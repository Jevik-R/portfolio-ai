# feature_builder.py  —  M3: Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
# Builds all inputs required for Black-Litterman portfolio optimization:
#
#   μ_prior  — market-implied equilibrium returns  (δ × Σ × w_mkt)
#   Σ        — historical covariance matrix
#   w_mkt    — market-cap based portfolio weights
#   Q        — LLM sentiment views (from FinBERT scores)
#   Ω        — view uncertainty matrix (Idzorek method via confidence)
#   μ_BL     — Black-Litterman posterior expected returns
#   Σ_BL     — Black-Litterman posterior covariance
#
# Also resolves USD → INR prices using the live FX rate.
# ─────────────────────────────────────────────────────────────────────────────

import os
import time
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from pypfopt import risk_models
from pypfopt.risk_models import fix_nonpositive_semidefinite as fix_nonpsd
from pypfopt.black_litterman import BlackLittermanModel, market_implied_prior_returns

from sentiment_engine import load_sentiment_scores, get_bl_views, dynamic_alpha

warnings.filterwarnings("ignore")

DATA_DIR         = "data"
STOCKS           = [
    "TCS.NS",       "INFY.NS",       "WIPRO.NS",      "HCLTECH.NS",
    "HDFCBANK.NS",  "ICICIBANK.NS",  "SBIN.NS",       "KOTAKBANK.NS",
    "SUNPHARMA.NS", "DRREDDY.NS",
    "HINDUNILVR.NS","ITC.NS",
    "RELIANCE.NS",  "ONGC.NS",
    "LT.NS",        "BHARTIARTL.NS",
]
DELTA            = 2.5      # market risk-aversion coefficient
TAU              = 0.025    # BL scaling — confidence in market prior
VIEW_SENSITIVITY = 0.5      # sentiment → return adjustment magnitude (halved from 1.0)

SECTOR_MAP = {
    "TCS.NS":        "Technology",     "INFY.NS":       "Technology",
    "WIPRO.NS":      "Technology",     "HCLTECH.NS":    "Technology",
    "HDFCBANK.NS":   "Finance",        "ICICIBANK.NS":  "Finance",
    "SBIN.NS":       "Finance",        "KOTAKBANK.NS":  "Finance",
    "SUNPHARMA.NS":  "Healthcare",     "DRREDDY.NS":    "Healthcare",
    "HINDUNILVR.NS": "Consumer",       "ITC.NS":        "Consumer",
    "RELIANCE.NS":   "Energy",         "ONGC.NS":       "Energy",
    "LT.NS":         "Infrastructure", "BHARTIARTL.NS": "Telecom",
}


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYST CONSENSUS SIGNAL
# ══════════════════════════════════════════════════════════════════════════════

def get_analyst_consensus(ticker: str) -> float:
    """
    Fetch analyst recommendation consensus from yfinance.

    Scores the distribution of Strong Buy / Buy / Hold / Sell / Strong Sell
    into a single value in [-1.0, +1.0]:
      +1.0 = unanimous strong buy   |   -1.0 = unanimous strong sell
      0.0  = no data or all holds

    This is blended into BL views as a mild secondary signal.
    """
    try:
        t   = yf.Ticker(ticker)
        rec = t.recommendations_summary
        if rec is not None and not rec.empty:
            row         = rec.iloc[0]
            strong_buy  = float(row.get("strongBuy",  0))
            buy         = float(row.get("buy",         0))
            hold        = float(row.get("hold",        0))
            sell        = float(row.get("sell",        0))
            strong_sell = float(row.get("strongSell",  0))
            total = strong_buy + buy + hold + sell + strong_sell
            if total > 0:
                score = (strong_buy * 1.0 + buy * 0.5 + hold * 0.0
                         + sell * -0.5 + strong_sell * -1.0) / total
                return round(score, 4)
    except Exception:
        pass
    return 0.0


def fetch_all_analyst_consensus(tickers: list) -> pd.Series:
    """Fetch analyst consensus for all tickers. Returns Series indexed by ticker."""
    scores = {}
    for ticker in tickers:
        scores[ticker] = get_analyst_consensus(ticker)
        time.sleep(0.05)
    return pd.Series(scores, name="analyst_consensus")


def get_sector_sentiment(sentiment_df: pd.DataFrame, sector_map: dict) -> dict:
    """
    Aggregate per-stock FinBERT scores into sector-level averages.

    Returns dict: { "Technology": 0.12, "Finance": -0.05, ... }
    Used in dashboard heatmap and as a macro context signal.
    """
    sector_scores: dict = {}
    for ticker in sentiment_df.index:
        sector = sector_map.get(ticker, "Other")
        score  = float(sentiment_df.loc[ticker, "final_score"])
        sector_scores.setdefault(sector, []).append(score)
    return {s: round(float(np.mean(v)), 4) for s, v in sorted(sector_scores.items())}


# ══════════════════════════════════════════════════════════════════════════════
#  EARNINGS SURPRISE SIGNAL
# ══════════════════════════════════════════════════════════════════════════════

def get_earnings_surprise(ticker: str) -> float:
    """
    Fetch the most recent earnings surprise for a stock via yfinance.

    Returns surprise as a fraction: (actual - estimate) / |estimate|
      +0.10 = beat by 10%   |   -0.15 = missed by 15%   |   0.0 = no data

    Blended into BL views alongside sentiment: positive surprise nudges Q up,
    negative surprise nudges Q down, scaled by a small beta = 0.15.
    """
    try:
        t = yf.Ticker(ticker)
        # earnings_dates includes both historical and upcoming dates
        dates = t.earnings_dates
        if dates is not None and not dates.empty:
            hist = dates.dropna(subset=["Reported EPS", "EPS Estimate"])
            if not hist.empty:
                row      = hist.sort_index(ascending=False).iloc[0]
                actual   = float(row["Reported EPS"])
                estimate = float(row["EPS Estimate"])
                if estimate != 0:
                    return round((actual - estimate) / abs(estimate), 4)
    except Exception:
        pass
    return 0.0


def fetch_all_earnings_surprises(tickers: list) -> pd.Series:
    """Fetch earnings surprises for all tickers. Returns Series indexed by ticker."""
    surprises = {}
    for ticker in tickers:
        surprises[ticker] = get_earnings_surprise(ticker)
        time.sleep(0.1)   # gentle rate limiting
    return pd.Series(surprises, name="earnings_surprise")


# ══════════════════════════════════════════════════════════════════════════════
#  LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_prices_and_returns():
    for path in (f"{DATA_DIR}/prices.csv", f"{DATA_DIR}/returns.csv"):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found — run: python data_collector.py"
            )
    prices  = pd.read_csv(f"{DATA_DIR}/prices.csv",  index_col=0, parse_dates=True)
    returns = pd.read_csv(f"{DATA_DIR}/returns.csv", index_col=0, parse_dates=True)
    return prices, returns


def _load_market_caps() -> pd.Series:
    path = f"{DATA_DIR}/market_caps.csv"
    if not os.path.exists(path):
        # Fallback: equal caps so BL still runs
        print("  ⚠️  market_caps.csv not found — using equal market caps (run data_collector.py)")
        return pd.Series({t: 1.0 for t in STOCKS})
    df = pd.read_csv(path, index_col=0)
    return df.iloc[:, 0]   # first column = market_cap_usd


def _load_fx_rate() -> float:
    """NSE stock prices are already in INR — return 1.0 (identity)."""
    return 1.0


# ══════════════════════════════════════════════════════════════════════════════
#  CORE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_features(lookback_days: int = 252) -> dict:
    """
    Build the complete feature set for portfolio optimisation.

    Parameters
    ----------
    lookback_days : int
        Number of trading days of history to use for covariance estimation.

    Returns
    -------
    dict with keys:
        mu_bl        — BL posterior expected returns (pd.Series)
        S_bl         — BL posterior covariance matrix (pd.DataFrame)
        mu_prior     — market-implied prior returns (pd.Series)
        S            — historical sample covariance (pd.DataFrame)
        sentiment_df — FinBERT scores per ticker (pd.DataFrame)
        w_mkt        — market-cap weights (pd.Series)
        prices_inr   — latest stock prices in INR (pd.Series)
        fx_rate      — live USD/INR rate (float)
        tickers      — list of tickers actually used
    """
    # ── Load raw data ─────────────────────────────────────────────────────────
    prices, returns = _load_prices_and_returns()
    mcaps           = _load_market_caps()
    fx_rate         = _load_fx_rate()

    try:
        sentiment_df = load_sentiment_scores()
    except FileNotFoundError:
        sentiment_df = pd.DataFrame()

    # ── Align tickers ─────────────────────────────────────────────────────────
    # Tickers must exist in price/returns data. Sentiment is optional — if the
    # saved sentiment_scores.csv has stale tickers (e.g. old US stocks), we fall
    # back to neutral scores so the BL model still runs purely on market prior.
    price_available = [t for t in STOCKS
                       if t in prices.columns and t in returns.columns]

    if len(price_available) < 3:
        raise ValueError(f"Only {len(price_available)} tickers in prices.csv — "
                         f"run: python data_collector.py")

    sentiment_available = [t for t in price_available if t in sentiment_df.index]
    stale_sentiment     = len(sentiment_available) < 3

    if stale_sentiment:
        # sentiment_scores.csv has different tickers — build neutral placeholder
        print("  ⚠️  sentiment_scores.csv has no matching NSE tickers. "
              "Run: python sentiment_engine.py   (using neutral scores for now)")
        idx = pd.Index(price_available, name="ticker")
        sentiment_df = pd.DataFrame({
            "final_score":   0.0,
            "label":         "neutral",
            "confidence":    0.5,
            "num_headlines": 0,
            "pct_positive":  0.0,
            "pct_negative":  0.0,
            "pct_neutral":   1.0,
            "sentiment_std": 0.0,
        }, index=idx)

    available = price_available

    stock_prices  = prices[available].dropna()

    # ── Covariance matrix ─────────────────────────────────────────────────────
    # sample_cov expects PRICES (it does pct_change internally).
    # Slice to the most recent lookback window (+1 row so pct_change has full lookback).
    prices_window = stock_prices.iloc[-(lookback_days + 1):]
    S_raw = risk_models.sample_cov(prices_window, frequency=252)
    # Ledoit-Wolf shrinkage gives a guaranteed positive-definite matrix
    try:
        S = risk_models.CovarianceShrinkage(prices_window, frequency=252).ledoit_wolf()
    except Exception:
        S = pd.DataFrame(fix_nonpsd(S_raw), index=available, columns=available)

    # ── Market-cap weights ────────────────────────────────────────────────────
    mcap_aligned = mcaps.reindex(available).fillna(mcaps.mean())
    w_mkt        = mcap_aligned / mcap_aligned.sum()

    # ── Market-implied prior returns  μ_eq = δ × Σ × w_mkt ──────────────────
    mu_prior = market_implied_prior_returns(w_mkt, DELTA, S)

    # ── Align sentiment to available tickers (must come before all downstream calls) ──
    sent_aligned = sentiment_df.loc[available]

    # ── Sector sentiment aggregation ─────────────────────────────────────────
    sector_sentiment = get_sector_sentiment(sent_aligned, SECTOR_MAP)
    print("  Sector sentiment: " +
          "  ".join(f"{s}:{v:+.3f}" for s, v in sector_sentiment.items()))

    # ── Analyst consensus signal ──────────────────────────────────────────────
    print("  Fetching analyst consensus...")
    analyst_consensus = fetch_all_analyst_consensus(available)
    nonzero_ac = [(t, v) for t, v in analyst_consensus.items() if v != 0]
    if nonzero_ac:
        print(f"  Analyst consensus: " +
              "  ".join(f"{t}:{v:+.2f}" for t, v in nonzero_ac))

    # ── Earnings surprise signal ──────────────────────────────────────────────
    print("  Fetching earnings surprises...")
    earnings_surprise = fetch_all_earnings_surprises(available)
    print(f"  Earnings surprises: " +
          "  ".join(f"{t}:{v:+.2f}" for t, v in earnings_surprise.items() if v != 0))

    # ── LLM views (Q, confidence) via FinBERT sentiment + earnings ───────────
    viewdict, confidences = get_bl_views(
        sent_aligned, mu_prior, S,
        view_sensitivity=VIEW_SENSITIVITY,
        earnings_surprise=earnings_surprise,
    )

    # ── Black-Litterman posterior ─────────────────────────────────────────────
    bl = BlackLittermanModel(
        S,
        absolute_views   = viewdict,
        pi               = mu_prior,
        omega            = "idzorek",
        view_confidences = confidences,
        tau              = TAU,
    )
    mu_bl   = bl.bl_returns()
    # Regularise BL covariance to guarantee positive-definiteness for the solver
    S_bl_raw = np.array(bl.bl_cov())
    S_bl_reg = S_bl_raw + np.eye(len(available)) * 1e-8
    S_bl     = pd.DataFrame(fix_nonpsd(S_bl_reg), index=available, columns=available)

    # ── Current prices in INR ─────────────────────────────────────────────────
    latest_usd = stock_prices.iloc[-1]
    prices_inr = (latest_usd * fx_rate).round(2)

    # ── Debug summary ─────────────────────────────────────────────────────────
    print(f"\n📐 FEATURE BUILDER — Black-Litterman")
    print("=" * 55)
    print(f"  Tickers        : {len(available)}")
    print(f"  Lookback       : {len(prices_window)} trading days")
    print(f"  Market         : NSE India (prices in ₹)")
    print(f"  τ (BL prior)   : {TAU}    δ (risk aversion): {DELTA}")
    print(f"\n  {'TICKER':<7} {'μ_prior':>9} {'Q(view)':>9} {'μ_BL':>9}  SENTIMENT")
    print(f"  {'─'*52}")
    for t in available:
        print(f"  {t:<7} {mu_prior[t]:>+9.3%} {viewdict[t]:>+9.3%} "
              f"{mu_bl[t]:>+9.3%}  {sent_aligned.loc[t, 'label']}")

    return {
        "mu_bl":            mu_bl,
        "S_bl":             S_bl,
        "mu_prior":         mu_prior,
        "S":                S,
        "sentiment_df":     sent_aligned,
        "w_mkt":            w_mkt,
        "prices_inr":       prices_inr,
        "fx_rate":          fx_rate,
        "tickers":          available,
        "analyst_consensus":analyst_consensus,
        "sector_sentiment": sector_sentiment,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    features = build_features()
    print(f"\n✅ Features ready. BL posterior μ range: "
          f"{features['mu_bl'].min():.2%} → {features['mu_bl'].max():.2%}")
    print(f"\nCurrent prices in INR:")
    for t, p in features["prices_inr"].items():
        print(f"  {t:<6} ₹{p:>10,.2f}")
