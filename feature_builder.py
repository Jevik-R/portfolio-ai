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

from llm_views import get_bl_views, dynamic_alpha
from scorer import load_factor_scores

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
#  VIEW HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _neutral_views(tickers: list) -> pd.DataFrame:
    """Fallback: neutral views for all tickers when no view data is available."""
    idx = pd.Index(tickers, name="ticker")
    return pd.DataFrame({
        "final_score":   0.0,
        "label":         "neutral",
        "confidence":    0.5,
        "num_headlines": 0,
        "pct_positive":  0.0,
        "pct_negative":  0.0,
        "pct_neutral":   1.0,
        "sentiment_std": 0.0,
        "q":             0.0,
        "omega":         0.01,
    }, index=idx)


def _load_combined_views(tickers: list, alpha: float = 0.5) -> pd.DataFrame:
    """
    Blend LLM views and FinBERT sentiment scores.
    alpha = weight given to LLM views; (1-alpha) = weight given to sentiment.
    Agreement between signals → higher confidence; disagreement → lower.
    """
    # Load LLM views
    try:
        from llm_views import load_sentiment_scores as _load_llm
        llm_df = _load_llm()
    except Exception:
        llm_df = pd.DataFrame()

    # Load FinBERT sentiment
    try:
        from sentiment_engine import load_sentiment_scores as _load_sent
        sent_df = _load_sent()
    except Exception:
        sent_df = pd.DataFrame()

    # If only one is available, use that
    if llm_df.empty and sent_df.empty:
        return _neutral_views(tickers)
    if llm_df.empty:
        print("  ℹ️  LLM views missing — using sentiment only")
        return sent_df
    if sent_df.empty:
        print("  ℹ️  Sentiment missing — using LLM views only")
        return llm_df

    # Both available — blend them
    blended = llm_df.copy()

    for ticker in tickers:
        llm_available  = ticker in llm_df.index
        sent_available = ticker in sent_df.index

        if not llm_available and not sent_available:
            continue
        elif llm_available and not sent_available:
            continue  # keep llm as-is
        elif not llm_available and sent_available:
            blended.loc[ticker] = sent_df.loc[ticker]
            continue

        # Both available for this ticker
        llm_score  = float(llm_df.loc[ticker,  "final_score"])
        sent_score = float(sent_df.loc[ticker, "final_score"])
        llm_conf   = float(llm_df.loc[ticker,  "confidence"])
        sent_conf  = float(sent_df.loc[ticker, "confidence"])

        blended_score = alpha * llm_score + (1 - alpha) * sent_score

        both_positive = llm_score > 0 and sent_score > 0
        both_negative = llm_score < 0 and sent_score < 0
        conf_adjustment = +0.10 if (both_positive or both_negative) else -0.10

        blended_conf = float(np.clip(
            (alpha * llm_conf + (1 - alpha) * sent_conf) + conf_adjustment,
            0.05, 0.95
        ))

        if   blended_score >=  0.30: label = "bullish"
        elif blended_score >=  0.05: label = "slightly_bullish"
        elif blended_score <= -0.30: label = "bearish"
        elif blended_score <= -0.05: label = "slightly_bearish"
        else:                        label = "neutral"

        blended.loc[ticker, "final_score"] = round(blended_score, 4)
        blended.loc[ticker, "confidence"]  = round(blended_conf,  4)
        blended.loc[ticker, "label"]       = label

        # Store both individual scores for dashboard display
        blended.loc[ticker, "llm_score"]  = round(llm_score,  4)
        blended.loc[ticker, "sent_score"] = round(sent_score, 4)
        blended.loc[ticker, "signals_agree"] = int(both_positive or both_negative)

    return blended


# ══════════════════════════════════════════════════════════════════════════════
#  CORE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_features(lookback_days: int = 252, analysis_method: str = "llm") -> dict:
    """
    Build the complete feature set for portfolio optimisation.

    Parameters
    ----------
    lookback_days : int
        Number of trading days of history to use for covariance estimation.
    analysis_method : str
        One of "llm" (Groq/LLaMA views), "sentiment" (FinBERT news),
        or "combined" (blended). Falls back to neutral if data is missing.

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

    # ── Align tickers (prices first, views second) ────────────────────────────
    price_available = [t for t in STOCKS
                       if t in prices.columns and t in returns.columns]

    if len(price_available) < 3:
        raise ValueError(f"Only {len(price_available)} tickers in prices.csv — "
                         f"run: python data_collector.py")

    available = price_available

    # ── Load views based on selected method ───────────────────────────────────
    if analysis_method == "llm":
        try:
            from llm_views import load_sentiment_scores as _load_llm
            sentiment_df = _load_llm()
            print(f"  📊 Using LLM views (Groq/LLaMA)")
        except FileNotFoundError:
            print("  ⚠️  llm_views.csv not found — run llm_views.py first")
            sentiment_df = _neutral_views(available)
        except Exception as e:
            print(f"  ⚠️  LLM views error: {e} — using neutral views")
            sentiment_df = _neutral_views(available)

    elif analysis_method == "sentiment":
        try:
            from sentiment_engine import load_sentiment_scores as _load_sent
            sentiment_df = _load_sent()
            print(f"  📰 Using FinBERT sentiment")
            matching = [t for t in available if t in sentiment_df.index]
            if len(matching) < 3:
                print("  ⚠️  Sentiment file has no NSE tickers — using neutral views")
                sentiment_df = _neutral_views(available)
        except FileNotFoundError:
            print("  ⚠️  sentiment_scores.csv not found — run sentiment_engine.py first")
            sentiment_df = _neutral_views(available)
        except Exception as e:
            print(f"  ⚠️  Sentiment error: {e} — using neutral views")
            sentiment_df = _neutral_views(available)

    elif analysis_method == "combined":
        sentiment_df = _load_combined_views(available)
        print(f"  🔥 Using combined views (LLM + FinBERT sentiment)")

    else:
        sentiment_df = _neutral_views(available)

    # Final safety net — if we still have no matching tickers, use neutral
    if sentiment_df.empty or len([t for t in available if t in sentiment_df.index]) < 3:
        print("  ⚠️  No matching view data — falling back to neutral views")
        sentiment_df = _neutral_views(available)

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

    # ── Factor score blending  (graceful skip if factor_scores.csv missing) ──
    factor_scores = load_factor_scores()
    if factor_scores is not None:
        # Keep only tickers that are both in BL universe and factor scores
        factor_tickers = [t for t in available if t in factor_scores.index]
        if factor_tickers:
            print(f"\n  📐 Blending factor scores for {len(factor_tickers)} tickers "
                  f"(60% LLM + 40% factor)")
            for ticker in factor_tickers:
                llm_view      = viewdict.get(ticker, float(mu_prior[ticker]))
                combined_score = float(factor_scores.loc[ticker, "combined_score"])
                # factor_score_view = market prior × (1 + combined_score)
                # combined_score is 0→1; map to a ±view by centring at 0.5
                factor_view    = float(mu_prior[ticker]) * (1 + (combined_score - 0.5))
                # Blend: 60% LLM, 40% factor
                blended        = 0.60 * llm_view + 0.40 * factor_view
                viewdict[ticker] = round(blended, 6)
        else:
            factor_scores = None   # no overlap — skip factor display
    else:
        print("  ℹ️  factor_scores.csv not found — using pure LLM views "
              "(run: python scorer.py to enable factor blending)")

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
        "factor_scores":    factor_scores,   # pd.DataFrame or None
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    method = sys.argv[1] if len(sys.argv) > 1 else "llm"
    features = build_features(analysis_method=method)
    print(f"\n✅ Features ready. BL posterior μ range: "
          f"{features['mu_bl'].min():.2%} → {features['mu_bl'].max():.2%}")
    print(f"\nCurrent prices in INR:")
    for t, p in features["prices_inr"].items():
        print(f"  {t:<6} ₹{p:>10,.2f}")
