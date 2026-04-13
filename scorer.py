# scorer.py  —  Quantitative Factor Scoring Engine
# ─────────────────────────────────────────────────────────────────────────────
# Scores every stock on 3 proven quant factors each time it runs:
#
#   1. MOMENTUM  (40%) — 6-month price return, skip last 1 month (t-126:t-21)
#   2. QUALITY   (40%) — ROE + Debt/Equity + EPS growth from yfinance
#   3. VOLATILITY(20%) — 60-day realized vol, inverted (low vol = high score)
#
# COMBINED = 0.4×momentum + 0.4×quality + 0.2×(1 - volatility)
#
# Top 15 stocks by combined score enter the portfolio universe.
# Saves: data/factor_scores.csv
#
# Run: python scorer.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = "data"

STOCKS = [
    "TCS.NS",       "INFY.NS",       "WIPRO.NS",      "HCLTECH.NS",
    "HDFCBANK.NS",  "ICICIBANK.NS",  "SBIN.NS",       "KOTAKBANK.NS",
    "SUNPHARMA.NS", "DRREDDY.NS",
    "HINDUNILVR.NS","ITC.NS",
    "RELIANCE.NS",  "ONGC.NS",
    "LT.NS",        "BHARTIARTL.NS",
]

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

TOP_N = 15   # number of stocks to select for portfolio universe


# ══════════════════════════════════════════════════════════════════════════════
#  1. MOMENTUM SCORE  (40% weight)
# ══════════════════════════════════════════════════════════════════════════════

def compute_momentum_scores(prices: pd.DataFrame) -> pd.Series:
    """
    6-month price return, skipping last 1 month to avoid short-term reversal.
    Formula: return from price[t-126] to price[t-21].

    Returns per-stock momentum score ranked 0→1 within universe.
    """
    available = [t for t in STOCKS if t in prices.columns]
    raw_momentum = {}

    for ticker in available:
        p = prices[ticker].dropna()
        if len(p) < 130:
            print(f"  ⚠️  {ticker}: insufficient history for momentum ({len(p)} days)")
            raw_momentum[ticker] = np.nan
            continue
        p_start = float(p.iloc[-126])
        p_end   = float(p.iloc[-21])
        if p_start > 0:
            raw_momentum[ticker] = (p_end / p_start) - 1.0
        else:
            raw_momentum[ticker] = np.nan

    s = pd.Series(raw_momentum)
    s = s.fillna(s.median())   # fill missing with median to stay neutral

    # Rank 0→1 (percentile rank within universe)
    ranked = s.rank(pct=True)
    return ranked.rename("momentum_score")


# ══════════════════════════════════════════════════════════════════════════════
#  2. QUALITY SCORE  (40% weight)
# ══════════════════════════════════════════════════════════════════════════════

def compute_quality_scores(fundamentals: pd.DataFrame) -> pd.Series:
    """
    Combine ROE, Debt/Equity, and EPS growth into a single 0→1 quality score.

    - ROE          : higher = better → rank ascending
    - Debt/Equity  : lower  = better → rank descending (inverted)
    - EPS growth   : higher = better → rank ascending
    """
    available = [t for t in STOCKS if t in fundamentals.index]
    if not available:
        print("  ⚠️  No fundamentals data — using equal quality scores")
        return pd.Series({t: 0.5 for t in STOCKS}, name="quality_score")

    df = fundamentals.reindex(available).copy()

    # ── ROE ──────────────────────────────────────────────────────────────────
    roe = df["returnOnEquity"].fillna(0.0)
    roe = roe.clip(-2.0, 5.0)          # clip extreme outliers
    roe_rank = roe.rank(pct=True)

    # ── Debt / Equity  (lower = better → invert) ─────────────────────────────
    de = df["debtToEquity"].fillna(df["debtToEquity"].median())
    de = de.clip(0.0, 10.0)
    de_rank = 1.0 - de.rank(pct=True)  # invert

    # ── EPS Growth ───────────────────────────────────────────────────────────
    eps_g = df["earningsGrowth"].fillna(0.0)
    eps_g = eps_g.clip(-1.0, 3.0)
    eps_rank = eps_g.rank(pct=True)

    quality = (roe_rank + de_rank + eps_rank) / 3.0

    # Back-fill any tickers not in fundamentals with 0.5 (neutral)
    all_quality = pd.Series({t: float(quality.get(t, 0.5)) for t in STOCKS})
    return all_quality.rename("quality_score")


# ══════════════════════════════════════════════════════════════════════════════
#  3. VOLATILITY SCORE  (20% weight)
# ══════════════════════════════════════════════════════════════════════════════

def compute_volatility_scores(returns: pd.DataFrame, window: int = 60) -> pd.Series:
    """
    60-day realized annualized volatility, inverted.
    Lower volatility → higher score (rank 0→1, then invert).
    """
    available = [t for t in STOCKS if t in returns.columns]
    raw_vol = {}

    for ticker in available:
        r = returns[ticker].dropna()
        if len(r) < window:
            print(f"  ⚠️  {ticker}: insufficient history for vol ({len(r)} days)")
            raw_vol[ticker] = np.nan
            continue
        raw_vol[ticker] = float(r.iloc[-window:].std() * np.sqrt(252))

    v = pd.Series(raw_vol)
    v = v.fillna(v.median())

    # Rank then invert: low vol gets high score
    v_ranked = 1.0 - v.rank(pct=True)

    all_vol = pd.Series({t: float(v_ranked.get(t, 0.5)) for t in STOCKS})
    return all_vol.rename("volatility_score")


# ══════════════════════════════════════════════════════════════════════════════
#  4. COMBINED FACTOR SCORE
# ══════════════════════════════════════════════════════════════════════════════

def compute_factor_scores() -> pd.DataFrame:
    """
    Load prices, returns, and fundamentals.
    Compute and combine all 3 factor scores.
    Select top 15 stocks by combined score.

    Returns DataFrame with columns:
        momentum_score, quality_score, volatility_score, combined_score,
        sector, selected (bool), raw_momentum, raw_volatility
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # ── Load price / return data ──────────────────────────────────────────────
    prices_path  = f"{DATA_DIR}/prices.csv"
    returns_path = f"{DATA_DIR}/returns.csv"

    if not os.path.exists(prices_path):
        raise FileNotFoundError("prices.csv not found — run: python data_collector.py")
    if not os.path.exists(returns_path):
        raise FileNotFoundError("returns.csv not found — run: python data_collector.py")

    prices  = pd.read_csv(prices_path,  index_col=0, parse_dates=True)
    returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)

    # ── Load fundamentals ─────────────────────────────────────────────────────
    fundamentals_path = f"{DATA_DIR}/fundamentals.csv"
    if os.path.exists(fundamentals_path):
        fundamentals = pd.read_csv(fundamentals_path, index_col=0)
        print(f"  📂 Loaded fundamentals for {len(fundamentals)} tickers")
    else:
        print("  ⚠️  fundamentals.csv not found — quality scores will be neutral 0.5")
        fundamentals = pd.DataFrame(index=STOCKS)

    print("\n📐 FACTOR SCORING ENGINE")
    print("=" * 55)

    # ── 1. Momentum ───────────────────────────────────────────────────────────
    print("\n  [1/3] Computing momentum scores (6m return, skip 1m)...")
    mom_scores = compute_momentum_scores(prices)
    print(f"       Range: {mom_scores.min():.3f} → {mom_scores.max():.3f}")

    # ── 2. Quality ────────────────────────────────────────────────────────────
    print("\n  [2/3] Computing quality scores (ROE / D:E / EPS growth)...")
    qual_scores = compute_quality_scores(fundamentals)
    print(f"       Range: {qual_scores.min():.3f} → {qual_scores.max():.3f}")

    # ── 3. Volatility ─────────────────────────────────────────────────────────
    print("\n  [3/3] Computing volatility scores (60-day realized vol, inverted)...")
    vol_scores = compute_volatility_scores(returns)
    print(f"       Range: {vol_scores.min():.3f} → {vol_scores.max():.3f}")

    # ── Combined ──────────────────────────────────────────────────────────────
    all_tickers = [t for t in STOCKS
                   if t in mom_scores.index and t in qual_scores.index and t in vol_scores.index]

    combined = (
        0.40 * mom_scores.reindex(all_tickers) +
        0.40 * qual_scores.reindex(all_tickers) +
        0.20 * vol_scores.reindex(all_tickers)
    )

    # ── Also store raw (unranked) values for display ──────────────────────────
    raw_momentum = {}
    for ticker in all_tickers:
        p = prices[ticker].dropna() if ticker in prices.columns else pd.Series()
        if len(p) >= 130:
            raw_momentum[ticker] = round((float(p.iloc[-21]) / float(p.iloc[-126])) - 1.0, 6)
        else:
            raw_momentum[ticker] = 0.0

    raw_vol = {}
    for ticker in all_tickers:
        r = returns[ticker].dropna() if ticker in returns.columns else pd.Series()
        if len(r) >= 60:
            raw_vol[ticker] = round(float(r.iloc[-60:].std() * np.sqrt(252)), 6)
        else:
            raw_vol[ticker] = 0.0

    # ── Build result DataFrame ────────────────────────────────────────────────
    top15 = set(combined.nlargest(TOP_N).index)

    df = pd.DataFrame({
        "momentum_score":  mom_scores.reindex(all_tickers).round(4),
        "quality_score":   qual_scores.reindex(all_tickers).round(4),
        "volatility_score":vol_scores.reindex(all_tickers).round(4),
        "combined_score":  combined.round(4),
        "sector":          pd.Series({t: SECTOR_MAP.get(t, "Other") for t in all_tickers}),
        "selected":        pd.Series({t: t in top15 for t in all_tickers}),
        "raw_momentum_6m": pd.Series(raw_momentum).round(4),
        "raw_vol_60d":     pd.Series(raw_vol).round(4),
    })
    df.index.name = "ticker"
    df = df.sort_values("combined_score", ascending=False)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n  {'TICKER':<14} {'MOMENTUM':>9} {'QUALITY':>9} {'VOL':>9} {'COMBINED':>9}  SELECTED")
    print(f"  {'─'*65}")
    for ticker, row in df.iterrows():
        mark = "✅" if row["selected"] else "  "
        print(f"  {ticker:<14} {row['momentum_score']:>9.3f} "
              f"{row['quality_score']:>9.3f} "
              f"{row['volatility_score']:>9.3f} "
              f"{row['combined_score']:>9.3f}  {mark}")

    print(f"\n  Top {TOP_N} selected: {sorted(t for t in top15)}")

    # ── Save ──────────────────────────────────────────────────────────────────
    df.to_csv(f"{DATA_DIR}/factor_scores.csv")
    print(f"\n  💾 Saved: {DATA_DIR}/factor_scores.csv")

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  LOADER  (used by feature_builder.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_factor_scores() -> pd.DataFrame | None:
    """
    Load saved factor scores. Returns None if file doesn't exist (graceful skip).
    Called by feature_builder.py to blend into BL views.
    """
    path = f"{DATA_DIR}/factor_scores.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col="ticker")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    scores = compute_factor_scores()

    print(f"\n\n✅ Factor scoring complete!")
    print(f"   Universe: {len(scores)} stocks  |  Selected: {scores['selected'].sum()}")
    print(f"   Top 5 by combined score:")
    for ticker, row in scores.head(5).iterrows():
        print(f"     {ticker:<14}  combined={row['combined_score']:.3f}  "
              f"mom={row['raw_momentum_6m']:+.1%}  "
              f"sector={row['sector']}")

    print(f"\n   Next step: run  python llm_views.py")
