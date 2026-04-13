# risk_manager.py  —  Portfolio Risk Management Layer
# ─────────────────────────────────────────────────────────────────────────────
# Three risk controls applied after Black-Litterman optimisation:
#
#   1. VOLATILITY TARGETING
#      - Target annualised portfolio vol = 12%
#      - Scale factor = target_vol / realised_20d_vol,  clipped [0.5, 1.5]
#      - Remaining weight → cash buffer
#
#   2. DRAWDOWN CONTROL
#      - DD > 15% → reduce equity to 70% (hold 30% cash)
#      - DD > 25% → reduce equity to 50%
#      - Resume full exposure when DD < 5%
#
#   3. POSITION LIMITS
#      - Single stock max:   15%
#      - Single sector max:  35%
#      - Min position:        2% (else zero — no tiny dust positions)
#
# All functions are pure (no side effects) and return (new_weights, cash_fraction).
# ─────────────────────────────────────────────────────────────────────────────

import warnings
import numpy as np
import pandas as pd
import os

warnings.filterwarnings("ignore")

# ── Default parameters ────────────────────────────────────────────────────────
TARGET_VOL      = 0.12   # 12% annualised
SCALE_MIN       = 0.50   # never go below 50% equity
SCALE_MAX       = 1.50   # never lever beyond 150% (practically 100% here)
VOL_WINDOW      = 20     # trading days for realised vol estimate

DD_THRESHOLD_1  = 0.15   # drawdown > 15% → 70% equity
DD_THRESHOLD_2  = 0.25   # drawdown > 25% → 50% equity
DD_RECOVER      = 0.05   # resume full equity when drawdown < 5%

MAX_SINGLE_WT   = 0.15   # single stock
MAX_SECTOR_WT   = 0.35   # single sector
MIN_POSITION    = 0.02   # below this → zero out (no tiny sliver positions)

DATA_DIR = "data"

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
#  HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _normalise(weights: dict) -> dict:
    """Normalise weights so all positive weights sum to 1.0."""
    total = sum(v for v in weights.values() if v > 0)
    if total <= 0:
        n = len(weights)
        return {k: 1.0 / n for k in weights}
    return {k: v / total for k, v in weights.items()}


def _load_returns() -> pd.DataFrame | None:
    """Load returns.csv if available. Returns None on failure."""
    path = f"{DATA_DIR}/returns.csv"
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  1. VOLATILITY TARGETING
# ══════════════════════════════════════════════════════════════════════════════

def apply_volatility_targeting(
    weights:        dict,
    returns_df:     pd.DataFrame | None = None,
    target_vol:     float = TARGET_VOL,
    window:         int   = VOL_WINDOW,
) -> tuple[dict, float, float]:
    """
    Scale portfolio weights so that realised portfolio volatility ≈ target_vol.

    Parameters
    ----------
    weights    : dict  — {ticker: weight}  (should sum to 1.0)
    returns_df : DataFrame  — daily returns (rows=dates, cols=tickers)
                              If None, loads data/returns.csv automatically.
    target_vol : float  — target annualised volatility (default 12%)
    window     : int    — lookback window in trading days (default 20)

    Returns
    -------
    (scaled_weights, cash_fraction, scale_factor)
    scaled_weights : dict  — equity weights after scaling (sum < 1 when cash > 0)
    cash_fraction  : float — portion to hold as cash / liquid fund
    scale_factor   : float — the raw scale applied [0.5, 1.5]
    """
    if returns_df is None:
        returns_df = _load_returns()

    if returns_df is None or returns_df.empty:
        print("  ⚠️  risk_manager: returns.csv unavailable — volatility targeting skipped")
        return weights, 0.0, 1.0

    # ── Compute realised portfolio vol ────────────────────────────────────────
    tickers_in_df = [t for t in weights if t in returns_df.columns and weights[t] > 0]
    if not tickers_in_df:
        return weights, 0.0, 1.0

    recent = returns_df[tickers_in_df].dropna().tail(window)
    if len(recent) < 5:
        print("  ⚠️  risk_manager: not enough history for vol targeting")
        return weights, 0.0, 1.0

    w_arr = np.array([weights.get(t, 0.0) for t in tickers_in_df])
    w_arr = w_arr / w_arr.sum()   # normalise subset

    port_returns = recent.values @ w_arr
    realised_vol = float(port_returns.std() * np.sqrt(252))

    if realised_vol < 1e-6:
        return weights, 0.0, 1.0

    scale = float(np.clip(target_vol / realised_vol, SCALE_MIN, SCALE_MAX))
    cash  = max(0.0, 1.0 - scale)   # remainder to cash (only when scale < 1)

    scaled = {t: w * scale for t, w in weights.items()}

    print(f"  📉 Vol Targeting | Realised: {realised_vol:.1%}  "
          f"Target: {target_vol:.1%}  Scale: {scale:.2f}  Cash: {cash:.1%}")

    return scaled, round(cash, 4), round(scale, 4)


# ══════════════════════════════════════════════════════════════════════════════
#  2. DRAWDOWN CONTROL
# ══════════════════════════════════════════════════════════════════════════════

def apply_drawdown_control(
    weights:           dict,
    portfolio_history: pd.Series | None = None,
) -> tuple[dict, float, str]:
    """
    Reduce equity exposure if portfolio is in a significant drawdown.

    Loads backtest_results.csv as portfolio_history if not provided.
    If no history is available, returns weights unchanged.

    Thresholds:
      - Drawdown > 25% → hold only 50% equity
      - Drawdown > 15% → hold only 70% equity
      - Drawdown < 5%  → full equity (1.0 scale)

    Returns
    -------
    (adjusted_weights, cash_fraction, regime_label)
    """
    # ── Get portfolio history (cumulative P&L series) ─────────────────────────
    if portfolio_history is None:
        bt_path = f"{DATA_DIR}/backtest_results.csv"
        if os.path.exists(bt_path):
            try:
                bt = pd.read_csv(bt_path, index_col="date", parse_dates=True)
                if "ret_sentiment" in bt.columns:
                    portfolio_history = (1 + bt["ret_sentiment"]).cumprod()
            except Exception:
                pass

    if portfolio_history is None or len(portfolio_history) < 2:
        # No history available — cannot compute drawdown, skip control
        return weights, 0.0, "no_history"

    # ── Compute current drawdown from peak ────────────────────────────────────
    peak     = float(portfolio_history.cummax().iloc[-1])
    current  = float(portfolio_history.iloc[-1])
    drawdown = (current - peak) / peak if peak > 0 else 0.0

    # ── Apply reduction rules ─────────────────────────────────────────────────
    if drawdown < -DD_THRESHOLD_2:
        equity_scale  = 0.50
        regime_label  = f"severe_drawdown ({drawdown:.1%})"
    elif drawdown < -DD_THRESHOLD_1:
        equity_scale  = 0.70
        regime_label  = f"moderate_drawdown ({drawdown:.1%})"
    elif drawdown > -DD_RECOVER:
        equity_scale  = 1.0
        regime_label  = f"normal ({drawdown:.1%})"
    else:
        # Between -5% and -15%: continue whatever was applied (no change)
        equity_scale  = 1.0
        regime_label  = f"recovering ({drawdown:.1%})"

    cash = max(0.0, 1.0 - equity_scale)
    adjusted = {t: w * equity_scale for t, w in weights.items()}

    if abs(drawdown) > DD_RECOVER:
        print(f"  🛡️  Drawdown Control | DD={drawdown:.1%}  "
              f"Equity={equity_scale:.0%}  Cash={cash:.0%}  [{regime_label}]")

    return adjusted, round(cash, 4), regime_label


# ══════════════════════════════════════════════════════════════════════════════
#  3. POSITION LIMITS
# ══════════════════════════════════════════════════════════════════════════════

def apply_position_limits(
    weights:      dict,
    sector_map:   dict  | None = None,
    max_single:   float = MAX_SINGLE_WT,
    max_sector:   float = MAX_SECTOR_WT,
    min_position: float = MIN_POSITION,
) -> dict:
    """
    Enforce per-stock and per-sector caps, and remove dust positions.

    1. Cap any single stock at max_single (default 15%)
    2. Cap any single sector at max_sector (default 35%)
       — reduce proportionally within over-weight sectors
    3. Zero out positions below min_position (default 2%)
    4. Re-normalise so equity weights sum to original total

    Returns cleaned weight dict (may sum < 1 if some weights were zeroed).
    """
    if sector_map is None:
        sector_map = SECTOR_MAP

    w = {k: max(0.0, float(v)) for k, v in weights.items()}
    equity_total = sum(w.values())

    if equity_total <= 0:
        return weights

    # ── Step 1: Per-stock cap ─────────────────────────────────────────────────
    cap_headroom = 0.0
    for ticker in list(w):
        if w[ticker] > max_single:
            cap_headroom += w[ticker] - max_single
            w[ticker] = max_single

    # ── Step 2: Per-sector cap ────────────────────────────────────────────────
    sector_wts: dict[str, float] = {}
    for ticker, wt in w.items():
        sec = sector_map.get(ticker, "Other")
        sector_wts[sec] = sector_wts.get(sec, 0.0) + wt

    for sector, sec_total in sector_wts.items():
        if sec_total > max_sector:
            trim_ratio = max_sector / sec_total
            for ticker in w:
                if sector_map.get(ticker, "Other") == sector:
                    w[ticker] = w[ticker] * trim_ratio

    # ── Step 3: Remove dust positions ────────────────────────────────────────
    new_total = sum(w.values())
    for ticker in list(w):
        frac = w[ticker] / new_total if new_total > 0 else 0
        if 0 < frac < min_position:
            w[ticker] = 0.0

    # ── Step 4: Re-normalise to same equity_total ─────────────────────────────
    new_total = sum(w.values())
    if new_total > 0:
        w = {k: v * equity_total / new_total for k, v in w.items()}

    trimmed = [t for t in w if w[t] < weights.get(t, 0) - 1e-6]
    if trimmed:
        print(f"  🔒 Position Limits | Capped/trimmed: {', '.join(trimmed)}")

    return w


# ══════════════════════════════════════════════════════════════════════════════
#  COMBINED PIPELINE  (called by optimizer.py)
# ══════════════════════════════════════════════════════════════════════════════

def apply_all_risk_controls(
    weights:           dict,
    returns_df:        pd.DataFrame | None = None,
    portfolio_history: pd.Series    | None = None,
    sector_map:        dict         | None = None,
) -> tuple[dict, float, dict]:
    """
    Apply all three risk controls in sequence:
      position limits → volatility targeting → drawdown control

    Position limits run first (clean up weights before vol calc).

    Returns
    -------
    (final_weights, total_cash_fraction, risk_info_dict)
    """
    print("\n  ⚙️  Applying risk controls...")

    # 1. Position limits (always run first)
    w = apply_position_limits(weights, sector_map)

    # 2. Volatility targeting
    w, cash_vol, scale = apply_volatility_targeting(w, returns_df)

    # 3. Drawdown control
    w, cash_dd, dd_regime = apply_drawdown_control(w, portfolio_history)

    # Total cash = max of the two (don't double-count)
    total_cash = min(1.0, cash_vol + cash_dd)

    risk_info = {
        "vol_scale":      round(scale, 4),
        "cash_vol":       round(cash_vol, 4),
        "cash_dd":        round(cash_dd,  4),
        "total_cash":     round(total_cash, 4),
        "dd_regime":      dd_regime,
    }

    return w, round(total_cash, 4), risk_info


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT  (smoke test)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Quick smoke test with equal weights
    test_weights = {t: 1 / 16 for t in [
        "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS",
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS",
        "SUNPHARMA.NS", "DRREDDY.NS", "HINDUNILVR.NS", "ITC.NS",
        "RELIANCE.NS", "ONGC.NS", "LT.NS", "BHARTIARTL.NS",
    ]}

    final, cash, info = apply_all_risk_controls(test_weights)

    print(f"\n✅ Risk Manager smoke test")
    print(f"   Input equity total : {sum(test_weights.values()):.4f}")
    print(f"   Output equity total: {sum(final.values()):.4f}")
    print(f"   Cash buffer        : {cash:.2%}")
    print(f"   Vol scale          : {info['vol_scale']:.2f}")
    print(f"   DD regime          : {info['dd_regime']}")
    print(f"\n   Weight sample:")
    for t, w in list(final.items())[:5]:
        print(f"     {t:<14}  {w:.4f}")
