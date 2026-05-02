# optimizer.py  —  M4: Portfolio Optimisation Engine
# ─────────────────────────────────────────────────────────────────────────────
# Three public functions:
#
#   optimize_fresh_investment(investment_inr, risk_profile)
#       → User has ₹X to invest fresh. Returns optimal allocation with
#         exact INR amounts and approximate shares per stock.
#
#   optimize_rebalancing(current_holdings, additional_inr, risk_profile)
#       → User already holds stocks. Returns a rebalancing plan:
#         which stocks to BUY / SELL / HOLD and by how much (₹).
#
#   run_walk_forward_backtest()
#       → Historical walk-forward backtest (June 2018 → present).
#         Saves data/backtest_results.csv and data/backtest_metrics.csv.
#
# Usage:
#   python optimizer.py                   → single-date allocation (fresh ₹1L)
#   python optimizer.py --backtest        → full walk-forward backtest
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import warnings
import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, EfficientCVaR, risk_models, expected_returns
from pypfopt.risk_models import fix_nonpositive_semidefinite as fix_nonpsd
from pypfopt.black_litterman import BlackLittermanModel, market_implied_prior_returns

from feature_builder import build_features
from macro_overlay import get_macro_snapshot, apply_macro_overlay
from risk_manager import apply_all_risk_controls

warnings.filterwarnings("ignore")

DATA_DIR       = "data"
STOCKS         = [
    "TCS.NS",       "INFY.NS",       "WIPRO.NS",      "HCLTECH.NS",
    "HDFCBANK.NS",  "ICICIBANK.NS",  "SBIN.NS",       "KOTAKBANK.NS",
    "SUNPHARMA.NS", "DRREDDY.NS",
    "HINDUNILVR.NS","ITC.NS",
    "RELIANCE.NS",  "ONGC.NS",
    "LT.NS",        "BHARTIARTL.NS",
]
RISK_FREE_INR  = 0.065    # Indian 10-yr G-Sec yield (~6.5%)
BROKERAGE_PCT  = 0.001    # 0.1% per transaction (conservative estimate)
MIN_TRADE_INR  = 500      # ignore rebalancing trades smaller than ₹500
LOOKBACK       = 252      # trading days for backtest covariance window
DELTA          = 2.5
TAU            = 0.025

# Risk profile → optimisation objective mapping
RISK_PROFILES = {
    "conservative": {"method": "min_cvar",        "max_weight": 0.20},  # minimise tail risk
    "moderate":     {"method": "max_sharpe",      "max_weight": 0.30},  # maximise Sharpe
    "aggressive":   {"method": "max_sharpe",      "max_weight": 0.40},  # concentrated Sharpe
}


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_inr(amount: float) -> str:
    """Indian number formatting: ₹1,23,456."""
    if amount >= 1e7:
        return f"₹{amount/1e7:.2f} Cr"
    if amount >= 1e5:
        return f"₹{amount/1e5:.2f} L"
    if amount >= 1e3:
        return f"₹{amount/1e3:.1f} K"
    return f"₹{amount:.0f}"


def _run_ef(mu_bl, S_bl, sentiment_df, risk_profile: str,
            analysis_method: str = "llm") -> dict:
    """
    Run EfficientFrontier on BL posterior returns with sentiment constraints.
    Returns cleaned weights dict.
    """
    profile    = RISK_PROFILES.get(risk_profile, RISK_PROFILES["moderate"])
    max_weight = profile["max_weight"]

    # Sentiment-derived per-stock weight bounds (import conditionally to avoid
    # forcing sentiment_engine when the user selected the llm method)
    try:
        if analysis_method == "sentiment":
            from sentiment_engine import get_sentiment_constraints
        else:
            from llm_views import get_sentiment_constraints
        constraints = get_sentiment_constraints(sentiment_df)
    except Exception:
        constraints = {}
    lower_bounds = constraints.get("lower_bounds", {})
    upper_bounds = constraints.get("upper_bounds", {})

    tickers = list(mu_bl.index)
    bounds  = [
        (float(lower_bounds.get(t, 0.0)),
         min(float(upper_bounds.get(t, max_weight)), max_weight))
        for t in tickers
    ]

    method = profile["method"]

    # If all expected returns are below the risk-free rate, max_sharpe is
    # ill-defined (maximising a universally negative Sharpe ratio). Switch to
    # min_volatility which is always well-defined regardless of return level.
    effective_method = method
    if method == "max_sharpe" and all(float(v) < RISK_FREE_INR for v in mu_bl):
        print(f"  ℹ️  All BL returns < risk-free ({RISK_FREE_INR:.1%}) — "
              f"switching to min_volatility for a valid solution.")
        effective_method = "min_volatility"

    def _attempt_mvo(mu, cov):
        ef = EfficientFrontier(mu, cov, weight_bounds=bounds)
        if effective_method in ("min_volatility", "min_cvar"):
            ef.min_volatility()
        else:
            ef.max_sharpe(risk_free_rate=RISK_FREE_INR)
        return ef.clean_weights()

    def _attempt_cvar(mu, prices_path):
        """CVaR uses historical returns, not a covariance matrix."""
        try:
            prices_df = pd.read_csv(prices_path, index_col=0, parse_dates=True)
            hist_cols = [t for t in tickers if t in prices_df.columns]
            if len(hist_cols) < 3:
                return None
            hist_prices = prices_df[hist_cols].dropna().iloc[-252:]
            hist_returns = hist_prices.pct_change().dropna()
            mu_cvar = expected_returns.mean_historical_return(hist_prices)
            ef_cvar = EfficientCVaR(mu_cvar, hist_returns,
                                    weight_bounds=[(b[0], b[1]) for b in bounds[:len(hist_cols)]])
            ef_cvar.min_cvar()
            w = ef_cvar.clean_weights()
            # fill missing tickers with 0
            return {t: w.get(t, 0.0) for t in tickers}
        except Exception:
            return None

    # CVaR path (conservative profile)
    if method == "min_cvar":
        prices_path = f"{DATA_DIR}/prices.csv"
        w = _attempt_cvar(mu_bl, prices_path)
        if w is not None:
            return w
        # fallback to min_volatility if CVaR fails

    # MVO — attempt 1: BL posterior covariance
    try:
        return _attempt_mvo(mu_bl, S_bl)
    except Exception:
        pass

    # MVO — attempt 2: stronger regularisation
    try:
        S_reg = pd.DataFrame(
            np.array(S_bl) + np.eye(len(mu_bl)) * 1e-4,
            index=S_bl.index, columns=S_bl.columns,
        )
        return _attempt_mvo(mu_bl, S_reg)
    except Exception:
        pass

    # Fallback: equal weights
    n = len(tickers)
    return {t: 1.0 / n for t in tickers}


def _apply_macro(weights: dict) -> tuple:
    """
    Load prices.csv, run macro snapshot, and scale weights by regime × VIX.

    Returns (scaled_weights, macro_snapshot, cash_buffer, combined_scale).
    Falls back gracefully if macro_overlay fails.
    """
    try:
        prices = pd.read_csv(f"{DATA_DIR}/prices.csv", index_col=0, parse_dates=True)
        snap   = get_macro_snapshot(prices)
        scaled, cash_buf, scale = apply_macro_overlay(
            weights, snap["regime"], snap["vix"]
        )
        print(f"\n  🌍 Macro Overlay | {snap['regime']['label']}  "
              f"VIX {snap['vix']['vix']:.1f} ({snap['vix']['fear_level']}) | "
              f"Scale {scale:.0%} | Cash buffer {cash_buf:.0%}")
        return scaled, snap, cash_buf, scale
    except Exception as e:
        print(f"  ⚠️  Macro overlay skipped: {e}")
        return weights, None, 0.0, 1.0


def _weights_to_allocation(weights: dict, total_inr: float,
                            prices_inr: pd.Series) -> pd.DataFrame:
    """
    Convert portfolio weights + total INR → allocation table.

    Returns DataFrame with columns:
        ticker, target_weight, target_inr, price_inr, shares, invested_inr, cash_leftover
    """
    rows = []
    for ticker, weight in weights.items():
        if weight <= 0:
            continue
        target_inr = weight * total_inr
        price      = float(prices_inr.get(ticker, 0))
        if price <= 0:
            shares        = 0.0
            invested_inr  = 0.0
        else:
            # Support fractional shares (as offered by most international brokers)
            shares       = round(target_inr / price, 4)
            invested_inr = round(shares * price, 2)

        rows.append({
            "ticker":        ticker,
            "target_weight": round(weight, 4),
            "target_inr":    round(target_inr, 2),
            "price_inr":     round(price, 2),
            "shares":        shares,
            "invested_inr":  invested_inr,
        })

    df = pd.DataFrame(rows).sort_values("target_weight", ascending=False).reset_index(drop=True)
    df["cash_leftover"] = round(total_inr - df["invested_inr"].sum(), 2)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API — MODE 1: FRESH INVESTMENT
# ══════════════════════════════════════════════════════════════════════════════

def optimize_fresh_investment(
    investment_inr:  float,
    risk_profile:    str = "moderate",
    analysis_method: str = "llm",
) -> dict:
    """
    Optimise a fresh investment of `investment_inr` (₹).

    Parameters
    ----------
    investment_inr : float   — total amount to deploy, in INR
    risk_profile   : str     — 'conservative' | 'moderate' | 'aggressive'

    Returns
    -------
    dict:
        allocation   : pd.DataFrame  — per-stock amounts, shares, weights
        weights      : dict          — raw weights
        summary      : dict          — Sharpe, expected return, vol, total deployed
        features     : dict          — BL features (for display)
    """
    print(f"\n💰 FRESH INVESTMENT OPTIMISER  |  {_fmt_inr(investment_inr)}  |  {risk_profile}  |  {analysis_method}")
    print("=" * 60)

    features     = build_features(analysis_method=analysis_method)
    mu_bl        = features["mu_bl"]
    S_bl         = features["S_bl"]
    sentiment_df = features["sentiment_df"]
    prices_inr   = features["prices_inr"]

    weights    = _run_ef(mu_bl, S_bl, sentiment_df, risk_profile, analysis_method)
    weights, macro_snap, cash_buf, macro_scale = _apply_macro(weights)

    # ── Risk Manager: vol targeting + drawdown control + position limits ──────
    try:
        weights, rm_cash, rm_info = apply_all_risk_controls(weights)
        cash_buf = max(cash_buf, rm_info["total_cash"])
        print(f"  ✅ Risk controls applied  |  Cash buffer: {rm_info['total_cash']:.1%}  "
              f"Vol scale: {rm_info['vol_scale']:.2f}  Regime: {rm_info['dd_regime']}")
    except Exception as e:
        rm_info = {}
        print(f"  ⚠️  Risk manager skipped: {e}")

    allocation = _weights_to_allocation(weights, investment_inr, prices_inr)

    # Portfolio-level performance estimate
    w_arr    = np.array([weights.get(t, 0) for t in mu_bl.index])
    exp_ret  = float(w_arr @ mu_bl.values)
    port_var = float(w_arr @ features["S_bl"].values @ w_arr)
    port_vol = float(np.sqrt(port_var))
    sharpe   = (exp_ret - RISK_FREE_INR) / port_vol if port_vol > 0 else 0.0

    cash_inr = investment_inr * cash_buf

    summary = {
        "total_investment_inr": investment_inr,
        "total_deployed_inr":   float(allocation["invested_inr"].sum()),
        "cash_buffer_inr":      round(cash_inr, 2),
        "cash_leftover_inr":    float(allocation["cash_leftover"].iloc[0]) if not allocation.empty else 0.0,
        "expected_return":      round(exp_ret,  4),
        "expected_volatility":  round(port_vol, 4),
        "sharpe_ratio":         round(sharpe,   4),
        "fx_rate":              features["fx_rate"],
        "risk_profile":         risk_profile,
        "macro_scale":          round(macro_scale, 4),
        "macro_cash_buffer":    round(cash_buf, 4),
        "rm_vol_scale":         round(rm_info.get("vol_scale", 1.0), 4),
        "rm_dd_regime":         rm_info.get("dd_regime", "n/a"),
    }

    # Print
    print(f"\n  {'STOCK':<7} {'WEIGHT':>7}  {'AMOUNT':>12}  {'SHARES':>8}  SENTIMENT")
    print(f"  {'─'*55}")
    for _, row in allocation.iterrows():
        t    = row["ticker"]
        sent = sentiment_df.loc[t, "label"] if t in sentiment_df.index else "n/a"
        print(f"  {t:<7} {row['target_weight']:>7.2%}  "
              f"{_fmt_inr(row['invested_inr']):>12}  "
              f"{row['shares']:>8.4f}  {sent}")

    print(f"\n  Total deployed : {_fmt_inr(summary['total_deployed_inr'])}")
    print(f"  Cash leftover  : {_fmt_inr(summary['cash_leftover_inr'])}")
    print(f"  Expected return: {summary['expected_return']:.2%}  "
          f"| Volatility: {summary['expected_volatility']:.2%}  "
          f"| Sharpe: {summary['sharpe_ratio']:.2f}")

    return {"allocation": allocation, "weights": weights,
            "summary": summary, "features": features, "macro_snapshot": macro_snap}


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API — MODE 2: PORTFOLIO REBALANCER
# ══════════════════════════════════════════════════════════════════════════════

def optimize_rebalancing(
    current_holdings: dict,
    additional_inr:   float = 0.0,
    risk_profile:     str   = "moderate",
    analysis_method:  str   = "llm",
) -> dict:
    """
    Generate a rebalancing plan for an existing portfolio.

    Parameters
    ----------
    current_holdings : dict  — {ticker: current_value_in_INR}
                               (e.g. {"AAPL": 15000, "MSFT": 8000})
    additional_inr   : float — extra INR the user wants to add now
    risk_profile     : str   — 'conservative' | 'moderate' | 'aggressive'

    Returns
    -------
    dict:
        allocation    : pd.DataFrame — target allocation with actions
        rebalance     : pd.DataFrame — BUY / SELL / HOLD per stock
        summary       : dict         — totals, transaction costs, net cash
        features      : dict         — BL features
    """
    current_total    = sum(current_holdings.values())
    total_capital    = current_total + additional_inr

    print(f"\n🔄 PORTFOLIO REBALANCER  |  current: {_fmt_inr(current_total)}  "
          f"| adding: {_fmt_inr(additional_inr)}  | total: {_fmt_inr(total_capital)}  |  {analysis_method}")
    print("=" * 65)

    features     = build_features(analysis_method=analysis_method)
    mu_bl        = features["mu_bl"]
    S_bl         = features["S_bl"]
    sentiment_df = features["sentiment_df"]
    prices_inr   = features["prices_inr"]

    weights    = _run_ef(mu_bl, S_bl, sentiment_df, risk_profile, analysis_method)
    weights, macro_snap, cash_buf, macro_scale = _apply_macro(weights)

    # ── Risk Manager ──────────────────────────────────────────────────────────
    try:
        weights, rm_cash, rm_info = apply_all_risk_controls(weights)
        cash_buf = max(cash_buf, rm_info["total_cash"])
    except Exception as e:
        rm_info = {}
        print(f"  ⚠️  Risk manager skipped: {e}")

    allocation = _weights_to_allocation(weights, total_capital, prices_inr)

    # ── Rebalancing actions ───────────────────────────────────────────────────
    rebalance_rows = []
    sells_total    = 0.0
    buys_total     = 0.0

    all_tickers = list(set(list(allocation["ticker"]) + list(current_holdings.keys())))

    for ticker in all_tickers:
        current_val = float(current_holdings.get(ticker, 0.0))
        target_row  = allocation[allocation["ticker"] == ticker]
        target_val  = float(target_row["invested_inr"].values[0]) if not target_row.empty else 0.0
        target_wt   = float(target_row["target_weight"].values[0]) if not target_row.empty else 0.0

        diff        = target_val - current_val
        abs_diff    = abs(diff)

        if diff > MIN_TRADE_INR:
            action = "BUY"
            buys_total += abs_diff
        elif diff < -MIN_TRADE_INR:
            action = "SELL"
            sells_total += abs_diff
        else:
            action   = "HOLD"
            abs_diff = 0.0

        price_inr    = float(prices_inr.get(ticker, 0))
        shares_delta = round(diff / price_inr, 4) if price_inr > 0 else 0.0

        sent = sentiment_df.loc[ticker, "label"] if ticker in sentiment_df.index else "n/a"

        rebalance_rows.append({
            "ticker":        ticker,
            "sentiment":     sent,
            "current_inr":   round(current_val, 2),
            "target_inr":    round(target_val,  2),
            "diff_inr":      round(diff,        2),
            "action":        action,
            "trade_inr":     round(abs_diff, 2),
            "shares_delta":  shares_delta,
            "target_weight": round(target_wt, 4),
            "price_inr":     round(price_inr, 2),
        })

    rebalance_df = (pd.DataFrame(rebalance_rows)
                    .sort_values(["action", "trade_inr"], ascending=[True, False])
                    .reset_index(drop=True))

    # Transaction costs — real Zerodha formula (brokerage + STT + exchange + GST + SEBI + stamp)
    try:
        from data_collector import total_trade_cost as _zerodha_cost
        buy_cost  = _zerodha_cost(buys_total,  "buy")
        sell_cost = _zerodha_cost(sells_total, "sell")
        transaction_cost = round(buy_cost + sell_cost, 2)
    except Exception:
        # Fallback: conservative 0.1% estimate if import fails
        sell_cost = sells_total * BROKERAGE_PCT / 2
        transaction_cost = round((buys_total + sells_total) * BROKERAGE_PCT, 2)
    net_cash_from_sell = round(sells_total - sell_cost, 2)

    # Portfolio metrics
    w_arr   = np.array([weights.get(t, 0) for t in mu_bl.index])
    exp_ret = float(w_arr @ mu_bl.values)
    port_vol= float(np.sqrt(w_arr @ features["S_bl"].values @ w_arr))
    sharpe  = (exp_ret - RISK_FREE_INR) / port_vol if port_vol > 0 else 0.0

    summary = {
        "current_value_inr":    current_total,
        "additional_inr":       additional_inr,
        "total_capital_inr":    total_capital,
        "sells_inr":            sells_total,
        "buys_inr":             buys_total,
        "transaction_cost_inr": transaction_cost,
        "net_cash_from_sells":  net_cash_from_sell,
        "cash_buffer_inr":      round(total_capital * cash_buf, 2),
        "expected_return":      round(exp_ret,  4),
        "expected_volatility":  round(port_vol, 4),
        "sharpe_ratio":         round(sharpe,   4),
        "fx_rate":              features["fx_rate"],
        "risk_profile":         risk_profile,
        "macro_scale":          round(macro_scale, 4),
        "macro_cash_buffer":    round(cash_buf, 4),
        "rm_vol_scale":         round(rm_info.get("vol_scale", 1.0), 4),
        "rm_dd_regime":         rm_info.get("dd_regime", "n/a"),
    }

    # Print rebalancing plan
    print(f"\n  {'STOCK':<7} {'CURRENT':>12}  {'TARGET':>12}  {'ACTION':<5}  {'TRADE':>12}  SENTIMENT")
    print(f"  {'─'*65}")
    for _, row in rebalance_df.iterrows():
        arrow = {"BUY": "▲", "SELL": "▼", "HOLD": "─"}.get(row["action"], "─")
        print(f"  {row['ticker']:<7} {_fmt_inr(row['current_inr']):>12}  "
              f"{_fmt_inr(row['target_inr']):>12}  "
              f"{arrow} {row['action']:<4}  "
              f"{_fmt_inr(row['trade_inr']):>12}  {row['sentiment']}")

    print(f"\n  Total to SELL  : {_fmt_inr(sells_total)}")
    print(f"  Total to BUY   : {_fmt_inr(buys_total)}")
    print(f"  Est. brokerage : {_fmt_inr(transaction_cost)}")
    print(f"  Expected return: {exp_ret:.2%}  | Vol: {port_vol:.2%}  | Sharpe: {sharpe:.2f}")

    return {"allocation": allocation, "rebalance": rebalance_df,
            "summary": summary, "features": features, "macro_snapshot": macro_snap}


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API — MODE 3: WALK-FORWARD BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

def run_walk_forward_backtest(alpha_view: float = 0.5) -> tuple:
    """
    Walk-forward backtest over all bi-weekly rebalancing dates.
    No look-ahead bias: at each date, only uses data up to that point.

    Since sentiment is live-only, the backtest uses fixed today's sentiment
    as a static signal — this is noted clearly in output.

    Saves data/backtest_results.csv and data/backtest_metrics.csv.
    Returns (results_df, metrics_df).
    """
    print("\n📅 WALK-FORWARD BACKTEST  |  June 2018 → Present")
    print("=" * 65)

    # Load raw data
    prices_path  = f"{DATA_DIR}/prices.csv"
    returns_path = f"{DATA_DIR}/returns.csv"
    if not os.path.exists(prices_path):
        raise FileNotFoundError("Run: python data_collector.py first.")

    prices  = pd.read_csv(prices_path,  index_col=0, parse_dates=True)
    returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)

    stock_prices  = prices[[t for t in STOCKS if t in prices.columns]].dropna()
    stock_returns = returns[[t for t in STOCKS if t in returns.columns]].dropna()
    sp500_returns = returns["NIFTY50"].dropna() if "NIFTY50" in returns.columns else pd.Series(dtype=float)

    # Load sentiment (static for backtest)
    try:
        from sentiment_engine import load_sentiment_scores
        sentiment_df = load_sentiment_scores()
    except FileNotFoundError:
        sentiment_df = None
        print("  ⚠️  No sentiment — running pure-quant baseline only.")

    # Load market caps
    mcap_path = f"{DATA_DIR}/market_caps.csv"
    if os.path.exists(mcap_path):
        mcaps = pd.read_csv(mcap_path, index_col=0).iloc[:, 0]
    else:
        mcaps = pd.Series({t: 1.0 for t in STOCKS})

    # Rebalancing dates
    all_dates   = stock_returns.index.tolist()
    rebal_dates = all_dates[::10]   # every 10 trading days (~2 weeks)
    rebal_dates = [d for d in rebal_dates
                   if stock_returns.index.get_loc(d) >= LOOKBACK]

    print(f"   Rebalancing periods : {len(rebal_dates)}")
    print(f"   Sentiment signal    : {'FinBERT (static)' if sentiment_df is not None else 'None (baseline only)'}\n")

    records = []

    for i, rebal_date in enumerate(rebal_dates):
        idx   = stock_returns.index.get_loc(rebal_date)
        hist  = stock_returns.iloc[max(0, idx - LOOKBACK):idx].dropna()

        if hist.shape[0] < 40:
            continue

        try:
            # sample_cov needs PRICES — slice the price window for this period
            price_hist = stock_prices.loc[:rebal_date].iloc[-(LOOKBACK + 1):]
            price_hist = price_hist[[c for c in hist.columns if c in price_hist.columns]]
            try:
                S = risk_models.CovarianceShrinkage(price_hist, frequency=252).ledoit_wolf()
            except Exception:
                S_raw = risk_models.sample_cov(price_hist, frequency=252)
                S = pd.DataFrame(fix_nonpsd(S_raw), index=price_hist.columns, columns=price_hist.columns)
            w_mkt_s = mcaps.reindex(hist.columns).fillna(mcaps.mean())
            w_mkt_s = w_mkt_s / w_mkt_s.sum()
            mu_eq   = market_implied_prior_returns(w_mkt_s, DELTA, S)

            # BL with sentiment
            if sentiment_df is not None:
                from sentiment_engine import get_bl_views
                sent_al = sentiment_df.reindex(hist.columns)
                viewdict, confs = get_bl_views(sent_al, mu_eq, S)
                bl    = BlackLittermanModel(S, absolute_views=viewdict,
                                            pi=mu_eq, omega="idzorek",
                                            view_confidences=confs, tau=TAU)
                mu_bl    = bl.bl_returns()
                S_bl_arr = np.array(bl.bl_cov()) + np.eye(len(hist.columns)) * 1e-8
                S_bl     = pd.DataFrame(fix_nonpsd(S_bl_arr),
                                        index=hist.columns, columns=hist.columns)
                constraints = get_sentiment_constraints(sent_al)
                lo = constraints["lower_bounds"]
                hi = constraints["upper_bounds"]
                bounds_sent = [(float(lo.get(t, 0.0)), float(hi.get(t, 0.30))) for t in mu_bl.index]
            else:
                mu_bl, S_bl = mu_eq, S
                bounds_sent = [(0.0, 0.30)] * len(mu_eq)

            # Baseline: pure-quant max Sharpe
            bounds_base = [(0.0, 0.30)] * len(mu_eq)

            def _opt(mu, cov, bounds):
                try:
                    ef = EfficientFrontier(mu, cov, weight_bounds=bounds)
                    ef.max_sharpe(risk_free_rate=RISK_FREE_INR)
                    return ef.clean_weights()
                except Exception:
                    n = len(mu)
                    return {t: 1.0 / n for t in mu.index}

            w_sent = _opt(mu_bl, S_bl, bounds_sent)
            w_base = _opt(mu_eq,  S,   bounds_base)

        except Exception:
            continue

        # Realised returns for next period
        next_date     = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else stock_returns.index[-1]
        period_mask   = (stock_returns.index >= rebal_date) & (stock_returns.index < next_date)
        period_rets   = stock_returns[period_mask]

        if period_rets.empty:
            continue

        period_compound = (1 + period_rets).prod() - 1

        def _port_ret(w):
            return sum(w.get(t, 0) * period_compound.get(t, 0) for t in w)

        ret_sent  = _port_ret(w_sent)
        ret_base  = _port_ret(w_base)
        sp_period = sp500_returns[period_mask]
        ret_sp500 = float((1 + sp_period).prod() - 1) if not sp_period.empty else float("nan")

        records.append({
            "date":          rebal_date,
            "ret_sentiment": round(ret_sent,  6),
            "ret_baseline":  round(ret_base,  6),
            "ret_nifty50":   round(ret_sp500, 6) if not np.isnan(ret_sp500) else np.nan,
        })

        if (i + 1) % 20 == 0:
            print(f"  [{i+1:3d}/{len(rebal_dates)}] {str(rebal_date.date())}  "
                  f"BL={ret_sent:+.2%}  base={ret_base:+.2%}  nifty={ret_sp500:+.2%}")

    if not records:
        print("❌ No backtest records generated.")
        return pd.DataFrame(), pd.DataFrame()

    results_df = pd.DataFrame(records).set_index("date")

    def _metrics(s: pd.Series, label: str) -> dict:
        s   = s.dropna()
        if s.empty:
            return {"label": label}
        PPY = 26   # periods per year (bi-weekly)
        cum = (1 + s).cumprod()
        tot = cum.iloc[-1] - 1
        ann = (1 + tot) ** (PPY / len(s)) - 1
        vol = s.std() * np.sqrt(PPY)
        shr = (ann - RISK_FREE_INR) / vol if vol > 0 else 0
        mdd = ((cum - cum.cummax()) / cum.cummax()).min()
        cal = ann / abs(mdd) if mdd != 0 else 0
        return {"label": label, "cumulative_ret": round(tot, 4),
                "ann_return": round(ann, 4), "ann_vol": round(vol, 4),
                "sharpe": round(shr, 4), "max_drawdown": round(mdd, 4),
                "calmar": round(cal, 4), "n_periods": len(s)}

    m_s   = _metrics(results_df["ret_sentiment"], "BL + Sentiment")
    m_b   = _metrics(results_df["ret_baseline"],  "Pure Quant (Baseline)")
    m_sp  = _metrics(results_df["ret_nifty50"].dropna(), "Nifty 50")
    alpha = round(m_s.get("ann_return", 0) - m_b.get("ann_return", 0), 4)
    m_s["sentiment_alpha"] = alpha
    m_b["sentiment_alpha"] = 0.0
    m_sp["sentiment_alpha"] = float("nan")

    metrics_df = pd.DataFrame([m_s, m_b, m_sp])

    os.makedirs(DATA_DIR, exist_ok=True)
    results_df.to_csv(f"{DATA_DIR}/backtest_results.csv")
    metrics_df.to_csv(f"{DATA_DIR}/backtest_metrics.csv", index=False)

    # Summary print
    print(f"\n{'═'*65}\n   BACKTEST SUMMARY\n{'═'*65}")
    print(f"  {'Metric':<22} {'BL+Sent':>12} {'Baseline':>12} {'Nifty 50':>12}")
    print(f"  {'─'*60}")
    for display, key, fmt in [
        ("Cumul. Return",  "cumulative_ret", ".1%"),
        ("Ann. Return",    "ann_return",     ".2%"),
        ("Volatility",     "ann_vol",        ".2%"),
        ("Sharpe Ratio",   "sharpe",         ".3f"),
        ("Max Drawdown",   "max_drawdown",   ".2%"),
        ("Calmar Ratio",   "calmar",         ".3f"),
        ("Sentiment Alpha","sentiment_alpha", ".2%"),
    ]:
        vals = [metrics_df[metrics_df["label"] == lbl][key].values
                for lbl in ["BL + Sentiment", "Pure Quant (Baseline)", "Nifty 50"]]
        def fv(v):
            return "n/a".rjust(12) if not len(v) or pd.isna(v[0]) else format(v[0], fmt).rjust(12)
        print(f"  {display:<22} {fv(vals[0])} {fv(vals[1])} {fv(vals[2])}")

    print(f"\n  Saved: data/backtest_results.csv  &  data/backtest_metrics.csv")
    return results_df, metrics_df


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if "--backtest" in sys.argv:
        run_walk_forward_backtest()
    else:
        _method = "combined" if "--combined" in sys.argv else \
                  "sentiment" if "--sentiment" in sys.argv else "llm"
        result = optimize_fresh_investment(100_000, risk_profile="moderate", analysis_method=_method)
        print("\nTip: run with --backtest flag for full historical backtest.")
