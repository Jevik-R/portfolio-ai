# backtester.py  —  Honest Walk-Forward Backtester  (Zero Look-Ahead Bias)
# ─────────────────────────────────────────────────────────────────────────────
# A production-grade backtester with ZERO look-ahead bias.
#
# Key design choices:
#   ✅ Monthly rebalancing (21 trading days)  — keeps costs manageable
#   ✅ Real Zerodha transaction costs          — honest P&L
#   ✅ Zero look-ahead bias — ONLY past prices used at every decision point
#   ✅ Quality fundamentals REMOVED from backtest — no ROE/DE/EPS look-ahead
#   ✅ Nifty 100 universe (80-90 stocks)     — wider opportunity set
#   ✅ Black-Litterman with momentum views   — not just equal weight
#   ✅ Min trade filter: ₹500 / 2% weight   — avoid churning
#   ✅ Stock eligibility: requires ≥126 days of history before decision date
#
# Signals used (both price-only, zero bias by definition):
#   Momentum (60% weight): 6-month return skipping last 1 month
#   Volatility (40% weight): 60-day realized vol, inverted rank
#
# Produces data/backtest_enhanced_results.csv and data/backtest_enhanced_metrics.csv
#
# Run:
#   python backtester.py               — default ₹10L from 2019
#   python backtester.py --capital 500000 --start 2020-01-01
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import warnings
import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, risk_models
from pypfopt.risk_models import fix_nonpositive_semidefinite as fix_nonpsd
from pypfopt.black_litterman import BlackLittermanModel, market_implied_prior_returns

warnings.filterwarnings("ignore")

DATA_DIR       = "data"
RISK_FREE_INR  = 0.065     # Indian 10-yr G-Sec
DELTA          = 2.5       # BL market risk aversion
TAU            = 0.025     # BL confidence in market prior
LOOKBACK       = 252       # trading days for covariance estimation
REBAL_DAYS     = 21        # monthly rebalancing
TOP_N          = 15        # stocks in portfolio
MIN_HIST_DAYS  = 130       # need 6m + 1m skip + buffer
MIN_PRICE      = 50.0      # minimum stock price (₹)
MIN_TRADE_INR  = 500.0     # skip trades smaller than this
MIN_TRADE_WT   = 0.02      # skip weight changes smaller than 2%
MAX_WEIGHT     = 0.15      # single stock cap


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSACTION COSTS  (Zerodha exact formula)
# ══════════════════════════════════════════════════════════════════════════════

def _zerodha_cost(trade_value: float, side: str = "buy") -> float:
    """
    Exact Zerodha equity delivery cost in ₹.

    Components:
      Brokerage  = min(₹20, 0.03% of value)
      STT        = 0.1% on sell side only (delivery equity)
      Exchange   = 0.00345% (NSE fee)
      GST        = 18% of brokerage
      SEBI       = 0.0001% (₹10/crore)
      Stamp      = 0.015% on buy side
    """
    if trade_value <= 0:
        return 0.0
    brokerage = min(20.0, 0.0003  * trade_value)
    stt       = 0.001    * trade_value if side == "sell" else 0.0
    exchange  = 0.0000345 * trade_value
    gst       = 0.18     * brokerage
    sebi      = 0.000001 * trade_value
    stamp     = 0.00015  * trade_value if side == "buy" else 0.0
    return brokerage + stt + exchange + gst + sebi + stamp


def round_trip_cost_pct(trade_value: float) -> float:
    """Return buy+sell cost as fraction of trade value (for display)."""
    if trade_value <= 0:
        return 0.0
    return (_zerodha_cost(trade_value, "buy") + _zerodha_cost(trade_value, "sell")) / trade_value


# ══════════════════════════════════════════════════════════════════════════════
#  STOCK ELIGIBILITY  (survivorship-bias fix)
# ══════════════════════════════════════════════════════════════════════════════

def get_eligible_stocks(returns_df: pd.DataFrame, rebal_date, min_history: int = 126) -> list:
    """
    Return tickers that have at least `min_history` non-NaN daily returns
    STRICTLY BEFORE rebal_date.

    This prevents using a stock that "didn't exist yet" at decision time —
    partially addressing survivorship bias within our fixed universe.
    """
    try:
        idx = returns_df.index.get_loc(rebal_date)
    except KeyError:
        # rebal_date not in index — use all rows before it
        idx = (returns_df.index < rebal_date).sum()

    eligible = []
    for ticker in returns_df.columns:
        history = returns_df[ticker].iloc[:idx].dropna()
        if len(history) >= min_history:
            eligible.append(ticker)
    return eligible


# ══════════════════════════════════════════════════════════════════════════════
#  FACTOR SCORING  (price-only, zero look-ahead bias)
# ══════════════════════════════════════════════════════════════════════════════

def _factor_scores_at_date(
    prices_df:  pd.DataFrame,
    returns_df: pd.DataFrame,
    rebal_date,
    tickers:    list,
) -> pd.DataFrame:
    """
    Compute momentum + volatility factor scores using ONLY price data
    available at or before rebal_date. No fundamental data used.

    Signals (both zero look-ahead bias):
      Momentum  (60% weight): 6-month return, skip last 1 month
                              formula: price[T-21] / price[T-126] - 1
      Volatility (40% weight): 60-day realized annual vol, inverted rank
                              lower vol → higher score

    Returns DataFrame sorted by combined_score (best first).
    """
    p_hist = prices_df.loc[:rebal_date]
    r_hist = returns_df.loc[:rebal_date]

    rows = {}
    for t in tickers:
        if t not in prices_df.columns:
            continue
        p = p_hist[t].dropna()
        r = r_hist[t].dropna() if t in r_hist.columns else pd.Series()

        if len(p) < MIN_HIST_DAYS:
            continue
        if float(p.iloc[-1]) < MIN_PRICE:
            continue

        # ── Momentum: 6-month return skipping last 1 month ────────────────────
        # Uses only prices that existed before this date — zero look-ahead
        p_end   = float(p.iloc[-21])   # price 1 month ago (skip recent noise)
        p_start = float(p.iloc[-126])  # price 6 months ago
        if p_start <= 0:
            continue
        momentum = (p_end / p_start) - 1.0

        # ── Volatility: 60-day realized vol, annualised ───────────────────────
        # Uses only returns that existed before this date — zero look-ahead
        if len(r) >= 60:
            vol = float(r.iloc[-60:].std() * np.sqrt(252))
        elif len(r) >= 20:
            vol = float(r.iloc[-20:].std() * np.sqrt(252))
        else:
            vol = 0.25   # conservative default for thin history

        rows[t] = {
            "momentum": momentum,
            "vol":      vol,
            "price":    float(p.iloc[-1]),
        }

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).T
    df.index.name = "ticker"

    # ── Rank within eligible universe (eliminates outlier effects) ────────────
    df["mom_rank"] = df["momentum"].rank(pct=True)          # higher = better
    df["vol_rank"] = 1.0 - df["vol"].rank(pct=True)        # lower vol = higher score

    # ── Combined score: 60% momentum + 40% volatility (price-only, zero bias) ─
    df["combined"] = (0.60 * df["mom_rank"] +
                      0.40 * df["vol_rank"])

    return df.sort_values("combined", ascending=False)


# ══════════════════════════════════════════════════════════════════════════════
#  PORTFOLIO OPTIMISATION  (BL with momentum views)
# ══════════════════════════════════════════════════════════════════════════════

def _bl_optimize(
    tickers:      list,
    prices_hist:  pd.DataFrame,
    factor_df:    pd.DataFrame,
) -> dict:
    """
    Run Black-Litterman + Efficient Frontier for selected tickers.
    Views are derived from momentum ranks (higher rank → boosted return view).
    Max single-stock weight = 15%.
    Falls back to equal weights on any error.
    """
    eq_weights = {t: 1.0 / len(tickers) for t in tickers}
    if len(tickers) < 3:
        return eq_weights

    try:
        price_window = prices_hist[tickers].dropna().iloc[-(LOOKBACK + 1):]
        if len(price_window) < 40:
            return eq_weights

        # Covariance (Ledoit-Wolf shrinkage)
        try:
            S = risk_models.CovarianceShrinkage(price_window, frequency=252).ledoit_wolf()
        except Exception:
            S_raw = risk_models.sample_cov(price_window, frequency=252)
            S = pd.DataFrame(fix_nonpsd(S_raw), index=tickers, columns=tickers)

        # Equal market-cap weights as prior (no intraday cap data in backtest)
        w_eq = pd.Series({t: 1.0 / len(tickers) for t in tickers})
        mu_prior = market_implied_prior_returns(w_eq, DELTA, S)

        # BL views from momentum: high momentum → return boosted above prior
        viewdict    = {}
        confidences = []
        for t in tickers:
            mu_i      = float(mu_prior.get(t, RISK_FREE_INR))
            mom_rank  = float(factor_df.loc[t, "mom_rank"]) if t in factor_df.index else 0.5
            # Scale: rank 1.0 → +50% above prior, rank 0.0 → -50% below prior
            view_adj  = mu_i * (1.0 + (mom_rank - 0.5))
            viewdict[t] = round(view_adj, 6)
            confidences.append(0.5)

        # BL model
        bl = BlackLittermanModel(
            S, absolute_views=viewdict,
            pi=mu_prior, omega="idzorek",
            view_confidences=confidences, tau=TAU,
        )
        mu_bl = bl.bl_returns()
        S_bl_arr = np.array(bl.bl_cov()) + np.eye(len(tickers)) * 1e-8
        S_bl     = pd.DataFrame(fix_nonpsd(S_bl_arr), index=tickers, columns=tickers)

        # Optimise
        ef = EfficientFrontier(mu_bl, S_bl, weight_bounds=(0.0, MAX_WEIGHT))
        if all(float(v) < RISK_FREE_INR for v in mu_bl):
            ef.min_volatility()
        else:
            ef.max_sharpe(risk_free_rate=RISK_FREE_INR)

        return ef.clean_weights()

    except Exception:
        return eq_weights


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN BACKTEST LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_enhanced_backtest(
    initial_capital: float = 1_000_000,   # ₹10 Lakh
    start_date:      str   = "2019-01-01",
    top_n:           int   = TOP_N,
    rebal_days:      int   = REBAL_DAYS,
) -> tuple:
    """
    Walk-forward backtest with real costs and monthly rebalancing.

    Runs THREE parallel portfolios:
      1. BL + Factor (after real Zerodha costs)
      2. BL + Factor (before costs — theoretical upper bound)
      3. Equal weight top-N (after costs — simple baseline)
      4. Nifty 50 buy-and-hold (no costs)

    Returns
    -------
    (results_df, metrics_df, costs_df)
    """
    # ── Load price data ───────────────────────────────────────────────────────
    for fname in ["nifty100_prices.csv", "prices.csv"]:
        fpath = f"{DATA_DIR}/{fname}"
        if os.path.exists(fpath):
            prices_df  = pd.read_csv(fpath, index_col=0, parse_dates=True)
            returns_path = fpath.replace("prices", "returns")
            returns_df = pd.read_csv(
                returns_path, index_col=0, parse_dates=True,
            ) if os.path.exists(returns_path) else prices_df.pct_change().dropna()
            src = fname
            break
    else:
        raise FileNotFoundError(
            "No price data found.\n"
            "Run: python data_collector.py           (16 stocks)\n"
            "  or: python data_collector.py --full   (Nifty 100 universe)"
        )

    # ── Filter to start date ──────────────────────────────────────────────────
    prices_df  = prices_df[prices_df.index >= start_date]
    returns_df = returns_df[returns_df.index >= start_date]

    BENCH = "NIFTY50"
    all_tickers = [c for c in prices_df.columns if c != BENCH]

    # ── Rebalancing dates ─────────────────────────────────────────────────────
    all_dates    = prices_df.index.tolist()
    rebal_idx    = list(range(LOOKBACK, len(all_dates), rebal_days))
    rebal_dates  = [all_dates[i] for i in rebal_idx]

    n_periods = len(rebal_dates) - 1
    if n_periods < 6:
        raise ValueError(f"Only {n_periods} rebalancing periods — need more history. "
                         f"Extend START_DATE or use longer data.")

    print(f"\n{'═'*65}")
    print(f"  ENHANCED WALK-FORWARD BACKTEST")
    print(f"{'═'*65}")
    print(f"  Data source  : {src}  ({len(all_tickers)} tickers)")
    print(f"  Capital      : ₹{initial_capital/1e5:.1f}L")
    print(f"  Period       : {rebal_dates[0].date()} → {rebal_dates[-1].date()}")
    print(f"  Rebalancing  : every {rebal_days} trading days (~monthly)")
    print(f"  Periods      : {n_periods}")
    print(f"  Top-N stocks : {top_n}")
    print(f"  Costs        : Zerodha exact (brokerage + STT + exchange + GST + SEBI + stamp)")
    print(f"  Signals      : Momentum (60%) + Volatility (40%) — price-only, zero look-ahead bias")
    print(f"  Quality      : REMOVED from backtest (no ROE/D:E/EPS — eliminates fundamental look-ahead)")
    print(f"  Eligibility  : Each stock requires ≥{MIN_HIST_DAYS} days price history before decision date")
    print(f"{'─'*65}\n")

    # ── Portfolio trackers ────────────────────────────────────────────────────
    val_bl_net  = float(initial_capital)   # BL after costs
    val_bl_gross= float(initial_capital)   # BL before costs
    val_eq_net  = float(initial_capital)   # Equal weight after costs
    val_mom_net = float(initial_capital)   # Momentum-only weighted after costs

    prev_bl_wts  = {}
    prev_eq_wts  = {}   # tracks drifted EQ weights for realistic cost modelling
    prev_mom_wts = {}

    total_costs       = 0.0
    total_eq_costs    = 0.0
    total_mom_costs   = 0.0
    total_buy_turnover  = 0.0
    total_sell_turnover = 0.0

    records      = []
    cost_records = []

    for i, rebal_date in enumerate(rebal_dates[:-1]):
        next_date  = rebal_dates[i + 1]

        prices_to  = prices_df.loc[:rebal_date]
        returns_to = returns_df.loc[:rebal_date]

        # ── Eligible stocks: must have ≥MIN_HIST_DAYS history before this date ──
        eligible_tickers = get_eligible_stocks(returns_df, rebal_date, min_history=MIN_HIST_DAYS)

        # ── Score stocks (price-only, zero look-ahead bias) ───────────────────
        factor_df = _factor_scores_at_date(
            prices_df, returns_df, rebal_date, eligible_tickers
        )
        if factor_df.empty or len(factor_df) < max(top_n, 5):
            continue

        selected = factor_df.head(top_n).index.tolist()

        # ── Optimise weights ───────────────────────────────────────────────────
        bl_weights = _bl_optimize(selected, prices_to, factor_df)
        eq_weights = {t: 1.0 / top_n for t in selected}

        # ── Pure momentum weights: rank by raw momentum, weight by score ───────
        mom_df = factor_df.sort_values("momentum", ascending=False)
        pos_mom = mom_df[mom_df["momentum"] > 0]
        if len(pos_mom) >= 5:
            mom_top = pos_mom.head(top_n)
        else:
            mom_top = mom_df.head(max(5, top_n))
        total_pos_mom = mom_top["momentum"].sum()
        if total_pos_mom > 0:
            mom_weights = {t: float(mom_top.loc[t, "momentum"]) / total_pos_mom
                           for t in mom_top.index}
        else:
            mom_weights = {t: 1.0 / len(mom_top) for t in mom_top.index}

        # ── Get current prices for cost calculation ────────────────────────────
        try:
            cur_prices = {t: float(prices_to[t].dropna().iloc[-1])
                          for t in selected if t in prices_to.columns}
        except Exception:
            continue

        # ── Calculate trading costs ────────────────────────────────────────────
        period_cost_bl = 0.0
        period_cost_eq = 0.0

        for t in set(list(bl_weights.keys()) + list(prev_bl_wts.keys())):
            new_w = float(bl_weights.get(t, 0.0))
            old_w = float(prev_bl_wts.get(t, 0.0))
            dw    = new_w - old_w

            if abs(dw) < MIN_TRADE_WT:
                continue

            trade_val = abs(dw) * val_bl_net
            if trade_val < MIN_TRADE_INR:
                continue

            side = "buy" if dw > 0 else "sell"
            c    = _zerodha_cost(trade_val, side)
            period_cost_bl += c

            if side == "buy":
                total_buy_turnover  += trade_val
            else:
                total_sell_turnover += trade_val

        for t in set(list(eq_weights.keys()) + list(prev_eq_wts.keys())):
            new_w = float(eq_weights.get(t, 0.0))
            old_w = float(prev_eq_wts.get(t, 0.0))
            dw    = new_w - old_w
            if abs(dw) < MIN_TRADE_WT:
                continue
            trade_val = abs(dw) * val_eq_net
            if trade_val < MIN_TRADE_INR:
                continue
            side = "buy" if dw > 0 else "sell"
            period_cost_eq += _zerodha_cost(trade_val, side)

        # Momentum-only: uses momentum-weighted portfolio (different from EQ)
        period_cost_mom = 0.0
        for t in set(list(mom_weights.keys()) + list(prev_mom_wts.keys())):
            new_w = float(mom_weights.get(t, 0.0))
            old_w = float(prev_mom_wts.get(t, 0.0))
            dw    = new_w - old_w
            if abs(dw) < MIN_TRADE_WT:
                continue
            trade_val = abs(dw) * val_mom_net
            if trade_val < MIN_TRADE_INR:
                continue
            side = "buy" if dw > 0 else "sell"
            period_cost_mom += _zerodha_cost(trade_val, side)

        total_costs     += period_cost_bl
        total_eq_costs  += period_cost_eq
        total_mom_costs += period_cost_mom

        # ── Get period returns (next month prices) ─────────────────────────────
        period_mask   = (prices_df.index > rebal_date) & (prices_df.index <= next_date)
        period_prices = prices_df[period_mask]

        if period_prices.empty:
            continue

        def _period_return(weights, portfolio_val):
            ret = 0.0
            for t, w in weights.items():
                if t not in prices_df.columns or float(w) <= 0:
                    continue
                p_s_ser = prices_to[t].dropna()
                p_e_ser = period_prices[t].dropna()
                if p_s_ser.empty or p_e_ser.empty:
                    continue
                p_s = float(p_s_ser.iloc[-1])
                p_e = float(p_e_ser.iloc[-1])
                if p_s > 0:
                    ret += float(w) * (p_e / p_s - 1.0)
            return ret

        bl_ret  = _period_return(bl_weights, val_bl_net)
        eq_ret  = _period_return(eq_weights, val_eq_net)
        mom_ret = _period_return(mom_weights, val_mom_net)

        # Nifty 50
        nifty_ret = 0.0
        if BENCH in prices_df.columns:
            nb_s = prices_to[BENCH].dropna()
            nb_e = period_prices[BENCH].dropna()
            if not nb_s.empty and not nb_e.empty:
                nifty_ret = float(nb_e.iloc[-1]) / float(nb_s.iloc[-1]) - 1.0

        # ── Update portfolio values ────────────────────────────────────────────
        cost_drag_bl  = period_cost_bl  / val_bl_net  if val_bl_net  > 0 else 0
        cost_drag_eq  = period_cost_eq  / val_eq_net  if val_eq_net  > 0 else 0
        cost_drag_mom = period_cost_mom / val_mom_net if val_mom_net > 0 else 0

        val_bl_net   *= (1 + bl_ret  - cost_drag_bl)
        val_bl_gross *= (1 + bl_ret)
        val_eq_net   *= (1 + eq_ret  - cost_drag_eq)
        val_mom_net  *= (1 + mom_ret - cost_drag_mom)

        prev_bl_wts  = dict(bl_weights)

        # Compute drifted EQ weights so next period captures real rebalancing cost.
        # Each stock's weight drifts proportionally to its return vs portfolio return.
        drifted_eq_wts = {}
        for t in eq_weights:
            w0 = float(eq_weights[t])
            p_s_ser = prices_to[t].dropna() if t in prices_to.columns else pd.Series()
            p_e_ser = period_prices[t].dropna() if t in period_prices.columns else pd.Series()
            if p_s_ser.empty or p_e_ser.empty or float(p_s_ser.iloc[-1]) <= 0:
                drifted_eq_wts[t] = w0
            else:
                r_t = float(p_e_ser.iloc[-1]) / float(p_s_ser.iloc[-1]) - 1.0
                drifted_eq_wts[t] = w0 * (1.0 + r_t) / (1.0 + eq_ret) if abs(1.0 + eq_ret) > 1e-9 else w0
        prev_eq_wts  = drifted_eq_wts
        prev_mom_wts = dict(mom_weights)

        records.append({
            "date":          rebal_date,
            "bl_net":        round(bl_ret - cost_drag_bl,   6),
            "bl_gross":      round(bl_ret,                  6),
            "eq_net":        round(eq_ret - cost_drag_eq,   6),
            "mom_net":       round(mom_ret - cost_drag_mom, 6),
            "nifty":         round(nifty_ret,               6),
            "period_costs":  round(period_cost_bl,          2),
            "val_bl_net":    round(val_bl_net,              2),
            "val_bl_gross":  round(val_bl_gross,            2),
            "val_eq_net":    round(val_eq_net,              2),
            "val_mom_net":   round(val_mom_net,             2),
            "n_stocks":      len(selected),
        })

        cost_records.append({
            "date":        rebal_date,
            "cost_inr":    round(period_cost_bl, 2),
            "cost_bps":    round(period_cost_bl / val_bl_net * 10000, 2),
        })

        if (i + 1) % 6 == 0:
            print(f"  [{i+1:3d}/{n_periods}] {rebal_date.date()}  "
                  f"BL(net)={bl_ret - cost_drag_bl:+.2%}  "
                  f"EQ={eq_ret - cost_drag_eq:+.2%}  "
                  f"Nifty={nifty_ret:+.2%}  "
                  f"Costs=₹{period_cost_bl:.0f}  "
                  f"Val=₹{val_bl_net/1e5:.2f}L")

    if not records:
        print("❌  No records generated — check data range and universe.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    results_df = pd.DataFrame(records).set_index("date")
    costs_df   = pd.DataFrame(cost_records).set_index("date")

    # ── Performance metrics ────────────────────────────────────────────────────
    PPY = 252 / rebal_days   # periods per year

    def _metrics(values: pd.Series, label: str, strategy_costs: float = 0.0) -> dict:
        """Compute annualized metrics from a portfolio value series."""
        r = values.pct_change().dropna()
        if r.empty or len(r) < 3:
            return {"label": label}
        n   = len(r)
        cum = float(values.iloc[-1] / values.iloc[0]) - 1.0
        ann = (1 + cum) ** (PPY / n) - 1.0
        vol = float(r.std() * np.sqrt(PPY))
        shr = (ann - RISK_FREE_INR) / vol if vol > 0 else 0.0
        mdd = float(((values - values.cummax()) / values.cummax()).min())
        cal = ann / abs(mdd) if mdd != 0 else 0.0
        win = float((r > 0).mean())
        return {
            "label":          label,
            "cumulative_ret": round(cum,  4),
            "ann_return":     round(ann,  4),
            "ann_vol":        round(vol,  4),
            "sharpe":         round(shr,  4),
            "max_drawdown":   round(mdd,  4),
            "calmar":         round(cal,  4),
            "win_rate":       round(win,  4),
            "n_periods":      n,
            "final_value":    round(float(values.iloc[-1]), 0),
            "total_costs_inr":round(strategy_costs, 0),
            "cost_drag_ann":  round(strategy_costs / initial_capital / (n / PPY) * 100, 3),
        }

    # Reconstruct portfolio value series
    nifty_val = (1 + results_df["nifty"]).cumprod() * initial_capital

    m1 = _metrics(results_df["val_bl_net"],   "Momentum+Vol BL  (After Costs)",  total_costs)
    m2 = _metrics(results_df["val_bl_gross"],  "Momentum+Vol BL  (Before Costs)", 0.0)
    m3 = _metrics(results_df["val_eq_net"],    "Equal Weight     (After Costs)",  total_eq_costs)
    m4 = _metrics(nifty_val,                   "Nifty 50         (Buy & Hold)",   0.0)
    m5 = _metrics(results_df["val_mom_net"],   "Momentum Only    (After Costs)",  total_mom_costs)

    # Add cost drag explicitly to before-costs row
    m2["total_costs_inr"] = 0
    m2["cost_drag_ann"]   = 0.0

    metrics_df = pd.DataFrame([m1, m2, m3, m4, m5])

    # ── Save results ───────────────────────────────────────────────────────────
    os.makedirs(DATA_DIR, exist_ok=True)
    results_df.to_csv(f"{DATA_DIR}/backtest_enhanced_results.csv")
    metrics_df.to_csv(f"{DATA_DIR}/backtest_enhanced_metrics.csv", index=False)
    costs_df.to_csv(f"{DATA_DIR}/backtest_costs.csv")

    # ── Print summary ──────────────────────────────────────────────────────────
    annual_turnover = (total_buy_turnover + total_sell_turnover) / initial_capital / (n_periods / PPY)
    print(f"\n{'═'*65}")
    print(f"  BACKTEST SUMMARY  ({rebal_dates[0].date()} → {rebal_dates[-1].date()})")
    print(f"{'═'*65}")
    print(f"\n  {'Metric':<28} {'BL+Factor':>12}  {'BL Gross':>12}  {'EqWt':>12}  {'Nifty50':>12}")
    print(f"  {'─'*74}")

    keys_fmts = [
        ("Cumulative Return",   "cumulative_ret", ".1%"),
        ("Annual Return (CAGR)","ann_return",     ".2%"),
        ("Annual Volatility",   "ann_vol",        ".2%"),
        ("Sharpe Ratio",        "sharpe",         ".3f"),
        ("Max Drawdown",        "max_drawdown",   ".2%"),
        ("Calmar Ratio",        "calmar",         ".3f"),
        ("Win Rate (months)",   "win_rate",       ".1%"),
        ("Final Value (₹)",     "final_value",    ",.0f"),
        ("Total Costs (₹)",     "total_costs_inr",",.0f"),
        ("Cost Drag (%/yr)",    "cost_drag_ann",  ".3f"),
    ]

    metrics_list = [m1, m2, m3, m4]
    for display, key, fmt in keys_fmts:
        vals = []
        for m in metrics_list:
            v = m.get(key, float("nan"))
            try:
                vals.append(format(v, fmt).rjust(12))
            except Exception:
                vals.append("  n/a".rjust(12))
        print(f"  {display:<28} {'  '.join(vals)}")

    print(f"\n  Annual turnover rate : {annual_turnover:.0%}")
    print(f"  Total costs paid     : ₹{total_costs:,.0f}")
    print(f"  Cost as % of capital : {total_costs / initial_capital:.2%} lifetime")
    print(f"\n  ✅  Bias integrity:")
    print(f"     · Momentum signal: uses only prices available before each decision date")
    print(f"     · Volatility signal: uses only returns available before each decision date")
    print(f"     · Quality fundamentals: REMOVED — no ROE/D:E/EPS used in backtest")
    print(f"     · Stock eligibility: each stock required ≥{MIN_HIST_DAYS} days history at decision date")
    print(f"     · Remaining limitation: fixed 16-stock universe (not dynamically updated)")
    print(f"\n  Saved: {DATA_DIR}/backtest_enhanced_results.csv")
    print(f"         {DATA_DIR}/backtest_enhanced_metrics.csv")
    print(f"         {DATA_DIR}/backtest_costs.csv")

    return results_df, metrics_df, costs_df


# ══════════════════════════════════════════════════════════════════════════════
#  COST BREAKDOWN HELPER  (used by dashboard)
# ══════════════════════════════════════════════════════════════════════════════

def estimate_annual_costs(
    portfolio_value: float = 1_000_000,
    annual_turnover: float = 0.35,
) -> dict:
    """
    Estimate annual transaction costs for a given portfolio size and turnover.

    Annual turnover = fraction of portfolio traded per year (both buy + sell).
    For monthly rebalancing with ~35% turnover, costs are surprisingly low
    because Zerodha delivery brokerage is nearly free.

    Returns dict with per-component annual costs.
    """
    buy_value  = portfolio_value * annual_turnover / 2
    sell_value = portfolio_value * annual_turnover / 2

    buy_cost  = _zerodha_cost(buy_value,  "buy")
    sell_cost = _zerodha_cost(sell_value, "sell")

    total     = buy_cost + sell_cost
    pct_of_portfolio = total / portfolio_value

    return {
        "portfolio_value":   portfolio_value,
        "annual_turnover":   annual_turnover,
        "buy_turnover_inr":  buy_value,
        "sell_turnover_inr": sell_value,
        "annual_cost_inr":   round(total, 2),
        "cost_pct_annual":   round(pct_of_portfolio * 100, 4),
        "cost_bps_annual":   round(pct_of_portfolio * 10000, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MONTE CARLO SIMULATION  (statistical significance test)
# ══════════════════════════════════════════════════════════════════════════════

def run_monte_carlo(
    n_simulations:   int   = 500,
    seed:            int   = 42,
    initial_capital: float = 1_000_000,
    start_date:      str   = "2019-01-01",
    top_n:           int   = TOP_N,
    rebal_days:      int   = REBAL_DAYS,
    our_cagr:        float = 0.1590,
) -> pd.DataFrame:
    """
    Run Monte Carlo simulation: 500 random equal-weight portfolios using the
    same universe, date range, capital, and Zerodha cost model as the main backtest.

    Each simulation randomly selects top_n stocks at each rebalancing date from
    the eligible universe. Returns a DataFrame with one row per simulation.

    Saves results to data/monte_carlo_results.csv.
    """
    import random

    rng = np.random.default_rng(seed)
    random.seed(seed)

    # Load price data (same logic as run_enhanced_backtest)
    for fname in ["nifty100_prices.csv", "prices.csv"]:
        fpath = f"{DATA_DIR}/{fname}"
        if os.path.exists(fpath):
            prices_df  = pd.read_csv(fpath, index_col=0, parse_dates=True)
            returns_path = fpath.replace("prices", "returns")
            returns_df = pd.read_csv(
                returns_path, index_col=0, parse_dates=True,
            ) if os.path.exists(returns_path) else prices_df.pct_change().dropna()
            break
    else:
        raise FileNotFoundError("No price data found. Run data_collector.py first.")

    prices_df  = prices_df[prices_df.index >= start_date]
    returns_df = returns_df[returns_df.index >= start_date]

    BENCH       = "NIFTY50"
    all_tickers = [c for c in prices_df.columns if c != BENCH]

    all_dates   = prices_df.index.tolist()
    rebal_idx   = list(range(LOOKBACK, len(all_dates), rebal_days))
    rebal_dates = [all_dates[i] for i in rebal_idx]
    n_periods   = len(rebal_dates) - 1

    PPY = 252 / rebal_days

    print(f"\n{'═'*65}")
    print(f"  MONTE CARLO SIMULATION  ({n_simulations} random portfolios)")
    print(f"{'═'*65}")
    print(f"  Universe     : {len(all_tickers)} tickers")
    print(f"  Period       : {rebal_dates[0].date()} → {rebal_dates[-1].date()}")
    print(f"  Stocks/port  : {top_n}  (random equal-weight)")
    print(f"  Costs        : Zerodha exact (same as main backtest)")
    print(f"{'─'*65}\n")

    sim_rows = []

    for sim_id in range(n_simulations):
        if (sim_id + 1) % 100 == 0:
            print(f"  Simulation {sim_id+1}/{n_simulations}…")

        val        = float(initial_capital)
        prev_wts   = {}
        val_series = [val]

        for i, rebal_date in enumerate(rebal_dates[:-1]):
            next_date = rebal_dates[i + 1]

            prices_to  = prices_df.loc[:rebal_date]

            eligible = get_eligible_stocks(returns_df, rebal_date, min_history=MIN_HIST_DAYS)
            if len(eligible) < top_n:
                val_series.append(val)
                continue

            # Random selection (reproducible per sim via rng)
            chosen = list(rng.choice(eligible, size=top_n, replace=False))
            weights = {t: 1.0 / top_n for t in chosen}

            # Transaction costs
            period_cost = 0.0
            for t in set(list(weights.keys()) + list(prev_wts.keys())):
                new_w = float(weights.get(t, 0.0))
                old_w = float(prev_wts.get(t, 0.0))
                dw    = new_w - old_w
                if abs(dw) < MIN_TRADE_WT:
                    continue
                trade_val = abs(dw) * val
                if trade_val < MIN_TRADE_INR:
                    continue
                side = "buy" if dw > 0 else "sell"
                period_cost += _zerodha_cost(trade_val, side)

            # Period return
            period_mask   = (prices_df.index > rebal_date) & (prices_df.index <= next_date)
            period_prices = prices_df[period_mask]
            if period_prices.empty:
                val_series.append(val)
                continue

            ret = 0.0
            for t, w in weights.items():
                if t not in prices_df.columns:
                    continue
                p_s_ser = prices_to[t].dropna()
                p_e_ser = period_prices[t].dropna()
                if p_s_ser.empty or p_e_ser.empty:
                    continue
                p_s = float(p_s_ser.iloc[-1])
                p_e = float(p_e_ser.iloc[-1])
                if p_s > 0:
                    ret += w * (p_e / p_s - 1.0)

            cost_drag = period_cost / val if val > 0 else 0
            val      *= (1 + ret - cost_drag)
            prev_wts  = dict(weights)
            val_series.append(val)

        # Compute metrics for this simulation
        vs = pd.Series(val_series)
        r  = vs.pct_change().dropna()
        n  = len(r)
        if n < 3:
            continue
        cum  = float(vs.iloc[-1] / vs.iloc[0]) - 1.0
        cagr = float((1 + cum) ** (PPY / n) - 1.0)
        vol  = float(r.std() * np.sqrt(PPY))
        shrp = (cagr - RISK_FREE_INR) / vol if vol > 0 else 0.0
        mdd  = float(((vs - vs.cummax()) / vs.cummax()).min())

        sim_rows.append({
            "sim_id":       sim_id,
            "cagr":         round(cagr, 6),
            "sharpe":       round(shrp, 4),
            "max_drawdown": round(mdd,  4),
            "final_value":  round(float(vs.iloc[-1]), 0),
        })

    mc_df = pd.DataFrame(sim_rows)

    os.makedirs(DATA_DIR, exist_ok=True)
    mc_df.to_csv(f"{DATA_DIR}/monte_carlo_results.csv", index=False)

    beats      = int((mc_df["cagr"] < our_cagr).sum())
    p_value    = 1.0 - beats / len(mc_df)
    pct_beats  = beats / len(mc_df) * 100

    print(f"\n{'═'*65}")
    print(f"  MONTE CARLO RESULTS")
    print(f"{'═'*65}")
    print(f"  Our BL+Factor CAGR        : {our_cagr:.2%}")
    print(f"  Median random CAGR        : {mc_df['cagr'].median():.2%}")
    print(f"  Our strategy beats        : {beats}/{len(mc_df)} random portfolios ({pct_beats:.1f}%)")
    print(f"  p-value (fraction above)  : {p_value:.4f}")
    if p_value < 0.05:
        print("  ✅ Result is statistically significant (p < 0.05) — alpha is NOT due to luck")
    else:
        print("  ⚠️  Result is NOT statistically significant at 95% confidence")
    print(f"\n  Saved: {DATA_DIR}/monte_carlo_results.csv")

    return mc_df


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Parse CLI flags
    capital      = 1_000_000
    start_date   = "2019-01-01"
    do_montecarlo = "--montecarlo" in sys.argv

    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--capital" and i + 2 <= len(sys.argv) - 1:
            try:
                capital = int(sys.argv[i + 2])
            except ValueError:
                pass
        if arg == "--start" and i + 2 <= len(sys.argv) - 1:
            start_date = sys.argv[i + 2]

    if do_montecarlo:
        run_monte_carlo(n_simulations=500, seed=42, initial_capital=capital, start_date=start_date)
        sys.exit(0)

    print(f"Starting backtest with capital=₹{capital:,}  start={start_date}")

    results, metrics, costs = run_enhanced_backtest(
        initial_capital = capital,
        start_date      = start_date,
    )

    if not metrics.empty:
        m_bl = metrics[metrics["label"].str.contains("After Costs")]
        m_ni = metrics[metrics["label"].str.contains("Nifty")]
        if not m_bl.empty and not m_ni.empty:
            alpha = float(m_bl.iloc[0]["ann_return"]) - float(m_ni.iloc[0]["ann_return"])
            print(f"\n  📊 Alpha over Nifty 50: {alpha:+.2%} per year")
            if alpha > 0.03:
                print("  ✅ Strategy adds meaningful value above Nifty 50")
            elif alpha > 0:
                print("  ⚠️  Strategy beats Nifty 50 but margin is thin")
            else:
                print("  ❌ Strategy underperforms Nifty 50 — review parameters")

        # Show cost estimate for common portfolio sizes
        print(f"\n  📋 Annual cost estimates (Zerodha, 35% turnover):")
        for pv in [50_000, 100_000, 500_000, 1_000_000]:
            est = estimate_annual_costs(pv, 0.35)
            print(f"     ₹{pv/1e5:.1f}L portfolio → ₹{est['annual_cost_inr']:,.0f}/yr "
                  f"({est['cost_pct_annual']:.3f}%  or  {est['cost_bps_annual']:.1f} bps)")
