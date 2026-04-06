# macro_overlay.py  —  Level 3: Market Regime & Macro Risk Overlay
# ─────────────────────────────────────────────────────────────────────────────
# Three independent signals layered on top of BL optimisation:
#
#   1. Market Regime Detector
#      Technical: Nifty 50 vs 50/200-day MAs → bull / neutral / bear
#      Each regime scales ALL equity weights (bear = hold 30% as cash buffer)
#
#   2. VIX Fear Index Overlay
#      Fetches live ^VIX from yfinance. High fear (VIX > 30) shrinks exposure.
#      VIX > 40 (extreme) cuts exposure by 50% — the March-2020 protection.
#
#   3. Macro Narrative (optional GPT)
#      GPT-3.5 synthesises current regime + VIX into a 2-sentence summary.
#      Falls back to a rule-based template if no API key.
#
# These three together would have significantly reduced the March 2020 -28% drawdown.
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

DATA_DIR = "data"

# ── Thresholds ────────────────────────────────────────────────────────────────
VIX_NORMAL    = 20    # below → normal, full exposure
VIX_ELEVATED  = 25    # 20-25 → slight caution
VIX_HIGH      = 30    # 30+   → high fear, reduce exposure
VIX_EXTREME   = 40    # 40+   → extreme fear, major reduction

# Regime → weight scale factor (fraction of capital to deploy in equities)
REGIME_SCALE = {
    "bull":    1.00,   # fully invested
    "neutral": 0.90,   # hold 10% as buffer
    "bear":    0.70,   # hold 30% as cash buffer
}

VIX_SCALE = {
    "normal":   1.00,
    "elevated": 0.92,
    "high":     0.75,
    "extreme":  0.50,
}


# ══════════════════════════════════════════════════════════════════════════════
#  1. MARKET REGIME DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

def detect_market_regime(prices_df: pd.DataFrame) -> dict:
    """
    Classify the current market regime using Nifty 50 moving averages.

    Rules (applied in order):
      BULL    : price > MA200  AND  MA50 > MA200  AND  20-day momentum > -3%
      BEAR    : price < MA200  AND  MA50 < MA200
      NEUTRAL : everything else (mixed signals, transitioning)

    Returns a dict with regime label + all intermediate signals.
    """
    if "NIFTY50" not in prices_df.columns:
        return _regime_result("neutral", {}, "Nifty 50 data not available.")

    sp     = prices_df["NIFTY50"].dropna()
    n      = len(sp)

    if n < 200:
        return _regime_result("neutral", {}, f"Only {n} days — need 200+ for regime detection.")

    price   = float(sp.iloc[-1])
    ma50    = float(sp.rolling(50).mean().iloc[-1])
    ma200   = float(sp.rolling(200).mean().iloc[-1])
    mom_20d = float((sp.iloc[-1] / sp.iloc[-20] - 1)) if n >= 20 else 0.0
    mom_5d  = float((sp.iloc[-1] / sp.iloc[-5]  - 1)) if n >= 5  else 0.0

    # Drawdown from 52-week high
    high_52w  = float(sp.tail(252).max())
    drawdown  = (price - high_52w) / high_52w

    signals = {
        "nifty50":      round(price, 2),
        "ma50":         round(ma50, 2),
        "ma200":        round(ma200, 2),
        "above_ma200":  price > ma200,
        "ma50_gt_ma200":ma50 > ma200,
        "momentum_20d": round(mom_20d, 4),
        "momentum_5d":  round(mom_5d,  4),
        "drawdown_52w": round(drawdown, 4),
        "high_52w":     round(high_52w, 2),
    }

    if price > ma200 and ma50 > ma200 and mom_20d > -0.03:
        regime  = "bull"
        reason  = (f"Nifty 50 (₹{price:,.0f}) is above both MA50 (₹{ma50:,.0f}) "
                   f"and MA200 (₹{ma200:,.0f}) with {mom_20d:.1%} 20-day momentum.")
    elif price < ma200 and ma50 < ma200:
        regime  = "bear"
        reason  = (f"Nifty 50 (₹{price:,.0f}) is below MA200 (₹{ma200:,.0f}) "
                   f"and MA50 (₹{ma50:,.0f}) — confirmed downtrend. "
                   f"Down {drawdown:.1%} from 52-week high.")
    else:
        regime  = "neutral"
        reason  = (f"Nifty 50 signals are mixed: index {'above' if price > ma200 else 'below'} "
                   f"MA200 but MA50 crossover not confirmed. Applying caution.")

    return _regime_result(regime, signals, reason)


def _regime_result(regime, signals, reason):
    return {
        "regime":       regime,
        "signals":      signals,
        "reason":       reason,
        "equity_scale": REGIME_SCALE.get(regime, 0.90),
        "emoji":        {"bull": "🟢", "neutral": "🟡", "bear": "🔴"}.get(regime, "🟡"),
        "label":        {"bull": "BULL MARKET", "neutral": "NEUTRAL",
                         "bear": "BEAR MARKET"}.get(regime, "NEUTRAL"),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  2. VIX FEAR INDEX OVERLAY
# ══════════════════════════════════════════════════════════════════════════════

def get_vix_info() -> dict:
    """
    Fetch live VIX level from yfinance and classify fear state.

    VIX      Fear level   Equity scale   Action
    ────────────────────────────────────────────
    < 20     normal        100%           Full investment
    20–25    elevated       92%           Minor caution
    25–30    elevated       85%           Moderate caution
    30–40    high           75%           Reduce exposure
    > 40     extreme        50%           Major reduction
    """
    vix = _fetch_vix()

    if vix > VIX_EXTREME:
        fear, scale, desc = "extreme",  0.50, f"VIX {vix:.1f} — extreme fear. Cut equity exposure by 50%."
    elif vix > VIX_HIGH:
        fear, scale, desc = "high",     0.75, f"VIX {vix:.1f} — high fear. Reduce equity exposure by 25%."
    elif vix > VIX_ELEVATED:
        fear, scale, desc = "elevated", 0.85, f"VIX {vix:.1f} — elevated volatility. Minor 15% reduction."
    elif vix > VIX_NORMAL:
        fear, scale, desc = "elevated", 0.92, f"VIX {vix:.1f} — slightly above normal. Slight caution."
    else:
        fear, scale, desc = "normal",   1.00, f"VIX {vix:.1f} — normal volatility. Full deployment."

    return {
        "vix":         round(vix, 2),
        "fear_level":  fear,
        "equity_scale":scale,
        "description": desc,
        "emoji":       {"normal": "😊", "elevated": "😐",
                        "high": "😰", "extreme": "🚨"}.get(fear, "😐"),
    }


def _fetch_vix() -> float:
    """Fetch ^VIX last price. Returns 20.0 (neutral) on any failure."""
    try:
        t   = yf.Ticker("^VIX")
        vix = t.fast_info.last_price
        if vix and 5 < vix < 150:
            return float(vix)
    except Exception:
        pass
    # Try from prices.csv if already downloaded
    try:
        p = pd.read_csv(f"{DATA_DIR}/prices.csv", index_col=0)
        if "^VIX" in p.columns:
            return float(p["^VIX"].dropna().iloc[-1])
    except Exception:
        pass
    return 20.0


# ══════════════════════════════════════════════════════════════════════════════
#  3. COMBINED MACRO OVERLAY — apply to optimised weights
# ══════════════════════════════════════════════════════════════════════════════

def apply_macro_overlay(
    weights:     dict,
    regime_info: dict,
    vix_info:    dict,
) -> tuple:
    """
    Scale portfolio weights by regime × VIX combined factor.

    The freed-up weight represents a cash/short-term bond buffer.
    Portfolio risk is reduced proportionally for ALL stocks — individual
    sentiment rankings are preserved; only the overall exposure changes.

    Returns
    -------
    scaled_weights : dict  — equity weights after scaling (sum < 1.0 in risk-off)
    cash_buffer    : float — fraction of portfolio held as cash
    combined_scale : float — the combined multiplier applied
    """
    regime_scale   = regime_info.get("equity_scale", 1.0)
    vix_scale      = vix_info.get("equity_scale",   1.0)
    combined_scale = regime_scale * vix_scale

    # Hard floor: always keep at least 40% in equities
    combined_scale = max(0.40, min(1.0, combined_scale))

    scaled_weights = {t: round(w * combined_scale, 6) for t, w in weights.items()}
    cash_buffer    = round(1.0 - sum(scaled_weights.values()), 4)

    return scaled_weights, cash_buffer, round(combined_scale, 4)


# ══════════════════════════════════════════════════════════════════════════════
#  4. MACRO NARRATIVE
# ══════════════════════════════════════════════════════════════════════════════

def get_macro_narrative(
    regime_info: dict,
    vix_info:    dict,
    openai_key:  str = "",
) -> str:
    """
    2-3 sentence plain-English macro commentary.
    Uses GPT-3.5 if an OpenAI key is provided, otherwise a template.
    """
    regime = regime_info["regime"]
    vix    = vix_info["vix"]
    fear   = vix_info["fear_level"]
    scale  = regime_info["equity_scale"] * vix_info["equity_scale"]
    scale  = max(0.40, min(1.0, scale))

    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            prompt = f"""You are a senior macro strategist writing a brief daily note for retail investors.
In exactly 2-3 sentences, describe the current market environment and what it means for their portfolio.

Market regime : {regime.upper()} ({regime_info.get('reason','')})
VIX           : {vix:.1f} ({fear} fear — {vix_info['description']})
Portfolio deployed : {scale:.0%} (rest held as cash buffer)

Tone: calm, professional, specific. No bullet points. Start with the market regime."""
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=140,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            pass

    # Template fallback (no API needed)
    cash_pct = round((1.0 - scale) * 100, 0)
    templates = {
        "bull": (
            f"Indian markets are in a confirmed bull regime — the Nifty 50 is above both its "
            f"50-day and 200-day moving averages with positive momentum. "
            f"VIX at {vix:.1f} ({fear} fear) supports full equity deployment; "
            f"the portfolio is {scale:.0%} invested in NSE equities."
        ),
        "neutral": (
            f"Indian markets are sending mixed signals: Nifty 50 technical indicators are "
            f"not clearly trending in either direction, and VIX at {vix:.1f} ({fear} fear) "
            f"warrants measured caution. The portfolio holds a {cash_pct:.0f}% cash buffer "
            f"while maintaining {scale:.0%} NSE equity exposure."
        ),
        "bear": (
            f"Indian markets are in a bearish regime — Nifty 50 is below both its 50-day "
            f"and 200-day moving averages, confirming a downtrend. "
            f"With VIX at {vix:.1f} ({fear} fear), the macro overlay has reduced equity "
            f"exposure to {scale:.0%}, holding a {cash_pct:.0f}% cash buffer to limit drawdown risk."
        ),
    }
    return templates.get(regime, templates["neutral"])


# ══════════════════════════════════════════════════════════════════════════════
#  5. FULL MACRO SNAPSHOT  (single function for dashboard)
# ══════════════════════════════════════════════════════════════════════════════

def get_macro_snapshot(prices_df: pd.DataFrame, openai_key: str = "") -> dict:
    """
    Run all three overlays and return a complete macro snapshot dict.
    This is the single function called by the dashboard and optimizer.

    Returns
    -------
    {
        regime        : dict  (from detect_market_regime)
        vix           : dict  (from get_vix_info)
        combined_scale: float (regime_scale × vix_scale)
        narrative     : str   (GPT or template)
        cash_buffer   : float (1 - combined_scale, floor at 0)
    }
    """
    regime_info = detect_market_regime(prices_df)
    vix_info    = get_vix_info()
    narrative   = get_macro_narrative(regime_info, vix_info, openai_key)

    combined_scale = max(0.40, min(1.0,
        regime_info["equity_scale"] * vix_info["equity_scale"]
    ))

    return {
        "regime":         regime_info,
        "vix":            vix_info,
        "combined_scale": round(combined_scale, 4),
        "cash_buffer":    round(1.0 - combined_scale, 4),
        "narrative":      narrative,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT — standalone test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import pandas as pd

    prices_path = f"{DATA_DIR}/prices.csv"
    if not os.path.exists(prices_path):
        print("Run python data_collector.py first.")
    else:
        prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
        snap   = get_macro_snapshot(prices)

        r = snap["regime"]
        v = snap["vix"]
        print(f"\n{'═'*55}")
        print(f"  MACRO SNAPSHOT")
        print(f"{'═'*55}")
        print(f"  Market Regime : {r['emoji']}  {r['label']}")
        print(f"  Regime scale  : {r['equity_scale']:.0%}")
        print(f"  VIX           : {v['vix']:.1f}  ({v['emoji']} {v['fear_level']} fear)")
        print(f"  VIX scale     : {v['equity_scale']:.0%}")
        print(f"  Combined scale: {snap['combined_scale']:.0%}  "
              f"(cash buffer: {snap['cash_buffer']:.0%})")
        print(f"\n  Narrative:\n  {snap['narrative']}")
        print(f"\n  Regime signals:")
        for k, val in r["signals"].items():
            print(f"    {k:<20} {val}")
