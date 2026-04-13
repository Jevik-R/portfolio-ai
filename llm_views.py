# llm_views.py  —  LLM View Generator (Groq + LLaMA)
# ─────────────────────────────────────────────────────────────────────────────
# Replaces sentiment_engine.py with direct LLM-based return predictions
# exactly as described in the CIKM 2025 paper:
#
#   For each stock at each rebalancing date:
#     1. Send last 2 weeks of price data + sector + company info to LLaMA
#     2. Query N=10 times → get N return predictions
#     3. mean(predictions)     → View vector q  (expected return)
#     4. variance(predictions) → Confidence Ω   (uncertainty)
#
# These replace the FinBERT sentiment scores in feature_builder.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import time
import warnings
import numpy as np
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
MODEL          = "llama-3.1-8b-instant"   # same model family as paper
N_QUERIES      = 10                        # queries per stock (paper uses 100,
                                           # we use 10 to stay within free tier)
DATA_DIR       = "data"
OUTPUT_FILE    = f"{DATA_DIR}/llm_views.csv"

STOCKS = [
    "TCS.NS",       "INFY.NS",       "WIPRO.NS",      "HCLTECH.NS",
    "HDFCBANK.NS",  "ICICIBANK.NS",  "SBIN.NS",       "KOTAKBANK.NS",
    "SUNPHARMA.NS", "DRREDDY.NS",
    "HINDUNILVR.NS","ITC.NS",
    "RELIANCE.NS",  "ONGC.NS",
    "LT.NS",        "BHARTIARTL.NS",
]

STOCK_META = {
    "TCS.NS":        {"name": "Tata Consultancy Services", "sector": "Technology",     "sub": "IT Services"},
    "INFY.NS":       {"name": "Infosys Ltd",               "sector": "Technology",     "sub": "IT Services"},
    "WIPRO.NS":      {"name": "Wipro Ltd",                 "sector": "Technology",     "sub": "IT Services"},
    "HCLTECH.NS":    {"name": "HCL Technologies",          "sector": "Technology",     "sub": "IT Services"},
    "HDFCBANK.NS":   {"name": "HDFC Bank Ltd",             "sector": "Finance",        "sub": "Private Sector Bank"},
    "ICICIBANK.NS":  {"name": "ICICI Bank Ltd",            "sector": "Finance",        "sub": "Private Sector Bank"},
    "SBIN.NS":       {"name": "State Bank of India",       "sector": "Finance",        "sub": "Public Sector Bank"},
    "KOTAKBANK.NS":  {"name": "Kotak Mahindra Bank",       "sector": "Finance",        "sub": "Private Sector Bank"},
    "SUNPHARMA.NS":  {"name": "Sun Pharmaceutical",        "sector": "Healthcare",     "sub": "Pharmaceuticals"},
    "DRREDDY.NS":    {"name": "Dr. Reddy's Laboratories",  "sector": "Healthcare",     "sub": "Pharmaceuticals"},
    "HINDUNILVR.NS": {"name": "Hindustan Unilever",        "sector": "Consumer",       "sub": "FMCG"},
    "ITC.NS":        {"name": "ITC Ltd",                   "sector": "Consumer",       "sub": "FMCG & Cigarettes"},
    "RELIANCE.NS":   {"name": "Reliance Industries",       "sector": "Energy",         "sub": "Oil & Gas + Retail"},
    "ONGC.NS":       {"name": "Oil & Natural Gas Corp",    "sector": "Energy",         "sub": "Upstream Oil & Gas"},
    "LT.NS":         {"name": "Larsen & Toubro",           "sector": "Infrastructure", "sub": "Engineering & Construction"},
    "BHARTIARTL.NS": {"name": "Bharti Airtel",             "sector": "Telecom",        "sub": "Wireless Telecom"},
}

# Nifty 50 sector index proxies (used as sector returns in prompt)
SECTOR_INDEX = {
    "Technology":     "^NSEI",   # fallback to Nifty if sector ETF unavailable
    "Finance":        "^NSEI",
    "Healthcare":     "^NSEI",
    "Consumer":       "^NSEI",
    "Energy":         "^NSEI",
    "Infrastructure": "^NSEI",
    "Telecom":        "^NSEI",
}
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — PROMPT BUILDER  (exactly as described in paper Appendix A)
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are providing analysis on {date}. Predict the average daily return for the next two weeks based on the information provided about a stock's past performance.

You will receive the following inputs:
- Daily Returns: The stock's daily returns, a time-series from the past two weeks (in %).
- Company Sector: The company's sector classification.
- Sector Returns: The sector's daily returns, a time-series from the past two weeks (in %).
- Market Returns: The Nifty 50's daily returns, a time-series from the past two weeks (in %).
- Company Information: Ticker, Company Name, Sector, Sub-Industry.

Steps:
1. Analyze the Time-Series Data: Review the historical daily returns to identify patterns, trends, or anomalies.
2. Consider Sector Performance: Analyze how the market and sector performance might influence the stock's future returns.
3. Incorporate Company Information: Use the sector and sub-industry to contextualize the predicted performance.
4. Predict Future Returns: Estimate the average daily return for the next two weeks.

Output Format:
Return ONLY a single float value representing the predicted average daily return (in %) for the next two weeks. No explanation, no text, just the number.

Example valid outputs: 0.15 or -0.08 or 0.32

Notes:
- Returns are in percentage terms (e.g. -0.36 means -0.36% daily return)
- Make calculations based on statistical trends from daily returns data
- Pay attention to momentum within both stock and market returns
- Consider sector relevance to refine predictions"""


def build_user_prompt(
    ticker:         str,
    stock_returns:  list,
    sector_returns: list,
    market_returns: list,
    date:           str,
) -> str:
    """
    Build the user prompt for a specific stock at a specific date.
    Returns are already scaled to % (×100) as per paper Appendix A.
    """
    meta = STOCK_META.get(ticker, {})

    def fmt_list(lst):
        return "[" + ", ".join(f"{x:.2f}" for x in lst) + "]"

    return f"""Daily Returns: {fmt_list(stock_returns)}
Company Sector: {meta.get('sector', 'Unknown')}
Sector Returns: {fmt_list(sector_returns)}
Market Returns: {fmt_list(market_returns)}
Company Information:
  Ticker: {ticker.replace('.NS', '')}
  Company Name: {meta.get('name', ticker)}
  Sector: {meta.get('sector', 'Unknown')}
  Sub-Industry: {meta.get('sub', 'Unknown')}"""


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — GROQ API CALLER
# ══════════════════════════════════════════════════════════════════════════════

def _init_groq_client() -> Groq:
    """Initialise Groq client. Raises if API key missing."""
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY not found!\n"
            "Add it to your .env file: GROQ_API_KEY=your_key_here\n"
            "Get a free key at: https://console.groq.com"
        )
    return Groq(api_key=GROQ_API_KEY)


def query_llm_once(
    client:        Groq,
    system_prompt: str,
    user_prompt:   str,
    temperature:   float = 0.7,
) -> float | None:
    """
    Send one query to LLaMA via Groq.
    Returns the predicted return as a float, or None if parsing fails.
    Temperature > 0 ensures variation across repeated queries.
    """
    try:
        response = client.chat.completions.create(
            model       = MODEL,
            messages    = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature = temperature,
            max_tokens  = 20,      # we only need a single number
        )
        raw = response.choices[0].message.content.strip()

        # Parse: strip any accidental text, extract first float found
        import re
        matches = re.findall(r"[-+]?\d*\.?\d+", raw)
        if matches:
            val = float(matches[0])
            # Sanity check: daily return should be in [-10%, +10%] range
            if -10.0 <= val <= 10.0:
                return val
        return None

    except Exception as e:
        print(f"      ⚠️  Groq API error: {e}")
        time.sleep(2)
        return None


def query_llm_n_times(
    client:        Groq,
    system_prompt: str,
    user_prompt:   str,
    n:             int = N_QUERIES,
) -> list[float]:
    """
    Query LLaMA N times for the same stock.
    Returns list of valid float predictions.
    Implements retry logic for failed queries.
    """
    predictions = []
    attempts    = 0
    max_attempts = n * 2   # allow up to 2× retries

    while len(predictions) < n and attempts < max_attempts:
        val = query_llm_once(client, system_prompt, user_prompt)
        if val is not None:
            predictions.append(val)
        attempts += 1
        # Small delay to respect Groq free tier rate limits
        time.sleep(0.3)

    return predictions


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — VIEW & CONFIDENCE CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

def predictions_to_view(predictions: list[float]) -> dict:
    """
    Convert N LLM predictions → Black-Litterman view components.

    Exactly as described in paper Section 3.3.2:
      q_i   = mean(predictions)      ← expected return view
      Ω_ii  = variance(predictions)  ← uncertainty (confidence matrix diagonal)

    Returns dict with all stats needed for BL model and dashboard display.
    """
    if not predictions:
        return {
            "q":            0.0,
            "omega":        1.0,      # high uncertainty = low confidence
            "mean_pct":     0.0,
            "std_pct":      1.0,
            "n_valid":      0,
            "min_pct":      0.0,
            "max_pct":      0.0,
            "predictions":  [],
            "label":        "neutral",
            "final_score":  0.0,
            "confidence":   0.0,
        }

    arr      = np.array(predictions)
    mean_pct = float(np.mean(arr))
    std_pct  = float(np.std(arr))

    # Convert % → decimal for BL model
    # e.g. mean_pct=0.15% → q=0.0015 daily return
    q     = mean_pct / 100.0
    omega = (std_pct / 100.0) ** 2   # variance in decimal

    # Map to sentiment-like label for dashboard compatibility
    if   mean_pct >=  0.20: label = "bullish"
    elif mean_pct >=  0.05: label = "slightly_bullish"
    elif mean_pct <= -0.20: label = "bearish"
    elif mean_pct <= -0.05: label = "slightly_bearish"
    else:                   label = "neutral"

    # Normalise to [-1, +1] score for dashboard display
    final_score = float(np.clip(mean_pct / 0.5, -1.0, 1.0))

    # Confidence: lower variance = higher confidence
    # Maps std of 0% → conf=1.0, std of 1% → conf=0.0
    confidence = float(np.clip(1.0 - std_pct, 0.05, 0.95))

    return {
        "q":           q,
        "omega":       max(omega, 1e-8),   # floor to avoid division by zero
        "mean_pct":    round(mean_pct, 4),
        "std_pct":     round(std_pct,  4),
        "n_valid":     len(predictions),
        "min_pct":     round(float(arr.min()), 4),
        "max_pct":     round(float(arr.max()), 4),
        "predictions": predictions,
        "label":       label,
        "final_score": round(final_score, 4),
        "confidence":  round(confidence,  4),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_llm_view_pipeline(
    lookback_days: int = 10,
) -> pd.DataFrame:
    """
    Full pipeline — runs at each rebalancing date (or on-demand for dashboard).

    For each stock:
      1. Load last `lookback_days` of returns from returns.csv
      2. Build prompt (stock returns + sector + market returns)
      3. Query LLaMA N times
      4. Compute mean (view q) and variance (confidence Ω)
      5. Save results to data/llm_views.csv

    Returns DataFrame indexed by ticker — compatible with
    feature_builder.py's load_sentiment_scores() interface.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # ── Load price data ───────────────────────────────────────────────────────
    returns_path = f"{DATA_DIR}/returns.csv"
    if not os.path.exists(returns_path):
        raise FileNotFoundError(
            "returns.csv not found — run: python data_collector.py first"
        )

    returns_df = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    latest_date = returns_df.index[-1].strftime("%Y-%m-%d")

    # Last lookback_days of returns
    recent = returns_df.tail(lookback_days)

    # Market returns (Nifty 50)
    market_col    = "NIFTY50" if "NIFTY50" in recent.columns else None
    market_rets   = (recent[market_col] * 100).round(2).tolist() if market_col else [0.0] * lookback_days

    # ── Init Groq ─────────────────────────────────────────────────────────────
    client = _init_groq_client()

    print(f"\n{'═'*60}")
    print(f"   LLM VIEW GENERATOR  |  {latest_date}")
    print(f"   Model    : {MODEL}")
    print(f"   Queries  : {N_QUERIES} per stock")
    print(f"   Lookback : {lookback_days} trading days")
    print(f"   Stocks   : {len(STOCKS)}")
    print(f"{'═'*60}")

    rows = []

    for ticker in STOCKS:
        meta = STOCK_META.get(ticker, {})
        print(f"\n{'─'*50}")
        print(f"  {ticker}  —  {meta.get('name', ticker)}")

        # ── Get stock returns ─────────────────────────────────────────────────
        if ticker not in recent.columns:
            print(f"  ⚠️  {ticker} not in returns.csv — skipping")
            continue

        stock_rets = (recent[ticker] * 100).round(2).tolist()

        # ── Get sector returns (use Nifty as proxy) ───────────────────────────
        # In a production system you'd use sector ETF returns here
        # For simplicity we use Nifty 50 as market proxy for all sectors
        sector_rets = market_rets   # same as market for NSE (no sector ETFs loaded)

        # ── Build prompts ─────────────────────────────────────────────────────
        system = SYSTEM_PROMPT.format(date=latest_date)
        user   = build_user_prompt(
            ticker        = ticker,
            stock_returns = stock_rets,
            sector_returns= sector_rets,
            market_returns= market_rets,
            date          = latest_date,
        )

        print(f"  Stock returns (last {lookback_days}d): {stock_rets}")
        print(f"  Querying LLaMA {N_QUERIES} times...")

        # ── Query LLM N times ─────────────────────────────────────────────────
        predictions = query_llm_n_times(client, system, user, n=N_QUERIES)

        print(f"  Valid predictions: {len(predictions)}/{N_QUERIES}")
        print(f"  Raw predictions: {[round(p,3) for p in predictions]}")

        # ── Compute view and confidence ───────────────────────────────────────
        view = predictions_to_view(predictions)

        print(f"  Mean (q):    {view['mean_pct']:+.4f}%  per day")
        print(f"  Std  (Ω):    {view['std_pct']:.4f}%")
        print(f"  Label:       {view['label']}")
        print(f"  Score:       {view['final_score']:+.4f}")

        rows.append({
            "ticker":        ticker,
            "date":          latest_date,
            "company":       meta.get("name", ticker),
            "sector":        meta.get("sector", "Unknown"),
            # BL inputs
            "q":             view["q"],
            "omega":         view["omega"],
            # Dashboard-compatible fields (mirrors sentiment_scores.csv)
            "final_score":   view["final_score"],
            "label":         view["label"],
            "confidence":    view["confidence"],
            "num_headlines": view["n_valid"],    # reused for "n queries"
            "pct_positive":  1.0 if view["mean_pct"] > 0 else 0.0,
            "pct_negative":  1.0 if view["mean_pct"] < 0 else 0.0,
            "pct_neutral":   1.0 if view["mean_pct"] == 0 else 0.0,
            "sentiment_std": view["std_pct"] / 100.0,
            # Extra LLM-specific fields
            "mean_pct":      view["mean_pct"],
            "std_pct":       view["std_pct"],
            "min_pct":       view["min_pct"],
            "max_pct":       view["max_pct"],
            "n_valid":       view["n_valid"],
        })

        # Groq free tier: ~30 req/min → small pause between stocks
        time.sleep(1.0)

    # ── Build DataFrame ───────────────────────────────────────────────────────
    df = pd.DataFrame(rows).set_index("ticker")

    # ── Save ─────────────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_FILE)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n\n{'═'*60}")
    print(f"   LLM VIEW SUMMARY  —  {latest_date}")
    print(f"{'═'*60}")
    print(f"  {'TICKER':<14} {'q(view)':>9}  {'Ω(uncert)':>10}  {'LABEL':<20}  PREDICTIONS")
    print(f"  {'─'*70}")
    for ticker, row in df.iterrows():
        preds_preview = str([round(p, 2) for p in rows[
            next(i for i, r in enumerate(rows) if r["ticker"] == ticker)
        ]["n_valid"] and [] or []])
        print(
            f"  {ticker:<14} "
            f"{row['mean_pct']:>+9.4f}%  "
            f"{row['std_pct']:>10.4f}%  "
            f"{row['label']:<20}"
        )

    print(f"\n  💾 Saved: {OUTPUT_FILE}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — INTERFACE FOR feature_builder.py
#  These functions mirror sentiment_engine.py's interface exactly
#  so feature_builder.py works with zero changes to its logic
# ══════════════════════════════════════════════════════════════════════════════

def load_sentiment_scores() -> pd.DataFrame:
    """
    Load saved LLM views. Called by feature_builder.py.
    Mirrors sentiment_engine.load_sentiment_scores() interface exactly.
    """
    # Try LLM views first, fall back to sentiment scores
    for path in [OUTPUT_FILE, f"{DATA_DIR}/sentiment_scores.csv"]:
        if os.path.exists(path):
            df = pd.read_csv(path, index_col="ticker")
            print(f"  📂 Loaded views from: {path}")
            return df

    raise FileNotFoundError(
        f"No view data found.\n"
        f"Run: python llm_views.py   to generate LLM views\n"
        f"  or: python sentiment_engine.py   to use FinBERT sentiment"
    )


def get_bl_views(
    sentiment_df:      pd.DataFrame,
    mu_prior:          pd.Series,
    S:                 pd.DataFrame,
    view_sensitivity:  float = 0.5,
    earnings_surprise: "pd.Series | None" = None,
    earnings_beta:     float = 0.15,
) -> tuple:
    """
    Convert LLM views → Black-Litterman (Q, confidences).

    If llm_views.csv is loaded (has 'q' column), uses direct LLM predictions.
    If sentiment_scores.csv is loaded (legacy), falls back to sentiment-based views.

    Returns (viewdict, confidences) — same interface as sentiment_engine.get_bl_views()
    """
    viewdict    = {}
    confidences = []

    for ticker in mu_prior.index:
        if ticker not in sentiment_df.index:
            # No view — use market prior
            viewdict[ticker] = float(mu_prior[ticker])
            confidences.append(0.5)
            continue

        row = sentiment_df.loc[ticker]

        # ── LLM direct view path (new) ────────────────────────────────────────
        if "q" in row.index and pd.notna(row["q"]) and row["q"] != 0.0:
            # q is already in daily decimal return (e.g. 0.0015)
            # Annualise for BL model (BL works with annual returns)
            q_daily  = float(row["q"])
            q_annual = q_daily * 252        # annualise daily → annual

            # Add earnings surprise if available
            if earnings_surprise is not None and ticker in earnings_surprise.index:
                sigma_i  = float(np.sqrt(S.loc[ticker, ticker])) if ticker in S.columns else 0.20
                surprise = float(earnings_surprise[ticker])
                q_annual += earnings_beta * surprise * sigma_i

            viewdict[ticker] = round(q_annual, 6)

            # Confidence from LLM variance
            omega      = float(row.get("omega", 0.01))
            confidence = float(np.clip(1.0 - omega * 1000, 0.05, 0.95))
            confidences.append(confidence)

        # ── Sentiment fallback path (legacy FinBERT) ──────────────────────────
        else:
            from sentiment_engine import get_bl_views as _legacy_bl_views
            # Just get this one ticker's view from legacy function
            score    = float(row.get("final_score", 0.0))
            conf     = float(row.get("confidence",  0.5))
            sent_std = float(row.get("sentiment_std", 0.1))
            sigma_i  = float(np.sqrt(S.loc[ticker, ticker])) if ticker in S.columns else 0.20
            mu_i     = float(mu_prior[ticker])
            alpha    = float(np.clip(0.10 + (conf - 0.5) * 0.15 - sent_std * 0.20, 0.02, 0.20))
            Q_i      = mu_i + alpha * score * sigma_i * view_sensitivity
            viewdict[ticker] = round(Q_i, 6)
            confidences.append(float(np.clip(conf, 0.05, 0.95)))

    return viewdict, confidences


def get_sentiment_constraints(sentiment_df: pd.DataFrame) -> dict:
    """
    Convert LLM view labels → portfolio weight constraints.
    Mirrors sentiment_engine.get_sentiment_constraints() exactly.
    """
    lower, upper = {}, {}

    for ticker, row in sentiment_df.iterrows():
        label      = row.get("label", "neutral")
        confidence = float(row.get("confidence", 0.5))

        if label == "bearish":
            cap = 0.08 if confidence > 0.85 else 0.12 if confidence > 0.70 else 0.20
            lower[ticker] = 0.0
            upper[ticker] = cap

        elif label == "bullish":
            floor = 0.08 if confidence > 0.85 else 0.05 if confidence > 0.70 else 0.02
            lower[ticker] = floor
            upper[ticker] = 0.40

        else:
            lower[ticker] = 0.0
            upper[ticker] = 0.30

    return {
        "lower_bounds": pd.Series(lower),
        "upper_bounds": pd.Series(upper),
    }


def dynamic_alpha(confidence: float, sentiment_std: float) -> float:
    """Mirrors sentiment_engine.dynamic_alpha() — used by feature_builder."""
    base_alpha          = 0.10
    confidence_bonus    = (confidence - 0.5) * 0.15
    uncertainty_penalty = sentiment_std * 0.20
    return round(max(0.02, min(0.20, base_alpha + confidence_bonus - uncertainty_penalty)), 4)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    df = run_llm_view_pipeline(lookback_days=10)

    print(f"\n\n✅ LLM views ready!")
    print(f"   Bullish  stocks: {(df['label'] == 'bullish').sum()}")
    print(f"   Slightly bullish: {(df['label'] == 'slightly_bullish').sum()}")
    print(f"   Neutral  stocks: {(df['label'] == 'neutral').sum()}")
    print(f"   Slightly bearish: {(df['label'] == 'slightly_bearish').sum()}")
    print(f"   Bearish  stocks: {(df['label'] == 'bearish').sum()}")
    print(f"\n   Mean predicted daily return: {df['mean_pct'].mean():+.4f}%")
    print(f"   Range: {df['mean_pct'].min():+.4f}% to {df['mean_pct'].max():+.4f}%")
    print(f"\n   Next step: run python feature_builder.py")