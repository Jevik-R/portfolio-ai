# sentiment_engine.py
# ─────────────────────────────────────────────────────────────────────────────
#  LLM Sentiment Engine  —  Portfolio Optimization  (BTech Mini Project)
#  News source : Serper API  (https://serper.dev — 2500 free queries/month)
#  Scoring     : FinBERT (free, local)  OR  OpenAI GPT-3.5 (optional)
# ─────────────────────────────────────────────────────────────────────────────
#  Setup:
#    1. Get free API key at https://serper.dev (takes 30 seconds)
#    2. pip install requests transformers torch pandas numpy tqdm
#    3. Set SERPER_API_KEY below or as environment variable
#    4. python sentiment_engine.py
# ─────────────────────────────────────────────────────────────────────────────

import os, json, time, warnings
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG  —  only edit this section
# ══════════════════════════════════════════════════════════════════════════════

SERPER_API_KEY  = os.getenv("SERPER_API_KEY", "8e8ea27f93117a01b10421f2f704764284e4b422")

USE_OPENAI      = False          # False = FinBERT (free, recommended)
                                 # True  = GPT-3.5 (needs OPENAI_API_KEY)
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")

# Fetch 20 headlines per stock — larger sample reduces clickbait noise
MAX_HEADLINES   = 20

# Time window for news:  "qdr:d" = past day  |  "qdr:w" = past week  |  "qdr:m" = past month
NEWS_TIMEFRAME  = "qdr:w"

DATA_DIR        = "data"
OUTPUT_FILE     = f"{DATA_DIR}/sentiment_scores.csv"
DETAIL_FILE     = f"{DATA_DIR}/sentiment_detail.csv"

# ── Stocks (must match data_collector.py) ─────────────────────────────────────
STOCKS = [
    "TCS.NS",       "INFY.NS",       "WIPRO.NS",      "HCLTECH.NS",
    "HDFCBANK.NS",  "ICICIBANK.NS",  "SBIN.NS",       "KOTAKBANK.NS",
    "SUNPHARMA.NS", "DRREDDY.NS",
    "HINDUNILVR.NS","ITC.NS",
    "RELIANCE.NS",  "ONGC.NS",
    "LT.NS",        "BHARTIARTL.NS",
]

STOCK_META = {
    "TCS.NS":        {"name": "Tata Consultancy Services", "search": "TCS Tata Consultancy Services NSE earnings revenue IT"},
    "INFY.NS":       {"name": "Infosys Ltd",               "search": "Infosys INFY NSE earnings guidance IT services"},
    "WIPRO.NS":      {"name": "Wipro Ltd",                 "search": "Wipro NSE stock earnings IT outsourcing"},
    "HCLTECH.NS":    {"name": "HCL Technologies",          "search": "HCL Technologies NSE stock earnings IT services"},
    "HDFCBANK.NS":   {"name": "HDFC Bank Ltd",             "search": "HDFC Bank NSE earnings NPA credit growth"},
    "ICICIBANK.NS":  {"name": "ICICI Bank Ltd",            "search": "ICICI Bank NSE earnings NPA loan growth"},
    "SBIN.NS":       {"name": "State Bank of India",       "search": "SBI State Bank India NSE earnings NPA profit"},
    "KOTAKBANK.NS":  {"name": "Kotak Mahindra Bank",       "search": "Kotak Mahindra Bank NSE earnings profit"},
    "SUNPHARMA.NS":  {"name": "Sun Pharmaceutical",        "search": "Sun Pharma NSE earnings USFDA drug approval"},
    "DRREDDY.NS":    {"name": "Dr. Reddy's Laboratories",  "search": "Dr Reddy Laboratories NSE earnings USFDA pipeline"},
    "HINDUNILVR.NS": {"name": "Hindustan Unilever",        "search": "Hindustan Unilever HUL NSE earnings FMCG volume"},
    "ITC.NS":        {"name": "ITC Ltd",                   "search": "ITC Ltd NSE earnings cigarette FMCG hotel"},
    "RELIANCE.NS":   {"name": "Reliance Industries",       "search": "Reliance Industries RIL NSE earnings Jio retail"},
    "ONGC.NS":       {"name": "Oil & Natural Gas Corp",    "search": "ONGC NSE earnings oil gas production crude"},
    "LT.NS":         {"name": "Larsen & Toubro",           "search": "Larsen Toubro L&T NSE earnings order book infra"},
    "BHARTIARTL.NS": {"name": "Bharti Airtel",             "search": "Bharti Airtel NSE earnings ARPU subscriber 5G"},
}

# ── Source credibility weights ─────────────────────────────────────────────────
# Premium financial sources are trusted more; opinion sites weighted down
SOURCE_WEIGHTS = {
    # Indian financial media (high credibility)
    "Economic Times":       1.4,
    "Business Standard":    1.4,
    "Mint":                 1.4,
    "Livemint":             1.4,
    "Financial Express":    1.3,
    "Business Today":       1.2,
    "Moneycontrol":         1.2,
    "CNBC-TV18":            1.2,
    "The Hindu Business":   1.1,
    "Hindustan Times":      1.0,
    # Global financial media
    "Reuters":              1.5,
    "Bloomberg":            1.5,
    "Wall Street Journal":  1.3,
    "Financial Times":      1.3,
    # General / lower credibility
    "Yahoo Finance":        1.0,
    "Seeking Alpha":        0.7,
    "Investopedia":         0.8,
}

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — SERPER NEWS FETCHER
# ══════════════════════════════════════════════════════════════════════════════

SERPER_URL = "https://google.serper.dev/news"

def fetch_serper_news(ticker: str) -> list:
    """
    Call Serper's /news endpoint for a stock ticker.
    Returns list of dicts: {title, snippet, source, date, link, ticker}

    Serper /news endpoint returns real Google News results.
    Each article has:
      - title   : headline text  (what we score)
      - snippet : 1-2 sentence summary  (extra context)
      - source  : publisher name (Reuters, Bloomberg, etc.)
      - date    : "2 hours ago", "1 day ago", etc.
      - link    : original article URL
    """
    if SERPER_API_KEY == "YOUR_SERPER_API_KEY_HERE":
        raise ValueError(
            "SERPER_API_KEY not set!\n"
            "Get your free key at https://serper.dev\n"
            "Then set: SERPER_API_KEY = 'your_key_here' in this file"
        )

    query = STOCK_META[ticker]["search"]

    payload = {
        "q":   query,
        "num": MAX_HEADLINES,
        "tbs": NEWS_TIMEFRAME,    # time filter
        "gl":  "us",              # geo: United States
        "hl":  "en",              # language: English
    }
    headers = {
        "X-API-KEY":    SERPER_API_KEY,
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            SERPER_URL,
            headers = headers,
            json    = payload,
            timeout = 15,
        )
        response.raise_for_status()
        data     = response.json()
        articles = data.get("news", [])

        results = []
        for a in articles:
            title   = a.get("title",   "").strip()
            snippet = a.get("snippet", "").strip()

            # Skip empty or very short titles
            if not title or len(title) < 10:
                continue

            # Combine title + snippet for richer sentiment signal
            # FinBERT scores the full_text but we display only title
            full_text = title
            if snippet:
                full_text = f"{title}. {snippet}"

            results.append({
                "ticker":    ticker,
                "title":     title,
                "snippet":   snippet,
                "full_text": full_text,
                "source":    a.get("source", "Unknown"),
                "date":      a.get("date",   "Unknown"),
                "link":      a.get("link",   ""),
            })

        return results

    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response else "?"
        if status == 401:
            print(f"  ✗ [{ticker}] Invalid Serper API key — check SERPER_API_KEY")
        elif status == 429:
            print(f"  ✗ [{ticker}] Serper rate limit hit — wait a minute and retry")
        else:
            print(f"  ✗ [{ticker}] HTTP {status}: {e}")
        return []

    except requests.exceptions.ConnectionError:
        print(f"  ✗ [{ticker}] No internet connection — Serper unreachable")
        return []

    except Exception as e:
        print(f"  ✗ [{ticker}] Unexpected error: {e}")
        return []


def fetch_serper_news_safe(ticker: str, retries: int = 2) -> list:
    """Wrapper with retry logic and delay between calls."""
    for attempt in range(retries + 1):
        results = fetch_serper_news(ticker)
        if results:
            return results
        if attempt < retries:
            print(f"    Retry {attempt + 1}/{retries} for {ticker}...")
            time.sleep(2)
    return []


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — SENTIMENT SCORING
# ══════════════════════════════════════════════════════════════════════════════

LABEL_TO_SCORE = {"positive": +1.0, "negative": -1.0, "neutral": 0.0}

# ── 2A. FinBERT  (recommended — free, finance-specific, no API key) ───────────

_finbert_pipe = None

def load_finbert():
    global _finbert_pipe
    if _finbert_pipe is None:
        from transformers import pipeline
        print("\n  Loading FinBERT model (~440 MB, cached after first run)...")
        _finbert_pipe = pipeline(
            task       = "text-classification",
            model      = "ProsusAI/finbert",
            tokenizer  = "ProsusAI/finbert",
            top_k      = None,      # returns all 3 class scores (works on all versions)
            truncation = True,
            max_length = 512,
        )
        print("  FinBERT ready.\n")
    return _finbert_pipe


def _parse_finbert_result(res):
    """
    Handle both output formats across transformers versions:
      - New (top_k=None) : [{"label":"positive","score":0.87}, ...]   <- list of dicts
      - Old single-label  : {"label":"positive","score":0.87}         <- single dict
    Always returns a list of dicts with 'label' and 'score' keys.
    """
    # Single dict (old format without return_all_scores)
    if isinstance(res, dict):
        return [res]
    # List of dicts — normal top_k=None output
    if isinstance(res, list) and res and isinstance(res[0], dict):
        return res
    # Nested list [[{...}, {...}]] — some versions wrap in extra list
    if isinstance(res, list) and res and isinstance(res[0], list):
        return res[0]
    # Fallback
    return [{"label": "neutral", "score": 1.0}]


def score_with_finbert(articles: list) -> list:
    """
    Score each article using FinBERT.
    Input : list of article dicts (must have 'full_text' key)
    Output: same list with added keys — label, confidence, raw_score
    """
    if not articles:
        return []

    pipe  = load_finbert()
    texts = [a["full_text"] for a in articles]

    # Batch inference — much faster than one-by-one
    batch_results = pipe(texts, batch_size=8)

    scored = []
    for article, raw_res in zip(articles, batch_results):

        # Normalise output format (handles all transformers versions)
        res = _parse_finbert_result(raw_res)

        best       = max(res, key=lambda x: x["score"])
        label      = best["label"].lower()      # "positive" / "negative" / "neutral"
        confidence = round(best["score"], 4)
        direction  = LABEL_TO_SCORE.get(label, 0.0)
        raw_score  = round(direction * confidence, 4)

        # All 3 class probabilities (useful for reporting)
        all_probs  = {r["label"].lower(): round(r["score"], 4) for r in res}

        scored.append({
            **article,
            "label":      label,
            "confidence": confidence,
            "raw_score":  raw_score,
            "prob_pos":   all_probs.get("positive", 0.0),
            "prob_neg":   all_probs.get("negative", 0.0),
            "prob_neu":   all_probs.get("neutral",  0.0),
        })

    return scored


# ── 2B. OpenAI GPT-3.5  (optional) ───────────────────────────────────────────

def score_with_openai(ticker: str, articles: list) -> list:
    """
    Score articles using GPT-3.5 via OpenAI API.
    Sends all headlines in one batch prompt to save API calls.
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
    except ImportError:
        raise ImportError("Run: pip install openai")

    company  = STOCK_META[ticker]["name"]
    numbered = "\n".join(
        f"{i+1}. {a['title']}" + (f" — {a['snippet'][:80]}" if a.get("snippet") else "")
        for i, a in enumerate(articles)
    )

    prompt = f"""You are a quantitative financial analyst.

Analyze these {len(articles)} recent news items about {company} ({ticker}).
For each item, output its sentiment from an equity investor's perspective.

Rules:
- "positive" = likely to push stock price UP (earnings beat, new product, upgrade, buyback)
- "negative" = likely to push stock price DOWN (miss, lawsuit, downgrade, macro headwind)
- "neutral"  = no clear directional impact

Return ONLY a valid JSON array, no markdown:
[{{"num": 1, "label": "positive", "confidence": 0.85}}, ...]

News items:
{numbered}"""

    try:
        resp   = client.chat.completions.create(
            model       = "gpt-3.5-turbo",
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0,
            max_tokens  = 800,
        )
        parsed = json.loads(resp.choices[0].message.content.strip())

        scored = []
        for article, r in zip(articles, parsed):
            label      = r.get("label", "neutral").lower()
            confidence = float(r.get("confidence", 0.5))
            scored.append({
                **article,
                "label":      label,
                "confidence": round(confidence, 4),
                "raw_score":  round(LABEL_TO_SCORE.get(label, 0.0) * confidence, 4),
                "prob_pos":   confidence if label == "positive" else 0.0,
                "prob_neg":   confidence if label == "negative" else 0.0,
                "prob_neu":   confidence if label == "neutral"  else 0.0,
            })
        return scored

    except json.JSONDecodeError:
        print(f"  GPT returned non-JSON — falling back to neutral for {ticker}")
        return [{**a, "label": "neutral", "confidence": 0.5,
                 "raw_score": 0.0, "prob_pos": 0.0,
                 "prob_neg": 0.0, "prob_neu": 1.0} for a in articles]

    except Exception as e:
        print(f"  OpenAI error for {ticker}: {e}")
        return [{**a, "label": "neutral", "confidence": 0.5,
                 "raw_score": 0.0, "prob_pos": 0.0,
                 "prob_neg": 0.0, "prob_neu": 1.0} for a in articles]


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — AGGREGATION  (headline → stock-level signal)
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_to_stock_score(scored_articles: list) -> dict:
    """
    Aggregate N headline scores into one stock-level sentiment signal.

    Method: confidence-weighted average
      - A headline scored "positive" at 90% confidence contributes more
        than one scored at 55% confidence
      - Final score is clipped to [-1.0, +1.0]
      - Label thresholds: bullish (>0.30), slightly_bullish (>0.05),
        neutral, slightly_bearish (<-0.05), bearish (<-0.30)

    Additional outputs used by the optimizer:
      - sentiment_std  : disagreement across headlines (high = uncertain)
      - pct_positive/negative/neutral : distribution for reporting
    """
    if not scored_articles:
        return {
            "final_score":   0.0,
            "label":         "neutral",
            "confidence":    0.0,
            "num_headlines": 0,
            "pct_positive":  0.0,
            "pct_negative":  0.0,
            "pct_neutral":   1.0,
            "sentiment_std": 0.0,
        }

    labels = [a["label"] for a in scored_articles]

    # Source-credibility × confidence weighted average
    # Premium sources (Reuters, Bloomberg) count more; opinion sites count less
    effective_weights = []
    weighted_scores   = []
    for a in scored_articles:
        src_wt  = SOURCE_WEIGHTS.get(a.get("source", ""), SOURCE_WEIGHTS.get("default", 1.0))
        eff_wt  = a["confidence"] * src_wt
        effective_weights.append(eff_wt)
        weighted_scores.append(a["raw_score"] * eff_wt)

    total_wt     = sum(effective_weights) or 1.0
    weighted_avg = sum(weighted_scores) / total_wt
    final_score  = round(max(-1.0, min(1.0, weighted_avg)), 4)

    n = len(labels)
    pct_pos = round(labels.count("positive") / n, 3)
    pct_neg = round(labels.count("negative") / n, 3)
    pct_neu = round(labels.count("neutral")  / n, 3)

    # Map score → label
    if   final_score >=  0.30: sentiment_label = "bullish"
    elif final_score >=  0.05: sentiment_label = "slightly_bullish"
    elif final_score <= -0.30: sentiment_label = "bearish"
    elif final_score <= -0.05: sentiment_label = "slightly_bearish"
    else:                      sentiment_label = "neutral"

    raw_scores    = [a["raw_score"]  for a in scored_articles]
    confidences   = [a["confidence"] for a in scored_articles]
    sentiment_std = round(float(np.std(raw_scores)), 4) if len(raw_scores) > 1 else 0.0
    avg_conf      = round(sum(confidences) / n, 4)

    return {
        "final_score":   final_score,
        "label":         sentiment_label,
        "confidence":    avg_conf,
        "num_headlines": n,
        "pct_positive":  pct_pos,
        "pct_negative":  pct_neg,
        "pct_neutral":   pct_neu,
        "sentiment_std": sentiment_std,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_sentiment_pipeline(stocks: list = None) -> pd.DataFrame:
    """
    Full pipeline for all stocks:
      1. Fetch news from Serper API
      2. Score with FinBERT or GPT
      3. Aggregate to per-stock score
      4. Save sentiment_scores.csv + sentiment_detail.csv

    Returns: DataFrame indexed by ticker
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    stocks = stocks or STOCKS
    today  = datetime.today().strftime("%Y-%m-%d")
    model  = "OpenAI GPT-3.5" if USE_OPENAI else "FinBERT (local)"

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f"   LLM SENTIMENT ENGINE   |   {today}")
    print(f"   News    : Serper API  (Google News)  |  timeframe: {NEWS_TIMEFRAME}")
    print(f"   Scoring : {model}")
    print(f"   Stocks  : {len(stocks)}   |   Max headlines/stock: {MAX_HEADLINES}")
    print(f"{'═'*65}")

    summary_rows = []
    detail_rows  = []
    total_api_calls = 0

    for ticker in tqdm(stocks, desc="Processing", ncols=72):

        meta = STOCK_META.get(ticker, {})
        print(f"\n{'─'*50}")
        print(f"  {ticker}  —  {meta.get('name', ticker)}")
        print(f"  Query: \"{meta.get('search', ticker)}\"")

        # ── Step 1: Fetch news ─────────────────────────────────────────────
        articles = fetch_serper_news_safe(ticker)
        total_api_calls += 1
        print(f"  Fetched : {len(articles)} articles", end="")
        if articles:
            sources = list(set(a["source"] for a in articles))[:4]
            print(f"  |  sources: {', '.join(sources)}")
        else:
            print("  ← no results returned")

        # Print headlines so you can verify live data
        if articles:
            print(f"  Headlines fetched:")
            for i, a in enumerate(articles, 1):
                date_str = f"[{a['date']}]" if a.get("date") else ""
                print(f"    {i:2d}. {a['title'][:72]}  {date_str}")

        # ── Step 2: Score ──────────────────────────────────────────────────
        if articles:
            if USE_OPENAI:
                scored = score_with_openai(ticker, articles)
            else:
                scored = score_with_finbert(articles)
        else:
            scored = []

        # ── Step 3: Aggregate ──────────────────────────────────────────────
        agg = aggregate_to_stock_score(scored)

        # ── Display result ─────────────────────────────────────────────────
        if scored:
            print(f"\n  Sentiment scores per headline:")
            print(f"  {'#':<4} {'Label':<10} {'Conf':>6}  {'Score':>7}  Headline")
            print(f"  {'─'*65}")
            for i, a in enumerate(scored, 1):
                icon = {"positive": "▲", "negative": "▼", "neutral": "●"}.get(a["label"], "●")
                print(
                    f"  {i:<4} {icon} {a['label']:<8} "
                    f"{a['confidence']:>6.2f}  "
                    f"{a['raw_score']:>+7.3f}  "
                    f"{a['title'][:48]}"
                )

        # Visual bar
        filled = int((agg["final_score"] + 1) / 2 * 30)
        bar    = "█" * filled + "░" * (30 - filled)
        print(f"\n  AGGREGATE  |{bar}|  {agg['final_score']:+.3f}")
        print(f"  Label    : {agg['label'].upper()}")
        print(f"  Confidence: {agg['confidence']:.2f}  |  "
              f"Pos: {agg['pct_positive']:.0%}  "
              f"Neg: {agg['pct_negative']:.0%}  "
              f"Neu: {agg['pct_neutral']:.0%}  |  "
              f"Std: {agg['sentiment_std']:.3f}")

        # ── Collect rows ───────────────────────────────────────────────────
        summary_rows.append({
            "date":    today,
            "ticker":  ticker,
            "company": meta.get("name", ticker),
            **agg,
        })

        for a in scored:
            detail_rows.append({
                "date":       today,
                "ticker":     ticker,
                "title":      a["title"],
                "snippet":    a.get("snippet", ""),
                "source":     a.get("source", ""),
                "news_date":  a.get("date", ""),
                "link":       a.get("link", ""),
                "label":      a["label"],
                "confidence": a["confidence"],
                "raw_score":  a["raw_score"],
                "prob_pos":   a.get("prob_pos", 0.0),
                "prob_neg":   a.get("prob_neg", 0.0),
                "prob_neu":   a.get("prob_neu", 0.0),
            })

        # Serper rate limit: ~1 req/sec on free tier
        time.sleep(1.2)

    # ── Build & save DataFrames ────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows).set_index("ticker")
    detail_df  = pd.DataFrame(detail_rows) if detail_rows else pd.DataFrame()

    summary_df.to_csv(OUTPUT_FILE)
    if not detail_df.empty:
        detail_df.to_csv(DETAIL_FILE, index=False)

    # ── Final summary table ────────────────────────────────────────────────────
    print(f"\n\n{'═'*65}")
    print(f"   FINAL SENTIMENT SUMMARY  —  {today}")
    print(f"   Serper API calls used: {total_api_calls} / 2500 monthly free")
    print(f"{'═'*65}")
    print(f"  {'TICKER':<8} {'SCORE':>7}  {'LABEL':<20} {'CONF':>6}  "
          f"{'NEWS':>5}  {'%POS':>6}  {'%NEG':>6}")
    print(f"  {'─'*63}")

    for ticker, row in summary_df.iterrows():
        direction_arrow = (
            "▲▲" if row["label"] == "bullish"          else
            "▲"  if row["label"] == "slightly_bullish"  else
            "▼▼" if row["label"] == "bearish"           else
            "▼"  if row["label"] == "slightly_bearish"  else
            "─"
        )
        print(
            f"  {ticker:<8} {row['final_score']:>+7.3f}  "
            f"{direction_arrow} {row['label']:<18} "
            f"{row['confidence']:>6.2f}  "
            f"{int(row['num_headlines']):>5}  "
            f"{row['pct_positive']:>6.0%}  "
            f"{row['pct_negative']:>6.0%}"
        )

    print(f"\n  Saved: {OUTPUT_FILE}")
    print(f"  Saved: {DETAIL_FILE}")

    return summary_df


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — OPTIMIZER INTERFACE
#  These functions are imported by feature_builder.py and optimizer.py
# ══════════════════════════════════════════════════════════════════════════════

def load_sentiment_scores() -> pd.DataFrame:
    """Load saved sentiment scores. Called by feature_builder.py."""
    if not os.path.exists(OUTPUT_FILE):
        raise FileNotFoundError(
            f"{OUTPUT_FILE} not found.\n"
            f"Run: python sentiment_engine.py first."
        )
    return pd.read_csv(OUTPUT_FILE, index_col="ticker")


def get_sentiment_adjusted_returns(
    expected_returns: pd.Series,
    sentiment_df:     pd.DataFrame,
    alpha:            float = 0.25,
) -> pd.Series:
    """
    Blend quantitative expected returns with LLM sentiment.

    Formula:
        μ_adj = (1 - α) × μ_quant  +  α × sentiment_score × |μ_quant|

    α = 0.25 means sentiment can shift expected returns by up to ±25%.
    Higher alpha = more weight given to news sentiment.

    Called by: optimizer.py
    """
    adjusted = expected_returns.copy()
    for ticker in expected_returns.index:
        if ticker in sentiment_df.index:
            s = float(sentiment_df.loc[ticker, "final_score"])
            m = float(expected_returns[ticker])
            adjusted[ticker] = (1 - alpha) * m + alpha * s * abs(m)
    return adjusted


def dynamic_alpha(confidence: float, sentiment_std: float) -> float:
    """
    Compute a dynamic sentiment blend weight (alpha) based on signal quality.

    High confidence + low disagreement across headlines → trust sentiment more.
    Low confidence or high disagreement             → stay close to quant.

    Range: [0.02, 0.20]  (quant always dominates; sentiment is a controlled tilt)
    """
    base_alpha          = 0.10
    confidence_bonus    = (confidence - 0.5) * 0.15    # up to +0.075 at conf=1.0
    uncertainty_penalty = sentiment_std * 0.20          # high std = penalise
    return round(max(0.02, min(0.20, base_alpha + confidence_bonus - uncertainty_penalty)), 4)


def get_sentiment_constraints(
    sentiment_df: pd.DataFrame,
) -> dict:
    """
    Convert sentiment labels into portfolio weight constraints.

    Constraint aggressiveness scales with FinBERT confidence — a low-confidence
    bearish signal barely constrains the stock, while a high-confidence one
    applies a tight cap.

    Confidence tiers:
      > 0.85  → strong signal  → tight constraint
      > 0.70  → moderate       → moderate constraint
      ≤ 0.70  → weak           → near-standard bounds (don't override quant)

    Label   Conf > 0.85   Conf 0.70–0.85   Conf ≤ 0.70
    ──────────────────────────────────────────────────
    bearish   cap 8%         cap 12%         cap 20%
    bullish   floor 8%       floor 5%        floor 2%
    others    standard [0%, 30%]
    """
    lower, upper = {}, {}
    for ticker, row in sentiment_df.iterrows():
        label      = row["label"]
        confidence = float(row.get("confidence", 0.5))

        if label == "bearish":
            if confidence > 0.85:
                cap = 0.08
            elif confidence > 0.70:
                cap = 0.12
            else:
                cap = 0.20
            lower[ticker] = 0.0
            upper[ticker] = cap

        elif label == "bullish":
            if confidence > 0.85:
                floor = 0.08
            elif confidence > 0.70:
                floor = 0.05
            else:
                floor = 0.02
            lower[ticker] = floor
            upper[ticker] = 0.40

        else:
            lower[ticker] = 0.0
            upper[ticker] = 0.30

    return {
        "lower_bounds": pd.Series(lower),
        "upper_bounds": pd.Series(upper),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — BLACK-LITTERMAN VIEW INTERFACE
#  Converts FinBERT sentiment scores into BL "investor views" (Q, Omega)
# ══════════════════════════════════════════════════════════════════════════════

def get_bl_views(
    sentiment_df:      pd.DataFrame,
    mu_prior:          pd.Series,
    S:                 pd.DataFrame,
    view_sensitivity:  float = 0.5,
    earnings_surprise: "pd.Series | None" = None,
    earnings_beta:     float = 0.15,
) -> tuple:
    """
    Map FinBERT sentiment scores → Black-Litterman absolute views (Q, confidence).

    Combined view formula:
        Q_i = μ_prior_i
            + α_dynamic × sentiment_score_i × σ_i × view_sensitivity   ← sentiment tilt
            + β × earnings_surprise_i × σ_i                             ← earnings tilt

    - α_dynamic scales with signal quality: high-confidence, low-disagreement
      headlines shift Q more; weak, mixed signals barely move it.
    - β=0.15 keeps earnings as a secondary signal (sentiment dominates).
    - σ_i grounds both adjustments in the stock's own volatility, so a 20%
      vol stock gets a proportionally smaller absolute return shift than a 40% one.

    Returns
    -------
    viewdict    : {ticker: expected_return}  for BlackLittermanModel(absolute_views=...)
    confidences : list[float] in [0.05, 0.95]  for Idzorek Omega calculation
    """
    viewdict    = {}
    confidences = []

    for ticker in mu_prior.index:
        if ticker not in sentiment_df.index:
            viewdict[ticker] = float(mu_prior[ticker])
            confidences.append(0.5)
            continue

        score    = float(sentiment_df.loc[ticker, "final_score"])
        conf     = float(sentiment_df.loc[ticker, "confidence"])
        sent_std = float(sentiment_df.loc[ticker, "sentiment_std"])
        sigma_i  = float(np.sqrt(S.loc[ticker, ticker])) if ticker in S.columns else 0.20
        mu_i     = float(mu_prior[ticker])

        # Dynamic alpha — scales with signal quality
        alpha_dyn = dynamic_alpha(conf, sent_std)

        # Sentiment component
        Q_i = mu_i + alpha_dyn * score * sigma_i * view_sensitivity

        # Earnings surprise component (if available)
        if earnings_surprise is not None and ticker in earnings_surprise.index:
            surprise = float(earnings_surprise[ticker])
            Q_i += earnings_beta * surprise * sigma_i

        # Idzorek confidence: clamp to [0.05, 0.95]
        idzorek_conf = float(np.clip(conf, 0.05, 0.95))

        viewdict[ticker] = round(Q_i, 6)
        confidences.append(idzorek_conf)

    return viewdict, confidences


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Run the full pipeline ─────────────────────────────────────────────────
    sentiment_df = run_sentiment_pipeline(STOCKS)

    # ── Show how the optimizer will use these scores ──────────────────────────
    print(f"\n\n{'═'*65}")
    print(f"   OPTIMIZER INTERFACE PREVIEW  (alpha = 0.25)")
    print(f"{'═'*65}")

    # Simulate quantitative expected returns (normally from returns.csv)
    np.random.seed(42)
    mock_mu = pd.Series(
        np.random.uniform(-0.005, 0.025, len(STOCKS)),
        index=STOCKS
    )

    adj_mu  = get_sentiment_adjusted_returns(mock_mu, sentiment_df, alpha=0.25)
    constr  = get_sentiment_constraints(sentiment_df)

    print(f"\n  {'TICKER':<8} {'μ_quant':>9} {'μ_adj':>9} {'Δ':>8}  {'MAX_WT':>7}  SENTIMENT_LABEL")
    print(f"  {'─'*65}")
    for t in STOCKS:
        delta = adj_mu[t] - mock_mu[t]
        maxwt = constr["upper_bounds"].get(t, 0.30)
        lbl   = sentiment_df.loc[t, "label"] if t in sentiment_df.index else "n/a"
        print(
            f"  {t:<8} {mock_mu[t]:>9.4f} {adj_mu[t]:>9.4f} "
            f"{delta:>+8.4f}  {maxwt:>7.0%}  {lbl}"
        )

    print(f"\n  Ready for feature_builder.py  →  optimizer.py")