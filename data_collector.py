# data_collector.py  —  M1: Data Collection (Indian Markets)
# ─────────────────────────────────────────────────────────────────────────────
# Downloads:
#   - Adjusted closing prices for 16 NSE equities + Nifty 50  (core 16)
#   - Nifty 100 universe (~90 stocks) for backtester.py
#   - Market capitalisations (for Black-Litterman equilibrium weights)
#   - Fundamentals (ROE, D/E, EPS growth) for quality factor scoring
#
# Also provides:  calculate_zerodha_costs(trade_value, side)
#
# Run: python data_collector.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import time
import warnings
import requests
import io
import yfinance as yf
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
STOCKS = [
    "TCS.NS",       "INFY.NS",       "WIPRO.NS",      "HCLTECH.NS",    # Technology (4)
    "HDFCBANK.NS",  "ICICIBANK.NS",  "SBIN.NS",       "KOTAKBANK.NS",  # Finance (4)
    "SUNPHARMA.NS", "DRREDDY.NS",                                        # Healthcare (2)
    "HINDUNILVR.NS","ITC.NS",                                            # Consumer/FMCG (2)
    "RELIANCE.NS",  "ONGC.NS",                                           # Energy (2)
    "LT.NS",        "BHARTIARTL.NS",                                     # Infra/Telecom (2)
]

STOCK_INFO = {
    "TCS.NS":        {"name": "Tata Consultancy Services", "sector": "Technology"},
    "INFY.NS":       {"name": "Infosys Ltd",               "sector": "Technology"},
    "WIPRO.NS":      {"name": "Wipro Ltd",                 "sector": "Technology"},
    "HCLTECH.NS":    {"name": "HCL Technologies",          "sector": "Technology"},
    "HDFCBANK.NS":   {"name": "HDFC Bank Ltd",             "sector": "Finance"},
    "ICICIBANK.NS":  {"name": "ICICI Bank Ltd",            "sector": "Finance"},
    "SBIN.NS":       {"name": "State Bank of India",       "sector": "Finance"},
    "KOTAKBANK.NS":  {"name": "Kotak Mahindra Bank",       "sector": "Finance"},
    "SUNPHARMA.NS":  {"name": "Sun Pharmaceutical",        "sector": "Healthcare"},
    "DRREDDY.NS":    {"name": "Dr. Reddy's Laboratories",  "sector": "Healthcare"},
    "HINDUNILVR.NS": {"name": "Hindustan Unilever",        "sector": "Consumer"},
    "ITC.NS":        {"name": "ITC Ltd",                   "sector": "Consumer"},
    "RELIANCE.NS":   {"name": "Reliance Industries",       "sector": "Energy"},
    "ONGC.NS":       {"name": "Oil & Natural Gas Corp",    "sector": "Energy"},
    "LT.NS":         {"name": "Larsen & Toubro",           "sector": "Infrastructure"},
    "BHARTIARTL.NS": {"name": "Bharti Airtel",             "sector": "Telecom"},
}

NIFTY50      = "^NSEI"      # Nifty 50 index (benchmark)
START_DATE   = "2018-06-01"
END_DATE     = datetime.today().strftime("%Y-%m-%d")
DATA_DIR     = "data"

# ── Nifty 100 universe ─────────────────────────────────────────────────────
# ~90 well-established NSE large/mid-caps with data going back to 2018.
# Source: Nifty 50 + Nifty Next 50 (current as of 2025).
# backtester.py uses this wider universe; the 16-stock core flow is unchanged.
NIFTY_100_URL = "https://www.niftyindices.com/IndexConstituents/ind_nifty100list.csv"

NIFTY_100_FALLBACK = [
    # ── Technology ──────────────────────────────────────────────────────────
    "TCS.NS",        "INFY.NS",        "WIPRO.NS",       "HCLTECH.NS",
    "TECHM.NS",      "MPHASIS.NS",     "OFSS.NS",        "PERSISTENT.NS",
    # ── Finance / Banking ───────────────────────────────────────────────────
    "HDFCBANK.NS",   "ICICIBANK.NS",   "SBIN.NS",        "KOTAKBANK.NS",
    "AXISBANK.NS",   "BAJFINANCE.NS",  "BAJAJFINSV.NS",  "INDUSINDBK.NS",
    "SBILIFE.NS",    "HDFCLIFE.NS",    "ICICIPRULI.NS",  "ICICIGI.NS",
    "BANKBARODA.NS", "CANBK.NS",       "PNB.NS",
    "CHOLAFIN.NS",   "MUTHOOTFIN.NS",  "SHRIRAMFIN.NS",
    # ── Consumer / FMCG ─────────────────────────────────────────────────────
    "HINDUNILVR.NS", "ITC.NS",         "BRITANNIA.NS",   "NESTLEIND.NS",
    "TATACONSUM.NS", "DABUR.NS",       "GODREJCP.NS",    "COLPAL.NS",
    "BERGEPAINT.NS",
    # ── Automobile ──────────────────────────────────────────────────────────
    "MARUTI.NS",     "M&M.NS",         "BAJAJ-AUTO.NS",  "HEROMOTOCO.NS",
    "EICHERMOT.NS",  "TATAMOTORS.NS",  "TVSMOTOR.NS",
    # ── Metals / Materials ──────────────────────────────────────────────────
    "TATASTEEL.NS",  "JSWSTEEL.NS",    "HINDALCO.NS",    "VEDL.NS",
    "NMDC.NS",       "JINDALSTEL.NS",  "SAIL.NS",
    # ── Energy ──────────────────────────────────────────────────────────────
    "RELIANCE.NS",   "ONGC.NS",        "BPCL.NS",        "COALINDIA.NS",
    "NTPC.NS",       "POWERGRID.NS",   "TATAPOWER.NS",
    # ── Infrastructure / Construction ───────────────────────────────────────
    "LT.NS",         "DLF.NS",         "GODREJPROP.NS",  "ULTRACEMCO.NS",
    "GRASIM.NS",     "AMBUJACEM.NS",   "OBEROIRLTY.NS",
    # ── Telecom / Infra ─────────────────────────────────────────────────────
    "BHARTIARTL.NS", "TATACOMM.NS",
    # ── Healthcare / Pharma ─────────────────────────────────────────────────
    "SUNPHARMA.NS",  "DRREDDY.NS",     "DIVISLAB.NS",    "CIPLA.NS",
    "APOLLOHOSP.NS", "LUPIN.NS",       "TORNTPHARM.NS",  "AUROPHARMA.NS",
    "ZYDUSLIFE.NS",
    # ── Conglomerate / Adani ────────────────────────────────────────────────
    "ADANIENT.NS",   "ADANIPORTS.NS",
    # ── Consumer Discretionary ──────────────────────────────────────────────
    "TITAN.NS",      "ASIANPAINT.NS",  "TRENT.NS",       "INDHOTEL.NS",
    "HAVELLS.NS",    "SIEMENS.NS",     "ABB.NS",         "BOSCHLTD.NS",
    "PIDILITIND.NS", "SRF.NS",         "POLYCAB.NS",
    # ── Other ───────────────────────────────────────────────────────────────
    "BAJAJHLDNG.NS", "MRF.NS",         "NAUKRI.NS",
    "CONCOR.NS",     "RECLTD.NS",      "NHPC.NS",        "BEL.NS",
    "ZOMATO.NS",
]

# Liquidity / quality filters for backtester universe
MIN_DAILY_VOLUME_CR = 5.0    # minimum average daily turnover (₹ crore)
MIN_PRICE_INR       = 50.0   # exclude penny stocks
MIN_HISTORY_DAYS    = 252    # need at least 1 year of data

# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  ZERODHA TRANSACTION COST CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

def calculate_zerodha_costs(trade_value: float, side: str = "buy") -> dict:
    """
    Calculate exact Zerodha trading costs for NSE equity delivery orders.

    Formula (per SEBI/NSE schedules 2024-25):
        brokerage  = min(₹20, 0.03% of trade_value)
        stt        = 0.1% of trade_value on SELL side only
        exchange   = 0.00345% of trade_value
        gst        = 18% of brokerage (not on STT)
        sebi       = 0.0001% of trade_value (₹10/crore)
        stamp      = 0.015% on BUY side (delivery)

    Parameters
    ----------
    trade_value : float — trade amount in INR
    side        : str   — "buy" or "sell"

    Returns
    -------
    dict with individual cost components and total
    """
    if trade_value <= 0:
        return {"brokerage": 0, "stt": 0, "exchange": 0,
                "gst": 0, "sebi": 0, "stamp": 0, "total": 0, "total_pct": 0}

    brokerage = min(20.0, 0.0003 * trade_value)
    stt       = 0.001  * trade_value if side == "sell" else 0.0
    exchange  = 0.0000345 * trade_value
    gst       = 0.18   * brokerage
    sebi      = 0.000001 * trade_value
    stamp     = 0.00015  * trade_value if side == "buy" else 0.0  # 0.015%

    total     = brokerage + stt + exchange + gst + sebi + stamp

    return {
        "brokerage":   round(brokerage, 4),
        "stt":         round(stt,       4),
        "exchange":    round(exchange,  4),
        "gst":         round(gst,       4),
        "sebi":        round(sebi,      4),
        "stamp":       round(stamp,     4),
        "total":       round(total,     4),
        "total_pct":   round(total / trade_value * 100, 4),
    }


def total_trade_cost(trade_value: float, side: str = "buy") -> float:
    """Convenience function — returns total cost in ₹."""
    return calculate_zerodha_costs(trade_value, side)["total"]


# ══════════════════════════════════════════════════════════════════════════════
#  1. PRICE & RETURNS
# ══════════════════════════════════════════════════════════════════════════════

def download_prices():
    """Download adjusted closing prices for all NSE stocks + Nifty 50."""
    os.makedirs(DATA_DIR, exist_ok=True)

    print("📥 Downloading NSE stock prices + Nifty 50...")
    all_tickers = STOCKS + [NIFTY50]
    raw    = yf.download(all_tickers, start=START_DATE, end=END_DATE, auto_adjust=True)
    prices = raw["Close"].rename(columns={NIFTY50: "NIFTY50"})

    # Drop columns that are all NaN
    prices = prices.dropna(axis=1, how="all")

    print(f"✅ {len(prices)} trading days for {len(prices.columns)} tickers  "
          f"({prices.index[0].date()} → {prices.index[-1].date()})")

    returns = prices.pct_change().dropna()
    prices.to_csv(f"{DATA_DIR}/prices.csv")
    returns.to_csv(f"{DATA_DIR}/returns.csv")
    print(f"💾 Saved prices.csv and returns.csv")
    return prices, returns


# ══════════════════════════════════════════════════════════════════════════════
#  2. MARKET CAPITALISATIONS  (for Black-Litterman equilibrium weights)
# ══════════════════════════════════════════════════════════════════════════════

def download_market_caps() -> pd.Series:
    """
    Fetch market caps via yfinance fast_info.
    For NSE stocks, yfinance returns market cap in INR.
    Saves data/market_caps.csv.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    DEFAULT_MCAP = 5e12   # ₹5 Lakh Crore fallback (approx large-cap)

    print("📊 Fetching NSE market capitalisations...")
    mcaps = {}
    for ticker in STOCKS:
        try:
            t    = yf.Ticker(ticker)
            mcap = t.fast_info.market_cap
            mcaps[ticker] = float(mcap) if mcap and mcap > 0 else DEFAULT_MCAP
            time.sleep(0.15)
        except Exception:
            mcaps[ticker] = DEFAULT_MCAP

    series = pd.Series(mcaps, name="market_cap_inr")
    series.to_csv(f"{DATA_DIR}/market_caps.csv", header=True)
    print("✅ Market caps saved: " +
          "  ".join(f"{t.replace('.NS','')}=₹{v/1e12:.1f}T" for t, v in mcaps.items()))
    return series


# ══════════════════════════════════════════════════════════════════════════════
#  3. FUNDAMENTALS  (for quality factor scoring)
# ══════════════════════════════════════════════════════════════════════════════

def download_fundamentals() -> pd.DataFrame:
    """
    Fetch fundamental data for all STOCKS via yfinance Ticker.info.

    Fields fetched per stock:
        returnOnEquity   — Return on Equity (e.g. 0.25 = 25%)
        debtToEquity     — Debt-to-Equity ratio  (lower = better)
        earningsGrowth   — YoY EPS growth (e.g. 0.12 = 12%)
        trailingEps      — Trailing twelve-month EPS (₹)

    Falls back to NaN if yfinance returns None so scorer.py handles it.
    Saves: data/fundamentals.csv
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    FIELDS   = ["returnOnEquity", "debtToEquity", "earningsGrowth", "trailingEps"]
    DEFAULTS = {
        "returnOnEquity": float("nan"),
        "debtToEquity":   float("nan"),
        "earningsGrowth": float("nan"),
        "trailingEps":    float("nan"),
    }

    print("📊 Fetching fundamental data (ROE / D:E / EPS growth)...")
    rows = {}

    for ticker in STOCKS:
        row = dict(DEFAULTS)   # start with NaN defaults
        try:
            info = yf.Ticker(ticker).info
            for field in FIELDS:
                val = info.get(field, None)
                if val is not None:
                    try:
                        row[field] = float(val)
                    except (TypeError, ValueError):
                        pass
        except Exception as e:
            print(f"  ⚠️  {ticker}: yfinance error ({e}) — using NaN defaults")

        rows[ticker] = row
        time.sleep(0.2)   # gentle rate-limiting

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "ticker"
    df.to_csv(f"{DATA_DIR}/fundamentals.csv")

    print(f"✅ Fundamentals saved ({len(df)} stocks)")
    for ticker, row in df.iterrows():
        roe  = f"{row['returnOnEquity']:.1%}" if pd.notna(row["returnOnEquity"]) else "n/a"
        de   = f"{row['debtToEquity']:.2f}"   if pd.notna(row["debtToEquity"])   else "n/a"
        epsg = f"{row['earningsGrowth']:.1%}" if pd.notna(row["earningsGrowth"]) else "n/a"
        name = STOCK_INFO.get(ticker, {}).get("name", ticker)
        print(f"  {ticker.replace('.NS',''):<14}  ROE={roe:>7}  D/E={de:>6}  EPSg={epsg:>7}  {name}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  4. NIFTY 100 UNIVERSE  (for backtester.py)
# ══════════════════════════════════════════════════════════════════════════════

def get_nifty100_tickers() -> list:
    """
    Fetch current Nifty 100 constituents from NSE website.
    Falls back to the hardcoded NIFTY_100_FALLBACK list on any error.
    """
    try:
        resp = requests.get(NIFTY_100_URL, timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (compatible; PortfolioAI/1.0)"
        })
        resp.raise_for_status()
        df  = pd.read_csv(io.StringIO(resp.text))
        # NSE CSV usually has a 'Symbol' column
        sym_col = next((c for c in df.columns if "Symbol" in c or "symbol" in c), None)
        if sym_col and len(df) > 50:
            tickers = [s.strip().upper() + ".NS" for s in df[sym_col].dropna()]
            # Fix common ticker name differences in yfinance
            tickers = [t.replace("BAJAJ-AUTO.NS", "BAJAJ-AUTO.NS")
                        .replace("M&M.NS", "M&M.NS") for t in tickers]
            print(f"  ✅ Downloaded Nifty 100 list from NSE ({len(tickers)} stocks)")
            return tickers
    except Exception as e:
        print(f"  ⚠️  NSE download failed ({e}) — using hardcoded Nifty 100 fallback")

    return list(NIFTY_100_FALLBACK)


def download_nifty100_universe(
    apply_filters: bool = True,
    min_vol_cr:    float = MIN_DAILY_VOLUME_CR,
    min_price:     float = MIN_PRICE_INR,
) -> list:
    """
    Get filtered Nifty 100 universe for backtesting.

    Filters:
        - Price >= ₹50   (exclude penny stocks)
        - Avg daily turnover >= ₹5 Cr  (exclude illiquid stocks)
        - At least 252 trading days of data

    Saves filtered list to data/universe.csv.
    Returns list of valid tickers.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    tickers = get_nifty100_tickers()

    if not apply_filters:
        pd.Series(tickers, name="ticker").to_csv(f"{DATA_DIR}/universe.csv", index=False)
        return tickers

    print(f"\n🔍 Filtering {len(tickers)} Nifty 100 candidates "
          f"(price≥₹{min_price:.0f}, turnover≥₹{min_vol_cr:.0f}Cr)...")

    # Download last 30 days of data for all candidates at once
    try:
        raw = yf.download(
            tickers + [NIFTY50], period="30d",
            auto_adjust=True, progress=False,
        )
        close  = raw["Close"]  if "Close"  in raw else raw.xs("Close",  axis=1, level=0)
        volume = raw["Volume"] if "Volume" in raw else raw.xs("Volume", axis=1, level=0)
    except Exception as e:
        print(f"  ⚠️  Filter download failed ({e}) — returning unfiltered list")
        pd.Series(tickers, name="ticker").to_csv(f"{DATA_DIR}/universe.csv", index=False)
        return tickers

    filtered = []
    for t in tickers:
        try:
            t_col = t  # column name might match exactly
            if t_col not in close.columns:
                continue
            p_series = close[t_col].dropna()
            v_series = volume[t_col].dropna()

            if len(p_series) < 5:
                continue

            last_price    = float(p_series.iloc[-1])
            avg_volume    = float(v_series.mean())
            daily_turnover_cr = (last_price * avg_volume) / 1e7   # ₹ crore

            if last_price >= min_price and daily_turnover_cr >= min_vol_cr:
                filtered.append(t)
        except Exception:
            pass   # skip problematic tickers

    # Also check history length using START_DATE
    # (already handled in backtester via MIN_HISTORY_DAYS check per date)

    print(f"  ✅ {len(filtered)} stocks passed filters  "
          f"(removed {len(tickers) - len(filtered)} illiquid/cheap)")

    pd.DataFrame({"ticker": filtered}).to_csv(f"{DATA_DIR}/universe.csv", index=False)
    print(f"  💾 Saved: {DATA_DIR}/universe.csv")
    return filtered


def download_nifty100_data(tickers: list = None) -> tuple:
    """
    Download full price history for the Nifty 100 universe.

    Saves:
        data/nifty100_prices.csv  — daily OHLCV adjusted close
        data/nifty100_returns.csv — daily returns

    Parameters
    ----------
    tickers : list (optional) — if None, loads from data/universe.csv or uses fallback

    Returns
    -------
    (prices_df, returns_df)
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if tickers is None:
        univ_path = f"{DATA_DIR}/universe.csv"
        if os.path.exists(univ_path):
            tickers = pd.read_csv(univ_path)["ticker"].tolist()
            print(f"  Loaded universe: {len(tickers)} tickers from universe.csv")
        else:
            tickers = list(NIFTY_100_FALLBACK)
            print(f"  Using fallback universe: {len(tickers)} tickers")

    all_tickers = list(tickers) + [NIFTY50]

    print(f"\n📥 Downloading Nifty 100 price history ({len(tickers)} stocks + benchmark)...")
    print(f"   Period: {START_DATE} → {END_DATE}")

    raw    = yf.download(all_tickers, start=START_DATE, end=END_DATE, auto_adjust=True)
    prices = raw["Close"].rename(columns={NIFTY50: "NIFTY50"})
    prices = prices.dropna(axis=1, how="all")

    n_tickers = len([c for c in prices.columns if c != "NIFTY50"])
    print(f"✅ Downloaded {n_tickers} stocks  ({prices.index[0].date()} → {prices.index[-1].date()})")

    returns = prices.pct_change().dropna()

    prices.to_csv(f"{DATA_DIR}/nifty100_prices.csv")
    returns.to_csv(f"{DATA_DIR}/nifty100_returns.csv")
    print(f"💾 Saved: nifty100_prices.csv  nifty100_returns.csv")

    return prices, returns


# ══════════════════════════════════════════════════════════════════════════════
#  5. REBALANCING DATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_rebalancing_dates(returns, frequency_weeks=2):
    """Generate bi-weekly rebalancing dates (every 10 trading days). Kept for backward compat."""
    all_dates   = returns.index.tolist()
    rebal_dates = all_dates[::frequency_weeks * 5]
    print(f"📅 Generated {len(rebal_dates)} rebalancing dates")
    return rebal_dates


def get_lookback_window(returns, rebal_date, lookback_days=252):
    """Historical returns window up to (not including) rebal_date."""
    idx       = returns.index.get_loc(rebal_date)
    start_idx = max(0, idx - lookback_days)
    return returns.iloc[start_idx:idx]


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    full_mode = "--full" in sys.argv   # python data_collector.py --full downloads Nifty 100

    # ── Core 16-stock data (always downloaded) ────────────────────────────────
    prices, returns = download_prices()
    mcaps           = download_market_caps()
    fundamentals    = download_fundamentals()

    print(f"\n📈 Latest prices (₹):")
    stock_cols = [t for t in STOCKS if t in prices.columns]
    latest = prices[stock_cols].iloc[-1].round(2)
    for ticker, price in latest.items():
        name = STOCK_INFO.get(ticker, {}).get("name", ticker)
        print(f"  {ticker.replace('.NS',''):<14} ₹{price:>10,.2f}  —  {name}")

    # ── Nifty 100 universe (for backtester.py — opt-in with --full) ───────────
    if full_mode:
        print("\n" + "═"*55)
        print("  DOWNLOADING NIFTY 100 UNIVERSE (--full mode)")
        print("  This downloads ~90 stocks and takes 2-3 minutes")
        print("═"*55)
        universe = download_nifty100_universe(apply_filters=True)
        n100_prices, n100_returns = download_nifty100_data(universe)
        print(f"\n✅ Nifty 100 data ready: {len(universe)} stocks")
        print("   Now run:  python backtester.py")
    else:
        print("\n💡 Tip: run  python data_collector.py --full  to also download the")
        print("   full Nifty 100 universe for backtester.py (takes ~3 min)")

    rebal_dates = get_rebalancing_dates(returns)
    print(f"\n🗓️  Next step: python scorer.py  →  python llm_views.py")
