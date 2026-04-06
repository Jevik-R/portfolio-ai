# data_collector.py  —  M1: Data Collection (Indian Markets)
# ─────────────────────────────────────────────────────────────────────────────
# Downloads:
#   - Adjusted closing prices for 16 NSE equities + Nifty 50
#   - Market capitalisations (for Black-Litterman equilibrium weights)
#
# Run: python data_collector.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import time
import yfinance as yf
import pandas as pd
from datetime import datetime

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
# ─────────────────────────────────────────────────────────────────────────────


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
#  3. REBALANCING DATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_rebalancing_dates(returns, frequency_weeks=2):
    """Generate bi-weekly rebalancing dates (every 10 trading days)."""
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
    prices, returns = download_prices()
    mcaps           = download_market_caps()

    print(f"\n📈 Latest prices (₹):")
    stock_cols = [t for t in STOCKS if t in prices.columns]
    latest = prices[stock_cols].iloc[-1].round(2)
    for ticker, price in latest.items():
        name = STOCK_INFO.get(ticker, {}).get("name", ticker)
        print(f"  {ticker.replace('.NS',''):<14} ₹{price:>10,.2f}  —  {name}")

    rebal_dates = get_rebalancing_dates(returns)
    print(f"\n🗓️  First 5 rebalancing dates: {[str(d.date()) for d in rebal_dates[:5]]}")
