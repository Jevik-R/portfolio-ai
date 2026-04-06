# 📈 PortfolioAI — GenAI-Driven Portfolio Optimisation for Indian Markets

> **BTech Minor Project 2024–25**  
> AI-powered NSE portfolio optimisation combining Black-Litterman model with FinBERT sentiment analysis

---

## 🎯 Overview

PortfolioAI is a full-stack portfolio optimisation system that helps Indian retail investors intelligently allocate capital across 16 Nifty-50 stocks. It combines:

- **Black-Litterman Model** — Market-implied equilibrium returns as prior, updated by LLM sentiment views
- **FinBERT Sentiment** — Pre-trained financial NLP model scores live news headlines per stock
- **Macro Regime Overlay** — Nifty 50 moving averages + VIX fear index scales equity exposure
- **CVaR Optimisation** — Minimises worst-case tail losses for conservative profiles
- **Streamlit Dashboard** — Product-level UI with fresh investment and portfolio rebalancing flows

---

## 🏗️ Architecture

```
data_collector.py      ← M1: Downloads NSE prices, Nifty 50, market caps
        ↓
sentiment_engine.py    ← M2: Fetches news (Serper API) → FinBERT scoring
        ↓
feature_builder.py     ← M3: BL prior μ, covariance Σ, LLM views Q, posterior μ_BL
        ↓
optimizer.py           ← M4: EfficientFrontier / CVaR / walk-forward backtest
        ↓
dashboard.py           ← M5: Streamlit product dashboard
macro_overlay.py       ← L3: Nifty regime detector + VIX overlay
```

---

## 🚀 Quick Start

### 1. Clone & set up environment

```bash
git clone https://github.com/Jevik-R/portfolio-ai.git
cd portfolio-ai

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API keys

Create a `.env` file (see `.env.example`):

```bash
SERPER_API_KEY=your_serper_key      # https://serper.dev — 2500 free queries/month
OPENAI_API_KEY=your_openai_key      # Optional — for GPT rationale & narratives
```

Or set `SERPER_API_KEY` directly in `sentiment_engine.py` line 27.

### 3. Run the pipeline

```bash
# Step 1 — Download 7 years of NSE market data
python data_collector.py

# Step 2 — Fetch live news & score with FinBERT (~5 min, downloads model once)
python sentiment_engine.py

# Step 3 — Launch the dashboard
streamlit run dashboard.py
```

Open **http://localhost:8501** in your browser.

---

## 📊 Stock Universe — 16 NSE Large-Caps

| Sector | Stocks |
|--------|--------|
| Technology | TCS, Infosys, Wipro, HCL Technologies |
| Finance | HDFC Bank, ICICI Bank, SBI, Kotak Mahindra Bank |
| Healthcare | Sun Pharma, Dr. Reddy's |
| Consumer/FMCG | Hindustan Unilever, ITC |
| Energy | Reliance Industries, ONGC |
| Infrastructure | Larsen & Toubro |
| Telecom | Bharti Airtel |

---

## 🧠 Technical Methodology

### Black-Litterman Model

```
μ_prior  =  δ × Σ × w_mkt          (market-implied equilibrium returns)
Q_i      =  μ_prior_i
           + α_dynamic × sentiment_score × σ_i × 0.5    (FinBERT tilt)
           + β × earnings_surprise × σ_i                 (earnings signal)
μ_BL     =  BL posterior via Idzorek confidence weighting
```

### FinBERT Sentiment Pipeline

1. Fetch 20 headlines per stock via Serper API (Google News)
2. Score each headline with `ProsusAI/finbert` → {positive, negative, neutral}
3. Confidence-weighted aggregation with source credibility weights
   - Reuters / Bloomberg: 1.5×  |  Seeking Alpha: 0.7×  |  ET / Mint: 1.4×
4. Dynamic alpha: `α = 0.10 + (conf − 0.5) × 0.15 − std × 0.20`

### Risk Profiles

| Profile | Objective | Max Weight |
|---------|-----------|-----------|
| Conservative | Min CVaR (5% tail) | 20% |
| Moderate | Min Volatility / Max Sharpe | 30% |
| Aggressive | Max Sharpe | 40% |

### Macro Overlay (Level 3)

```
Regime scale:  Bull 100% | Neutral 90% | Bear 70%   (Nifty 50 vs MA50/MA200)
VIX scale:     Normal 100% | Elevated 85-92% | High 75% | Extreme 50%
Combined:      max(40%, regime_scale × vix_scale)
```

---

## 📁 Project Structure

```
portfolio-ai/
├── data_collector.py      # M1 — NSE data download
├── sentiment_engine.py    # M2 — FinBERT news sentiment
├── feature_builder.py     # M3 — BL feature engineering
├── optimizer.py           # M4 — Portfolio optimisation + backtest
├── dashboard.py           # M5 — Streamlit dashboard
├── macro_overlay.py       # L3 — Market regime + VIX overlay
├── data/
│   ├── market_caps.csv    # NSE market capitalisations (BL weights)
│   ├── sentiment_scores.csv  # Latest FinBERT scores
│   └── backtest_metrics.csv  # Walk-forward backtest results
├── requirements.txt
└── README.md
```

---

## 🖥️ Dashboard Features

### 💰 Fresh Investment Flow
- Enter ₹ amount → Black-Litterman optimisation → allocation with exact shares
- Donut chart coloured by sector | Horizontal bar chart by sentiment signal
- AI-generated rationale per stock (template or GPT-3.5)

### 🔄 Portfolio Rebalancer
- Enter current holdings (₹) → BUY / SELL / HOLD plan
- Cash flow summary | Transaction cost estimate (0.1% brokerage)
- Current vs target allocation comparison

### 📉 Backtest Tab
- Walk-forward backtest: June 2018 → present (bi-weekly rebalancing)
- BL + Sentiment vs Pure Quant vs Nifty 50 cumulative return chart
- Look-ahead bias warning (sentiment is live-only)

### 🌍 Macro Panel
- Live Nifty 50 regime badge (🟢 Bull / 🟡 Neutral / 🔴 Bear)
- Live VIX fear gauge | Combined equity scale | Cash buffer %
- Sector sentiment heatmap | Analyst consensus scores

---

## ⚙️ Configuration

| Parameter | File | Default | Description |
|-----------|------|---------|-------------|
| `DELTA` | feature_builder.py | 2.5 | Market risk aversion |
| `TAU` | feature_builder.py | 0.025 | BL prior confidence |
| `VIEW_SENSITIVITY` | feature_builder.py | 0.5 | Sentiment → return magnitude |
| `MAX_HEADLINES` | sentiment_engine.py | 20 | News per stock |
| `RISK_FREE_INR` | optimizer.py | 0.065 | Indian G-Sec yield |
| `LOOKBACK` | optimizer.py | 252 | Covariance window (trading days) |

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `yfinance` | NSE price data & market caps |
| `pypfopt` | Black-Litterman, EfficientFrontier, CVaR |
| `transformers` | FinBERT model (`ProsusAI/finbert`) |
| `torch` | FinBERT inference backend |
| `streamlit` | Web dashboard |
| `plotly` | Interactive charts |
| `requests` | Serper API news fetching |

---

## ⚠️ Disclaimer

This tool is for **educational purposes only** and does not constitute financial advice.  
NSE equity investments are subject to market risk. Brokerage fees, STT, and capital gains tax (LTCG/STCG) apply.  
Consult a **SEBI-registered investment advisor (RIA)** before investing.

---

## 👨‍💻 Author

Built as a **BTech Minor Project**  
Powered by: FinBERT · Black-Litterman · PyPortfolioOpt · Streamlit
