# dashboard.py  —  M5: Product Dashboard
# ─────────────────────────────────────────────────────────────────────────────
# Run:  streamlit run dashboard.py
#
# Two user flows:
#   💰 Fresh Investment  — enter ₹ amount → get optimal allocation
#   🔄 Portfolio Rebalancer — enter current holdings → get rebalancing plan
#
# Powered by: FinBERT sentiment + Black-Litterman MVO + INR pricing
# ─────────────────────────────────────────────────────────────────────────────

import os, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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

STOCK_META = {
    "TCS.NS":        {"name": "Tata Consultancy Services", "sector": "Technology",      "flag": "💻"},
    "INFY.NS":       {"name": "Infosys Ltd",               "sector": "Technology",      "flag": "🔷"},
    "WIPRO.NS":      {"name": "Wipro Ltd",                 "sector": "Technology",      "flag": "🌐"},
    "HCLTECH.NS":    {"name": "HCL Technologies",          "sector": "Technology",      "flag": "⚙️"},
    "HDFCBANK.NS":   {"name": "HDFC Bank Ltd",             "sector": "Finance",         "flag": "🏦"},
    "ICICIBANK.NS":  {"name": "ICICI Bank Ltd",            "sector": "Finance",         "flag": "🏛️"},
    "SBIN.NS":       {"name": "State Bank of India",       "sector": "Finance",         "flag": "🇮🇳"},
    "KOTAKBANK.NS":  {"name": "Kotak Mahindra Bank",       "sector": "Finance",         "flag": "💼"},
    "SUNPHARMA.NS":  {"name": "Sun Pharmaceutical",        "sector": "Healthcare",      "flag": "💊"},
    "DRREDDY.NS":    {"name": "Dr. Reddy's Labs",          "sector": "Healthcare",      "flag": "🔬"},
    "HINDUNILVR.NS": {"name": "Hindustan Unilever",        "sector": "Consumer",        "flag": "🛒"},
    "ITC.NS":        {"name": "ITC Ltd",                   "sector": "Consumer",        "flag": "🏭"},
    "RELIANCE.NS":   {"name": "Reliance Industries",       "sector": "Energy",          "flag": "⛽"},
    "ONGC.NS":       {"name": "Oil & Natural Gas Corp",    "sector": "Energy",          "flag": "🛢️"},
    "LT.NS":         {"name": "Larsen & Toubro",           "sector": "Infrastructure",  "flag": "🏗️"},
    "BHARTIARTL.NS": {"name": "Bharti Airtel",             "sector": "Telecom",         "flag": "📡"},
}

SECTOR_COLORS = {
    "Technology":     "#3b82f6",
    "Finance":        "#f59e0b",
    "Healthcare":     "#10b981",
    "Consumer":       "#8b5cf6",
    "Energy":         "#ef4444",
    "Infrastructure": "#06b6d4",
    "Telecom":        "#f97316",
}

SENTIMENT_CONFIG = {
    "bullish":          {"color": "#16a34a", "icon": "▲▲", "badge": "🟢"},
    "slightly_bullish": {"color": "#86efac", "icon": "▲",  "badge": "🟡"},
    "neutral":          {"color": "#94a3b8", "icon": "─",  "badge": "⚪"},
    "slightly_bearish": {"color": "#fca5a5", "icon": "▼",  "badge": "🟠"},
    "bearish":          {"color": "#dc2626", "icon": "▼▼", "badge": "🔴"},
}

ACTION_COLORS = {
    "BUY":  "#16a34a",
    "SELL": "#dc2626",
    "HOLD": "#94a3b8",
}


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def fmt_inr(amount: float, compact: bool = False) -> str:
    """Indian currency formatting."""
    if compact:
        if amount >= 1e7:  return f"₹{amount/1e7:.1f}Cr"
        if amount >= 1e5:  return f"₹{amount/1e5:.1f}L"
        if amount >= 1e3:  return f"₹{amount/1e3:.0f}K"
        return f"₹{amount:.0f}"
    # Full formatting with Indian comma system
    if amount >= 1e7:  return f"₹{amount/1e7:.2f} Crore"
    if amount >= 1e5:  return f"₹{amount/1e5:.2f} Lakh"
    return f"₹{amount:,.0f}"


def check_data_files() -> dict:
    required = {
        "prices.csv":           f"{DATA_DIR}/prices.csv",
        "returns.csv":          f"{DATA_DIR}/returns.csv",
        "sentiment_scores.csv": f"{DATA_DIR}/sentiment_scores.csv",
        "market_caps.csv":      f"{DATA_DIR}/market_caps.csv",
    }
    optional = {
        "backtest_results.csv": f"{DATA_DIR}/backtest_results.csv",
        "backtest_metrics.csv": f"{DATA_DIR}/backtest_metrics.csv",
    }
    return {
        "required_ok": all(os.path.exists(p) for p in required.values()),
        "required":    {k: os.path.exists(v) for k, v in required.items()},
        "optional":    {k: os.path.exists(v) for k, v in optional.items()},
    }


@st.cache_data(ttl=300)
def load_sentiment_df():
    p = f"{DATA_DIR}/sentiment_scores.csv"
    if not os.path.exists(p):
        return None
    return pd.read_csv(p, index_col="ticker")


@st.cache_data(ttl=300)
def load_backtest():
    rp = f"{DATA_DIR}/backtest_results.csv"
    mp = f"{DATA_DIR}/backtest_metrics.csv"
    results = pd.read_csv(rp, index_col="date", parse_dates=True) if os.path.exists(rp) else None
    metrics = pd.read_csv(mp) if os.path.exists(mp) else None
    return results, metrics


@st.cache_data(ttl=300)
def load_prices_inr():
    """Latest NSE stock prices (already in ₹ — no FX conversion needed)."""
    p = f"{DATA_DIR}/prices.csv"
    if not os.path.exists(p):
        return None
    prices = pd.read_csv(p, index_col=0, parse_dates=True)
    latest = prices[[t for t in STOCKS if t in prices.columns]].iloc[-1]
    return latest.round(2)


@st.cache_data(ttl=3600)
def load_macro_snapshot():
    """Run market regime + VIX overlay (cached 1 hour — regime changes slowly)."""
    p = f"{DATA_DIR}/prices.csv"
    if not os.path.exists(p):
        return None
    try:
        from macro_overlay import get_macro_snapshot
        prices = pd.read_csv(p, index_col=0, parse_dates=True)
        return get_macro_snapshot(prices)
    except Exception:
        return None


def run_fresh_optimizer(investment_inr, risk_profile):
    """Run optimizer and cache result in session state."""
    from optimizer import optimize_fresh_investment
    return optimize_fresh_investment(investment_inr, risk_profile)


def run_rebalance_optimizer(current_holdings, additional_inr, risk_profile):
    """Run rebalancer and cache result in session state."""
    from optimizer import optimize_rebalancing
    return optimize_rebalancing(current_holdings, additional_inr, risk_profile)


# ══════════════════════════════════════════════════════════════════════════════
#  CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def chart_allocation_donut(allocation_df, sentiment_df, title="Portfolio Allocation"):
    """Donut chart coloured by sector."""
    labels, values, colors, hovers = [], [], [], []
    for _, row in allocation_df.iterrows():
        t     = row["ticker"]
        meta  = STOCK_META.get(t, {})
        sent  = sentiment_df.loc[t, "label"] if sentiment_df is not None and t in sentiment_df.index else "n/a"
        s_cfg = SENTIMENT_CONFIG.get(sent, {})
        labels.append(f"{t}")
        values.append(row["invested_inr"])
        colors.append(SECTOR_COLORS.get(meta.get("sector", ""), "#94a3b8"))
        hovers.append(f"<b>{t}</b> — {meta.get('name','')}<br>"
                      f"Amount: ₹{row['invested_inr']:,.0f}<br>"
                      f"Weight: {row['target_weight']:.1%}<br>"
                      f"Sentiment: {s_cfg.get('badge','') } {sent}")

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker_colors=colors,
        hole=0.55,
        hovertext=hovers, hoverinfo="text",
        textinfo="label+percent",
        textfont_size=13,
    ))
    fig.update_layout(
        title=dict(text=title, font_size=16),
        showlegend=False,
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def chart_allocation_bar(allocation_df, sentiment_df):
    """Horizontal bar chart with sentiment colour coding."""
    df  = allocation_df.copy()
    df  = df[df["invested_inr"] > 0].sort_values("target_weight")

    colors = []
    for t in df["ticker"]:
        sent = sentiment_df.loc[t, "label"] if sentiment_df is not None and t in sentiment_df.index else "neutral"
        colors.append(SENTIMENT_CONFIG.get(sent, {}).get("color", "#94a3b8"))

    fig = go.Figure(go.Bar(
        x    = df["invested_inr"],
        y    = df["ticker"],
        orientation = "h",
        marker_color = colors,
        text = [f"₹{v:,.0f}  ({w:.1%})"
                for v, w in zip(df["invested_inr"], df["target_weight"])],
        textposition = "outside",
        hovertemplate = "<b>%{y}</b><br>₹%{x:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title  = "Invested Amount (₹)",
        plot_bgcolor = "white",
        height       = 400,
        margin       = dict(l=80, r=120, t=20, b=40),
        xaxis        = dict(showgrid=True, gridcolor="#e2e8f0"),
    )
    return fig


def chart_rebalance(rebalance_df):
    """Waterfall-style bar chart: green = BUY, red = SELL."""
    df = rebalance_df[rebalance_df["action"] != "HOLD"].copy()
    if df.empty:
        return None

    df["amount_signed"] = df.apply(
        lambda r: r["trade_inr"] if r["action"] == "BUY" else -r["trade_inr"], axis=1
    )
    df = df.sort_values("amount_signed")
    colors = [ACTION_COLORS.get(a, "#94a3b8") for a in df["action"]]

    fig = go.Figure(go.Bar(
        x    = df["ticker"],
        y    = df["amount_signed"],
        marker_color = colors,
        text = [f"{'+'if r>0 else ''}₹{abs(r):,.0f}" for r in df["amount_signed"]],
        textposition = "outside",
        hovertemplate = "<b>%{x}</b><br>%{text}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#1e293b", line_width=1)
    fig.update_layout(
        title       = "Rebalancing Trades Required",
        yaxis_title = "Amount (₹)  — positive = BUY  |  negative = SELL",
        plot_bgcolor = "white",
        height      = 380,
        margin      = dict(l=60, r=60, t=50, b=40),
    )
    return fig


def chart_current_vs_target(rebalance_df, total_inr):
    """Grouped bar: current vs target allocation."""
    df = rebalance_df.copy()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name = "Current Holdings",
        x    = df["ticker"],
        y    = df["current_inr"],
        marker_color = "#94a3b8",
        text = [f"₹{v:,.0f}" for v in df["current_inr"]],
        textposition = "outside",
    ))
    fig.add_trace(go.Bar(
        name = "Target Allocation",
        x    = df["ticker"],
        y    = df["target_inr"],
        marker_color = "#3b82f6",
        text = [f"₹{v:,.0f}" for v in df["target_inr"]],
        textposition = "outside",
    ))
    fig.update_layout(
        barmode     = "group",
        title       = "Current vs Optimised Target",
        yaxis_title = "Value (₹)",
        plot_bgcolor = "white",
        height      = 400,
        legend      = dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def chart_cumulative_return(backtest_df):
    cum_sent = (1 + backtest_df["ret_sentiment"]).cumprod() - 1
    cum_base = (1 + backtest_df["ret_baseline"]).cumprod()  - 1
    cum_sp   = (1 + backtest_df["ret_nifty50"].fillna(0)).cumprod() - 1

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3],
                        subplot_titles=("Cumulative Return", "Drawdown"))

    fig.add_trace(go.Scatter(x=backtest_df.index, y=cum_sent,
                             name="BL + Sentiment", line=dict(color="#3b82f6", width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=backtest_df.index, y=cum_base,
                             name="Pure Quant",     line=dict(color="#f59e0b", width=1.8, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=backtest_df.index, y=cum_sp,
                             name="Nifty 50",       line=dict(color="#64748b", width=1.5, dash="dot")), row=1, col=1)

    cum_port = (1 + backtest_df["ret_sentiment"]).cumprod()
    drawdown = (cum_port - cum_port.cummax()) / cum_port.cummax()
    fig.add_trace(go.Scatter(x=backtest_df.index, y=drawdown,
                             fill="tozeroy", line=dict(color="#dc2626"),
                             name="Drawdown", showlegend=False), row=2, col=1)

    fig.update_layout(
        height      = 540,
        plot_bgcolor = "white",
        yaxis_tickformat  = ".0%",
        yaxis2_tickformat = ".0%",
        legend      = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def chart_sentiment_scores(sentiment_df):
    df = sentiment_df.reset_index().sort_values("final_score", ascending=True)
    colors = [SENTIMENT_CONFIG.get(l, {}).get("color", "#94a3b8")
              for l in df["label"]]

    fig = go.Figure(go.Bar(
        x    = df["final_score"],
        y    = df["ticker"],
        orientation  = "h",
        marker_color = colors,
        text = [f"{s:+.3f}" for s in df["final_score"]],
        textposition = "outside",
        hovertemplate = (
            "<b>%{y}</b><br>Score: %{x:.3f}<br>"
            "Headlines: %{customdata[0]}<br>"
            "% Positive: %{customdata[1]:.0%}<extra></extra>"
        ),
        customdata = df[["num_headlines", "pct_positive"]].values,
    ))
    fig.add_vline(x=0.30,  line_dash="dash", line_color="#16a34a",
                  annotation_text="Bullish", annotation_position="top")
    fig.add_vline(x=-0.30, line_dash="dash", line_color="#dc2626",
                  annotation_text="Bearish", annotation_position="top")
    fig.add_vline(x=0, line_color="#1e293b", line_width=1)
    fig.update_layout(
        xaxis_range  = [-1.1, 1.3],
        xaxis_title  = "FinBERT Sentiment Score",
        plot_bgcolor = "white",
        height       = 400,
        margin       = dict(l=60, r=60, t=20, b=40),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  RATIONALE GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_rationale(ticker, weight, target_inr, sentiment_row,
                        mu_bl_val, mu_prior_val, prices_inr,
                        use_openai=False, openai_key="") -> str:
    meta  = STOCK_META.get(ticker, {})
    name  = meta.get("name", ticker)
    score = float(sentiment_row.get("final_score", 0.0))
    label = sentiment_row.get("label", "neutral")
    n_news = int(sentiment_row.get("num_headlines", 0))
    pct_pos = float(sentiment_row.get("pct_positive", 0.0))
    price_inr = float(prices_inr.get(ticker, 0))
    shares = round(target_inr / price_inr, 4) if price_inr > 0 else 0

    stance = ("overweight" if weight >= 0.12 else
               "underweight" if weight <= 0.04 else "market-weight")
    alignment = ("aligned" if (score > 0 and mu_bl_val > mu_prior_val) or
                              (score < 0 and mu_bl_val < mu_prior_val) else "divergent")

    if use_openai and openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            prompt = f"""You are a portfolio analyst for Indian retail investors.
Write a 3-sentence rationale for this allocation. Be specific and cite numbers.
Mention: the BL posterior return, the sentiment signal, and the weight rationale.

Stock  : {name} ({ticker}) | Sector: {meta.get('sector','')}
Weight : {weight:.1%} ({stance}) | Target INR: ₹{target_inr:,.0f} | Approx. shares: {shares}
Sentiment: {label} ({score:+.3f}) | {n_news} headlines | {pct_pos:.0%} positive
BL Posterior Return: {mu_bl_val:.2%} | Market Prior: {mu_prior_val:.2%}
Price: ₹{price_inr:,.0f} per share

Write for an Indian retail investor. Do not use bullet points."""
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3, max_tokens=180,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            pass

    # Template rationale
    view_shift = mu_bl_val - mu_prior_val
    shift_dir  = "upward" if view_shift > 0 else "downward"

    return (
        f"**{name}** is assigned a **{stance} position of {weight:.1%}** "
        f"(≈{shares:.4g} shares at ₹{price_inr:,.0f}/share = ₹{target_inr:,.0f}). "
        f"FinBERT scored {n_news} recent headlines at **{score:+.3f}** ({label}), "
        f"with {pct_pos:.0%} of articles carrying a positive signal — "
        f"shifting the Black-Litterman posterior return {shift_dir} "
        f"from the market prior of **{mu_prior_val:.2%}** to **{mu_bl_val:.2%}**. "
        f"Sentiment and price signals are **{alignment}**, "
        f"{'supporting the overweight thesis.' if stance == 'overweight' else 'suggesting caution at current valuations.' if stance == 'underweight' else 'warranting a neutral allocation.'}"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  LEVEL 3: MACRO REGIME PANEL
# ══════════════════════════════════════════════════════════════════════════════

def chart_sector_heatmap(sector_sentiment: dict) -> go.Figure:
    """Single-row heatmap showing sector-level FinBERT sentiment."""
    sectors = list(sector_sentiment.keys())
    scores  = [sector_sentiment[s] for s in sectors]
    colors  = [SECTOR_COLORS.get(s, "#94a3b8") for s in sectors]

    fig = go.Figure(go.Bar(
        x    = sectors,
        y    = scores,
        marker_color = [
            "#16a34a" if v > 0.15 else
            "#86efac" if v > 0.03 else
            "#fca5a5" if v < -0.03 else
            "#dc2626" if v < -0.15 else "#94a3b8"
            for v in scores
        ],
        text = [f"{v:+.3f}" for v in scores],
        textposition = "outside",
        hovertemplate = "<b>%{x}</b><br>Avg Sentiment: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#1e293b", line_width=1)
    fig.update_layout(
        yaxis_title  = "Avg FinBERT Score",
        plot_bgcolor = "white",
        height       = 260,
        margin       = dict(l=40, r=40, t=20, b=40),
        yaxis        = dict(range=[-0.7, 0.7], showgrid=True, gridcolor="#e2e8f0"),
    )
    return fig


def render_macro_panel(snap: dict, sector_sentiment: dict = None, key_prefix: str = "macro"):
    """
    Level 3 Market Regime & Macro Risk panel.
    Shows regime badge, VIX gauge, combined scale, narrative, and sector heatmap.
    """
    r = snap["regime"]
    v = snap["vix"]

    regime_bg = {"bull": "#f0fdf4", "neutral": "#fefce8", "bear": "#fef2f2"}.get(
        r["regime"], "#f8fafc"
    )
    regime_border = {"bull": "#16a34a", "neutral": "#ca8a04", "bear": "#dc2626"}.get(
        r["regime"], "#94a3b8"
    )

    with st.expander(
        f"{r['emoji']} **Market Regime: {r['label']}** | "
        f"VIX {v['vix']:.1f} {v['emoji']} | "
        f"Equity Deployment: {snap['combined_scale']:.0%} | "
        f"Cash Buffer: {snap['cash_buffer']:.0%}",
        expanded=True,
    ):
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Market Regime",    f"{r['emoji']} {r['label']}")
        m2.metric("Regime Scale",     f"{r['equity_scale']:.0%}",
                  help="Equity scale from Nifty 50 moving average regime")
        m3.metric("VIX",              f"{v['vix']:.1f}",
                  delta=f"{v['emoji']} {v['fear_level']} fear",
                  delta_color="off")
        m4.metric("VIX Scale",        f"{v['equity_scale']:.0%}",
                  help="Equity scale from VIX fear index")
        m5.metric("Cash Buffer",      f"{snap['cash_buffer']:.0%}",
                  delta=f"{snap['combined_scale']:.0%} deployed",
                  delta_color="off",
                  help="Fraction held as cash/short bonds due to macro risk-off")

        st.markdown(
            f'<div style="border-left:4px solid {regime_border}; background:{regime_bg}; '
            f'padding:0.6rem 1rem; border-radius:4px; margin:0.5rem 0;">'
            f'<b>Macro Narrative:</b> {snap["narrative"]}</div>',
            unsafe_allow_html=True,
        )

        if r.get("signals"):
            sig = r["signals"]
            cols = st.columns(4)
            cols[0].caption(f"Nifty 50: **₹{sig.get('nifty50',0):,.0f}**")
            cols[1].caption(f"MA50: **₹{sig.get('ma50',0):,.0f}** | MA200: **₹{sig.get('ma200',0):,.0f}**")
            cols[2].caption(f"20-day momentum: **{sig.get('momentum_20d',0):.1%}**")
            cols[3].caption(f"52-week drawdown: **{sig.get('drawdown_52w',0):.1%}**")

        if sector_sentiment:
            st.markdown("**Sector Sentiment Heatmap**")
            st.plotly_chart(chart_sector_heatmap(sector_sentiment), use_container_width=True, key=f"{key_prefix}_panel_sector_heatmap")


# ══════════════════════════════════════════════════════════════════════════════
#  SETUP GATE — check data availability before rendering
# ══════════════════════════════════════════════════════════════════════════════

def render_setup_gate(data_status):
    st.error("⚠️  Some required data files are missing. Please run the setup pipeline first.")
    st.markdown("### Setup Pipeline")
    st.code(
        "# Step 1: Download market data + FX rate + market caps\n"
        "python data_collector.py\n\n"
        "# Step 2: Fetch news and score sentiment with FinBERT\n"
        "python sentiment_engine.py\n\n"
        "# Step 3 (optional but recommended): Run historical backtest\n"
        "python optimizer.py --backtest",
        language="bash",
    )
    st.markdown("### Data File Status")
    cols = st.columns(3)
    items = list(data_status["required"].items())
    for i, (fname, exists) in enumerate(items):
        cols[i % 3].metric(
            label=fname,
            value="✅ Ready" if exists else "❌ Missing",
        )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Page config ───────────────────────────────────────────────────────────
    st.set_page_config(
        page_title = "PortfolioAI — Smart US Stock Investing in ₹",
        page_icon  = "📈",
        layout     = "wide",
        initial_sidebar_state = "expanded",
    )

    # ── Global CSS ────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    .metric-card {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 12px; padding: 1rem 1.2rem;
    }
    .action-buy  { color: #16a34a; font-weight: 700; }
    .action-sell { color: #dc2626; font-weight: 700; }
    .action-hold { color: #64748b; font-weight: 500; }
    .badge {
        display: inline-block; padding: 2px 10px;
        border-radius: 999px; font-size: 0.75rem; font-weight: 600;
    }
    .disclaimer {
        background: #fefce8; border: 1px solid #fde047;
        border-radius: 8px; padding: 0.8rem 1rem;
        font-size: 0.8rem; color: #713f12; margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Session state initialisation ──────────────────────────────────────────
    for key, default in [
        ("mode",        "fresh"),
        ("result",      None),
        ("run_count",   0),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 📈 PortfolioAI")
        st.caption("AI-driven NSE portfolio optimisation powered by FinBERT + Black-Litterman")
        st.divider()

        mode = st.radio(
            "**Choose your investment mode**",
            options=["💰 Fresh Investment", "🔄 Portfolio Rebalancer"],
            index=0 if st.session_state.mode == "fresh" else 1,
        )
        st.session_state.mode = "fresh" if "Fresh" in mode else "rebalance"

        st.divider()

        risk_profile = st.select_slider(
            "**Risk Profile**",
            options=["conservative", "moderate", "aggressive"],
            value="moderate",
            help="Conservative: min volatility | Moderate & Aggressive: max Sharpe ratio",
        )

        risk_desc = {
            "conservative": "📉 Min CVaR — minimises worst-case tail losses (5% VaR)",
            "moderate":     "⚖️  Max Sharpe — balances return and risk optimally",
            "aggressive":   "📈 Max Sharpe — higher concentration, more upside/downside",
        }
        st.caption(risk_desc[risk_profile])

        st.divider()

        # ── Mode-specific inputs ──────────────────────────────────────────────
        if st.session_state.mode == "fresh":
            st.markdown("**💰 Investment Amount**")
            inv_preset = st.selectbox(
                "Quick select",
                ["₹25,000", "₹50,000", "₹1,00,000", "₹2,50,000", "₹5,00,000", "Custom"],
                index=2,
            )
            preset_map = {
                "₹25,000": 25_000, "₹50,000": 50_000,
                "₹1,00,000": 100_000, "₹2,50,000": 250_000,
                "₹5,00,000": 500_000,
            }
            if inv_preset == "Custom":
                investment_inr = st.number_input(
                    "Enter amount (₹)", min_value=5000, max_value=50_000_000,
                    value=100_000, step=5000,
                )
            else:
                investment_inr = preset_map[inv_preset]
            st.metric("Investment Amount", fmt_inr(investment_inr))

        else:  # rebalance
            st.markdown("**🔄 Enter Current Holdings**")
            st.caption("Enter your current portfolio values in ₹")
            additional_inr = st.number_input(
                "Additional investment (₹)", min_value=0,
                value=0, step=5000,
                help="Extra money to add alongside rebalancing",
            )

        st.divider()

        # ── Advanced / optional ───────────────────────────────────────────────
        with st.expander("🔑 GPT-3.5 Rationale (optional)"):
            use_openai = st.toggle("Use GPT-3.5 for explanations", value=False)
            openai_key = ""
            if use_openai:
                openai_key = st.text_input("OpenAI API Key", type="password")
                st.caption("Used only for generating natural language rationale.")

        st.divider()

        # Data file status
        data_status = check_data_files()
        with st.expander("📁 Data Files"):
            for fname, exists in {**data_status["required"], **data_status["optional"]}.items():
                st.markdown(f"{'✅' if exists else '❌'} `{fname}`")
            if not data_status["required_ok"]:
                st.error("Run setup pipeline — see main page")

    # ── MAIN AREA ─────────────────────────────────────────────────────────────
    sentiment_df        = load_sentiment_df()
    backtest_df, metrics_df = load_backtest()
    prices_inr          = load_prices_inr()

    # ── Header ───────────────────────────────────────────────────────────────
    col_h1, col_h2, col_h3 = st.columns([4, 1, 1])
    col_h1.markdown("## 📈 PortfolioAI — NSE India Portfolio Optimiser")
    col_h2.metric("Universe", "16 NSE Stocks")
    col_h3.metric("Strategy", "BL + FinBERT")

    if not data_status["required_ok"]:
        render_setup_gate(data_status)
        return

    # ── Level 3: Macro Regime Panel ───────────────────────────────────────────
    macro_snap = load_macro_snapshot()
    if macro_snap:
        render_macro_panel(macro_snap, key_prefix="top")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    #  FRESH INVESTMENT FLOW
    # ══════════════════════════════════════════════════════════════════════════
    if st.session_state.mode == "fresh":
        st.markdown(f"### 💰 Fresh Investment — {fmt_inr(investment_inr)}")

        if st.button("🚀 Optimise My Portfolio", type="primary", use_container_width=True):
            with st.spinner("Running Black-Litterman optimisation with FinBERT sentiment…"):
                try:
                    result = run_fresh_optimizer(investment_inr, risk_profile)
                    st.session_state.result    = result
                    st.session_state.run_count += 1
                except Exception as e:
                    st.error(f"Optimisation failed: {e}")
                    st.stop()

        result = st.session_state.result

        if result:
            alloc        = result["allocation"]
            summary      = result["summary"]
            feats        = result["features"]
            mu_bl        = feats["mu_bl"]
            mu_prior     = feats["mu_prior"]
            s_df         = feats["sentiment_df"]
            sec_sent     = feats.get("sector_sentiment", {})
            result_macro = result.get("macro_snapshot")

            # ── Macro overlay summary (if weights were scaled) ─────────────────
            macro_scale = summary.get("macro_scale", 1.0)
            macro_cash  = summary.get("macro_cash_buffer", 0.0)
            if result_macro and macro_cash > 0.02:
                r_reg = result_macro["regime"]
                r_vix = result_macro["vix"]
                st.info(
                    f"{r_reg['emoji']} **Macro Overlay Applied** — "
                    f"{r_reg['label']} regime × VIX {r_vix['vix']:.1f} ({r_vix['fear_level']}) "
                    f"→ equity exposure scaled to **{macro_scale:.0%}**, "
                    f"holding **{fmt_inr(investment_inr * macro_cash, compact=True)} cash buffer** "
                    f"({macro_cash:.0%} of capital)."
                )

            # ── KPI row ───────────────────────────────────────────────────────
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Total Deployed",  fmt_inr(summary["total_deployed_inr"]))
            k2.metric("Cash Leftover",   fmt_inr(summary["cash_leftover_inr"]))
            k3.metric("Expected Return", f"{summary['expected_return']:.2%}")
            k4.metric("Volatility",      f"{summary['expected_volatility']:.2%}")
            k5.metric("Sharpe Ratio",    f"{summary['sharpe_ratio']:.2f}")

            st.divider()

            # ── Tabs ──────────────────────────────────────────────────────────
            t1, t2, t3, t4, t5, t6 = st.tabs([
                "📊 Allocation", "🧠 Sentiment", "📉 Backtest",
                "💬 Rationale", "🌍 Macro & Sectors", "🔢 Details",
            ])

            with t1:
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.plotly_chart(chart_allocation_donut(alloc, s_df), use_container_width=True, key="fresh_alloc_donut")
                with c2:
                    st.plotly_chart(chart_allocation_bar(alloc, s_df), use_container_width=True, key="fresh_alloc_bar")

                # Sector legend
                sectors = {}
                for _, row in alloc.iterrows():
                    sec = STOCK_META.get(row["ticker"], {}).get("sector", "Other")
                    sectors[sec] = sectors.get(sec, 0) + row["invested_inr"]
                st.markdown("**Sector Breakdown:**  " + "  |  ".join(
                    f'<span style="color:{SECTOR_COLORS.get(s,"#94a3b8")}">■</span> '
                    f'**{s}** {v/summary["total_deployed_inr"]:.0%}'
                    for s, v in sorted(sectors.items(), key=lambda x: -x[1])
                ), unsafe_allow_html=True)

                st.divider()

                # Allocation table
                st.markdown("#### Recommended Allocation")
                display = []
                for _, row in alloc.iterrows():
                    t   = row["ticker"]
                    sent = s_df.loc[t, "label"] if t in s_df.index else "n/a"
                    cfg  = SENTIMENT_CONFIG.get(sent, {})
                    display.append({
                        "Stock":      f"{cfg.get('badge','⚪')} {t}",
                        "Company":    STOCK_META.get(t, {}).get("name", t),
                        "Sector":     STOCK_META.get(t, {}).get("sector", ""),
                        "Weight":     f"{row['target_weight']:.2%}",
                        "Amount":     fmt_inr(row["invested_inr"]),
                        "Shares":     f"{row['shares']:.4g}",
                        "Price/Share":fmt_inr(row["price_inr"]),
                        "Sentiment":  f"{cfg.get('icon','─')} {sent}",
                    })
                st.dataframe(pd.DataFrame(display), use_container_width=True, hide_index=True)

                # Buy instructions
                st.info(
                    f"💡 **How to invest:** Place buy orders for {fmt_inr(summary['total_deployed_inr'])} "
                    f"through your broker (Zerodha, Groww, Angel One, HDFC Securities, etc.) "
                    f"on the NSE. All prices are in INR — no currency conversion required. "
                    f"Cash leftover after rounding: {fmt_inr(summary['cash_leftover_inr'])}."
                )

            with t2:
                st.markdown("#### FinBERT Sentiment Scores — Today's Signal")
                if sentiment_df is not None:
                    st.plotly_chart(chart_sentiment_scores(s_df), use_container_width=True, key="fresh_sentiment_chart")

                    sent_table = []
                    for ticker in s_df.index:
                        row  = s_df.loc[ticker]
                        cfg  = SENTIMENT_CONFIG.get(row["label"], {})
                        sent_table.append({
                            "Stock":      f"{cfg.get('badge','⚪')} {ticker}",
                            "Company":    STOCK_META.get(ticker, {}).get("name", ticker),
                            "Score":      f"{row['final_score']:+.3f}",
                            "Signal":     f"{cfg.get('icon','─')} {row['label']}",
                            "Confidence": f"{row['confidence']:.0%}",
                            "Headlines":  int(row["num_headlines"]),
                            "% Positive": f"{row['pct_positive']:.0%}",
                            "% Negative": f"{row['pct_negative']:.0%}",
                            "BL View (Q)":f"{mu_bl[ticker]:+.2%}",
                            "Market Prior":f"{mu_prior[ticker]:+.2%}",
                        })
                    st.dataframe(pd.DataFrame(sent_table), use_container_width=True, hide_index=True)

                    st.markdown("""
                    **How sentiment drives allocation:**
                    - **Bullish (>+0.30):** Minimum 8% floor weight + BL view boosted above market prior
                    - **Bearish (<-0.30):** Maximum 5% cap weight + BL view pulled below market prior
                    - **Neutral:** Standard [0%, 30%] bounds, BL view stays near market prior
                    """)

            with t3:
                if backtest_df is not None and metrics_df is not None:
                    st.plotly_chart(chart_cumulative_return(backtest_df), use_container_width=True, key="fresh_backtest_chart")

                    st.warning(
                        "⚠️ **Look-Ahead Bias Notice** — The historical backtest applies "
                        "**today's** FinBERT sentiment scores to all past rebalancing dates. "
                        "This means the backtest sentiment signal is not available in the real world "
                        "for periods before today. The Sentiment Alpha figure is therefore "
                        "**overstated** — treat it as an upper-bound estimate. "
                        "To fully eliminate this bias, a historical news archive API "
                        "(EODHD, Benzinga, Tiingo) is required to re-scrape news for each "
                        "past rebalancing date. The Pure Quant Baseline is unaffected."
                    )

                    st.markdown("#### Performance Metrics")
                    m_bl  = metrics_df[metrics_df["label"] == "BL + Sentiment"]
                    m_base= metrics_df[metrics_df["label"] == "Pure Quant (Baseline)"]
                    m_sp  = metrics_df[metrics_df["label"] == "Nifty 50"]

                    if not m_bl.empty:
                        r = m_bl.iloc[0]
                        mc1, mc2, mc3, mc4 = st.columns(4)
                        mc1.metric("Cumul. Return (BL)",    f"{r['cumulative_ret']:.1%}")
                        mc2.metric("Sharpe Ratio (BL)",     f"{r['sharpe']:.2f}")
                        mc3.metric("Max Drawdown (BL)",     f"{r['max_drawdown']:.1%}")
                        mc4.metric("Sentiment Alpha",        f"{r.get('sentiment_alpha',0):.2%}",
                                   help="Incremental return vs pure-quant baseline")

                    # Full table
                    fmt_cols = {"cumulative_ret": ".1%", "ann_return": ".2%",
                                "ann_vol": ".2%", "sharpe": ".2f",
                                "max_drawdown": ".2%", "calmar": ".2f",
                                "sentiment_alpha": ".2%"}
                    disp = metrics_df.copy()
                    for col, fmt in fmt_cols.items():
                        if col in disp.columns:
                            disp[col] = disp[col].apply(
                                lambda x: format(x, fmt) if pd.notna(x) else "n/a"
                            )
                    st.dataframe(disp, use_container_width=True, hide_index=True)
                else:
                    st.info("Backtest data not found. Run: `python optimizer.py --backtest`")

            with t4:
                st.markdown("#### AI-Generated Investment Rationale")
                st.caption("Each allocation is explained using FinBERT sentiment, "
                            "Black-Litterman posterior returns, and portfolio weight logic.")
                for _, row in alloc.iterrows():
                    t = row["ticker"]
                    if t not in s_df.index:
                        continue
                    sent_row = s_df.loc[t]
                    sent     = sent_row["label"]
                    cfg      = SENTIMENT_CONFIG.get(sent, {})
                    weight   = row["target_weight"]
                    rationale= generate_rationale(
                        t, weight, row["invested_inr"],
                        sent_row, mu_bl[t], mu_prior[t], prices_inr,
                        use_openai, openai_key,
                    )
                    with st.expander(
                        f"{cfg.get('badge','⚪')} **{t}** — "
                        f"{STOCK_META.get(t,{}).get('name',t)}  |  "
                        f"{weight:.1%}  |  {fmt_inr(row['invested_inr'])}  |  {sent}",
                        expanded=False,
                    ):
                        st.markdown(
                            f'<div style="border-left:4px solid {cfg.get("color","#94a3b8")}; '
                            f'padding:0.6rem 1rem; background:#f8fafc; border-radius:4px;">'
                            f'{rationale}</div>',
                            unsafe_allow_html=True,
                        )
                        rc1, rc2, rc3, rc4 = st.columns(4)
                        rc1.metric("Sentiment Score",   f"{sent_row['final_score']:+.3f}")
                        rc2.metric("Market Prior",      f"{mu_prior[t]:.2%}")
                        rc3.metric("BL Posterior",      f"{mu_bl[t]:.2%}")
                        rc4.metric("Weight",            f"{weight:.2%}")

            with t5:
                st.markdown("#### Market Regime & Macro Risk")
                if result_macro:
                    render_macro_panel(result_macro, sec_sent, key_prefix="fresh")
                else:
                    st.info("Macro snapshot unavailable — ensure prices.csv exists.")

                if sec_sent:
                    st.markdown("#### Sector Sentiment (FinBERT Aggregated)")
                    st.plotly_chart(chart_sector_heatmap(sec_sent), use_container_width=True, key="fresh_sector_heatmap")
                    sec_rows = [
                        {"Sector": s, "Avg Sentiment": f"{v:+.3f}",
                         "Signal": "🟢 Bullish" if v > 0.15 else
                                   "🟡 Slightly Bullish" if v > 0.03 else
                                   "🔴 Bearish" if v < -0.15 else
                                   "🟠 Slightly Bearish" if v < -0.03 else "⚪ Neutral"}
                        for s, v in sec_sent.items()
                    ]
                    st.dataframe(pd.DataFrame(sec_rows), use_container_width=True, hide_index=True)

                analyst_cons = feats.get("analyst_consensus")
                if analyst_cons is not None and not analyst_cons.empty:
                    st.markdown("#### Analyst Consensus (Wall Street)")
                    ac_rows = []
                    for ticker in analyst_cons.index:
                        score = float(analyst_cons[ticker])
                        ac_rows.append({
                            "Stock":     ticker,
                            "Company":   STOCK_META.get(ticker, {}).get("name", ticker),
                            "Score":     f"{score:+.2f}",
                            "Signal":    "🟢 Strong Buy" if score > 0.6 else
                                         "🟡 Buy" if score > 0.2 else
                                         "🔴 Sell" if score < -0.2 else
                                         "🟠 Strong Sell" if score < -0.6 else "⚪ Hold",
                        })
                    st.dataframe(pd.DataFrame(ac_rows), use_container_width=True, hide_index=True)
                    st.caption("Analyst consensus score: +1.0 = all Strong Buy, -1.0 = all Strong Sell, 0 = Hold")

            with t6:
                st.markdown("#### Full Allocation Details")
                detail = alloc.copy()
                detail["target_weight"] = detail["target_weight"].map("{:.2%}".format)
                detail["target_inr"]    = detail["target_inr"].map("₹{:,.0f}".format)
                detail["price_inr"]     = detail["price_inr"].map("₹{:,.2f}".format)
                detail["invested_inr"]  = detail["invested_inr"].map("₹{:,.0f}".format)
                st.dataframe(detail, use_container_width=True, hide_index=True)

                st.markdown("#### Summary")
                for k, v in summary.items():
                    if isinstance(v, float) and abs(v) < 10:
                        st.write(f"**{k}:** {v:.4f}")
                    elif isinstance(v, float):
                        st.write(f"**{k}:** {fmt_inr(v)}")
                    else:
                        st.write(f"**{k}:** {v}")

        else:
            # Landing state — show stock preview
            st.info("👈 Set your investment amount in the sidebar, then click **Optimise My Portfolio**.")
            if prices_inr is not None and sentiment_df is not None:
                st.markdown("#### Current Prices & Sentiment at a Glance")
                preview = []
                for t in STOCKS:
                    if t not in sentiment_df.index:
                        continue
                    s    = sentiment_df.loc[t, "final_score"]
                    lbl  = sentiment_df.loc[t, "label"]
                    cfg  = SENTIMENT_CONFIG.get(lbl, {})
                    p    = float(prices_inr.get(t, 0))
                    preview.append({
                        "Stock":     f"{cfg.get('badge','⚪')} {t}",
                        "Company":   STOCK_META.get(t, {}).get("name", t),
                        "Sector":    STOCK_META.get(t, {}).get("sector", ""),
                        "Price":     fmt_inr(p),
                        "Sentiment": f"{cfg.get('icon','─')} {lbl}",
                        "Score":     f"{s:+.3f}",
                    })
                st.dataframe(pd.DataFrame(preview), use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  PORTFOLIO REBALANCER FLOW
    # ══════════════════════════════════════════════════════════════════════════
    else:
        st.markdown("### 🔄 Portfolio Rebalancer")
        st.caption("Enter what you currently hold — get a clear BUY / SELL / HOLD plan.")

        # ── Holdings input table ──────────────────────────────────────────────
        st.markdown("#### Step 1 — Enter Your Current Holdings")
        st.caption("Enter the current market value (in ₹) of each stock you hold. Leave 0 if you don't hold it.")

        # Build default table
        if prices_inr is not None:
            default_prices = {t: float(prices_inr.get(t, 0)) for t in STOCKS}
        else:
            default_prices = {t: 0.0 for t in STOCKS}

        template_df = pd.DataFrame([
            {
                "Ticker":        t,
                "Company":       STOCK_META.get(t, {}).get("name", t),
                "Sector":        STOCK_META.get(t, {}).get("sector", ""),
                "Current Price (₹)": default_prices.get(t, 0),
                "Your Holdings (₹)": 0.0,
            }
            for t in STOCKS
        ])

        edited = st.data_editor(
            template_df,
            use_container_width=True,
            disabled=["Ticker", "Company", "Sector", "Current Price (₹)"],
            column_config={
                "Your Holdings (₹)": st.column_config.NumberColumn(
                    "Your Holdings (₹)",
                    help="Enter the current ₹ value of your position in this stock",
                    min_value=0, max_value=50_000_000, step=500,
                    format="₹%d",
                ),
                "Current Price (₹)": st.column_config.NumberColumn(
                    "Current Price (₹)", format="₹%.2f", disabled=True,
                ),
            },
            num_rows="fixed",
            hide_index=True,
        )

        current_holdings = {
            row["Ticker"]: float(row["Your Holdings (₹)"])
            for _, row in edited.iterrows()
            if float(row["Your Holdings (₹)"]) > 0
        }
        total_current = sum(current_holdings.values())
        total_capital = total_current + additional_inr

        c_a, c_b, c_c = st.columns(3)
        c_a.metric("Current Portfolio",    fmt_inr(total_current))
        c_b.metric("Adding Now",           fmt_inr(additional_inr))
        c_c.metric("Total Capital",        fmt_inr(total_capital))

        st.markdown("---")

        if st.button("🔄 Generate Rebalancing Plan", type="primary",
                     use_container_width=True,
                     disabled=(total_capital < 1000)):
            if not current_holdings and additional_inr < 1000:
                st.warning("Please enter your current holdings or an additional investment amount.")
            else:
                with st.spinner("Analysing your portfolio and running BL optimisation…"):
                    try:
                        result = run_rebalance_optimizer(
                            current_holdings, additional_inr, risk_profile
                        )
                        st.session_state.result    = result
                        st.session_state.run_count += 1
                    except Exception as e:
                        st.error(f"Optimisation failed: {e}")
                        st.stop()

        result = st.session_state.result if (
            st.session_state.result and "rebalance" in st.session_state.result
        ) else None

        if result:
            rebal        = result["rebalance"]
            alloc        = result["allocation"]
            summary      = result["summary"]
            feats        = result["features"]
            mu_bl        = feats["mu_bl"]
            mu_prior     = feats["mu_prior"]
            s_df         = feats["sentiment_df"]
            sec_sent_rb  = feats.get("sector_sentiment", {})
            result_macro = result.get("macro_snapshot")

            macro_scale = summary.get("macro_scale", 1.0)
            macro_cash  = summary.get("macro_cash_buffer", 0.0)

            st.divider()

            # ── Macro overlay info ────────────────────────────────────────────
            if result_macro and macro_cash > 0.02:
                r_reg = result_macro["regime"]
                r_vix = result_macro["vix"]
                st.info(
                    f"{r_reg['emoji']} **Macro Overlay Applied** — "
                    f"{r_reg['label']} regime × VIX {r_vix['vix']:.1f} ({r_vix['fear_level']}) "
                    f"→ equity exposure scaled to **{macro_scale:.0%}**, "
                    f"holding **{fmt_inr(total_capital * macro_cash, compact=True)} cash buffer** "
                    f"({macro_cash:.0%} of capital)."
                )

            # ── Action KPIs ───────────────────────────────────────────────────
            buys  = rebal[rebal["action"] == "BUY"]
            sells = rebal[rebal["action"] == "SELL"]
            holds = rebal[rebal["action"] == "HOLD"]

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Stocks to BUY",  len(buys),  delta=f"₹{summary['buys_inr']:,.0f}")
            k2.metric("Stocks to SELL", len(sells), delta=f"-₹{summary['sells_inr']:,.0f}",
                      delta_color="inverse")
            k3.metric("Stocks to HOLD", len(holds))
            k4.metric("Est. Brokerage", fmt_inr(summary["transaction_cost_inr"]))
            k5.metric("Post-Rebal Sharpe", f"{summary['sharpe_ratio']:.2f}")

            rb1, rb2, rb3, rb4, rb5 = st.tabs([
                "📋 Rebalancing Plan", "📊 Portfolio View",
                "🧠 Sentiment", "💬 Rationale", "🌍 Macro & Sectors",
            ])

            with rb1:
                # Main rebalancing chart
                rb_chart = chart_rebalance(rebal)
                if rb_chart:
                    st.plotly_chart(rb_chart, use_container_width=True, key="rebal_trades_chart")

                st.markdown("#### Detailed Action Plan")

                # Colour-coded action table
                action_rows = []
                for _, row in rebal.sort_values(
                    ["action", "trade_inr"], ascending=[True, False]
                ).iterrows():
                    t   = row["ticker"]
                    act = row["action"]
                    cfg = SENTIMENT_CONFIG.get(row["sentiment"], {})
                    action_rows.append({
                        "Action":        act,
                        "Stock":         f"{cfg.get('badge','⚪')} {t}",
                        "Company":       STOCK_META.get(t, {}).get("name", t),
                        "Current (₹)":   fmt_inr(row["current_inr"]),
                        "Target (₹)":    fmt_inr(row["target_inr"]),
                        "Trade (₹)":     fmt_inr(row["trade_inr"]) if act != "HOLD" else "─",
                        "Shares Δ":      f"{row['shares_delta']:+.4g}" if act != "HOLD" else "─",
                        "Price/Share":   fmt_inr(row["price_inr"]),
                        "Sentiment":     f"{cfg.get('icon','─')} {row['sentiment']}",
                        "Target Weight": f"{row['target_weight']:.1%}",
                    })

                action_df = pd.DataFrame(action_rows)

                # Highlight rows by action
                def _style(row):
                    colors = {"BUY": "background-color:#f0fdf4",
                              "SELL": "background-color:#fef2f2",
                              "HOLD": ""}
                    return [colors.get(row["Action"], "")] * len(row)

                st.dataframe(
                    action_df.style.apply(_style, axis=1),
                    use_container_width=True, hide_index=True,
                )

                # Cash flow summary
                net_cash = summary["sells_inr"] - summary["transaction_cost_inr"] / 2
                st.markdown(f"""
                #### Cash Flow Summary
                | | Amount |
                |---|---|
                | Cash from SELL orders | {fmt_inr(summary['sells_inr'])} |
                | Additional investment | {fmt_inr(summary['additional_inr'])} |
                | Total available for BUY | {fmt_inr(net_cash + summary['additional_inr'])} |
                | BUY orders total | {fmt_inr(summary['buys_inr'])} |
                | Estimated brokerage (0.1%) | {fmt_inr(summary['transaction_cost_inr'])} |
                """)

                if summary["buys_inr"] > (net_cash + summary["additional_inr"]) * 1.05:
                    st.warning(
                        "⚠️ BUY orders exceed available cash from sells + additional investment. "
                        "Consider increasing additional investment or reducing position sizes."
                    )
                else:
                    st.success("✅ Rebalancing is self-funding — sells cover the buys.")

            with rb2:
                st.plotly_chart(
                    chart_current_vs_target(rebal, total_capital),
                    use_container_width=True,
                    key="rebal_current_vs_target",
                )

                # Pie: current vs target
                cc1, cc2 = st.columns(2)
                current_for_pie = rebal[rebal["current_inr"] > 0].copy()
                current_for_pie["label"] = current_for_pie["ticker"]
                if not current_for_pie.empty:
                    with cc1:
                        fig_curr = go.Figure(go.Pie(
                            labels=current_for_pie["ticker"],
                            values=current_for_pie["current_inr"],
                            hole=0.5, title="Current"
                        ))
                        fig_curr.update_layout(height=320, margin=dict(l=0,r=0,t=40,b=0))
                        st.plotly_chart(fig_curr, use_container_width=True, key="rebal_pie_current")

                with cc2:
                    target_for_pie = alloc[alloc["invested_inr"] > 0].copy()
                    fig_tgt = go.Figure(go.Pie(
                        labels=target_for_pie["ticker"],
                        values=target_for_pie["invested_inr"],
                        hole=0.5, title="Target"
                    ))
                    fig_tgt.update_layout(height=320, margin=dict(l=0,r=0,t=40,b=0))
                    st.plotly_chart(fig_tgt, use_container_width=True, key="rebal_pie_target")

            with rb3:
                if sentiment_df is not None:
                    st.plotly_chart(chart_sentiment_scores(s_df), use_container_width=True, key="rebal_sentiment_chart")
                    st.markdown("*Sentiment directly influences BUY/SELL decisions: "
                                "bearish stocks are capped at 5% weight, bullish stocks "
                                "have a minimum 8% floor.*")

            with rb4:
                st.markdown("#### AI Rationale for Each Target Position")
                for _, row in alloc.iterrows():
                    t = row["ticker"]
                    if t not in s_df.index:
                        continue
                    sent_row = s_df.loc[t]
                    sent     = sent_row["label"]
                    cfg      = SENTIMENT_CONFIG.get(sent, {})
                    weight   = row["target_weight"]
                    reb_row  = rebal[rebal["ticker"] == t]
                    action   = reb_row["action"].values[0] if not reb_row.empty else "HOLD"
                    act_icon = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(action, "⚪")
                    rationale= generate_rationale(
                        t, weight, row["invested_inr"], sent_row,
                        mu_bl[t], mu_prior[t], prices_inr, use_openai, openai_key,
                    )
                    with st.expander(
                        f"{act_icon} **{action}** — {cfg.get('badge','⚪')} **{t}** "
                        f"— {STOCK_META.get(t,{}).get('name',t)}  |  "
                        f"Target: {weight:.1%}  |  {sent}",
                        expanded=False,
                    ):
                        st.markdown(
                            f'<div style="border-left:4px solid {cfg.get("color","#94a3b8")}; '
                            f'padding:0.6rem 1rem; background:#f8fafc; border-radius:4px;">'
                            f'{rationale}</div>',
                            unsafe_allow_html=True,
                        )
                        rc1, rc2, rc3, rc4 = st.columns(4)
                        rc1.metric("Action",         action)
                        rc2.metric("Sentiment",      f"{sent_row['final_score']:+.3f}")
                        rc3.metric("BL Return",      f"{mu_bl[t]:.2%}")
                        rc4.metric("Target Weight",  f"{weight:.2%}")

            with rb5:
                st.markdown("#### Market Regime & Macro Risk")
                if result_macro:
                    render_macro_panel(result_macro, sec_sent_rb, key_prefix="rebal")
                else:
                    st.info("Macro snapshot unavailable — ensure prices.csv exists.")

                if sec_sent_rb:
                    st.markdown("#### Sector Sentiment (FinBERT Aggregated)")
                    st.plotly_chart(chart_sector_heatmap(sec_sent_rb), use_container_width=True, key="rebal_sector_heatmap")

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("""
    <div class="disclaimer">
    ⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only and does not constitute
    financial advice. NSE equity investments are subject to market risk. Brokerage fees, STT
    (Securities Transaction Tax), exchange charges, and capital gains tax (LTCG 10% above ₹1L /
    STCG 15%) apply. Past backtest performance does not guarantee future results. Please consult
    a SEBI-registered investment advisor (RIA) before making investment decisions.
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        "PortfolioAI | Black-Litterman + FinBERT Sentiment | "
        "16 NSE Equities | Macro Regime Overlay | Built for BTech Minor Project 2024–25"
    )


if __name__ == "__main__":
    main()
