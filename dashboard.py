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
from financial_planner import (
    FinancialProfile, FinancialAnalyzer, RiskProfiler,
    AssetAllocator, FinancialPlanGenerator, RISK_QUESTIONS,
)

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
        "fundamentals.csv":     f"{DATA_DIR}/fundamentals.csv",
        "factor_scores.csv":    f"{DATA_DIR}/factor_scores.csv",
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
def load_factor_scores_df():
    p = f"{DATA_DIR}/factor_scores.csv"
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
def load_enhanced_backtest():
    """Load enhanced backtester results (from backtester.py)."""
    rp = f"{DATA_DIR}/backtest_enhanced_results.csv"
    mp = f"{DATA_DIR}/backtest_enhanced_metrics.csv"
    cp = f"{DATA_DIR}/backtest_costs.csv"
    results = pd.read_csv(rp, index_col="date", parse_dates=True) if os.path.exists(rp) else None
    metrics = pd.read_csv(mp) if os.path.exists(mp) else None
    costs   = pd.read_csv(cp, index_col="date", parse_dates=True) if os.path.exists(cp) else None
    return results, metrics, costs


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


def run_fresh_optimizer(investment_inr, risk_profile, analysis_method="llm"):
    """Run optimizer and cache result in session state."""
    from optimizer import optimize_fresh_investment
    return optimize_fresh_investment(investment_inr, risk_profile, analysis_method)


def run_rebalance_optimizer(current_holdings, additional_inr, risk_profile, analysis_method="llm"):
    """Run rebalancer and cache result in session state."""
    from optimizer import optimize_rebalancing
    return optimize_rebalancing(current_holdings, additional_inr, risk_profile, analysis_method)


def run_full_pipeline(analysis_method="llm"):
    """
    Refresh all data needed for optimisation, based on the chosen analysis method.
    Displays progress spinners for each step. Errors are shown as warnings — the
    app never crashes, it just falls back to cached data.
    """
    import data_collector as dc

    # Step 1 — Always refresh prices
    with st.spinner("📥 Step 1 — Downloading fresh market data…"):
        try:
            dc.download_prices()
            st.toast("✅ Market data ready!", icon="📈")
        except Exception as e:
            st.warning(f"Using cached market data: {e}")

    # Step 2 — LLM views (Groq/LLaMA) if needed
    if analysis_method in ("llm", "combined"):
        with st.spinner("🤖 Step 2 — LLaMA analysing stocks…"):
            try:
                import llm_views as lv
                lv.run_llm_view_pipeline(lookback_days=10)
                st.toast("✅ LLM views ready!", icon="🧠")
            except Exception as e:
                st.warning(f"LLM views failed: {e}")

    # Step 3 — FinBERT news sentiment if needed
    if analysis_method in ("sentiment", "combined"):
        with st.spinner("📰 Step 3 — Reading news sentiment…"):
            try:
                from sentiment_engine import run_sentiment_pipeline, STOCKS as SE_STOCKS
                run_sentiment_pipeline(SE_STOCKS)
                st.toast("✅ Sentiment ready!", icon="📰")
            except Exception as e:
                st.warning(f"Sentiment failed: {e} — using LLM views only")

    # Step 4 — Factor scoring
    with st.spinner("📊 Scoring stocks on momentum & quality…"):
        try:
            import scorer as sc
            sc.compute_factor_scores()
            st.toast("✅ Factor scores ready!", icon="📊")
        except Exception as e:
            st.warning(f"Factor scores skipped: {e}")


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
#  FACTOR SCORES CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def chart_combined_factor_scores(factor_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of combined factor scores, coloured by sector."""
    df = factor_df.reset_index().sort_values("combined_score", ascending=True)
    colors = [SECTOR_COLORS.get(s, "#94a3b8") for s in df["sector"]]
    selected_text = ["✅" if s else "" for s in df["selected"]]

    fig = go.Figure(go.Bar(
        x    = df["combined_score"],
        y    = df["ticker"],
        orientation  = "h",
        marker_color = colors,
        text = [f"{v:.3f} {m}" for v, m in zip(df["combined_score"], selected_text)],
        textposition = "outside",
        hovertemplate = (
            "<b>%{y}</b><br>"
            "Combined: %{x:.3f}<br>"
            "Momentum: %{customdata[0]:.3f}<br>"
            "Quality:  %{customdata[1]:.3f}<br>"
            "Volatility: %{customdata[2]:.3f}<br>"
            "6m Return: %{customdata[3]:.1%}<extra></extra>"
        ),
        customdata = df[["momentum_score","quality_score","volatility_score","raw_momentum_6m"]].values,
    ))
    fig.add_vline(x=df["combined_score"].quantile(0.0625),  # 1/16 = bottom cut
                  line_dash="dash", line_color="#94a3b8", line_width=1)
    fig.update_layout(
        title        = "Combined Factor Score (Momentum 40% + Quality 40% + Vol 20%)",
        xaxis_range  = [0, 1.15],
        xaxis_title  = "Combined Score (0→1)",
        plot_bgcolor = "white",
        height       = 480,
        margin       = dict(l=80, r=120, t=50, b=40),
        xaxis        = dict(showgrid=True, gridcolor="#e2e8f0"),
    )
    return fig


def chart_individual_factor(factor_df: pd.DataFrame, col: str, title: str,
                             color: str = "#3b82f6") -> go.Figure:
    """Compact horizontal bar for one factor."""
    df = factor_df.reset_index().sort_values(col, ascending=True)
    fig = go.Figure(go.Bar(
        x    = df[col],
        y    = df["ticker"],
        orientation  = "h",
        marker_color = color,
        text = [f"{v:.3f}" for v in df[col]],
        textposition = "outside",
        hovertemplate = "<b>%{y}</b><br>" + title + ": %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title        = title,
        xaxis_range  = [0, 1.2],
        plot_bgcolor = "white",
        height       = 380,
        margin       = dict(l=80, r=60, t=40, b=30),
        xaxis        = dict(showgrid=True, gridcolor="#e2e8f0"),
    )
    return fig


def render_factor_scores_tab(factor_df: pd.DataFrame, key_prefix: str = "f"):
    """Render the full 📊 Factor Scores tab content."""
    st.markdown("""
    <div style="background:#f0f9ff; border:1px solid #bae6fd; border-radius:8px;
                padding:0.7rem 1rem; margin-bottom:1rem; font-size:0.85rem; color:#0c4a6e;">
    <b>How factor scoring works:</b> Stocks are ranked by combined factor score.
    Top 15 enter the portfolio. LLM views then fine-tune the exact weights within this universe.
    </div>
    """, unsafe_allow_html=True)

    # ── KPIs ─────────────────────────────────────────────────────────────────
    selected = factor_df[factor_df["selected"] == True]
    top_sector = factor_df[factor_df["selected"]]["sector"].value_counts().idxmax() \
                 if not selected.empty else "N/A"

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Stocks Scored", len(factor_df))
    k2.metric("Selected for Portfolio", len(selected))
    k3.metric("Top Sector", top_sector)
    k4.metric("Avg Combined Score", f"{factor_df['combined_score'].mean():.3f}")

    st.divider()

    # ── Combined score bar chart ──────────────────────────────────────────────
    st.plotly_chart(chart_combined_factor_scores(factor_df),
                    use_container_width=True, key=f"{key_prefix}_combined_bar")

    # Sector legend
    st.markdown("**Sector colour legend:**  " + "  |  ".join(
        f'<span style="color:{SECTOR_COLORS.get(s,"#94a3b8")}">■</span> **{s}**'
        for s in sorted(SECTOR_COLORS)
    ), unsafe_allow_html=True)

    st.divider()

    # ── Three factor columns ──────────────────────────────────────────────────
    st.markdown("#### Individual Factor Scores")
    fc1, fc2, fc3 = st.columns(3)

    with fc1:
        st.plotly_chart(
            chart_individual_factor(factor_df, "momentum_score",
                                    "Momentum Score (6m return)", "#3b82f6"),
            use_container_width=True, key=f"{key_prefix}_mom_bar",
        )
        st.caption("6-month price return (skip last month). "
                   "Higher rank = stronger trend.")

    with fc2:
        st.plotly_chart(
            chart_individual_factor(factor_df, "quality_score",
                                    "Quality Score (ROE / D:E / EPS)", "#10b981"),
            use_container_width=True, key=f"{key_prefix}_qual_bar",
        )
        st.caption("Combines ROE, Debt/Equity, and EPS growth. "
                   "Higher = fundamentally stronger.")

    with fc3:
        st.plotly_chart(
            chart_individual_factor(factor_df, "volatility_score",
                                    "Low-Vol Score (inverted 60d vol)", "#f59e0b"),
            use_container_width=True, key=f"{key_prefix}_vol_bar",
        )
        st.caption("60-day realised volatility, inverted. "
                   "Higher = lower risk / smoother returns.")

    st.divider()

    # ── Detailed table ────────────────────────────────────────────────────────
    st.markdown("#### Full Factor Score Table")
    table_rows = []
    for ticker, row in factor_df.sort_values("combined_score", ascending=False).iterrows():
        meta = STOCK_META.get(ticker, {})
        table_rows.append({
            "Ticker":        ticker,
            "Company":       meta.get("name", ticker),
            "Sector":        row.get("sector", ""),
            "Momentum":      f"{row['momentum_score']:.3f}",
            "Quality":       f"{row['quality_score']:.3f}",
            "Volatility":    f"{row['volatility_score']:.3f}",
            "Combined":      f"{row['combined_score']:.3f}",
            "6m Return":     f"{row.get('raw_momentum_6m', 0):.1%}",
            "60d Ann.Vol":   f"{row.get('raw_vol_60d', 0):.1%}",
            "Selected":      "✅ Yes" if row["selected"] else "—",
        })

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    st.markdown("""
    > **Interpretation:**
    > - **Momentum score** ranks stocks by 6-month price momentum (rank 0→1)
    > - **Quality score** ranks stocks by ROE, low debt, and earnings growth (rank 0→1)
    > - **Low-Vol score** ranks stocks by **inverse** realised volatility (rank 0→1)
    > - **Combined = 0.4×Momentum + 0.4×Quality + 0.2×Low-Vol**
    > - Top 15 by combined score enter the portfolio. LLM views then fine-tune exact weights.
    """)


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
    Friendly Market Conditions panel with traffic light, VIX mood meter, and plain-English narrative.
    """
    r = snap["regime"]
    v = snap["vix"]

    # ── Traffic light ────────────────────────────────────────────────────────
    _regime_key = r.get("regime", "neutral")
    _tl_config = {
        "bull":    ("🟢", "Green light", "Markets are healthy. Good time to invest.",
                    "#16a34a", "#f0fdf4", "#bbf7d0"),
        "neutral": ("🟡", "Yellow light", "Markets are mixed. Invest carefully.",
                    "#ca8a04", "#fefce8", "#fde68a"),
        "bear":    ("🔴", "Red light",    "Markets are stressed. We're being cautious.",
                    "#dc2626", "#fef2f2", "#fecaca"),
    }.get(_regime_key, ("🟡", "Yellow light", "Markets are mixed.", "#ca8a04", "#fefce8", "#fde68a"))
    _tl_icon, _tl_title, _tl_desc, _tl_color, _tl_bg, _tl_border = _tl_config

    # ── VIX mood ─────────────────────────────────────────────────────────────
    _vix_val = v.get("vix", 20)
    if _vix_val < 15:
        _vix_face, _vix_mood, _vix_color = "😊", "Calm markets", "#16a34a"
    elif _vix_val < 20:
        _vix_face, _vix_mood, _vix_color = "😐", "Slightly nervous", "#ca8a04"
    elif _vix_val < 30:
        _vix_face, _vix_mood, _vix_color = "😰", "Fearful", "#ea580c"
    else:
        _vix_face, _vix_mood, _vix_color = "🚨", "Panic mode", "#dc2626"

    _deployed_pct = int(snap.get("combined_scale", 1.0) * 100)
    _cash_pct     = int(snap.get("cash_buffer", 0.0) * 100)

    st.markdown(f"""
    <div style="display:flex; gap:1rem; flex-wrap:wrap; margin-bottom:1rem;">

        <!-- Traffic light card -->
        <div style="flex:1; min-width:220px; background:{_tl_bg}; border:2px solid {_tl_border};
                    border-radius:12px; padding:1rem 1.2rem;">
            <div style="font-size:2.2rem; margin-bottom:0.3rem;">{_tl_icon}</div>
            <div style="font-size:1rem; font-weight:800; color:{_tl_color};">{_tl_title}</div>
            <div style="font-size:0.85rem; color:#475569; margin-top:0.2rem;">{_tl_desc}</div>
            <div style="font-size:0.78rem; color:{_tl_color}; font-weight:600; margin-top:0.5rem;">
                {r.get('emoji','')} {r.get('label','').title()} regime
            </div>
        </div>

        <!-- VIX fear meter -->
        <div style="flex:1; min-width:220px; background:white; border:1px solid #e2e8f0;
                    border-radius:12px; padding:1rem 1.2rem;">
            <div style="font-size:0.78rem; color:#64748b; font-weight:600; text-transform:uppercase;
                        letter-spacing:0.04em; margin-bottom:0.4rem;">Market Fear Meter</div>
            <div style="display:flex; align-items:center; gap:0.6rem;">
                <span style="font-size:1.8rem;">{_vix_face}</span>
                <div>
                    <div style="font-size:1.3rem; font-weight:800; color:{_vix_color};">{_vix_mood}</div>
                    <div style="font-size:0.78rem; color:#94a3b8;">VIX = {_vix_val:.1f}</div>
                </div>
            </div>
            <div style="background:#f1f5f9; border-radius:6px; height:8px; margin-top:0.7rem; overflow:hidden;">
                <div style="background:{_vix_color}; width:{min(int(_vix_val/50*100), 100)}%; height:100%; border-radius:6px;"></div>
            </div>
            <div style="display:flex; justify-content:space-between; margin-top:0.2rem;">
                <span style="font-size:0.68rem; color:#94a3b8;">Calm</span>
                <span style="font-size:0.68rem; color:#94a3b8;">Panic</span>
            </div>
        </div>

        <!-- Deployment card -->
        <div style="flex:1; min-width:220px; background:white; border:1px solid #e2e8f0;
                    border-radius:12px; padding:1rem 1.2rem;">
            <div style="font-size:0.78rem; color:#64748b; font-weight:600; text-transform:uppercase;
                        letter-spacing:0.04em; margin-bottom:0.4rem;">Your money going in</div>
            <div style="font-size:1.8rem; font-weight:900; color:#1e293b;">{_deployed_pct}%</div>
            <div style="font-size:0.82rem; color:#64748b; margin-bottom:0.5rem;">invested in stocks</div>
            <div style="background:#f1f5f9; border-radius:6px; height:8px; overflow:hidden;">
                <div style="background:#3b82f6; width:{_deployed_pct}%; height:100%; border-radius:6px;"></div>
            </div>
            <div style="font-size:0.78rem; color:#94a3b8; margin-top:0.3rem;">
                {_cash_pct}% kept as safety buffer
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Narrative ────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:{_tl_bg}; border-left:4px solid {_tl_color}; border-radius:0 8px 8px 0;
                padding:0.8rem 1.1rem; margin-bottom:1rem; font-size:0.9rem; color:#1e293b;">
        <strong>What this means for you:</strong> {snap.get("narrative", "")}
    </div>
    """, unsafe_allow_html=True)

    # ── Technical signals (collapsed) ────────────────────────────────────────
    if r.get("signals"):
        sig = r["signals"]
        with st.expander("📐 Technical signals (Nifty 50 moving averages)"):
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Nifty 50",        f"₹{sig.get('nifty50',0):,.0f}")
            sc2.metric("50-day average",  f"₹{sig.get('ma50',0):,.0f}")
            sc3.metric("200-day average", f"₹{sig.get('ma200',0):,.0f}")
            sc4.metric("20-day momentum", f"{sig.get('momentum_20d',0):.1%}")


# ══════════════════════════════════════════════════════════════════════════════
#  ENHANCED BACKTEST CHARTS  (from backtester.py)
# ══════════════════════════════════════════════════════════════════════════════

def chart_enhanced_cumulative(results_df: pd.DataFrame) -> go.Figure:
    """Cumulative returns for BL (net), BL (gross), Equal Weight, Nifty 50."""
    cum_bl_net   = (1 + results_df["bl_net"]).cumprod()   - 1
    cum_bl_gross = (1 + results_df["bl_gross"]).cumprod() - 1
    cum_eq       = (1 + results_df["eq_net"]).cumprod()   - 1
    cum_nifty    = (1 + results_df["nifty"]).cumprod()    - 1

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        subplot_titles=("Cumulative Return", "Monthly Period Costs (₹)"))

    fig.add_trace(go.Scatter(x=results_df.index, y=cum_bl_net,
        name="BL+Factor (After Costs)", line=dict(color="#3b82f6", width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=results_df.index, y=cum_bl_gross,
        name="BL+Factor (Before Costs)", line=dict(color="#93c5fd", width=1.5, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=results_df.index, y=cum_eq,
        name="Equal Weight", line=dict(color="#f59e0b", width=1.8, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=results_df.index, y=cum_nifty,
        name="Nifty 50", line=dict(color="#64748b", width=1.5, dash="dot")), row=1, col=1)

    fig.add_trace(go.Bar(x=results_df.index, y=results_df["period_costs"],
        name="Period Costs", marker_color="#fca5a5", showlegend=False), row=2, col=1)

    fig.update_layout(
        height=560, plot_bgcolor="white",
        yaxis_tickformat=".0%", yaxis2_title="Cost (₹)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def render_enhanced_backtest_tab(results_df, metrics_df, costs_df):
    """Render the full enhanced backtest tab content."""

    # ── ₹1 Lakh story cards ──────────────────────────────────────────────────
    m_bl = metrics_df[metrics_df["label"].str.contains("After Costs", na=False)]
    m_eq = metrics_df[metrics_df["label"].str.contains("Equal", na=False)]
    m_ni = metrics_df[metrics_df["label"].str.contains("Nifty", na=False)]

    if not m_bl.empty:
        # Estimate years from date range
        _years = max(1, (results_df.index[-1] - results_df.index[0]).days / 365.25)
        _start = 100_000

        _bl_cum  = float(m_bl.iloc[0]["cumulative_ret"]) if "cumulative_ret" in m_bl.columns else 0
        _ni_cum  = float(m_ni.iloc[0]["cumulative_ret"]) if (not m_ni.empty and "cumulative_ret" in m_ni.columns) else 0
        _fd_cum  = (1.065 ** _years) - 1  # approx 6.5% FD

        _bl_end  = round(_start * (1 + _bl_cum))
        _ni_end  = round(_start * (1 + _ni_cum))
        _fd_end  = round(_start * (1 + _fd_cum))
        _yrs_lbl = f"{_years:.0f} years"

        st.markdown(f"""
        <div style="text-align:center; padding:0.5rem 0 1rem;">
            <h4 style="color:#1e293b; margin:0;">What if you had invested ₹1 Lakh? 💭</h4>
            <p style="color:#64748b; font-size:0.88rem; margin:0.3rem 0 0;">
                Honest backtest over {_yrs_lbl} — zero look-ahead bias, real trading costs included ✅
            </p>
        </div>
        """, unsafe_allow_html=True)

        _sc1, _sc2, _sc3 = st.columns(3)

        _sc1.markdown(f"""
        <div class="finance-card" style="text-align:center; border-top:4px solid #94a3b8; opacity:0.9;">
            <div style="font-size:1.5rem; margin-bottom:0.4rem;">🏦</div>
            <div style="font-weight:700; color:#1e293b; margin-bottom:0.75rem;">Fixed Deposit</div>
            <div style="color:#64748b; font-size:0.9rem;">₹1,00,000</div>
            <div style="font-size:1.4rem; color:#64748b; margin:0.3rem 0;">↓</div>
            <div style="font-size:1.6rem; font-weight:800; color:#64748b;">
                {fmt_inr(_fd_end, compact=True)}
            </div>
            <div style="color:#94a3b8; font-size:0.82rem; margin-top:0.4rem;">
                +{_fd_cum:.0%} in {_yrs_lbl}
            </div>
        </div>
        """, unsafe_allow_html=True)

        _sc2.markdown(f"""
        <div class="finance-card" style="text-align:center; border-top:4px solid #64748b; opacity:0.9;">
            <div style="font-size:1.5rem; margin-bottom:0.4rem;">📊</div>
            <div style="font-weight:700; color:#1e293b; margin-bottom:0.75rem;">Nifty 50 Index</div>
            <div style="color:#64748b; font-size:0.9rem;">₹1,00,000</div>
            <div style="font-size:1.4rem; color:#64748b; margin:0.3rem 0;">↓</div>
            <div style="font-size:1.6rem; font-weight:800; color:#1e293b;">
                {fmt_inr(_ni_end, compact=True)}
            </div>
            <div style="color:#64748b; font-size:0.82rem; margin-top:0.4rem;">
                +{_ni_cum:.0%} in {_yrs_lbl}
            </div>
        </div>
        """, unsafe_allow_html=True)

        _sc3.markdown(f"""
        <div style="background:linear-gradient(135deg,#1e3a5f 0%,#2563eb 100%);
                    border-radius:16px; padding:1.5rem; text-align:center; color:white;
                    box-shadow:0 4px 16px rgba(37,99,235,0.3);">
            <div style="font-size:1.5rem; margin-bottom:0.4rem;">🤖</div>
            <div style="font-weight:800; font-size:1.05rem; margin-bottom:0.75rem;">PortfolioAI</div>
            <div style="opacity:0.8; font-size:0.9rem;">₹1,00,000</div>
            <div style="font-size:1.4rem; opacity:0.8; margin:0.3rem 0;">↓</div>
            <div style="font-size:1.9rem; font-weight:900;">
                {fmt_inr(_bl_end, compact=True)}
            </div>
            <div style="opacity:0.85; font-size:0.88rem; margin-top:0.4rem;">
                +{_bl_cum:.0%} in {_yrs_lbl}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Here's how each investment grew month by month:**")

    st.markdown("""
    <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:10px;
                padding:0.85rem 1.2rem; margin-bottom:0.75rem;">
        <div style="font-weight:700; color:#14532d; font-size:0.95rem; margin-bottom:0.5rem;">
            ✅ Honest Backtest — Zero Look-Ahead Bias
        </div>
        <div style="font-size:0.82rem; color:#166534; line-height:1.75;">
            • <strong>Momentum signal:</strong> uses only prices that existed before each decision date<br>
            • <strong>Volatility signal:</strong> uses only returns that existed before each decision date<br>
            • <strong>Quality fundamentals:</strong> completely removed — no ROE/D:E/EPS in backtest<br>
            • <strong>Stock eligibility:</strong> each stock must have ≥130 days of history at decision time<br>
            • <strong>Transaction costs:</strong> real Zerodha rates (brokerage + STT + GST + stamp)<br>
            • <em>Note: our 16-stock universe is fixed. A larger universe would improve results further.</em>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.plotly_chart(chart_enhanced_cumulative(results_df),
                    use_container_width=True, key="enhanced_bt_chart")

    # ── Plain English KPIs ────────────────────────────────────────────────────
    if not m_bl.empty and not m_ni.empty:
        bl_cagr = float(m_bl.iloc[0]["ann_return"])
        ni_cagr = float(m_ni.iloc[0]["ann_return"])
        alpha   = bl_cagr - ni_cagr
        total_c = float(m_bl.iloc[0]["total_costs_inr"])
        drag    = float(m_bl.iloc[0]["cost_drag_ann"])
        bl_dd   = float(m_bl.iloc[0]["max_drawdown"]) if "max_drawdown" in m_bl.columns else 0
        bl_shr  = float(m_bl.iloc[0]["sharpe"]) if "sharpe" in m_bl.columns else 0

        st.markdown("#### In plain English")
        _pk1, _pk2, _pk3, _pk4 = st.columns(4)
        _pk1.markdown(f"""
        <div class="finance-card" style="text-align:center;">
            <div style="color:#64748b; font-size:0.8rem;">Yearly growth rate</div>
            <div style="font-size:1.6rem; font-weight:800; color:#2563eb; margin:0.3rem 0;">
                {bl_cagr:.2%}
            </div>
            <div style="color:#94a3b8; font-size:0.75rem;">like a FD giving {bl_cagr:.1%}/yr!</div>
        </div>
        """, unsafe_allow_html=True)
        _pk2.markdown(f"""
        <div class="finance-card" style="text-align:center;">
            <div style="color:#64748b; font-size:0.8rem;">Beat Nifty 50 by</div>
            <div style="font-size:1.6rem; font-weight:800; color:{"#16a34a" if alpha>0 else "#dc2626"}; margin:0.3rem 0;">
                {alpha:+.2%}/yr
            </div>
            <div style="color:#94a3b8; font-size:0.75rem;">{"above" if alpha>0 else "below"} index after costs</div>
        </div>
        """, unsafe_allow_html=True)
        _pk3.markdown(f"""
        <div class="finance-card" style="text-align:center;">
            <div style="color:#64748b; font-size:0.8rem;">Worst loss from peak</div>
            <div style="font-size:1.6rem; font-weight:800; color:#ea580c; margin:0.3rem 0;">
                {bl_dd:.1%}
            </div>
            <div style="color:#94a3b8; font-size:0.75rem;">happened during market crash</div>
        </div>
        """, unsafe_allow_html=True)
        _pk4.markdown(f"""
        <div class="finance-card" style="text-align:center;">
            <div style="color:#64748b; font-size:0.8rem;">Risk efficiency</div>
            <div style="font-size:1.6rem; font-weight:800; color:#7c3aed; margin:0.3rem 0;">
                {bl_shr:.2f}
            </div>
            <div style="color:#94a3b8; font-size:0.75rem;">return per unit of risk taken</div>
        </div>
        """, unsafe_allow_html=True)

        if alpha > 0.02:
            st.markdown(
                f'<div class="insight-card good">'
                f'🎉 Strategy beats Nifty 50 by <strong>{alpha:.2%}/year</strong> after real costs. '
                f'Total trading costs paid: <strong>₹{total_c:,.0f}</strong> ({drag:.3f}%/yr drag).'
                f'</div>',
                unsafe_allow_html=True,
            )
        elif alpha > 0:
            st.markdown(
                f'<div class="insight-card warning">'
                f'Strategy beats Nifty 50 by {alpha:.2%}/year after costs. '
                f'Consider longer rebalancing periods to cut costs further.'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.warning(f"Strategy underperforms Nifty 50 by {abs(alpha):.2%}/year after costs.")

    # ── Full metrics table ────────────────────────────────────────────────────
    with st.expander("📐 Full performance metrics table"):
        fmt_cols = {
            "cumulative_ret": ".1%", "ann_return": ".2%", "ann_vol": ".2%",
            "sharpe": ".3f", "max_drawdown": ".2%", "calmar": ".3f",
            "win_rate": ".1%", "final_value": ",.0f",
            "total_costs_inr": ",.0f", "cost_drag_ann": ".3f",
        }
        disp = metrics_df[["label"] + [c for c in fmt_cols if c in metrics_df.columns]].copy()
        for col, fmt in fmt_cols.items():
            if col in disp.columns:
                disp[col] = disp[col].apply(
                    lambda x: format(x, fmt) if pd.notna(x) else "n/a"
                )
        disp.columns = [c.replace("_", " ").title() for c in disp.columns]
        st.dataframe(disp, use_container_width=True, hide_index=True)

    if costs_df is not None and not costs_df.empty:
        st.markdown("#### Monthly Cost History")
        c_fig = go.Figure(go.Bar(
            x=costs_df.index, y=costs_df["cost_inr"],
            marker_color="#fca5a5",
            hovertemplate="<b>%{x}</b><br>₹%{y:.0f}<extra></extra>",
        ))
        c_fig.update_layout(
            title="Transaction Costs per Rebalancing Period (₹)",
            yaxis_title="Cost (₹)", height=250, plot_bgcolor="white",
            margin=dict(l=40, r=20, t=40, b=40),
        )
        st.plotly_chart(c_fig, use_container_width=True, key="cost_hist_chart")

    st.markdown("""
    <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:8px;
                padding:0.6rem 1rem; font-size:0.82rem; color:#14532d; margin-top:0.5rem;">
        ✅ <strong>No look-ahead bias.</strong>
        Every decision in this backtest used only information that was available
        at that point in time. No future data was used at any step.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PORTFOLIO TRACKER
# ══════════════════════════════════════════════════════════════════════════════

_TRACKER_TEMPLATE = [
    {"Ticker": t, "Company": STOCK_META.get(t, {}).get("name", t), "Qty": 0, "Buy Price (₹)": 0.0}
    for t in STOCKS
]


def _calc_tracker_pnl(holdings_df: pd.DataFrame, prices_inr: pd.Series) -> pd.DataFrame | None:
    """Calculate P&L for each holding using current prices."""
    active = holdings_df[(holdings_df["Qty"] > 0) & (holdings_df["Buy Price (₹)"] > 0)].copy()
    if active.empty:
        return None

    rows = []
    for _, r in active.iterrows():
        t         = r["Ticker"]
        qty       = float(r["Qty"])
        buy_price = float(r["Buy Price (₹)"])
        cur_price = float(prices_inr.get(t, 0)) if prices_inr is not None else 0.0

        if cur_price <= 0:
            cur_price = buy_price   # fallback

        cost      = buy_price * qty
        cur_val   = cur_price  * qty
        pnl       = cur_val - cost
        ret_pct   = (cur_price / buy_price) - 1.0 if buy_price > 0 else 0.0
        sector    = STOCK_META.get(t, {}).get("sector", "Other")

        rows.append({
            "Ticker":      t,
            "Company":     STOCK_META.get(t, {}).get("name", t),
            "Sector":      sector,
            "Qty":         int(qty),
            "Buy Price":   buy_price,
            "Cur Price":   cur_price,
            "Cost":        round(cost,    2),
            "Cur Value":   round(cur_val, 2),
            "P&L":         round(pnl,     2),
            "Return %":    round(ret_pct, 4),
        })

    return pd.DataFrame(rows)


def _next_action_signal(holdings_df: pd.DataFrame, factor_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Generate BUY / SELL / HOLD signals by comparing current holdings
    with top-15 factor scores.
    """
    if factor_df is None or factor_df.empty:
        return pd.DataFrame()

    held_tickers = set(
        holdings_df[holdings_df["Qty"] > 0]["Ticker"].tolist()
    )

    top15 = set(factor_df[factor_df["selected"] == True].index.tolist())

    rows = []
    # Stocks to SELL (held but not in top 15)
    for t in held_tickers:
        if t not in top15:
            score = float(factor_df.loc[t, "combined_score"]) if t in factor_df.index else 0.0
            rows.append({"Action": "SELL", "Ticker": t,
                         "Company": STOCK_META.get(t, {}).get("name", t),
                         "Reason": "Dropped from top-15 factor score",
                         "Combined Score": f"{score:.3f}",
                         "Urgency": "This month"})

    # Stocks to BUY (top 15 but not held)
    for t in top15:
        if t not in held_tickers:
            score = float(factor_df.loc[t, "combined_score"]) if t in factor_df.index else 0.0
            rows.append({"Action": "BUY", "Ticker": t,
                         "Company": STOCK_META.get(t, {}).get("name", t),
                         "Reason": "Entered top-15 by factor score",
                         "Combined Score": f"{score:.3f}",
                         "Urgency": "This month"})

    # Stocks to HOLD (held and in top 15)
    for t in held_tickers:
        if t in top15:
            score = float(factor_df.loc[t, "combined_score"]) if t in factor_df.index else 0.0
            rows.append({"Action": "HOLD", "Ticker": t,
                         "Company": STOCK_META.get(t, {}).get("name", t),
                         "Reason": "Still in top-15",
                         "Combined Score": f"{score:.3f}",
                         "Urgency": "─"})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    order = {"SELL": 0, "BUY": 1, "HOLD": 2}
    df["_order"] = df["Action"].map(order)
    return df.sort_values(["_order", "Combined Score"]).drop(columns="_order").reset_index(drop=True)


def render_portfolio_tracker():
    """
    Full Portfolio Tracker page.
    User enters actual holdings → sees P&L, vs Nifty 50, and next month signals.
    """
    from datetime import date as _date
    _today_str = _date.today().strftime("%d %b %Y")

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0f172a 0%,#1e40af 100%);
                border-radius:16px; padding:1.5rem 2rem; margin-bottom:1.5rem; color:white;">
        <div style="font-size:0.8rem; opacity:0.7; margin-bottom:0.3rem;">As of {_today_str}</div>
        <h2 style="margin:0 0 0.3rem; font-size:1.5rem;">📊 My Portfolio</h2>
        <p style="margin:0; opacity:0.85; font-size:0.9rem;">
            Enter your holdings to see exactly how your money is performing — and what to do next.
        </p>
    </div>
    """, unsafe_allow_html=True)

    prices_inr   = load_prices_inr()
    factor_df    = load_factor_scores_df()

    # ── Step 1: Holdings input ────────────────────────────────────────────────
    st.markdown("#### Your holdings")
    st.caption("How many shares do you own? What did you pay per share? Fill in the rows below.")

    # Pre-fill current prices as reference
    ref_prices = {}
    if prices_inr is not None:
        for t in STOCKS:
            ref_prices[t] = float(prices_inr.get(t, 0))

    template = pd.DataFrame([
        {
            "Ticker":       t,
            "Company":      STOCK_META.get(t, {}).get("name", t),
            "Sector":       STOCK_META.get(t, {}).get("sector", ""),
            "Current Price (₹)": ref_prices.get(t, 0.0),
            "Qty":          0,
            "Buy Price (₹)":0.0,
        }
        for t in STOCKS
    ])

    edited = st.data_editor(
        template,
        use_container_width=True,
        disabled=["Ticker", "Company", "Sector", "Current Price (₹)"],
        column_config={
            "Current Price (₹)": st.column_config.NumberColumn(format="₹%.2f", disabled=True),
            "Qty": st.column_config.NumberColumn(
                "Qty (shares)", help="How many shares you hold", min_value=0, step=1),
            "Buy Price (₹)": st.column_config.NumberColumn(
                "Buy Price (₹)", help="Your average cost per share",
                min_value=0.0, format="₹%.2f"),
        },
        num_rows="fixed",
        hide_index=True,
        key="tracker_holdings",
    )

    # ── Calculate P&L ─────────────────────────────────────────────────────────
    pnl_df = _calc_tracker_pnl(edited, prices_inr)
    if pnl_df is None or pnl_df.empty:
        st.markdown("""
        <div style="background:#f0f9ff; border:1px solid #bae6fd; border-radius:12px;
                    padding:1.2rem 1.5rem; text-align:center; margin:0.5rem 0 1rem;">
            <div style="font-size:1.8rem; margin-bottom:0.5rem;">👆</div>
            <div style="font-size:1rem; font-weight:700; color:#0369a1; margin-bottom:0.3rem;">
                Fill in your holdings above to see your portfolio health
            </div>
            <div style="font-size:0.85rem; color:#0284c7;">
                Set <strong>Qty</strong> (number of shares) and <strong>Buy Price</strong> for each stock you own.
                We'll instantly show your profit/loss and what to do next.
            </div>
        </div>
        """, unsafe_allow_html=True)
        # Still show next-action even with no holdings
        if factor_df is not None:
            st.divider()
            st.markdown("#### 🎯 Next Month's Top 15 Stocks (Factor Score)")
            st.caption("Even without holdings entered, here's what our model recommends now:")
            top15 = factor_df[factor_df["selected"] == True].reset_index()
            top15 = top15[["ticker","sector","momentum_score","quality_score","volatility_score","combined_score"]].copy()
            top15.columns = ["Ticker","Sector","Momentum","Quality","Low-Vol","Combined"]
            for c in ["Momentum","Quality","Low-Vol","Combined"]:
                top15[c] = top15[c].apply(lambda x: f"{x:.3f}")
            st.dataframe(top15, use_container_width=True, hide_index=True)
        return

    # ── KPI row ───────────────────────────────────────────────────────────────
    total_cost    = pnl_df["Cost"].sum()
    total_val     = pnl_df["Cur Value"].sum()
    total_pnl     = pnl_df["P&L"].sum()
    total_ret     = (total_val / total_cost - 1) if total_cost > 0 else 0
    n_stocks      = len(pnl_df)

    # Nifty 50 return over approximate holding period (use 1-year as proxy if no dates)
    nifty_1yr = 0.0
    nifty_path = f"{DATA_DIR}/prices.csv"
    if os.path.exists(nifty_path):
        try:
            px = pd.read_csv(nifty_path, index_col=0, parse_dates=True)
            if "NIFTY50" in px.columns:
                nifty_series = px["NIFTY50"].dropna()
                if len(nifty_series) >= 252:
                    nifty_1yr = float(nifty_series.iloc[-1] / nifty_series.iloc[-252]) - 1
        except Exception:
            pass

    _pnl_color   = "#16a34a" if total_pnl >= 0 else "#dc2626"
    _pnl_icon    = "▲" if total_pnl >= 0 else "▼"
    _pnl_label   = "You're in profit! 🎉" if total_pnl >= 0 else "Currently at a loss"
    _vs_nifty    = total_ret - nifty_1yr
    _vs_label    = f"{'Beating' if _vs_nifty >= 0 else 'Behind'} Nifty by {abs(_vs_nifty):.1%}"
    _vs_icon     = "🏆" if _vs_nifty >= 0 else "📉"

    st.markdown(f"""
    <div style="background:white; border:1px solid #e2e8f0; border-radius:14px;
                padding:1.2rem 1.5rem; margin-bottom:1rem;">
        <div style="display:flex; align-items:baseline; gap:0.6rem; flex-wrap:wrap;">
            <div style="font-size:2rem; font-weight:900; color:#1e293b;">{fmt_inr(total_val)}</div>
            <div style="font-size:0.85rem; color:#64748b;">current value of your portfolio</div>
        </div>
        <div style="display:flex; gap:1.5rem; flex-wrap:wrap; margin-top:0.8rem;">
            <div>
                <div style="font-size:0.75rem; color:#64748b;">Total invested</div>
                <div style="font-size:1rem; font-weight:700; color:#1e293b;">{fmt_inr(total_cost)}</div>
            </div>
            <div>
                <div style="font-size:0.75rem; color:#64748b;">Overall gain/loss</div>
                <div style="font-size:1rem; font-weight:700; color:{_pnl_color};">
                    {_pnl_icon} {fmt_inr(abs(total_pnl))} ({total_ret:+.1%})
                </div>
                <div style="font-size:0.72rem; color:{_pnl_color};">{_pnl_label}</div>
            </div>
            <div>
                <div style="font-size:0.75rem; color:#64748b;">vs Nifty 50 (1 yr)</div>
                <div style="font-size:1rem; font-weight:700; color:{'#16a34a' if _vs_nifty >= 0 else '#dc2626'};">
                    {_vs_icon} {_vs_label}
                </div>
                <div style="font-size:0.72rem; color:#94a3b8;">Nifty returned {nifty_1yr:+.1%}</div>
            </div>
            <div>
                <div style="font-size:0.75rem; color:#64748b;">Stocks</div>
                <div style="font-size:1rem; font-weight:700; color:#1e293b;">{n_stocks} holdings</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Groww-style per-stock list ────────────────────────────────────────────
    st.markdown("#### Your stocks at a glance")
    _stocks_sorted = pnl_df.sort_values("Return %", ascending=False)
    for _, _sr in _stocks_sorted.iterrows():
        _tk      = _sr["Ticker"]
        _co      = _sr.get("Company", _tk)
        _sec     = _sr.get("Sector", "")
        _ret_pct = _sr["Return %"]
        _pnl_v   = _sr["P&L"]
        _cur_val = _sr["Cur Value"]
        _ret_c   = "#16a34a" if _ret_pct >= 0 else "#dc2626"
        _ret_ic  = "▲" if _ret_pct >= 0 else "▼"
        _meta_t  = STOCK_META.get(_tk, {})

        st.markdown(f"""
        <div style="background:white; border:1px solid #e2e8f0; border-radius:10px;
                    padding:0.75rem 1rem; margin:0.3rem 0;
                    display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:0.5rem;">
            <div style="display:flex; align-items:center; gap:0.7rem; min-width:200px;">
                <div style="font-size:1.4rem;">{_meta_t.get('flag','🏢')}</div>
                <div>
                    <div style="font-weight:700; color:#1e293b; font-size:0.95rem;">{_tk}</div>
                    <div style="font-size:0.75rem; color:#94a3b8;">{_sec}</div>
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-weight:700; color:#1e293b;">{fmt_inr(_cur_val)}</div>
                <div style="font-size:0.82rem; color:{_ret_c}; font-weight:600;">
                    {_ret_ic} {abs(_ret_pct):.1%} &nbsp;·&nbsp; {_ret_ic} {fmt_inr(abs(_pnl_v), compact=True)}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tr1, tr2, tr3 = st.tabs(["📋 Full Details", "📊 Attribution", "🎯 Next Month Signal"])

    with tr1:
        # Format table for display
        display = pnl_df.copy()
        display["Buy Price"] = display["Buy Price"].apply(lambda x: f"₹{x:,.2f}")
        display["Cur Price"] = display["Cur Price"].apply(lambda x: f"₹{x:,.2f}")
        display["Cost"]      = display["Cost"].apply(lambda x: fmt_inr(x))
        display["Cur Value"] = display["Cur Value"].apply(lambda x: fmt_inr(x))
        display["P&L"]       = display["P&L"].apply(
            lambda x: f"{'▲' if x>=0 else '▼'} {fmt_inr(abs(x))}"
        )
        display["Return %"]  = display["Return %"].apply(lambda x: f"{x:+.2%}")

        def _colour_pnl(row):
            colour = "#f0fdf4" if "▲" in str(row.get("P&L","")) else "#fef2f2"
            return [colour] * len(row)

        st.dataframe(display.style.apply(_colour_pnl, axis=1),
                     use_container_width=True, hide_index=True)

        # Sector breakdown
        sec_agg = pnl_df.groupby("Sector")[["Cost","Cur Value","P&L"]].sum()
        sec_agg["Return %"] = sec_agg["Cur Value"] / sec_agg["Cost"] - 1
        sec_display = sec_agg.copy()
        sec_display["Return %"] = sec_display["Return %"].apply(lambda x: f"{x:+.2%}")
        sec_display["Cur Value"] = sec_display["Cur Value"].apply(lambda x: fmt_inr(x))
        st.markdown("**Sector Breakdown:**")
        st.dataframe(sec_display, use_container_width=True)

    with tr2:
        # Attribution: contribution of each stock to total return
        pnl_df["Contribution"] = pnl_df["P&L"] / total_cost
        contrib_sorted = pnl_df.sort_values("Contribution")

        colors = ["#16a34a" if c >= 0 else "#dc2626" for c in contrib_sorted["Contribution"]]
        fig_attr = go.Figure(go.Bar(
            x=contrib_sorted["Contribution"],
            y=contrib_sorted["Ticker"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.2%}" for v in contrib_sorted["Contribution"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Contribution: %{x:.2%}<extra></extra>",
        ))
        fig_attr.add_vline(x=0, line_color="#1e293b", line_width=1)
        fig_attr.update_layout(
            title="Portfolio Return Attribution (each stock's contribution)",
            xaxis_title="Contribution to Total Return",
            xaxis_tickformat=".2%",
            plot_bgcolor="white", height=400,
            margin=dict(l=80, r=80, t=50, b=40),
        )
        st.plotly_chart(fig_attr, use_container_width=True, key="tracker_attribution")

        # Winners vs Losers
        winners = pnl_df[pnl_df["P&L"] > 0].sort_values("P&L", ascending=False)
        losers  = pnl_df[pnl_df["P&L"] < 0].sort_values("P&L")
        wl1, wl2 = st.columns(2)
        with wl1:
            st.markdown("**Top Winners**")
            if not winners.empty:
                w_disp = winners[["Ticker","Company","P&L","Return %"]].copy()
                w_disp["P&L"] = w_disp["P&L"].apply(lambda x: f"₹{x:,.0f}")
                w_disp["Return %"] = w_disp["Return %"].apply(lambda x: f"{x:+.2%}")
                st.dataframe(w_disp, hide_index=True, use_container_width=True)
        with wl2:
            st.markdown("**Stocks in Loss**")
            if not losers.empty:
                l_disp = losers[["Ticker","Company","P&L","Return %"]].copy()
                l_disp["P&L"] = l_disp["P&L"].apply(lambda x: f"₹{x:,.0f}")
                l_disp["Return %"] = l_disp["Return %"].apply(lambda x: f"{x:+.2%}")
                st.dataframe(l_disp, hide_index=True, use_container_width=True)

    with tr3:
        st.markdown("#### 🎯 What should you do this month?")
        st.caption(
            "Based on which stocks have the best momentum + quality right now. "
            "Review every 3–4 weeks."
        )

        if factor_df is None:
            st.markdown("""
            <div style="background:#fefce8; border:1px solid #fde68a; border-radius:10px;
                        padding:1rem 1.2rem; color:#92400e;">
                📊 Signal data not available yet.
                Run <code>python scorer.py</code> to generate factor scores.
            </div>
            """, unsafe_allow_html=True)
        else:
            signals = _next_action_signal(edited, factor_df)
            if signals.empty:
                st.markdown("""
                <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:10px;
                            padding:1rem 1.2rem; text-align:center;">
                    <div style="font-size:1.5rem; margin-bottom:0.4rem;">✅</div>
                    <div style="font-weight:700; color:#16a34a; font-size:1rem;">No changes needed this month!</div>
                    <div style="color:#166534; font-size:0.85rem; margin-top:0.2rem;">
                        Your holdings already match our top picks. Just hold and relax.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                sells = signals[signals["Action"] == "SELL"]
                buys  = signals[signals["Action"] == "BUY"]
                holds = signals[signals["Action"] == "HOLD"]

                # Summary callout
                if not sells.empty or not buys.empty:
                    sell_list = ", ".join(sells["Ticker"].tolist()) or "None"
                    buy_list  = ", ".join(buys["Ticker"].tolist())  or "None"
                    st.markdown(f"""
                    <div style="background:#f0f9ff; border:1px solid #bae6fd; border-radius:10px;
                                padding:1rem 1.2rem; margin-bottom:0.8rem;">
                        <div style="font-weight:700; color:#0369a1; font-size:0.95rem; margin-bottom:0.5rem;">
                            📋 This month's action plan:
                        </div>
                        <div style="display:flex; gap:1.5rem; flex-wrap:wrap;">
                            <div><span style="color:#dc2626; font-weight:700;">🔴 SELL</span>
                                 <span style="color:#1e293b; margin-left:0.5rem;">{sell_list}</span></div>
                            <div><span style="color:#16a34a; font-weight:700;">🟢 BUY</span>
                                 <span style="color:#1e293b; margin-left:0.5rem;">{buy_list}</span></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Colour-coded action table
                action_rows = []
                for _, r in signals.iterrows():
                    act    = r["Action"]
                    badge  = {"SELL":"🔴","BUY":"🟢","HOLD":"⚪"}.get(act, "⚪")
                    action_rows.append({
                        "Action":  f"{badge} {act}",
                        "Ticker":  r["Ticker"],
                        "Company": r["Company"],
                        "Reason":  r["Reason"],
                        "Score":   r["Combined Score"],
                        "When":    r["Urgency"],
                    })

                sig_df = pd.DataFrame(action_rows)

                def _style_action(row):
                    if "SELL" in str(row["Action"]):
                        return ["background-color:#fef2f2"] * len(row)
                    if "BUY" in str(row["Action"]):
                        return ["background-color:#f0fdf4"] * len(row)
                    return [""] * len(row)

                st.dataframe(sig_df.style.apply(_style_action, axis=1),
                             use_container_width=True, hide_index=True)

                if not sells.empty:
                    st.markdown("**Why sell?** These stocks have weakened momentum or deteriorating "
                                "quality scores and have dropped out of the top-15 universe. "
                                "Sell them and reinvest proceeds in the BUY list.")
                if not buys.empty:
                    st.markdown("**Why buy?** These stocks have entered the top-15 by combined "
                                "factor score — strong momentum AND quality. Buy equal or "
                                "factor-weighted positions.")

        # Show factor scores for reference
        if factor_df is not None:
            with st.expander("📐 Full Factor Score Table (reference)"):
                top15_ref = factor_df.reset_index().copy()
                top15_ref["In Portfolio?"] = top15_ref["ticker"].apply(
                    lambda t: "✅ Yes" if t in set(edited[edited["Qty"]>0]["Ticker"]) else "─"
                )
                cols_show = ["ticker","sector","momentum_score","quality_score",
                             "volatility_score","combined_score","selected","In Portfolio?"]
                cols_show = [c for c in cols_show if c in top15_ref.columns]
                st.dataframe(top15_ref[cols_show], use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("""
    <div class="disclaimer">
    ⚠️ <strong>Tracker Disclaimer:</strong>
    P&L calculations use live NSE prices from the last downloaded prices.csv.
    Prices update only when you re-run <code>python data_collector.py</code>.
    This tool is for personal tracking and educational purposes only, not financial advice.
    Consult a SEBI-registered advisor before making investment decisions.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  FINANCIAL PLANNER PAGE
# ══════════════════════════════════════════════════════════════════════════════

def render_financial_planner() -> None:
    """Full Financial Planner page — warm, friendly, guided experience."""

    step = st.session_state.get("planner_step", 1)

    # ── Step Indicator ────────────────────────────────────────────────────────
    _steps = [("Your Details", 1), ("Health Score", 2), ("Risk Profile", 3), ("Your Plan", 4)]
    _parts = []
    for _lbl, _s in _steps:
        if _s < step:
            _parts.append(f'<span class="step-complete">✓ {_lbl}</span>')
        elif _s == step:
            _parts.append(f'<span class="step-active">● {_lbl}</span>')
        else:
            _parts.append(f'<span class="step-pending">○ {_lbl}</span>')
    st.markdown(
        '<div style="text-align:center; padding:0.75rem 0 1.5rem; font-size:0.9rem;">'
        + '  →  '.join(_parts) + '</div>',
        unsafe_allow_html=True,
    )

    # ── STEP 1: Conversational Input Form ────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding:0.25rem 0 1.5rem;">
        <h2 style="color:#1e293b; font-size:1.9rem; font-weight:800; margin:0;">
            Let's figure out your money 💰
        </h2>
        <p style="color:#64748b; font-size:1rem; max-width:480px; margin:0.6rem auto 0;">
            Takes 2 minutes. We'll tell you exactly how much you can invest — safely.
        </p>
    </div>
    """, unsafe_allow_html=True)

    dep_options = ["0 (only me)", "1", "2", "3", "4+"]
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**What's your monthly take-home salary? 👋**")
        income = st.number_input(
            "Monthly income", label_visibility="collapsed",
            min_value=0, step=1000,
            value=int(st.session_state.get("fp_income", 80_000)),
            key="fp_income_w",
        )
        st.caption("After tax, in-hand amount")

        st.markdown("**How much goes to fixed expenses? 🏠**")
        fixed_exp = st.number_input(
            "Fixed expenses", label_visibility="collapsed",
            min_value=0, step=500,
            value=int(st.session_state.get("fp_fixed_exp", 25_000)),
            key="fp_fixed_w",
        )
        st.caption("Rent, EMI, subscriptions, school fees")

        st.markdown("**What about daily spending? 🛒**")
        var_exp = st.number_input(
            "Variable expenses", label_visibility="collapsed",
            min_value=0, step=500,
            value=int(st.session_state.get("fp_var_exp", 15_000)),
            key="fp_var_w",
        )
        st.caption("Food, transport, shopping, dining out")

        st.markdown("**Any loan EMIs? 💳**")
        loan_emi = st.number_input(
            "Loan EMI", label_visibility="collapsed",
            min_value=0, step=500,
            value=int(st.session_state.get("fp_loan_emi", 0)),
            key="fp_loan_w",
        )
        st.caption("Home, car, personal loan EMIs")

        # Live mini-preview
        total_spend = fixed_exp + var_exp + loan_emi
        left_over   = income - total_spend
        if income > 0:
            left_pct = left_over / income * 100
            preview_color = "#16a34a" if left_pct >= 20 else "#ea580c" if left_pct >= 5 else "#dc2626"
            st.markdown(
                f'<div style="background:#f8fafc; border-radius:10px; padding:0.65rem 1rem; '
                f'margin-top:0.5rem; border:1px solid #e2e8f0; font-size:0.9rem;">'
                f'You spend <strong style="color:#1e293b;">{fmt_inr(total_spend, compact=True)}/mo</strong>'
                f' &nbsp;→&nbsp; '
                f'<strong style="color:{preview_color};">{fmt_inr(left_over, compact=True)} left over</strong>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("**How much have you saved so far? 🏦**")
        savings = st.number_input(
            "Existing savings", label_visibility="collapsed",
            min_value=0, step=5000,
            value=int(st.session_state.get("fp_savings", 200_000)),
            key="fp_savings_w",
        )
        st.caption("FD + savings account total")

        st.markdown("**Existing investments? 📊**")
        investments = st.number_input(
            "Existing investments", label_visibility="collapsed",
            min_value=0, step=5000,
            value=int(st.session_state.get("fp_investments", 0)),
            key="fp_investments_w",
        )
        st.caption("Stocks, mutual funds already held")

        st.markdown("**Running monthly SIPs? 🔄**")
        sip = st.number_input(
            "Monthly SIP", label_visibility="collapsed",
            min_value=0, step=500,
            value=int(st.session_state.get("fp_sip", 0)),
            key="fp_sip_w",
        )
        st.caption("Auto-investments each month")

        st.markdown("**Annual insurance premium? 🛡️**")
        insurance = st.number_input(
            "Insurance premium/year", label_visibility="collapsed",
            min_value=0, step=1000,
            value=int(st.session_state.get("fp_insurance", 12_000)),
            key="fp_insurance_w",
        )
        st.caption("Sum of all insurance premiums per year")

        st.markdown("**How old are you? 🎂**")
        age = st.slider(
            "Age", label_visibility="collapsed",
            min_value=18, max_value=70,
            value=int(st.session_state.get("fp_age", 30)),
            key="fp_age_w",
        )
        st.caption(f"{age} years old")

        st.markdown("**Who depends on your income? 👨‍👩‍👧**")
        _dep_labels = {"0 (only me)": "Just me 🙋", "1": "Spouse 👫",
                       "2": "Small family 👨‍👩‍👧", "3": "Family 👨‍👩‍👧‍👦", "4+": "Big family 🏠"}
        saved_dep_idx = min(int(st.session_state.get("fp_dep_idx", 0)), len(dep_options) - 1)
        dependents_str = st.selectbox(
            "Dependents", dep_options,
            index=saved_dep_idx,
            label_visibility="collapsed",
            key="fp_dep_w",
        )
        st.caption(_dep_labels.get(dependents_str, ""))

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button(
        "Show me my financial picture →",
        type="primary", use_container_width=True, key="fp_submit_btn",
    ):
        dep_map = {"0 (only me)": 0, "1": 1, "2": 2, "3": 3, "4+": 4}
        dep_val = dep_map.get(dependents_str, 0)

        if income == 0:
            st.error("Please enter your monthly income to continue.")
            return
        if fixed_exp == 0 and var_exp == 0:
            st.warning("Please fill in your expenses for an accurate analysis.")
            return

        try:
            FinancialProfile(
                monthly_income=float(income),
                fixed_expenses=float(fixed_exp),
                variable_expenses=float(var_exp),
                existing_savings=float(savings),
                age=int(age),
                dependents=dep_val,
                existing_investments=float(investments),
                monthly_sip=float(sip),
                insurance_premium_annual=float(insurance),
                loan_emi=float(loan_emi),
            )
        except ValueError as exc:
            st.error(f"Oops! {exc}")
            return

        for k, v in [
            ("fp_income", int(income)), ("fp_fixed_exp", int(fixed_exp)),
            ("fp_var_exp", int(var_exp)), ("fp_loan_emi", int(loan_emi)),
            ("fp_savings", int(savings)), ("fp_investments", int(investments)),
            ("fp_sip", int(sip)), ("fp_insurance", int(insurance)),
            ("fp_age", int(age)),
            ("fp_dep_idx", dep_options.index(dependents_str)),
        ]:
            st.session_state[k] = v

        st.session_state["fp_profile_dict"] = {
            "monthly_income":           float(income),
            "fixed_expenses":           float(fixed_exp),
            "variable_expenses":        float(var_exp),
            "existing_savings":         float(savings),
            "age":                      int(age),
            "dependents":               dep_val,
            "existing_investments":     float(investments),
            "monthly_sip":              float(sip),
            "insurance_premium_annual": float(insurance),
            "loan_emi":                 float(loan_emi),
        }
        st.session_state["planner_step"]   = 2
        st.session_state["financial_plan"] = None
        st.session_state["risk_answers"]   = None
        st.session_state["quiz_q_index"]   = 0
        for _qi in range(5):
            st.session_state.pop(f"quiz_q{_qi}_score", None)
        st.rerun()

    # ── STEP 2: Financial Health ──────────────────────────────────────────────
    if st.session_state.get("planner_step", 1) < 2:
        return

    profile_dict = st.session_state.get("fp_profile_dict")
    if not profile_dict:
        return

    profile = FinancialProfile(**profile_dict)
    az      = FinancialAnalyzer(profile)
    score, health_label = az.financial_health_score()
    net_disp   = az.net_disposable_income()
    sav_rate   = az.savings_rate()
    ef_gap     = az.emergency_fund_gap()
    ef_req     = az.emergency_fund_required()
    em_alloc, inv_amt = az.safe_monthly_investment()

    st.divider()

    # Big friendly health score reveal
    _score_emoji = "🌟" if score >= 80 else "😊" if score >= 70 else "🙂" if score >= 60 else "😐" if score >= 50 else "💪"
    _score_msg   = (
        "You're a money superstar!" if score >= 80 else
        "You're doing great!" if score >= 70 else
        "Good start, room to grow" if score >= 60 else
        "Let's improve this together" if score >= 50 else
        "Don't worry, we'll fix this!"
    )
    _score_color = "#16a34a" if score >= 80 else "#f59e0b" if score >= 60 else "#f97316" if score >= 40 else "#dc2626"
    _score_bg    = "#f0fdf4" if score >= 80 else "#fefce8" if score >= 60 else "#fff7ed" if score >= 40 else "#fef2f2"
    _sav_pct     = round(sav_rate * 100, 1)

    st.markdown(f"""
    <div style="background:{_score_bg}; border-radius:20px; padding:2rem;
                text-align:center; border:1px solid {_score_color}33; margin-bottom:1.5rem;">
        <div style="font-size:3rem; margin-bottom:0.4rem;">{_score_emoji}</div>
        <div style="font-size:5rem; font-weight:900; color:{_score_color}; line-height:1;">{score}</div>
        <div style="font-size:1.1rem; color:{_score_color}; font-weight:600; margin-top:0.2rem;">
            out of 100 — {health_label.split()[0]}
        </div>
        <div style="font-size:1rem; color:#1e293b; margin-top:0.75rem; font-weight:500;">
            {_score_msg}
        </div>
        <div style="color:#64748b; font-size:0.9rem; margin-top:0.4rem;">
            You save <strong>{_sav_pct}%</strong> of your income — keep it up!
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 3 key number cards
    _c1, _c2, _c3 = st.columns(3)

    _c1.markdown(f"""
    <div class="finance-card" style="text-align:center; border-top:4px solid #16a34a;">
        <div style="font-size:1.4rem;">💵</div>
        <div style="color:#64748b; font-size:0.82rem; margin:0.3rem 0;">Money left after bills</div>
        <div style="font-size:1.9rem; font-weight:800; color:#16a34a;">{fmt_inr(net_disp, compact=True)}</div>
        <div style="color:#64748b; font-size:0.78rem; margin-top:0.2rem;">every month</div>
        <div style="color:#94a3b8; font-size:0.75rem; margin-top:0.4rem;">This is what you work with</div>
    </div>
    """, unsafe_allow_html=True)

    _c2.markdown(f"""
    <div class="finance-card" style="text-align:center; border-top:4px solid #2563eb;">
        <div style="font-size:1.4rem;">🚀</div>
        <div style="color:#64748b; font-size:0.82rem; margin:0.3rem 0;">Safe to invest</div>
        <div style="font-size:1.9rem; font-weight:800; color:#2563eb;">{fmt_inr(inv_amt, compact=True)}</div>
        <div style="color:#64748b; font-size:0.78rem; margin-top:0.2rem;">per month</div>
        <div style="color:#94a3b8; font-size:0.75rem; margin-top:0.4rem;">After buffer for surprises</div>
    </div>
    """, unsafe_allow_html=True)

    _ef_color = "#ea580c" if ef_gap > 0 else "#16a34a"
    _ef_icon  = "🏦" if ef_gap > 0 else "✅"
    _ef_label = fmt_inr(ef_gap, compact=True) + " more needed" if ef_gap > 0 else "Fully funded"
    _ef_sub   = "Emergency fund not yet complete" if ef_gap > 0 else "Emergency fund complete"

    _c3.markdown(f"""
    <div class="finance-card" style="text-align:center; border-top:4px solid {_ef_color};">
        <div style="font-size:1.4rem;">{_ef_icon}</div>
        <div style="color:#64748b; font-size:0.82rem; margin:0.3rem 0;">Emergency fund</div>
        <div style="font-size:1.9rem; font-weight:800; color:{_ef_color};">{_ef_label}</div>
        <div style="color:#94a3b8; font-size:0.75rem; margin-top:0.4rem;">{_ef_sub}</div>
    </div>
    """, unsafe_allow_html=True)

    # Friendly insights
    st.markdown("#### 💬 Your personalised insights")
    _insight_types = ["good", "warning", "great", "good", "warning"]
    for _ii, _insight in enumerate(az.generate_insights()):
        st.markdown(
            f'<div class="insight-card {_insight_types[_ii % len(_insight_types)]}">💬 {_insight}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Continue to Risk Profile →", type="primary",
                 use_container_width=True, key="fp_proceed_quiz"):
        st.session_state["planner_step"] = 3
        st.session_state["quiz_q_index"] = 0
        st.rerun()

    # ── STEP 3: Risk Quiz — One Question at a Time ────────────────────────────
    if st.session_state.get("planner_step", 1) < 3:
        return

    st.divider()
    st.markdown("""
    <div style="text-align:center; padding:0.5rem 0 1rem;">
        <h3 style="color:#1e293b; font-weight:800; margin:0;">What kind of investor are you? 🎯</h3>
        <p style="color:#64748b; margin:0.4rem 0 0;">5 quick questions — no wrong answers!</p>
    </div>
    """, unsafe_allow_html=True)

    questions = RISK_QUESTIONS
    q_idx = st.session_state.get("quiz_q_index", 0)

    if q_idx < len(questions):
        q = questions[q_idx]
        opt_labels = [o[0] for o in q["options"]]
        opt_scores = [o[1] for o in q["options"]]

        # Progress indicator
        st.markdown(
            f'<div style="text-align:center; font-size:1.2rem; font-weight:800; '
            f'color:#2563eb; margin-bottom:0.5rem;">Q{q_idx + 1} of {len(questions)}</div>',
            unsafe_allow_html=True,
        )
        st.progress(q_idx / len(questions))
        st.markdown(f"### {q['text']}")
        st.markdown("<br>", unsafe_allow_html=True)

        saved_idx = min(int(st.session_state.get(f"fp_q{q_idx+1}_idx", 0)), len(opt_labels) - 1)
        chosen = st.radio(
            f"Question {q_idx + 1}",
            options=opt_labels,
            index=saved_idx,
            key=f"fp_radio_q{q_idx + 1}",
            label_visibility="collapsed",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        btn_lbl = "Next Question →" if q_idx < len(questions) - 1 else "See My Risk Profile 🎯"
        if st.button(btn_lbl, type="primary", use_container_width=True, key=f"quiz_next_{q_idx}"):
            opt_idx = opt_labels.index(chosen)
            st.session_state[f"fp_q{q_idx + 1}_idx"]  = opt_idx
            st.session_state[f"quiz_q{q_idx}_score"]   = opt_scores[opt_idx]
            st.session_state["quiz_q_index"]            = q_idx + 1
            st.rerun()

    else:
        # All 5 done — collect scores and show risk result
        _scores = []
        for _qi, _q in enumerate(questions):
            _sv = st.session_state.get(f"quiz_q{_qi}_score")
            if _sv is None:
                _idx = int(st.session_state.get(f"fp_q{_qi+1}_idx", 0))
                _sv  = _q["options"][min(_idx, len(_q["options"]) - 1)][1]
            _scores.append(_sv)

        _total, _rlabel, _remoji, _rdesc = RiskProfiler.score(_scores)

        _badge_colors = {
            "Aggressive": "#dc2626", "Moderate": "#f59e0b",
            "Conservative": "#16a34a", "Very Conservative": "#3b82f6",
        }
        _bc = _badge_colors.get(_rlabel, "#6b7280")

        st.markdown(f"""
        <div class="finance-card" style="text-align:center; padding:2rem;">
            <div style="font-size:3rem; margin-bottom:1rem;">{_remoji}</div>
            <div style="background:{_bc}; color:white; font-size:1.25rem; font-weight:800;
                        padding:0.6rem 2rem; border-radius:10px;
                        display:inline-block; margin-bottom:1.2rem; letter-spacing:1px;">
                {_rlabel.upper()} INVESTOR
            </div>
            <p style="font-size:0.98rem; color:#1e293b; max-width:500px;
                      margin:0 auto; line-height:1.65;">{_rdesc}</p>
            <div style="margin-top:1rem; color:#64748b; font-size:0.88rem;">
                Your score: {_total} out of 20
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Get My Complete Money Plan →", type="primary",
                     use_container_width=True, key="quiz_gen_plan"):
            st.balloons()
            _profile = FinancialProfile(**st.session_state["fp_profile_dict"])
            _plan    = FinancialPlanGenerator.generate_plan(_profile, _scores)
            st.session_state["financial_plan"] = _plan
            st.session_state["risk_answers"]   = _scores
            st.session_state["planner_step"]   = 4
            st.rerun()

    # ── STEP 4: Complete Financial Plan ───────────────────────────────────────
    plan = st.session_state.get("financial_plan")
    if plan is None or st.session_state.get("planner_step", 1) < 4:
        return

    st.divider()

    alloc  = plan["asset_allocation"]
    rp     = plan["risk_profile"]
    budget = plan["monthly_budget"]
    ef     = plan["emergency_fund"]

    st.markdown(f"""
    <div style="text-align:center; padding:0.25rem 0 1.5rem;">
        <h3 style="color:#1e293b; font-weight:800; margin:0;">Your personalised money plan 📋</h3>
        <p style="color:#64748b; margin:0.5rem 0 0;">Here's exactly what to do with your
            <strong style="color:#2563eb;">{fmt_inr(budget['investment_amount'], compact=True)}</strong>
            investable amount
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Risk badge
    _badge_colors = {
        "Aggressive": "#dc2626", "Moderate": "#f59e0b",
        "Conservative": "#16a34a", "Very Conservative": "#3b82f6",
    }
    _bc = _badge_colors.get(alloc["risk_label"], "#6b7280")
    st.markdown(
        f'<div style="text-align:center; margin-bottom:0.75rem;">'
        f'<span style="background:{_bc}; color:white; font-size:1.05rem; font-weight:700;'
        f'             padding:0.45rem 1.4rem; border-radius:8px;">{rp["label"]} Investor</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.progress(rp["score"] / 20, text=f"Risk score: {rp['score']}/20")
    st.markdown("<br>", unsafe_allow_html=True)

    # Asset allocation — rupee-first cards
    st.markdown("#### Where your money goes each month")
    _ac1, _ac2, _ac3, _ac4 = st.columns(4)
    _asset_cards = [
        (_ac1, "📈", "Stocks",  alloc["equity_amount"], alloc["products"]["equity"], "#3b82f6"),
        (_ac2, "🏛️", "Bonds",   alloc["debt_amount"],   alloc["products"]["debt"],   "#10b981"),
        (_ac3, "🥇", "Gold",    alloc["gold_amount"],    alloc["products"]["gold"],   "#f59e0b"),
        (_ac4, "💵", "Cash",    alloc["cash_amount"],    alloc["products"]["cash"],   "#94a3b8"),
    ]
    for _col, _icon, _name, _amt, _product, _color in _asset_cards:
        _col.markdown(f"""
        <div class="finance-card" style="text-align:center; border-top:4px solid {_color}; min-height:175px;">
            <div style="font-size:1.7rem;">{_icon}</div>
            <div style="font-weight:700; color:#1e293b; font-size:1rem; margin:0.35rem 0;">{_name}</div>
            <div style="font-size:1.55rem; font-weight:800; color:{_color}; line-height:1.1;">
                {fmt_inr(_amt, compact=True)}
            </div>
            <div style="color:#64748b; font-size:0.78rem; margin-top:0.2rem;">per month</div>
            <div style="color:#94a3b8; font-size:0.73rem; margin-top:0.5rem; line-height:1.3;">{_product}</div>
        </div>
        """, unsafe_allow_html=True)

    # Equity breakdown expander
    bkdn    = alloc["equity_breakdown"]
    eq_total = alloc["equity_amount"]
    with st.expander("📊 Equity breakdown detail"):
        b1, b2, b3 = st.columns(3)
        b1.metric("Large Cap", fmt_inr(bkdn["largecap"], compact=True) + "/mo",
                  f"{bkdn['largecap']/eq_total*100:.0f}%" if eq_total > 0 else "0%")
        b2.metric("Mid Cap",   fmt_inr(bkdn["midcap"],   compact=True) + "/mo",
                  f"{bkdn['midcap']/eq_total*100:.0f}%"   if eq_total > 0 else "0%")
        b3.metric("Small Cap", fmt_inr(bkdn["smallcap"], compact=True) + "/mo",
                  f"{bkdn['smallcap']/eq_total*100:.0f}%" if eq_total > 0 else "0%")

    # Emergency fund progress
    st.markdown("#### Emergency Fund Progress 🏦")
    if ef["gap"] > 0:
        _ef_prog = min(1.0, ef["existing"] / ef["required"]) if ef["required"] > 0 else 0.0
        st.progress(_ef_prog, text=f"{fmt_inr(ef['existing'])} saved of {fmt_inr(ef['required'])}")
        st.markdown(
            f'<div class="insight-card warning">'
            f'Almost there! 💪 You need just <strong>{fmt_inr(ef["gap"], compact=True)}</strong> more '
            f'in your emergency fund. At your savings rate, you\'ll be done in '
            f'<strong>{ef["months_to_complete"]:.0f} months</strong>! 🎯'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="insight-card good">'
            '🎉 Emergency fund fully funded! You\'re investing with a safety net. '
            'That\'s the right way to do it!'
            '</div>',
            unsafe_allow_html=True,
        )

    # Action checklist
    st.markdown("#### Your money checklist ✅")
    for i, step_txt in enumerate(plan["action_plan"]):
        st.checkbox(step_txt, value=False, key=f"fp_action_{i}")

    # Connect to AI stock picker
    st.divider()
    nse_amt = alloc["nse_stock_amount"]

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1e3a5f 0%,#2563eb 100%);
                border-radius:16px; padding:1.5rem 2rem; margin:0.5rem 0; color:white;">
        <div style="font-size:1.35rem; font-weight:800; margin-bottom:0.4rem;">
            🤖 Let AI pick your stocks
        </div>
        <div style="opacity:0.9; margin-bottom:0.5rem; font-size:1rem;">
            Your stock budget: <strong>{fmt_inr(nse_amt)}/month</strong>
        </div>
        <div style="opacity:0.75; font-size:0.88rem;">
            Our AI will analyse all NSE stocks and tell you exactly what to buy this month.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button(
        "🚀 Pick My Stocks with AI →",
        type="primary", use_container_width=True, key="fp_connect_btn",
    ):
        st.session_state.planned_investment = nse_amt
        st.session_state.mode = "fresh"
        st.rerun()

    # Download summary
    st.divider()
    from datetime import date as _dt
    today_str = _dt.today().strftime("%d %b %Y")
    file_date = _dt.today().strftime("%Y%m%d")

    summary_lines = [
        "=" * 62,
        "  PORTFOLIOAI — PERSONAL FINANCIAL PLAN",
        f"  Generated on: {today_str}",
        "=" * 62,
        "",
        "FINANCIAL HEALTH",
        f"  Score : {plan['health_score']}/100  ({plan['health_label']})",
        "",
        "MONTHLY BUDGET",
        f"  Income               : ₹{budget['income']:>10,.0f}",
        f"  Total Expenses       : ₹{budget['total_expenses']:>10,.0f}",
        f"  Net Disposable       : ₹{budget['net_disposable']:>10,.0f}",
        f"  Savings Rate         : {budget['savings_rate']:>9.1f}%",
        f"  Emergency Allocation : ₹{budget['emergency_allocation']:>10,.0f}/mo",
        f"  Safe to Invest       : ₹{budget['investment_amount']:>10,.0f}/mo",
        "",
        "EMERGENCY FUND",
        f"  Required : ₹{ef['required']:,.0f}",
        f"  Existing : ₹{ef['existing']:,.0f}",
        f"  Gap      : ₹{ef['gap']:,.0f}",
        "",
        "RISK PROFILE",
        f"  Score: {rp['score']}/20  →  {rp['label']}",
        f"  {rp['description']}",
        "",
        "ASSET ALLOCATION (per month)",
        f"  Equity : {alloc['equity_pct']*100:.0f}%  →  ₹{alloc['equity_amount']:,.0f}",
        f"  Debt   : {alloc['debt_pct']*100:.0f}%  →  ₹{alloc['debt_amount']:,.0f}",
        f"  Gold   : {alloc['gold_pct']*100:.0f}%  →  ₹{alloc['gold_amount']:,.0f}",
        f"  Cash   : {alloc['cash_pct']*100:.0f}%  →  ₹{alloc['cash_amount']:,.0f}",
        f"  NSE Stock Amount (→ BL Optimizer): ₹{alloc['nse_stock_amount']:,.0f}",
        "",
        "ACTION PLAN",
    ] + [f"  ☐  {s}" for s in plan["action_plan"]] + [
        "",
        "INSIGHTS",
    ] + [f"  • {ins}" for ins in plan["insights"]] + [
        "",
        "=" * 62,
        "  PortfolioAI | Black-Litterman + LLaMA Views | NSE India",
        "  For Indian retail investors (₹20K–₹10L)",
        "=" * 62,
    ]

    st.download_button(
        label="📄 Download My Financial Plan",
        data="\n".join(summary_lines),
        file_name=f"financial_plan_{file_date}.txt",
        mime="text/plain",
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SETUP GATE — check data availability before rendering
# ══════════════════════════════════════════════════════════════════════════════

def render_setup_gate(data_status):
    st.error("⚠️  Some required data files are missing. Please run the setup pipeline first.")
    st.markdown("### Setup Pipeline")
    st.code(
        "# Step 1: Download market data + market caps + fundamentals\n"
        "python data_collector.py\n\n"
        "# Step 2: Score stocks on momentum / quality / volatility factors\n"
        "python scorer.py\n\n"
        "# Step 3: Generate LLM views (requires GROQ_API_KEY in .env)\n"
        "python llm_views.py\n\n"
        "# Step 4 (optional): Run historical backtest\n"
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
        page_title = "PortfolioAI — NSE India Portfolio Optimiser",
        page_icon  = "📈",
        layout     = "wide",
        initial_sidebar_state = "expanded",
    )

    # ── Global CSS ────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    /* ── Base ── */
    .main { background: #f8fafc; }
    header { visibility: hidden; }
    [data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }

    /* ── Finance Cards ── */
    .finance-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
    }

    /* ── Big Numbers ── */
    .big-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a202c;
        line-height: 1.1;
    }

    /* ── Text Colours ── */
    .text-green  { color: #16a34a; }
    .text-red    { color: #dc2626; }
    .text-blue   { color: #2563eb; }
    .text-orange { color: #ea580c; }
    .text-purple { color: #7c3aed; }
    .text-grey   { color: #64748b; }

    /* ── Step Indicators ── */
    .step-complete { color: #16a34a; font-weight: 600; }
    .step-active   { color: #2563eb; font-weight: 600; }
    .step-pending  { color: #94a3b8; }

    /* ── Buttons ── */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.75rem 2rem !important;
        transition: transform 0.1s !important;
    }
    .stButton > button:hover { transform: translateY(-2px) !important; }

    /* ── Insight Cards ── */
    .insight-card {
        background: #f0f9ff;
        border-left: 4px solid #0ea5e9;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        font-size: 1rem;
        color: #0c4a6e;
        line-height: 1.5;
    }
    .insight-card.good    { background:#f0fdf4; border-left-color:#16a34a; color:#14532d; }
    .insight-card.warning { background:#fffbeb; border-left-color:#f59e0b; color:#78350f; }
    .insight-card.great   { background:#fdf4ff; border-left-color:#a855f7; color:#581c87; }

    /* ── Quiz Radio as Cards ── */
    div[data-testid="stRadio"] label {
        display: flex !important;
        align-items: center !important;
        background: white !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 0.9rem 1.2rem !important;
        margin: 0.4rem 0 !important;
        cursor: pointer !important;
        transition: border-color 0.15s, background 0.15s !important;
    }
    div[data-testid="stRadio"] label:hover {
        border-color: #2563eb !important;
        background: #f0f9ff !important;
    }
    div[data-testid="stRadio"] label:has(input:checked) {
        border-color: #2563eb !important;
        background: #eff6ff !important;
        box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
    }

    /* ── Sidebar Nav Radio as Cards ── */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
        background: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
        padding: 0.65rem 0.9rem !important;
        margin: 0.2rem 0 !important;
        font-size: 0.93rem !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) {
        border-color: #2563eb !important;
        background: #eff6ff !important;
        color: #2563eb !important;
        font-weight: 700 !important;
        box-shadow: none !important;
    }

    /* ── Compatibility (keep existing classes working) ── */
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
        ("mode",               "planner"),  # financial planner is the entry point
        ("result",             None),
        ("run_count",          0),
        ("financial_plan",     None),
        ("risk_answers",       None),
        ("planner_step",       1),
        ("planned_investment", 0.0),
        ("quiz_q_index",       0),          # tracks current quiz question (0-4, 5=done)
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        # Logo & tagline
        st.markdown("""
        <div style="padding: 0.6rem 0 0.8rem;">
            <div style="font-size:1.45rem; font-weight:900; color:#1e293b; letter-spacing:-0.5px;">
                📊 PortfolioAI
            </div>
            <div style="color:#64748b; font-size:0.82rem; margin-top:3px;">
                Your AI money manager
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        # Navigation — styled radio buttons (CSS makes them look like cards)
        mode = st.radio(
            "Navigate",
            options=[
                "💰 Plan My Finances",
                "📈 Invest Now",
                "🔄 Rebalance",
                "📊 My Performance",
            ],
            index={"planner": 0, "fresh": 1, "rebalance": 2, "tracker": 3}.get(
                st.session_state.mode, 0
            ),
            label_visibility="collapsed",
        )

        _mode_map = {
            "💰 Plan My Finances": "planner",
            "📈 Invest Now":       "fresh",
            "🔄 Rebalance":       "rebalance",
            "📊 My Performance":  "tracker",
        }
        st.session_state.mode = _mode_map.get(mode, "planner")

        _mode_subtitles = {
            "💰 Plan My Finances": "Figure out how much to invest · 5 mins",
            "📈 Invest Now":       "AI picks the best stocks for you",
            "🔄 Rebalance":       "Adjust your existing holdings",
            "📊 My Performance":  "See your profit/loss and what to do next",
        }
        st.caption(_mode_subtitles.get(mode, ""))

        st.divider()

        risk_profile = st.select_slider(
            "**How do you feel about risk?**",
            options=["conservative", "moderate", "aggressive"],
            value="moderate",
            help="Conservative = play it safe. Moderate = balanced. Aggressive = go for higher returns, accept more ups and downs.",
        )

        risk_desc = {
            "conservative": "🛡️ Play it safe — steady, lower returns, fewer surprises",
            "moderate":     "⚖️ Balanced — good returns without too many sleepless nights",
            "aggressive":   "🚀 Go big — higher potential, but bigger swings too",
        }
        st.caption(risk_desc[risk_profile])

        st.divider()

        # ── AI Analysis Method selector ───────────────────────────────────────
        st.markdown("**🧠 AI Analysis Method**")
        analysis_method = st.selectbox(
            label="Choose method",
            options=["llm", "sentiment", "combined"],
            format_func=lambda x: {
                "llm":       "🚀 LLM Views — Fast (Groq/LLaMA)",
                "sentiment": "📰 News Sentiment (FinBERT)",
                "combined":  "🔥 Combined — Best Results",
            }[x],
            index=0,
            help=(
                "LLM Views uses recent price data. "
                "News Sentiment uses headlines. "
                "Combined uses both for best accuracy."
            ),
        )
        st.session_state.analysis_method = analysis_method

        _am_desc = {
            "llm":       "LLaMA analyses recent price patterns to predict returns. Fast and reliable.",
            "sentiment": "FinBERT reads recent news headlines for each stock. Needs sentiment data file.",
            "combined":  "Both signals combined. Best accuracy but takes slightly longer.",
        }
        st.caption(_am_desc[analysis_method])

        # ── Refresh data pipeline button ──────────────────────────────────────
        if st.button("🔄 Refresh Data", use_container_width=True,
                     help="Re-download prices and regenerate AI views for the selected method"):
            run_full_pipeline(analysis_method)
            st.cache_data.clear()
            st.rerun()

        st.divider()

        # ── Mode-specific inputs ──────────────────────────────────────────────
        if st.session_state.mode == "planner":
            st.markdown("**💰 Financial Planner**")
            st.caption("Complete your financial profile to get a personalised investment plan.")
            investment_inr = 0

        elif st.session_state.mode == "fresh":
            st.markdown("**💰 Investment Amount**")

            # Pre-fill from Financial Planner if user came via the connect button
            planned = float(st.session_state.get("planned_investment", 0.0))
            if planned > 0:
                st.success(f"💰 Amount from your financial plan")
                investment_inr = planned
                st.metric("Investment Amount", fmt_inr(investment_inr))
                if st.button("✏️ Change amount", key="fp_clear_planned"):
                    st.session_state.planned_investment = 0.0
                    st.rerun()
            else:
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

        elif st.session_state.mode == "tracker":
            st.markdown("**📊 Portfolio Tracker**")
            st.caption("Enter your holdings on the main page to see real P&L and next-month action signals.")
            investment_inr = 0
            additional_inr = 0

        else:  # rebalance
            st.markdown("**🔄 Enter Current Holdings**")
            st.caption("Enter your current portfolio values in ₹")
            investment_inr = 0
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
    enh_results, enh_metrics, enh_costs = load_enhanced_backtest()
    prices_inr          = load_prices_inr()

    # ── Financial Planner — runs without data files (no generic header) ─────────
    if st.session_state.mode == "planner":
        render_financial_planner()
        st.divider()
        st.markdown("""
        <div class="disclaimer">
        ⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only and does not
        constitute financial advice. Consult a SEBI-registered investment advisor (RIA) before
        making investment decisions. All projections are estimates based on inputs provided.
        </div>
        """, unsafe_allow_html=True)
        st.caption(
            "PortfolioAI | Personal Finance Planner + AI-powered NSE Portfolio Optimisation | "
            "For Indian retail investors"
        )
        return

    if not data_status["required_ok"]:
        render_setup_gate(data_status)
        return

    # ── Header for investment modes ───────────────────────────────────────────
    _n100_path = f"{DATA_DIR}/nifty100_prices.csv"
    _universe_label = "Top 100 Indian companies" if os.path.exists(_n100_path) else "16 Indian stocks"

    # ── Level 3: Macro Regime Panel — collapsed so it doesn't overwhelm ─────────
    macro_snap = load_macro_snapshot()
    if macro_snap:
        _r = macro_snap["regime"]
        _v = macro_snap["vix"]
        _tl_emoji = {"bull": "🟢", "neutral": "🟡", "bear": "🔴"}.get(_r.get("regime","neutral"), "🟡")
        with st.expander(f"{_tl_emoji} Market conditions today: **{_r.get('label','').title()}** · VIX {_v.get('vix',0):.1f} — click to see details"):
            render_macro_panel(macro_snap, key_prefix="top")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    #  FRESH INVESTMENT FLOW
    # ══════════════════════════════════════════════════════════════════════════
    if st.session_state.mode == "fresh":
        # Welcome message — different tone if coming from planner
        _planned = float(st.session_state.get("planned_investment", 0.0))
        if _planned > 0:
            st.markdown(f"""
            <div style="text-align:center; padding:0.5rem 0 1rem;">
                <h2 style="color:#1e293b; font-weight:800; margin:0;">
                    Perfect! Let's put your {fmt_inr(investment_inr, compact=True)} to work 🚀
                </h2>
                <p style="color:#64748b; margin:0.5rem 0 0;">
                    Our AI is about to analyse NSE stocks and build your personalised portfolio
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="text-align:center; padding:0.5rem 0 1rem;">
                <h2 style="color:#1e293b; font-weight:800; margin:0;">
                    AI Stock Portfolio — {fmt_inr(investment_inr)} 📈
                </h2>
                <p style="color:#64748b; margin:0.5rem 0 0;">
                    Set your amount in the sidebar, then let AI build your portfolio
                </p>
            </div>
            """, unsafe_allow_html=True)

        # What happens explanation + button
        st.markdown("""
        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:14px;
                    padding:1.2rem 1.5rem; margin-bottom:1rem;">
            <div style="font-weight:700; color:#1e293b; margin-bottom:0.6rem;">
                🧠 What happens when you click:
            </div>
            <div style="color:#64748b; font-size:0.9rem; line-height:1.9;">
                📰 &nbsp;AI reads the latest news for every stock on the list<br>
                📊 &nbsp;Checks which stocks look strong right now<br>
                🎯 &nbsp;Figures out the best mix for your risk level<br>
                ✅ &nbsp;Gives you a ready-to-execute plan with exact rupee amounts
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚀 Start AI Analysis →", type="primary", use_container_width=True):
            _am = st.session_state.get("analysis_method", "llm")
            with st.spinner("Reading the news, checking market signals, building your portfolio… ~30 seconds"):
                try:
                    result = run_fresh_optimizer(investment_inr, risk_profile, _am)
                    st.session_state.result    = result
                    st.session_state.run_count += 1
                    st.success("🎉 Your portfolio is ready! Scroll down to see your picks.")
                except Exception as e:
                    st.error(
                        f"Oops! Something went wrong 😅\n\n"
                        f"Details: {e}\n\n"
                        f"Try refreshing or running the setup pipeline again."
                    )
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
                st.markdown(f"""
                <div style="background:#fefce8; border:1px solid #fde68a; border-left:4px solid #ca8a04;
                            border-radius:0 10px 10px 0; padding:0.8rem 1.1rem; margin-bottom:0.5rem;
                            font-size:0.9rem; color:#92400e;">
                    {r_reg['emoji']} <strong>Markets are a bit cautious right now</strong> —
                    we're putting <strong>{macro_scale:.0%} of your money</strong> into stocks
                    and keeping <strong>{fmt_inr(investment_inr * macro_cash, compact=True)}</strong>
                    ({macro_cash:.0%}) as a safety buffer.
                    This is automatically adjusted as markets change.
                </div>
                """, unsafe_allow_html=True)

            # ── Friendly KPI row ──────────────────────────────────────────────
            st.markdown("#### Your portfolio at a glance")
            _fk1, _fk2, _fk3, _fk4 = st.columns(4)
            _fk1.markdown(f"""
            <div class="finance-card" style="text-align:center; border-top:4px solid #16a34a;">
                <div style="color:#64748b; font-size:0.82rem;">Money invested</div>
                <div style="font-size:1.5rem; font-weight:800; color:#16a34a; margin:0.3rem 0;">
                    {fmt_inr(summary["total_deployed_inr"], compact=True)}
                </div>
                <div style="color:#94a3b8; font-size:0.75rem;">across NSE stocks</div>
            </div>
            """, unsafe_allow_html=True)
            _fk2.markdown(f"""
            <div class="finance-card" style="text-align:center; border-top:4px solid #2563eb;">
                <div style="color:#64748b; font-size:0.82rem;">Expected yearly return</div>
                <div style="font-size:1.5rem; font-weight:800; color:#2563eb; margin:0.3rem 0;">
                    {summary['expected_return']:.1%}
                </div>
                <div style="color:#94a3b8; font-size:0.75rem;">based on historical data</div>
            </div>
            """, unsafe_allow_html=True)
            _fk3.markdown(f"""
            <div class="finance-card" style="text-align:center; border-top:4px solid #7c3aed;">
                <div style="color:#64748b; font-size:0.82rem;">Risk efficiency</div>
                <div style="font-size:1.5rem; font-weight:800; color:#7c3aed; margin:0.3rem 0;">
                    {summary['sharpe_ratio']:.2f}
                </div>
                <div style="color:#94a3b8; font-size:0.75rem;">Sharpe ratio — higher is better</div>
            </div>
            """, unsafe_allow_html=True)
            _cb_inr = summary.get("cash_buffer_inr", 0)
            _fk4.markdown(f"""
            <div class="finance-card" style="text-align:center; border-top:4px solid #f59e0b;">
                <div style="color:#64748b; font-size:0.82rem;">Cash buffer kept</div>
                <div style="font-size:1.5rem; font-weight:800; color:#f59e0b; margin:0.3rem 0;">
                    {fmt_inr(_cb_inr, compact=True)}
                </div>
                <div style="color:#94a3b8; font-size:0.75rem;">safety reserve</div>
            </div>
            """, unsafe_allow_html=True)

            st.divider()

            # ── Tabs ──────────────────────────────────────────────────────────
            t1, t2, t_factor, t3, t4, t5, t6 = st.tabs([
                "📊 Allocation", "🤖 AI Analysis", "📐 Stock Scorecard",
                "📉 Past Performance", "💡 Why these stocks?", "🌍 Market Conditions", "🔢 Details",
            ])

            with t1:
                # Header
                st.markdown(f"""
                <div style="text-align:center; padding:0.5rem 0 1.2rem;">
                    <h4 style="color:#1e293b; margin:0;">Your AI-picked portfolio this month 🎯</h4>
                    <p style="color:#64748b; font-size:0.9rem; margin:0.3rem 0 0;">
                        Total to invest: <strong style="color:#16a34a;">{fmt_inr(summary["total_deployed_inr"])}</strong>
                        &nbsp;·&nbsp; Expected return: <strong>{summary["expected_return"]:.1%} annually*</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Stock cards — color-coded by sentiment
                for _, _row in alloc.iterrows():
                    _t    = _row["ticker"]
                    _meta = STOCK_META.get(_t, {})
                    _sent = s_df.loc[_t, "label"] if _t in s_df.index else "neutral"
                    _cfg  = SENTIMENT_CONFIG.get(_sent, {})
                    _border_color = _cfg.get("color", "#94a3b8")
                    _ai_label = {
                        "bullish": "📈 Bullish",
                        "slightly_bullish": "📈 Slightly Bullish",
                        "neutral": "➡️ Neutral",
                        "slightly_bearish": "📉 Slightly Bearish",
                        "bearish": "📉 Bearish",
                    }.get(_sent, "➡️ Neutral")

                    _col_a, _col_b, _col_c, _col_d, _col_e = st.columns([3, 2, 1.5, 1.5, 1.5])
                    _col_a.markdown(f"""
                    <div style="border-left:4px solid {_border_color}; padding:0.7rem 1rem;
                                background:white; border-radius:0 10px 10px 0; margin:0.25rem 0;
                                box-shadow:0 1px 4px rgba(0,0,0,0.06);">
                        <div style="font-size:0.97rem; font-weight:700; color:#1e293b;">
                            {_meta.get("flag","📈")} {_meta.get("name", _t)}
                        </div>
                        <div style="color:#64748b; font-size:0.8rem; margin-top:1px;">
                            {_meta.get("sector","")} &nbsp;·&nbsp; {_t.replace(".NS","")}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    _col_b.metric("Buy", fmt_inr(_row["invested_inr"]))
                    _col_c.metric("Shares", f"~{_row['shares']:.1f}")
                    _col_d.metric("Weight", f"{_row['target_weight']:.1%}")
                    _col_e.metric("AI View", _ai_label)

                st.caption(f"*Based on historical backtest. Cash leftover after rounding: {fmt_inr(summary['cash_leftover_inr'])}")

                st.divider()
                # Charts
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

                # Full allocation table (reference)
                st.markdown("#### Full Allocation Table")
                display = []
                for _, row in alloc.iterrows():
                    t    = row["ticker"]
                    sent = s_df.loc[t, "label"] if t in s_df.index else "n/a"
                    cfg  = SENTIMENT_CONFIG.get(sent, {})
                    display.append({
                        "Stock":       f"{cfg.get('badge','⚪')} {t}",
                        "Company":     STOCK_META.get(t, {}).get("name", t),
                        "Sector":      STOCK_META.get(t, {}).get("sector", ""),
                        "Weight":      f"{row['target_weight']:.2%}",
                        "Amount":      fmt_inr(row["invested_inr"]),
                        "Shares":      f"{row['shares']:.4g}",
                        "Price/Share": fmt_inr(row["price_inr"]),
                        "AI Signal":   f"{cfg.get('icon','─')} {sent}",
                    })
                st.dataframe(pd.DataFrame(display), use_container_width=True, hide_index=True)

                st.markdown(
                    f'<div class="insight-card good">'
                    f'💡 Place buy orders for <strong>{fmt_inr(summary["total_deployed_inr"])}</strong> '
                    f'through your broker (Zerodha, Groww, Angel One, etc.) on the NSE. '
                    f'All prices are in ₹ — no conversion needed.'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            with t2:
                st.markdown("""
                <div style="padding:0.8rem 0 1.2rem;">
                    <h4 style="color:#1e293b; margin:0 0 0.3rem;">What is the AI reading right now? 🤖</h4>
                    <p style="color:#64748b; font-size:0.9rem; margin:0;">
                        Our AI reads recent news and price signals for each stock and gives it a label —
                        like <strong>Strong Buy</strong> or <strong>Caution</strong>.
                        This directly affects how much money goes into each stock.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # ── Show which analysis method was used ───────────────────────
                _used_method = st.session_state.get("analysis_method", "llm")
                _method_labels = {
                    "llm":       "🚀 LLM Views (Groq/LLaMA)",
                    "sentiment": "📰 News Sentiment (FinBERT)",
                    "combined":  "🔥 Combined (LLM + News)",
                }
                _method_descs = {
                    "llm":
                        "LLaMA analysed recent price patterns for each stock and predicted "
                        "whether it will go up or down.",
                    "sentiment":
                        "FinBERT read recent news headlines for each stock and assessed "
                        "whether the news is positive or negative.",
                    "combined":
                        "Both LLaMA price analysis AND news sentiment were combined. "
                        "When both agree the signal is stronger. "
                        "When they disagree confidence is lower.",
                }
                st.info(
                    f"**Analysis method used: {_method_labels.get(_used_method, _used_method)}**\n\n"
                    + _method_descs.get(_used_method, "")
                )

                if s_df is not None and not s_df.empty:
                    # Signal legend
                    st.markdown("""
                    <div style="display:flex; gap:0.6rem; flex-wrap:wrap; margin-bottom:1rem;">
                        <span style="background:#f0fdf4; color:#16a34a; border:1px solid #bbf7d0; padding:0.2rem 0.7rem; border-radius:20px; font-size:0.78rem; font-weight:600;">🚀 Strong Buy</span>
                        <span style="background:#f0fdf4; color:#22c55e; border:1px solid #bbf7d0; padding:0.2rem 0.7rem; border-radius:20px; font-size:0.78rem; font-weight:600;">📈 Buy</span>
                        <span style="background:#f8fafc; color:#64748b; border:1px solid #e2e8f0; padding:0.2rem 0.7rem; border-radius:20px; font-size:0.78rem; font-weight:600;">➡️ Hold</span>
                        <span style="background:#fff7ed; color:#ea580c; border:1px solid #fed7aa; padding:0.2rem 0.7rem; border-radius:20px; font-size:0.78rem; font-weight:600;">⚠️ Caution</span>
                        <span style="background:#fef2f2; color:#dc2626; border:1px solid #fecaca; padding:0.2rem 0.7rem; border-radius:20px; font-size:0.78rem; font-weight:600;">🚫 Avoid</span>
                    </div>
                    """, unsafe_allow_html=True)

                    _SIGNAL_MAP = {
                        "bullish":          ("🚀 Strong Buy",  "#16a34a", "#f0fdf4", "#bbf7d0", 100),
                        "slightly_bullish": ("📈 Buy",         "#22c55e", "#f0fdf4", "#dcfce7",  70),
                        "neutral":          ("➡️ Hold",        "#64748b", "#f8fafc", "#e2e8f0",  50),
                        "slightly_bearish": ("⚠️ Caution",     "#ea580c", "#fff7ed", "#fed7aa",  30),
                        "bearish":          ("🚫 Avoid",       "#dc2626", "#fef2f2", "#fecaca",  10),
                    }

                    for _ticker in s_df.index:
                        _srow   = s_df.loc[_ticker]
                        _label  = _srow["label"]
                        _meta   = STOCK_META.get(_ticker, {})
                        _sig_label, _sig_color, _sig_bg, _sig_border, _bar_pct = _SIGNAL_MAP.get(
                            _label, ("➡️ Hold", "#64748b", "#f8fafc", "#e2e8f0", 50)
                        )
                        _conf_pct = int(_srow["confidence"] * 100)
                        _headlines = int(_srow.get("num_headlines", 0))
                        _pct_pos   = int(_srow.get("pct_positive", 0) * 100)
                        _pct_neg   = int(_srow.get("pct_negative", 0) * 100)

                        # Plain-English confidence label
                        if _conf_pct >= 75:
                            _conf_label = "Very confident"
                        elif _conf_pct >= 55:
                            _conf_label = "Fairly confident"
                        else:
                            _conf_label = "Less certain"

                        st.markdown(f"""
                        <div style="background:white; border:1px solid #e2e8f0; border-left:4px solid {_sig_color};
                                    border-radius:10px; padding:0.85rem 1.1rem; margin:0.4rem 0;
                                    display:flex; align-items:center; gap:1rem; flex-wrap:wrap;">
                            <div style="min-width:160px;">
                                <div style="font-size:1rem; font-weight:700; color:#1e293b;">
                                    {_meta.get('flag','🏢')} {_ticker}
                                </div>
                                <div style="font-size:0.78rem; color:#64748b;">{_meta.get('name', _ticker)}</div>
                            </div>
                            <div style="flex:1; min-width:180px;">
                                <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.35rem;">
                                    <span style="background:{_sig_bg}; color:{_sig_color}; border:1px solid {_sig_border};
                                                 padding:0.15rem 0.65rem; border-radius:20px; font-size:0.8rem; font-weight:700;">
                                        {_sig_label}
                                    </span>
                                    <span style="color:#94a3b8; font-size:0.75rem;">{_conf_label} · {_headlines} headlines</span>
                                </div>
                                <div style="background:#f1f5f9; border-radius:6px; height:8px; width:100%; overflow:hidden;">
                                    <div style="background:{_sig_color}; width:{_bar_pct}%; height:100%; border-radius:6px;
                                                transition: width 0.6s ease;"></div>
                                </div>
                                <div style="display:flex; justify-content:space-between; margin-top:0.2rem;">
                                    <span style="font-size:0.7rem; color:#16a34a;">👍 {_pct_pos}% positive news</span>
                                    <span style="font-size:0.7rem; color:#dc2626;">👎 {_pct_neg}% negative</span>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("""
                    <div style="background:#f0f9ff; border:1px solid #bae6fd; border-radius:8px;
                                padding:0.7rem 1rem; margin-top:1rem; font-size:0.83rem; color:#0369a1;">
                        💡 <strong>How this affects your money:</strong>
                        Stocks marked <strong>Strong Buy</strong> get a bigger slice (min 8%).
                        Stocks marked <strong>Avoid</strong> get capped at 5% so your risk stays low.
                        Everything in between gets a balanced share.
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander("📊 See detailed numbers (for the curious)"):
                        sent_table = []
                        _is_combined = _used_method == "combined"
                        for ticker in s_df.index:
                            row  = s_df.loc[ticker]
                            cfg  = SENTIMENT_CONFIG.get(row["label"], {})
                            entry = {
                                "Stock":       f"{cfg.get('badge','⚪')} {ticker}",
                                "Company":     STOCK_META.get(ticker, {}).get("name", ticker),
                                "AI Score":    f"{row['final_score']:+.3f}",
                                "Signal":      f"{cfg.get('icon','─')} {row['label']}",
                                "Confidence":  f"{row['confidence']:.0%}",
                                "Headlines":   int(row["num_headlines"]),
                                "% Positive":  f"{row['pct_positive']:.0%}",
                                "% Negative":  f"{row['pct_negative']:.0%}",
                                "AI View":     f"{mu_bl[ticker]:+.2%}",
                                "Market Avg":  f"{mu_prior[ticker]:+.2%}",
                            }
                            if _is_combined:
                                llm_s  = row.get("llm_score",  float("nan"))
                                sent_s = row.get("sent_score", float("nan"))
                                agree  = row.get("signals_agree", float("nan"))
                                entry["LLM Score"]  = f"{llm_s:+.3f}"  if not pd.isna(llm_s)  else "n/a"
                                entry["News Score"] = f"{sent_s:+.3f}" if not pd.isna(sent_s) else "n/a"
                                entry["Agreement"]  = "✅ Agree" if agree == 1 else ("⚠️ Mixed" if agree == 0 else "n/a")
                            sent_table.append(entry)
                        st.dataframe(pd.DataFrame(sent_table), use_container_width=True, hide_index=True)

                else:
                    st.markdown("""
                    <div style="background:#fefce8; border:1px solid #fde68a; border-radius:10px;
                                padding:1rem 1.2rem; color:#92400e;">
                        📊 No AI signal data yet. Click <strong>🔄 Refresh Data</strong> in the sidebar,
                        then run the analysis again.
                    </div>
                    """, unsafe_allow_html=True)

            with t_factor:
                st.markdown("""
                <div style="padding:0.6rem 0 1rem;">
                    <h4 style="color:#1e293b; margin:0 0 0.3rem;">How strong is each stock? 📐</h4>
                    <p style="color:#64748b; font-size:0.9rem; margin:0;">
                        We score every stock on 3 things: <strong>momentum</strong> (is it trending up?),
                        <strong>quality</strong> (is the company financially healthy?),
                        and <strong>stability</strong> (how much does it jump around?).
                        Higher score = stronger stock.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                factor_df = feats.get("factor_scores")
                if factor_df is None:
                    factor_df = load_factor_scores_df()
                if factor_df is not None:
                    render_factor_scores_tab(factor_df, key_prefix="fresh_factor")
                else:
                    st.markdown("""
                    <div style="background:#fefce8; border:1px solid #fde68a; border-radius:10px;
                                padding:1rem 1.2rem; color:#92400e;">
                        📊 Stock scores not available yet — the data pipeline needs to run first.
                    </div>
                    """, unsafe_allow_html=True)

            with t3:
                st.markdown("""
                <div style="padding:0.6rem 0 1rem;">
                    <h4 style="color:#1e293b; margin:0 0 0.3rem;">How has this strategy done in the past? 📉</h4>
                    <p style="color:#64748b; font-size:0.9rem; margin:0;">
                        We ran this same AI strategy on historical data to see how it would have performed.
                        <em>Past results don't guarantee the future — but it's a useful reference.</em>
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Show enhanced backtest (from backtester.py) if available,
                # otherwise fall back to the legacy BL+Sentiment backtest
                if enh_results is not None and enh_metrics is not None:
                    render_enhanced_backtest_tab(enh_results, enh_metrics, enh_costs)
                elif backtest_df is not None and metrics_df is not None:
                    st.plotly_chart(chart_cumulative_return(backtest_df), use_container_width=True, key="fresh_backtest_chart")
                    fmt_cols = {"cumulative_ret": ".1%", "ann_return": ".2%",
                                "ann_vol": ".2%", "sharpe": ".2f",
                                "max_drawdown": ".2%", "calmar": ".2f"}
                    disp = metrics_df.copy()
                    for col, fmt in fmt_cols.items():
                        if col in disp.columns:
                            disp[col] = disp[col].apply(
                                lambda x: format(x, fmt) if pd.notna(x) else "n/a"
                            )
                    st.dataframe(disp, use_container_width=True, hide_index=True)
                else:
                    st.markdown("""
                    <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
                                padding:1.2rem 1.5rem; text-align:center; color:#64748b;">
                        📊 Historical performance data isn't available yet.
                        Run the backtester to generate it.
                    </div>
                    """, unsafe_allow_html=True)

            with t4:
                st.markdown("""
                <div style="padding:0.6rem 0 1.2rem;">
                    <h4 style="color:#1e293b; margin:0 0 0.3rem;">Why did the AI pick these stocks? 💡</h4>
                    <p style="color:#64748b; font-size:0.9rem; margin:0;">
                        Here's the plain-English reasoning behind each pick.
                        No jargon — just honest explanations you can actually understand.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                for _, row in alloc.iterrows():
                    t = row["ticker"]
                    if t not in s_df.index:
                        continue
                    sent_row = s_df.loc[t]
                    sent     = sent_row["label"]
                    cfg      = SENTIMENT_CONFIG.get(sent, {})
                    weight   = row["target_weight"]
                    _meta    = STOCK_META.get(t, {})
                    rationale= generate_rationale(
                        t, weight, row["invested_inr"],
                        sent_row, mu_bl[t], mu_prior[t], prices_inr,
                        use_openai, openai_key,
                    )

                    # Friendly signal label
                    _friendly_sent = {
                        "bullish":          "🚀 Strong Buy",
                        "slightly_bullish": "📈 Buy",
                        "neutral":          "➡️ Hold",
                        "slightly_bearish": "⚠️ Caution",
                        "bearish":          "🚫 Avoid",
                    }.get(sent, "➡️ Hold")

                    _border_c = cfg.get("color", "#94a3b8")
                    _amount   = fmt_inr(row["invested_inr"])
                    _weight_pct = f"{weight:.0%}"

                    # AI expected return vs market baseline
                    _ai_view   = mu_bl[t]
                    _mkt_view  = mu_prior[t]
                    _edge      = _ai_view - _mkt_view
                    _edge_text = (
                        f"AI expects <strong>{_ai_view:.1%}/yr</strong> — "
                        f"{'better' if _edge >= 0 else 'lower'} than market baseline of {_mkt_view:.1%}"
                    )

                    # 2–3 quick checkpoints
                    _chk1 = "✅ News sentiment is positive" if sent in ("bullish","slightly_bullish") else \
                            "⚠️ News is mixed — lower allocation" if sent == "neutral" else \
                            "🔴 News is negative — minimal allocation"
                    _chk2 = "✅ AI model sees above-average return potential" if _ai_view > _mkt_view else \
                            "⚪ Return potential is near market average"
                    _chk3 = f"✅ Gets {_weight_pct} of your portfolio ({_amount})"

                    with st.expander(
                        f"{_meta.get('flag','🏢')} {t} — {_meta.get('name',t)}  ·  {_friendly_sent}  ·  {_weight_pct}",
                        expanded=False,
                    ):
                        st.markdown(f"""
                        <div style="border-left:4px solid {_border_c}; padding:0.8rem 1.1rem;
                                    background:#f8fafc; border-radius:0 8px 8px 0; margin-bottom:0.8rem;">
                            {_chk1}<br>{_chk2}<br>{_chk3}
                            <div style="margin-top:0.6rem; color:#64748b; font-size:0.82rem;">{_edge_text}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(rationale)
                        with st.expander("🔢 Technical details"):
                            rc1, rc2, rc3, rc4 = st.columns(4)
                            rc1.metric("AI Score",      f"{sent_row['final_score']:+.3f}")
                            rc2.metric("Market Avg",    f"{mu_prior[t]:.2%}")
                            rc3.metric("AI Estimate",   f"{mu_bl[t]:.2%}")
                            rc4.metric("Weight",        f"{weight:.2%}")

            with t5:
                st.markdown("""
                <div style="padding:0.6rem 0 1.2rem;">
                    <h4 style="color:#1e293b; margin:0 0 0.3rem;">Is this a good time to invest? 🌍</h4>
                    <p style="color:#64748b; font-size:0.9rem; margin:0;">
                        We look at the overall market health — like a weather forecast for investing.
                        Your portfolio is automatically adjusted based on what we see here.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                if result_macro:
                    render_macro_panel(result_macro, sec_sent, key_prefix="fresh")
                else:
                    st.markdown("""
                    <div style="background:#fefce8; border:1px solid #fde68a; border-radius:10px;
                                padding:1rem 1.2rem; color:#92400e;">
                        📡 Market condition data not loaded yet. Run the app with prices.csv available.
                    </div>
                    """, unsafe_allow_html=True)

                if sec_sent:
                    st.markdown("#### Which sectors look good right now? 🏭")
                    _sec_cards_html = ""
                    for _sec, _val in sorted(sec_sent.items(), key=lambda x: -x[1]):
                        if _val > 0.15:
                            _sc_label, _sc_color, _sc_bg = "🟢 Bullish",  "#16a34a", "#f0fdf4"
                        elif _val > 0.03:
                            _sc_label, _sc_color, _sc_bg = "🟡 Positive", "#ca8a04", "#fefce8"
                        elif _val < -0.15:
                            _sc_label, _sc_color, _sc_bg = "🔴 Bearish",  "#dc2626", "#fef2f2"
                        elif _val < -0.03:
                            _sc_label, _sc_color, _sc_bg = "🟠 Caution",  "#ea580c", "#fff7ed"
                        else:
                            _sc_label, _sc_color, _sc_bg = "⚪ Neutral",  "#64748b", "#f8fafc"
                        _sec_cards_html += f"""
                        <div style="background:{_sc_bg}; border:1px solid #e2e8f0; border-radius:8px;
                                    padding:0.55rem 0.9rem; margin:0.3rem 0; display:flex;
                                    justify-content:space-between; align-items:center;">
                            <span style="font-weight:600; color:#1e293b; font-size:0.9rem;">{_sec}</span>
                            <span style="font-weight:700; color:{_sc_color}; font-size:0.85rem;">{_sc_label}</span>
                        </div>"""
                    st.markdown(_sec_cards_html, unsafe_allow_html=True)

                analyst_cons = feats.get("analyst_consensus")
                if analyst_cons is not None and not analyst_cons.empty:
                    st.markdown("#### What do professional analysts say? 👔")
                    st.caption("These are ratings from brokerage analysts — just one more data point we consider.")
                    ac_rows = []
                    for ticker in analyst_cons.index:
                        score = float(analyst_cons[ticker])
                        ac_rows.append({
                            "Stock":     ticker,
                            "Company":   STOCK_META.get(ticker, {}).get("name", ticker),
                            "Signal":    "🟢 Strong Buy" if score > 0.6 else
                                         "🟡 Buy" if score > 0.2 else
                                         "🔴 Sell" if score < -0.2 else
                                         "🟠 Strong Sell" if score < -0.6 else "⚪ Hold",
                        })
                    st.dataframe(pd.DataFrame(ac_rows), use_container_width=True, hide_index=True)

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
            st.markdown("""
            <div style="background:#f0f9ff; border:1px solid #bae6fd; border-radius:12px;
                        padding:1.2rem 1.5rem; text-align:center; margin:0.5rem 0 1.2rem;">
                <div style="font-size:1.8rem; margin-bottom:0.4rem;">👈</div>
                <div style="font-size:1rem; font-weight:700; color:#0369a1; margin-bottom:0.25rem;">
                    One step left — set your amount and click Invest!
                </div>
                <div style="font-size:0.85rem; color:#0284c7;">
                    Enter how much you want to invest in the sidebar on the left, then hit the button.
                    We'll build your personalised portfolio in seconds.
                </div>
            </div>
            """, unsafe_allow_html=True)
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
    #  PORTFOLIO TRACKER FLOW
    # ══════════════════════════════════════════════════════════════════════════
    elif st.session_state.mode == "tracker":
        if not data_status["required_ok"]:
            render_setup_gate(data_status)
            return
        render_portfolio_tracker()
        # Disclaimer already rendered inside render_portfolio_tracker
        st.caption(
            "PortfolioAI | Black-Litterman + LLaMA Views + Factor Scoring | "
            "16 NSE Equities | Macro Regime Overlay | Vol Targeting + Drawdown Control | "
            "Built for BTech Minor Project 2024–25"
        )
        return

    # ══════════════════════════════════════════════════════════════════════════
    #  PORTFOLIO REBALANCER FLOW
    # ══════════════════════════════════════════════════════════════════════════
    else:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#1e3a5f 0%,#1e40af 100%);
                    border-radius:16px; padding:1.5rem 2rem; margin-bottom:1.5rem; color:white;">
            <h2 style="margin:0 0 0.3rem; font-size:1.5rem;">Time to rebalance? Let's check 📊</h2>
            <p style="margin:0; opacity:0.85; font-size:0.95rem;">
                Tell us what you currently hold. We'll tell you exactly what to buy, sell, or keep —
                so your money stays optimally invested.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Holdings input table ──────────────────────────────────────────────
        st.markdown("#### Step 1 — What do you currently hold?")
        st.caption("Enter the ₹ value of each stock you hold today. Leave 0 if you don't own it.")

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

        st.markdown(f"""
        <div style="display:flex; gap:0.8rem; flex-wrap:wrap; margin:0.8rem 0 1rem;">
            <div style="flex:1; min-width:150px; background:white; border:1px solid #e2e8f0;
                        border-top:3px solid #3b82f6; border-radius:10px; padding:0.8rem 1rem; text-align:center;">
                <div style="font-size:0.75rem; color:#64748b; text-transform:uppercase; letter-spacing:0.04em;">You hold today</div>
                <div style="font-size:1.4rem; font-weight:800; color:#1e293b; margin:0.2rem 0;">{fmt_inr(total_current, compact=True)}</div>
            </div>
            <div style="flex:1; min-width:150px; background:white; border:1px solid #e2e8f0;
                        border-top:3px solid #16a34a; border-radius:10px; padding:0.8rem 1rem; text-align:center;">
                <div style="font-size:0.75rem; color:#64748b; text-transform:uppercase; letter-spacing:0.04em;">Adding now</div>
                <div style="font-size:1.4rem; font-weight:800; color:#16a34a; margin:0.2rem 0;">{fmt_inr(additional_inr, compact=True)}</div>
            </div>
            <div style="flex:1; min-width:150px; background:white; border:1px solid #e2e8f0;
                        border-top:3px solid #6366f1; border-radius:10px; padding:0.8rem 1rem; text-align:center;">
                <div style="font-size:0.75rem; color:#64748b; text-transform:uppercase; letter-spacing:0.04em;">Total to work with</div>
                <div style="font-size:1.4rem; font-weight:800; color:#6366f1; margin:0.2rem 0;">{fmt_inr(total_capital, compact=True)}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        if st.button("📊 Generate My Rebalancing Plan", type="primary",
                     use_container_width=True,
                     disabled=(total_capital < 1000)):
            if not current_holdings and additional_inr < 1000:
                st.warning("Please enter your current holdings or an additional investment amount.")
            else:
                with st.spinner("Crunching the numbers… building your rebalancing plan 📊"):
                    try:
                        result = run_rebalance_optimizer(
                            current_holdings, additional_inr, risk_profile,
                            st.session_state.get("analysis_method", "llm"),
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
                st.markdown(f"""
                <div style="background:#fefce8; border:1px solid #fde68a; border-left:4px solid #ca8a04;
                            border-radius:0 10px 10px 0; padding:0.8rem 1.1rem; margin-bottom:0.5rem;
                            font-size:0.9rem; color:#92400e;">
                    {r_reg['emoji']} <strong>Market caution applied</strong> —
                    {r_reg['label']} market × VIX {r_vix['vix']:.1f}
                    → only <strong>{macro_scale:.0%} of your money</strong> goes into stocks.
                    <strong>{fmt_inr(total_capital * macro_cash, compact=True)}</strong>
                    ({macro_cash:.0%}) is kept as cash for safety.
                </div>
                """, unsafe_allow_html=True)

            # ── Action KPIs ───────────────────────────────────────────────────
            buys  = rebal[rebal["action"] == "BUY"]
            sells = rebal[rebal["action"] == "SELL"]
            holds = rebal[rebal["action"] == "HOLD"]

            st.markdown(f"""
            <div style="display:flex; gap:0.8rem; flex-wrap:wrap; margin:0.5rem 0 1rem;">
                <div style="flex:1; min-width:150px; background:#f0fdf4; border:1px solid #bbf7d0;
                            border-left:4px solid #16a34a; border-radius:10px; padding:0.8rem 1rem;">
                    <div style="font-size:0.75rem; color:#16a34a; font-weight:700; text-transform:uppercase;">Buy</div>
                    <div style="font-size:1.5rem; font-weight:900; color:#1e293b;">{len(buys)} stocks</div>
                    <div style="font-size:0.82rem; color:#16a34a; font-weight:600;">{fmt_inr(summary['buys_inr'], compact=True)} to invest</div>
                </div>
                <div style="flex:1; min-width:150px; background:#fef2f2; border:1px solid #fecaca;
                            border-left:4px solid #dc2626; border-radius:10px; padding:0.8rem 1rem;">
                    <div style="font-size:0.75rem; color:#dc2626; font-weight:700; text-transform:uppercase;">Sell</div>
                    <div style="font-size:1.5rem; font-weight:900; color:#1e293b;">{len(sells)} stocks</div>
                    <div style="font-size:0.82rem; color:#dc2626; font-weight:600;">{fmt_inr(summary['sells_inr'], compact=True)} to free up</div>
                </div>
                <div style="flex:1; min-width:150px; background:#f8fafc; border:1px solid #e2e8f0;
                            border-left:4px solid #64748b; border-radius:10px; padding:0.8rem 1rem;">
                    <div style="font-size:0.75rem; color:#64748b; font-weight:700; text-transform:uppercase;">Hold</div>
                    <div style="font-size:1.5rem; font-weight:900; color:#1e293b;">{len(holds)} stocks</div>
                    <div style="font-size:0.82rem; color:#64748b;">no action needed</div>
                </div>
                <div style="flex:1; min-width:150px; background:white; border:1px solid #e2e8f0;
                            border-left:4px solid #f59e0b; border-radius:10px; padding:0.8rem 1rem;">
                    <div style="font-size:0.75rem; color:#92400e; font-weight:700; text-transform:uppercase;">Brokerage Cost</div>
                    <div style="font-size:1.5rem; font-weight:900; color:#1e293b;">{fmt_inr(summary['transaction_cost_inr'], compact=True)}</div>
                    <div style="font-size:0.82rem; color:#92400e;">est. Zerodha fees</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            rb1, rb2, rb3, rb_factor, rb4, rb5 = st.tabs([
                "📋 Rebalancing Plan", "📊 Portfolio View",
                "🤖 AI Analysis", "📐 Stock Scorecard", "💡 Why these stocks?", "🌍 Market Conditions",
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
                        "Action":         act,
                        "Stock":          f"{cfg.get('badge','⚪')} {t}",
                        "Company":        STOCK_META.get(t, {}).get("name", t),
                        "You have (₹)":   fmt_inr(row["current_inr"]),
                        "Target (₹)":     fmt_inr(row["target_inr"]),
                        "Buy/Sell (₹)":   fmt_inr(row["trade_inr"]) if act != "HOLD" else "─",
                        "No. of shares":  f"{row['shares_delta']:+.4g}" if act != "HOLD" else "─",
                        "Share price":    fmt_inr(row["price_inr"]),
                        "AI Signal":      f"{cfg.get('icon','─')} {row['sentiment']}",
                        "Goal %":         f"{row['target_weight']:.1%}",
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
                _total_avail = net_cash + summary["additional_inr"]
                st.markdown(f"""
                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px;
                            padding:1rem 1.2rem; margin-top:0.8rem;">
                    <div style="font-weight:700; color:#1e293b; margin-bottom:0.6rem; font-size:0.95rem;">
                        💸 Where the money goes
                    </div>
                    <div style="display:flex; flex-direction:column; gap:0.4rem; font-size:0.88rem;">
                        <div style="display:flex; justify-content:space-between;">
                            <span style="color:#64748b;">Money you'll get from selling</span>
                            <span style="font-weight:700; color:#16a34a;">{fmt_inr(summary['sells_inr'])}</span>
                        </div>
                        <div style="display:flex; justify-content:space-between;">
                            <span style="color:#64748b;">Extra money you're adding</span>
                            <span style="font-weight:700; color:#1e293b;">{fmt_inr(summary['additional_inr'])}</span>
                        </div>
                        <div style="border-top:1px dashed #e2e8f0; padding-top:0.4rem; display:flex; justify-content:space-between;">
                            <span style="color:#64748b;">Total available to buy</span>
                            <span style="font-weight:700; color:#1e293b;">{fmt_inr(_total_avail)}</span>
                        </div>
                        <div style="display:flex; justify-content:space-between;">
                            <span style="color:#64748b;">Total you'll spend buying</span>
                            <span style="font-weight:700; color:#3b82f6;">{fmt_inr(summary['buys_inr'])}</span>
                        </div>
                        <div style="display:flex; justify-content:space-between;">
                            <span style="color:#64748b;">Broker fees (Zerodha est.)</span>
                            <span style="font-weight:700; color:#92400e;">{fmt_inr(summary['transaction_cost_inr'])}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if summary["buys_inr"] > _total_avail * 1.05:
                    st.warning(
                        "⚠️ The stocks you're buying cost a bit more than what you'll have from selling. "
                        "Consider adding more money or the plan will adjust slightly."
                    )
                else:
                    st.success("✅ The sells fully cover the buys — no extra cash needed from you.")

            with rb2:
                st.caption("Left = what you have now. Right = what you'll have after rebalancing.")
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
                st.markdown("""
                <div style="padding:0.6rem 0 1rem;">
                    <h4 style="color:#1e293b; margin:0 0 0.3rem;">AI signal for each stock 🤖</h4>
                    <p style="color:#64748b; font-size:0.9rem; margin:0;">
                        Our AI reads recent news and market signals to decide which stocks
                        get a bigger allocation and which get capped.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                if sentiment_df is not None:
                    _SIGNAL_MAP_RB = {
                        "bullish":          ("🚀 Strong Buy",  "#16a34a", "#f0fdf4", "#bbf7d0", 100),
                        "slightly_bullish": ("📈 Buy",         "#22c55e", "#f0fdf4", "#dcfce7",  70),
                        "neutral":          ("➡️ Hold",        "#64748b", "#f8fafc", "#e2e8f0",  50),
                        "slightly_bearish": ("⚠️ Caution",     "#ea580c", "#fff7ed", "#fed7aa",  30),
                        "bearish":          ("🚫 Avoid",       "#dc2626", "#fef2f2", "#fecaca",  10),
                    }
                    for _ticker in s_df.index:
                        _srow = s_df.loc[_ticker]
                        _label = _srow["label"]
                        _meta  = STOCK_META.get(_ticker, {})
                        _sig_label, _sig_color, _sig_bg, _sig_border, _bar_pct = _SIGNAL_MAP_RB.get(
                            _label, ("➡️ Hold", "#64748b", "#f8fafc", "#e2e8f0", 50)
                        )
                        _conf_pct  = int(_srow["confidence"] * 100)
                        _headlines = int(_srow.get("num_headlines", 0))
                        _pct_pos   = int(_srow.get("pct_positive", 0) * 100)
                        _pct_neg   = int(_srow.get("pct_negative", 0) * 100)
                        _conf_label = "Very confident" if _conf_pct >= 75 else \
                                      "Fairly confident" if _conf_pct >= 55 else "Less certain"

                        st.markdown(f"""
                        <div style="background:white; border:1px solid #e2e8f0; border-left:4px solid {_sig_color};
                                    border-radius:10px; padding:0.85rem 1.1rem; margin:0.4rem 0;
                                    display:flex; align-items:center; gap:1rem; flex-wrap:wrap;">
                            <div style="min-width:160px;">
                                <div style="font-size:1rem; font-weight:700; color:#1e293b;">
                                    {_meta.get('flag','🏢')} {_ticker}
                                </div>
                                <div style="font-size:0.78rem; color:#64748b;">{_meta.get('name', _ticker)}</div>
                            </div>
                            <div style="flex:1; min-width:180px;">
                                <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.35rem;">
                                    <span style="background:{_sig_bg}; color:{_sig_color}; border:1px solid {_sig_border};
                                                 padding:0.15rem 0.65rem; border-radius:20px; font-size:0.8rem; font-weight:700;">
                                        {_sig_label}
                                    </span>
                                    <span style="color:#94a3b8; font-size:0.75rem;">{_conf_label} · {_headlines} headlines</span>
                                </div>
                                <div style="background:#f1f5f9; border-radius:6px; height:8px; width:100%; overflow:hidden;">
                                    <div style="background:{_sig_color}; width:{_bar_pct}%; height:100%; border-radius:6px;"></div>
                                </div>
                                <div style="display:flex; justify-content:space-between; margin-top:0.2rem;">
                                    <span style="font-size:0.7rem; color:#16a34a;">👍 {_pct_pos}% positive news</span>
                                    <span style="font-size:0.7rem; color:#dc2626;">👎 {_pct_neg}% negative</span>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("""
                    <div style="background:#f0f9ff; border:1px solid #bae6fd; border-radius:8px;
                                padding:0.7rem 1rem; margin-top:0.8rem; font-size:0.83rem; color:#0369a1;">
                        💡 <strong>How this affects your rebalance:</strong>
                        Stocks marked <em>Strong Buy</em> have a higher target weight.
                        Stocks marked <em>Avoid</em> are capped at 5%.
                    </div>
                    """, unsafe_allow_html=True)

            with rb_factor:
                st.markdown("""
                <div style="padding:0.6rem 0 1rem;">
                    <h4 style="color:#1e293b; margin:0 0 0.3rem;">How strong is each stock? 📐</h4>
                    <p style="color:#64748b; font-size:0.9rem; margin:0;">
                        We score every stock on <strong>momentum</strong> (trending up?),
                        <strong>quality</strong> (financially healthy?), and
                        <strong>stability</strong> (not too jumpy?). Higher = stronger.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                rb_factor_df = feats.get("factor_scores")
                if rb_factor_df is None:
                    rb_factor_df = load_factor_scores_df()
                if rb_factor_df is not None:
                    render_factor_scores_tab(rb_factor_df, key_prefix="rebal_factor")
                else:
                    st.markdown("""
                    <div style="background:#fefce8; border:1px solid #fde68a; border-radius:10px;
                                padding:1rem 1.2rem; color:#92400e;">
                        📊 Stock scores not available yet.
                    </div>
                    """, unsafe_allow_html=True)

            with rb4:
                st.markdown("""
                <div style="padding:0.6rem 0 1.2rem;">
                    <h4 style="color:#1e293b; margin:0 0 0.3rem;">Why these target positions? 💡</h4>
                    <p style="color:#64748b; font-size:0.9rem; margin:0;">
                        Here's why we're recommending each buy, sell, or hold — in plain language.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                for _, row in alloc.iterrows():
                    t = row["ticker"]
                    if t not in s_df.index:
                        continue
                    sent_row = s_df.loc[t]
                    sent     = sent_row["label"]
                    cfg      = SENTIMENT_CONFIG.get(sent, {})
                    weight   = row["target_weight"]
                    _meta_rb = STOCK_META.get(t, {})
                    reb_row  = rebal[rebal["ticker"] == t]
                    action   = reb_row["action"].values[0] if not reb_row.empty else "HOLD"
                    act_icon = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(action, "⚪")
                    _act_color = {"BUY": "#16a34a", "SELL": "#dc2626", "HOLD": "#64748b"}.get(action, "#64748b")
                    rationale = generate_rationale(
                        t, weight, row["invested_inr"], sent_row,
                        mu_bl[t], mu_prior[t], prices_inr, use_openai, openai_key,
                    )

                    _friendly_sent_rb = {
                        "bullish":          "🚀 Strong Buy",
                        "slightly_bullish": "📈 Buy",
                        "neutral":          "➡️ Hold",
                        "slightly_bearish": "⚠️ Caution",
                        "bearish":          "🚫 Avoid",
                    }.get(sent, "➡️ Hold")

                    _chk1_rb = "✅ Positive news sentiment" if sent in ("bullish","slightly_bullish") else \
                               "⚠️ Mixed news" if sent == "neutral" else "🔴 Negative news — reduced allocation"
                    _chk2_rb = "✅ AI expects above-average returns" if mu_bl[t] > mu_prior[t] else \
                               "⚪ Return near market average"
                    _chk3_rb = f"✅ Target: {weight:.0%} of portfolio"

                    with st.expander(
                        f"{act_icon} {action} — {_meta_rb.get('flag','🏢')} {t} "
                        f"— {_meta_rb.get('name',t)}  ·  {_friendly_sent_rb}",
                        expanded=False,
                    ):
                        st.markdown(f"""
                        <div style="border-left:4px solid {_act_color}; padding:0.8rem 1.1rem;
                                    background:#f8fafc; border-radius:0 8px 8px 0; margin-bottom:0.8rem;">
                            <div style="font-weight:700; color:{_act_color}; margin-bottom:0.4rem;">
                                {act_icon} Recommendation: {action}
                            </div>
                            {_chk1_rb}<br>{_chk2_rb}<br>{_chk3_rb}
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(rationale)
                        with st.expander("🔢 Technical details"):
                            rc1, rc2, rc3, rc4 = st.columns(4)
                            rc1.metric("Action",         action)
                            rc2.metric("AI Score",       f"{sent_row['final_score']:+.3f}")
                            rc3.metric("AI Estimate",    f"{mu_bl[t]:.2%}")
                            rc4.metric("Target Weight",  f"{weight:.2%}")

            with rb5:
                st.markdown("""
                <div style="padding:0.6rem 0 1rem;">
                    <h4 style="color:#1e293b; margin:0 0 0.3rem;">What does the market look like? 🌍</h4>
                    <p style="color:#64748b; font-size:0.9rem; margin:0;">
                        Your rebalance plan already accounts for current market conditions.
                        Here's what our AI is reading right now.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                if result_macro:
                    render_macro_panel(result_macro, sec_sent_rb, key_prefix="rebal")
                else:
                    st.markdown("""
                    <div style="background:#fefce8; border:1px solid #fde68a; border-radius:10px;
                                padding:1rem 1.2rem; color:#92400e;">
                        📡 Market condition data not loaded yet. Run the app with prices.csv available.
                    </div>
                    """, unsafe_allow_html=True)

                if sec_sent_rb:
                    st.markdown("#### Sector signals")
                    _sec_rb_html = ""
                    for _sec, _val in sorted(sec_sent_rb.items(), key=lambda x: -x[1]):
                        if _val > 0.15:
                            _sc_l, _sc_c, _sc_b = "🟢 Bullish",  "#16a34a", "#f0fdf4"
                        elif _val > 0.03:
                            _sc_l, _sc_c, _sc_b = "🟡 Positive", "#ca8a04", "#fefce8"
                        elif _val < -0.15:
                            _sc_l, _sc_c, _sc_b = "🔴 Bearish",  "#dc2626", "#fef2f2"
                        elif _val < -0.03:
                            _sc_l, _sc_c, _sc_b = "🟠 Caution",  "#ea580c", "#fff7ed"
                        else:
                            _sc_l, _sc_c, _sc_b = "⚪ Neutral",  "#64748b", "#f8fafc"
                        _sec_rb_html += f"""
                        <div style="background:{_sc_b}; border:1px solid #e2e8f0; border-radius:8px;
                                    padding:0.55rem 0.9rem; margin:0.3rem 0; display:flex;
                                    justify-content:space-between; align-items:center;">
                            <span style="font-weight:600; color:#1e293b; font-size:0.9rem;">{_sec}</span>
                            <span style="font-weight:700; color:{_sc_c}; font-size:0.85rem;">{_sc_l}</span>
                        </div>"""
                    st.markdown(_sec_rb_html, unsafe_allow_html=True)

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
        "PortfolioAI | Black-Litterman + LLaMA Views + Factor Scoring | "
        "Nifty 100 Universe | Macro Regime Overlay | Vol Targeting + Drawdown Control | "
        "For Indian retail investors (₹50K–₹10L)"
    )


if __name__ == "__main__":
    main()
