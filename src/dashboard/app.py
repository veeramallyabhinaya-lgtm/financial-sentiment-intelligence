"""
Financial Sentiment Intelligence Dashboard
Run with: streamlit run src/dashboard/app.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.storage.db import (
    get_company_sentiment_summary,
    get_recent_articles,
    get_recent_signals,
    get_sector_sentiment_timeseries,
)

SECTORS = ["Technology", "Finance", "Healthcare", "Energy", "Consumer", "Industrials", "Telecom", "Materials"]

SECTOR_COLORS = {
    "Technology":  "#6366f1",
    "Finance":     "#10b981",
    "Healthcare":  "#f59e0b",
    "Energy":      "#ef4444",
    "Consumer":    "#8b5cf6",
    "Industrials": "#06b6d4",
    "Telecom":     "#84cc16",
    "Materials":   "#f97316",
}

SIGNAL_EMOJI = {
    "reversal_up":    "📈",
    "reversal_down":  "📉",
    "spike_positive": "🚀",
    "spike_negative": "⚠️",
}

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Financial Sentiment Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 1.6rem; }
    .signal-card {
        background: #1e293b;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        border-left: 4px solid;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    st.title("📊 Sentiment Intel")
    st.caption("AI-Powered Financial Signal Tracker")
    st.divider()

    lookback = st.slider("Lookback window (hours)", 6, 72, 24, step=6)
    selected_sectors = st.multiselect("Filter sectors", SECTORS, default=SECTORS)
    min_articles = st.slider("Min articles per company", 1, 10, 2)

    st.divider()
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()
    st.caption(f"Showing last {lookback}h of data")

# ──────────────────────────────────────────────
# Data loading (cached)
# ──────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_data(lookback_h: int):
    articles  = get_recent_articles(hours=lookback_h)
    companies = get_company_sentiment_summary(hours=lookback_h)
    signals   = get_recent_signals(hours=lookback_h * 3)
    return articles, companies, signals


articles, companies, signals = load_data(lookback)

# ──────────────────────────────────────────────
# Header metrics
# ──────────────────────────────────────────────

st.title("📊 Financial Sentiment Intelligence")
st.caption("Real-time NLP pipeline · Groq LLaMA-3 · 45+ feeds · 8 sectors · 25+ companies")
st.divider()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("📰 Articles Ingested", len(articles))
col2.metric("🏢 Companies Tracked", len(companies))
col3.metric("🚨 Signals Detected", len(signals))

all_scores = [
    v["score"]
    for a in articles
    for v in a.get("sentiment_scores", {}).values()
    if isinstance(v, dict)
]
overall_sentiment = sum(all_scores) / len(all_scores) if all_scores else 0
col4.metric("📈 Market Sentiment", f"{overall_sentiment:+.2f}", delta=None)

processed_pct = sum(1 for a in articles if a.get("pipeline_ran")) / max(len(articles), 1) * 100
col5.metric("⚙️ Pipeline Coverage", f"{processed_pct:.0f}%")

st.divider()

# ──────────────────────────────────────────────
# Row 1: Sentiment heatmap + Signals
# ──────────────────────────────────────────────

left, right = st.columns([2, 1])

with left:
    st.subheader("🌡️ Sector Sentiment Heatmap")

    heatmap_data = []
    for sector in selected_sectors:
        ts = get_sector_sentiment_timeseries(sector, hours=lookback)
        for point in ts[-12:]:  # Last 12 hours
            heatmap_data.append({
                "Sector": sector,
                "Hour": point["hour"][-5:],  # HH:MM
                "Sentiment": point["avg_sentiment"],
                "Volume": point["volume"],
            })

    if heatmap_data:
        df_heat = pd.DataFrame(heatmap_data)
        df_pivot = df_heat.pivot_table(index="Sector", columns="Hour", values="Sentiment", aggfunc="mean")
        fig = px.imshow(
            df_pivot,
            color_continuous_scale="RdYlGn",
            zmin=-1, zmax=1,
            aspect="auto",
            labels={"color": "Sentiment"},
        )
        fig.update_layout(
            height=280,
            margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            coloraxis_colorbar=dict(tickvals=[-1, 0, 1], ticktext=["Bearish", "Neutral", "Bullish"]),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run the pipeline to populate heatmap data.")

with right:
    st.subheader("🚨 Active Signals")
    if signals:
        for sig in signals[:8]:
            emoji = SIGNAL_EMOJI.get(sig["direction"], "📊")
            direction_label = sig["direction"].replace("_", " ").title()
            st.markdown(f"""
**{emoji} {sig['name']}**
`{direction_label}` · Δ{sig['delta']:+.2f}
""")
            st.caption(f"Detected: {sig['detected_at'][:16].replace('T', ' ')}")
            st.divider()
    else:
        st.info("No signals detected yet in this window.")

# ──────────────────────────────────────────────
# Row 2: Company sentiment bar + Sector trend
# ──────────────────────────────────────────────

left2, right2 = st.columns([3, 2])

with left2:
    st.subheader("🏢 Company Sentiment Rankings")
    df_co = pd.DataFrame([c for c in companies if c["article_count"] >= min_articles])

    if not df_co.empty:
        df_co = df_co.nlargest(15, "avg_sentiment")
        df_co["color"] = df_co["avg_sentiment"].apply(lambda s: "#10b981" if s >= 0 else "#ef4444")
        fig2 = go.Figure(go.Bar(
            x=df_co["avg_sentiment"],
            y=df_co["company"],
            orientation="h",
            marker_color=df_co["color"],
            text=[f"{s:+.2f}" for s in df_co["avg_sentiment"]],
            textposition="outside",
        ))
        fig2.update_layout(
            height=360,
            xaxis=dict(range=[-1.1, 1.1], tickvals=[-1, -0.5, 0, 0.5, 1]),
            yaxis=dict(autorange="reversed"),
            margin=dict(l=0, r=40, t=20, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            shapes=[dict(type="line", x0=0, x1=0, y0=-0.5, y1=len(df_co) - 0.5,
                         line=dict(color="#475569", width=1, dash="dot"))],
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No company data with the current filters.")

with right2:
    st.subheader("📈 Sector Trend Lines")
    sector_pick = st.selectbox("Select sector", selected_sectors, key="trend_sector")
    ts_data = get_sector_sentiment_timeseries(sector_pick, hours=lookback)

    if ts_data:
        df_ts = pd.DataFrame(ts_data)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=df_ts["hour"], y=df_ts["avg_sentiment"],
            mode="lines+markers",
            line=dict(color=SECTOR_COLORS.get(sector_pick, "#6366f1"), width=2.5),
            fill="tozeroy",
            fillcolor=f"rgba(99,102,241,0.1)",
        ))
        fig3.add_hline(y=0, line_dash="dot", line_color="#475569")
        fig3.update_layout(
            height=300,
            yaxis=dict(range=[-1.1, 1.1]),
            margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            xaxis=dict(showticklabels=False),
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.caption(f"Avg sentiment over last {lookback}h for {sector_pick}")
    else:
        st.info("No timeseries data for this sector yet.")

# ──────────────────────────────────────────────
# Row 3: Recent articles feed
# ──────────────────────────────────────────────

st.divider()
st.subheader("📰 Latest Processed Articles")
cols = st.columns([2, 1, 1, 1])
cols[0].markdown("**Title**")
cols[1].markdown("**Source**")
cols[2].markdown("**Entities**")
cols[3].markdown("**Avg Sentiment**")

for article in articles[:20]:
    scores = article.get("sentiment_scores", {})
    score_vals = [v["score"] for v in scores.values() if isinstance(v, dict)]
    avg = sum(score_vals) / len(score_vals) if score_vals else None
    entities = article.get("entities", [])

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        st.markdown(f"[{article['title'][:75]}...]({article.get('url', '#')})", unsafe_allow_html=False)
    c2.caption(article.get("source", "—"))
    c3.caption(", ".join(entities[:2]) if entities else "—")
    if avg is not None:
        color = "🟢" if avg > 0.1 else ("🔴" if avg < -0.1 else "⚪")
        c4.caption(f"{color} {avg:+.2f}")
    else:
        c4.caption("—")
