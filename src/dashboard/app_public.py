"""
India Market Sentiment Intelligence — Public Dashboard (Streamlit Cloud)
Run: streamlit run src/dashboard/app_public.py
"""
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.storage.db import (
    get_company_sentiment_summary,
    get_recent_articles,
    get_recent_signals,
    get_sector_sentiment_timeseries,
)

SECTORS = ["Technology","Finance","Healthcare","Energy","Consumer","Industrials","Telecom","Materials"]
SECTOR_COLORS = {
    "Technology":"#3b82f6","Finance":"#06b6d4","Healthcare":"#8b5cf6",
    "Energy":"#f59e0b","Consumer":"#10b981","Industrials":"#6366f1",
    "Telecom":"#ec4899","Materials":"#84cc16",
}
SIGNAL_ICONS = {
    "reversal_up":   ("📈","#10b981","Sector Reversal ↑"),
    "reversal_down": ("📉","#ef4444","Sector Reversal ↓"),
    "spike_positive":("🚀","#3b82f6","Bullish Spike"),
    "spike_negative":("⚠️","#f59e0b","Bearish Spike"),
}

st.set_page_config(page_title="Sentiment Intel",page_icon="📊",layout="wide",initial_sidebar_state="collapsed")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');
*,*::before,*::after{box-sizing:border-box}
html,body,[data-testid="stAppViewContainer"]{background:#e8f4fd !important;font-family:'DM Sans',sans-serif !important}
[data-testid="stAppViewContainer"]{background:radial-gradient(ellipse at 8% 15%,rgba(147,197,253,.28) 0%,transparent 45%),radial-gradient(ellipse at 92% 85%,rgba(186,230,253,.22) 0%,transparent 45%),#e8f4fd !important}
[data-testid="collapsedControl"]{display:none !important}
header[data-testid="stHeader"]{background:transparent !important;box-shadow:none !important}
#MainMenu,footer{visibility:hidden}
[data-testid="stMainBlockContainer"]{padding:0 2rem 3rem !important;max-width:1420px !important;margin:0 auto !important}
h1,h2,h3,h4{font-family:'Syne',sans-serif !important;color:#0f2d5e !important}
.hero{background:linear-gradient(135deg,#1d4ed8 0%,#2563eb 45%,#0ea5e9 100%);border-radius:20px;padding:1.8rem 2.5rem;margin-bottom:1.25rem;position:relative;overflow:hidden}
.hero::before{content:'';position:absolute;top:-40%;right:-5%;width:380px;height:380px;background:radial-gradient(circle,rgba(255,255,255,.07) 0%,transparent 70%)}
.hero-title{font-family:'Syne',sans-serif;font-size:1.9rem;font-weight:800;color:#fff;margin:0 0 .25rem;letter-spacing:-.03em}
.hero-sub{font-family:'DM Sans',sans-serif;font-size:.82rem;color:rgba(255,255,255,.72);margin:0}
.metric-tile{background:rgba(255,255,255,.78);backdrop-filter:blur(8px);border:1px solid rgba(186,230,253,.7);border-radius:14px;padding:1rem 1.1rem;text-align:center;box-shadow:0 2px 12px rgba(37,99,235,.05)}
.metric-value{font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:700;color:#1d4ed8;line-height:1;margin-bottom:.2rem}
.metric-label{font-size:.68rem;color:#64748b;font-weight:500;text-transform:uppercase;letter-spacing:.08em}
.filter-bar{background:rgba(219,234,254,.55);backdrop-filter:blur(8px);border:1px solid rgba(147,197,253,.4);border-radius:16px;padding:1rem 1.75rem 1.1rem;margin-bottom:1.1rem}
.filter-label{font-family:'Syne',sans-serif;font-size:.67rem;font-weight:700;text-transform:uppercase;letter-spacing:.12em;color:#1d4ed8;margin-bottom:.35rem;display:block}
.card{background:rgba(255,255,255,.72);backdrop-filter:blur(12px);border:1px solid rgba(186,230,253,.55);border-radius:16px;padding:1.1rem 1.4rem;box-shadow:0 4px 20px rgba(37,99,235,.05);margin-bottom:.75rem}
.section-title{font-family:'Syne',sans-serif;font-size:.95rem;font-weight:700;color:#0f2d5e;margin:0 0 .7rem;letter-spacing:-.01em}
.signal-row{background:rgba(255,255,255,.55);border-radius:10px;padding:.6rem .85rem;margin-bottom:.45rem;border-left:3px solid}
.article-card{background:rgba(255,255,255,.72);border:1px solid rgba(186,230,253,.45);border-radius:12px;padding:.85rem 1.05rem;margin-bottom:.55rem;transition:box-shadow .18s ease,transform .18s ease}
.article-card:hover{box-shadow:0 5px 18px rgba(37,99,235,.1);transform:translateY(-1px)}
.article-title{font-family:'Syne',sans-serif;font-size:.83rem;font-weight:600;color:#0f2d5e;line-height:1.35;margin-bottom:.25rem}
.article-summary{font-size:.76rem;color:#475569;line-height:1.5;margin-bottom:.35rem}
.badge{display:inline-block;padding:.13rem .45rem;border-radius:5px;font-size:.63rem;font-weight:600}
.bp{background:#dcfce7;color:#15803d}
.bn{background:#fee2e2;color:#b91c1c}
.bz{background:#f1f5f9;color:#475569}
.bs{background:#dbeafe;color:#1d4ed8}
.be{background:#ede9fe;color:#6d28d9}
/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"]{background:rgba(219,234,254,.5) !important;border-radius:12px !important;padding:.25rem !important;gap:.25rem !important;border:1px solid rgba(147,197,253,.3) !important;width:100% !important;display:flex !important}
[data-testid="stTabs"] [data-baseweb="tab"]{border-radius:8px !important;font-family:'DM Sans',sans-serif !important;font-size:.82rem !important;font-weight:600 !important;flex:1 1 0 !important;text-align:center !important;justify-content:center !important;padding:.5rem .25rem !important;white-space:nowrap !important;color:#334155 !important;transition:background .15s ease !important}
[data-testid="stTabs"] [data-baseweb="tab"]:hover{background:rgba(37,99,235,.08) !important}
[data-testid="stTabs"] [aria-selected="true"]{background:#2563eb !important;color:white !important;box-shadow:0 2px 8px rgba(37,99,235,.25) !important}
[data-testid="stTabs"] [data-baseweb="tab-highlight"]{display:none !important}
[data-testid="stTabs"] [data-baseweb="tab-border"]{display:none !important}
/* ── Fix all dropdowns / selects (Streamlit Cloud dark override) ── */
[data-baseweb="select"] > div{background:rgba(255,255,255,.92) !important;border:1px solid rgba(147,197,253,.7) !important;border-radius:10px !important;color:#0f2d5e !important}
[data-baseweb="select"] span,[data-baseweb="select"] div{color:#0f2d5e !important}
[data-baseweb="popover"],[data-baseweb="menu"]{background:#ffffff !important;border:1px solid rgba(147,197,253,.5) !important;border-radius:10px !important}
[role="option"]{background:#ffffff !important;color:#0f2d5e !important}
[role="option"]:hover,[role="option"][aria-selected="true"]{background:#dbeafe !important;color:#1d4ed8 !important}
/* Radio */
[data-testid="stRadio"] label{color:#0f2d5e !important;font-family:'DM Sans',sans-serif !important;font-size:.82rem !important}
[data-testid="stRadio"] [data-baseweb="radio"] div[class]{border-color:#2563eb !important}
/* Multiselect tag */
[data-baseweb="tag"]{background:#dbeafe !important;border:1px solid #93c5fd !important;color:#1d4ed8 !important;border-radius:6px !important}
/* Slider */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"]{background:#2563eb !important;border-color:#2563eb !important}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def _chart(h=300, lm=160, rm=48, tm=32, bm=24):
    return dict(
        height=h, margin=dict(l=lm,r=rm,t=tm,b=bm),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#334155", size=11),
        xaxis=dict(gridcolor="rgba(147,197,253,.2)", showline=False, tickfont=dict(size=10)),
        yaxis=dict(gridcolor="rgba(147,197,253,.2)", showline=False, tickfont=dict(size=10)),
    )


@st.cache_data(ttl=300)
def _load(lb, ma):
    arts = get_recent_articles(hours=lb)
    cos  = [c for c in get_company_sentiment_summary(hours=lb) if c["article_count"] >= ma]
    sigs = get_recent_signals(hours=lb * 3)
    return arts, cos, sigs


# ── HERO ──────────────────────────────────────────────────────────────────────
HERO_SUB = "Real-time NLP analysis of Indian financial news &nbsp;·&nbsp; Nifty 50 constituents &nbsp;·&nbsp; 8 sectors &nbsp;·&nbsp; ET · Mint · Moneycontrol · Business Standard · RBI · SEBI"
st.markdown(f"""
<div class="hero">
  <div style="display:flex;align-items:center;gap:.75rem;margin-bottom:.4rem;">
    <span style="font-size:1.5rem;">📡</span>
    <p class="hero-title">India Market Sentiment Intelligence</p>
  </div>
  <p class="hero-sub">{HERO_SUB}</p>
</div>
""", unsafe_allow_html=True)

# ── FILTER BAR ────────────────────────────────────────────────────────────────
st.markdown('<div class="filter-bar">', unsafe_allow_html=True)
fc1, fc2, fc3, fc4 = st.columns([2,3,2,1])
with fc1:
    st.markdown('<span class="filter-label">⏱ Lookback window</span>', unsafe_allow_html=True)
    lookback = st.slider("lb", 6, 72, 24, step=6, label_visibility="collapsed")
with fc2:
    st.markdown('<span class="filter-label">🏭 Sectors</span>', unsafe_allow_html=True)
    sel_sectors = st.multiselect("sec", SECTORS, default=SECTORS, label_visibility="collapsed")
with fc3:
    st.markdown('<span class="filter-label">📰 Min articles / company</span>', unsafe_allow_html=True)
    min_art = st.slider("ma", 1, 10, 1, label_visibility="collapsed")
with fc4:
    st.markdown('<span class="filter-label">↻</span>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:.68rem;color:#475569;padding-top:.5rem;">Daily updated</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

articles, companies, signals = _load(lookback, min_art)

last_update = articles[0].get("fetched_at","")[:16].replace("T"," ") if articles else "—"
st.markdown(f"""
<div style="background:rgba(219,234,254,.5);border:1px solid rgba(147,197,253,.4);
     border-radius:10px;padding:.6rem 1.1rem;margin-bottom:.75rem;
     font-size:.75rem;color:#1e40af;display:flex;align-items:center;gap:.5rem;">
  <span>ℹ️</span>
  <span>Read-only analysis dashboard. Last updated: <b>{last_update} UTC</b>.
  View the full project on
  <a href="https://github.com/harshbokadia/financial-sentiment-intelligence"
     target="_blank" style="color:#1d4ed8;font-weight:600;">GitHub →</a></span>
</div>
""", unsafe_allow_html=True)

# ── METRICS ───────────────────────────────────────────────────────────────────
all_scores = [v["score"] for a in articles for v in a.get("sentiment_scores",{}).values() if isinstance(v,dict)]
overall = sum(all_scores)/len(all_scores) if all_scores else 0
pct = sum(1 for a in articles if a.get("pipeline_ran"))/max(len(articles),1)*100

for col,val,lbl in zip(
    st.columns(5),
    [len(articles),len(companies),len(signals),f"{overall:+.2f}",f"{pct:.0f}%"],
    ["Articles Ingested","Companies Tracked","Signals Detected","Market Sentiment","Pipeline Coverage"]
):
    col.markdown(f'<div class="metric-tile"><div class="metric-value">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4 = st.tabs(["🌡️ Heatmap & Signals","🏢 Company Rankings","📈 Sector Trends","📰 Article Feed"])

# ── TAB 1: HEATMAP + SIGNALS ──────────────────────────────────────────────────
with t1:
    col_h, col_s = st.columns([3,2])
    with col_h:
        st.markdown('<div class="card"><p class="section-title">🌡️ Sector Sentiment Heatmap</p>', unsafe_allow_html=True)
        hd = []
        for s in sel_sectors:
            for pt in get_sector_sentiment_timeseries(s, hours=lookback)[-16:]:
                hd.append({"Sector":s, "Hour":pt["hour"][5:13].replace("T"," "), "Sentiment":pt["avg_sentiment"]})
        if hd:
            piv = pd.DataFrame(hd).pivot_table(index="Sector", columns="Hour", values="Sentiment", aggfunc="mean", fill_value=0)
            fh = px.imshow(piv, color_continuous_scale=[[0,"#fca5a5"],[.5,"#e0f2fe"],[1,"#34d399"]], zmin=-1, zmax=1, aspect="auto")
            fh.update_layout(
                height=300, margin=dict(l=140,r=95,t=28,b=52),
                paper_bgcolor="rgba(255,255,255,0)", plot_bgcolor="rgba(255,255,255,0)",
                font=dict(family="DM Sans, sans-serif", color="#1e293b", size=12),
                xaxis=dict(
                    title=dict(text="Hour (MM-DD HH)", font=dict(size=11, color="#1e293b")),
                    tickfont=dict(size=10, color="#1e293b"),
                    tickangle=-30, showline=False,
                ),
                yaxis=dict(
                    tickfont=dict(size=12, color="#1e293b"),
                    showline=False,
                ),
                coloraxis_colorbar=dict(
                    tickvals=[-1,0,1], ticktext=["Bearish","Neutral","Bullish"],
                    len=0.85, thickness=14, outlinewidth=0,
                    tickfont=dict(size=11, color="#1e293b"), title="",
                )
            )
            fh.update_traces(hovertemplate="<b>%{y}</b><br>%{x}<br>Sentiment: %{z:.2f}<extra></extra>")
            st.plotly_chart(fh, use_container_width=True, config={"displayModeBar":False})
        else:
            st.info("Run the pipeline to populate heatmap data.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_s:
        st.markdown('<div class="card"><p class="section-title">🚨 Active Signals</p>', unsafe_allow_html=True)
        if signals:
            for sig in signals[:10]:
                icon, color, lbl = SIGNAL_ICONS.get(sig["direction"],("📊","#6b7280",sig["direction"]))
                det = sig["detected_at"][:16].replace("T"," ")
                st.markdown(f"""
                <div class="signal-row" style="border-left-color:{color};">
                  <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                    <div>
                      <span style="font-family:'Syne',sans-serif;font-size:.8rem;font-weight:700;color:#0f2d5e;">{icon} {sig['name']}</span>
                      <div style="font-size:.67rem;color:{color};font-weight:600;margin-top:.08rem;">{lbl} &nbsp;·&nbsp; Δ{sig['delta']:+.2f}</div>
                    </div>
                    <span style="font-size:.6rem;color:#94a3b8;">{det}</span>
                  </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No signals in this window yet.")
        st.markdown('</div>', unsafe_allow_html=True)

# ── TAB 2: COMPANY RANKINGS ───────────────────────────────────────────────────
with t2:
    st.markdown('<div class="card"><p class="section-title">🏢 Company Sentiment Rankings</p>', unsafe_allow_html=True)
    r1, r2 = st.columns([2,2])
    with r1: sort_by = st.selectbox("Sort by", ["Absolute Score","Most Bullish","Most Bearish","Most Coverage"])
    with r2: top_n   = st.slider("Top N companies", 5, 30, 15)
    df_co = pd.DataFrame(companies)
    if not df_co.empty:
        if sort_by == "Most Bullish":    df_co = df_co.nlargest(top_n, "avg_sentiment")
        elif sort_by == "Most Bearish":  df_co = df_co.nsmallest(top_n, "avg_sentiment")
        elif sort_by == "Most Coverage": df_co = df_co.nlargest(top_n, "article_count")
        else:                            df_co = df_co.reindex(df_co["avg_sentiment"].abs().nlargest(top_n).index)
        df_co["clr"] = df_co["avg_sentiment"].apply(lambda s:"#10b981" if s>=.1 else("#ef4444" if s<=-.1 else"#94a3b8"))
        fig2 = go.Figure(go.Bar(
            x=df_co["avg_sentiment"], y=df_co["company"], orientation="h",
            marker=dict(color=df_co["clr"], opacity=0.85, line=dict(width=0)),
            text=[f"{s:+.2f}" for s in df_co["avg_sentiment"]],
            textposition="outside", textfont=dict(size=10, color="#334155"),
            customdata=df_co[["article_count"]].values,
            hovertemplate="<b>%{y}</b><br>Sentiment: %{x:.3f}<br>Articles: %{customdata[0]}<extra></extra>",
        ))
        fig2.update_layout(
            height=max(320, len(df_co)*30),
            margin=dict(l=185, r=65, t=32, b=24),
            paper_bgcolor="rgba(255,255,255,0)", plot_bgcolor="rgba(255,255,255,0)",
            font=dict(family="DM Sans, sans-serif", color="#1e293b", size=12),
            xaxis=dict(
                range=[-1.18,1.18], tickvals=[-1,-.5,0,.5,1],
                ticktext=["-1.0","-0.5","0","+0.5","+1.0"],
                gridcolor="rgba(147,197,253,.25)", showline=False,
                tickfont=dict(size=11, color="#1e293b"),
            ),
            yaxis=dict(
                autorange="reversed", showline=False,
                gridcolor="rgba(147,197,253,.25)",
                tickfont=dict(size=12, color="#1e293b"),
            ),
            shapes=[dict(type="line",x0=0,x1=0,y0=-.5,y1=len(df_co)-.5,
                         line=dict(color="#93c5fd",width=1,dash="dot"))],
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})
    else:
        st.info("No company data with current filters.")
    st.markdown('</div>', unsafe_allow_html=True)

# ── TAB 3: SECTOR TRENDS ──────────────────────────────────────────────────────
with t3:
    st.markdown('<div class="card"><p class="section-title">📈 Sector Sentiment Over Time</p>', unsafe_allow_html=True)
    vmode = st.radio("View", ["Single sector","All sectors overlay"], horizontal=True)
    if vmode == "Single sector":
        sp = st.selectbox("Sector", sel_sectors or SECTORS)
        ts = get_sector_sentiment_timeseries(sp, hours=lookback)
        if ts:
            df_ts = pd.DataFrame(ts)
            clr = SECTOR_COLORS.get(sp,"#2563eb")
            rv,gv,bv = int(clr[1:3],16),int(clr[3:5],16),int(clr[5:7],16)
            f3 = go.Figure()
            f3.add_trace(go.Scatter(
                x=df_ts["hour"], y=df_ts["avg_sentiment"],
                mode="lines+markers", name=sp,
                line=dict(color=clr,width=2.5,shape="spline"),
                marker=dict(size=5,color=clr),
                fill="tozeroy", fillcolor=f"rgba({rv},{gv},{bv},.07)",
                hovertemplate="%{x}<br>Sentiment: %{y:.3f}<extra></extra>",
            ))
            f3.add_hline(y=0, line_dash="dot", line_color="#93c5fd", line_width=1)
            f3.update_layout(
                height=340, margin=dict(l=80,r=28,t=32,b=72),
                paper_bgcolor="rgba(255,255,255,0)", plot_bgcolor="rgba(255,255,255,0)",
                font=dict(family="DM Sans, sans-serif", color="#1e293b", size=12),
                xaxis=dict(
                    showticklabels=True, tickangle=-30,
                    tickfont=dict(size=10, color="#1e293b"),
                    gridcolor="rgba(147,197,253,.25)", showline=False,
                ),
                yaxis=dict(
                    range=[-1.1,1.1], tickvals=[-1,-.5,0,.5,1],
                    ticktext=["-1.0","-0.5","0","+0.5","+1.0"],
                    title=dict(text="Sentiment", font=dict(size=11, color="#1e293b")),
                    gridcolor="rgba(147,197,253,.25)", showline=False,
                    tickfont=dict(size=11, color="#1e293b"),
                ),
            )
            st.plotly_chart(f3, use_container_width=True, config={"displayModeBar":False})
        else:
            st.info("No data for this sector yet.")
    else:
        fa = go.Figure()
        has_d = False
        for s in (sel_sectors or SECTORS):
            ts = get_sector_sentiment_timeseries(s, hours=lookback)
            if not ts: continue
            has_d = True
            df_ts = pd.DataFrame(ts)
            clr = SECTOR_COLORS.get(s,"#2563eb")
            fa.add_trace(go.Scatter(
                x=df_ts["hour"], y=df_ts["avg_sentiment"],
                mode="lines", name=s,
                line=dict(color=clr,width=2,shape="spline"),
                hovertemplate=f"<b>{s}</b><br>%{{x}}<br>%{{y:.3f}}<extra></extra>",
            ))
        if has_d:
            fa.add_hline(y=0, line_dash="dot", line_color="#93c5fd", line_width=1)
            fa.update_layout(
                height=400, margin=dict(l=80,r=28,t=32,b=100),
                paper_bgcolor="rgba(255,255,255,0)", plot_bgcolor="rgba(255,255,255,0)",
                font=dict(family="DM Sans, sans-serif", color="#1e293b", size=12),
                xaxis=dict(
                    showticklabels=True, tickangle=-30,
                    tickfont=dict(size=10, color="#1e293b"),
                    gridcolor="rgba(147,197,253,.25)", showline=False,
                ),
                yaxis=dict(
                    range=[-1.1,1.1], tickvals=[-1,-.5,0,.5,1],
                    ticktext=["-1.0","-0.5","0","+0.5","+1.0"],
                    title=dict(text="Sentiment", font=dict(size=11, color="#1e293b")),
                    gridcolor="rgba(147,197,253,.25)", showline=False,
                    tickfont=dict(size=11, color="#1e293b"),
                ),
                legend=dict(
                    orientation="h", y=-0.22,
                    font=dict(size=11, color="#1e293b"),
                    bgcolor="rgba(255,255,255,.7)",
                    bordercolor="rgba(147,197,253,.5)", borderwidth=1,
                ),
            )
            st.plotly_chart(fa, use_container_width=True, config={"displayModeBar":False})
        else:
            st.info("No timeseries data in this window.")
    st.markdown('</div>', unsafe_allow_html=True)

# ── TAB 4: ARTICLE FEED ───────────────────────────────────────────────────────
with t4:
    af1, af2, af3 = st.columns(3)
    with af1: src_f = st.selectbox("Source", ["All"]+sorted(set(a.get("source","") for a in articles)))
    with af2: sen_f = st.selectbox("Sentiment", ["All","Positive","Negative","Neutral"])
    with af3: sec_f = st.selectbox("Sector", ["All"]+SECTORS)
    show_n = st.slider("Articles to show", 5, 50, 20)

    def _avg(a):
        s = [v["score"] for v in a.get("sentiment_scores",{}).values() if isinstance(v,dict)]
        return sum(s)/len(s) if s else None
    def _lbl(a):
        v = _avg(a)
        if v is None: return "Neutral"
        return "Positive" if v>.1 else ("Negative" if v<-.1 else "Neutral")

    filt = articles
    if src_f != "All": filt = [a for a in filt if a.get("source")==src_f]
    if sec_f != "All": filt = [a for a in filt if sec_f in a.get("sectors",[])]
    if sen_f != "All": filt = [a for a in filt if _lbl(a)==sen_f]

    st.caption(f"Showing {min(show_n,len(filt))} of {len(filt)} articles")

    for art in filt[:show_n]:
        avg  = _avg(art)
        ents = art.get("entities",[])
        summ = art.get("summary","")
        pub  = art.get("published_at","")[:16].replace("T"," ")

        if avg is None:   sb,sc = '<span class="badge bz">Unscored</span>',"—"
        elif avg>.1:      sb,sc = '<span class="badge bp">● Positive</span>',f"{avg:+.2f}"
        elif avg<-.1:     sb,sc = '<span class="badge bn">● Negative</span>',f"{avg:+.2f}"
        else:             sb,sc = '<span class="badge bz">● Neutral</span>',f"{avg:+.2f}"

        eb = " ".join(f'<span class="badge be">{e}</span>' for e in ents[:3])
        sr = f'<span class="badge bs">{art.get("source","")}</span>'
        su = f'<div class="article-summary">{summ}</div>' if summ else ""

        nt = art.get("news_type","")
        nt_map = {"sector_wide":("🌐 Sector","#0369a1","#e0f2fe"),"macro":("📊 Macro","#7c3aed","#ede9fe"),"company_specific":("🏢 Company","#065f46","#dcfce7")}
        nt_label,nt_color,nt_bg = nt_map.get(nt,("","#6b7280","#f1f5f9"))
        nt_badge = f'<span style="background:{nt_bg};color:{nt_color};padding:.13rem .45rem;border-radius:5px;font-size:.62rem;font-weight:600;">{nt_label}</span>' if nt_label else ""
        imp = art.get("news_importance",None)
        imp_badge = f'<span style="background:#fef3c7;color:#92400e;padding:.13rem .4rem;border-radius:5px;font-size:.62rem;font-weight:600;">⚡ {imp:.2f}</span>' if imp else ""

        st.markdown(f"""
        <div class="article-card">
          <div class="article-title"><a href="{art.get('url','#')}" target="_blank" style="color:#0f2d5e;text-decoration:none;">{art.get('title','Untitled')}</a></div>
          {su}
          <div style="display:flex;align-items:center;gap:.4rem;flex-wrap:wrap;font-size:.65rem;color:#94a3b8;">
            {sr} {sb} {nt_badge} {imp_badge} {eb}
            <span style="margin-left:auto;">{pub} &nbsp;·&nbsp; {sc}</span>
          </div>
        </div>""", unsafe_allow_html=True)

    if not filt:
        st.info("No articles match the current filters.")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:1.5rem 0 .5rem;font-size:.68rem;color:#94a3b8;font-family:'DM Sans',sans-serif;">
  India Market Sentiment Intelligence &nbsp;·&nbsp;
  Built by <a href="https://linkedin.com/in/-harsh-bokadia/" target="_blank" style="color:#2563eb;text-decoration:none;font-weight:600;">Harsh Bokadia</a>
  &nbsp;·&nbsp;
  <a href="https://github.com/harshbokadia/financial-sentiment-intelligence" target="_blank" style="color:#2563eb;text-decoration:none;">View on GitHub</a>
  &nbsp;·&nbsp; Data updated daily
</div>
""", unsafe_allow_html=True)