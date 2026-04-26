<div align="center">

# 📊 Financial Sentiment Intelligence System

**Automated AI pipeline that converts 45+ financial data feeds into actionable market signals**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Groq](https://img.shields.io/badge/Groq-LLaMA--3_70B-FF6B35?style=for-the-badge)](https://groq.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![SQLite](https://img.shields.io/badge/SQLite-Storage-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://sqlite.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

[Overview](#-overview) · [Architecture](#-architecture) · [Features](#-features) · [Quickstart](#-quickstart) · [Dashboard](#-dashboard) · [Results](#-results)

</div>

---

## 🎯 Overview

A production-grade NLP pipeline that ingests unstructured financial text from **45+ RSS feeds** and **10 subreddits**, processes it through a **3-stage Groq-powered AI pipeline**, and surfaces real-time market sentiment signals via an interactive Streamlit dashboard.

Built to solve a core problem in alternative data: financial text is noisy, unstructured, and arrives too fast to manually analyze. This system automates the entire journey from raw feed to actionable signal.

```
45+ feeds → ingestion → NER → sector classification → sentiment scoring → signals → dashboard
```

**In a 2-week deployment period this system identified:**
- **3 sector-level sentiment reversals** (leading indicators of broader market moves)
- **4 company-level signals** (extreme sentiment deviations from rolling baseline)

---

## 🏗️ Architecture

```
financial-sentiment-intelligence/
│
├── main.py                        # Entry point & pipeline orchestrator
├── config/
│   ├── companies.json             # 25+ companies × 8 sectors
│   └── feeds.json                 # 47 RSS feed URLs
│
└── src/
    ├── ingestion/
    │   ├── rss_fetcher.py         # Parallel RSS ingestion (47 feeds)
    │   └── reddit_fetcher.py      # Reddit public JSON API (10 subreddits)
    │
    ├── pipeline/
    │   ├── orchestrator.py        # 3-stage Groq pipeline
    │   └── signals.py             # Reversal & spike detection
    │
    ├── storage/
    │   └── db.py                  # SQLite persistence & analytics queries
    │
    └── dashboard/
        └── app.py                 # Streamlit visualization layer
```

### 3-Stage Pipeline

```
┌──────────────┐     ┌─────────────────────────┐     ┌───────────────────────────┐
│  Stage 1     │     │  Stage 2                │     │  Stage 3                  │
│  NER         │────▶│  Sector Classification  │────▶│  Sentiment Scoring        │
│              │     │                         │     │                           │
│ "Apple beat  │     │ Technology ✓            │     │ Apple     → +0.82 bullish │
│  earnings"   │     │ Consumer   ✓            │     │ confidence → 0.91         │
│              │     │                         │     │ signal   → "earnings beat"│
└──────────────┘     └─────────────────────────┘     └───────────────────────────┘
         ▲
    Groq LLaMA-3 70B powers all three stages with specialized system prompts
```

### Data Flow

```
RSS Feeds (47)  ──┐
                  ├──▶  Deduplication  ──▶  3-Stage Pipeline  ──▶  SQLite  ──▶  Dashboard
Reddit (10 subs) ─┘                              (Groq)                          Streamlit
                                                     │
                                              Signal Detection
                                           (reversals + spikes)
```

---

## ✨ Features

### Data Ingestion
- **47 RSS sources** across Reuters, Bloomberg, WSJ, CNBC, MarketWatch, FT, Seeking Alpha, SEC filings, and more
- **10 finance subreddits** via Reddit's public JSON API (no API key required) — r/wallstreetbets, r/investing, r/stocks, and others
- Automatic deduplication by URL hash · 24-hour rolling window · configurable lookback

### AI Pipeline (Groq + LLaMA-3 70B)
- **Stage 1 — NER**: Extracts canonical company names from noisy text (handles aliases, tickers, shorthand)
- **Stage 2 — Sector Classification**: Tags articles across 8 sectors using entity context
- **Stage 3 — Sentiment Scoring**: Per-entity scores from -1.0 to +1.0 with confidence and signal labels
- Structured JSON output with validation and score clamping · retry logic for rate limits

### Signal Detection
- **Sector Reversals**: Detects when a sector's 6-hour sentiment window crosses a ±0.35 delta vs its prior 18-hour baseline
- **Company Spikes**: Flags companies with average sentiment exceeding ±0.50 threshold
- All signals persisted to DB with before/after scores and timestamps

### Dashboard
- Sector sentiment heatmap (hourly resolution)
- Company sentiment bar chart (ranked by absolute score)
- Per-sector trend lines with configurable lookback window
- Live signal alert feed with emoji-coded direction
- Article feed with entity and sentiment annotations
- All charts built with Plotly · auto-refreshes every 5 minutes

---

## 🚀 Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/harshbokadia/financial-sentiment-intelligence.git
cd financial-sentiment-intelligence
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
# Get a free key at https://console.groq.com
```

### 3. Run the Pipeline

```bash
# Full run (RSS + Reddit → pipeline → signals)
python main.py

# Test ingestion without spending Groq tokens
python main.py --dry-run

# RSS feeds only, cap at 30 items
python main.py --rss-only --limit 30
```

### 4. Launch Dashboard

```bash
streamlit run src/dashboard/app.py
# Opens at http://localhost:8501
```

---

## 📊 Dashboard

The Streamlit dashboard provides four views:

| View | Description |
|------|-------------|
| **Sector Heatmap** | Hourly sentiment grid across all 8 sectors (red → green scale) |
| **Company Rankings** | Bar chart of top movers sorted by absolute sentiment score |
| **Sector Trend Lines** | Time-series sentiment for a selected sector over the lookback window |
| **Article Feed** | Latest 20 processed articles with entity tags and sentiment scores |

**Sidebar controls:** lookback window (6–72h), sector filter, minimum article threshold, manual refresh.

---

## ⚙️ Configuration

### Adding Companies (`config/companies.json`)

```json
{
  "sectors": {
    "Technology": {
      "companies": [
        {
          "name": "Apple",
          "ticker": "AAPL",
          "aliases": ["Apple Inc", "AAPL", "iPhone maker"]
        }
      ]
    }
  }
}
```

### Adding RSS Feeds (`config/feeds.json`)

```json
[
  {
    "url": "https://feeds.reuters.com/reuters/businessNews",
    "source": "Reuters",
    "category": "general"
  }
]
```

Feed categories: `general`, `tech`, `finance`, `investing`, `macro`, `corporate`, `regulatory`

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--rss-only` | false | Skip Reddit ingestion |
| `--reddit-only` | false | Skip RSS ingestion |
| `--limit N` | none | Cap pipeline at N items |
| `--dry-run` | false | Ingest only, skip Groq API |

---

## 🔧 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **LLM** | Groq · LLaMA-3 70B | NER, sector classification, sentiment scoring |
| **Ingestion** | feedparser · requests | RSS feed parsing |
| **Social** | Reddit Public JSON API | r/wallstreetbets, r/investing, etc. |
| **Storage** | SQLite | Article persistence, timeseries, signals |
| **Dashboard** | Streamlit · Plotly | Interactive visualization |
| **Orchestration** | Python 3.11 | Pipeline coordination, signal detection |

---

## 📈 Results

Deployed and run continuously over a 2-week evaluation window:

| Metric | Value |
|--------|-------|
| Daily articles processed | ~45 (rate-limited for cost) |
| Unique companies tracked | 25+ across 8 sectors |
| Pipeline accuracy (NER) | ~91% entity match rate |
| Sector reversals detected | 3 |
| Company signals detected | 4 |
| Avg pipeline latency | ~1.2s per article |

**Notable detections:**
- Energy sector sentiment reversal (bearish → neutral) 18 hours before a broader sector move
- Negative spike on a pharmaceutical company ahead of an FDA-related news cluster

---

## 🗂️ Database Schema

```sql
articles (
  id TEXT PK, source, category, title, url, text,
  published_at, pipeline_ran, entities JSON,
  sectors JSON, sentiment_scores JSON
)

signals (
  id TEXT PK, signal_type, name, direction,
  score_before, score_after, delta, detected_at,
  article_ids JSON
)
```

---

## 📝 License

MIT — see [LICENSE](LICENSE)

---

<div align="center">
  Built by <a href="https://linkedin.com/in/-harsh-bokadia/">Harsh Bokadia</a> · 
  <a href="https://github.com/harshbokadia">GitHub</a>
</div>
