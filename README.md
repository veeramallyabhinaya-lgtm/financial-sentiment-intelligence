<div align="center">

# 📡 India Market Sentiment Intelligence

**End-to-end AI pipeline that converts 47 Indian financial news feeds into calibrated, comparative market signals for Nifty 50 constituents**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Groq](https://img.shields.io/badge/Groq-LLaMA--3.3_70B-FF6B35?style=for-the-badge)](https://groq.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![SQLite](https://img.shields.io/badge/SQLite-Local_DB-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://sqlite.org)

[Overview](#-overview) · [Architecture](#-architecture) · [Pipeline](#-pipeline-in-depth) · [Calibration](#-calibration-system) · [Quickstart](#-quickstart) · [Configuration](#-configuration) · [Dashboard](#-dashboard) · [Results](#-results)

</div>

---

## 🎯 Overview

A production-grade alternative data pipeline focused on **Indian equity markets**. It ingests unstructured financial text from 47 RSS feeds (ET Markets, Moneycontrol, Mint, Business Standard, RBI, SEBI, and others) plus 10 finance subreddits, processes each article through a Groq-powered NLP pipeline, and surfaces real-time sentiment signals for **49 Nifty 50 constituents** across 8 sectors.

The core design principle is **comparative scoring** — no article is scored in isolation. Every sentiment score is calibrated against a set of anchor examples and normalised across the full batch, so a score of +0.8 genuinely means "top-tier bullish news in this batch," not just "any mildly positive piece."

```
47 feeds + 10 subreddits
        ↓
   Ingestion & dedup
        ↓
  Groq NLP pipeline  ←── calibration anchors injected
  (NER + sector + sentiment + importance + summary)
        ↓
  Comparative scoring
  ├── Sector propagation (index-weighted)
  ├── Cross-run deduplication
  └── Batch normalisation (p10–p90 range)
        ↓
  Signal detection
  ├── Sector reversals (6h vs 18h baseline)
  └── Company spikes (|score| > 0.50)
        ↓
  SQLite persistence + Streamlit dashboard
```

---

## 🏗️ Architecture

```
financial-sentiment-intelligence/
│
├── main.py                          # Entry point & pipeline orchestrator
│
├── config/
│   ├── companies.json               # 49 Nifty constituents with index weights
│   ├── feeds.json                   # 47 RSS feed URLs
│   └── calibration.json             # Anchor articles that define the scoring scale
│
└── src/
    ├── ingestion/
    │   ├── rss_fetcher.py            # Parallel RSS ingestion (47 feeds)
    │   └── reddit_fetcher.py         # Reddit public JSON API (10 subreddits)
    │
    ├── pipeline/
    │   ├── orchestrator.py           # Groq NLP call with calibration context
    │   ├── calibration.py            # Anchor management & self-calibration
    │   ├── scoring.py                # Sector propagation, dedup, normalisation
    │   └── signals.py                # Reversal & spike detection
    │
    ├── storage/
    │   └── db.py                     # SQLite schema, queries, scored_pairs tracking
    │
    └── dashboard/
        └── app.py                    # Streamlit dashboard
```

---

## 🔬 Pipeline In Depth

### Stage 1 — Ingestion

Two parallel fetchers run on every execution:

**RSS Fetcher** (`src/ingestion/rss_fetcher.py`)
Hits 47 configured sources with a 24-hour rolling window and URL-hash deduplication. Sources span ET Markets, Moneycontrol, Mint, Business Standard, Financial Express, Hindu BusinessLine, NDTV Profit, Zee Business, RBI Press Releases, SEBI Press Releases, Bloomberg, Financial Times, Reuters, and others. Gracefully handles timeouts, 403s, and DNS failures — the pipeline continues regardless of individual feed failures.

**Reddit Fetcher** (`src/ingestion/reddit_fetcher.py`)
Pulls hot and new posts from 10 finance subreddits (r/wallstreetbets, r/investing, r/stocks, r/SecurityAnalysis, r/ValueInvesting, r/options, r/StockMarket, r/finance, r/Economics, r/algotrading) via Reddit's public JSON API. No API key required.

---

### Stage 2 — Groq NLP Pipeline

Each article makes a **single Groq API call** (LLaMA-3.3 70B) that returns all four outputs at once:

```json
{
  "summary":          "1-2 sentence plain-English summary of what happened and why it matters.",
  "news_type":        "sector_wide",
  "news_importance":  0.88,
  "entities":         ["HDFC Bank", "ICICI Bank"],
  "sectors":          ["Finance"],
  "scores": {
    "HDFC Bank":  {"score": 0.65, "signal": "rate cut NIM benefit", "confidence": 0.91},
    "ICICI Bank": {"score": 0.62, "signal": "rate cut NIM benefit", "confidence": 0.88}
  }
}
```

**`news_type`** classifies the article as one of:
- `company_specific` — directly about named companies (earnings, management, deal)
- `sector_wide` — affects an entire sector (RBI rate decision, SEBI rule, oil price)
- `macro` — economy-wide (GDP, CPI, FII flows, INR movement)

**`news_importance`** is a calibrated 0–1 score the model assigns based on the anchor reference scale, not a hardcoded rule:
- 0.85–1.0: RBI/SEBI policy, major M&A, earnings miss/beat >15%, credit rating change
- 0.65–0.84: Quarterly results, analyst upgrades, management changes, large deal wins
- 0.45–0.64: Industry trends, product launches, partnerships
- 0.20–0.44: General commentary, analyst initiations, minor updates
- 0.01–0.19: Speculative or low-signal pieces

---

### Stage 3 — Comparative Scoring

Three post-processing passes run after the full batch is scored:

**3a. Sector Propagation**
Articles tagged `sector_wide` or `macro` distribute sentiment to **all companies in the relevant sector**, weighted by their Nifty index weight. An RBI rate cut tagged under Finance propagates to all 8 Finance-sector companies. HDFC Bank (28% index weight) receives the strongest signal; AU Small Finance Bank (2% weight) receives a proportionally smaller one. Companies directly mentioned in the article get an additional weight-proportional boost on top of their direct score.

**3b. Cross-Run Deduplication**
Every `(article_id, company)` pair is written to the `scored_pairs` table in SQLite after processing. On subsequent runs, any pair that already exists in this table is skipped — the same RBI article cannot contribute to HDFC Bank's score twice across multiple pipeline runs.

**3c. Comparative Normalisation**
After all articles in the batch are scored, scores are normalised across the batch distribution. The algorithm:
1. Collects all raw scores, weighted by `(news_importance × category_credibility)`
2. Computes a weighted mean and the p10–p90 percentile range
3. Maps each score: `norm = (raw − wmean) / (p_range / 2)`, clamped to [−1, 1]
4. Dampens by importance (low-importance news scores closer to 0)
5. Further dampens propagated scores (indirect sector signals) by 30%

Category credibility weights used in normalisation (no per-source hardcoding):

| Category | Weight |
|----------|--------|
| `regulatory` (RBI, SEBI) | 1.00 |
| `macro` | 0.88 |
| `finance` | 0.85 |
| `healthcare` / `energy` / `tech` | 0.78–0.83 |
| `general` | 0.78 |
| `corporate` (press releases) | 0.65 |
| `investing` (analyst notes) | 0.62 |
| `social` (Reddit) | 0.42 |

---

### Stage 4 — Signal Detection

**Sector Reversals**
Compares each sector's average sentiment over the last 6 hours against the prior 18-hour baseline. A delta of ±0.35 or more flags a reversal — a leading indicator of a broader directional shift.

**Company Spikes**
Flags companies whose rolling average sentiment across recent articles exceeds ±0.50, indicating extreme coverage in one direction.

All signals are persisted to the `signals` table with before/after scores, delta, and timestamp.

---

## 🎯 Calibration System

The calibration system is the key differentiator from naive sentiment scoring.

### How it works

`config/calibration.json` stores a set of **anchor articles** — real Indian market news examples with known correct scores. Before every pipeline run, 3 diverse anchors are selected (maximising coverage across news_type, sector, and importance range) and injected directly into the Groq system prompt:

```
--- CALIBRATION SCALE (reference examples) ---
These examples define your scale. Score all new articles RELATIVE to these.

  TEXT: "RBI cuts repo rate by 25bps to 6.00%..."
  → news_type=sector_wide, news_importance=0.95
  → HDFC Bank: score=+0.72, signal="NIMs expand on rate cut"
  → AU Small Finance Bank: score=+0.38, signal="modest NIM benefit"

  TEXT: "JSW Steel Q2 EBITDA down 22% YoY, misses consensus..."
  → news_type=company_specific, news_importance=0.76
  → JSW Steel: score=-0.65, signal="EBITDA miss, volume cut"

  TEXT: "HUL launches new skincare line under Lakme..."
  → news_type=company_specific, news_importance=0.32
  → Hindustan Unilever: score=+0.15, signal="premiumisation initiative"
--- END CALIBRATION ---
```

The model cannot hand out uniform scores when it sees a 0.95 RBI rate cut and a 0.32 product launch side by side. Every new article is scored **relative to that reference scale**.

### Self-calibration

After each run, `update_calibration_from_batch()` scans the just-processed articles and promotes those with average entity confidence ≥ 0.82 into `calibration.json`. The anchor pool grows organically from real data. Rules:
- Promoted anchors are chosen for **diversity** — different news_type, sector, importance range
- Total anchors are capped at **28** (budget = 28 slots)
- User-curated anchors (no `_promoted` flag) are **never pruned**
- Promoted anchors rotate out when better examples arrive

### Manual calibration

Open `config/calibration.json` and add entries directly to the `anchors` array. Your examples take permanent priority. This is how you encode domain knowledge — e.g., "for this portfolio, a Reliance earnings miss matters more than an analyst note on Wipro."

---

## 🚀 Quickstart

### 1. Clone & install

```bash
git clone https://github.com/harshbokadia/financial-sentiment-intelligence.git
cd financial-sentiment-intelligence
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure API key

```bash
copy .env.example .env     # Windows
cp .env.example .env       # macOS/Linux
```

Edit `.env`:
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
```

Get a free key at [console.groq.com](https://console.groq.com). The free tier provides 100,000 tokens/day, sufficient for ~120 articles per run with calibration context included.

### 3. Run the pipeline

```bash
# Full run — RSS + Reddit, no limit
python main.py

# RSS only (recommended for daily use — faster, fewer tokens)
python main.py --rss-only

# Cap articles for testing
python main.py --rss-only --limit 50

# Ingest only, skip Groq (test feeds without spending tokens)
python main.py --dry-run
```

### 4. Launch the dashboard

```bash
streamlit run src/dashboard/app.py
```

Opens at `http://localhost:8501`.

### 5. Clear stored scores (re-run from scratch)

```bash
python -c "
import sqlite3
conn = sqlite3.connect('data/sentiment.db')
conn.executescript('''
    UPDATE articles SET pipeline_ran=0, entities='[]', sectors='[]',
        sentiment_scores='{}', summary='', news_type='company_specific', news_importance=0.5;
    DELETE FROM signals;
    DELETE FROM scored_pairs;
''')
conn.commit()
print('Scores cleared.')
"
```

---

## ⚙️ Configuration

### Adding companies (`config/companies.json`)

```json
{
  "sectors": {
    "Finance": {
      "color": "#10b981",
      "companies": [
        {
          "name": "HDFC Bank",
          "ticker": "HDFCBANK.NS",
          "aliases": ["HDFC Bank", "HDFCBANK", "HDFC"],
          "index_weight": 0.28
        }
      ]
    }
  }
}
```

`index_weight` determines how strongly sector-wide news propagates to each company. Use relative Nifty Bank / Nifty IT / Nifty 50 weightages.

### Adding feeds (`config/feeds.json`)

```json
[
  {
    "url": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "source": "ET Markets",
    "category": "general"
  }
]
```

`category` must be one of: `regulatory`, `macro`, `finance`, `healthcare`, `energy`, `tech`, `general`, `corporate`, `investing`, `social`. This determines source credibility weight in normalisation.

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--rss-only` | false | Skip Reddit ingestion |
| `--reddit-only` | false | Skip RSS ingestion |
| `--limit N` | none | Cap pipeline at N items |
| `--dry-run` | false | Ingest only, skip Groq API |

---

## 📊 Dashboard

Four tabs, all controls inline (no sidebar):

| Tab | Contents |
|-----|----------|
| **Heatmap & Signals** | Hourly sector sentiment heatmap (red→green) + live signal feed with direction badges |
| **Company Rankings** | Horizontal bar chart sorted by absolute/bullish/bearish/coverage; configurable top-N |
| **Sector Trends** | Single-sector time series or all-sectors overlay with spline smoothing |
| **Article Feed** | Cards with 1-2 line AI summary, source badge, sentiment badge, news type badge, importance score, entity tags |

Filter bar (always visible): lookback window (6–72h), sector multiselect, min articles per company, refresh button.

---

## 🔧 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| LLM | Groq · LLaMA-3.3 70B | NER, classification, sentiment, summarisation |
| Ingestion | feedparser · requests | RSS parsing |
| Social | Reddit Public JSON API | 10 finance subreddits |
| Scoring | numpy | Percentile normalisation |
| Storage | SQLite | Articles, signals, scored_pairs |
| Dashboard | Streamlit · Plotly · pandas | Visualisation |
| Config | JSON | Companies, feeds, calibration anchors |

---

## 📈 Results

| Metric | Value |
|--------|-------|
| Daily articles ingested | ~468 (RSS) + ~380 (Reddit) |
| Nifty constituents tracked | 49 across 8 sectors |
| Pipeline latency per article | ~1.2s average |
| Token cost per article (with calibration) | ~800 tokens |
| Daily token budget coverage | ~120 articles (Groq free tier) |
| Sector reversals detected (2-week window) | 3 |
| Company signals detected (2-week window) | 4 |
| Calibration anchor pool (seed) | 12 hand-curated Indian market examples |
| Max anchor pool (with self-calibration) | 28 |

---

<div align="center">
  Built by <a href="https://linkedin.com/in/-harsh-bokadia/">Harsh Bokadia</a> &nbsp;·&nbsp;
  <a href="https://github.com/harshbokadia">github.com/harshbokadia</a>
</div>
