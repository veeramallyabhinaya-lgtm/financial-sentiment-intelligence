"""
Storage Layer
Persists pipeline outputs to a local SQLite database.
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent.parent / "data" / "sentiment.db"


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS articles (
                id          TEXT PRIMARY KEY,
                source      TEXT,
                category    TEXT,
                title       TEXT,
                url         TEXT,
                text        TEXT,
                published_at TEXT,
                fetched_at  TEXT,
                pipeline_ran INTEGER DEFAULT 0,
                entities    TEXT,   -- JSON array
                sectors     TEXT,   -- JSON array
                sentiment_scores TEXT, -- JSON object
                inserted_at TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at);
            CREATE INDEX IF NOT EXISTS idx_articles_source    ON articles(source);
            CREATE INDEX IF NOT EXISTS idx_articles_pipeline  ON articles(pipeline_ran);

            CREATE TABLE IF NOT EXISTS signals (
                id          TEXT PRIMARY KEY,
                signal_type TEXT,   -- 'company' or 'sector'
                name        TEXT,
                direction   TEXT,   -- 'reversal_up', 'reversal_down', 'spike_positive', 'spike_negative'
                score_before REAL,
                score_after  REAL,
                delta        REAL,
                detected_at  TEXT,
                article_ids  TEXT   -- JSON array
            );

            CREATE INDEX IF NOT EXISTS idx_signals_detected ON signals(detected_at);
            CREATE INDEX IF NOT EXISTS idx_signals_name     ON signals(name);
        """)
    logger.info("Database initialized at %s", DB_PATH)


def upsert_articles(items: list[dict[str, Any]]) -> int:
    """Insert or replace processed articles. Returns count of new rows."""
    if not items:
        return 0

    rows = []
    for item in items:
        rows.append((
            item["id"],
            item.get("source", ""),
            item.get("category", ""),
            item.get("title", ""),
            item.get("url", ""),
            item.get("text", ""),
            item.get("published_at", ""),
            item.get("fetched_at", ""),
            int(item.get("pipeline_ran", False)),
            json.dumps(item.get("entities", [])),
            json.dumps(item.get("sectors", [])),
            json.dumps(item.get("sentiment_scores", {})),
        ))

    with _get_conn() as conn:
        conn.executemany("""
            INSERT OR REPLACE INTO articles
            (id, source, category, title, url, text, published_at, fetched_at,
             pipeline_ran, entities, sectors, sentiment_scores)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, rows)

    logger.info("Upserted %d articles.", len(rows))
    return len(rows)


def get_recent_articles(hours: int = 24, pipeline_ran: bool = True) -> list[dict]:
    """Fetch articles from the last N hours."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    query = "SELECT * FROM articles WHERE published_at >= ?"
    if pipeline_ran:
        query += " AND pipeline_ran = 1"
    query += " ORDER BY published_at DESC"

    with _get_conn() as conn:
        rows = conn.execute(query, (cutoff,)).fetchall()

    result = []
    for row in rows:
        d = dict(row)
        d["entities"] = json.loads(d.get("entities") or "[]")
        d["sectors"] = json.loads(d.get("sectors") or "[]")
        d["sentiment_scores"] = json.loads(d.get("sentiment_scores") or "{}")
        result.append(d)
    return result


def get_sector_sentiment_timeseries(sector: str, hours: int = 48) -> list[dict]:
    """Compute hourly average sentiment for a given sector."""
    articles = get_recent_articles(hours=hours)
    hourly: dict[str, list[float]] = {}

    for article in articles:
        if sector not in article.get("sectors", []):
            continue
        pub_hour = article["published_at"][:13]  # YYYY-MM-DDTHH
        scores = [v["score"] for v in article["sentiment_scores"].values() if isinstance(v, dict)]
        if scores:
            hourly.setdefault(pub_hour, []).extend(scores)

    return [
        {"hour": h, "avg_sentiment": sum(s) / len(s), "volume": len(s)}
        for h, s in sorted(hourly.items())
    ]


def get_company_sentiment_summary(hours: int = 24) -> list[dict]:
    """Aggregate per-company sentiment across recent articles."""
    articles = get_recent_articles(hours=hours)
    company_data: dict[str, dict] = {}

    for article in articles:
        for company, data in article.get("sentiment_scores", {}).items():
            if not isinstance(data, dict):
                continue
            if company not in company_data:
                company_data[company] = {"scores": [], "signals": [], "articles": 0}
            company_data[company]["scores"].append(data.get("score", 0.0))
            company_data[company]["signals"].append(data.get("signal", ""))
            company_data[company]["articles"] += 1

    result = []
    for company, d in company_data.items():
        scores = d["scores"]
        result.append({
            "company": company,
            "avg_sentiment": round(sum(scores) / len(scores), 3),
            "min_sentiment": round(min(scores), 3),
            "max_sentiment": round(max(scores), 3),
            "article_count": d["articles"],
            "top_signals": list(set(d["signals"]))[:3],
        })

    return sorted(result, key=lambda x: abs(x["avg_sentiment"]), reverse=True)


def save_signal(signal: dict) -> None:
    """Persist a detected market signal."""
    with _get_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO signals
            (id, signal_type, name, direction, score_before, score_after, delta, detected_at, article_ids)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            signal["id"], signal["signal_type"], signal["name"], signal["direction"],
            signal.get("score_before", 0), signal.get("score_after", 0), signal.get("delta", 0),
            signal.get("detected_at", datetime.now(timezone.utc).isoformat()),
            json.dumps(signal.get("article_ids", [])),
        ))


def get_recent_signals(hours: int = 72) -> list[dict]:
    """Retrieve recently detected signals."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM signals WHERE detected_at >= ? ORDER BY detected_at DESC", (cutoff,)
        ).fetchall()

    result = []
    for row in rows:
        d = dict(row)
        d["article_ids"] = json.loads(d.get("article_ids") or "[]")
        result.append(d)
    return result
