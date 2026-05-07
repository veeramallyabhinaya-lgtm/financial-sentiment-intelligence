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
                summary          TEXT,
                news_type        TEXT DEFAULT 'company_specific',
                news_importance  REAL DEFAULT 0.5,
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

            CREATE TABLE IF NOT EXISTS scored_pairs (
                pair_key    TEXT PRIMARY KEY,  -- "{article_id}::{company}"
                scored_at   TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_scored_pairs ON scored_pairs(pair_key);

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
    # Migrations: add new columns if missing (safe on existing DBs)
    with _get_conn() as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(articles)").fetchall()]
        for col, defn in [
            ("summary",         "TEXT"),
            ("news_type",       "TEXT DEFAULT 'company_specific'"),
            ("news_importance", "REAL DEFAULT 0.5"),
        ]:
            if col not in cols:
                conn.execute(f"ALTER TABLE articles ADD COLUMN {col} {defn}")
                logger.info("Migration: added %s column to articles.", col)
    logger.info("Database initialized at %s", DB_PATH)


def upsert_articles(items: list[dict[str, Any]]) -> int:
    """
    Persist articles without ever overwriting existing scores.

    Strategy:
    - New articles (not in DB yet): inserted in full.
    - Existing unscored articles (pipeline_ran=0): updated with new
      pipeline results if this run scored them.
    - Existing scored articles (pipeline_ran=1): metadata may update
      (fetched_at), but scores, entities, sectors, summary are NEVER
      overwritten — preserving accumulated scoring history across runs.
    """
    if not items:
        return 0

    with _get_conn() as conn:
        # Fetch which article IDs already exist and whether they're scored
        existing_ids = {
            row[0]: row[1]  # id -> pipeline_ran
            for row in conn.execute(
                "SELECT id, pipeline_ran FROM articles WHERE id IN ({})".format(
                    ",".join("?" * len(items))
                ),
                [item["id"] for item in items],
            ).fetchall()
        }

        new_rows      = []   # brand new articles
        score_updates = []   # existing unscored → now scored
        touch_updates = []   # existing scored → just update fetched_at

        for item in items:
            aid = item["id"]
            already_scored = existing_ids.get(aid, -1)

            if already_scored == -1:
                # Never seen before — full insert
                new_rows.append((
                    aid,
                    item.get("source", ""),
                    item.get("category", ""),
                    item.get("title", ""),
                    item.get("url", ""),
                    item.get("text", ""),
                    item.get("summary", ""),
                    item.get("news_type", "company_specific"),
                    float(item.get("news_importance", 0.5)),
                    item.get("published_at", ""),
                    item.get("fetched_at", ""),
                    int(item.get("pipeline_ran", False)),
                    json.dumps(item.get("entities", [])),
                    json.dumps(item.get("sectors", [])),
                    json.dumps(item.get("sentiment_scores", {})),
                ))

            elif already_scored == 0 and item.get("pipeline_ran"):
                # Was unscored, now has results — write the scores in
                score_updates.append((
                    item.get("summary", ""),
                    item.get("news_type", "company_specific"),
                    float(item.get("news_importance", 0.5)),
                    item.get("fetched_at", ""),
                    int(item.get("pipeline_ran", False)),
                    json.dumps(item.get("entities", [])),
                    json.dumps(item.get("sectors", [])),
                    json.dumps(item.get("sentiment_scores", {})),
                    aid,
                ))

            else:
                # Already scored — only touch fetched_at, preserve everything else
                touch_updates.append((
                    item.get("fetched_at", ""),
                    aid,
                ))

        if new_rows:
            conn.executemany("""
                INSERT INTO articles
                (id, source, category, title, url, text, summary,
                 news_type, news_importance, published_at, fetched_at,
                 pipeline_ran, entities, sectors, sentiment_scores)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, new_rows)

        if score_updates:
            conn.executemany("""
                UPDATE articles SET
                    summary          = ?,
                    news_type        = ?,
                    news_importance  = ?,
                    fetched_at       = ?,
                    pipeline_ran     = ?,
                    entities         = ?,
                    sectors          = ?,
                    sentiment_scores = ?
                WHERE id = ? AND pipeline_ran = 0
            """, score_updates)

        if touch_updates:
            conn.executemany(
                "UPDATE articles SET fetched_at = ? WHERE id = ?",
                touch_updates,
            )

    total = len(new_rows) + len(score_updates) + len(touch_updates)
    logger.info(
        "Articles: %d new | %d scored | %d already-scored (preserved) | %d total.",
        len(new_rows), len(score_updates), len(touch_updates), total,
    )
    return len(new_rows)


def get_recent_articles(hours: int | None = None, pipeline_ran: bool = True) -> list[dict]:
    """
    Fetch scored articles from the database.

    When hours=None (default), returns ALL scored articles ever stored —
    nothing is filtered out. Articles accumulate permanently across runs.

    When hours is set, filters to articles whose inserted_at (DB write time)
    or published_at falls within the window — used by chart queries only.
    """
    if hours is not None:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        query = """
            SELECT * FROM articles
            WHERE (inserted_at >= ? OR published_at >= ?)
        """
        params: list = [cutoff, cutoff]
    else:
        query = "SELECT * FROM articles WHERE 1=1"
        params = []

    if pipeline_ran:
        query += " AND pipeline_ran = 1"
    query += " ORDER BY published_at DESC"

    with _get_conn() as conn:
        rows = conn.execute(query, params).fetchall()

    result = []
    for row in rows:
        d = dict(row)
        d["entities"]         = json.loads(d.get("entities")         or "[]")
        d["sectors"]          = json.loads(d.get("sectors")          or "[]")
        d["sentiment_scores"] = json.loads(d.get("sentiment_scores") or "{}")
        result.append(d)
    return result


def get_total_articles_count() -> dict:
    """Return lifetime ingestion stats — never filtered by time."""
    with _get_conn() as conn:
        total     = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        scored    = conn.execute("SELECT COUNT(*) FROM articles WHERE pipeline_ran=1").fetchone()[0]
        sources   = conn.execute("SELECT COUNT(DISTINCT source) FROM articles").fetchone()[0]
        companies = conn.execute(
            "SELECT COUNT(DISTINCT json_each.value) FROM articles, json_each(entities) WHERE pipeline_ran=1"
        ).fetchone()[0]
    return {
        "total_ingested":   total,
        "total_scored":     scored,
        "unique_sources":   sources,
        "unique_companies": companies,
    }


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

def get_scored_pairs() -> set[str]:
    """Return all tracked (article_id::company) pairs from previous runs."""
    with _get_conn() as conn:
        rows = conn.execute("SELECT pair_key FROM scored_pairs").fetchall()
    return {r["pair_key"] for r in rows}


def save_scored_pairs(pairs: set[str]) -> None:
    """Persist new scored pairs to prevent cross-run double-scoring."""
    if not pairs:
        return
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO scored_pairs (pair_key, scored_at) VALUES (?,?)",
            [(p, now) for p in pairs],
        )