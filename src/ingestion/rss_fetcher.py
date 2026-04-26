"""
RSS Feed Fetcher
Ingests articles from 45+ financial news RSS feeds.
"""

import feedparser
import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

FEEDS_PATH = Path(__file__).parent.parent.parent / "config" / "feeds.json"
REQUEST_TIMEOUT = 15
MAX_AGE_HOURS = 24


def _load_feeds() -> list[dict]:
    with open(FEEDS_PATH) as f:
        return json.load(f)


def _article_id(url: str, title: str) -> str:
    return hashlib.md5(f"{url}{title}".encode()).hexdigest()


def _parse_date(entry) -> datetime:
    for field in ("published_parsed", "updated_parsed"):
        val = getattr(entry, field, None)
        if val:
            try:
                return datetime(*val[:6], tzinfo=timezone.utc)
            except Exception:
                pass
    return datetime.now(timezone.utc)


def _is_recent(dt: datetime, max_age_hours: int = MAX_AGE_HOURS) -> bool:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    return dt >= cutoff


def fetch_feed(feed_config: dict) -> list[dict[str, Any]]:
    """Fetch and parse a single RSS feed."""
    url = feed_config["url"]
    source = feed_config["source"]
    articles = []

    try:
        headers = {"User-Agent": "FinancialSentimentBot/1.0 (+github.com/harshbokadia/financial-sentiment-intelligence)"}
        response = requests.get(url, timeout=REQUEST_TIMEOUT, headers=headers)
        response.raise_for_status()
        feed = feedparser.parse(response.text)

        for entry in feed.entries:
            title = getattr(entry, "title", "").strip()
            link = getattr(entry, "link", "").strip()
            summary = getattr(entry, "summary", getattr(entry, "description", "")).strip()

            if not title or not link:
                continue

            published_at = _parse_date(entry)
            if not _is_recent(published_at):
                continue

            articles.append({
                "id": _article_id(link, title),
                "source": source,
                "category": feed_config.get("category", "general"),
                "title": title,
                "url": link,
                "text": f"{title}. {summary}",
                "published_at": published_at.isoformat(),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            })

    except requests.exceptions.Timeout:
        logger.warning("Timeout fetching feed: %s", source)
    except requests.exceptions.HTTPError as e:
        logger.warning("HTTP %s for feed: %s", e.response.status_code, source)
    except Exception as e:
        logger.warning("Error fetching feed %s: %s", source, e)

    return articles


def fetch_all_feeds(max_age_hours: int = MAX_AGE_HOURS) -> list[dict[str, Any]]:
    """
    Fetch articles from all configured RSS feeds.

    Returns:
        Deduplicated list of article dicts, sorted newest-first.
    """
    feeds = _load_feeds()
    seen_ids: set[str] = set()
    all_articles: list[dict] = []

    logger.info("Fetching %d RSS feeds...", len(feeds))

    for feed_config in feeds:
        articles = fetch_feed(feed_config)
        for article in articles:
            if article["id"] not in seen_ids:
                seen_ids.add(article["id"])
                all_articles.append(article)

    all_articles.sort(key=lambda a: a["published_at"], reverse=True)
    logger.info("Fetched %d unique articles from RSS feeds.", len(all_articles))
    return all_articles
