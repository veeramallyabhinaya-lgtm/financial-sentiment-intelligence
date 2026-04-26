"""
Reddit Feed Fetcher
Ingests posts and top comments from finance-focused subreddits via the public JSON API.
No API key required — uses Reddit's public JSON endpoints.
"""

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any

import requests

logger = logging.getLogger(__name__)

SUBREDDITS = [
    "wallstreetbets",
    "investing",
    "stocks",
    "SecurityAnalysis",
    "ValueInvesting",
    "options",
    "StockMarket",
    "finance",
    "Economics",
    "algotrading",
]

REDDIT_HEADERS = {
    "User-Agent": "FinancialSentimentBot/1.0 (+github.com/harshbokadia/financial-sentiment-intelligence)"
}
POST_LIMIT = 50
MIN_SCORE = 5
REQUEST_TIMEOUT = 10


def _post_id(subreddit: str, post_id: str) -> str:
    return hashlib.md5(f"reddit:{subreddit}:{post_id}".encode()).hexdigest()


def _epoch_to_iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def fetch_subreddit(subreddit: str, sort: str = "hot") -> list[dict[str, Any]]:
    """
    Fetch top posts from a subreddit using the public JSON API.

    Args:
        subreddit: Subreddit name (without r/)
        sort: 'hot', 'new', or 'top'
    """
    url = f"https://www.reddit.com/r/{subreddit}/{sort}.json?limit={POST_LIMIT}"
    posts = []

    try:
        response = requests.get(url, headers=REDDIT_HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        for child in data.get("data", {}).get("children", []):
            post = child.get("data", {})

            score = post.get("score", 0)
            if score < MIN_SCORE:
                continue

            title = post.get("title", "").strip()
            selftext = post.get("selftext", "").strip()
            if not title:
                continue

            # Combine title + body, cap at 1000 chars for efficiency
            full_text = f"{title}. {selftext}"[:1000].strip()

            posts.append({
                "id": _post_id(subreddit, post.get("id", "")),
                "source": f"r/{subreddit}",
                "category": "social",
                "title": title,
                "url": f"https://reddit.com{post.get('permalink', '')}",
                "text": full_text,
                "score": score,
                "num_comments": post.get("num_comments", 0),
                "published_at": _epoch_to_iso(post.get("created_utc", 0)),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            })

    except requests.exceptions.Timeout:
        logger.warning("Timeout fetching r/%s", subreddit)
    except Exception as e:
        logger.warning("Error fetching r/%s: %s", subreddit, e)

    return posts


def fetch_all_subreddits() -> list[dict[str, Any]]:
    """
    Fetch posts from all configured subreddits (hot + new sort).

    Returns:
        Deduplicated list of post dicts, sorted by score descending.
    """
    seen_ids: set[str] = set()
    all_posts: list[dict] = []

    logger.info("Fetching %d subreddits...", len(SUBREDDITS))

    for subreddit in SUBREDDITS:
        for sort in ("hot", "new"):
            posts = fetch_subreddit(subreddit, sort=sort)
            for post in posts:
                if post["id"] not in seen_ids:
                    seen_ids.add(post["id"])
                    all_posts.append(post)

    all_posts.sort(key=lambda p: p.get("score", 0), reverse=True)
    logger.info("Fetched %d unique Reddit posts.", len(all_posts))
    return all_posts
