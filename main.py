"""
Financial Sentiment Intelligence System
Entry point — run the full ingestion → pipeline → signal detection cycle.

Usage:
    python main.py                  # Full run (RSS + Reddit)
    python main.py --rss-only       # RSS feeds only
    python main.py --reddit-only    # Reddit only
    python main.py --limit 20       # Cap items for testing
    python main.py --dry-run        # Ingest + skip Groq pipeline
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")


def parse_args():
    parser = argparse.ArgumentParser(description="Financial Sentiment Intelligence Pipeline")
    parser.add_argument("--rss-only",    action="store_true", help="Fetch RSS feeds only")
    parser.add_argument("--reddit-only", action="store_true", help="Fetch Reddit only")
    parser.add_argument("--limit",       type=int, default=None, help="Max items to process")
    parser.add_argument("--dry-run",     action="store_true", help="Skip Groq pipeline (test ingestion)")
    return parser.parse_args()


def main():
    args = parse_args()
    start = datetime.now(timezone.utc)
    logger.info("=" * 60)
    logger.info("Financial Sentiment Intelligence — Run started at %s", start.strftime("%Y-%m-%d %H:%M UTC"))
    logger.info("=" * 60)

    # ── 1. Initialize DB ──────────────────────────────────────
    from src.storage.db import init_db
    init_db()

    # ── 2. Ingestion ──────────────────────────────────────────
    items = []

    if not args.reddit_only:
        from src.ingestion.rss_fetcher import fetch_all_feeds
        rss_items = fetch_all_feeds()
        logger.info("RSS: %d articles fetched.", len(rss_items))
        items.extend(rss_items)

    if not args.rss_only:
        from src.ingestion.reddit_fetcher import fetch_all_subreddits
        reddit_items = fetch_all_subreddits()
        logger.info("Reddit: %d posts fetched.", len(reddit_items))
        items.extend(reddit_items)

    if args.limit:
        items = items[:args.limit]
        logger.info("Capped at %d items (--limit flag).", len(items))

    if not items:
        logger.warning("No items ingested. Check your network / API keys.")
        return

    # ── 3. Pipeline ───────────────────────────────────────────
    if args.dry_run:
        logger.info("Dry run mode — skipping Groq pipeline.")
        processed = [{**item, "pipeline_ran": False, "entities": [], "sectors": [], "sentiment_scores": {}} for item in items]
    else:
        from src.pipeline.orchestrator import run_pipeline_batch
        processed = run_pipeline_batch(items, batch_size=1, delay=3)

    # ── 4. Store ──────────────────────────────────────────────
    from src.storage.db import upsert_articles
    upsert_articles(processed)

    # ── 5. Signal Detection ───────────────────────────────────
    if not args.dry_run:
        from src.pipeline.signals import run_signal_detection
        signal_summary = run_signal_detection()

        from src.notify.telegram import notify_signals
        notify_signals(signal_summary)
        
        sr = len(signal_summary.get("sector_reversals", []))
        cs = len(signal_summary.get("company_spikes", []))
        logger.info("Signals: %d sector reversals, %d company spikes.", sr, cs)

    # ── 6. Summary ────────────────────────────────────────────
    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    ran = sum(1 for p in processed if p.get("pipeline_ran"))
    logger.info("=" * 60)
    logger.info("Run complete in %.1fs", elapsed)
    logger.info("  Items ingested   : %d", len(items))
    logger.info("  Pipeline ran     : %d", ran)
    logger.info("  Launch dashboard : streamlit run src/dashboard/app.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
