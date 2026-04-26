"""
Signal Detection
Identifies sector reversals and company-level sentiment spikes
by comparing recent windows against the rolling baseline.
"""

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any

from src.storage.db import get_company_sentiment_summary, get_sector_sentiment_timeseries, save_signal

logger = logging.getLogger(__name__)

SECTORS = ["Technology", "Finance", "Healthcare", "Energy", "Consumer", "Industrials", "Telecom", "Materials"]

REVERSAL_THRESHOLD = 0.35    # sector avg sentiment swing
SPIKE_THRESHOLD    = 0.50    # company sentiment spike


def _signal_id(name: str, direction: str, hour: str) -> str:
    return hashlib.md5(f"{name}:{direction}:{hour}".encode()).hexdigest()


def detect_sector_reversals() -> list[dict[str, Any]]:
    """
    Compare the last 6-hour sentiment window vs the prior 18-hour window.
    Flag sectors where the average crosses the REVERSAL_THRESHOLD.
    """
    signals = []

    for sector in SECTORS:
        ts = get_sector_sentiment_timeseries(sector, hours=24)
        if len(ts) < 4:
            continue

        recent = [p["avg_sentiment"] for p in ts[-3:]]
        baseline = [p["avg_sentiment"] for p in ts[:-3]]

        if not recent or not baseline:
            continue

        recent_avg   = sum(recent)   / len(recent)
        baseline_avg = sum(baseline) / len(baseline)
        delta        = recent_avg - baseline_avg

        if abs(delta) >= REVERSAL_THRESHOLD:
            direction = "reversal_up" if delta > 0 else "reversal_down"
            now = datetime.now(timezone.utc).isoformat()
            signal = {
                "id": _signal_id(sector, direction, now[:13]),
                "signal_type": "sector",
                "name": sector,
                "direction": direction,
                "score_before": round(baseline_avg, 3),
                "score_after":  round(recent_avg, 3),
                "delta":        round(delta, 3),
                "detected_at":  now,
                "article_ids":  [],
            }
            save_signal(signal)
            signals.append(signal)
            logger.info("Sector reversal detected: %s → %s (Δ%.3f)", sector, direction, delta)

    return signals


def detect_company_spikes() -> list[dict[str, Any]]:
    """
    Scan per-company sentiment averages for extreme scores
    that exceed the SPIKE_THRESHOLD in either direction.
    """
    signals = []
    summaries = get_company_sentiment_summary(hours=24)

    for row in summaries:
        score = row["avg_sentiment"]
        if abs(score) < SPIKE_THRESHOLD:
            continue

        direction = "spike_positive" if score > 0 else "spike_negative"
        now = datetime.now(timezone.utc).isoformat()
        signal = {
            "id": _signal_id(row["company"], direction, now[:13]),
            "signal_type": "company",
            "name": row["company"],
            "direction": direction,
            "score_before": 0.0,
            "score_after":  score,
            "delta":        score,
            "detected_at":  now,
            "article_ids":  [],
        }
        save_signal(signal)
        signals.append(signal)
        logger.info("Company signal detected: %s → %s (score=%.3f)", row["company"], direction, score)

    return signals


def run_signal_detection() -> dict[str, list]:
    """Run all signal detectors and return a summary."""
    reversals = detect_sector_reversals()
    spikes    = detect_company_spikes()
    logger.info("Signal detection complete: %d sector reversals, %d company spikes.", len(reversals), len(spikes))
    return {"sector_reversals": reversals, "company_spikes": spikes}
