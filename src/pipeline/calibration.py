"""
Calibration Engine
==================
Manages anchor articles that define the scoring scale for the Groq model.

How it works
------------
Instead of hardcoded source weights, we maintain a set of "anchor" articles
in config/calibration.json. Each anchor has a known correct score, importance,
and news_type. These are injected as few-shot examples into every Groq call
so the model scores NEW articles relative to them — not in isolation.

Three anchor sources (merged in priority order):
  1. config/calibration.json  — user-curated, hand-verified ground truth
  2. DB-promoted articles      — high-confidence past scores auto-promoted
                                 after each run (self-calibrating over time)
  3. (future) user-provided    — pass your own news list via CLI to recalibrate

Self-calibration
----------------
After each pipeline run, `promote_from_db()` selects up to N diverse articles
with confidence >= threshold and adds them to the DB anchor pool. On the next
run, those become part of the few-shot context. The system's scoring scale
naturally stabilises and improves as the DB grows.

User recalibration
------------------
Edit config/calibration.json directly. Add examples that reflect YOUR
interpretation of the scale (e.g. "this JSW Steel miss was a 0.5, not 0.65").
The model will use your examples as the reference on the next run.
"""

import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CALIBRATION_PATH = Path(__file__).parent.parent.parent / "config" / "calibration.json"

# How many anchors to inject per call (too many = wasted tokens)
MAX_ANCHORS_IN_PROMPT = 6

# Minimum confidence for a DB article to be promoted as an anchor
PROMOTE_CONFIDENCE_THRESHOLD = 0.82

# How many DB-promoted anchors to mix in alongside user anchors
MAX_DB_ANCHORS = 3

# Maximum anchors to keep in calibration.json (user + promoted combined)
# Beyond this, lowest-value promoted anchors are pruned to make room
MAX_TOTAL_ANCHORS = 28


# ── Load & Save ───────────────────────────────────────────────────────────────

def load_anchors() -> list[dict]:
    """Load user-curated anchors from calibration.json."""
    if not CALIBRATION_PATH.exists():
        logger.warning("calibration.json not found. Scoring without anchors.")
        return []
    with open(CALIBRATION_PATH) as f:
        data = json.load(f)
    anchors = data.get("anchors", [])
    logger.info("Loaded %d calibration anchors from config.", len(anchors))
    return anchors


def save_anchors(anchors: list[dict]) -> None:
    """Write anchors back to calibration.json (preserves _comment and version)."""
    if not CALIBRATION_PATH.exists():
        return
    with open(CALIBRATION_PATH) as f:
        data = json.load(f)
    data["anchors"] = anchors
    data["last_updated"] = datetime.now(timezone.utc).isoformat()[:10]
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(data, f, indent=2)


# ── Anchor Selection ──────────────────────────────────────────────────────────

def _coverage_score(anchor: dict, already_selected: list[dict]) -> float:
    """
    Score how much diversity an anchor adds over already-selected ones.
    Penalise same news_type or sector overlap to maximise scale coverage.
    """
    selected_types    = {a.get("news_type") for a in already_selected}
    selected_sectors  = {s for a in already_selected for s in a.get("sectors", [])}

    type_bonus   = 0.4 if anchor.get("news_type") not in selected_types else 0.0
    sector_bonus = 0.3 if not set(anchor.get("sectors", [])) & selected_sectors else 0.0
    range_bonus  = abs(anchor.get("news_importance", 0.5) - 0.5) * 0.3  # prefer extremes

    return type_bonus + sector_bonus + range_bonus


def select_diverse_anchors(anchors: list[dict], n: int = MAX_ANCHORS_IN_PROMPT) -> list[dict]:
    """
    Greedily select N anchors that maximise coverage of:
      - news_type variety (sector_wide, company_specific, macro)
      - sector variety
      - importance range (extremes are most informative)
    """
    if len(anchors) <= n:
        return anchors

    selected: list[dict] = []
    pool = list(anchors)

    while len(selected) < n and pool:
        best = max(pool, key=lambda a: _coverage_score(a, selected))
        selected.append(best)
        pool.remove(best)

    return selected


# ── DB Promotion ──────────────────────────────────────────────────────────────

def promote_from_db(db_articles: list[dict], existing_ids: set[str]) -> list[dict]:
    """
    Select high-confidence, diverse articles from a processed batch to
    promote as new DB-level anchors.

    These are returned separately (not saved to calibration.json) and are
    passed at runtime alongside the user anchors.

    Args:
        db_articles:  Recently processed pipeline items.
        existing_ids: IDs already in calibration.json (skip duplicates).

    Returns:
        Up to MAX_DB_ANCHORS promoted anchor dicts.
    """
    candidates = []
    for item in db_articles:
        if not item.get("pipeline_ran"):
            continue
        if item.get("id") in existing_ids:
            continue

        scores = item.get("sentiment_scores", {})
        if not scores:
            continue

        # Average confidence across all scored entities
        confidences = [
            v.get("confidence", 0.0)
            for v in scores.values()
            if isinstance(v, dict) and not v.get("propagated", False)
        ]
        if not confidences:
            continue
        avg_conf = sum(confidences) / len(confidences)

        if avg_conf >= PROMOTE_CONFIDENCE_THRESHOLD:
            candidates.append({
                "id":               item.get("id"),
                "text":             item.get("text", "")[:400],
                "news_type":        item.get("news_type", "company_specific"),
                "news_importance":  item.get("news_importance", 0.5),
                "sectors":          item.get("sectors", []),
                "scores":           {
                    k: {"score": v["score"], "signal": v.get("signal",""), "confidence": v.get("confidence",0)}
                    for k, v in scores.items()
                    if isinstance(v, dict) and not v.get("propagated", False)
                },
                "_promoted": True,
                "_avg_confidence": avg_conf,
            })

    if not candidates:
        return []

    # Pick diverse ones
    candidates.sort(key=lambda x: x["_avg_confidence"], reverse=True)
    promoted = select_diverse_anchors(candidates, n=MAX_DB_ANCHORS)
    logger.info("Promoted %d DB articles as runtime calibration anchors.", len(promoted))
    return promoted


# ── Prompt Builder ────────────────────────────────────────────────────────────

def _format_anchor(anchor: dict) -> str:
    """Format a single anchor as a compact prompt example."""
    scores_str = "\n".join(
        f'      "{company}": score={data["score"]:+.2f}, signal="{data.get("signal","")}", confidence={data.get("confidence",0):.2f}'
        for company, data in anchor.get("scores", {}).items()
    )
    return (
        f'  TEXT: "{anchor["text"][:220]}"\n'
        f'  → news_type={anchor["news_type"]}, news_importance={anchor["news_importance"]:.2f}\n'
        f'  → scores:\n{scores_str}'
    )


def build_calibration_context(
    user_anchors: list[dict],
    db_anchors:   list[dict] | None = None,
) -> str:
    """
    Build the few-shot calibration block to inject into the system prompt.

    Selects the most diverse subset from combined user + DB anchors,
    formats them as labelled examples the model uses as its reference scale.
    """
    all_anchors = list(user_anchors) + (db_anchors or [])
    if not all_anchors:
        return ""

    selected = select_diverse_anchors(all_anchors, n=MAX_ANCHORS_IN_PROMPT)

    examples = "\n\n".join(_format_anchor(a) for a in selected)
    return (
        "\n\n--- CALIBRATION SCALE (reference examples — score all new articles RELATIVE to these) ---\n"
        "These examples define your scale. A score of ±1.0 should be as extreme as the most extreme "
        "example below. A 0.5 importance should feel like a middle example. Do not inflate scores.\n\n"
        f"{examples}\n"
        "--- END CALIBRATION ---\n"
    )


# ── Persist after run ─────────────────────────────────────────────────────────

def _prune_to_budget(anchors: list[dict], budget: int) -> list[dict]:
    """
    Prune anchors down to budget while maximising diversity.
    User-curated anchors (no _promoted flag) are NEVER pruned.
    Among promoted anchors, lowest-scoring by a composite diversity+confidence
    metric are dropped first.
    """
    user    = [a for a in anchors if not a.get("_promoted", False)]
    promoted = [a for a in anchors if a.get("_promoted", False)]

    if len(user) + len(promoted) <= budget:
        return anchors   # nothing to prune

    slots = max(0, budget - len(user))  # slots available for promoted

    if slots == 0:
        return user  # no room for any promoted anchors

    # Score each promoted anchor by confidence × diversity contribution
    # (greedy: pick greedily to maximise coverage)
    selected = select_diverse_anchors(
        sorted(promoted, key=lambda x: x.get("_avg_confidence", 0.0), reverse=True),
        n=slots,
    )
    return user + selected


def update_calibration_from_batch(processed_batch: list[dict]) -> int:
    """
    Promote high-confidence articles from this batch into calibration.json
    so future runs calibrate against an ever-improving set of real examples.

    Rules:
    - Only articles with avg entity confidence >= PROMOTE_CONFIDENCE_THRESHOLD
    - Articles already in calibration.json are skipped (dedup by id)
    - New anchors are merged with existing ones and pruned to MAX_TOTAL_ANCHORS
    - User-curated anchors (no _promoted flag) are never pruned or overwritten
    - A timestamp (_promoted_at) is added to each new anchor for auditability

    Returns:
        Number of new anchors added.
    """
    existing  = load_anchors()
    exist_ids = {a.get("id", "") for a in existing}

    new_anchors = promote_from_db(processed_batch, exist_ids)
    if not new_anchors:
        logger.info("Calibration update: no new anchors to add.")
        return 0

    # Stamp with promotion timestamp
    now = datetime.now(timezone.utc).isoformat()[:19]
    for a in new_anchors:
        a["_promoted_at"] = now

    merged = existing + new_anchors
    pruned = _prune_to_budget(merged, MAX_TOTAL_ANCHORS)
    save_anchors(pruned)

    added = len(new_anchors)
    total = len(pruned)
    logger.info(
        "Calibration updated: +%d new anchors, %d total (budget %d).",
        added, total, MAX_TOTAL_ANCHORS,
    )
    return added


# ── Public entry point ─────────────────────────────────────────────────────────

def get_calibration_context(processed_batch: list[dict] | None = None) -> str:
    """
    Load anchors and build the calibration context string for injection
    into the Groq system prompt.

    Args:
        processed_batch: If provided, promotes high-confidence articles
                         from this batch as additional runtime anchors.
                         Pass results from a PREVIOUS run, not the current one.

    Returns:
        Formatted calibration block (empty string if no anchors found).
    """
    user_anchors = load_anchors()
    existing_ids = {a.get("id", "") for a in user_anchors}

    db_anchors: list[dict] = []
    if processed_batch:
        db_anchors = promote_from_db(processed_batch, existing_ids)

    return build_calibration_context(user_anchors, db_anchors)