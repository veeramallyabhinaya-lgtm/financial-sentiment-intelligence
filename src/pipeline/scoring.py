"""
Comparative Scoring Engine
==========================
Transforms raw per-article Groq scores into a properly calibrated,
batch-relative sentiment model:

  1. Sector Propagation  — sector_wide / macro articles distribute sentiment
                           to ALL companies in the sector, weighted by their
                           Nifty index weight.  Larger-cap companies get a
                           proportionally larger share of the signal.

  2. Deduplication       — (article_id, company) pairs are tracked in SQLite.
                           The same article cannot score the same company twice
                           across any number of pipeline runs.

  3. Comparative Norm.   — Raw scores are percentile-normalised across the
                           full batch, so 0.8 means "top-tier article in this
                           batch" rather than "any mildly positive piece".
                           Importance and source weight scale the final values.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

COMPANIES_PATH = Path(__file__).parent.parent.parent / "config" / "companies.json"

# Category-based credibility weights — derived from the feed's category field,
# not the source name. This generalises cleanly to any new source added to feeds.json.
# Values reflect signal-to-noise ratio typical of each category:
#   regulatory > macro > sector-specific > general > investing opinions > social
CATEGORY_WEIGHTS: dict[str, float] = {
    "regulatory": 1.00,   # RBI, SEBI official releases
    "macro":       0.88,   # Economy, GDP, inflation, central bank commentary
    "finance":     0.85,   # Banking, financial sector news
    "healthcare":  0.83,   # Pharma, biotech
    "energy":      0.82,   # Oil, gas, renewables
    "tech":        0.80,   # Technology, IT
    "general":     0.78,   # Mixed business/market news
    "corporate":   0.65,   # Press releases, corporate announcements
    "investing":   0.62,   # Analyst notes, opinions, seeking alpha type
    "social":      0.42,   # Reddit, forums
}
DEFAULT_CATEGORY_WEIGHT = 0.60


# ── Config helpers ─────────────────────────────────────────────────────────────

def _load_sector_map() -> dict[str, list[dict]]:
    """Return {sector: [company_dict, ...]} with index_weight per company."""
    with open(COMPANIES_PATH) as f:
        cfg = json.load(f)
    return {s: d["companies"] for s, d in cfg["sectors"].items()}


def _max_weight(companies: list[dict]) -> float:
    return max((c.get("index_weight", 0.1) for c in companies), default=1.0)


# ── Step 1: Sector Propagation ─────────────────────────────────────────────────

def _propagate(items: list[dict], sector_map: dict[str, list[dict]]) -> list[dict]:
    """
    For sector_wide / macro articles, distribute sentiment to all companies
    in the relevant sectors.

    - Direct mentions (already in scores) get a weight-proportional boost.
    - Unmentioned companies receive a propagated score scaled by their
      index weight relative to the sector's top-weight company.
    - Propagated confidence is capped at 0.6 to reflect lower certainty.
    """
    result = []
    for item in items:
        news_type = item.get("news_type", "company_specific")
        sectors   = item.get("sectors", [])
        scores    = dict(item.get("sentiment_scores", {}))  # copy

        if news_type in ("sector_wide", "macro") and sectors:
            # Base sentiment = average of explicitly mentioned companies
            direct = [v["score"] for v in scores.values() if isinstance(v, dict)]
            base   = sum(direct) / len(direct) if direct else 0.0

            for sector in sectors:
                cos      = sector_map.get(sector, [])
                max_w    = _max_weight(cos)

                for co in cos:
                    name = co["name"]
                    w    = co.get("index_weight", 0.1)
                    rel  = w / max_w          # 0.0 – 1.0, relative within sector

                    if name in scores and isinstance(scores[name], dict):
                        # Boost direct score proportionally to index weight
                        # Max boost: +20% for top-weight company
                        boost = rel * 0.20
                        raw   = scores[name]["score"]
                        boosted = max(-1.0, min(1.0, raw + boost * abs(raw) * (1 if raw >= 0 else -1)))
                        scores[name] = {**scores[name], "score": round(boosted, 4), "propagated": False}
                    else:
                        # Propagated score
                        prop = round(base * rel, 4)
                        if abs(prop) < 0.02:
                            continue   # too small to bother
                        scores[name] = {
                            "score":      prop,
                            "raw_score":  prop,
                            "label":      "positive" if prop > 0.05 else ("negative" if prop < -0.05 else "neutral"),
                            "signal":     f"{sector.lower()} sector signal",
                            "confidence": round(rel * 0.55, 3),
                            "propagated": True,
                        }

        result.append({**item, "sentiment_scores": scores})
    return result


# ── Step 2: Deduplication ──────────────────────────────────────────────────────

def _dedup(items: list[dict], already_scored: set[str]) -> tuple[list[dict], set[str]]:
    """
    Remove scores for (article_id, company) pairs already processed.
    Mutates already_scored in-place and returns cleaned items.
    """
    result = []
    for item in items:
        aid    = item.get("id", "")
        clean  = {}
        for company, data in item.get("sentiment_scores", {}).items():
            key = f"{aid}::{company}"
            if key not in already_scored:
                clean[company] = data
                already_scored.add(key)
            else:
                logger.debug("Dedup: skipping %s for article %s", company, aid[:8])
        result.append({**item, "sentiment_scores": clean})
    return result, already_scored


# ── Step 3: Comparative Normalisation ─────────────────────────────────────────

def _normalise(items: list[dict]) -> list[dict]:
    """
    Normalise all sentiment scores relative to the batch distribution.

    Algorithm:
      - Collect raw scores, weighted by (news_importance × source_weight).
      - Compute importance-weighted mean and the 90th/10th percentile range.
      - Map each score through: norm = (raw - wmean) / (p90-p10)/2
      - Clamp to [-1, 1], then dampen by importance and propagation flag.

    This ensures:
      - A score of ±1.0 represents genuinely extreme news in this batch.
      - Low-importance / propagated scores cluster toward 0.
      - Cross-company comparisons are meaningful.
    """
    # Flatten all scores
    flat: list[dict] = []
    for i, item in enumerate(items):
        # Use category-based credibility (generalises to any source)
        src_w  = CATEGORY_WEIGHTS.get(item.get("category", ""), DEFAULT_CATEGORY_WEIGHT)
        imp    = float(item.get("news_importance", 0.5))
        weight = src_w * imp           # composite weight for this article

        for company, data in item.get("sentiment_scores", {}).items():
            if not isinstance(data, dict):
                continue
            flat.append({
                "item_idx":   i,
                "company":    company,
                "raw":        float(data.get("score", 0.0)),
                "weight":     weight,
                "importance": imp,
                "propagated": bool(data.get("propagated", False)),
                "data":       data,
            })

    if len(flat) < 4:
        logger.info("Too few scores to normalise (%d). Returning raw.", len(flat))
        return items

    raws    = np.array([s["raw"]    for s in flat])
    weights = np.array([s["weight"] for s in flat])

    # Weighted mean
    wmean   = float(np.average(raws, weights=weights))

    # Percentile range (90th – 10th) as the scale denominator
    p90     = float(np.percentile(raws, 90))
    p10     = float(np.percentile(raws, 10))
    p_range = max(p90 - p10, 0.10)       # avoid division by zero

    half    = p_range / 2.0

    def _norm_score(raw: float, importance: float, propagated: bool) -> float:
        n  = (raw - wmean) / half
        n  = max(-1.0, min(1.0, n))
        # Dampen: importance 1.0 → full signal; 0.0 → 50% signal
        n *= (0.50 + 0.50 * importance)
        # Extra dampen for propagated (indirect) scores
        if propagated:
            n *= 0.70
        return round(n, 4)

    # Build lookup: item_idx → company → normalised
    lookup: dict[int, dict[str, float]] = {}
    for s in flat:
        lookup.setdefault(s["item_idx"], {})[s["company"]] = _norm_score(
            s["raw"], s["importance"], s["propagated"]
        )

    # Apply back to items
    result = []
    for i, item in enumerate(items):
        if i not in lookup:
            result.append(item)
            continue
        new_scores = {}
        for company, data in item.get("sentiment_scores", {}).items():
            if not isinstance(data, dict):
                continue
            norm = lookup[i].get(company)
            if norm is None:
                new_scores[company] = data
            else:
                new_scores[company] = {
                    **data,
                    "raw_score": data.get("score", 0.0),   # preserve pre-norm
                    "score":     norm,
                    "label":     "positive" if norm > 0.08 else (
                                 "negative" if norm < -0.08 else "neutral"),
                }
        result.append({**item, "sentiment_scores": new_scores})

    logger.info(
        "Normalised %d scores | wmean=%.3f p10=%.2f p90=%.2f range=%.2f",
        len(flat), wmean, p10, p90, p_range,
    )
    return result


# ── Public entry point ─────────────────────────────────────────────────────────

def run_scoring_pipeline(
    items: list[dict[str, Any]],
    already_scored: set[str] | None = None,
) -> tuple[list[dict[str, Any]], set[str]]:
    """
    Run the full post-processing pipeline on a processed batch.

    Args:
        items:          List of pipeline-processed article dicts.
        already_scored: Set of "article_id::company" keys already persisted.
                        Pass the result of db.get_scored_pairs() for
                        cross-run deduplication.

    Returns:
        (enriched_items, updated_already_scored)
    """
    if already_scored is None:
        already_scored = set()

    sector_map = _load_sector_map()

    logger.info("Scoring pipeline start: %d items", len(items))

    items = _propagate(items, sector_map)
    logger.info("Sector propagation done.")

    items, already_scored = _dedup(items, already_scored)
    logger.info("Dedup done. %d unique pairs.", len(already_scored))

    items = _normalise(items)
    logger.info("Comparative normalisation done.")

    return items, already_scored