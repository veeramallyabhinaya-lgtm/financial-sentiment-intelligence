"""
3-Stage Sentiment Pipeline
Powered by Groq (LLaMA-3 70B)

Stage 1 → NER: Extract company mentions from text
Stage 2 → Sector Classification: Tag the dominant sector(s)
Stage 3 → Sentiment Scoring: Score sentiment per entity (-1.0 to 1.0)
"""

import json
import logging
import os
import re
import time
from typing import Any

from groq import Groq

logger = logging.getLogger(__name__)

MODEL = "llama-3.3-70b-versatile"
MAX_RETRIES = 3
RETRY_DELAY = 2.0

SECTORS = ["Technology", "Finance", "Healthcare", "Energy", "Consumer", "Industrials", "Telecom", "Materials"]

# System prompt is built dynamically — calibration context injected at call time
_SYSTEM_BASE = """You are a senior financial analyst specialising in Indian markets. Analyse a news article and return structured JSON.

Return ONLY a valid JSON object in exactly this format:
{{
  "summary": "One to two sentence plain-English summary. What happened and why it matters for investors.",
  "news_type": "company_specific",
  "news_importance": 0.75,
  "entities": ["Company A", "Company B"],
  "sectors": ["Technology", "Finance"],
  "scores": {{
    "Company A": {{"score": 0.75, "signal": "strong earnings beat", "confidence": 0.9}},
    "Company B": {{"score": -0.4, "signal": "regulatory headwind", "confidence": 0.7}}
  }}
}}

Field rules:
- summary: 1-2 plain English sentences. Max 220 chars. No jargon.

- news_type: one of three values only:
    "company_specific" — directly about named companies (earnings, management, product)
    "sector_wide"      — affects entire sector (RBI rate change, SEBI rule, IT sector slowdown, oil price)
    "macro"            — economy-wide (GDP, inflation, FII flows, global recession, INR movement)

- news_importance: float 0.0–1.0. Be discriminating — most news is 0.3–0.6.
    0.85–1.0 : RBI/SEBI policy, major M&A, earnings miss/beat >15%, credit rating change
    0.65–0.84: Quarterly results (routine), analyst upgrade/downgrade, management change, large deal win
    0.45–0.64: Industry trend, product launch, partnership, minor regulatory update
    0.20–0.44: General market commentary, analyst initiation, minor corporate update
    0.01–0.19: Speculative or low-quality piece

- entities: canonical company names for publicly listed Indian companies (use BSE/NSE names).
    Use "HDFC Bank" not "HDFC" alone. Use "TCS" not "Tata Consultancy Services".
    Empty list [] if no specific company is mentioned.

- sectors: from this list only — {', '.join(SECTORS)}. Most relevant first. Empty [] if none apply.

- scores: assign ONLY to entities in your entities list.
    score: -1.0 (very bearish) to +1.0 (very bullish). Be calibrated:
      ±0.7–1.0: Exceptional/severe — landmark deals, massive misses, regulatory action
      ±0.4–0.69: Significant — strong quarterly beat, key management exit, meaningful downgrade
      ±0.2–0.39: Moderate — in-line results, analyst note, minor positive/negative
      ±0.0–0.19: Marginal or ambiguous sentiment
    signal: 2-5 word reason e.g. "RBI rate cut benefit", "margin compression risk", "deal win"
    confidence: 0.0–1.0 based on how clearly the article supports the score
    scores {{}} if entities is empty.

- No extra keys, no markdown, no explanation. Return the JSON object only."""


def _build_system_prompt(calibration_context: str = "") -> str:
    """Assemble system prompt with injected calibration anchors."""
    return _SYSTEM_BASE + calibration_context


def _groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set. Add it to your .env file.")
    return Groq(api_key=api_key)


def _call_groq(client: Groq, text: str, calibration_context: str = "") -> str:
    """Single combined API call with retry on rate limit."""
    prompt = f"Article text: {text[:700]}"
    system = _build_system_prompt(calibration_context)
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=500,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err = str(e).lower()
            if "rate_limit" in err and "tokens per day" in err:
                raise  # Daily limit — no point retrying
            if "rate_limit" in err and attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAY * (attempt + 1)
                logger.warning("Rate limit hit, retrying in %ss...", wait)
                time.sleep(wait)
            else:
                raise
    return ""


def _parse_json_response(raw: str) -> Any:
    """Safely extract JSON from model output."""
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    return json.loads(clean)


def _validate_result(result: dict) -> tuple[str, str, float, list, list, dict]:
    """Validate and sanitize pipeline output."""
    summary    = str(result.get("summary", "")).strip()[:300]
    news_type  = result.get("news_type", "company_specific")
    if news_type not in ("company_specific", "sector_wide", "macro"):
        news_type = "company_specific"
    news_importance = max(0.0, min(1.0, float(result.get("news_importance", 0.5))))

    entities = [str(e).strip() for e in result.get("entities", []) if e]
    sectors  = [s for s in result.get("sectors", []) if s in SECTORS]

    raw_scores = result.get("scores", {})
    scores = {}
    for company, data in raw_scores.items():
        if isinstance(data, dict):
            raw = max(-1.0, min(1.0, float(data.get("score", 0.0))))
            scores[company] = {
                "score":      raw,
                "raw_score":  raw,
                "label":      "positive" if raw > 0.08 else ("negative" if raw < -0.08 else "neutral"),
                "signal":     data.get("signal", ""),
                "confidence": max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
                "propagated": False,
            }

    return summary, news_type, news_importance, entities, sectors, scores


# ──────────────────────────────────────────────
# Full Pipeline (single call)
# ──────────────────────────────────────────────

def run_pipeline(item: dict[str, Any], calibration_context: str = "") -> dict[str, Any]:
    """
    Run the combined single-call pipeline on one article or post.
    One Groq call returns entities + sectors + sentiment together.
    Calibration context (few-shot anchors) is injected into the system prompt.
    """
    client = _groq_client()
    text = item.get("text", "")

    if not text.strip():
        return {**item, "entities": [], "sectors": [], "sentiment_scores": {}, "pipeline_ran": False}

    try:
        raw = _call_groq(client, text, calibration_context)
        result = _parse_json_response(raw)
        summary, news_type, news_importance, entities, sectors, sentiment_scores = _validate_result(result)

        logger.debug("[%s] type=%s imp=%.2f entities=%s sectors=%s",
                     item.get("id", "?")[:8], news_type, news_importance, entities, sectors)

        return {
            **item,
            "summary":          summary,
            "news_type":        news_type,
            "news_importance":  news_importance,
            "entities":         entities,
            "sectors":          sectors,
            "sentiment_scores": sentiment_scores,
            "pipeline_ran":     True,
        }

    except Exception as e:
        logger.error("Pipeline failed for item %s: %s", item.get("id", "?"), e)
        return {**item, "summary": "", "news_type": "company_specific", "news_importance": 0.5,
                "entities": [], "sectors": [], "sentiment_scores": {}, "pipeline_ran": False, "error": str(e)}


def run_pipeline_batch(
    items: list[dict],
    batch_size: int = 10,
    delay: float = 1.0,
    previous_batch: list[dict] | None = None,
) -> list[dict]:
    """
    Run the pipeline on a batch of items with rate-limit-safe pacing.

    Calibration anchors are loaded ONCE at the start of the batch:
      - User-curated anchors from config/calibration.json
      - High-confidence articles from previous_batch (if provided)
    The same calibration context is injected into every Groq call in this batch,
    so all scores are on the same relative scale.

    Args:
        items:          List of article/post dicts to process.
        batch_size:     Items per mini-batch before pausing (rate limit).
        delay:          Seconds to wait between mini-batches.
        previous_batch: Processed items from a prior run to promote as
                        runtime anchors (self-calibration).

    Returns:
        List of enriched dicts with calibrated scores.
    """
    from src.pipeline.calibration import get_calibration_context

    # Build calibration context once for the whole batch
    calibration_context = get_calibration_context(processed_batch=previous_batch)
    anchor_count = calibration_context.count("TEXT:") if calibration_context else 0
    logger.info("Calibration: %d anchors injected into system prompt.", anchor_count)

    results = []
    total = len(items)
    logger.info("Running pipeline on %d items...", total)

    for i, item in enumerate(items, 1):
        result = run_pipeline(item, calibration_context=calibration_context)
        results.append(result)

        if i % batch_size == 0:
            logger.info("Progress: %d/%d (%.0f%%)", i, total, i / total * 100)
            time.sleep(delay)

    ran = sum(1 for r in results if r.get("pipeline_ran"))
    logger.info("Pipeline complete. %d/%d items processed.", ran, total)
    return results