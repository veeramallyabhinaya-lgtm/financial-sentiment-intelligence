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

# Single combined prompt — 1 API call per article instead of 3
COMBINED_SYSTEM = f"""You are a financial NLP analyst. Given a news article or post, perform four tasks in one response.

Return ONLY a valid JSON object in exactly this format:
{{
  "summary": "One to two sentence plain-English summary of what this article is actually about.",
  "entities": ["Company A", "Company B"],
  "sectors": ["Technology", "Finance"],
  "scores": {{
    "Company A": {{"score": 0.75, "label": "positive", "signal": "earnings beat", "confidence": 0.9}},
    "Company B": {{"score": -0.4, "label": "negative", "signal": "regulatory risk", "confidence": 0.7}}
  }}
}}

Rules:
- summary: 1-2 plain English sentences. What happened and why it matters. No jargon. Max 200 chars.
- entities: canonical names of publicly traded companies mentioned (e.g. "Apple" not "AAPL"). Empty list if none.
- sectors: from this list only — {', '.join(SECTORS)}. Most relevant first. Empty list if none apply.
- scores: sentiment per entity. score is -1.0 (very bearish) to 1.0 (very bullish). label is positive/negative/neutral. signal is 2-4 words. confidence is 0.0-1.0.
- If no entities, scores must be {{}}
- No explanation, no markdown, no extra text. Only the JSON object."""


def _groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set. Add it to your .env file.")
    return Groq(api_key=api_key)


def _call_groq(client: Groq, text: str) -> str:
    """Single combined API call with retry on rate limit."""
    prompt = f"Article text: {text[:700]}"
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": COMBINED_SYSTEM},
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


def _validate_result(result: dict) -> tuple[str, list, list, dict]:
    """Validate and sanitize pipeline output."""
    summary = str(result.get("summary", "")).strip()[:300]

    entities = [str(e).strip() for e in result.get("entities", []) if e]

    sectors = [s for s in result.get("sectors", []) if s in SECTORS]

    raw_scores = result.get("scores", {})
    scores = {}
    for company, data in raw_scores.items():
        if isinstance(data, dict):
            scores[company] = {
                "score":      max(-1.0, min(1.0, float(data.get("score", 0.0)))),
                "label":      data.get("label", "neutral"),
                "signal":     data.get("signal", ""),
                "confidence": max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
            }

    return summary, entities, sectors, scores


# ──────────────────────────────────────────────
# Full Pipeline (single call)
# ──────────────────────────────────────────────

def run_pipeline(item: dict[str, Any]) -> dict[str, Any]:
    """
    Run the combined single-call pipeline on one article or post.
    One Groq call returns entities + sectors + sentiment together.
    """
    client = _groq_client()
    text = item.get("text", "")

    if not text.strip():
        return {**item, "entities": [], "sectors": [], "sentiment_scores": {}, "pipeline_ran": False}

    try:
        raw = _call_groq(client, text)
        result = _parse_json_response(raw)
        summary, entities, sectors, sentiment_scores = _validate_result(result)

        logger.debug("[%s] entities=%s sectors=%s scores=%s",
                     item.get("id", "?")[:8], entities, sectors, list(sentiment_scores.keys()))

        return {
            **item,
            "summary": summary,
            "entities": entities,
            "sectors": sectors,
            "sentiment_scores": sentiment_scores,
            "pipeline_ran": True,
        }

    except Exception as e:
        logger.error("Pipeline failed for item %s: %s", item.get("id", "?"), e)
        return {**item, "summary": "", "entities": [], "sectors": [], "sentiment_scores": {}, "pipeline_ran": False, "error": str(e)}


def run_pipeline_batch(items: list[dict], batch_size: int = 10, delay: float = 1.0) -> list[dict]:
    """
    Run the pipeline on a batch of items with rate-limit-safe pacing.

    Args:
        items: List of article/post dicts
        batch_size: Items per batch before pausing
        delay: Seconds to wait between batches

    Returns:
        List of enriched dicts.
    """
    results = []
    total = len(items)
    logger.info("Running pipeline on %d items...", total)

    for i, item in enumerate(items, 1):
        result = run_pipeline(item)
        results.append(result)

        if i % batch_size == 0:
            logger.info("Progress: %d/%d (%.0f%%)", i, total, i / total * 100)
            time.sleep(delay)

    ran = sum(1 for r in results if r.get("pipeline_ran"))
    logger.info("Pipeline complete. %d/%d items processed.", ran, total)
    return results