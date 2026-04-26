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
RETRY_DELAY = 1.5


def _groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set. Add it to your .env file.")
    return Groq(api_key=api_key)


def _call_groq(client: Groq, prompt: str, system: str) -> str:
    """Call Groq with retries on rate limit."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "rate_limit" in str(e).lower() and attempt < MAX_RETRIES - 1:
                logger.warning("Rate limit hit, retrying in %ss...", RETRY_DELAY)
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                raise
    return ""


def _parse_json_response(raw: str) -> Any:
    """Safely extract JSON from model output."""
    # Strip markdown fences if present
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    # Find JSON object or array
    match = re.search(r"(\{.*\}|\[.*\])", clean, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    return json.loads(clean)


# ──────────────────────────────────────────────
# Stage 1: Named Entity Recognition
# ──────────────────────────────────────────────

NER_SYSTEM = """You are a financial NER system. Extract company names from text.
Return ONLY a JSON object: {"entities": ["Company A", "Company B"]}
Rules:
- Use canonical company names (e.g. "Apple", not "AAPL" or "Apple Inc.")
- Return [] if no companies are mentioned
- Include only publicly traded companies
- No explanations, no markdown, just the JSON."""

def stage1_ner(client: Groq, text: str) -> list[str]:
    """
    Stage 1: Extract company entity mentions from text.

    Returns:
        List of canonical company names found in the text.
    """
    prompt = f"Text: {text[:600]}"
    raw = _call_groq(client, prompt, NER_SYSTEM)
    try:
        result = _parse_json_response(raw)
        entities = result.get("entities", [])
        return [str(e).strip() for e in entities if e]
    except Exception:
        logger.debug("NER parse failed for text: %s...", text[:80])
        return []


# ──────────────────────────────────────────────
# Stage 2: Sector Classification
# ──────────────────────────────────────────────

SECTORS = ["Technology", "Finance", "Healthcare", "Energy", "Consumer", "Industrials", "Telecom", "Materials"]

SECTOR_SYSTEM = f"""You are a financial sector classifier. Given a text, identify which sectors it primarily discusses.
Valid sectors: {', '.join(SECTORS)}
Return ONLY a JSON object: {{"sectors": ["Sector1", "Sector2"]}}
Rules:
- Return only sectors from the valid list
- Return [] if no sector is clearly relevant
- Order by relevance (most relevant first)
- No explanations, no markdown, just JSON."""

def stage2_sector(client: Groq, text: str, entities: list[str]) -> list[str]:
    """
    Stage 2: Classify the text into relevant sectors.

    Returns:
        List of sector names, ordered by relevance.
    """
    entities_str = ", ".join(entities) if entities else "unknown"
    prompt = f"Entities mentioned: {entities_str}\nText: {text[:600]}"
    raw = _call_groq(client, prompt, SECTOR_SYSTEM)
    try:
        result = _parse_json_response(raw)
        sectors = result.get("sectors", [])
        return [s for s in sectors if s in SECTORS]
    except Exception:
        logger.debug("Sector parse failed.")
        return []


# ──────────────────────────────────────────────
# Stage 3: Sentiment Scoring
# ──────────────────────────────────────────────

SENTIMENT_SYSTEM = """You are a financial sentiment analyst. Score sentiment for each company in context of the text.
Return ONLY a JSON object:
{
  "scores": {
    "CompanyName": {
      "score": 0.75,
      "label": "positive",
      "signal": "earnings beat",
      "confidence": 0.9
    }
  }
}
Rules:
- score: float from -1.0 (very negative) to 1.0 (very positive)
- label: "positive", "negative", or "neutral"
- signal: 2-4 word reason (e.g. "earnings beat", "regulatory risk", "product launch")
- confidence: 0.0 to 1.0 based on clarity of sentiment in text
- No explanations, no markdown, just JSON."""

def stage3_sentiment(client: Groq, text: str, entities: list[str]) -> dict[str, dict]:
    """
    Stage 3: Score sentiment for each entity extracted in Stage 1.

    Returns:
        Dict mapping company name → sentiment dict with score, label, signal, confidence.
    """
    if not entities:
        return {}

    entities_str = ", ".join(entities)
    prompt = f"Companies to score: {entities_str}\nText: {text[:600]}"
    raw = _call_groq(client, prompt, SENTIMENT_SYSTEM)
    try:
        result = _parse_json_response(raw)
        scores = result.get("scores", {})
        # Validate and clamp scores
        validated = {}
        for company, data in scores.items():
            if isinstance(data, dict):
                data["score"] = max(-1.0, min(1.0, float(data.get("score", 0.0))))
                data["confidence"] = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
                validated[company] = data
        return validated
    except Exception:
        logger.debug("Sentiment parse failed.")
        return {}


# ──────────────────────────────────────────────
# Full Pipeline
# ──────────────────────────────────────────────

def run_pipeline(item: dict[str, Any]) -> dict[str, Any]:
    """
    Run the full 3-stage pipeline on a single article or post.

    Args:
        item: dict with at least {"id", "text", "source", "published_at"}

    Returns:
        Enriched dict with pipeline results added.
    """
    client = _groq_client()
    text = item.get("text", "")

    if not text.strip():
        return {**item, "entities": [], "sectors": [], "sentiment_scores": {}, "pipeline_ran": False}

    try:
        # Stage 1: Entity Extraction
        entities = stage1_ner(client, text)
        logger.debug("[%s] Stage 1 → entities: %s", item.get("id", "?")[:8], entities)

        # Stage 2: Sector Classification
        sectors = stage2_sector(client, text, entities)
        logger.debug("[%s] Stage 2 → sectors: %s", item.get("id", "?")[:8], sectors)

        # Stage 3: Sentiment Scoring
        sentiment_scores = stage3_sentiment(client, text, entities) if entities else {}
        logger.debug("[%s] Stage 3 → scores: %s", item.get("id", "?")[:8], list(sentiment_scores.keys()))

        return {
            **item,
            "entities": entities,
            "sectors": sectors,
            "sentiment_scores": sentiment_scores,
            "pipeline_ran": True,
        }

    except Exception as e:
        logger.error("Pipeline failed for item %s: %s", item.get("id", "?"), e)
        return {**item, "entities": [], "sectors": [], "sentiment_scores": {}, "pipeline_ran": False, "error": str(e)}


def run_pipeline_batch(items: list[dict], batch_size: int = 5, delay: float = 0.3) -> list[dict]:
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
