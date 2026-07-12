"""
Telegram notification module for financial-sentiment-intelligence.

Pushes detected signals (sector reversals + company spikes) to a Telegram
chat after each pipeline run, so you don't have to check the Streamlit
dashboard manually.

SETUP:
1. Create a bot via @BotFather on Telegram -> get a bot token
2. Get your chat ID via @userinfobot on Telegram
3. Add both to your .env file (see .env.example):
     TELEGRAM_BOT_TOKEN=xxxxx
     TELEGRAM_CHAT_ID=xxxxx
4. In main.py, after run_signal_detection(), call notify_signals(signal_summary)
   (see integration snippet at the bottom of this file)

This module is intentionally defensive about the exact shape of each signal
dict, since field names can vary slightly by implementation. It tries a few
common key names and falls back to a raw dump rather than crashing the
pipeline run if something doesn't match.
"""

import os
import logging
import requests

logger = logging.getLogger("notify.telegram")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"

# Only notify for signals at or above these thresholds (tune to taste).
# Set to 0 to notify on everything the pipeline already flagged.
MIN_REVERSAL_DELTA = 0.0
MIN_SPIKE_SCORE = 0.0


def _get(d, *keys, default=None):
    """Try several possible key names on a dict, return first match."""
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return default


def _format_reversal(item):
    sector = _get(item, "sector", "sector_name", "name", default="Unknown sector")
    before = _get(item, "before", "before_score", "baseline", "prior")
    after = _get(item, "after", "after_score", "current", "recent")
    delta = _get(item, "delta", "change")

    if delta is None and before is not None and after is not None:
        try:
            delta = after - before
        except TypeError:
            delta = None

    direction = "📈 turning bullish" if (delta or 0) > 0 else "📉 turning bearish"

    if before is not None and after is not None:
        return (f"🔄 <b>{sector}</b> sector reversal — {direction}\n"
                f"   {before:.2f} → {after:.2f} (Δ {delta:+.2f})" if isinstance(delta, (int, float))
                else f"🔄 <b>{sector}</b> sector reversal — {direction}\n   {before} → {after}")
    return f"🔄 <b>{sector}</b> sector reversal detected\n   raw: {item}"


def _format_spike(item):
    company = _get(item, "company", "name", "entity", default="Unknown company")
    score = _get(item, "score", "avg_score", "sentiment_score")
    signal = _get(item, "signal", "reason", "summary", default="")

    direction = "🚀 bullish spike" if (score or 0) > 0 else "⚠️ bearish spike"

    if isinstance(score, (int, float)):
        line = f"{direction} — <b>{company}</b>: {score:+.2f}"
    else:
        line = f"{direction} — <b>{company}</b>"

    if signal:
        line += f"\n   {signal}"
    return line


def build_alert_messages(signal_summary):
    """
    Turn a signal_summary dict (as returned by run_signal_detection())
    into a list of formatted Telegram message strings.
    """
    messages = []

    reversals = signal_summary.get("sector_reversals", []) or []
    for item in reversals:
        delta = _get(item, "delta", "change", default=0) or 0
        try:
            if abs(delta) < MIN_REVERSAL_DELTA:
                continue
        except TypeError:
            pass
        messages.append(_format_reversal(item))

    spikes = signal_summary.get("company_spikes", []) or []
    for item in spikes:
        score = _get(item, "score", "avg_score", "sentiment_score", default=0) or 0
        try:
            if abs(score) < MIN_SPIKE_SCORE:
                continue
        except TypeError:
            pass
        messages.append(_format_spike(item))

    return messages


def send_telegram_message(text):
    """Send a single message to the configured Telegram chat."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning(
            "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set — skipping Telegram notification. "
            "Add them to your .env file to enable this."
        )
        return False

    url = TELEGRAM_API_URL.format(token=TELEGRAM_BOT_TOKEN)
    try:
        resp = requests.post(url, data={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }, timeout=15)
        if resp.status_code != 200:
            logger.error("Telegram send failed: %s %s", resp.status_code, resp.text)
            return False
        return True
    except requests.RequestException as e:
        logger.error("Telegram request error: %s", e)
        return False


def notify_signals(signal_summary):
    """
    Main entry point — call this from main.py after run_signal_detection().
    Sends one Telegram message per detected signal (batched isn't done here
    to keep each alert scannable; adjust to a single combined message if
    you'd rather get one ping per run instead of several).
    """
    messages = build_alert_messages(signal_summary)

    if not messages:
        logger.info("No signals above threshold — no Telegram notification sent.")
        return

    logger.info("Sending %d signal notification(s) to Telegram...", len(messages))
    for msg in messages:
        send_telegram_message(msg)


# ---------------------------------------------------------------------------
# INTEGRATION SNIPPET — add this to main.py inside the
# "5. Signal Detection" section, right after signal_summary is computed:
#
#     from src.pipeline.signals import run_signal_detection
#     signal_summary = run_signal_detection()
#
#     from src.notify.telegram import notify_signals   # <-- add this line
#     notify_signals(signal_summary)                    # <-- add this line
#
#     sr = len(signal_summary.get("sector_reversals", []))
#     cs = len(signal_summary.get("company_spikes", []))
#     ...
# ---------------------------------------------------------------------------
