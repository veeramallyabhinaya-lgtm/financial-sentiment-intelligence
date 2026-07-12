"""
Telegram notification module for financial-sentiment-intelligence.

Pushes detected signals (sector reversals + company spikes) to one or more
Telegram chats after each pipeline run, so you don't have to check the
Streamlit dashboard manually.

SETUP:
1. Create a bot via @BotFather on Telegram -> get a bot token
2. Get chat ID(s) via @userinfobot on Telegram (message it from each
   account/group you want alerts sent to)
3. Add to your .env file (see .env.example):
     TELEGRAM_BOT_TOKEN=xxxxx
     TELEGRAM_CHAT_IDS=111111111,222222222,-100333333333
   (single-recipient setups can still use TELEGRAM_CHAT_ID=111111111 —
   both variables are supported, see _load_chat_ids() below)
4. In main.py, after run_signal_detection(), call notify_signals(signal_summary)
   (see integration snippet at the bottom of this file)

Schema confirmed directly from src/pipeline/signals.py: both sector reversal
and company spike signals share the same dict shape (signal_type, name,
direction, score_before, score_after, delta, detected_at, article_ids).
"""

import os
import time
import logging
import requests

logger = logging.getLogger("notify.telegram")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"

# Only notify for signals at or above these thresholds (tune to taste).
# Set to 0 to notify on everything the pipeline already flagged.
MIN_REVERSAL_DELTA = 0.0
MIN_SPIKE_SCORE = 0.0

# Small delay between sends to different chats, to stay well under
# Telegram's rate limits when broadcasting to several recipients.
SEND_DELAY_SECONDS = 0.1


def _load_chat_ids():
    """
    Supports multiple recipients via TELEGRAM_CHAT_IDS (comma-separated),
    while still honoring a single TELEGRAM_CHAT_ID for backward
    compatibility with the original single-chat setup. Duplicates and
    blank entries are ignored; order is preserved.
    """
    ids = []

    multi = os.environ.get("TELEGRAM_CHAT_IDS", "")
    for raw in multi.split(","):
        chat_id = raw.strip()
        if chat_id and chat_id not in ids:
            ids.append(chat_id)

    single = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if single and single not in ids:
        ids.append(single)

    return ids


TELEGRAM_CHAT_IDS = _load_chat_ids()


# Exact schema confirmed from src/pipeline/signals.py — both sector reversal
# and company spike signals share the SAME shape:
#   {
#     "id": str, "signal_type": "sector" | "company", "name": str,
#     "direction": "reversal_up" | "reversal_down" | "spike_positive" | "spike_negative",
#     "score_before": float, "score_after": float, "delta": float,
#     "detected_at": iso timestamp str, "article_ids": list,
#   }

_DIRECTION_LABELS = {
    "reversal_up":     "📈 sector turning bullish",
    "reversal_down":   "📉 sector turning bearish",
    "spike_positive":  "🚀 bullish spike",
    "spike_negative":  "⚠️ bearish spike",
}


def _format_signal(item):
    name = item.get("name", "Unknown")
    direction = item.get("direction", "")
    label = _DIRECTION_LABELS.get(direction, direction or "signal detected")
    before = item.get("score_before")
    after = item.get("score_after")
    delta = item.get("delta")

    if item.get("signal_type") == "sector":
        return (f"🔄 <b>{name}</b> — {label}\n"
                f"   {before:+.2f} → {after:+.2f}  (Δ {delta:+.2f})")
    else:
        return f"{label} — <b>{name}</b>: {after:+.2f} (Δ {delta:+.2f})"


def build_alert_messages(signal_summary):
    """
    Turn a signal_summary dict (as returned by run_signal_detection())
    into a list of formatted Telegram message strings.
    """
    messages = []

    for item in signal_summary.get("sector_reversals", []) or []:
        if abs(item.get("delta", 0)) < MIN_REVERSAL_DELTA:
            continue
        messages.append(_format_signal(item))

    for item in signal_summary.get("company_spikes", []) or []:
        if abs(item.get("score_after", 0)) < MIN_SPIKE_SCORE:
            continue
        messages.append(_format_signal(item))

    return messages


def send_telegram_message(text, chat_ids=None):
    """
    Send a single message to one or more Telegram chats.
    Returns a dict of {chat_id: success_bool} so callers can see exactly
    which recipients did/didn't receive it.
    """
    targets = chat_ids if chat_ids is not None else TELEGRAM_CHAT_IDS

    if not TELEGRAM_BOT_TOKEN or not targets:
        logger.warning(
            "TELEGRAM_BOT_TOKEN and/or chat IDs not set — skipping Telegram notification. "
            "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_IDS (or TELEGRAM_CHAT_ID) in your .env."
        )
        return {}

    url = TELEGRAM_API_URL.format(token=TELEGRAM_BOT_TOKEN)
    results = {}

    for chat_id in targets:
        try:
            resp = requests.post(url, data={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }, timeout=15)
            if resp.status_code == 200:
                results[chat_id] = True
            else:
                results[chat_id] = False
                logger.error("Telegram send failed for chat_id=%s: %s %s",
                             chat_id, resp.status_code, resp.text)
        except requests.RequestException as e:
            results[chat_id] = False
            logger.error("Telegram request error for chat_id=%s: %s", chat_id, e)

        if len(targets) > 1:
            time.sleep(SEND_DELAY_SECONDS)

    return results


def notify_signals(signal_summary):
    """
    Main entry point — call this from main.py after run_signal_detection().
    Sends one Telegram message per detected signal, broadcast to every
    chat ID configured in TELEGRAM_CHAT_IDS / TELEGRAM_CHAT_ID.
    """
    messages = build_alert_messages(signal_summary)

    if not messages:
        logger.info("No signals above threshold — no Telegram notification sent.")
        return

    if not TELEGRAM_CHAT_IDS:
        logger.warning("No Telegram chat IDs configured — skipping notification.")
        return

    logger.info("Sending %d signal notification(s) to %d Telegram chat(s)...",
                len(messages), len(TELEGRAM_CHAT_IDS))

    for msg in messages:
        results = send_telegram_message(msg)
        failed = [cid for cid, ok in results.items() if not ok]
        if failed:
            logger.warning("Failed to deliver to chat_id(s): %s", ", ".join(failed))


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
