"""Production SRE Webhook Alerting Dispatcher for Discord, Slack, and Telegram."""
from __future__ import annotations
import json
import os
import threading
from typing import Any, Dict, Optional
from loguru import logger
import requests


def escape_telegram_markdown(text: str) -> str:
    """Escapes markdown formatting characters that cause Telegram parse errors."""
    for ch in ("_", "*", "[", "]", "`"):
        text = text.replace(ch, f"\\{ch}")
    return text


def send_discord_alert(webhook_url: str, message: str, level: str = "CRITICAL") -> bool:
    """Dispatches formatted embed or text alert to Discord webhook."""
    if not webhook_url:
        return False
    color = 0xFF0000 if level == "CRITICAL" else 0xFFA500
    payload = {
        "content": f"🚨 **[{level}] Bio-Quant Sovereign SRE Alert**",
        "embeds": [{
            "title": f"Bio-Quant Alert: {level}",
            "description": message,
            "color": color,
        }],
    }
    try:
        resp = requests.post(webhook_url, json=payload, timeout=3.0)
        return resp.status_code in (200, 204)
    except Exception as exc:
        logger.error(f"[Alert] Discord dispatch error: {exc}")
        return False


def send_slack_alert(webhook_url: str, message: str, level: str = "CRITICAL") -> bool:
    """Dispatches formatted message payload to Slack webhook."""
    if not webhook_url:
        return False
    payload = {
        "text": f":warning: *[{level}] Bio-Quant SRE Alert*\n{message}",
    }
    try:
        resp = requests.post(webhook_url, json=payload, timeout=3.0)
        return resp.status_code == 200
    except Exception as exc:
        logger.error(f"[Alert] Slack dispatch error: {exc}")
        return False


def send_telegram_alert(bot_token: str, chat_id: str, message: str, level: str = "CRITICAL") -> bool:
    """Dispatches alert message to Telegram bot chat with Markdown safety and auto-fallback."""
    if not (bot_token and chat_id):
        return False
    clean_message = escape_telegram_markdown(message)
    text = f"⚠️ *[{level}] Bio-Quant SRE Alert*\n\n{clean_message}"
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    try:
        resp = requests.post(url, json=payload, timeout=3.0)
        if resp.status_code == 200:
            return True
        plain_text = f"⚠️ [{level}] Bio-Quant SRE Alert\n\n{message}"
        plain_resp = requests.post(url, json={"chat_id": chat_id, "text": plain_text}, timeout=3.0)
        return plain_resp.status_code == 200
    except Exception as exc:
        logger.error(f"[Alert] Telegram dispatch error: {exc}")
        return False


def send_critical_alert(
    message: str,
    level: str = "CRITICAL",
    channel: str = "all",
    async_dispatch: bool = False,
) -> Dict[str, bool]:
    """Unified alert dispatcher routing critical events to configured webhooks."""
    if async_dispatch:
        t = threading.Thread(
            target=send_critical_alert,
            args=(message, level, channel, False),
            daemon=True
        )
        t.start()
        return {"discord": True, "slack": True, "telegram": True}

    results = {"discord": False, "slack": False, "telegram": False}

    discord_url = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if (channel in ("all", "discord")) and discord_url:
        results["discord"] = send_discord_alert(discord_url, message, level=level)

    slack_url = os.getenv("SLACK_WEBHOOK_URL", "").strip()
    if (channel in ("all", "slack")) and slack_url:
        results["slack"] = send_slack_alert(slack_url, message, level=level)

    tg_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    tg_chat = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if (channel in ("all", "telegram")) and tg_token and tg_chat:
        results["telegram"] = send_telegram_alert(tg_token, tg_chat, message, level=level)

    logger.info(f"[Alert] Dispatched '{level}' alert. Results: {results}")
    return results
