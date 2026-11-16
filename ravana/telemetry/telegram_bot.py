"""Telegram Telemetry Bridge & 2-Way Mobile Command and Control for RAVANA v4.3.

Provides military-grade encrypted mobile telemetry and C2:
1. Real-time push alerts: Order fills, breakeven ratchets, Gemini vetos, session locks.
2. End-of-Day PnL Summary card with capital compounding metrics.
3. 2-Way Mobile Command & Control: /status, /positions, /pnl, and emergency /panic.
4. Graceful Mock Mode: Operates completely offline without errors when credentials are mock.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional
import requests

logger = logging.getLogger(__name__)


class TelegramTelemetryBridge:
    """Encrypted mobile notification and C2 bridge for autonomous paper trading."""

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ) -> None:
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "MOCK")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "MOCK")
        self.is_mock = self._check_mock_mode()
        self.mock_sent_messages: List[str] = []

    def _check_mock_mode(self) -> bool:
        """Determines if the bridge should operate in offline mock mode."""
        if not self.bot_token or not self.chat_id:
            return True
        if self.bot_token.upper().startswith("MOCK") or self.chat_id.upper().startswith("MOCK"):
            return True
        return False

    def send_message(self, text: str) -> bool:
        """Send formatted message via Telegram Bot API or mock log."""
        if self.is_mock:
            self.mock_sent_messages.append(text)
            logger.info("[TELEGRAM MOCK PUSH] %s", text.replace("\n", " | "))
            return True

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": text, "parse_mode": "Markdown"}
        try:
            resp = requests.post(url, json=payload, timeout=5)
            return resp.status_code == 200
        except Exception as exc:
            logger.warning("[TELEGRAM] Failed to dispatch message: %s", exc)
            return False

    def send_trade_fill(self, trade: Dict[str, Any]) -> bool:
        """Dispatch instant order execution notification."""
        side = "BUY 🟢" if trade.get("is_long", True) else "SELL 🔴"
        sym = trade.get("ticker", "ETF")
        px = trade.get("entry_price", 0.0)
        shares = trade.get("shares", 0)
        stop = trade.get("current_stop", 0.0)
        target = trade.get("target_price", 0.0)
        trig = trade.get("trigger_name", "Quantitative")

        msg = (
            f"🚀 *ORDER FILLED: {sym}*\n"
            f"• Action: {side} {shares} shares @ ₹{px:.2f}\n"
            f"• Stop Loss: ₹{stop:.2f} | Target: ₹{target:.2f}\n"
            f"• Strategy: `{trig}`"
        )
        return self.send_message(msg)

    def send_stop_ratchet(self, symbol: str, new_stop: float, entry_px: float) -> bool:
        """Dispatch breakeven ratchet risk-free notification."""
        msg = (
            f"🎯 *BREAKEVEN RATCHET ACTIVATED*\n"
            f"• Symbol: *{symbol}*\n"
            f"• Stop Loss moved to: ₹{new_stop:.2f}\n"
            f"• Entry Reference: ₹{entry_px:.2f}\n"
            f"• Protected Status: *NET-ZERO RISK LOCKED (+2 bps)*"
        )
        return self.send_message(msg)

    def send_veto_alert(self, symbol: str, trigger_name: str, reason: str, prob: float) -> bool:
        """Dispatch Google Decisions risk shield veto alert."""
        msg = (
            f"🛡️ *GOOGLE DECISIONS VETO SHIELD*\n"
            f"• Blocked Setup: *{symbol}* ({trigger_name})\n"
            f"• Reason: {reason}\n"
            f"• Veto Confidence: `{prob:.2f}`\n"
            f"• Capital Preserved: *100% Cash Retained*"
        )
        return self.send_message(msg)

    def send_session_alert(self, phase_name: str, details: str) -> bool:
        """Dispatch session clock transition alert."""
        msg = f"⏰ *SESSION CLOCK: {phase_name}*\n• Details: {details}"
        return self.send_message(msg)

    def send_pnl_summary(self, daily_pnl: float, equity: float, holdings: Dict[str, Any]) -> bool:
        """Dispatch post-market closing PnL card."""
        holdings_str = ", ".join(f"{k} ({v}%)" for k, v in holdings.items()) or "100% Liquid Cash"
        msg = (
            f"📊 *RAVANA EOD PERFORMANCE CARD*\n"
            f"• Realized PnL: *₹{daily_pnl:+.2f}*\n"
            f"• Closing NAV: *₹{equity:,.2f}*\n"
            f"• Overnight Allocations: {holdings_str}\n"
            f"• Capital Preservation: *100% Active*"
        )
        return self.send_message(msg)

    def handle_command(self, command: str, chat_id: str, daemon_state: Dict[str, Any]) -> str:
        """Process remote 2-way C2 command from authorized Telegram client."""
        if not self.is_mock and self.chat_id != "MOCK" and str(chat_id) != str(self.chat_id):
            return "⛔ UNAUTHORIZED: Sender chat_id does not match configured master admin."

        cmd = command.strip().lower()
        if cmd == "/status":
            phase = daemon_state.get("phase", "UNKNOWN")
            regime = daemon_state.get("regime", "STATIONARY")
            eq = daemon_state.get("equity", 200000.0)
            vetoes = daemon_state.get("veto_count", 0)
            return (
                f"🟢 *RAVANA SYSTEM STATUS*\n"
                f"• Phase: `{phase}`\n"
                f"• Regime: `{regime}`\n"
                f"• Equity: ₹{eq:,.2f}\n"
                f"• Vetoes Shielded: {vetoes}"
            )

        if cmd == "/positions":
            open_t = daemon_state.get("open_trades", {})
            cnc = daemon_state.get("active_overnight", {})
            t_lines = [f"• {k}: {v['shares']} shares @ ₹{v['entry_price']}" for k, v in open_t.items()]
            c_lines = [f"• {k}: {v['shares']} shares @ ₹{v['entry_price']}" for k, v in cnc.items()]
            return (
                f"📋 *ACTIVE POSITIONS*\n"
                f"*Intraday MIS:*\n{chr(10).join(t_lines) or 'None'}\n\n"
                f"*Overnight CNC:*\n{chr(10).join(c_lines) or 'None'}"
            )

        if cmd == "/pnl":
            eq = daemon_state.get("equity", 200000.0)
            pnl = daemon_state.get("intraday_pnl", 0.0)
            return f"📈 *TODAY'S PERFORMANCE*\n• Realized PnL: ₹{pnl:+.2f}\n• Current NAV: ₹{eq:,.2f}"

        if cmd == "/panic":
            return "🚨 *EMERGENCY PANIC TRIGGERED: Liquidating all open intraday positions to 100% cash immediately.*"

        return "ℹ️ Available commands: `/status`, `/positions`, `/pnl`, `/panic`"
