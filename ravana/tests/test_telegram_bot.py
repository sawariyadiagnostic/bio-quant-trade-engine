"""Unit tests for Telegram Telemetry Bridge and Mobile C2 Command Router."""
import unittest
from unittest.mock import MagicMock, patch
from ravana.telemetry.telegram_bot import TelegramTelemetryBridge


class TestTelegramTelemetryBridge(unittest.TestCase):
    """Test suite for TelegramTelemetryBridge."""

    def setUp(self) -> None:
        # Mock mode when token is MOCK or missing
        self.bridge = TelegramTelemetryBridge(bot_token="MOCK_TOKEN", chat_id="123456789")

    def test_mock_mode_active(self) -> None:
        """Verifies mock mode operates safely without network calls."""
        self.assertTrue(self.bridge.is_mock)

    def test_send_trade_fill_mock(self) -> None:
        """Trade fill message formats properly and returns True in mock mode."""
        trade = {
            "ticker": "PHARMABEES",
            "shares": 150,
            "entry_price": 38.25,
            "current_stop": 37.95,
            "target_price": 38.85,
            "trigger_name": "TwoWayChopMeanReversion_Long",
            "is_long": True,
        }
        ok = self.bridge.send_trade_fill(trade)
        self.assertTrue(ok)
        self.assertEqual(len(self.bridge.mock_sent_messages), 1)
        self.assertIn("PHARMABEES", self.bridge.mock_sent_messages[0])
        self.assertIn("BUY", self.bridge.mock_sent_messages[0])

    def test_send_stop_ratchet_mock(self) -> None:
        """Breakeven ratchet notification formats correctly."""
        ok = self.bridge.send_stop_ratchet(symbol="AUTOBEES", new_stop=284.50, entry_px=284.00)
        self.assertTrue(ok)
        self.assertIn("AUTOBEES", self.bridge.mock_sent_messages[-1])
        self.assertIn("BREAKEVEN", self.bridge.mock_sent_messages[-1])

    def test_send_veto_alert_mock(self) -> None:
        """Google Decisions Veto Shield alert formats correctly."""
        ok = self.bridge.send_veto_alert(
            symbol="AUTOBEES",
            trigger_name="TwoWayORB",
            reason="P(FalseBreakout) >= 0.72",
            prob=0.78,
        )
        self.assertTrue(ok)
        self.assertIn("VETO", self.bridge.mock_sent_messages[-1])
        self.assertIn("0.78", self.bridge.mock_sent_messages[-1])

    def test_send_session_alert_mock(self) -> None:
        """Session lifecycle transition alerts format properly."""
        ok = self.bridge.send_session_alert(phase_name="MORNING_LOCK_1130", details="All open MIS trades squared off.")
        self.assertTrue(ok)
        self.assertIn("MORNING_LOCK_1130", self.bridge.mock_sent_messages[-1])

    def test_send_pnl_summary_mock(self) -> None:
        """Daily closing PnL card formats properly."""
        ok = self.bridge.send_pnl_summary(
            daily_pnl=37.96,
            equity=200037.96,
            holdings={"PSUBNKBEES": 35.0, "BANKBEES": 22.5, "LIQUIDBEES": 42.5},
        )
        self.assertTrue(ok)
        self.assertIn("₹200,037.96", self.bridge.mock_sent_messages[-1])

    def test_handle_command_status(self) -> None:
        """Processes /status remote command."""
        daemon_state = {
            "phase": "MORNING_EXPANSION",
            "regime": "CHOP (VR: 0.62)",
            "equity": 200037.96,
            "open_trades_count": 1,
            "veto_count": 4,
        }
        res = self.bridge.handle_command("/status", chat_id="123456789", daemon_state=daemon_state)
        self.assertIn("MORNING_EXPANSION", res)
        self.assertIn("200,037.96", res)

    def test_handle_command_positions(self) -> None:
        """Processes /positions remote command."""
        daemon_state = {
            "open_trades": {
                "PHARMABEES": {"shares": 100, "entry_price": 38.0, "current_stop": 37.8, "target_price": 38.5}
            },
            "active_overnight": {"PSUBNKBEES": {"shares": 200, "entry_price": 85.0}},
        }
        res = self.bridge.handle_command("/positions", chat_id="123456789", daemon_state=daemon_state)
        self.assertIn("PHARMABEES", res)
        self.assertIn("PSUBNKBEES", res)

    def test_handle_command_panic(self) -> None:
        """Processes /panic emergency liquidation remote command."""
        daemon_state = {"equity": 200000.0}
        res = self.bridge.handle_command("/panic", chat_id="123456789", daemon_state=daemon_state)
        self.assertIn("PANIC", res)
        self.assertIn("EMERGENCY", res)

    def test_handle_command_unauthorized(self) -> None:
        """Rejects commands from unauthorized chat IDs."""
        real_bridge = TelegramTelemetryBridge(bot_token="REAL_TOKEN_123", chat_id="123456789")
        res = real_bridge.handle_command("/status", chat_id="999999999", daemon_state={})
        self.assertIn("UNAUTHORIZED", res)


if __name__ == "__main__":
    unittest.main()
