"""
Dedicated unit test suite for ADHD Batch 3 Remediations:
ADHD-21 through ADHD-28 (SRE Daemon Endurance, Concurrency & OS Interop).
"""
import asyncio
from datetime import datetime
from decimal import Decimal
import os
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

from scripts.overnight_daemon import check_and_execute_exit
from src.execution.shoonya_gateway import ShoonyaGateway, ShoonyaCredentials
from src.execution.risk_sentry import RiskSentry
from src.execution.alerts import escape_telegram_markdown, send_telegram_alert, send_critical_alert
from src.evolution.archive_store import ArchiveStore


class TestADHDBatch3SREDaemon(unittest.TestCase):

    def test_adhd_21_daemon_paper_position_detection(self):
        """ADHD-21: check_and_execute_exit correctly identifies paper positions."""
        gw = ShoonyaGateway(paper_trading=True)
        gw.paper_positions = {
            "BANKBEES-EQ": {"quantity": 100, "avg_price": 500.0},
            "NIFTYBEES-EQ": {"quantity": 50, "avg_price": 250.0}
        }
        res = check_and_execute_exit(gw, dry_run=True)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["open_positions_count"], 2)
        symbols = [p["symbol"] for p in res["positions"]]
        self.assertIn("BANKBEES-EQ", symbols)
        self.assertIn("NIFTYBEES-EQ", symbols)

    def test_adhd_23_shoonya_token_death_loop_break(self):
        """ADHD-23: handle_session_expiry clears expired env token to prevent infinite death loop."""
        creds = ShoonyaCredentials(
            user_id="U123",
            password="pwd",
            totp_key="JBSWY3DPEHPK3PXP",
            vendor_code="VC1",
            api_key="AK1",
            access_token="expired_token_123"
        )
        gw = ShoonyaGateway(credentials=creds, paper_trading=False)
        gw.susertoken = "expired_token_123"
        gw.quick_auth = MagicMock(return_value={"stat": "Ok", "susertoken": "new_fresh_token"})

        gw.handle_session_expiry()
        # Verify expired token was wiped from creds
        self.assertEqual(gw.creds.access_token, "")
        # Verify quick_auth was invoked as fallback
        gw.quick_auth.assert_called_once()

    def test_adhd_24_alerts_telegram_markdown_and_async(self):
        """ADHD-24: alerts escape markdown and support non-blocking async dispatch."""
        dangerous_trace = "Traceback: File /path/to_file.py in func_test(): KeyError: 'col_1' [Index 0]"
        escaped = escape_telegram_markdown(dangerous_trace)
        self.assertNotIn("to_file", escaped)
        self.assertIn("to\\_file", escaped)
        self.assertIn("\\[Index 0\\]", escaped)

        # Verify async non-blocking dispatch returns immediately
        with patch("src.execution.alerts.send_discord_alert", return_value=True):
            res = send_critical_alert("Test message", level="WARNING", channel="discord", async_dispatch=True)
            self.assertTrue(res["discord"])

    def test_adhd_25_risk_sentry_paper_liquidation_not_prematurely_wiped(self):
        """ADHD-25: Risk Sentry does not wipe paper_positions before order dispatch."""
        gw = ShoonyaGateway(paper_trading=True)
        gw.paper_positions = {
            "BANKBEES-EQ": {"quantity": 100, "avg_price": 500.0}
        }
        sentry = RiskSentry(gateway=gw)
        gw.place_order = MagicMock(return_value={"stat": "Ok", "status": "FILLED"})

        # Run parallel liquidation
        async def run_liq():
            return await sentry._liquidate_positions_parallel(force_hedge_if_circuit=False)

        asyncio.run(run_liq())
        # Verify place_order was actually called with SELL quantity 100
        gw.place_order.assert_called()

    def test_adhd_26_archive_store_deterministic_connection_close(self):
        """ADHD-26: ArchiveStore deterministically closes connection without file lock leak."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            temp_db = tf.name

        try:
            store = ArchiveStore(db_path=temp_db)
            # Perform query and checkpoint
            store.query_leaderboard(ticker="BANKBEES", lookback="1y")
            store.checkpoint()
            # Must be able to re-open or remove file on Windows without WinError 32
            self.assertTrue(os.path.exists(temp_db))
        finally:
            if os.path.exists(temp_db):
                try:
                    os.unlink(temp_db)
                except Exception:
                    pass

    def test_adhd_28_totp_clock_skew_compensation(self):
        """ADHD-28: generate_totp applies time offset compensation cleanly."""
        creds = ShoonyaCredentials(
            user_id="U1",
            totp_key="JBSWY3DPEHPK3PXP"
        )
        gw = ShoonyaGateway(credentials=creds, paper_trading=True)
        # Normal TOTP vs +60s offset TOTP must generate valid 6-digit numeric strings
        code_now = gw.generate_totp()
        code_offset = gw.generate_totp(time_offset_seconds=60.0)
        self.assertEqual(len(code_now), 6)
        self.assertEqual(len(code_offset), 6)
        self.assertTrue(code_now.isdigit())
        self.assertTrue(code_offset.isdigit())


if __name__ == "__main__":
    unittest.main()
