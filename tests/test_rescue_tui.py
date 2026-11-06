"""Unit and Functional Tests for Companion Headless Terminal Rescue TUI.

Validates:
1. Terminal Avionics dashboard layout rendering with Rich console.
2. Direct SharedMemory telemetry and SQLite WAL leaderboard display.
3. Single-keystroke 'Key: K' instant emergency freeze and liquidation trigger.
4. Single-keystroke 'Key: A' instant atomic abort trigger.
"""
import io
import unittest
from unittest.mock import MagicMock, patch, AsyncMock

from rich.console import Console

from src.pipeline.shared_telemetry import SharedMemoryBus
from src.ui.rescue_tui import RescueTUI


class TestRescueTUI(unittest.TestCase):
    def setUp(self):
        self.bus = SharedMemoryBus(create=True)
        self.mock_sentry = MagicMock()
        self.mock_sentry.freeze_stage1.return_value = {"status": "FROZEN", "elapsed_us": 4.2}
        self.mock_sentry.execute_stage2_liquidate = AsyncMock(return_value={
            "status": "LIQUIDATION_COMPLETE",
            "orders_cancelled_count": 3,
            "positions_liquidated_count": 1,
            "cash_secured": True
        })

    def tearDown(self):
        self.bus.close()

    def test_rescue_tui_render_snapshot(self):
        """Rescue TUI renders rich avionics layout with HUD metrics and champions."""
        # Setup telemetry in shared memory
        self.bus.update_header(
            current_gen=7,
            total_gens=10,
            resident_count=42,
            survivor_count=28,
            occupancy_pct=10.5,
            top_sharpe=2.85,
            best_net_return=45.2,
            worker_status=1
        )

        test_console = Console(file=io.StringIO(), force_terminal=True, width=120)
        tui = RescueTUI(
            shm_bus=self.bus,
            sentry=self.mock_sentry,
            console=test_console
        )

        output = tui.render_snapshot()
        self.assertIn("BIO-QUANT TERMINAL RESCUE TUI", output)
        self.assertIn("7/10", output)
        self.assertIn("2.85", output)
        self.assertIn("[K]", output)
        self.assertIn("EMERGENCY PANIC", output)
        self.assertIn("[A]", output)
        self.assertIn("ABORT WORKER", output)

    def test_keystroke_k_panic_trigger(self):
        """Single keystroke 'K' triggers Stage 1 freeze and Stage 2 liquidation."""
        tui = RescueTUI(shm_bus=self.bus, sentry=self.mock_sentry)
        handled = tui.handle_key("k")
        self.assertTrue(handled)
        self.mock_sentry.freeze_stage1.assert_called_once()
        self.mock_sentry.execute_stage2_liquidate.assert_called_once()

    def test_keystroke_a_abort_trigger(self):
        """Single keystroke 'A' flags abort requested in shared memory."""
        tui = RescueTUI(shm_bus=self.bus, sentry=self.mock_sentry)
        self.assertEqual(self.bus.header.abort_requested, 0)
        handled = tui.handle_key("a")
        self.assertTrue(handled)
        self.assertEqual(self.bus.header.abort_requested, 1)


if __name__ == "__main__":
    unittest.main()
