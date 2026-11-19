"""Unit tests for LiquidSweepManager in RAVANA v4.3."""
import unittest
import pandas as pd
from ravana.execution.liquid_sweep_manager import (
    LiquidSweepManager,
    SweepConfig,
    SweepOrderIntent,
)


class TestLiquidSweepManager(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = LiquidSweepManager()
        self.sweep_time = pd.Timestamp("2026-11-19 15:22:00+05:30")
        self.pre_sweep_time = pd.Timestamp("2026-11-19 14:30:00+05:30")

    def test_eod_sweep_threshold_trigger(self) -> None:
        """Verify sweep triggers only after 15:22 IST when cash exceeds threshold."""
        # Pre-sweep time: should return None
        res_early = self.manager.evaluate_eod_sweep(self.pre_sweep_time, free_cash_inr=50000.0)
        self.assertIsNone(res_early)

        # Cash below Rs. 25,000 threshold at 15:22: should return None
        res_small = self.manager.evaluate_eod_sweep(self.sweep_time, free_cash_inr=15000.0)
        self.assertIsNone(res_small)

        # Cash of Rs. 60,000: sweeps (60,000 - 10,000 buffer) = 50,000 / 1000 = 50 units
        res = self.manager.evaluate_eod_sweep(self.sweep_time, free_cash_inr=60000.0)
        self.assertIsNotNone(res)
        self.assertEqual(res.action, "BUY_SWEEP")
        self.assertEqual(res.units, 50)
        self.assertEqual(res.notional_inr, 50000.0)
        self.assertEqual(res.ticker, "LIQUIDBEES")

    def test_daily_accrued_yield_calculation(self) -> None:
        """Verify daily accrued yield matches 6.5% p.a. TREPS repo rate."""
        # 50 units = Rs. 50,000 notional. Daily yield ~ 12.50 to 12.90 INR
        accrued = self.manager.compute_daily_accrued_yield(held_units=50, calendar_days=1)
        self.assertGreater(accrued, 12.0)
        self.assertLess(accrued, 14.0)

    def test_morning_margin_release(self) -> None:
        """Verify morning release redeems only needed units when cash deficit occurs."""
        morning_time = pd.Timestamp("2026-11-20 09:12:00+05:30")
        # Required capital: 45,000, current cash: 10,000 -> Deficit: 35,000 -> 35 units needed
        release = self.manager.evaluate_morning_release(
            current_time=morning_time,
            required_capital_inr=45000.0,
            current_free_cash_inr=10000.0,
            held_liquid_units=50,
        )
        self.assertIsNotNone(release)
        self.assertEqual(release.action, "SELL_RELEASE")
        self.assertEqual(release.units, 35)
        self.assertEqual(release.notional_inr, 35000.0)


if __name__ == "__main__":
    unittest.main()
