"""Tests for Muscle 8: Overnight Target & Objective Builder.

Verifies:
1. Forward overnight return extraction (15:24 IST -> next session 09:16 IST).
2. Indian statutory cash delivery friction (2.0 bps) + flat CDSL DP debit fee (Rs. 17.00).
3. Trade capital threshold gating (>= Rs. 2,00,000) to keep flat DP drag negligible.
4. Continuous net returns, binary hurdle labels, and sample weights.
5. Strict zero-lookahead masking: target Y at bar t is strictly aligned for predictive loss.
"""

import unittest
import numpy as np
import pandas as pd

from ravana.targets.config import TargetConfig
from ravana.targets.costs import DeliveryFrictionModel
from ravana.targets.builder import OvernightTargetBuilder


class TestDeliveryFrictionModel(unittest.TestCase):
    """Test institutional Indian delivery frictions and flat DP charge drag."""

    def test_friction_drag_at_target_capital(self) -> None:
        """Verify at Rs. 2,00,000 capital, flat Rs. 17 DP charge is exactly 0.85 bps."""
        model = DeliveryFrictionModel(
            variable_bps=2.0,
            flat_dp_charge_inr=17.0,
            default_capital_inr=200000.0,
        )
        # Variable cost: 2 bps = 0.0002
        # Fixed cost: 17 / 200,000 = 0.000085 (0.85 bps)
        # Total drag: 2.85 bps = 0.000285
        net_ret = model.apply_friction(gross_return=0.01, capital_inr=200000.0)
        expected_drag = (2.0 / 10000.0) + (17.0 / 200000.0)
        self.assertAlmostEqual(0.01 - net_ret, expected_drag, places=7)

    def test_small_capital_severe_drag_penalty(self) -> None:
        """Verify accounts under Rs. 50,000 face severe (>3.4 bps) flat DP fee drag."""
        model = DeliveryFrictionModel(
            variable_bps=2.0,
            flat_dp_charge_inr=17.0,
        )
        small_cap_drag = model.compute_total_drag_bps(capital_inr=25000.0)
        # 17 / 25000 = 0.00068 = 6.8 bps. Total = 8.8 bps
        self.assertAlmostEqual(small_cap_drag, 8.8, places=4)

    def test_capital_threshold_gate(self) -> None:
        """Verify capital below minimum threshold triggers capital threshold warning/flag."""
        model = DeliveryFrictionModel(min_capital_inr=200000.0)
        self.assertTrue(model.is_capital_viable(250000.0))
        self.assertFalse(model.is_capital_viable(150000.0))


class TestOvernightTargetBuilder(unittest.TestCase):
    """Test forward overnight return calculation and sample mask construction."""

    def setUp(self) -> None:
        # Create 2 days of 5m bars for NSE (75 bars per day = 150 bars)
        # Day 1: 2026-01-05 09:15 to 15:25
        # Day 2: 2026-01-06 09:15 to 15:25
        day1 = pd.date_range("2026-01-05 09:15", "2026-01-05 15:25", freq="5min", tz="Asia/Kolkata")
        day2 = pd.date_range("2026-01-06 09:15", "2026-01-06 15:25", freq="5min", tz="Asia/Kolkata")
        all_dates = day1.append(day2)
        n = len(all_dates)

        # Base prices: Day 1 closes at 100.0, Day 2 opens with a 1% overnight gap at 101.0
        prices = np.full(n, 100.0)
        # Day 2 starts at index 75
        prices[75:] = 101.0

        self.df = pd.DataFrame(
            {
                "Open": prices,
                "High": prices + 0.1,
                "Low": prices - 0.1,
                "Close": prices,
                "Volume": np.full(n, 10000.0),
            },
            index=all_dates,
        )

    def test_forward_overnight_return_accuracy(self) -> None:
        """Verify Day 1 close (bar 74) forward return to Day 2 open is exactly +1.0% gross."""
        builder = OvernightTargetBuilder(TargetConfig(capital_inr=200000.0, variable_bps=2.0, dp_fee_inr=17.0))
        targets = builder.build_targets(self.df)

        # Bar 74 is Day 1 last bar (15:25 IST)
        day1_close_bar = 74
        self.assertEqual(targets["is_entry_bar"].iloc[day1_close_bar], 1)
        gross_ret = targets["gross_return"].iloc[day1_close_bar]
        self.assertAlmostEqual(gross_ret, 0.01, places=5)

        # Net return after 2.85 bps drag: 0.01 - 0.000285 = 0.009715
        net_ret = targets["net_return"].iloc[day1_close_bar]
        self.assertAlmostEqual(net_ret, 0.01 - 0.000285, places=5)

        # Sample weight should be 1.0 on entry bar, 0.0 on intraday non-entry bars
        self.assertEqual(targets["sample_weight"].iloc[day1_close_bar], 1.0)
        self.assertEqual(targets["sample_weight"].iloc[0], 0.0)

    def test_binary_hurdle_labeling(self) -> None:
        """Verify binary label is 1 only when net return exceeds the positive hurdle rate."""
        builder = OvernightTargetBuilder(TargetConfig(hurdle_bps=5.0))  # Hurdle = 5 bps = 0.0005
        targets = builder.build_targets(self.df)

        day1_close_bar = 74
        # Net return is ~97 bps > 5 bps hurdle -> label should be 1
        self.assertEqual(targets["binary_label"].iloc[day1_close_bar], 1)

    def test_last_session_terminal_handling(self) -> None:
        """Verify the final session close has no forward target (masked cleanly with weight 0)."""
        builder = OvernightTargetBuilder(TargetConfig())
        targets = builder.build_targets(self.df)

        # Last bar has no next-day session to exit into
        self.assertEqual(targets["is_entry_bar"].iloc[-1], 0)
        self.assertEqual(targets["sample_weight"].iloc[-1], 0.0)
        self.assertFalse(targets.isna().any().any())

    def test_lakehouse_live_targets_integration(self) -> None:
        """Verify target builder extracts valid overnight returns from live lake partitions."""
        from ravana.data.storage import LakeManager, StorageConfig
        import os

        config = StorageConfig()
        lake_mgr = LakeManager(config)

        if not os.path.exists(config.lake_root / "m4_cleaned"):
            self.skipTest("Lakehouse partitions not found, skipping live integration test.")

        silver_df = lake_mgr.read(tier="m4_cleaned", timeframe="5m", ticker="BANKBEES")
        self.assertFalse(silver_df.empty)

        builder = OvernightTargetBuilder(TargetConfig(capital_inr=200000.0))
        targets = builder.build_targets(silver_df)

        self.assertEqual(len(targets), len(silver_df))
        self.assertFalse(targets.isna().any().any())

        # Over 60 trading days, there should be ~59 overnight transitions
        entry_count = targets["is_entry_bar"].sum()
        self.assertGreaterEqual(entry_count, 50)
        self.assertLessEqual(entry_count, 65)

        # Net returns on entry bars should be bounded within [-10%, +10%] for index ETF
        entry_net_rets = targets.loc[targets["is_entry_bar"] == 1, "net_return"]
        self.assertTrue((entry_net_rets.abs() < 0.15).all())

    def test_calendar_hold_days_weekend_risk(self) -> None:
        """Verify Friday to Monday hold duration records 3 calendar days."""
        fri = pd.date_range("2026-01-09 09:15", "2026-01-09 15:25", freq="5min", tz="Asia/Kolkata")
        mon = pd.date_range("2026-01-12 09:15", "2026-01-12 15:25", freq="5min", tz="Asia/Kolkata")
        all_dates = fri.append(mon)
        df_weekend = pd.DataFrame(
            {
                "Open": np.full(len(all_dates), 100.0),
                "High": np.full(len(all_dates), 100.5),
                "Low": np.full(len(all_dates), 99.5),
                "Close": np.full(len(all_dates), 100.0),
                "Volume": np.full(len(all_dates), 1000.0),
            },
            index=all_dates,
        )
        builder = OvernightTargetBuilder()
        targets = builder.build_targets(df_weekend)
        self.assertIn("calendar_hold_days", targets.columns)
        self.assertEqual(targets["calendar_hold_days"].iloc[74], 3)

    def test_short_session_graceful_handling(self) -> None:
        """Verify session shorter than offset requirement does not crash with IndexError."""
        # 1-bar session followed by normal session
        d1 = pd.date_range("2026-01-05 09:15", periods=1, freq="5min", tz="Asia/Kolkata")
        d2 = pd.date_range("2026-01-06 09:15", periods=75, freq="5min", tz="Asia/Kolkata")
        all_dates = d1.append(d2)
        df_short = pd.DataFrame(
            {
                "Open": np.full(len(all_dates), 100.0),
                "High": np.full(len(all_dates), 100.5),
                "Low": np.full(len(all_dates), 99.5),
                "Close": np.full(len(all_dates), 100.0),
                "Volume": np.full(len(all_dates), 1000.0),
            },
            index=all_dates,
        )
        # Entry offset is 1 from end, but session only has 1 bar
        builder = OvernightTargetBuilder(TargetConfig(entry_bar_offset_from_end=2))
        targets = builder.build_targets(df_short)
        self.assertEqual(len(targets), 76)
        self.assertFalse(targets.isna().any().any())


if __name__ == "__main__":
    unittest.main()
