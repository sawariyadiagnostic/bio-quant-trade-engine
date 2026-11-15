"""Unit tests for Two-Way Morning Alpha and Overnight Bear Market Inverse Engine."""
import unittest
import numpy as np
import pandas as pd
from ravana.strategies.base import TriggerAction
from ravana.strategies.bear_inverse import AllWeatherOvernightAllocator
from ravana.strategies.two_way_orb import TwoWayMorningORBTrigger


class TestTwoWayMorningORBTrigger(unittest.TestCase):
    def setUp(self) -> None:
        self.trigger = TwoWayMorningORBTrigger(ofi_threshold=0.20, vr_threshold=0.85)

        # Construct a synthetic 5-minute morning dataset
        idx = pd.date_range("2026-11-13 09:15", "2026-11-13 11:35", freq="5min", tz="Asia/Kolkata")
        n = len(idx)
        # Establish OR between 09:15 and 09:25: High=101, Low=99
        df = pd.DataFrame(
            {
                "Open": [100.0] * n,
                "High": [100.5] * n,
                "Low": [99.5] * n,
                "Close": [100.0] * n,
                "Volume": [10000.0] * n,
                "VARIANCE_RATIO_20": [1.10] * n,
                "OFI_5": [0.0] * n,
                "ATR_14": [1.0] * n,
            },
            index=idx,
        )
        # Set 09:15-09:25 range
        df.loc["2026-11-13 09:15", "High"] = 101.0
        df.loc["2026-11-13 09:20", "Low"] = 99.0
        self.df = df

    def test_long_breakout_signal(self) -> None:
        # At 09:35, price breaks above 101.0 with positive OFI
        df = self.df.copy()
        df.loc["2026-11-13 09:35", "Close"] = 101.50
        df.loc["2026-11-13 09:35", "OFI_5"] = 0.35

        sigs = self.trigger.generate_two_way_signals(df, "ITBEES")
        buy_sigs = [s for s in sigs if s.action == TriggerAction.BUY]
        self.assertGreaterEqual(len(buy_sigs), 1)
        first_buy = buy_sigs[0]
        self.assertEqual(first_buy.action, TriggerAction.BUY)
        self.assertEqual(first_buy.trigger_name, "TwoWayMorningORBTrigger_Long")
        self.assertGreaterEqual(first_buy.conviction, 0.70)

    def test_short_breakdown_signal(self) -> None:
        # At 09:40, price breaks below 99.0 with negative OFI
        df = self.df.copy()
        df.loc["2026-11-13 09:40", "Close"] = 98.50
        df.loc["2026-11-13 09:40", "OFI_5"] = -0.35

        sigs = self.trigger.generate_two_way_signals(df, "BANKBEES")
        sell_sigs = [s for s in sigs if s.action == TriggerAction.SELL]
        self.assertGreaterEqual(len(sell_sigs), 1)
        first_sell = sell_sigs[0]
        self.assertEqual(first_sell.action, TriggerAction.SELL)
        self.assertEqual(first_sell.trigger_name, "TwoWayMorningORBTrigger_Short")
        self.assertGreaterEqual(first_sell.conviction, 0.70)

    def test_mandatory_1130_exit(self) -> None:
        sigs = self.trigger.generate_two_way_signals(self.df, "NIFTYBEES")
        exit_sigs = [s for s in sigs if s.action == TriggerAction.EXIT_ALL]
        self.assertEqual(len(exit_sigs), 1)
        self.assertEqual(exit_sigs[0].timestamp.strftime("%H:%M"), "11:30")


class TestAllWeatherOvernightAllocator(unittest.TestCase):
    def setUp(self) -> None:
        self.allocator = AllWeatherOvernightAllocator()

    def test_bull_regime_allocation(self) -> None:
        # Close > EMA_50 and VR >= 0.95
        bench_df = pd.DataFrame(
            {"Close": [25000.0], "EMA_50": [24000.0], "VR_20": [1.15]},
            index=[pd.Timestamp("2026-11-13")],
        )
        regime = self.allocator.determine_regime(bench_df, pd.Timestamp("2026-11-13").date())
        self.assertTrue(regime["is_bull"])
        self.assertFalse(regime["is_bear"])

        ranked = [("ITBEES", 42.0, 1.2), ("AUTOBEES", 280.0, 0.9)]
        targets = self.allocator.compute_allocations(regime, ranked)
        tickers = [t.ticker for t in targets]
        self.assertIn("ITBEES", tickers)
        self.assertIn("AUTOBEES", tickers)
        self.assertNotIn("INVERSE_NIFTY", tickers)

    def test_bear_regime_inverse_allocation(self) -> None:
        # Close < EMA_50 and VR < 0.95
        bench_df = pd.DataFrame(
            {"Close": [22500.0], "EMA_50": [24000.0], "VR_20": [0.85]},
            index=[pd.Timestamp("2026-09-15")],
        )
        regime = self.allocator.determine_regime(bench_df, pd.Timestamp("2026-09-15").date())
        self.assertTrue(regime["is_bear"])

        ranked = [("ITBEES", 38.0, -0.5)]
        targets = self.allocator.compute_allocations(regime, ranked)
        tickers = [t.ticker for t in targets]
        self.assertIn("GOLDBEES", tickers)
        self.assertIn("INVERSE_NIFTY", tickers)
        self.assertIn("LIQUIDBEES", tickers)


if __name__ == "__main__":
    unittest.main()
