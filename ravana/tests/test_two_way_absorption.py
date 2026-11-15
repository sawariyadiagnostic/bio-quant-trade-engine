"""Unit tests for TwoWayVWAPAbsorptionTrigger and TwoWayLiquidityStopSweepTrigger."""
from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from ravana.strategies.base import TriggerAction
from ravana.strategies.two_way_absorption import (
    TwoWayLiquidityStopSweepTrigger,
    TwoWayVWAPAbsorptionTrigger,
)


class TestTwoWayAbsorptionTriggers(unittest.TestCase):
    """Test suite for Two-Way Absorption and Stop-Sweep triggers."""

    def setUp(self) -> None:
        idx = pd.date_range("2026-09-01 09:15", "2026-09-01 15:30", freq="5min")
        n = len(idx)
        # Synthetic mean-reverting series
        close = 100.0 + np.sin(np.linspace(0, 4 * np.pi, n)) * 2.0
        self.df = pd.DataFrame(
            {
                "Open": close - 0.1,
                "High": close + 0.3,
                "Low": close - 0.3,
                "Close": close,
                "Volume": [10000.0] * n,
                "VARIANCE_RATIO_20": [0.75] * n,
                "OFI_5": [0.20] * n,
                "ATR_14": [0.5] * n,
            },
            index=idx,
        )

    def test_vwap_absorption_generates_signals(self) -> None:
        trigger = TwoWayVWAPAbsorptionTrigger()
        signals = trigger.generate_signals(self.df, "ITBEES")
        self.assertIsInstance(signals, list)
        for s in signals:
            self.assertIn(s.action, (TriggerAction.BUY, TriggerAction.SELL, TriggerAction.EXIT_ALL))

    def test_commodity_shield(self) -> None:
        trigger = TwoWayVWAPAbsorptionTrigger()
        signals_gold = trigger.generate_signals(self.df, "GOLDBEES")
        signals_silver = trigger.generate_signals(self.df, "SILVERBEES")
        self.assertEqual(len(signals_gold), 0)
        self.assertEqual(len(signals_silver), 0)

    def test_stop_sweep_generates_signals(self) -> None:
        trigger = TwoWayLiquidityStopSweepTrigger(sweep_window=5)
        signals = trigger.generate_signals(self.df, "BANKBEES")
        self.assertIsInstance(signals, list)


if __name__ == "__main__":
    unittest.main()
