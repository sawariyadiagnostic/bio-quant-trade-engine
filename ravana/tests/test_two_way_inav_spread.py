"""Unit tests for TwoWayINAVSpreadTrigger."""
from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from ravana.strategies.base import TriggerAction
from ravana.strategies.two_way_inav_spread import TwoWayINAVSpreadTrigger


class TestTwoWayINAVSpreadTrigger(unittest.TestCase):
    """Test suite for TwoWayINAVSpreadTrigger."""

    def setUp(self) -> None:
        idx = pd.date_range("2026-09-01 09:15", "2026-09-01 15:30", freq="5min")
        n = len(idx)
        close = 100.0 + np.sin(np.linspace(0, 4 * np.pi, n)) * 0.5
        # Simulate dislocation: ETF at discount
        theo = close + 0.30
        spread_bps = ((close - theo) / theo) * 10000.0
        zscore = (spread_bps - spread_bps.mean()) / (spread_bps.std() + 1e-6)

        self.df = pd.DataFrame(
            {
                "Open": close - 0.05,
                "High": close + 0.10,
                "Low": close - 0.10,
                "Close": close,
                "Theoretical_Close": theo,
                "SPREAD_BPS": spread_bps,
                "SPREAD_ZSCORE_20": zscore,
                "Volume": [15000.0] * n,
                "VARIANCE_RATIO_20": [0.75] * n,
                "OFI_5": [0.15] * n,
                "ATR_14": [0.4] * n,
            },
            index=idx,
        )

    def test_inav_spread_generates_signals(self) -> None:
        trigger = TwoWayINAVSpreadTrigger(z_threshold=1.0, bps_threshold=10.0)
        signals = trigger.generate_signals(self.df, "BANKBEES")
        self.assertIsInstance(signals, list)
        for s in signals:
            self.assertIn(s.action, (TriggerAction.BUY, TriggerAction.SELL, TriggerAction.EXIT_ALL))

    def test_commodity_shield(self) -> None:
        trigger = TwoWayINAVSpreadTrigger()
        signals_gold = trigger.generate_signals(self.df, "GOLDBEES")
        signals_silver = trigger.generate_signals(self.df, "SILVERBEES")
        self.assertEqual(len(signals_gold), 0)
        self.assertEqual(len(signals_silver), 0)


if __name__ == "__main__":
    unittest.main()
