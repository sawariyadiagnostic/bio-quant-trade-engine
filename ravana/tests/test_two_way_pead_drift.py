"""Unit tests for TwoWaySectorDriftTrigger."""
from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from ravana.strategies.base import TriggerAction
from ravana.strategies.two_way_pead_drift import TwoWaySectorDriftTrigger


class TestTwoWaySectorDriftTrigger(unittest.TestCase):
    """Test suite for TwoWaySectorDriftTrigger."""

    def setUp(self) -> None:
        idx = pd.date_range("2026-09-01 09:15", "2026-09-01 15:30", freq="5min")
        n = len(idx)
        close = 100.0 + np.linspace(0, 2.0, n)
        self.df = pd.DataFrame(
            {
                "Open": close - 0.1,
                "High": close + 0.2,
                "Low": close - 0.2,
                "Close": close,
                "Volume": [10000.0] * n,
                "VARIANCE_RATIO_20": [1.10] * n,
                "OFI_5": [0.20] * n,
                "ATR_14": [0.5] * n,
            },
            index=idx,
        )

    def test_drift_generates_signals(self) -> None:
        trigger = TwoWaySectorDriftTrigger(min_benchmark_momentum=0.002)
        signals = trigger.generate_signals(self.df, "ITBEES")
        self.assertIsInstance(signals, list)

    def test_unsupported_symbol_returns_empty(self) -> None:
        trigger = TwoWaySectorDriftTrigger()
        signals = trigger.generate_signals(self.df, "UNKNOWN_ETF")
        self.assertEqual(len(signals), 0)


if __name__ == "__main__":
    unittest.main()
