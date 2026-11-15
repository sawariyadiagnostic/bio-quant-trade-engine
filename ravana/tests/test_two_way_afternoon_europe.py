"""Unit tests for TwoWayAfternoonEuropeTrigger."""
from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from ravana.strategies.base import TriggerAction
from ravana.strategies.two_way_afternoon_europe import TwoWayAfternoonEuropeTrigger


class TestTwoWayAfternoonEuropeTrigger(unittest.TestCase):
    """Test suite for Two-Way Afternoon Europe Open trigger."""

    def setUp(self) -> None:
        idx = pd.date_range("2026-09-01 09:15", "2026-09-01 15:30", freq="5min")
        n = len(idx)
        close = 100.0 + np.cumsum(np.random.normal(0.02, 0.1, n))
        self.df = pd.DataFrame(
            {
                "Open": close - 0.1,
                "High": close + 0.3,
                "Low": close - 0.3,
                "Close": close,
                "Volume": [10000.0] * n,
                "ATR_14": [0.5] * n,
            },
            index=idx,
        )

    def test_signals_generation(self) -> None:
        trigger = TwoWayAfternoonEuropeTrigger()
        sigs = trigger.generate_signals(self.df, "BANKBEES")
        self.assertIsInstance(sigs, list)
        for s in sigs:
            self.assertIn(s.action, (TriggerAction.BUY, TriggerAction.SELL, TriggerAction.EXIT_ALL))


if __name__ == "__main__":
    unittest.main()
