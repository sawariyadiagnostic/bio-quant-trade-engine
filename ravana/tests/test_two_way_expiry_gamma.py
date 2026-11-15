"""Unit tests for TwoWayExpiryGammaTrigger."""
from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from ravana.strategies.base import TriggerAction
from ravana.strategies.two_way_expiry_gamma import TwoWayExpiryGammaTrigger


class TestTwoWayExpiryGammaTrigger(unittest.TestCase):
    """Test suite for Two-Way Thursday Expiry Gamma trigger."""

    def setUp(self) -> None:
        # Thursday 2026-09-03
        idx = pd.date_range("2026-09-03 09:15", "2026-09-03 15:30", freq="5min")
        n = len(idx)
        close = 100.0 + np.cumsum(np.random.normal(0.05, 0.2, n))
        self.df = pd.DataFrame(
            {
                "Open": close - 0.1,
                "High": close + 0.5,
                "Low": close - 0.5,
                "Close": close,
                "Volume": [20000.0] * n,
                "ATR_14": [0.5] * n,
            },
            index=idx,
        )

    def test_signals_generation(self) -> None:
        trigger = TwoWayExpiryGammaTrigger()
        sigs = trigger.generate_signals(self.df, "BANKBEES")
        self.assertIsInstance(sigs, list)
        for s in sigs:
            self.assertIn(s.action, (TriggerAction.BUY, TriggerAction.SELL, TriggerAction.EXIT_ALL))

    def test_non_index_asset_returns_empty(self) -> None:
        trigger = TwoWayExpiryGammaTrigger()
        sigs = trigger.generate_signals(self.df, "ITBEES")
        self.assertEqual(sigs, [])


if __name__ == "__main__":
    unittest.main()
