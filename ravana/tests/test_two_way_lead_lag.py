"""Unit tests for TwoWayLeadLagTrigger."""
from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from ravana.strategies.base import TriggerAction
from ravana.strategies.two_way_lead_lag import TwoWayLeadLagTrigger


class TestTwoWayLeadLagTrigger(unittest.TestCase):
    """Test suite for Two-Way Cross-Asset Lead-Lag trigger."""

    def setUp(self) -> None:
        idx = pd.date_range("2026-09-01 09:15", "2026-09-01 15:30", freq="5min")
        n = len(idx)
        # Synthetic leader and follower series
        leader_close = 100.0 + np.cumsum(np.random.normal(0.05, 0.2, n))
        follower_close = 100.0 + np.cumsum(np.random.normal(0.01, 0.1, n))

        self.df_leader = pd.DataFrame(
            {
                "Open": leader_close - 0.1,
                "High": leader_close + 0.3,
                "Low": leader_close - 0.3,
                "Close": leader_close,
                "Volume": [10000.0] * n,
                "ATR_14": [0.5] * n,
            },
            index=idx,
        )

        self.df_follower = pd.DataFrame(
            {
                "Open": follower_close - 0.1,
                "High": follower_close + 0.3,
                "Low": follower_close - 0.3,
                "Close": follower_close,
                "Volume": [5000.0] * n,
                "ATR_14": [0.3] * n,
            },
            index=idx,
        )

    def test_unknown_symbol_returns_empty(self) -> None:
        trigger = TwoWayLeadLagTrigger()
        sigs = trigger.generate_signals(self.df_follower, "UNKNOWN_BEES")
        self.assertEqual(sigs, [])

    def test_mock_leader_caching(self) -> None:
        trigger = TwoWayLeadLagTrigger()
        trigger._cached_leaders["BANKBEES"] = self.df_leader
        sigs = trigger.generate_signals(self.df_follower, "PSUBNKBEES")
        self.assertIsInstance(sigs, list)
        for s in sigs:
            self.assertIn(s.action, (TriggerAction.BUY, TriggerAction.SELL, TriggerAction.EXIT_ALL))


if __name__ == "__main__":
    unittest.main()
