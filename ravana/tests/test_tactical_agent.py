"""Unit tests for Tactical Execution Agent in RAVANA v4.2."""
import unittest
import pandas as pd
from ravana.backtest.orders import OrderSide
from ravana.execution.tactical_agent import (
    StrategicIntent,
    TacticalExecutionAgent,
    TacticalExecutionResult,
)


class TestTacticalExecutionAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = TacticalExecutionAgent(default_spread_bps=5.0)
        self.now = pd.Timestamp("2026-11-13 10:15:00+05:30")
        self.intent = StrategicIntent(
            ticker="ITBEES",
            side=OrderSide.BUY,
            shares=1000,
            alpha_timestamp=self.now,
            alpha_reference_price=42.50,
            trigger_name="OpeningRangeBreakoutTrigger",
            conviction=0.85,
            max_wait_bars=3,
            inav_max_premium_bps=15.0,
        )

    def test_passive_limit_price_calculation(self) -> None:
        bar = pd.Series({
            "Open": 42.50,
            "High": 42.60,
            "Low": 42.30,
            "Close": 42.45,
            "VWAP": 42.40,
        })
        limit_px = self.agent.calculate_passive_limit_price(self.intent, bar)
        # Limit price should be at or below close and within bar bounds
        self.assertLessEqual(limit_px, 42.45)
        self.assertGreaterEqual(limit_px, 42.30)

    def test_tactical_fill_on_dip(self) -> None:
        # Bar with low reaching below limit price
        bar = pd.Series({
            "Open": 42.50,
            "High": 42.55,
            "Low": 42.35,
            "Close": 42.45,
            "VWAP": 42.40,
            "SPREAD_BPS": 2.0,
        })
        res = self.agent.evaluate_tactical_fill(self.intent, bar, self.now, bars_waited=1)
        self.assertTrue(res.is_filled)
        self.assertEqual(res.filled_shares, 1000)
        self.assertLessEqual(res.executed_price, 42.45)
        self.assertGreater(res.spread_savings_bps, 0.0)
        self.assertEqual(res.reason, "LIMIT_DIP_FILLED")

    def test_inav_premium_block(self) -> None:
        # SPREAD_BPS is 25.0 (> 15.0 max allowed)
        bar = pd.Series({
            "Open": 42.50,
            "High": 42.55,
            "Low": 42.35,
            "Close": 42.45,
            "VWAP": 42.40,
            "SPREAD_BPS": 25.0,
        })
        res = self.agent.evaluate_tactical_fill(self.intent, bar, self.now, bars_waited=1)
        self.assertFalse(res.is_filled)
        self.assertEqual(res.reason, "INAV_PREMIUM_BLOCK")

    def test_tactical_timeout(self) -> None:
        bar = pd.Series({
            "Open": 42.50,
            "High": 42.55,
            "Low": 42.48,
            "Close": 42.50,
            "VWAP": 42.50,
            "SPREAD_BPS": 2.0,
        })
        res = self.agent.evaluate_tactical_fill(self.intent, bar, self.now, bars_waited=4)
        self.assertFalse(res.is_filled)
        self.assertEqual(res.reason, "TACTICAL_TIMEOUT")


if __name__ == "__main__":
    unittest.main()
