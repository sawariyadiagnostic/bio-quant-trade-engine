"""Unit tests for Two-Way Chop Mean-Reversion Trigger."""
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from ravana.strategies.base import TriggerAction, TriggerSignal
from ravana.strategies.two_way_chop_reversion import TwoWayChopMeanReversionTrigger


class TestTwoWayChopMeanReversion(unittest.TestCase):
    """Test suite for TwoWayChopMeanReversionTrigger."""

    def setUp(self) -> None:
        self.trigger = TwoWayChopMeanReversionTrigger(
            bb_lower=0.15,
            bb_upper=0.85,
            ofi_threshold=0.05,
            vr_ceiling=0.75,
            micro_stop_mult=0.40,
            micro_target_mult=0.50,
        )

    def _create_mock_df(
        self,
        n_bars: int = 40,
        base_px: float = 100.0,
        vr_val: float = 0.60,
        ofi_val: float = 0.10,
        bb_pct_scenario: str = "neutral",
    ) -> pd.DataFrame:
        """Create a mock 5m or 1m dataframe with proper session timestamps and technicals."""
        base_dt = datetime(2026, 11, 16, 9, 15)
        index = [base_dt + timedelta(minutes=5 * i) for i in range(n_bars)]

        closes = []
        opens = []
        highs = []
        lows = []
        vols = []

        for i in range(n_bars):
            # Normal quiet chop
            c = base_px + (0.05 * np.sin(i))
            o = c - 0.02
            h = c + 0.05
            l = c - 0.05
            v = 10000.0

            if i == 12:  # The test trigger bar at 10:15 IST (active morning window 09:35-11:15)
                if bb_pct_scenario == "oversold_bounce":
                    # Plunge to lower band then green close
                    c = base_px - 2.0
                    o = c - 0.20
                    h = c + 0.10
                    l = o - 0.10
                elif bb_pct_scenario == "overbought_fade":
                    # Spike to upper band then red close
                    c = base_px + 2.0
                    o = c + 0.20
                    h = o + 0.10
                    l = c - 0.10

            closes.append(c)
            opens.append(o)
            highs.append(h)
            lows.append(l)
            vols.append(v)

        df = pd.DataFrame(
            {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols},
            index=pd.DatetimeIndex(index),
        )
        df["ATR_14"] = 0.50
        df["VR_20"] = vr_val
        df["OFI_5"] = ofi_val
        df["VWAP"] = base_px
        return df

    def test_long_reversion_signal(self) -> None:
        """Triggers Long BUY when price hits lower Bollinger Band in chop regime."""
        df = self._create_mock_df(bb_pct_scenario="oversold_bounce", vr_val=0.60, ofi_val=0.15)
        sigs = self.trigger.generate_signals(df, "AUTOBEES")

        buy_sigs = [s for s in sigs if s.action == TriggerAction.BUY]
        self.assertGreaterEqual(len(buy_sigs), 1)
        first = buy_sigs[0]
        self.assertEqual(first.symbol, "AUTOBEES")
        self.assertIn("target_price", first.metadata)
        self.assertIn("stop_loss", first.metadata)
        self.assertGreater(first.metadata["target_price"], first.metadata["stop_loss"])

    def test_short_reversion_signal(self) -> None:
        """Triggers Short SELL when price hits upper Bollinger Band in chop regime."""
        df = self._create_mock_df(bb_pct_scenario="overbought_fade", vr_val=0.60, ofi_val=-0.15)
        sigs = self.trigger.generate_signals(df, "PHARMABEES")

        sell_sigs = [s for s in sigs if s.action == TriggerAction.SELL]
        self.assertGreaterEqual(len(sell_sigs), 1)
        first = sell_sigs[0]
        self.assertEqual(first.symbol, "PHARMABEES")
        self.assertIn("target_price", first.metadata)
        self.assertIn("stop_loss", first.metadata)
        self.assertLess(first.metadata["target_price"], first.metadata["stop_loss"])

    def test_trend_regime_rejection(self) -> None:
        """Strictly suppresses signals when Variance Ratio confirms trending regime (VR >= 0.75)."""
        df = self._create_mock_df(bb_pct_scenario="oversold_bounce", vr_val=0.98, ofi_val=0.15)
        sigs = self.trigger.generate_signals(df, "AUTOBEES")

        # Zero BUY or SELL signals when VR >= 0.75 (trending)
        trade_sigs = [s for s in sigs if s.action in (TriggerAction.BUY, TriggerAction.SELL)]
        self.assertEqual(len(trade_sigs), 0)

    def test_evaluate_series(self) -> None:
        """Legacy series evaluator returns non-zero values on valid signals."""
        df = self._create_mock_df(bb_pct_scenario="oversold_bounce", vr_val=0.60, ofi_val=0.15)
        series = self.trigger.evaluate(df, "AUTOBEES")
        self.assertIsInstance(series, pd.Series)
        self.assertGreaterEqual(series.max(), 0.80)


if __name__ == "__main__":
    unittest.main()
