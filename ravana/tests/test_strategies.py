"""Unit tests for RAVANA Structural Strategies, Triggers, and Zero-Giveback Stop Engine.

Covers:
1. Base abstractions: TriggerAction, StopLayerState, TriggerSignal immutability/bounds.
2. 5-Layer 'Zero-Giveback' Stop & Two-Tranche Scale-Out Engine.
3. Bucket 1 (Trend & Impulse): Triggers 1 to 4.
4. Bucket 2 (Mean-Reversion & Absorption): Triggers 5 to 8.
5. Bucket 3 (Macro & Overnight): Triggers 9 and 10.
"""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError
import numpy as np
import pandas as pd

from ravana.data.ingestion.session_gatekeeper import NSESessionGatekeeper
from ravana.strategies.base import (
    StopLayerState,
    StructuralTrigger,
    TriggerAction,
    TriggerSignal,
    ZeroGivebackStopEngine,
)
from ravana.strategies.triggers import (
    AfternoonBondEuropeTWAPTrigger,
    ClosingVWAPSnapbackTrigger,
    CrossSectorLeadLagTrigger,
    DIISIPVWAPAbsorptionTrigger,
    ICTUnicornStopSweepTrigger,
    INAVSpreadMeanReversionTrigger,
    OpeningRangeBreakoutTrigger,
    OvernightContinuationTrigger,
    SectorPEADInstitutionalDriftTrigger,
    ThursdayExpiryGammaTrigger,
    is_weekly_expiry,
)


class TestBaseAbstractions(unittest.TestCase):
    """Test suite for TriggerAction, StopLayerState, and TriggerSignal."""

    def test_trigger_action_enum(self) -> None:
        """Verify all mandatory trigger action states."""
        self.assertEqual(TriggerAction.BUY.value, "BUY")
        self.assertEqual(TriggerAction.SELL.value, "SELL")
        self.assertEqual(TriggerAction.HOLD.value, "HOLD")
        self.assertEqual(TriggerAction.EXIT_ALL.value, "EXIT_ALL")
        self.assertEqual(TriggerAction.EXIT_TRANCHE_1.value, "EXIT_TRANCHE_1")
        self.assertEqual(TriggerAction.EXIT_TIME_STOP.value, "EXIT_TIME_STOP")

    def test_trigger_signal_bounds_and_immutability(self) -> None:
        """Verify conviction bounding [0.0, 1.0] and frozen immutability."""
        ts = pd.Timestamp("2026-01-08 09:35:00", tz="Asia/Kolkata")

        # Upper bound clip
        sig_high = TriggerSignal(ts, "BANKBEES", TriggerAction.BUY, 1.45, "TestTrigger")
        self.assertEqual(sig_high.conviction, 1.0)

        # Lower bound clip
        sig_low = TriggerSignal(ts, "BANKBEES", TriggerAction.BUY, -0.50, "TestTrigger")
        self.assertEqual(sig_low.conviction, 0.0)

        # In-range preserves exact value
        sig_valid = TriggerSignal(ts, "BANKBEES", TriggerAction.BUY, 0.72, "TestTrigger")
        self.assertAlmostEqual(sig_valid.conviction, 0.72)

        # Immutability
        with self.assertRaises(FrozenInstanceError):
            sig_valid.conviction = 0.90  # type: ignore

    def test_stop_layer_state_dataclass(self) -> None:
        """Verify StopLayerState initialization and mutable tracking."""
        state = StopLayerState(
            initial_stop=498.0,
            breakeven_stop=500.10,
            is_breakeven_triggered=False,
            tranche_1_sold=False,
            chandelier_stop=0.0,
            active_stop=498.0,
            peak_price=500.0,
            bars_held=0,
        )
        self.assertEqual(state.initial_stop, 498.0)
        self.assertFalse(state.is_breakeven_triggered)
        self.assertFalse(state.tranche_1_sold)


class TestZeroGivebackStopEngine(unittest.TestCase):
    """Test suite validating all 5 layers of the Zero-Giveback Stop Engine."""

    def setUp(self) -> None:
        self.engine = ZeroGivebackStopEngine(
            max_invalidation_bps=35.0,
            breakeven_friction_bps=2.0,
            breakeven_atr_threshold=0.45,
            tranche1_atr_threshold=0.75,
            runner_snap_atr=0.25,
            chandelier_activation_atr=1.0,
            chandelier_distance_atr=0.35,
            dead_money_bars=4,
            dead_money_peak_atr=0.20,
        )
        self.entry_price = 1000.0
        self.atr = 10.0  # 1% ATR

    def test_layer_1_thesis_invalidation_stop_cap(self) -> None:
        """Layer 1: Thesis Invalidation Stop is capped at -35 bps max loss."""
        # Uncapped custom stop at -50 bps (995.0) -> capped to -35 bps (996.50)
        state = self.engine.initialize_position(self.entry_price, self.atr, custom_stop=995.0)
        floor = 1000.0 * (1.0 - 0.0035)  # 996.50
        self.assertAlmostEqual(state.initial_stop, floor)
        self.assertAlmostEqual(state.active_stop, floor)

        # Custom stop tighter than cap (e.g. 998.0 = -20 bps) -> preserved
        state_tight = self.engine.initialize_position(self.entry_price, self.atr, custom_stop=998.0)
        self.assertAlmostEqual(state_tight.initial_stop, 998.0)

        # Breach of active stop triggers EXIT_ALL
        st, act = self.engine.update(
            state_tight, self.entry_price, bar_high=998.5, bar_low=997.5, bar_close=997.8, atr=self.atr
        )
        self.assertEqual(act, TriggerAction.EXIT_ALL)

    def test_layer_2_breakeven_ratchet_at_045_atr(self) -> None:
        """Layer 2: +0.45 ATR price advance locks entry + 2.0 bps friction."""
        state = self.engine.initialize_position(self.entry_price, self.atr)
        # Advance to entry + 0.45 * 10 = 1004.50
        state, act = self.engine.update(
            state, self.entry_price, bar_high=1004.60, bar_low=1001.0, bar_close=1004.0, atr=self.atr
        )
        self.assertTrue(state.is_breakeven_triggered)
        expected_be = 1000.0 * (1.0 + 2.0 / 10000.0)  # 1000.20
        self.assertAlmostEqual(state.breakeven_stop, expected_be)
        self.assertAlmostEqual(state.active_stop, expected_be)

    def test_layer_3_two_tranche_scale_out_at_075_atr(self) -> None:
        """Layer 3: +0.75 ATR sells 50% shares and snaps runner stop to +0.25 ATR."""
        state = self.engine.initialize_position(self.entry_price, self.atr)
        # Bar moves to entry + 0.75 * 10 = 1007.50
        state, act = self.engine.update(
            state, self.entry_price, bar_high=1007.80, bar_low=1002.0, bar_close=1006.0, atr=self.atr, is_mis=True
        )
        self.assertEqual(act, TriggerAction.EXIT_TRANCHE_1)
        self.assertTrue(state.tranche_1_sold)
        expected_runner_stop = 1000.0 + 0.25 * 10.0  # 1002.50
        self.assertAlmostEqual(state.active_stop, expected_runner_stop)

        # CNC does not trigger tranche 1 scale out
        state_cnc = self.engine.initialize_position(self.entry_price, self.atr)
        state_cnc, act_cnc = self.engine.update(
            state_cnc, self.entry_price, bar_high=1008.0, bar_low=1002.0, bar_close=1007.0, atr=self.atr, is_mis=False
        )
        self.assertFalse(state_cnc.tranche_1_sold)
        self.assertNotEqual(act_cnc, TriggerAction.EXIT_TRANCHE_1)

    def test_layer_4_chandelier_ratchet_at_10_atr(self) -> None:
        """Layer 4: Chandelier ratchet locks peak - 0.35 ATR once peak >= +1.0 ATR."""
        state = self.engine.initialize_position(self.entry_price, self.atr)
        # Price spikes to 1015.0 (+1.5 ATR)
        state, act = self.engine.update(
            state, self.entry_price, bar_high=1015.0, bar_low=1005.0, bar_close=1014.0, atr=self.atr
        )
        expected_chandelier = 1015.0 - 0.35 * 10.0  # 1011.50
        self.assertAlmostEqual(state.chandelier_stop, expected_chandelier)
        self.assertAlmostEqual(state.active_stop, expected_chandelier)

    def test_layer_5_dead_money_time_stop(self) -> None:
        """Layer 5: 4-bar time stop exits flat if peak < +0.20 ATR."""
        state = self.engine.initialize_position(self.entry_price, self.atr)
        # Advance 3 bars with minimal movement (peak 1001.0 < 1000 + 2.0)
        for _ in range(3):
            state, act = self.engine.update(
                state, self.entry_price, bar_high=1001.0, bar_low=999.5, bar_close=1000.5, atr=self.atr
            )
            self.assertEqual(act, TriggerAction.HOLD)

        # 4th bar triggers EXIT_TIME_STOP
        state, act_4 = self.engine.update(
            state, self.entry_price, bar_high=1001.2, bar_low=999.8, bar_close=1000.2, atr=self.atr
        )
        self.assertEqual(act_4, TriggerAction.EXIT_TIME_STOP)


class TestStructuralTriggers(unittest.TestCase):
    """Test suite covering the 10 Hardened Structural Triggers across 3 Buckets."""

    def _build_session_df(
        self, date_str: str = "2026-01-08", base_price: float = 500.0, n_bars: int = 75
    ) -> pd.DataFrame:
        """Create standard 5-minute intraday NSE session bars (09:15 to 15:25)."""
        idx = pd.date_range(f"{date_str} 09:15", periods=n_bars, freq="5min", tz="Asia/Kolkata")
        close = pd.Series(base_price + np.zeros(n_bars), index=idx)
        df = pd.DataFrame(
            {
                "Open": close,
                "High": close + 0.50,
                "Low": close - 0.50,
                "Close": close,
                "Volume": pd.Series(10000.0, index=idx),
                "VWAP": close,
                "Theoretical_Close": close,
                "Spread_INR": pd.Series(0.05, index=idx),
            },
            index=idx,
        )
        return df

    def test_trigger_1_opening_range_breakout(self) -> None:
        """Trigger 1: 09:35-10:30 window, 15m range high breakout with OFI and VR."""
        df = self._build_session_df()
        # Set 15m range (09:15 to 09:25 bars) High to 501.0
        df.iloc[0:3, df.columns.get_loc("High")] = 501.0
        # 09:30 HFT trap bar: High 500.8, Close 500.5
        df.iloc[3, df.columns.get_loc("High")] = 500.8
        df.iloc[3, df.columns.get_loc("Close")] = 500.5

        # 09:35 bar (index 4): breakout Close 502.5 > 501.0, OFI = 0.40, VR = 1.20
        df.iloc[4, df.columns.get_loc("Close")] = 502.5
        df.iloc[4, df.columns.get_loc("High")] = 503.0
        df["OFI_5"] = pd.Series(0.10, index=df.index)
        df.iloc[4, df.columns.get_loc("OFI_5")] = 0.40
        df["VARIANCE_RATIO_20"] = pd.Series(1.20, index=df.index)

        trig = OpeningRangeBreakoutTrigger()
        signals = trig.generate_signals(df, "BANKBEES")

        self.assertGreaterEqual(len(signals), 1)
        sig = signals[0]
        self.assertEqual(sig.action, TriggerAction.BUY)
        self.assertEqual(sig.metadata["product_type"], "MIS")
        self.assertEqual(sig.timestamp.strftime("%H:%M"), "09:35")
        self.assertGreater(sig.conviction, 0.50)

    def test_trigger_2_thursday_expiry_gamma_and_holiday_shift(self) -> None:
        """Trigger 2: Thursday expiry unpinning & automatic holiday shift to Wednesday."""
        # 1. Normal Thursday expiry (2026-01-08)
        ts_thurs = pd.Timestamp("2026-01-08 14:30:00", tz="Asia/Kolkata")
        self.assertTrue(is_weekly_expiry(ts_thurs))

        # 2. Registered NSE Holiday: Independence Day 2024-08-15 (Thursday)
        # Register holiday explicitly if not already
        NSESessionGatekeeper.register_holidays(["2024-08-15"])
        ts_wed = pd.Timestamp("2024-08-14 14:30:00", tz="Asia/Kolkata")
        ts_thurs_hol = pd.Timestamp("2024-08-15 14:30:00", tz="Asia/Kolkata")

        self.assertTrue(is_weekly_expiry(ts_wed))
        self.assertFalse(is_weekly_expiry(ts_thurs_hol))

        # 3. Test Trigger execution at 14:20 on Thursday
        df = self._build_session_df("2026-01-08")
        # Bar at 14:20 is index 61
        idx_1420 = df.index.get_loc(pd.Timestamp("2026-01-08 14:20:00", tz="Asia/Kolkata"))
        df.iloc[idx_1420, df.columns.get_loc("Close")] = 505.0
        df.iloc[idx_1420, df.columns.get_loc("High")] = 506.0
        df.iloc[idx_1420, df.columns.get_loc("Volume")] = 50000.0  # 5x volume surge

        trig = ThursdayExpiryGammaTrigger()
        signals = trig.generate_signals(df, "BANKBEES")
        self.assertGreaterEqual(len(signals), 1)
        self.assertEqual(signals[0].metadata["product_type"], "MIS")

    def test_trigger_3_cross_sector_lead_lag(self) -> None:
        """Trigger 3: Leader t-1 return > 0.35% with OFI > 0.30 while follower lags < 0.12%."""
        df = self._build_session_df("2026-01-08")
        # Bar index 10 (10:05) leader jumps +0.50% at t-1 (index 9)
        leader_close = pd.Series(df["Close"].copy(), index=df.index)
        leader_close.iloc[9] = leader_close.iloc[8] * 1.0050  # +50 bps
        df["Leader_Close"] = leader_close
        df["Leader_OFI_5"] = pd.Series(0.40, index=df.index)
        # Follower lagged return at index 9: 0.0%
        df.iloc[9, df.columns.get_loc("Close")] = df.iloc[8, df.columns.get_loc("Close")]

        trig = CrossSectorLeadLagTrigger()
        signals = trig.generate_signals(df, "PSUBNKBEES")

        self.assertGreaterEqual(len(signals), 1)
        sig = signals[0]
        self.assertEqual(sig.action, TriggerAction.BUY)
        self.assertEqual(sig.metadata["order_type"], "LIMIT")
        self.assertEqual(sig.metadata["product_type"], "MIS")

    def test_trigger_4_sector_pead_institutional_drift(self) -> None:
        """Trigger 4: Volume z-score >= 1.8, log_ret_20 >= 0.006, VR >= 1.15, VWAP pullback."""
        df = self._build_session_df("2026-01-08")
        df["VARIANCE_RATIO_20"] = pd.Series(1.25, index=df.index)
        # Create 20-bar trend
        for i in range(25):
            df.iloc[i, df.columns.get_loc("Close")] = 500.0 + i * 0.40
            df.iloc[i, df.columns.get_loc("High")] = 500.0 + i * 0.40 + 0.50
            df.iloc[i, df.columns.get_loc("Low")] = 500.0 + i * 0.40 - 0.30
        df.iloc[22, df.columns.get_loc("Volume")] = 80000.0  # Lagged volume spike at t-1

        # At index 23 (11:10), price pulls back to rising VWAP
        df["VWAP"] = df["Close"] * 0.999
        df.iloc[23, df.columns.get_loc("Close")] = df["VWAP"].iloc[23] * 1.001
        df.iloc[23, df.columns.get_loc("Low")] = df["VWAP"].iloc[23]

        trig = SectorPEADInstitutionalDriftTrigger()
        signals = trig.generate_signals(df, "BANKBEES")
        self.assertGreaterEqual(len(signals), 1)
        self.assertEqual(signals[0].metadata["product_type"], "MIS")

    def test_trigger_5_inav_spread_mean_reversion_and_invalidation(self) -> None:
        """Trigger 5: ETF discount buy & thesis invalidation on underlying 3-bar low break."""
        df = self._build_session_df("2026-01-08")
        df["SPREAD_ZSCORE_20"] = pd.Series(0.0, index=df.index)
        df["SPREAD_BPS"] = pd.Series(0.0, index=df.index)
        df["VARIANCE_RATIO_20"] = pd.Series(0.95, index=df.index)  # Mean-reverting regime

        # Entry signal at 10:30 (index 15)
        df.iloc[15, df.columns.get_loc("SPREAD_ZSCORE_20")] = -2.1
        df.iloc[15, df.columns.get_loc("SPREAD_BPS")] = -25.0

        # Invalidation at 11:30 (index 27): underlying breaks 3-bar low
        df["Theoretical_Close"] = pd.Series(500.0, index=df.index)
        df.iloc[27, df.columns.get_loc("Theoretical_Close")] = 490.0

        trig = INAVSpreadMeanReversionTrigger()
        signals = trig.generate_signals(df, "BANKBEES")

        buys = [s for s in signals if s.action == TriggerAction.BUY]
        exits = [s for s in signals if s.action == TriggerAction.EXIT_ALL]

        self.assertGreaterEqual(len(buys), 1)
        self.assertGreaterEqual(len(exits), 1)
        self.assertEqual(buys[0].metadata["product_type"], "MIS")
        self.assertIn("target_price", buys[0].metadata)

    def test_trigger_6_closing_vwap_snapback_and_terminal_exit(self) -> None:
        """Trigger 6: 15:00-15:12 retail forced liquidation buy, scheduled 15:24 exit."""
        # 1-minute bars from 15:00 to 15:25
        idx = pd.date_range("2026-01-08 15:00", "2026-01-08 15:25", freq="1min", tz="Asia/Kolkata")
        df = pd.DataFrame(
            {
                "Open": pd.Series(505.0, index=idx),
                "High": pd.Series(506.0, index=idx),
                "Low": pd.Series(504.0, index=idx),
                "Close": pd.Series(505.0, index=idx),
                "Volume": pd.Series(10000.0, index=idx),
                "SPREAD_ZSCORE_20": pd.Series(0.0, index=idx),
                "LOG_RET_20": pd.Series(0.005, index=idx),
            },
            index=idx,
        )
        df.iloc[0, df.columns.get_loc("SPREAD_ZSCORE_20")] = -1.8  # 15:00 buy

        trig = ClosingVWAPSnapbackTrigger()
        signals = trig.generate_signals(df, "BANKBEES")

        buys = [s for s in signals if s.action == TriggerAction.BUY]
        exits = [s for s in signals if s.action == TriggerAction.EXIT_ALL]

        self.assertGreaterEqual(len(buys), 1)
        self.assertGreaterEqual(len(exits), 1)
        self.assertEqual(buys[0].timestamp.strftime("%H:%M"), "15:00")
        self.assertEqual(exits[0].timestamp.strftime("%H:%M"), "15:24")

    def test_trigger_7_dii_sip_vwap_absorption(self) -> None:
        """Trigger 7: Turn of month SIP window, lower BB <= 0.22, OFI > 0.20."""
        # 2026-01-05 is day 5 (within days 1..7 SIP window)
        df = self._build_session_df("2026-01-05")
        df["BOLLINGER_BAND_PCT_20"] = pd.Series(0.50, index=df.index)
        df["OFI_5"] = pd.Series(0.0, index=df.index)

        # Bar at 11:00 (index 21)
        df.iloc[21, df.columns.get_loc("BOLLINGER_BAND_PCT_20")] = 0.18
        df.iloc[21, df.columns.get_loc("OFI_5")] = 0.35

        trig = DIISIPVWAPAbsorptionTrigger()
        signals = trig.generate_signals(df, "NIFTYBEES")

        self.assertGreaterEqual(len(signals), 1)
        self.assertEqual(signals[0].action, TriggerAction.BUY)
        self.assertEqual(signals[0].metadata["product_type"], "MIS")

    def test_trigger_8_ict_unicorn_stop_sweep(self) -> None:
        """Trigger 8: Stop sweep reclaim, FVG / discount, OFI > 0.30, 3.3:1 R:R target."""
        df = self._build_session_df("2026-01-08")
        # Establish prior 10-bar low at 498.0
        df.iloc[0:15, df.columns.get_loc("Low")] = 498.0
        # At index 15 (10:30): Low dips to 497.0 (pierces), closes at 499.0 (reclaims)
        df.iloc[15, df.columns.get_loc("Low")] = 497.0
        df.iloc[15, df.columns.get_loc("Close")] = 499.0
        df["FVG_BULLISH_DRIFT"] = pd.Series(0.0, index=df.index)
        df.iloc[15, df.columns.get_loc("FVG_BULLISH_DRIFT")] = 0.002
        df["OFI_5"] = pd.Series(0.10, index=df.index)
        df.iloc[15, df.columns.get_loc("OFI_5")] = 0.45

        trig = ICTUnicornStopSweepTrigger()
        signals = trig.generate_signals(df, "BANKBEES")

        self.assertGreaterEqual(len(signals), 1)
        sig = signals[0]
        self.assertEqual(sig.action, TriggerAction.BUY)
        self.assertAlmostEqual(sig.metadata["stop_price"], 496.95)
        # Risk = 499.0 - 496.95 = 2.05 -> Target = 499.0 + 3.3 * 2.05 = 505.765
        expected_target = 499.0 + 3.3 * (499.0 - 496.95)
        self.assertAlmostEqual(sig.metadata["target_price"], expected_target)

    def test_trigger_9_afternoon_bond_europe_twap(self) -> None:
        """Trigger 9: 13:55-14:40 window, exit at 14:45 strictly before retail dump."""
        df = self._build_session_df("2026-01-08")
        df["MACRO_YIELD_RET_1"] = pd.Series(0.0, index=df.index)
        # Bar at 14:00 (index 57): bond yield eases -0.005
        df.iloc[57, df.columns.get_loc("MACRO_YIELD_RET_1")] = -0.005

        trig = AfternoonBondEuropeTWAPTrigger()
        signals = trig.generate_signals(df, "BANKBEES")

        buys = [s for s in signals if s.action == TriggerAction.BUY]
        exits = [s for s in signals if s.action == TriggerAction.EXIT_ALL]

        self.assertGreaterEqual(len(buys), 1)
        self.assertGreaterEqual(len(exits), 1)
        self.assertEqual(buys[0].metadata["exit_time"], "14:45")
        self.assertEqual(exits[0].timestamp.strftime("%H:%M"), "14:45")

    def test_trigger_10_overnight_continuation_and_gate(self) -> None:
        """Trigger 10: Whitelisted overnight entry at 15:20-15:25, continuation gate at 09:15."""
        # Multi-day DataFrame spanning Day 1 afternoon to Day 2 morning
        idx_day1 = pd.date_range("2026-01-08 15:00", "2026-01-08 15:25", freq="5min", tz="Asia/Kolkata")
        idx_day2 = pd.date_range("2026-01-09 09:15", "2026-01-09 10:00", freq="5min", tz="Asia/Kolkata")
        all_idx = idx_day1.append(idx_day2)

        df = pd.DataFrame(
            {
                "Open": pd.Series(500.0, index=all_idx),
                "High": pd.Series(501.0, index=all_idx),
                "Low": pd.Series(499.0, index=all_idx),
                "Close": pd.Series(500.0, index=all_idx),
                "Volume": pd.Series(20000.0, index=all_idx),
                "SPREAD_BPS": pd.Series(10.0, index=all_idx),
                "VWAP": pd.Series(500.0, index=all_idx),
            },
            index=all_idx,
        )

        # Day 2 morning gap up: Open 505.0, Close 506.0 -> strong continuation
        loc_0915 = all_idx.get_loc(pd.Timestamp("2026-01-09 09:15:00", tz="Asia/Kolkata"))
        df.iloc[loc_0915, df.columns.get_loc("Open")] = 505.0
        df.iloc[loc_0915, df.columns.get_loc("Close")] = 506.0

        trig = OvernightContinuationTrigger()
        signals = trig.generate_signals(df, "BANKBEES")

        buys = [s for s in signals if s.action == TriggerAction.BUY]
        holds = [s for s in signals if s.action == TriggerAction.HOLD]

        self.assertGreaterEqual(len(buys), 1)
        self.assertEqual(buys[0].metadata["product_type"], "CNC")
        self.assertGreaterEqual(len(holds), 1)
        self.assertEqual(holds[0].action, TriggerAction.HOLD)

        # Non-whitelisted symbol rejection
        rej_signals = trig.generate_signals(df, "RANDOM_STOCK")
        self.assertEqual(len(rej_signals), 0)


if __name__ == "__main__":
    unittest.main()
