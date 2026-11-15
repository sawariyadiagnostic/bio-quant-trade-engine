"""Master Integration Test Suite for RAVANA Institutional Structural Strategies.

Validates the complete 20-point test matrix covering the Trigger-First sovereign architecture:
1.  OpeningRangeBreakoutTrigger 09:35+ shield (avoids 09:31 bull trap)
2.  INAVSpreadMeanReversionTrigger dynamic ATR target & thesis stop on 3-bar low
3.  AfternoonBondEuropeTWAPTrigger 14:45 terminal exit before 15:00 retail dump
4.  ThursdayExpiryGammaTrigger holiday shift to Wednesday
5.  OvernightContinuationTrigger Tier-1 ETF whitelist
6.  CrossSectorLeadLagTrigger closed t-1 bar limit order placement
7.  ClosingVWAPSnapbackTrigger 15:00-15:12 entry and 15:24 closing exit
8.  DIISIPVWAPAbsorptionTrigger days 28-7 VWAP/Bollinger pullback absorption
9.  ICTUnicornStopSweepTrigger 10-bar sweep reclaim with FVG and 3.3:1 target
10. SectorPEADInstitutionalDriftTrigger volume drift entry with VR > 1.15
11. ZeroGivebackStopEngine 5-layer stop & two-tranche scale-out
12. BoundedASTModulator no saturation under extreme inputs [0.5x, 1.5x]
13. BoundedASTModulator 60-bar variance sentinel stagnation fallback
14. Zero Ghost Trades invariant (conviction 0.0 remains 0.0)
15. StrategyBanditAllocator 15-trade burn-in & [0.70x, 1.30x] clamping
16. StrategyRiskBridge 2.0x MIS risk-at-stop sizing capped at 0.25% equity
17. Minimum economic lot filter (rejection under ₹50,000 notional)
18. SQLiteStateLedger 11:32 AM reboot recovery reconciliation
19. ShoonyaBrokerGateway & Router continuation gate DP fee savings (₹15.93)
20. FastBacktester 100,000-bar sub-50ms benchmark with IEEE-754 NaN safety
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
import unittest
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from ravana.backtest.config import BacktestConfig
from ravana.backtest.fast_engine import FastBacktester
from ravana.domain.formula_ast import ASTNode
from ravana.execution.router import BrokerOrderRouter
from ravana.execution.shoonya_adapter import (
    ShoonyaBrokerGateway,
    ShoonyaGatewayConfig,
)
from ravana.risk.circuit_breaker import CircuitBreakerKillSwitch
from ravana.risk.config import DrawdownState
from ravana.risk.engine import AllocationResult
from ravana.risk.sizing import VolatilitySizedAllocation
from ravana.strategies.base import (
    TriggerAction,
    TriggerSignal,
    ZeroGivebackStopEngine,
)
from ravana.strategies.composite import StrategyBanditAllocator
from ravana.strategies.modulator import BoundedASTModulator
from ravana.strategies.risk_adapter import StrategyRiskBridge
from ravana.strategies.state_ledger import SQLiteStateLedger
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


class TestStructuralStrategiesMaster(unittest.TestCase):
    """Master Integration Test Suite executing the 20-point institutional matrix."""

    def setUp(self) -> None:
        """Initialize reusable test fixtures and isolated temp directories."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "master_test_ledger.db")

    def tearDown(self) -> None:
        """Clean up filesystem artifacts."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @staticmethod
    def _build_intraday_df(
        date_str: str,
        start_time: str,
        periods: int,
        base_price: float = 500.0,
        freq: str = "5min",
    ) -> pd.DataFrame:
        """Helper to construct realistic 5-minute OHLCV market data DataFrame."""
        ts_start = f"{date_str} {start_time}"
        idx = pd.date_range(ts_start, periods=periods, freq=freq, tz="Asia/Kolkata")
        return pd.DataFrame(
            {
                "Open": np.full(periods, base_price),
                "High": np.full(periods, base_price + 1.0),
                "Low": np.full(periods, base_price - 1.0),
                "Close": np.full(periods, base_price),
                "Volume": np.full(periods, 10000.0),
                "ATR_14": np.full(periods, 2.5),
                "VARIANCE_RATIO_20": np.full(periods, 1.20),
                "OFI_5": np.full(periods, 0.40),
            },
            index=idx,
        )

    # -------------------------------------------------------------------------
    # 1. Opening Range Breakout: 09:35+ Shield
    # -------------------------------------------------------------------------
    def test_opening_range_breakout_0935_shield(self) -> None:
        """Verify ORB entries start at 09:35+ with OFI > 0.25, avoiding 09:31 bull traps."""
        trigger = OpeningRangeBreakoutTrigger()
        # Create session bars: 09:15, 09:20, 09:25 (Opening Range), 09:30 (trap), 09:35 (valid)
        df = self._build_intraday_df("2026-01-08", "09:15", 6, base_price=500.0)
        # Establish Opening Range (09:15 - 09:25): High is 505.0
        df.loc[df.index[0], ["High", "Close"]] = [502.0, 501.0]
        df.loc[df.index[1], ["High", "Close"]] = [505.0, 504.0]
        df.loc[df.index[2], ["High", "Close"]] = [503.0, 502.0]

        # Bar 3 at 09:30: Retail bull trap breaks 505.0 with High=508.0, Close=507.0, OFI=0.50
        df.loc[df.index[3], ["Open", "High", "Low", "Close", "OFI_5"]] = [502.0, 508.0, 502.0, 507.0, 0.50]
        # Bar 4 at 09:35: True institutional breakout with Close=508.0, OFI=0.45
        df.loc[df.index[4], ["Open", "High", "Low", "Close", "OFI_5"]] = [507.0, 509.0, 506.0, 508.0, 0.45]
        # Bar 5 at 09:40: Breakout without OFI support (OFI=0.10 <= 0.25)
        df.loc[df.index[5], ["Open", "High", "Low", "Close", "OFI_5"]] = [508.0, 510.0, 507.0, 509.0, 0.10]

        conviction = trigger.evaluate(df, "BANKBEES")
        signals = trigger.generate_signals(df, "BANKBEES")

        # 09:30 bar must be strictly 0.0 (shielding against 09:31 trap)
        self.assertEqual(conviction.iloc[3], 0.0)
        self.assertNotIn(df.index[3], [s.timestamp for s in signals])

        # 09:35 bar must trigger BUY with conviction >= 0.70
        self.assertGreaterEqual(conviction.iloc[4], 0.70)
        sig_0935 = [s for s in signals if s.timestamp == df.index[4]][0]
        self.assertEqual(sig_0935.action, TriggerAction.BUY)
        self.assertEqual(sig_0935.metadata["product_type"], "MIS")
        self.assertEqual(sig_0935.metadata["regime"], "TREND_IMPULSE")

        # 09:40 bar with weak OFI must be rejected
        self.assertEqual(conviction.iloc[5], 0.0)

    # -------------------------------------------------------------------------
    # 2. iNAV Spread Mean Reversion: Dynamic ATR & Thesis Stop
    # -------------------------------------------------------------------------
    def test_inav_spread_mean_reversion_dynamic_atr(self) -> None:
        """Verify dynamic ATR profit target and thesis stop on 3-bar index low."""
        trigger = INAVSpreadMeanReversionTrigger()
        df = self._build_intraday_df("2026-01-08", "10:00", 6, base_price=100.0)
        df["VARIANCE_RATIO_20"] = 0.90  # Mean-reverting regime
        df["SPREAD_BPS"] = np.full(6, -5.0)
        df["SPREAD_ZSCORE_20"] = np.full(6, 0.0)
        df["Theoretical_Close"] = [100.0, 100.2, 100.1, 100.0, 100.3, 98.0]
        df["ATR_14"] = np.full(6, 2.0)

        # Bar 1 at 10:05: Abnormal discount trigger
        df.loc[df.index[1], "SPREAD_BPS"] = -25.0
        df.loc[df.index[1], "SPREAD_ZSCORE_20"] = -2.2
        df.loc[df.index[1], "Close"] = 100.0

        signals = trigger.generate_signals(df, "NIFTYBEES")

        # Verify BUY signal dynamic ATR target at 10:05 (Close + 1.5 * ATR = 100 + 3.0 = 103.0)
        buy_sigs = [s for s in signals if s.action == TriggerAction.BUY]
        self.assertGreaterEqual(len(buy_sigs), 1)
        self.assertEqual(buy_sigs[0].metadata["target_price"], 103.0)
        self.assertEqual(buy_sigs[0].metadata["regime"], "INAV_MEAN_REVERSION")

        # Bar 5: Underlying breaks 3-bar low (98.0 < min(100.1, 100.0, 100.3)) -> Thesis Invalidation Exit
        exit_sigs = [s for s in signals if s.action == TriggerAction.EXIT_ALL]
        self.assertGreaterEqual(len(exit_sigs), 1)
        self.assertEqual(exit_sigs[0].metadata["reason"], "underlying_broke_3_bar_low")

    # -------------------------------------------------------------------------
    # 3. Afternoon Bond / Europe TWAP: 14:45 Terminal Exit
    # -------------------------------------------------------------------------
    def test_afternoon_bond_twap_1445_exit(self) -> None:
        """Verify terminal exit at 14:45 before 15:00 retail MIS dump."""
        trigger = AfternoonBondEuropeTWAPTrigger()
        # Bars covering 14:00 to 15:00
        df = self._build_intraday_df("2026-01-08", "14:00", 13, base_price=500.0)
        df["MACRO_YIELD_RET_1"] = 0.0
        # 14:05: Bond easing triggers momentum buy
        df.loc[df.index[1], "MACRO_YIELD_RET_1"] = -0.02

        signals = trigger.generate_signals(df, "BANKBEES")

        # Check BUY entry
        buys = [s for s in signals if s.action == TriggerAction.BUY]
        self.assertGreaterEqual(len(buys), 1)
        self.assertEqual(buys[0].metadata["exit_time"], "14:45")

        # Check terminal exit at 14:45 IST (Bar 9: 14:00 + 9*5m = 14:45)
        exits = [s for s in signals if s.action == TriggerAction.EXIT_ALL]
        self.assertEqual(len(exits), 1)
        self.assertEqual(exits[0].timestamp.strftime("%H:%M"), "14:45")
        self.assertEqual(exits[0].metadata["reason"], "exit_before_1500_retail_mis_dump")

    # -------------------------------------------------------------------------
    # 4. Thursday Expiry Gamma: Exchange Holiday Shift to Wednesday
    # -------------------------------------------------------------------------
    def test_thursday_expiry_gamma_holiday_shift(self) -> None:
        """Verify gamma window shifts to Wednesday if Thursday is an exchange holiday."""
        # Standard Week: Thursday 2026-01-08 is expiry, Wednesday 2026-01-07 is not
        ts_wed = pd.Timestamp("2026-01-07 14:30", tz="Asia/Kolkata")
        ts_thu = pd.Timestamp("2026-01-08 14:30", tz="Asia/Kolkata")
        self.assertTrue(is_weekly_expiry(ts_thu, holidays=set()))
        self.assertFalse(is_weekly_expiry(ts_wed, holidays=set()))

        # Holiday Week: Thursday 2026-01-08 is closed -> Expiry shifts to Wednesday 2026-01-07
        holidays_set = {"2026-01-08"}
        self.assertTrue(is_weekly_expiry(ts_wed, holidays=holidays_set))
        self.assertFalse(is_weekly_expiry(ts_thu, holidays=holidays_set))

        # Test trigger execution on shifted Wednesday
        trigger = ThursdayExpiryGammaTrigger()
        df_wed = self._build_intraday_df("2026-01-07", "14:00", 6, base_price=250.0)
        # Bar at 14:30 (index 6 in 5m periods from 14:00 -> index 6 is 14:30)
        df_wed = self._build_intraday_df("2026-01-07", "14:00", 8, base_price=250.0)
        # Establish 6-bar high = 251.0, mean volume = 10000
        df_wed["Volume"] = 10000.0
        # 14:30 bar surges above high with 2.0x volume surge
        df_wed.loc[df_wed.index[6], ["High", "Close", "Volume"]] = [254.0, 253.5, 20000.0]

        # Patch KNOWN_NSE_HOLIDAYS check via monkeypatching helper or direct evaluation
        from unittest.mock import patch
        with patch("ravana.strategies.triggers.is_weekly_expiry", lambda ts: ts.day == 7):
            signals = trigger.generate_signals(df_wed, "NIFTYBEES")
            self.assertEqual(len(signals), 1)
            self.assertEqual(signals[0].action, TriggerAction.BUY)
            self.assertEqual(signals[0].metadata["regime"], "EXPIRY_GAMMA")

    # -------------------------------------------------------------------------
    # 5. Overnight Continuation: All-Weather Tier-1 Whitelist
    # -------------------------------------------------------------------------
    def test_overnight_continuation_all_weather_whitelist(self) -> None:
        """Verify only Tier-1 ETFs (BANKBEES, NIFTYBEES, ITBEES, GOLDBEES) enter overnight CNC."""
        trigger = OvernightContinuationTrigger()
        df = self._build_intraday_df("2026-01-08", "15:15", 3, base_price=500.0)
        df["SPREAD_BPS"] = 15.0  # Favorable tight spread
        df["VWAP"] = 499.50      # Close >= VWAP * 0.999

        # Whitelisted Tier-1 ETF
        for sym in ["BANKBEES", "NIFTYBEES", "ITBEES", "GOLDBEES"]:
            sigs = trigger.generate_signals(df, sym)
            self.assertGreaterEqual(len(sigs), 1)
            buy_sig = [s for s in sigs if s.action == TriggerAction.BUY][0]
            self.assertEqual(buy_sig.metadata["product_type"], "CNC")
            self.assertEqual(buy_sig.metadata["regime"], "ALL_WEATHER_OVERNIGHT")

        # Non-whitelisted ETFs / Equities strictly rejected
        for non_sym in ["LIQUIDBEES", "MIDCAPETF", "RELIANCE", "INFY"]:
            non_sigs = trigger.generate_signals(df, non_sym)
            self.assertEqual(len(non_sigs), 0)
            conv = trigger.evaluate(df, non_sym)
            self.assertEqual(float(conv.max()), 0.0)

    # -------------------------------------------------------------------------
    # 6. Cross Sector Lead-Lag: Closed Bar Limit Order
    # -------------------------------------------------------------------------
    def test_cross_sector_lead_lag_closed_bar_limit(self) -> None:
        """Verify leader return at t-1 Close generates half-spread limit order on follower."""
        trigger = CrossSectorLeadLagTrigger()
        # Follower = PSUBNKBEES, Leader = BANKBEES
        df = self._build_intraday_df("2026-01-08", "09:20", 4, base_price=100.0)
        df["Leader_Close"] = [500.0, 502.50, 503.0, 503.5]  # Bar 1 -> +0.50% jump (>0.35%)
        df["Leader_OFI_5"] = [0.10, 0.45, 0.40, 0.35]       # Leader OFI > 0.30
        df["Close"] = [100.0, 100.05, 100.10, 100.15]      # Follower lagged (<0.12%)
        df["Open"] = [100.0, 100.05, 100.10, 100.15]
        df["Spread_INR"] = 0.10                              # Half spread = 0.05

        signals = trigger.generate_signals(df, "PSUBNKBEES")
        # Bar 2 (09:30) evaluates causal t-1 (09:25) breakout
        sig_t2 = [s for s in signals if s.timestamp == df.index[2]]
        self.assertEqual(len(sig_t2), 1)
        self.assertEqual(sig_t2[0].action, TriggerAction.BUY)
        self.assertEqual(sig_t2[0].metadata["order_type"], "LIMIT")
        # Limit price: Open (100.10) + 0.5 * 0.10 = 100.15
        self.assertAlmostEqual(sig_t2[0].metadata["limit_price"], 100.15, places=2)

    # -------------------------------------------------------------------------
    # 7. Closing VWAP Snapback: 15:02 Entry and 15:24 Exit
    # -------------------------------------------------------------------------
    def test_closing_vwap_snapback_1502(self) -> None:
        """Verify entry on 15:00-15:12 retail dump and exit at 15:24 closing snapback."""
        trigger = ClosingVWAPSnapbackTrigger()
        df = self._build_intraday_df("2026-01-08", "14:55", 8, base_price=500.0)
        df["LOG_RET_20"] = 0.006      # Strong trend day
        df["SPREAD_ZSCORE_20"] = 0.0  # Normal spread

        # Bar 2 at 15:05: Retail MIS liquidation dump produces discount z-score -1.8
        df.loc[df.index[2], "SPREAD_ZSCORE_20"] = -1.8

        signals = trigger.generate_signals(df, "BANKBEES")
        buys = [s for s in signals if s.action == TriggerAction.BUY]
        self.assertEqual(len(buys), 1)
        self.assertEqual(buys[0].timestamp.strftime("%H:%M"), "15:05")
        self.assertEqual(buys[0].metadata["regime"], "VWAP_SNAPBACK")

        # Bar 5 at 15:24: Terminal closing exit
        df_exit = self._build_intraday_df("2026-01-08", "15:24", 1, base_price=500.0)
        exit_sigs = trigger.generate_signals(df_exit, "BANKBEES")
        self.assertEqual(len(exit_sigs), 1)
        self.assertEqual(exit_sigs[0].action, TriggerAction.EXIT_ALL)
        self.assertEqual(exit_sigs[0].metadata["reason"], "closing_vwap_terminal_exit_1524")

    # -------------------------------------------------------------------------
    # 8. DII SIP VWAP Absorption: Days 28-7 Pullback
    # -------------------------------------------------------------------------
    def test_dii_sip_vwap_absorption(self) -> None:
        """Verify entry on days 28-7 when pulling back to lower Bollinger/VWAP with positive OFI."""
        trigger = DIISIPVWAPAbsorptionTrigger()
        # SIP day: 2026-01-29 (Day 29)
        df_sip = self._build_intraday_df("2026-01-29", "11:00", 3, base_price=250.0)
        df_sip["BOLLINGER_BAND_PCT_20"] = 0.18  # Pullback to lower Bollinger Band (<= 0.22)
        df_sip["OFI_5"] = 0.35                  # Positive absorption (> 0.20)

        sigs_sip = trigger.generate_signals(df_sip, "NIFTYBEES")
        self.assertGreaterEqual(len(sigs_sip), 1)
        self.assertEqual(sigs_sip[0].action, TriggerAction.BUY)
        self.assertEqual(sigs_sip[0].metadata["regime"], "DII_SIP_ABSORPTION")

        # Non-SIP day: 2026-01-15 (Day 15) with identical technical levels
        df_nonsip = self._build_intraday_df("2026-01-15", "11:00", 3, base_price=250.0)
        df_nonsip["BOLLINGER_BAND_PCT_20"] = 0.18
        df_nonsip["OFI_5"] = 0.35

        sigs_nonsip = trigger.generate_signals(df_nonsip, "NIFTYBEES")
        self.assertEqual(len(sigs_nonsip), 0)

    # -------------------------------------------------------------------------
    # 9. ICT Unicorn Stop Sweep Reclaim
    # -------------------------------------------------------------------------
    def test_ict_unicorn_stop_sweep_reclaim(self) -> None:
        """Verify entry on prior 10-bar low sweep + reclaim with FVG and tight stop."""
        trigger = ICTUnicornStopSweepTrigger()
        # 12 bars: bars 0..9 establish 10-bar low of 100.0
        df = self._build_intraday_df("2026-01-08", "10:00", 12, base_price=101.0)
        df["Low"] = 100.5
        df.loc[df.index[5], "Low"] = 100.0  # 10-bar low = 100.0

        # Bar 11 at 10:55: Sweeps low to 99.70 (<100.0) and closes at 100.40 (>100.0)
        df.loc[df.index[11], "Low"] = 99.70
        df.loc[df.index[11], "Close"] = 100.40
        df["FVG_BULLISH_DRIFT"] = 0.25      # Bullish FVG
        df["OFI_5"] = 0.40                  # Institutional absorption

        signals = trigger.generate_signals(df, "BANKBEES")
        self.assertEqual(len(signals), 1)
        sig = signals[0]
        self.assertEqual(sig.action, TriggerAction.BUY)
        # Tight stop placed below wick: 99.70 - 0.05 = 99.65
        self.assertAlmostEqual(sig.metadata["stop_price"], 99.65, places=2)
        # 3.3:1 target: Risk = 100.40 - 99.65 = 0.75 -> Target = 100.40 + 3.3 * 0.75 = 102.875
        self.assertAlmostEqual(sig.metadata["target_price"], 102.875, places=3)
        self.assertEqual(sig.metadata["risk_reward_ratio"], 3.3)

    # -------------------------------------------------------------------------
    # 10. Sector PEAD Institutional Drift
    # -------------------------------------------------------------------------
    def test_sector_pead_institutional_drift(self) -> None:
        """Verify multi-day volume drift entry with VR > 1.15."""
        trigger = SectorPEADInstitutionalDriftTrigger()
        df = self._build_intraday_df("2026-01-08", "10:00", 4, base_price=100.0)
        df["VOLUME_ZSCORE_LAGGED"] = 2.2  # >= 1.8
        df["LOG_RET_20"] = 0.008          # >= 0.006
        df["VWAP"] = [99.5, 99.8, 100.0, 100.2]  # Rising VWAP
        df["Close"] = [99.6, 99.9, 100.1, 100.3] # Pullback near rising VWAP

        # VR >= 1.15 approves drift
        df["VARIANCE_RATIO_20"] = 1.25
        sigs_high_vr = trigger.generate_signals(df, "ITBEES")
        self.assertGreaterEqual(len(sigs_high_vr), 1)
        self.assertEqual(sigs_high_vr[0].metadata["regime"], "PEAD_INSTITUTIONAL_DRIFT")

        # VR < 1.15 rejects drift
        df["VARIANCE_RATIO_20"] = 1.05
        sigs_low_vr = trigger.generate_signals(df, "ITBEES")
        self.assertEqual(len(sigs_low_vr), 0)

    # -------------------------------------------------------------------------
    # 11. ZeroGivebackStopEngine: 5-Layer Stop and Scale-Out
    # -------------------------------------------------------------------------
    def test_5_layer_stop_and_two_tranche_scaleout(self) -> None:
        """Verify Thesis Stop, Breakeven Ratchet (+0.45 ATR), Tranche-1 Scale-Out (+0.75 ATR), Chandelier, and Time Stop."""
        engine = ZeroGivebackStopEngine()
        entry_price = 100.0
        atr = 2.0

        # Layer 1: Thesis Invalidation Stop
        state = engine.initialize_position(entry_price=entry_price, atr=atr)
        self.assertAlmostEqual(state.active_stop, 100.0 * (1.0 - 0.0035), places=3)

        # Layer 2: Breakeven Ratchet at +0.45 ATR (100 + 0.90 = 100.90)
        state, act1 = engine.update(state, entry_price, bar_high=101.0, bar_low=100.2, bar_close=100.8, atr=atr)
        self.assertTrue(state.is_breakeven_triggered)
        self.assertAlmostEqual(state.active_stop, 100.0 * (1.0 + 0.0002), places=3)
        self.assertEqual(act1, TriggerAction.HOLD)

        # Layer 3: Two-Tranche Scale-Out at +0.75 ATR (100 + 1.50 = 101.50)
        state, act2 = engine.update(state, entry_price, bar_high=101.60, bar_low=101.0, bar_close=101.4, atr=atr, is_mis=True)
        self.assertTrue(state.tranche_1_sold)
        self.assertEqual(act2, TriggerAction.EXIT_TRANCHE_1)
        # Runner stop snapped to +0.25 ATR (100.50)
        self.assertAlmostEqual(state.active_stop, 100.50, places=3)

        # Layer 4: Chandelier Ratchet at +1.0 ATR (100 + 2.0 = 102.0)
        state, act3 = engine.update(state, entry_price, bar_high=103.0, bar_low=102.0, bar_close=102.8, atr=atr)
        # Chandelier stop = Peak (103.0) - 0.35 * 2.0 = 102.30
        self.assertAlmostEqual(state.active_stop, 102.30, places=3)

        # Layer 5: Dead-Money Time Stop (flat exit after 4 bars under +0.20 ATR)
        flat_state = engine.initialize_position(entry_price=entry_price, atr=atr)
        for _ in range(3):
            flat_state, _ = engine.update(flat_state, entry_price, bar_high=100.1, bar_low=99.9, bar_close=100.0, atr=atr)
        flat_state, act_time = engine.update(flat_state, entry_price, bar_high=100.1, bar_low=99.9, bar_close=100.0, atr=atr)
        self.assertEqual(act_time, TriggerAction.EXIT_TIME_STOP)

    # -------------------------------------------------------------------------
    # 12. Bounded AST Modulator: No Saturation
    # -------------------------------------------------------------------------
    def test_bounded_ast_modulator_no_saturation(self) -> None:
        """Verify extreme inputs produce strictly bounded [0.5x, 1.5x] output."""
        modulator = BoundedASTModulator()
        df = self._build_intraday_df("2026-01-08", "10:00", 50, base_price=500.0)
        # Extreme feature swings (+10,000 to -10,000)
        df["RSI_14"] = np.concatenate([np.full(25, 10000.0), np.full(25, -10000.0)])
        node = ASTNode(op="FEATURE", val="RSI_14")

        mults, alert = modulator.compute_multipliers(node, df)
        self.assertFalse(alert)
        self.assertTrue(np.all(np.isfinite(mults)))
        self.assertTrue((mults >= 0.50).all())
        self.assertTrue((mults <= 1.50).all())

    # -------------------------------------------------------------------------
    # 13. AST Modulator: 60-Bar Variance Sentinel
    # -------------------------------------------------------------------------
    def test_ast_modulator_variance_sentinel(self) -> None:
        """Verify zero variance for 60 bars triggers stagnation alert and 1.0x fallback."""
        modulator = BoundedASTModulator()
        df = self._build_intraday_df("2026-01-08", "10:00", 70, base_price=500.0)
        df["RSI_14"] = 50.0  # Perfectly flat zero variance
        node = ASTNode(op="FEATURE", val="RSI_14")

        mults, alert = modulator.compute_multipliers(node, df)
        self.assertTrue(alert)
        self.assertTrue(modulator.stagnation_alert)
        # Must fall back strictly to 1.0x at bar 60+
        self.assertEqual(float(mults.iloc[-1]), 1.0)

    # -------------------------------------------------------------------------
    # 14. Zero Ghost Trades Invariant
    # -------------------------------------------------------------------------
    def test_zero_ghost_trades_invariant(self) -> None:
        """Verify modulated signal is strictly 0.0 when trigger is inactive."""
        modulator = BoundedASTModulator()
        # Direct scalar modulation
        self.assertEqual(modulator.modulate_conviction(0.0, 1.50), 0.0)
        self.assertEqual(modulator.modulate_conviction(0.0, 0.50), 0.0)

        # Vector series modulation
        df = self._build_intraday_df("2026-01-08", "10:00", 5, base_price=500.0)
        df["RSI_14"] = [10.0, 50.0, 90.0, 20.0, 80.0]
        node = ASTNode(op="FEATURE", val="RSI_14")
        base_conv = pd.Series([0.0, 0.85, 0.0, 0.60, 0.0], index=df.index)

        modulated = modulator.modulate_series(base_conv, node, df)
        self.assertEqual(float(modulated.iloc[0]), 0.0)
        self.assertGreater(float(modulated.iloc[1]), 0.0)
        self.assertEqual(float(modulated.iloc[2]), 0.0)
        self.assertGreater(float(modulated.iloc[3]), 0.0)
        self.assertEqual(float(modulated.iloc[4]), 0.0)

    # -------------------------------------------------------------------------
    # 15. Strategy Bandit Allocator: Burn-In and Clamping
    # -------------------------------------------------------------------------
    def test_strategy_bandit_allocator_burn_in_and_clamping(self) -> None:
        """Verify 15-trade burn-in and [0.70x, 1.30x] bounds."""
        bandit = StrategyBanditAllocator(burn_in_trades=15)
        trig = "OpeningRangeBreakoutTrigger"

        # First 14 trades: Burn-in gate strictly enforces 1.0x
        for _ in range(14):
            bandit.update(trig, net_pnl=1000.0)
            self.assertEqual(bandit.get_multiplier(trig), 1.0)

        # Trade 15: Burn-in completes, multiplier adapts
        bandit.update(trig, net_pnl=1000.0)
        self.assertGreater(bandit.get_multiplier(trig), 1.0)

        # Test extreme win streak clamping (<= 1.30x)
        for _ in range(40):
            bandit.update(trig, net_pnl=1000.0)
        self.assertAlmostEqual(bandit.get_multiplier(trig), 1.30, places=2)

        # Test extreme loss streak clamping (>= 0.70x)
        loss_trig = "INAVSpreadMeanReversionTrigger"
        for _ in range(50):
            bandit.update(loss_trig, net_pnl=-1000.0)
        self.assertAlmostEqual(bandit.get_multiplier(loss_trig), 0.70, places=2)

    # -------------------------------------------------------------------------
    # 16. Risk-at-Stop Leveraged Sizing: 2.0x MIS
    # -------------------------------------------------------------------------
    def test_risk_at_stop_leveraged_sizing(self) -> None:
        """Verify 2.0x MIS sizing caps max loss at stop to 0.25% of equity (₹500 on ₹2L)."""
        bridge = StrategyRiskBridge(risk_budget_pct=0.0025, max_mis_leverage=2.0)
        equity = 200000.0
        entry_price = 500.0
        stop_price = 496.25       # Distance = 3.75
        expected_slippage = 0.25  # Total risk/share = 4.00

        res = bridge.calculate_risk_at_stop_size(
            ticker="BANKBEES",
            entry_price=entry_price,
            stop_loss_price=stop_price,
            account_equity=equity,
            product_type="MIS",
            expected_slippage_inr=expected_slippage,
        )

        self.assertTrue(res.is_approved)
        # Shares = 500 / 4.00 = 125 shares -> Notional = ₹62,500
        self.assertEqual(res.shares, 125)
        self.assertEqual(res.notional, 62500.0)
        # Max loss at stop: 125 * 4.00 = ₹500.00 (exactly 0.25% of ₹200,000)
        self.assertLessEqual(res.risk_at_stop, 500.0 + 1e-5)
        self.assertLessEqual(res.risk_pct, 0.0025 + 1e-6)
        self.assertLessEqual(res.leverage_applied, 2.0)

    # -------------------------------------------------------------------------
    # 17. Minimum Economic Lot Filter (₹50,000 Floor)
    # -------------------------------------------------------------------------
    def test_minimum_economic_lot_filter(self) -> None:
        """Rejects trades sized under ₹50,000 to cap DP fee drag <= 3.0 bps."""
        bridge = StrategyRiskBridge(min_trade_notional=50000.0)
        equity = 200000.0

        # Sub-₹50,000 trade: 95 shares @ 500.0 = ₹47,500
        res_small = bridge.calculate_risk_at_stop_size(
            ticker="BANKBEES",
            entry_price=500.0,
            stop_loss_price=495.0,  # Risk/share = 5.25 -> 500 / 5.25 = 95 shares
            account_equity=equity,
            product_type="MIS",
            expected_slippage_inr=0.25,
        )
        self.assertFalse(res_small.is_approved)
        self.assertIn("below minimum ₹50,000.00", res_small.rejection_reason)

        # Above ₹50,000 trade: 125 shares @ 500.0 = ₹62,500
        res_valid = bridge.calculate_risk_at_stop_size(
            ticker="BANKBEES",
            entry_price=500.0,
            stop_loss_price=496.25,
            account_equity=equity,
            product_type="MIS",
            expected_slippage_inr=0.25,
        )
        self.assertTrue(res_valid.is_approved)

    # -------------------------------------------------------------------------
    # 18. SQLite State Ledger: 11:32 AM Reboot Recovery
    # -------------------------------------------------------------------------
    def test_sqlite_state_ledger_reboot_recovery(self) -> None:
        """Verify 11:32 AM reboot reconciliation against live broker positions."""
        ledger = SQLiteStateLedger(db_path=self.db_path)
        ledger.record_entry("BANKBEES", "TRIG_ORB", "MIS", 200, 500.0, 515.0, 495.0)
        ledger.record_entry("NIFTYBEES", "TRIG_INAV", "CNC", 100, 250.0, 260.0, 245.0)
        ledger.record_entry("ITBEES", "TRIG_PEAD", "MIS", 150, 40.0, 44.0, 38.0)

        # Midday snapshot from broker after server reboot
        broker_positions = {
            "BANKBEES": 200,  # In sync
            "ITBEES": 75,     # Tranche 1 half-exited
            "GOLDBEES": 500,  # Orphan position to adopt
        }
        prices = {"BANKBEES": 505.0, "ITBEES": 42.0, "GOLDBEES": 65.0}
        atrs = {"GOLDBEES": 1.50}

        report = ledger.reconcile_with_broker(broker_positions, prices, atrs)
        self.assertTrue(report["discrepancies_found"])

        # BANKBEES in sync
        self.assertEqual(len(report["in_sync"]), 1)
        self.assertEqual(report["in_sync"][0]["ticker"], "BANKBEES")

        # NIFTYBEES closed at broker
        self.assertEqual(len(report["closed"]), 1)
        self.assertEqual(report["closed"][0]["ticker"], "NIFTYBEES")
        self.assertNotIn("NIFTYBEES", ledger.get_active_positions())

        # ITBEES adjusted to 75 shares with tranche_1_filled = True
        self.assertEqual(len(report["adjusted"]), 1)
        it_pos = ledger.get_active_positions()["ITBEES"]
        self.assertEqual(it_pos["shares_remaining"], 75)
        self.assertTrue(it_pos["tranche_1_filled"])

        # GOLDBEES adopted with trailing stop P - 2.0*ATR (65.0 - 3.0 = 62.0)
        self.assertEqual(len(report["adopted"]), 1)
        gold_pos = ledger.get_active_positions()["GOLDBEES"]
        self.assertEqual(gold_pos["shares_remaining"], 500)
        self.assertEqual(gold_pos["trailing_stop_loss"], 62.0)

    # -------------------------------------------------------------------------
    # 19. Shoonya Continuation Gate: DP Fee Savings (₹15.93)
    # -------------------------------------------------------------------------
    def test_shoonya_continuation_gate_dp_fee_savings(self) -> None:
        """Verify 09:15 morning sell is skipped on continuation >= 0.50, saving ₹15.93 DP fee."""
        gateway = ShoonyaBrokerGateway(ShoonyaGatewayConfig(paper_trading=True, statutory_dp_fee_inr=15.93))
        router = BrokerOrderRouter(gateway=gateway, live_mode=False)

        t_morning = pd.Timestamp("2026-01-08 09:15:00", tz="Asia/Kolkata")
        empty_alloc = VolatilitySizedAllocation(
            base_allocation=AllocationResult(
                target_weights={}, target_shares={}, target_notionals={},
                drawdown_state=DrawdownState.NORMAL, drawdown_multiplier=1.0,
                cash_reserve_inr=100000.0, total_allocated_inr=0.0,
            ),
            sized_weights={}, sized_shares={}, sized_notionals={},
            vol_scalars={}, kelly_fraction_applied=0.5,
            cash_reserve_inr=100000.0, total_deployed_inr=0.0,
        )

        receipt = router.route_target_allocation(
            sized_allocation=empty_alloc,
            current_holdings={"NIFTYBEES": 100},
            current_prices={"NIFTYBEES": 250.0},
            timestamp=t_morning,
            hold_on_strong_continuation=True,
            continuation_threshold=0.50,
            continuation_signals={"NIFTYBEES": 0.70},  # Strong continuation >= 0.50
        )

        # Morning sell order must be skipped
        self.assertEqual(len(receipt.submitted_orders), 0)
        self.assertEqual(len(receipt.executed_trades), 0)
        self.assertEqual(router.continuation_holds_count, 1)
        self.assertEqual(gateway.continuation_holds_count, 1)
        # Statutory DP fee preserved: ₹15.93
        self.assertEqual(gateway.config.statutory_dp_fee_inr, 15.93)

    # -------------------------------------------------------------------------
    # 20. Numba Fast Backtest: Sub-50ms Benchmark & IEEE-754 Safety
    # -------------------------------------------------------------------------
    def test_numba_fast_backtest_sub_50ms_parity(self) -> None:
        """Verify Numba JIT backtest runs 100,000 bars in <50ms with IEEE-754 NaN safety."""
        n = 100000
        engine = FastBacktester(BacktestConfig(initial_capital=200000.0))
        dates = pd.date_range("2020-01-01 09:15", periods=n, freq="5min")
        rng = np.random.default_rng(42)
        base = 500.0 + np.cumsum(rng.normal(0, 0.1, n))

        df = pd.DataFrame(
            {
                "Open": base,
                "High": base + rng.uniform(0.1, 1.5, n),
                "Low": base - rng.uniform(0.1, 1.5, n),
                "Close": base + rng.normal(0, 0.2, n),
                "Volume": rng.uniform(10000, 50000, n),
                "ATR": np.full(n, 1.25),
            },
            index=dates,
        )
        signals = rng.uniform(-1.0, 1.0, n)
        # Inject IEEE-754 NaNs to verify robust NaN-safety without crashes
        signals[10:15] = np.nan
        signals[100:105] = np.nan

        # Warmup JIT compiler
        engine.run_fast(df.iloc[:20], signals[:20], signal_threshold=0.5)

        # Benchmark run_fast execution over 100,000 bars
        t0 = time.perf_counter()
        eq, cash, records, count = engine.run_fast(df, signals, signal_threshold=0.6, is_mis=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Verify performance contract: < 50ms
        self.assertLess(elapsed_ms, 50.0, f"Benchmark execution {elapsed_ms:.2f} ms exceeded 50ms limit!")
        self.assertEqual(len(eq), n)
        self.assertGreater(count, 0)
        self.assertTrue(np.all(np.isfinite(eq)))


if __name__ == "__main__":
    unittest.main()
