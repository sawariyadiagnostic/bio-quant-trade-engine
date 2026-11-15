"""Comprehensive Unit Tests for StrategyRiskBridge and SQLiteStateLedger."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

from ravana.risk.circuit_breaker import CircuitBreakerKillSwitch, CircuitBreakerStatus
from ravana.risk.config import DrawdownState
from ravana.strategies.risk_adapter import (
    CorrelationCheckResult,
    RiskBridgeDecision,
    SizingResult,
    StrategyRiskBridge,
)
from ravana.strategies.state_ledger import (
    OrderAuditRecord,
    PositionRecord,
    SQLiteStateLedger,
)


class TestStrategyRiskBridge(unittest.TestCase):
    """Test suite for Muscle 18 Interlock, Risk-at-Stop Sizing, and HRP Correlation Ceiling."""

    def setUp(self) -> None:
        self.kill_switch = CircuitBreakerKillSwitch(max_daily_loss_pct=0.02, max_consecutive_losses=4)
        self.bridge = StrategyRiskBridge(
            kill_switch=self.kill_switch,
            risk_budget_pct=0.0025,
            max_mis_leverage=2.0,
            max_cnc_leverage=0.98,
            min_trade_notional=50000.0,
            correlation_threshold=0.80,
            max_correlated_allocation_pct=0.40,
        )

    def test_circuit_breaker_blocks_buy_permits_sell(self) -> None:
        """Verify Kill Switch / Breaker strictly blocks BUY but permits liquidation / scale-outs."""
        allowed, reason = self.bridge.validate_order_gate("BUY")
        self.assertTrue(allowed)

        # Kill Switch Engaged
        self.kill_switch.trigger_kill_switch("Daily loss > 2.0% halt")
        self.assertTrue(self.kill_switch.is_killed)

        # BUY strictly blocked
        allowed_buy, reason_buy = self.bridge.validate_order_gate("BUY")
        self.assertFalse(allowed_buy)
        self.assertIn("strictly blocked", reason_buy)

        # Liquidations / Scale-outs explicitly permitted
        for side in ["SELL", "EXIT_ALL", "EXIT_TRANCHE_1", "SCALE_OUT", "LIQUIDATE"]:
            allowed_exit, reason_exit = self.bridge.validate_order_gate(side)
            self.assertTrue(allowed_exit, f"Expected {side} to be permitted")
            self.assertIn("explicitly permitted", reason_exit)

        # Unrecognized order side
        allowed_unknown, reason_unknown = self.bridge.validate_order_gate("UNKNOWN_SIDE")
        self.assertFalse(allowed_unknown)
        self.assertIn("Unrecognized order side", reason_unknown)

    def test_breaker_status_interlock(self) -> None:
        """Verify evaluate_order_proposal respects CircuitBreakerStatus tripping."""
        breaker_tripped = CircuitBreakerStatus(
            is_trading_allowed=False,
            kill_switch_engaged=False,
            drawdown_state=DrawdownState.HALTED,
            daily_return_pct=-2.2,
            trailing_drawdown_pct=2.5,
            consecutive_losses=4,
            in_cooldown=True,
            reason="4 consecutive losses cooldown",
        )

        decision_buy = self.bridge.evaluate_order_proposal(
            ticker="BANKBEES",
            side="BUY",
            entry_price=500.0,
            stop_loss_price=495.0,
            account_equity=200000.0,
            breaker_status=breaker_tripped,
        )
        self.assertFalse(decision_buy.is_allowed)
        self.assertEqual(decision_buy.action, "REJECTED")
        self.assertIn("Circuit Breaker", decision_buy.reason)

        decision_exit = self.bridge.evaluate_order_proposal(
            ticker="BANKBEES",
            side="EXIT_TRANCHE_1",
            entry_price=500.0,
            stop_loss_price=495.0,
            account_equity=200000.0,
            breaker_status=breaker_tripped,
        )
        self.assertTrue(decision_exit.is_allowed)
        self.assertEqual(decision_exit.action, "PERMITTED")

    def test_risk_at_stop_sizing_formula(self) -> None:
        """Verify formula: Shares = min( (0.0025 * Equity) / (|P_entry - S_0| + Slip), (2.0 * Equity) / P_entry )."""
        equity = 200000.0
        entry_price = 500.0
        stop_price = 495.0  # Stop distance = 5.0
        slippage_inr = 0.25  # Total per-share risk = 5.25

        # ₹47,500 < ₹50,000 floor -> Must be rejected due to DP fee drag
        res = self.bridge.calculate_risk_at_stop_size(
            ticker="BANKBEES",
            entry_price=entry_price,
            stop_loss_price=stop_price,
            account_equity=equity,
            product_type="MIS",
            expected_slippage_inr=slippage_inr,
        )
        self.assertFalse(res.is_approved)
        self.assertIn("below minimum ₹50,000.00", res.rejection_reason)

        # Tighter stop -> 125 shares -> ₹62,500 (>= ₹50,000)
        res2 = self.bridge.calculate_risk_at_stop_size(
            ticker="BANKBEES",
            entry_price=entry_price,
            stop_loss_price=496.25,
            account_equity=equity,
            product_type="MIS",
            expected_slippage_inr=slippage_inr,
        )
        self.assertTrue(res2.is_approved)
        self.assertEqual(res2.shares, 125)
        self.assertEqual(res2.notional, 62500.0)
        self.assertLessEqual(res2.risk_at_stop, 500.0 + 1e-6)
        self.assertLessEqual(res2.risk_pct, 0.0025 + 1e-6)
        self.assertEqual(res2.max_leverage_allowed, 2.0)
        self.assertEqual(res2.leverage_applied, 62500.0 / equity)

    def test_sizing_input_validation(self) -> None:
        """Verify rejection of NaN, Inf, non-positive equity, or prices."""
        res_nan = self.bridge.calculate_risk_at_stop_size("X", float("nan"), 100.0, 200000.0)
        self.assertFalse(res_nan.is_approved)

        res_neg_eq = self.bridge.calculate_risk_at_stop_size("X", 100.0, 95.0, -1000.0)
        self.assertFalse(res_neg_eq.is_approved)

        res_zero_px = self.bridge.calculate_risk_at_stop_size("X", 0.0, 95.0, 200000.0)
        self.assertFalse(res_zero_px.is_approved)

    def test_cnc_overnight_leverage_cap_0_98(self) -> None:
        """Verify OVERNIGHT CNC strictly caps leverage at 0.98x cash."""
        equity = 200000.0
        entry_price = 100.0
        stop_price = 99.90
        res_cnc = self.bridge.calculate_risk_at_stop_size(
            ticker="NIFTYBEES",
            entry_price=entry_price,
            stop_loss_price=stop_price,
            account_equity=equity,
            product_type="CNC",
            expected_slippage_inr=0.01,
        )
        self.assertTrue(res_cnc.is_approved)
        self.assertEqual(res_cnc.max_leverage_allowed, 0.98)
        self.assertEqual(res_cnc.shares, 1960)
        self.assertLessEqual(res_cnc.notional, 0.98 * equity + 1e-6)

    def test_hrp_sector_correlation_ceiling(self) -> None:
        """Verify pairwise correlated assets (>0.80) capped at 40% equity."""
        equity = 200000.0  # Max cluster = 40% = ₹80,000
        current_holdings = {
            "BANKBEES": 50000.0,
        }
        corr_matrix = pd.DataFrame(
            [
                [1.00, 0.92, 0.35],
                [0.92, 1.00, 0.30],
                [0.35, 0.30, 1.00],
            ],
            index=["BANKBEES", "PSUBNKBEES", "GOLDBEES"],
            columns=["BANKBEES", "PSUBNKBEES", "GOLDBEES"],
        )

        # Propose ₹35,000 in PSUBNKBEES -> Cluster total = 85k (> 80k) -> REJECT
        check_breach = self.bridge.check_hrp_correlation_ceiling(
            candidate_ticker="PSUBNKBEES",
            candidate_notional=35000.0,
            current_holdings_notional=current_holdings,
            correlation_matrix=corr_matrix,
            account_equity=equity,
        )
        self.assertFalse(check_breach.is_approved)
        self.assertIn("HRP Correlation Ceiling (40.0%) breached", check_breach.reason)
        self.assertEqual(check_breach.max_allowed_notional, 30000.0)

        # Propose ₹25,000 in PSUBNKBEES -> Cluster total = 75k (<= 80k) -> APPROVE
        check_pass = self.bridge.check_hrp_correlation_ceiling(
            candidate_ticker="PSUBNKBEES",
            candidate_notional=25000.0,
            current_holdings_notional=current_holdings,
            correlation_matrix=corr_matrix,
            account_equity=equity,
        )
        self.assertTrue(check_pass.is_approved)

        # Negative equity check
        check_neg = self.bridge.check_hrp_correlation_ceiling("X", 1000.0, {}, corr_matrix, -5000.0)
        self.assertFalse(check_neg.is_approved)

    def test_evaluate_order_proposal_full_chain(self) -> None:
        """Verify end-to-end order proposal evaluation across gate, sizing, and HRP ceiling."""
        equity = 200000.0
        corr_matrix = pd.DataFrame(
            [[1.0, 0.95], [0.95, 1.0]],
            index=["BANKBEES", "PSUBNKBEES"],
            columns=["BANKBEES", "PSUBNKBEES"],
        )
        holdings = {"BANKBEES": 60000.0}

        # Candidate PSUBNKBEES sized at ₹55,000 -> Total cluster = ₹115,000 (> ₹80,000 ceiling)
        decision = self.bridge.evaluate_order_proposal(
            ticker="PSUBNKBEES",
            side="BUY",
            entry_price=100.0,
            stop_loss_price=99.10,  # ~1.0 risk -> 500 shares -> 50k notional
            account_equity=equity,
            product_type="MIS",
            current_holdings_notional=holdings,
            correlation_matrix=corr_matrix,
            expected_slippage_inr=0.05,
        )
        self.assertFalse(decision.is_allowed)
        self.assertEqual(decision.action, "REJECTED")
        self.assertIn("Correlation ceiling rejected", decision.reason)


class TestSQLiteStateLedger(unittest.TestCase):
    """Test suite for ACID SQLite State Machine & 11:32 AM Reboot Recovery Protocol."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_ledger.db")
        self.ledger = SQLiteStateLedger(db_path=self.db_path)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_record_entry_and_get_active(self) -> None:
        """Verify position entry recording and reboot recovery state retrieval."""
        self.ledger.record_entry(
            ticker="BANKBEES",
            originating_trigger="BULL_ENGULFING_VWAP",
            product_type="MIS",
            shares_total=200,
            entry_price=510.50,
            dynamic_profit_target=520.00,
            trailing_stop_loss=505.00,
        )

        positions = self.ledger.get_active_positions()
        self.assertIn("BANKBEES", positions)
        pos = positions["BANKBEES"]
        self.assertEqual(pos["ticker"], "BANKBEES")
        self.assertEqual(pos["shares_total"], 200)
        self.assertEqual(pos["shares_remaining"], 200)
        self.assertFalse(pos["tranche_1_filled"])
        self.assertEqual(pos["entry_price"], 510.50)
        self.assertEqual(pos["peak_price_observed"], 510.50)
        self.assertEqual(pos["holding_bars"], 0)

    def test_update_ratchet(self) -> None:
        """Verify trailing stop ratchet, peak price updates, and holding bar increments."""
        self.ledger.record_entry(
            ticker="BANKBEES",
            originating_trigger="TEST",
            product_type="MIS",
            shares_total=100,
            entry_price=500.0,
            dynamic_profit_target=515.0,
            trailing_stop_loss=495.0,
        )

        success = self.ledger.update_ratchet(
            ticker="BANKBEES",
            trailing_stop_loss=498.0,
            peak_price_observed=505.0,
        )
        self.assertTrue(success)

        pos = self.ledger.get_active_positions()["BANKBEES"]
        self.assertEqual(pos["trailing_stop_loss"], 498.0)
        self.assertEqual(pos["peak_price_observed"], 505.0)
        self.assertEqual(pos["holding_bars"], 1)

        # Explicit holding bars
        self.ledger.update_ratchet("BANKBEES", 500.0, 508.0, holding_bars=5)
        pos2 = self.ledger.get_active_positions()["BANKBEES"]
        self.assertEqual(pos2["holding_bars"], 5)

    def test_tranche_1_scale_out(self) -> None:
        """Verify 50% scale-out fill records tranche_1_filled and adjusts remaining shares."""
        self.ledger.record_entry(
            ticker="NIFTYBEES",
            originating_trigger="TEST",
            product_type="MIS",
            shares_total=300,
            entry_price=250.0,
            dynamic_profit_target=255.0,
            trailing_stop_loss=248.0,
        )

        success, exited = self.ledger.record_tranche_1_exit("NIFTYBEES")
        self.assertTrue(success)
        self.assertEqual(exited, 150)

        pos = self.ledger.get_active_positions()["NIFTYBEES"]
        self.assertEqual(pos["shares_remaining"], 150)
        self.assertTrue(pos["tranche_1_filled"])

        # Non-existent position exit returns False, 0
        s_fake, e_fake = self.ledger.record_tranche_1_exit("NON_EXISTENT")
        self.assertFalse(s_fake)
        self.assertEqual(e_fake, 0)

    def test_close_position(self) -> None:
        """Verify close_position deletes row from active positions table."""
        self.ledger.record_entry(
            ticker="GOLDBEES",
            originating_trigger="TEST",
            product_type="CNC",
            shares_total=100,
            entry_price=60.0,
            dynamic_profit_target=65.0,
            trailing_stop_loss=58.0,
        )
        self.assertIn("GOLDBEES", self.ledger.get_active_positions())

        closed = self.ledger.close_position("GOLDBEES")
        self.assertTrue(closed)
        self.assertNotIn("GOLDBEES", self.ledger.get_active_positions())

    def test_persistence_across_reconnect(self) -> None:
        """Verify state persists in SQLite across independent connection instances."""
        self.ledger.record_entry(
            ticker="LIQUIDBEES",
            originating_trigger="TEST_INIT",
            product_type="CNC",
            shares_total=1000,
            entry_price=1000.0,
            dynamic_profit_target=1005.0,
            trailing_stop_loss=998.0,
        )
        self.ledger.record_order("ORD_INIT", "LIQUIDBEES", "BUY", 1000, "INIT")

        # New instance pointing to same db file
        fresh_ledger = SQLiteStateLedger(db_path=self.db_path)
        active = fresh_ledger.get_active_positions()
        self.assertIn("LIQUIDBEES", active)
        self.assertEqual(active["LIQUIDBEES"]["shares_remaining"], 1000)

        trail = fresh_ledger.get_order_audit_trail("LIQUIDBEES")
        self.assertEqual(len(trail), 1)
        self.assertEqual(trail[0]["client_order_id"], "ORD_INIT")

    def test_order_audit_trail_flow(self) -> None:
        """Verify immutable order audit trail insertion and reconciliation."""
        self.ledger.record_order(
            client_order_id="ORD_1001",
            ticker="BANKBEES",
            side="BUY",
            shares=100,
            trigger_source="VOLATILITY_BREAKOUT",
            limit_price=510.0,
        )
        trail = self.ledger.get_order_audit_trail("BANKBEES")
        self.assertEqual(len(trail), 1)
        self.assertEqual(trail[0]["status"], "SUBMITTED")

        reconciled = self.ledger.reconcile_order(
            client_order_id="ORD_1001",
            exchange_order_id="EXCH_99999",
            status="FILLED",
        )
        self.assertTrue(reconciled)

        updated_trail = self.ledger.get_order_audit_trail("BANKBEES")
        self.assertEqual(updated_trail[0]["status"], "FILLED")
        self.assertEqual(updated_trail[0]["exchange_order_id"], "EXCH_99999")

    def test_1132_reboot_recovery_protocol(self) -> None:
        """Verify the 11:32 AM Reboot Recovery Protocol."""
        self.ledger.record_entry("BANKBEES", "TRIG_A", "MIS", 200, 500.0, 515.0, 495.0)
        self.ledger.record_entry("NIFTYBEES", "TRIG_B", "CNC", 100, 250.0, 260.0, 245.0)
        self.ledger.record_entry("ITBEES", "TRIG_C", "MIS", 150, 40.0, 44.0, 38.0)

        broker_positions = {
            "BANKBEES": 200,
            "ITBEES": 75,
            "GOLDBEES": 500,
        }
        current_prices = {
            "BANKBEES": 505.0,
            "ITBEES": 42.0,
            "GOLDBEES": 65.0,
        }
        current_atrs = {
            "GOLDBEES": 1.50,
        }

        report = self.ledger.reconcile_with_broker(
            broker_positions=broker_positions,
            current_prices=current_prices,
            current_atrs=current_atrs,
        )

        self.assertTrue(report["discrepancies_found"])
        self.assertEqual(len(report["in_sync"]), 1)
        self.assertEqual(report["in_sync"][0]["ticker"], "BANKBEES")

        # NIFTYBEES closed
        self.assertEqual(len(report["closed"]), 1)
        self.assertEqual(report["closed"][0]["ticker"], "NIFTYBEES")
        self.assertNotIn("NIFTYBEES", self.ledger.get_active_positions())

        # ITBEES adjusted
        self.assertEqual(len(report["adjusted"]), 1)
        self.assertEqual(report["adjusted"][0]["ticker"], "ITBEES")
        it_pos = self.ledger.get_active_positions()["ITBEES"]
        self.assertEqual(it_pos["shares_remaining"], 75)
        self.assertTrue(it_pos["tranche_1_filled"])

        # GOLDBEES adopted with P - 2.0*ATR
        self.assertEqual(len(report["adopted"]), 1)
        self.assertEqual(report["adopted"][0]["ticker"], "GOLDBEES")
        self.assertEqual(report["adopted"][0]["trailing_stop_loss"], 62.0)

        gold_pos = self.ledger.get_active_positions()["GOLDBEES"]
        self.assertEqual(gold_pos["shares_remaining"], 500)
        self.assertEqual(gold_pos["trailing_stop_loss"], 62.0)
        self.assertEqual(gold_pos["dynamic_profit_target"], 68.0)
        self.assertEqual(gold_pos["originating_trigger"], "EMERGENCY_REBOOT_RECOVERY")


if __name__ == "__main__":
    unittest.main()
