"""Unit tests for DecisionVetoGuard.

Tests signal evaluation against Gemini Decision mock/synthetic responses,
threshold-based vetoes (false breakout trap, dead money risk), approved flow,
microstructure feature extraction, and SQLiteStateLedger audit logging.
"""

from __future__ import annotations

import gc
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
trade_clean_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
ravana_dir = os.path.abspath(os.path.join(current_dir, ".."))
for p in (trade_clean_dir, ravana_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from ravana.decisions.client import GeminiDecisionClient
from ravana.decisions.schema import BatchResponse, DecisionResult, DecisionType
from ravana.decisions.veto_guard import DecisionVetoGuard
from ravana.strategies.base import TriggerAction, TriggerSignal
from ravana.strategies.state_ledger import SQLiteStateLedger


class TestDecisionVetoGuard(unittest.TestCase):
    """Test suite for DecisionVetoGuard institutional veto logic."""

    def setUp(self) -> None:
        self.timestamp = pd.Timestamp("2026-11-16 10:30:00")
        self.signal = TriggerSignal(
            timestamp=self.timestamp,
            symbol="BANKBEES",
            action=TriggerAction.BUY,
            conviction=0.85,
            trigger_name="TwoWayVWAPAbsorptionTrigger",
            metadata={"timeframe": "5m"},
        )
        self.bar = pd.Series(
            {
                "Open": 450.0,
                "High": 455.0,
                "Low": 449.0,
                "Close": 454.5,
                "VWAP": 452.0,
                "ATR_14": 3.5,
                "SPREAD_Z": 0.8,
                "OFI_5": 0.35,
                "VARIANCE_RATIO_20": 0.82,
            },
            name=self.timestamp,
        )

    def test_default_mock_client_veto_behavior(self) -> None:
        """Default mock client returns prob 0.85, exceeding false_breakout_threshold (0.55)."""
        guard = DecisionVetoGuard(
            dead_money_threshold=0.60,
            false_breakout_threshold=0.55,
        )
        approved, reason, prob = guard.evaluate_signal(self.signal, self.bar)
        self.assertFalse(approved)
        self.assertEqual(reason, "VETO_FALSE_BREAKOUT_TRAP")
        self.assertAlmostEqual(prob, 0.85)

    def test_false_breakout_trap_veto(self) -> None:
        """When false breakout prob > threshold, reject immediately."""
        client = MagicMock(spec=GeminiDecisionClient)
        client.evaluate.return_value = BatchResponse(
            results={
                "is_false_breakout_trap": DecisionResult(
                    key="is_false_breakout_trap",
                    decision_type=DecisionType.NOUL,
                    selected_choice="true",
                    probability=0.72,
                ),
                "is_dead_money_risk": DecisionResult(
                    key="is_dead_money_risk",
                    decision_type=DecisionType.NOUL,
                    selected_choice="false",
                    probability=0.25,
                ),
                "is_volume_absorption_genuine": DecisionResult(
                    key="is_volume_absorption_genuine",
                    decision_type=DecisionType.NOUL,
                    selected_choice="true",
                    probability=0.80,
                ),
            },
            latency_ms=12.5,
            status="OK",
        )

        guard = DecisionVetoGuard(client=client, false_breakout_threshold=0.55)
        approved, reason, prob = guard.evaluate_signal(self.signal, self.bar)

        self.assertFalse(approved)
        self.assertEqual(reason, "VETO_FALSE_BREAKOUT_TRAP")
        self.assertAlmostEqual(prob, 0.72)

    def test_dead_money_risk_veto(self) -> None:
        """When false breakout is low but dead money prob > threshold, reject."""
        client = MagicMock(spec=GeminiDecisionClient)
        client.evaluate.return_value = BatchResponse(
            results={
                "is_false_breakout_trap": DecisionResult(
                    key="is_false_breakout_trap",
                    decision_type=DecisionType.NOUL,
                    selected_choice="false",
                    probability=0.30,
                ),
                "is_dead_money_risk": DecisionResult(
                    key="is_dead_money_risk",
                    decision_type=DecisionType.NOUL,
                    selected_choice="true",
                    probability=0.68,
                ),
                "is_volume_absorption_genuine": DecisionResult(
                    key="is_volume_absorption_genuine",
                    decision_type=DecisionType.NOUL,
                    selected_choice="true",
                    probability=0.75,
                ),
            },
            latency_ms=14.0,
            status="OK",
        )

        guard = DecisionVetoGuard(
            client=client,
            dead_money_threshold=0.60,
            false_breakout_threshold=0.55,
        )
        approved, reason, prob = guard.evaluate_signal(self.signal, self.bar)

        self.assertFalse(approved)
        self.assertEqual(reason, "VETO_DEAD_MONEY_RISK")
        self.assertAlmostEqual(prob, 0.68)

    def test_approved_signal_flow(self) -> None:
        """When both veto risks are below thresholds, approve with absorption prob."""
        client = MagicMock(spec=GeminiDecisionClient)
        client.evaluate.return_value = BatchResponse(
            results={
                "is_false_breakout_trap": DecisionResult(
                    key="is_false_breakout_trap",
                    decision_type=DecisionType.NOUL,
                    selected_choice="false",
                    probability=0.20,
                ),
                "is_dead_money_risk": DecisionResult(
                    key="is_dead_money_risk",
                    decision_type=DecisionType.NOUL,
                    selected_choice="false",
                    probability=0.35,
                ),
                "is_volume_absorption_genuine": DecisionResult(
                    key="is_volume_absorption_genuine",
                    decision_type=DecisionType.NOUL,
                    selected_choice="true",
                    probability=0.91,
                ),
            },
            latency_ms=15.0,
            status="OK",
        )

        guard = DecisionVetoGuard(
            client=client,
            dead_money_threshold=0.60,
            false_breakout_threshold=0.55,
        )
        approved, reason, prob = guard.evaluate_signal(self.signal, self.bar)

        self.assertTrue(approved)
        self.assertEqual(reason, "APPROVED")
        self.assertAlmostEqual(prob, 0.91)

    def test_custom_thresholds_override(self) -> None:
        """Customizing thresholds changes veto boundaries."""
        client = MagicMock(spec=GeminiDecisionClient)
        client.evaluate.return_value = BatchResponse(
            results={
                "is_false_breakout_trap": DecisionResult(
                    key="is_false_breakout_trap",
                    decision_type=DecisionType.NOUL,
                    selected_choice="true",
                    probability=0.62,
                ),
                "is_dead_money_risk": DecisionResult(
                    key="is_dead_money_risk",
                    decision_type=DecisionType.NOUL,
                    selected_choice="false",
                    probability=0.40,
                ),
                "is_volume_absorption_genuine": DecisionResult(
                    key="is_volume_absorption_genuine",
                    decision_type=DecisionType.NOUL,
                    selected_choice="true",
                    probability=0.88,
                ),
            },
            latency_ms=10.0,
            status="OK",
        )

        # Default 0.55 vetoes 0.62
        strict_guard = DecisionVetoGuard(client=client, false_breakout_threshold=0.55)
        self.assertFalse(strict_guard.evaluate_signal(self.signal, self.bar)[0])

        # Lenient 0.70 approves 0.62
        lenient_guard = DecisionVetoGuard(client=client, false_breakout_threshold=0.70)
        approved, reason, prob = lenient_guard.evaluate_signal(self.signal, self.bar)
        self.assertTrue(approved)
        self.assertEqual(reason, "APPROVED")
        self.assertAlmostEqual(prob, 0.88)

    def test_context_summary_and_missing_bar_fields(self) -> None:
        """Context summary is formatted and sparse bar does not raise exceptions."""
        client = MagicMock(spec=GeminiDecisionClient)
        client.evaluate.return_value = BatchResponse(
            results={
                "is_false_breakout_trap": DecisionResult(
                    key="is_false_breakout_trap",
                    decision_type=DecisionType.NOUL,
                    selected_choice="false",
                    probability=0.10,
                ),
                "is_dead_money_risk": DecisionResult(
                    key="is_dead_money_risk",
                    decision_type=DecisionType.NOUL,
                    selected_choice="false",
                    probability=0.15,
                ),
                "is_volume_absorption_genuine": DecisionResult(
                    key="is_volume_absorption_genuine",
                    decision_type=DecisionType.NOUL,
                    selected_choice="true",
                    probability=0.95,
                ),
            },
            latency_ms=8.0,
            status="OK",
        )

        sparse_bar = pd.Series({"Close": 100.0})
        guard = DecisionVetoGuard(client=client)
        approved, reason, prob = guard.evaluate_signal(
            self.signal,
            sparse_bar,
            context_summary="Nifty consolidating near VWAP support",
        )

        self.assertTrue(approved)
        client.evaluate.assert_called_once()
        call_kwargs = client.evaluate.call_args[1]
        context_str = call_kwargs["context"]
        self.assertIn("Nifty consolidating near VWAP support", context_str)
        self.assertIn("Symbol: BANKBEES", context_str)
        self.assertIn("Price: 100.00", context_str)

    def test_sqlite_state_ledger_audit_logging(self) -> None:
        """Audit records are persisted to SQLiteStateLedger on veto and approval."""
        tmpdir = tempfile.mkdtemp()
        try:
            db_path = os.path.join(tmpdir, "test_ledger.db")
            ledger = SQLiteStateLedger(db_path=db_path)

            client = MagicMock(spec=GeminiDecisionClient)
            client.evaluate.return_value = BatchResponse(
                results={
                    "is_false_breakout_trap": DecisionResult(
                        key="is_false_breakout_trap",
                        decision_type=DecisionType.NOUL,
                        selected_choice="true",
                        probability=0.75,
                    ),
                    "is_dead_money_risk": DecisionResult(
                        key="is_dead_money_risk",
                        decision_type=DecisionType.NOUL,
                        selected_choice="false",
                        probability=0.20,
                    ),
                    "is_volume_absorption_genuine": DecisionResult(
                        key="is_volume_absorption_genuine",
                        decision_type=DecisionType.NOUL,
                        selected_choice="true",
                        probability=0.60,
                    ),
                },
                latency_ms=25.0,
                status="OK",
            )

            guard = DecisionVetoGuard(client=client, ledger=ledger)
            approved, reason, prob = guard.evaluate_signal(self.signal, self.bar)
            self.assertFalse(approved)
            self.assertEqual(reason, "VETO_FALSE_BREAKOUT_TRAP")

            # Check ledger records
            decisions = ledger.get_recent_decisions(symbol="BANKBEES", cluster_id="veto")
            self.assertEqual(len(decisions), 3)

            trap_audit = next(d for d in decisions if d["question_key"] == "is_false_breakout_trap")
            self.assertEqual(trap_audit["status"], "VETOED")
            self.assertEqual(trap_audit["cluster_id"], "veto")
            self.assertAlmostEqual(trap_audit["probability_score"], 0.75)
            self.assertEqual(trap_audit["timeframe"], "5m")
        finally:
            del ledger
            gc.collect()
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
