"""Unit tests for UniverseAlphaCrown.

Tests top allocation selection across empty, single, and multiple candidate signals,
choice resolution, score to capital allocation fraction mapping, CASH selection,
and SQLiteStateLedger audit logging.
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
from ravana.decisions.universe_allocator import UniverseAlphaCrown
from ravana.strategies.base import TriggerAction, TriggerSignal
from ravana.strategies.state_ledger import SQLiteStateLedger


class TestUniverseAlphaCrown(unittest.TestCase):
    """Test suite for UniverseAlphaCrown allocation logic."""

    def setUp(self) -> None:
        self.ts = pd.Timestamp("2026-11-16 14:00:00")
        self.sig_bank = TriggerSignal(
            timestamp=self.ts,
            symbol="BANKBEES",
            action=TriggerAction.BUY,
            conviction=0.85,
            trigger_name="TwoWayVWAPAbsorptionTrigger",
            metadata={"timeframe": "5m"},
        )
        self.sig_it = TriggerSignal(
            timestamp=self.ts,
            symbol="ITBEES",
            action=TriggerAction.BUY,
            conviction=0.88,
            trigger_name="OpeningRangeBreakoutTrigger",
            metadata={"timeframe": "5m"},
        )
        self.sig_gold = TriggerSignal(
            timestamp=self.ts,
            symbol="GOLDBEES",
            action=TriggerAction.BUY,
            conviction=0.75,
            trigger_name="AfternoonBondEuropeTWAPTrigger",
            metadata={"timeframe": "5m"},
        )

        self.current_bars = {
            "BANKBEES": pd.Series({"Close": 450.0, "VWAP": 448.0, "ATR_14": 3.2, "VR_20": 0.85}),
            "ITBEES": pd.Series({"Close": 38.5, "VWAP": 38.2, "ATR_14": 0.45, "VR_20": 1.15}),
            "GOLDBEES": pd.Series({"Close": 58.0, "VWAP": 57.8, "ATR_14": 0.30, "VR_20": 0.95}),
        }

    def test_empty_candidate_signals(self) -> None:
        """Empty candidates returns (None, 0.0)."""
        crown = UniverseAlphaCrown()
        winner, fraction = crown.select_top_allocation([], self.current_bars)
        self.assertIsNone(winner)
        self.assertEqual(fraction, 0.0)

    def test_single_candidate_signal(self) -> None:
        """Single candidate receives 100% allocation without evaluating."""
        crown = UniverseAlphaCrown()
        winner, fraction = crown.select_top_allocation([self.sig_bank], self.current_bars)
        self.assertEqual(winner, self.sig_bank)
        self.assertEqual(fraction, 1.0)

    def test_multiple_candidates_winning_selection(self) -> None:
        """Evaluator selects ITBEES with score 3 (75% allocation)."""
        client = MagicMock(spec=GeminiDecisionClient)
        client.evaluate.return_value = BatchResponse(
            results={
                "best_etf_to_buy_now": DecisionResult(
                    key="best_etf_to_buy_now",
                    decision_type=DecisionType.CHOICE,
                    selected_choice="ITBEES",
                    probability=0.84,
                ),
                "capital_allocation_fraction": DecisionResult(
                    key="capital_allocation_fraction",
                    decision_type=DecisionType.SCORE,
                    score=3,
                    selected_choice="3",
                    probability=0.78,
                ),
            },
            latency_ms=18.0,
            status="OK",
        )

        crown = UniverseAlphaCrown(client=client)
        winner, fraction = crown.select_top_allocation(
            [self.sig_bank, self.sig_it, self.sig_gold],
            self.current_bars,
            capital=250000.0,
        )

        self.assertEqual(winner, self.sig_it)
        self.assertAlmostEqual(fraction, 0.75)

    def test_multiple_candidates_cash_selection(self) -> None:
        """When evaluator chooses CASH, return (None, 0.0)."""
        client = MagicMock(spec=GeminiDecisionClient)
        client.evaluate.return_value = BatchResponse(
            results={
                "best_etf_to_buy_now": DecisionResult(
                    key="best_etf_to_buy_now",
                    decision_type=DecisionType.CHOICE,
                    selected_choice="CASH",
                    probability=0.90,
                ),
                "capital_allocation_fraction": DecisionResult(
                    key="capital_allocation_fraction",
                    decision_type=DecisionType.SCORE,
                    score=0,
                    selected_choice="0",
                    probability=0.95,
                ),
            },
            latency_ms=15.0,
            status="OK",
        )

        crown = UniverseAlphaCrown(client=client)
        winner, fraction = crown.select_top_allocation(
            [self.sig_bank, self.sig_it],
            self.current_bars,
        )

        self.assertIsNone(winner)
        self.assertEqual(fraction, 0.0)

    def test_allocation_fraction_scores(self) -> None:
        """Test 0-4 score mappings to discrete fractions."""
        crown = UniverseAlphaCrown()

        score_map = {
            0: 0.0,
            1: 0.25,
            2: 0.50,
            3: 0.75,
            4: 1.0,
        }
        for score_val, expected_fraction in score_map.items():
            res = DecisionResult(
                key="capital_allocation_fraction",
                decision_type=DecisionType.SCORE,
                score=score_val,
                selected_choice=str(score_val),
            )
            fraction = crown._resolve_score_fraction(res)
            self.assertAlmostEqual(fraction, expected_fraction)

    def test_default_mock_client_behavior(self) -> None:
        """Default mock client selects first candidate symbol with score 2 (0.50)."""
        crown = UniverseAlphaCrown()
        winner, fraction = crown.select_top_allocation(
            [self.sig_bank, self.sig_it],
            self.current_bars,
        )
        self.assertEqual(winner, self.sig_bank)
        self.assertAlmostEqual(fraction, 0.50)

    def test_sqlite_state_ledger_audit_logging(self) -> None:
        """Audit records are logged to SQLiteStateLedger with cluster_id='alpha_crown'."""
        tmpdir = tempfile.mkdtemp()
        try:
            db_path = os.path.join(tmpdir, "test_allocator_ledger.db")
            ledger = SQLiteStateLedger(db_path=db_path)

            client = MagicMock(spec=GeminiDecisionClient)
            client.evaluate.return_value = BatchResponse(
                results={
                    "best_etf_to_buy_now": DecisionResult(
                        key="best_etf_to_buy_now",
                        decision_type=DecisionType.CHOICE,
                        selected_choice="BANKBEES",
                        probability=0.82,
                    ),
                    "capital_allocation_fraction": DecisionResult(
                        key="capital_allocation_fraction",
                        decision_type=DecisionType.SCORE,
                        score=4,
                        selected_choice="4",
                        probability=0.88,
                    ),
                },
                latency_ms=22.0,
                status="OK",
            )

            crown = UniverseAlphaCrown(client=client, ledger=ledger)
            winner, fraction = crown.select_top_allocation(
                [self.sig_bank, self.sig_it],
                self.current_bars,
            )
            self.assertEqual(winner, self.sig_bank)
            self.assertEqual(fraction, 1.0)

            # Check audit records in ledger
            decisions = ledger.get_recent_decisions(cluster_id="alpha_crown")
            self.assertEqual(len(decisions), 2)

            choice_record = next(d for d in decisions if d["question_key"] == "best_etf_to_buy_now")
            self.assertEqual(choice_record["status"], "APPLIED")
            self.assertEqual(choice_record["symbol"], "BANKBEES")
            self.assertEqual(choice_record["cluster_id"], "alpha_crown")

            score_record = next(d for d in decisions if d["question_key"] == "capital_allocation_fraction")
            self.assertEqual(score_record["status"], "APPLIED")
            self.assertEqual(score_record["selected_choice"], "4")
        finally:
            del ledger
            gc.collect()
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
