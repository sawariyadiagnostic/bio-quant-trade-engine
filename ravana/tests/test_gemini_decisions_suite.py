"""Comprehensive End-to-End Integration Test Suite for Gemini Decisions Architecture.

Validates all 8 decision clusters, client mechanics, rate limiting, and ACID persistence:
1. Rate Limiter and Resilient Client Mechanics
2. Dead-Money and False Breakout Trap Veto (DecisionVetoGuard)
3. Universe Alpha Crown Winner-Takes-All Allocation (UniverseAlphaCrown)
4. Elastic String 1-Minute Live Basis Stat-Arb (ElasticPairConfig)
5. Dynamic Volatility Adaptive Stop Loss (DecisionRiskShield)
6. Pre-15:00 Broker MIS Square-off Dump Shield (DecisionRiskShield)
7. Morning 09:15 Continuation Gate (CDSL DP Fee Preservation)
8. SQLite State Ledger ACID Audit Trail Persistence
"""

import os
import shutil
import tempfile
import unittest
from typing import Any, Dict, List
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from ravana.decisions.client import GeminiDecisionClient, RateLimiter
from ravana.decisions.schema import BatchResponse, DecisionResult, DecisionType, Question
from ravana.decisions.universe_allocator import UniverseAlphaCrown
from ravana.decisions.veto_guard import DecisionVetoGuard
from ravana.risk.decision_shield import DecisionRiskShield
from ravana.strategies.base import Action, TriggerSignal
from ravana.strategies.elastic_string import (
    CORE_ELASTIC_PAIRS,
    ElasticPairConfig,
    compute_basis_tension,
    generate_signals,
)
from ravana.strategies.state_ledger import DecisionAuditRecord, SQLiteStateLedger


class TestGeminiDecisionsEndToEndSuite(unittest.TestCase):
    """End-to-End test suite covering all 8 Gemini Decisions architectural clusters."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test_decisions.db")
        self.ledger = SQLiteStateLedger(db_path=self.db_path)

    def tearDown(self) -> None:
        if self.ledger is not None:
            self.ledger.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _create_mock_client(
        self,
        results_map: Dict[str, DecisionResult],
        latency_ms: float = 12.5,
    ) -> MagicMock:
        client = MagicMock(spec=GeminiDecisionClient)
        batch = BatchResponse(results=results_map, latency_ms=latency_ms, status="OK")
        client.evaluate.return_value = batch
        return client

    # Cluster 1: Rate Limiter and Client Mechanics
    def test_cluster_1_rate_limiter_and_client_mock(self) -> None:
        limiter = RateLimiter(max_rpm=120, max_tpm=10000)
        self.assertTrue(limiter.acquire(tokens=100))
        rpm, tpm = limiter.get_usage()
        self.assertEqual(tpm, 100)
        self.assertEqual(rpm, 1)

        mock_client = GeminiDecisionClient(mock=True)
        q = Question(key="test_noul", question="Test?", decision_type=DecisionType.NOUL)
        resp = mock_client.evaluate(context="Test context", questions=[q])
        self.assertEqual(resp.status, "OK")
        self.assertIn("test_noul", resp.results)
        self.assertAlmostEqual(resp.results["test_noul"].probability, 0.85, places=2)

    # Cluster 2: Dead-Money and False Breakout Trap Veto
    def test_cluster_2_veto_guard_blocks_trap_and_approves_genuine(self) -> None:
        bar = pd.Series({"Close": 520.0, "Volume": 450000, "ATR_14": 4.5, "VWAP": 518.0})
        signal = TriggerSignal(
            timestamp=pd.Timestamp("2026-03-01 09:30:00"),
            symbol="BANKBEES",
            action=Action.BUY,
            conviction=0.88,
            trigger_name="MorningOverflowTrigger",
            metadata={"timeframe": "5m", "price": 520.0, "stop_loss": 515.0},
        )

        trap_results = {
            "is_false_breakout_trap": DecisionResult(
                "is_false_breakout_trap", DecisionType.NOUL, "true", None, 0.88, {}
            ),
            "is_dead_money_risk": DecisionResult(
                "is_dead_money_risk", DecisionType.NOUL, "false", None, 0.15, {}
            ),
            "is_volume_absorption_genuine": DecisionResult(
                "is_volume_absorption_genuine", DecisionType.NOUL, "false", None, 0.20, {}
            ),
        }
        client_trap = self._create_mock_client(trap_results)
        guard = DecisionVetoGuard(client=client_trap, ledger=self.ledger, false_breakout_threshold=0.55)
        approved, reason, prob = guard.evaluate_signal(signal, bar)
        self.assertFalse(approved)
        self.assertEqual(reason, "VETO_FALSE_BREAKOUT_TRAP")
        self.assertAlmostEqual(prob, 0.88, places=2)

        genuine_results = {
            "is_false_breakout_trap": DecisionResult(
                "is_false_breakout_trap", DecisionType.NOUL, "false", None, 0.10, {}
            ),
            "is_dead_money_risk": DecisionResult(
                "is_dead_money_risk", DecisionType.NOUL, "false", None, 0.12, {}
            ),
            "is_volume_absorption_genuine": DecisionResult(
                "is_volume_absorption_genuine", DecisionType.NOUL, "true", None, 0.92, {}
            ),
        }
        client_genuine = self._create_mock_client(genuine_results)
        guard_genuine = DecisionVetoGuard(client=client_genuine, ledger=self.ledger)
        approved_gen, reason_gen, prob_gen = guard_genuine.evaluate_signal(signal, bar)
        self.assertTrue(approved_gen)
        self.assertEqual(reason_gen, "APPROVED")
        self.assertAlmostEqual(prob_gen, 0.92, places=2)

    # Cluster 3: Universe Alpha Crown Winner-Takes-All
    def test_cluster_3_universe_alpha_crown_winner_selection(self) -> None:
        candidates = [
            TriggerSignal(
                timestamp=pd.Timestamp("2026-03-01 09:30:00"),
                symbol="BANKBEES",
                action=Action.BUY,
                conviction=0.85,
                trigger_name="Drift",
                metadata={"timeframe": "5m"},
            ),
            TriggerSignal(
                timestamp=pd.Timestamp("2026-03-01 09:30:00"),
                symbol="ITBEES",
                action=Action.BUY,
                conviction=0.80,
                trigger_name="Drift",
                metadata={"timeframe": "5m"},
            ),
            TriggerSignal(
                timestamp=pd.Timestamp("2026-03-01 09:30:00"),
                symbol="AUTOBEES",
                action=Action.BUY,
                conviction=0.78,
                trigger_name="Drift",
                metadata={"timeframe": "5m"},
            ),
        ]
        results = {
            "best_etf_to_buy_now": DecisionResult(
                "best_etf_to_buy_now", DecisionType.CHOICE, "ITBEES", None, 0.82, {}
            ),
            "capital_allocation_fraction": DecisionResult(
                "capital_allocation_fraction", DecisionType.SCORE, "3", 3, 0.89, {}
            ),
        }
        client = self._create_mock_client(results)
        crown = UniverseAlphaCrown(client=client, ledger=self.ledger)
        bars = {
            "BANKBEES": pd.Series({"Close": 520.0, "Volume": 100000}),
            "ITBEES": pd.Series({"Close": 430.0, "Volume": 100000}),
            "AUTOBEES": pd.Series({"Close": 210.0, "Volume": 100000}),
        }
        winner, fraction = crown.select_top_allocation(candidates, current_bars=bars)

        self.assertIsNotNone(winner)
        self.assertEqual(winner.symbol, "ITBEES")
        self.assertAlmostEqual(fraction, 0.75, places=2)

    # Cluster 4: Elastic String 1-Minute Live Basis Stat-Arb
    def test_cluster_4_elastic_string_basis_tension(self) -> None:
        cfg = CORE_ELASTIC_PAIRS[0]
        self.assertEqual(cfg.etf_symbol, "BANKBEES")
        self.assertEqual(cfg.anchor_symbol, "^NSEBANK")

        idx = pd.date_range("2026-03-01 09:15", periods=35, freq="1min")
        anchor = pd.Series([50000.0] * 35, index=idx)
        etf = pd.Series([500.0 + (0.1 if i % 2 == 0 else -0.1) for i in range(35)], index=idx)
        etf.iloc[32:] = 480.0  # Sharp undervalued dislocation

        spread, z_score = compute_basis_tension(etf, anchor, lookback=20, beta=1.0)
        self.assertEqual(len(spread), 35)
        self.assertLess(float(z_score.min()), -1.8)

        signals = generate_signals(
            etf_df=pd.DataFrame({"Close": etf}, index=idx),
            anchor_df=pd.DataFrame({"Close": anchor}, index=idx),
            pair_config=cfg,
        )
        self.assertGreater(len(signals), 0)
        self.assertEqual(signals[0].action, Action.BUY)
        self.assertEqual(signals[0].symbol, "BANKBEES")

    # Cluster 5: Dynamic Volatility Adaptive Stop Loss
    def test_cluster_5_dynamic_volatility_adaptive_stop(self) -> None:
        bar = pd.Series({"Close": 520.0, "Volume": 100000})
        # Score 0 = Tight
        c0 = self._create_mock_client({
            "volatility_regime": DecisionResult("volatility_regime", DecisionType.SCORE, "0", 0, 0.90, {})
        })
        shield0 = DecisionRiskShield(client=c0, ledger=self.ledger)
        stop0 = shield0.compute_adaptive_stop("BANKBEES", 520.0, 510.0, bar)
        self.assertAlmostEqual(stop0, 515.0, places=2)

        # Score 3 = Shock flatten
        c3 = self._create_mock_client({
            "volatility_regime": DecisionResult("volatility_regime", DecisionType.SCORE, "3", 3, 0.95, {})
        })
        shield3 = DecisionRiskShield(client=c3, ledger=self.ledger)
        stop3 = shield3.compute_adaptive_stop("BANKBEES", 520.0, 510.0, bar)
        self.assertAlmostEqual(stop3, 520.0, places=2)

    # Cluster 6: Pre-15:00 Broker MIS Square-off Dump Shield
    def test_cluster_6_pre_dump_broker_shield(self) -> None:
        ts = pd.Timestamp("2026-03-01 14:40:00")
        positions = [{"ticker": "BANKBEES", "qty": 100, "entry_price": 520.0}]
        bars = {"BANKBEES": pd.Series({"Close": 522.0, "Volume": 60000})}

        dump_results = {
            "retail_mis_dump_intensity": DecisionResult(
                "retail_mis_dump_intensity", DecisionType.SCORE, "3", 3, 0.92, {}
            ),
            "should_force_exit_at_1440": DecisionResult(
                "should_force_exit_at_1440", DecisionType.NOUL, "true", None, 0.88, {}
            ),
        }
        client_dump = self._create_mock_client(dump_results)
        shield = DecisionRiskShield(client=client_dump, ledger=self.ledger)
        exits = shield.evaluate_pre_dump_exit(ts, positions, bars)
        self.assertEqual(len(exits), 1)
        self.assertEqual(exits[0], ("BANKBEES", "PRE_DUMP_SHIELD_EXIT"))

    # Cluster 7: Morning 09:15 Continuation Gate
    def test_cluster_7_morning_continuation_gate_saves_dp_fee(self) -> None:
        pos = {"ticker": "GOLDBEES", "entry_price": 60.0, "stop_loss": 58.5, "trailing_stop_loss": 59.0}
        today_bar = pd.Series({"Open": 61.0, "High": 61.5, "Low": 60.8, "Close": 61.2})

        continuation_results = {
            "should_hold_through_0915_open": DecisionResult(
                "should_hold_through_0915_open", DecisionType.NOUL, "true", None, 0.85, {}
            ),
            "morning_gap_fill_risk": DecisionResult(
                "morning_gap_fill_risk", DecisionType.NOUL, "false", None, 0.15, {}
            ),
        }
        client = self._create_mock_client(continuation_results)
        shield = DecisionRiskShield(client=client, ledger=self.ledger)
        hold, ratcheted_stop, reason = shield.evaluate_overnight_continuation("GOLDBEES", pos, today_bar)

        self.assertTrue(hold)
        self.assertEqual(reason, "CONTINUATION_HOLD_DP_SAVED")
        self.assertGreaterEqual(ratcheted_stop, 60.0)

    # Cluster 8: SQLite State Ledger ACID Audit Trail Persistence
    def test_cluster_8_sqlite_state_ledger_audit_persistence(self) -> None:
        rec = DecisionAuditRecord(
            decision_id="test_audit_rec_001",
            timestamp="2026-03-01 10:00:00",
            timeframe="5m",
            cluster_id="veto_guard",
            symbol="BANKBEES",
            question_key="is_false_breakout_trap",
            selected_choice="false",
            probability_score=0.12,
            latency_ms=15.4,
            status="APPLIED",
        )
        self.ledger.record_decision(rec)
        rows = self.ledger.get_recent_decisions(limit=10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["decision_id"], "test_audit_rec_001")
        self.assertEqual(rows[0]["symbol"], "BANKBEES")
        self.assertAlmostEqual(rows[0]["probability_score"], 0.12, places=2)


if __name__ == "__main__":
    unittest.main()
