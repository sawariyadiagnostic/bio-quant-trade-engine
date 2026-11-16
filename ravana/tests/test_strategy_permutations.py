"""Strategy Permutation Matrix QA Suite.

Validates the 5 architectural strategy combinations:
- Permutation A: Pure Deterministic Baseline (5 Core Triggers alone)
- Permutation B: Deterministic Triggers + Elastic String Stat-Arb
- Permutation C: Triggers + Gemini Veto Guard (Trap & Dead-Money filter)
- Permutation D: Triggers + Gemini Universe Alpha Crown (Winner-takes-all)
- Permutation E: Full Stack FullMax (Triggers + Elastic + Veto + Crown + Risk Shield)
"""

import os
import shutil
import tempfile
import unittest
from typing import Any, Dict, List
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from ravana.decisions.client import GeminiDecisionClient
from ravana.decisions.schema import BatchResponse, DecisionResult, DecisionType, Question
from ravana.decisions.universe_allocator import UniverseAlphaCrown
from ravana.decisions.veto_guard import DecisionVetoGuard
from ravana.risk.decision_shield import DecisionRiskShield
from ravana.strategies.base import Action, TriggerSignal
from ravana.strategies.elastic_string import (
    CORE_ELASTIC_PAIRS,
    ElasticPairConfig,
    compute_basis_tension,
    generate_signals as generate_elastic_signals,
)
from ravana.strategies.state_ledger import SQLiteStateLedger
from ravana.strategies.two_way_orb import TwoWayMorningORBTrigger


class TestStrategyPermutationsSuite(unittest.TestCase):
    """Rigorous verification of all 5 strategy permutations."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "perm_ledger.db")
        self.ledger = SQLiteStateLedger(db_path=self.db_path)

        # Baseline sample bar data for testing
        self.idx = pd.date_range("2026-03-02 09:15", periods=40, freq="5min")
        self.bar = pd.Series({
            "Open": 520.0,
            "High": 525.0,
            "Low": 519.0,
            "Close": 524.0,
            "Volume": 350000,
            "VWAP": 521.0,
            "ATR_14": 3.8,
        })

    def tearDown(self) -> None:
        if self.ledger is not None:
            self.ledger.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _create_mock_client(self, results: Dict[str, DecisionResult]) -> MagicMock:
        client = MagicMock(spec=GeminiDecisionClient)
        batch = BatchResponse(results=results, latency_ms=8.5, status="OK")
        client.evaluate.return_value = batch
        return client

    # Permutation A: Pure Deterministic Baseline
    def test_permutation_a_pure_deterministic(self) -> None:
        """Permutation A runs deterministic triggers with zero LLM dependence."""
        trigger = TwoWayMorningORBTrigger()
        df = pd.DataFrame({
            "Open": [500.0] * 20,
            "High": [502.0] * 20,
            "Low": [499.0] * 20,
            "Close": [501.0] * 20,
            "Volume": [100000] * 20,
            "VWAP": [500.5] * 20,
        }, index=self.idx[:20])

        # Breakout bar on bar 10
        df.loc[self.idx[9], "Close"] = 508.0
        df.loc[self.idx[9], "High"] = 509.0

        signals = trigger.generate_two_way_signals(df, "BANKBEES")
        self.assertIsInstance(signals, list)

    # Permutation B: Deterministic Triggers + Elastic String Stat-Arb
    def test_permutation_b_triggers_plus_elastic_string(self) -> None:
        """Permutation B fuses structural triggers with 1-min basis tension stat-arb."""
        cfg = CORE_ELASTIC_PAIRS[0]
        anchor = pd.Series([50000.0] * 35, index=self.idx[:35])
        etf = pd.Series([500.0] * 35, index=self.idx[:35])
        # Force basis stretching at bar 28
        etf.iloc[28:] = 482.0

        elastic_sigs = generate_elastic_signals(
            etf_df=pd.DataFrame({"Close": etf}, index=self.idx[:35]),
            anchor_df=pd.DataFrame({"Close": anchor}, index=self.idx[:35]),
            pair_config=cfg,
        )
        self.assertGreater(len(elastic_sigs), 0)
        self.assertEqual(elastic_sigs[0].symbol, "BANKBEES")
        self.assertIn("PULL_LONG", elastic_sigs[0].trigger_name)

    # Permutation C: Triggers + Gemini Veto Guard
    def test_permutation_c_triggers_plus_veto_guard(self) -> None:
        """Permutation C blocks false breakout traps using Veto Guard."""
        sig = TriggerSignal(
            timestamp=pd.Timestamp("2026-03-02 09:35"),
            symbol="ITBEES",
            action=Action.BUY,
            conviction=0.85,
            trigger_name="MorningORB",
            metadata={"timeframe": "5m", "entry_price": 430.0, "stop_loss": 425.0},
        )
        mock_results = {
            "is_false_breakout_trap": DecisionResult("is_false_breakout_trap", DecisionType.NOUL, "true", None, 0.82, {}),
            "is_dead_money_risk": DecisionResult("is_dead_money_risk", DecisionType.NOUL, "false", None, 0.10, {}),
            "is_volume_absorption_genuine": DecisionResult("is_volume_absorption_genuine", DecisionType.NOUL, "false", None, 0.25, {}),
        }
        guard = DecisionVetoGuard(client=self._create_mock_client(mock_results), ledger=self.ledger)
        approved, reason, prob = guard.evaluate_signal(sig, self.bar)
        self.assertFalse(approved)
        self.assertEqual(reason, "VETO_FALSE_BREAKOUT_TRAP")

    # Permutation D: Triggers + Gemini Universe Alpha Crown
    def test_permutation_d_triggers_plus_universe_crown(self) -> None:
        """Permutation D concentrates capital on the single strongest sector candidate."""
        signals = [
            TriggerSignal(
                timestamp=pd.Timestamp("2026-03-02 10:00"),
                symbol="BANKBEES",
                action=Action.BUY,
                conviction=0.75,
                trigger_name="PEAD",
                metadata={"timeframe": "5m", "entry_price": 520.0},
            ),
            TriggerSignal(
                timestamp=pd.Timestamp("2026-03-02 10:00"),
                symbol="SILVERBEES",
                action=Action.BUY,
                conviction=0.85,
                trigger_name="Spread",
                metadata={"timeframe": "5m", "entry_price": 95.0},
            ),
        ]
        results = {
            "best_etf_to_buy_now": DecisionResult("best_etf_to_buy_now", DecisionType.CHOICE, "SILVERBEES", None, 0.90, {}),
            "capital_allocation_fraction": DecisionResult("capital_allocation_fraction", DecisionType.SCORE, "4", 4, 0.94, {}),
        }
        crown = UniverseAlphaCrown(client=self._create_mock_client(results), ledger=self.ledger)
        bars = {
            "BANKBEES": self.bar,
            "SILVERBEES": self.bar,
        }
        winner, fraction = crown.select_top_allocation(signals, current_bars=bars)
        self.assertIsNotNone(winner)
        self.assertEqual(winner.symbol, "SILVERBEES")
        self.assertAlmostEqual(fraction, 1.0, places=2)

    # Permutation E: Full Stack FullMax Engine
    def test_permutation_e_full_stack_fullmax(self) -> None:
        """Permutation E verifies the full synchronized cascade end-to-end."""
        # 1. Candidate generation
        candidates = [
            TriggerSignal(
                timestamp=pd.Timestamp("2026-03-02 09:30"),
                symbol="BANKBEES",
                action=Action.BUY,
                conviction=0.82,
                trigger_name="ORB",
                metadata={"timeframe": "5m", "entry_price": 520.0, "stop_loss": 515.0},
            ),
            TriggerSignal(
                timestamp=pd.Timestamp("2026-03-02 09:30"),
                symbol="ITBEES",
                action=Action.BUY,
                conviction=0.80,
                trigger_name="ORB",
                metadata={"timeframe": "5m", "entry_price": 430.0, "stop_loss": 425.0},
            ),
        ]

        # 2. Alpha Crown Selection
        crown_res = {
            "best_etf_to_buy_now": DecisionResult("best_etf_to_buy_now", DecisionType.CHOICE, "BANKBEES", None, 0.88, {}),
            "capital_allocation_fraction": DecisionResult("capital_allocation_fraction", DecisionType.SCORE, "2", 2, 0.85, {}),
        }
        crown = UniverseAlphaCrown(client=self._create_mock_client(crown_res), ledger=self.ledger)
        bars = {
            "BANKBEES": self.bar,
            "ITBEES": self.bar,
        }
        winner, fraction = crown.select_top_allocation(candidates, current_bars=bars)
        self.assertIsNotNone(winner)
        self.assertEqual(winner.symbol, "BANKBEES")
        self.assertAlmostEqual(fraction, 0.50, places=2)

        # 3. Veto Guard Verification
        veto_res = {
            "is_false_breakout_trap": DecisionResult("is_false_breakout_trap", DecisionType.NOUL, "false", None, 0.12, {}),
            "is_dead_money_risk": DecisionResult("is_dead_money_risk", DecisionType.NOUL, "false", None, 0.18, {}),
            "is_volume_absorption_genuine": DecisionResult("is_volume_absorption_genuine", DecisionType.NOUL, "true", None, 0.91, {}),
        }
        guard = DecisionVetoGuard(client=self._create_mock_client(veto_res), ledger=self.ledger)
        approved, reason, _ = guard.evaluate_signal(winner, self.bar)
        self.assertTrue(approved)

        # 4. Adaptive Stop Computation
        vol_res = {"volatility_regime": DecisionResult("volatility_regime", DecisionType.SCORE, "1", 1, 0.85, {})}
        shield = DecisionRiskShield(client=self._create_mock_client(vol_res), ledger=self.ledger)
        entry_px = float(winner.metadata.get("entry_price", 520.0))
        def_stop = float(winner.metadata.get("stop_loss", 515.0))
        stop = shield.compute_adaptive_stop(winner.symbol, entry_px, def_stop, self.bar)
        self.assertEqual(stop, def_stop)

        # 5. Pre-Dump Shield Verification
        ts_dump = pd.Timestamp("2026-03-02 14:40")
        dump_res = {
            "retail_mis_dump_intensity": DecisionResult("retail_mis_dump_intensity", DecisionType.SCORE, "3", 3, 0.90, {}),
            "should_force_exit_at_1440": DecisionResult("should_force_exit_at_1440", DecisionType.NOUL, "true", None, 0.86, {}),
        }
        shield_dump = DecisionRiskShield(client=self._create_mock_client(dump_res), ledger=self.ledger)
        exits = shield_dump.evaluate_pre_dump_exit(ts_dump, [{"ticker": "BANKBEES"}], {"BANKBEES": self.bar})
        self.assertEqual(len(exits), 1)
        self.assertEqual(exits[0], ("BANKBEES", "PRE_DUMP_SHIELD_EXIT"))


if __name__ == "__main__":
    unittest.main()
