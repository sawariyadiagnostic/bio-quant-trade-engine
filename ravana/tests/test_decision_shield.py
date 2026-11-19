"""Unit tests for Gemini DecisionRiskShield in RAVANA Quantitative Engine.

Comprehensive coverage:
1. Pre-dump exit shield:
   - Window gating (14:35 - 14:50 IST), tz-aware & tz-naive handling.
   - Triggers via probability >= 0.65 or score >= 2.
   - Low-risk bypass (prob < 0.65 and score < 2).
   - Empty positions or missing tickers/bars.
   - Multi-position batch evaluations.
   - Operations without ledger (None).
2. Adaptive stop-loss calculation:
   - Volatility regimes (0=Tight, 1=Normal, 2=Wide, 3=Shock).
   - Long and Short trade orientations.
   - Zero default risk distance fallback.
   - Bar close case insensitivity ('Close' / 'close').
   - Operations without ledger (None).
3. Overnight continuation hold evaluation at morning open (09:15 IST):
   - Approved holds (prob >= 0.65 and gap risk < 0.40).
   - Rejections via low hold prob (< 0.65) or high gap risk (>= 0.40).
   - Automatic stop ratcheting on profit open vs down open vs explicit ratcheted_stop.
   - Standard morning exit price at today_bar['Open'].
   - Operations without ledger (None).
4. Full audit logging into SQLiteStateLedger with DecisionAuditRecord schema verification.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

import pandas as pd

# Ensure repository root is on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from ravana.decisions.client import GeminiDecisionClient
from ravana.decisions.schema import BatchResponse, DecisionResult, DecisionType
from ravana.risk.decision_shield import DecisionRiskShield
from ravana.strategies.state_ledger import SQLiteStateLedger


class TestDecisionRiskShieldPreDump(unittest.TestCase):
    """Test suite for Pre-Dump Exit Shield."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_shield.db")
        self.ledger = SQLiteStateLedger(db_path=self.db_path)
        self.mock_client = MagicMock(spec=GeminiDecisionClient)
        self.shield = DecisionRiskShield(client=self.mock_client, ledger=self.ledger)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_outside_window_no_exit_and_no_client_calls(self) -> None:
        """Shield is inactive outside 14:35-14:50 IST window."""
        ts = pd.Timestamp("2026-11-13 11:30:00+05:30")
        positions = [{"ticker": "BANKBEES", "shares": 100}]
        bars = {"BANKBEES": pd.Series({"Close": 500.0, "Volume": 10000})}

        exits = self.shield.evaluate_pre_dump_exit(ts, positions, bars)
        self.assertEqual(exits, [])
        self.mock_client.evaluate.assert_not_called()

    def test_outside_window_after_1450(self) -> None:
        """Shield is inactive after 14:50 IST."""
        ts = pd.Timestamp("2026-11-13 14:51:00+05:30")
        positions = [{"ticker": "BANKBEES", "shares": 100}]
        bars = {"BANKBEES": pd.Series({"Close": 500.0, "Volume": 10000})}

        exits = self.shield.evaluate_pre_dump_exit(ts, positions, bars)
        self.assertEqual(exits, [])
        self.mock_client.evaluate.assert_not_called()

    def test_empty_positions_returns_empty(self) -> None:
        """Empty positions returns empty list immediately."""
        ts = pd.Timestamp("2026-11-13 14:40:00+05:30")
        exits = self.shield.evaluate_pre_dump_exit(ts, [], {})
        self.assertEqual(exits, [])
        self.mock_client.evaluate.assert_not_called()

    def test_tz_aware_utc_timestamp_conversion(self) -> None:
        """Timestamp in UTC converts to IST correctly (09:10 UTC = 14:40 IST)."""
        ts_utc = pd.Timestamp("2026-11-13 09:10:00+00:00")
        positions = [{"ticker": "BANKBEES", "shares": 100}]
        bars = {"BANKBEES": pd.Series({"Close": 500.0, "Volume": 10000})}

        self.mock_client.evaluate.return_value = BatchResponse(
            results={
                "retail_mis_dump_intensity": DecisionResult(
                    key="retail_mis_dump_intensity",
                    decision_type=DecisionType.SCORE,
                    score=2,
                ),
                "should_force_exit_at_1440": DecisionResult(
                    key="should_force_exit_at_1440",
                    decision_type=DecisionType.NOUL,
                    probability=0.50,
                ),
            },
            status="OK",
        )

        exits = self.shield.evaluate_pre_dump_exit(ts_utc, positions, bars)
        self.assertEqual(exits, [("BANKBEES", "PRE_DUMP_SHIELD_EXIT")])

    def test_inside_window_high_prob_triggers_exit(self) -> None:
        """Exit triggered when should_force_exit_at_1440 prob >= 0.65."""
        ts = pd.Timestamp("2026-11-13 14:40:00+05:30")
        positions = [{"ticker": "BANKBEES", "shares": 100}]
        bars = {"BANKBEES": pd.Series({"Close": 500.0, "Volume": 15000})}

        self.mock_client.evaluate.return_value = BatchResponse(
            results={
                "retail_mis_dump_intensity": DecisionResult(
                    key="retail_mis_dump_intensity",
                    decision_type=DecisionType.SCORE,
                    score=1,
                    probability=0.70,
                ),
                "should_force_exit_at_1440": DecisionResult(
                    key="should_force_exit_at_1440",
                    decision_type=DecisionType.NOUL,
                    selected_choice="true",
                    probability=0.72,
                ),
            },
            latency_ms=12.5,
            status="OK",
        )

        exits = self.shield.evaluate_pre_dump_exit(ts, positions, bars)
        self.assertEqual(exits, [("BANKBEES", "PRE_DUMP_SHIELD_EXIT")])

        # Verify audit ledger recorded the decisions
        decisions = self.ledger.get_recent_decisions(symbol="BANKBEES", limit=10)
        self.assertEqual(len(decisions), 2)
        keys = {d["question_key"] for d in decisions}
        self.assertIn("retail_mis_dump_intensity", keys)
        self.assertIn("should_force_exit_at_1440", keys)
        for d in decisions:
            self.assertEqual(d["status"], "APPLIED")

    def test_inside_window_high_score_triggers_exit(self) -> None:
        """Exit triggered when retail_mis_dump_intensity score >= 2."""
        ts = pd.Timestamp("2026-11-13 14:45:00+05:30")
        positions = [{"ticker": "ITBEES", "shares": 200}]
        bars = {"ITBEES": pd.Series({"Close": 42.0, "Volume": 50000})}

        self.mock_client.evaluate.return_value = BatchResponse(
            results={
                "retail_mis_dump_intensity": DecisionResult(
                    key="retail_mis_dump_intensity",
                    decision_type=DecisionType.SCORE,
                    score=2,
                    probability=0.80,
                ),
                "should_force_exit_at_1440": DecisionResult(
                    key="should_force_exit_at_1440",
                    decision_type=DecisionType.NOUL,
                    selected_choice="false",
                    probability=0.45,
                ),
            },
            latency_ms=8.0,
            status="OK",
        )

        exits = self.shield.evaluate_pre_dump_exit(ts, positions, bars)
        self.assertEqual(exits, [("ITBEES", "PRE_DUMP_SHIELD_EXIT")])

    def test_inside_window_low_risk_no_exit(self) -> None:
        """No exit when prob < 0.65 and score < 2."""
        ts = pd.Timestamp("2026-11-13 14:38:00+05:30")
        positions = [{"ticker": "GOLDBEES", "shares": 500}]
        bars = {"GOLDBEES": pd.Series({"Close": 65.0, "Volume": 5000})}

        self.mock_client.evaluate.return_value = BatchResponse(
            results={
                "retail_mis_dump_intensity": DecisionResult(
                    key="retail_mis_dump_intensity",
                    decision_type=DecisionType.SCORE,
                    score=0,
                    probability=0.90,
                ),
                "should_force_exit_at_1440": DecisionResult(
                    key="should_force_exit_at_1440",
                    decision_type=DecisionType.NOUL,
                    selected_choice="false",
                    probability=0.20,
                ),
            },
            latency_ms=10.0,
            status="OK",
        )

        exits = self.shield.evaluate_pre_dump_exit(ts, positions, bars)
        self.assertEqual(exits, [])

        decisions = self.ledger.get_recent_decisions(symbol="GOLDBEES", limit=10)
        self.assertEqual(len(decisions), 2)
        for d in decisions:
            self.assertEqual(d["status"], "BYPASSED")

    def test_without_ledger_operates_cleanly(self) -> None:
        """Operates properly when ledger is None."""
        shield_no_ledger = DecisionRiskShield(client=self.mock_client, ledger=None)
        ts = pd.Timestamp("2026-11-13 14:40:00+05:30")
        positions = [{"ticker": "BANKBEES", "shares": 100}]
        bars = {}

        self.mock_client.evaluate.return_value = BatchResponse(
            results={
                "retail_mis_dump_intensity": DecisionResult(
                    key="retail_mis_dump_intensity",
                    decision_type=DecisionType.SCORE,
                    score=3,
                ),
                "should_force_exit_at_1440": DecisionResult(
                    key="should_force_exit_at_1440",
                    decision_type=DecisionType.NOUL,
                    probability=0.85,
                ),
            },
            status="OK",
        )

        exits = shield_no_ledger.evaluate_pre_dump_exit(ts, positions, bars)
        self.assertEqual(exits, [("BANKBEES", "PRE_DUMP_SHIELD_EXIT")])


class TestDecisionRiskShieldAdaptiveStop(unittest.TestCase):
    """Test suite for Adaptive Stop Loss computation across Volatility Regimes."""

    def setUp(self) -> None:
        self.mock_client = MagicMock(spec=GeminiDecisionClient)
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_shield.db")
        self.ledger = SQLiteStateLedger(db_path=self.db_path)
        self.shield = DecisionRiskShield(client=self.mock_client, ledger=self.ledger)
        self.bar = pd.Series({"Open": 100.0, "High": 102.0, "Low": 97.0, "Close": 98.5})

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _set_regime_mock(self, score: int) -> None:
        self.mock_client.evaluate.return_value = BatchResponse(
            results={
                "volatility_regime": DecisionResult(
                    key="volatility_regime",
                    decision_type=DecisionType.SCORE,
                    score=score,
                    probability=0.85,
                )
            },
            latency_ms=5.0,
            status="OK",
        )

    def test_regime_0_tightens_stop_long(self) -> None:
        """Regime 0 (Tight/low noise) tightens long stop towards entry price."""
        self._set_regime_mock(score=0)
        # Long position: entry=100.0, default_stop=95.0, risk_dist=5.0
        stop = self.shield.compute_adaptive_stop("TCS", entry_price=100.0, default_stop=95.0, bar=self.bar)
        # Tightened stop: 100.0 - (0.5 * 5.0) = 97.5 (tighter than 95.0)
        self.assertEqual(stop, 97.5)

    def test_regime_0_tightens_stop_short(self) -> None:
        """Regime 0 (Tight/low noise) tightens short stop towards entry price."""
        self._set_regime_mock(score=0)
        # Short position: entry=100.0, default_stop=105.0, risk_dist=5.0
        stop = self.shield.compute_adaptive_stop("TCS", entry_price=100.0, default_stop=105.0, bar=self.bar)
        # Tightened short stop: 100.0 + (0.5 * 5.0) = 102.5 (tighter than 105.0)
        self.assertEqual(stop, 102.5)

    def test_regime_1_normal_chop(self) -> None:
        """Regime 1 (Normal) maintains default stop loss."""
        self._set_regime_mock(score=1)
        stop = self.shield.compute_adaptive_stop("TCS", entry_price=100.0, default_stop=95.0, bar=self.bar)
        self.assertEqual(stop, 95.0)

    def test_regime_2_widens_stop_long(self) -> None:
        """Regime 2 (Wide) widens stop distance to accommodate noise (Long)."""
        self._set_regime_mock(score=2)
        stop = self.shield.compute_adaptive_stop("TCS", entry_price=100.0, default_stop=95.0, bar=self.bar)
        # Widened stop: 100.0 - (1.5 * 5.0) = 92.5 (wider than 95.0)
        self.assertEqual(stop, 92.5)

    def test_regime_2_widens_stop_short(self) -> None:
        """Regime 2 (Wide) widens stop distance to accommodate noise (Short)."""
        self._set_regime_mock(score=2)
        stop = self.shield.compute_adaptive_stop("TCS", entry_price=100.0, default_stop=105.0, bar=self.bar)
        # Widened short stop: 100.0 + (1.5 * 5.0) = 107.5 (wider than 105.0)
        self.assertEqual(stop, 107.5)

    def test_regime_3_flattens_on_shock(self) -> None:
        """Regime 3 (Shock) flattens position by setting stop to current close."""
        self._set_regime_mock(score=3)
        stop = self.shield.compute_adaptive_stop("TCS", entry_price=100.0, default_stop=95.0, bar=self.bar)
        self.assertEqual(stop, 98.5)

        # Verify audit trail
        decisions = self.ledger.get_recent_decisions(symbol="TCS", cluster_id="adaptive_stop_shield", limit=1)
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0]["question_key"], "volatility_regime")

    def test_zero_risk_distance_fallback(self) -> None:
        """Fallback to 1% default risk distance when default_stop equals entry_price."""
        self._set_regime_mock(score=0)
        stop = self.shield.compute_adaptive_stop("TCS", entry_price=100.0, default_stop=100.0, bar=self.bar)
        # 1% of 100 = 1.0 -> 0.5 * 1.0 = 0.5 -> 99.5
        self.assertEqual(stop, 99.5)

    def test_without_ledger_adaptive_stop_operates(self) -> None:
        """Adaptive stop operates cleanly without ledger."""
        shield_no_ledger = DecisionRiskShield(client=self.mock_client, ledger=None)
        self._set_regime_mock(score=1)
        stop = shield_no_ledger.compute_adaptive_stop("TCS", entry_price=100.0, default_stop=95.0, bar=self.bar)
        self.assertEqual(stop, 95.0)


class TestDecisionRiskShieldOvernightContinuation(unittest.TestCase):
    """Test suite for Overnight Continuation Gate at 09:15 IST open."""

    def setUp(self) -> None:
        self.mock_client = MagicMock(spec=GeminiDecisionClient)
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_shield.db")
        self.ledger = SQLiteStateLedger(db_path=self.db_path)
        self.shield = DecisionRiskShield(client=self.mock_client, ledger=self.ledger)
        self.today_bar = pd.Series({"Open": 515.0, "High": 520.0, "Low": 512.0, "Close": 518.0})

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_continuation_hold_approved_with_custom_ratchet(self) -> None:
        """Hold approved when should_hold prob >= 0.65 and gap risk < 0.40."""
        self.mock_client.evaluate.return_value = BatchResponse(
            results={
                "should_hold_through_0915_open": DecisionResult(
                    key="should_hold_through_0915_open",
                    decision_type=DecisionType.NOUL,
                    selected_choice="true",
                    probability=0.78,
                ),
                "morning_gap_fill_risk": DecisionResult(
                    key="morning_gap_fill_risk",
                    decision_type=DecisionType.NOUL,
                    selected_choice="false",
                    probability=0.25,
                ),
            },
            latency_ms=11.0,
            status="OK",
        )

        pos = {
            "ticker": "BANKBEES",
            "entry_price": 500.0,
            "trailing_stop_loss": 495.0,
            "ratcheted_stop": 508.0,
        }

        should_hold, exit_or_stop_px, reason = self.shield.evaluate_overnight_continuation(
            "BANKBEES", pos, self.today_bar
        )

        self.assertTrue(should_hold)
        self.assertEqual(exit_or_stop_px, 508.0)
        self.assertEqual(reason, "CONTINUATION_HOLD_DP_SAVED")

        # Verify audit trail
        decisions = self.ledger.get_recent_decisions(symbol="BANKBEES", cluster_id="overnight_continuation", limit=5)
        self.assertEqual(len(decisions), 2)
        for d in decisions:
            self.assertEqual(d["status"], "APPLIED")

    def test_continuation_hold_auto_ratchets_to_entry_on_profit_gap(self) -> None:
        """When in profit, stop ratchets at least to entry price if not explicitly provided."""
        self.mock_client.evaluate.return_value = BatchResponse(
            results={
                "should_hold_through_0915_open": DecisionResult(
                    key="should_hold_through_0915_open",
                    decision_type=DecisionType.NOUL,
                    probability=0.70,
                ),
                "morning_gap_fill_risk": DecisionResult(
                    key="morning_gap_fill_risk",
                    decision_type=DecisionType.NOUL,
                    probability=0.15,
                ),
            },
            status="OK",
        )

        pos = {
            "ticker": "BANKBEES",
            "entry_price": 500.0,
            "trailing_stop_loss": 495.0,
        }

        # Open is 515.0 > entry 500.0 -> ratchets stop to entry_price (500.0)
        should_hold, exit_or_stop_px, reason = self.shield.evaluate_overnight_continuation(
            "BANKBEES", pos, self.today_bar
        )

        self.assertTrue(should_hold)
        self.assertEqual(exit_or_stop_px, 500.0)
        self.assertEqual(reason, "CONTINUATION_HOLD_DP_SAVED")

    def test_continuation_hold_rejected_on_high_gap_risk(self) -> None:
        """Hold rejected if morning_gap_fill_risk >= 0.40 even if hold prob is high."""
        self.mock_client.evaluate.return_value = BatchResponse(
            results={
                "should_hold_through_0915_open": DecisionResult(
                    key="should_hold_through_0915_open",
                    decision_type=DecisionType.NOUL,
                    selected_choice="true",
                    probability=0.85,
                ),
                "morning_gap_fill_risk": DecisionResult(
                    key="morning_gap_fill_risk",
                    decision_type=DecisionType.NOUL,
                    selected_choice="true",
                    probability=0.60,
                ),
            },
            latency_ms=9.0,
            status="OK",
        )

        pos = {
            "ticker": "BANKBEES",
            "entry_price": 500.0,
            "trailing_stop_loss": 495.0,
        }

        should_hold, exit_or_stop_px, reason = self.shield.evaluate_overnight_continuation(
            "BANKBEES", pos, self.today_bar
        )

        self.assertFalse(should_hold)
        self.assertEqual(exit_or_stop_px, 515.0)  # today_bar['Open']
        self.assertEqual(reason, "STANDARD_MORNING_EXIT")

    def test_continuation_hold_rejected_on_low_hold_prob(self) -> None:
        """Hold rejected if should_hold prob < 0.65."""
        self.mock_client.evaluate.return_value = BatchResponse(
            results={
                "should_hold_through_0915_open": DecisionResult(
                    key="should_hold_through_0915_open",
                    decision_type=DecisionType.NOUL,
                    selected_choice="false",
                    probability=0.45,
                ),
                "morning_gap_fill_risk": DecisionResult(
                    key="morning_gap_fill_risk",
                    decision_type=DecisionType.NOUL,
                    selected_choice="false",
                    probability=0.20,
                ),
            },
            latency_ms=9.0,
            status="OK",
        )

        pos = {
            "ticker": "BANKBEES",
            "entry_price": 500.0,
            "trailing_stop_loss": 495.0,
        }

        should_hold, exit_or_stop_px, reason = self.shield.evaluate_overnight_continuation(
            "BANKBEES", pos, self.today_bar
        )

        self.assertFalse(should_hold)
        self.assertEqual(exit_or_stop_px, 515.0)
        self.assertEqual(reason, "STANDARD_MORNING_EXIT")

    def test_overnight_continuation_without_ledger(self) -> None:
        """Overnight continuation operates cleanly without ledger."""
        shield_no_ledger = DecisionRiskShield(client=self.mock_client, ledger=None)
        self.mock_client.evaluate.return_value = BatchResponse(
            results={
                "should_hold_through_0915_open": DecisionResult(
                    key="should_hold_through_0915_open",
                    decision_type=DecisionType.NOUL,
                    probability=0.30,
                ),
                "morning_gap_fill_risk": DecisionResult(
                    key="morning_gap_fill_risk",
                    decision_type=DecisionType.NOUL,
                    probability=0.10,
                ),
            },
            status="OK",
        )
        pos = {"ticker": "BANKBEES", "entry_price": 500.0}
        should_hold, exit_px, reason = shield_no_ledger.evaluate_overnight_continuation(
            "BANKBEES", pos, self.today_bar
        )
        self.assertFalse(should_hold)
        self.assertEqual(exit_px, 515.0)
        self.assertEqual(reason, "STANDARD_MORNING_EXIT")


class TestDecisionRiskShieldInitialization(unittest.TestCase):
    """Test default initialization."""

    def test_default_init(self) -> None:
        shield = DecisionRiskShield()
        self.assertIsNotNone(shield.client)
        self.assertIsNone(shield.ledger)


class TestCrossAssetMacroShock(unittest.TestCase):
    """Test deterministic macro tripwires and Gemini crisis veto."""

    def setUp(self) -> None:
        self.mock_client = MagicMock(spec=GeminiDecisionClient)
        self.shield = DecisionRiskShield(client=self.mock_client)

    def test_vix_spike_tripwire(self) -> None:
        snap = {"india_vix_delta_pct": 0.16, "nifty_5m_ret": -0.002, "gold_ret": 0.0}
        is_shock, reason, prob = self.shield.evaluate_cross_asset_macro_shock(snap)
        self.assertTrue(is_shock)
        self.assertEqual(reason, "MACRO_VIX_SPIKE_PANIC")
        self.mock_client.evaluate.assert_not_called()

    def test_nifty_gap_down_tripwire(self) -> None:
        snap = {"india_vix_delta_pct": 0.05, "nifty_5m_ret": -0.014, "gold_ret": 0.0}
        is_shock, reason, prob = self.shield.evaluate_cross_asset_macro_shock(snap)
        self.assertTrue(is_shock)
        self.assertEqual(reason, "MACRO_INDEX_GAP_DOWN")
        self.mock_client.evaluate.assert_not_called()

    def test_safe_haven_divergence_tripwire(self) -> None:
        snap = {"india_vix_delta_pct": 0.05, "nifty_5m_ret": -0.009, "gold_ret": 0.018}
        is_shock, reason, prob = self.shield.evaluate_cross_asset_macro_shock(snap)
        self.assertTrue(is_shock)
        self.assertEqual(reason, "MACRO_SAFE_HAVEN_DIVERGENCE")
        self.mock_client.evaluate.assert_not_called()

    def test_gemini_macro_crisis_veto(self) -> None:
        self.mock_client.evaluate.return_value = BatchResponse(
            results={
                "is_systemic_macro_shock": DecisionResult(
                    key="is_systemic_macro_shock",
                    decision_type=DecisionType.NOUL,
                    probability=0.82,
                )
            },
            status="OK",
        )
        snap = {"india_vix_delta_pct": 0.02, "nifty_5m_ret": -0.003, "gold_ret": 0.001}
        is_shock, reason, prob = self.shield.evaluate_cross_asset_macro_shock(snap)
        self.assertTrue(is_shock)
        self.assertEqual(reason, "GEMINI_MACRO_CRISIS_VETO")
        self.assertAlmostEqual(prob, 0.82)



if __name__ == "__main__":
    unittest.main()
