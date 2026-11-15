"""Unit test suite for Elastic String Statistical Arbitrage Engine.

Validates:
1. ElasticPairConfig data model and default parameters.
2. Pre-configured 5 Core Pairs.
3. Basis tension calculation (log spread, rolling mu, rolling sigma, rolling z-score).
4. Causality and zero forward lookahead.
5. Signal generation for long dislocations (PULL_LONG, ETF undervalued).
6. Signal generation for short dislocations (PULL_SHORT, ETF overvalued).
7. Conviction boundary and scaling (|Z| / 3.0 capped at 1.0).
8. Metadata completeness (spread_z, elastic_tension, entry_z, exit_z, stop_z).
9. Snap release logic (SNAP_EQUILIBRIUM and BROKEN_STRING_STOP).
10. Insufficient data edge cases.
"""
from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from ravana.strategies.base import TriggerAction, TriggerSignal
from ravana.strategies.elastic_string import (
    CORE_ELASTIC_PAIRS,
    CORE_ELASTIC_PAIRS_MAP,
    ElasticPairConfig,
    ElasticStringEngine,
    check_snap_release,
    compute_basis_tension,
)


class TestElasticString(unittest.TestCase):
    """Test suite for Elastic String engine and pair configurations."""

    def setUp(self) -> None:
        self.idx = pd.date_range("2026-09-01 09:15", periods=60, freq="5min")
        self.config = ElasticPairConfig(
            etf_symbol="BANKBEES",
            anchor_symbol="^NSEBANK",
            beta=1.0,
            lookback_bars=20,
            entry_z_threshold=1.8,
            exit_z_threshold=0.2,
            stop_z_threshold=3.2,
        )
        self.engine = ElasticStringEngine(self.config)

    def test_pair_config_and_core_pairs(self) -> None:
        """Verify ElasticPairConfig defaults and pre-configured 5 core pairs."""
        cfg = ElasticPairConfig(etf_symbol="TEST", anchor_symbol="ANCHOR")
        self.assertEqual(cfg.beta, 1.0)
        self.assertEqual(cfg.lookback_bars, 20)
        self.assertEqual(cfg.entry_z_threshold, 1.8)
        self.assertEqual(cfg.exit_z_threshold, 0.2)
        self.assertEqual(cfg.stop_z_threshold, 3.2)

        self.assertEqual(len(CORE_ELASTIC_PAIRS), 5)
        expected_pairs = [
            ("BANKBEES", "^NSEBANK"),
            ("PSUBNKBEES", "BANKBEES"),
            ("ITBEES", "^CNXIT"),
            ("SILVERBEES", "GOLDBEES"),
            ("CPSEETF", "BZ=F"),
        ]
        for i, (etf, anchor) in enumerate(expected_pairs):
            self.assertEqual(CORE_ELASTIC_PAIRS[i].etf_symbol, etf)
            self.assertEqual(CORE_ELASTIC_PAIRS[i].anchor_symbol, anchor)
            self.assertIn(etf, CORE_ELASTIC_PAIRS_MAP)

    def test_compute_basis_tension_formula(self) -> None:
        """Verify log spread calculation: log(etf) - beta * log(anchor)."""
        anchor_prices = pd.Series(np.linspace(100.0, 110.0, 60), index=self.idx)
        etf_prices = anchor_prices * 1.02  # ETF trades at constant 2% premium

        spread, z_score = compute_basis_tension(
            etf_prices, anchor_prices, lookback=20, beta=1.0
        )
        expected_spread = np.log(1.02)
        np.testing.assert_allclose(spread.values, expected_spread, rtol=1e-5)
        # Constant spread has 0 variance, z-score should be 0.0 without inf/nan
        self.assertTrue((z_score.fillna(0.0) == 0.0).all())

    def test_compute_basis_tension_beta_scaling(self) -> None:
        """Verify beta scaling: log(etf) - beta * log(anchor)."""
        anchor_prices = pd.Series([100.0] * 30, index=self.idx[:30])
        etf_prices = pd.Series([10.0] * 30, index=self.idx[:30])

        # When beta = 0.5: log(10) - 0.5 * log(100) = log(10) - log(10) = 0.0
        spread, _ = compute_basis_tension(
            etf_prices, anchor_prices, lookback=10, beta=0.5
        )
        np.testing.assert_allclose(spread.values, 0.0, atol=1e-6)

    def test_causality_zero_lookahead(self) -> None:
        """Verify rolling metrics never index future rows (strictly causal)."""
        np.random.seed(42)
        anchor_prices = pd.Series(100.0 + np.random.randn(50), index=self.idx[:50])
        etf_prices = anchor_prices * 1.01

        spread_1, z_1 = compute_basis_tension(
            etf_prices, anchor_prices, lookback=20, beta=1.0
        )

        # Mutate future bar at index 45
        etf_prices_mutated = etf_prices.copy()
        etf_prices_mutated.iloc[45] = 999.0

        spread_2, z_2 = compute_basis_tension(
            etf_prices_mutated, anchor_prices, lookback=20, beta=1.0
        )

        # Bars prior to index 45 must be IDENTICAL
        np.testing.assert_allclose(spread_1.iloc[:45], spread_2.iloc[:45])
        np.testing.assert_allclose(z_1.iloc[:45], z_2.iloc[:45])

    def test_signal_generation_pull_long(self) -> None:
        """Verify BUY signal (PULL_LONG) when ETF is undervalued (Z <= -entry_z)."""
        # Baseline anchor at 100
        anchor_prices = pd.Series([100.0] * 40, index=self.idx[:40])
        etf_prices = pd.Series([100.0] * 40, index=self.idx[:40])

        # Add minor fluctuation to establish historical volatility
        for i in range(25):
            etf_prices.iloc[i] = 100.0 + (0.1 if i % 2 == 0 else -0.1)

        # Sharp dislocation at bar 30: ETF drops to 96.0 (undervalued)
        etf_prices.iloc[30:] = 96.0

        etf_df = pd.DataFrame({"Close": etf_prices}, index=self.idx[:40])
        anchor_df = pd.DataFrame({"Close": anchor_prices}, index=self.idx[:40])

        signals = self.engine.generate_signals(etf_df, anchor_df, self.config)
        self.assertGreater(len(signals), 0)

        first_sig = signals[0]
        self.assertEqual(first_sig.action, TriggerAction.BUY)
        self.assertEqual(first_sig.symbol, "BANKBEES")
        self.assertGreaterEqual(first_sig.conviction, 0.60)
        self.assertLessEqual(first_sig.conviction, 1.0)

        # Check required metadata keys
        self.assertIn("spread_z", first_sig.metadata)
        self.assertIn("elastic_tension", first_sig.metadata)
        self.assertIn("entry_z", first_sig.metadata)
        self.assertIn("exit_z", first_sig.metadata)
        self.assertIn("stop_z", first_sig.metadata)
        self.assertLessEqual(first_sig.metadata["spread_z"], -self.config.entry_z_threshold)

    def test_signal_generation_pull_short(self) -> None:
        """Verify SELL signal (PULL_SHORT) when ETF is overvalued (Z >= +entry_z)."""
        anchor_prices = pd.Series([100.0] * 40, index=self.idx[:40])
        etf_prices = pd.Series([100.0] * 40, index=self.idx[:40])

        for i in range(25):
            etf_prices.iloc[i] = 100.0 + (0.1 if i % 2 == 0 else -0.1)

        # Sharp dislocation at bar 30: ETF surges to 104.0 (overvalued)
        etf_prices.iloc[30:] = 104.0

        etf_df = pd.DataFrame({"Close": etf_prices}, index=self.idx[:40])
        anchor_df = pd.DataFrame({"Close": anchor_prices}, index=self.idx[:40])

        signals = self.engine.generate_signals(etf_df, anchor_df, self.config)
        self.assertGreater(len(signals), 0)

        first_sig = signals[0]
        self.assertEqual(first_sig.action, TriggerAction.SELL)
        self.assertEqual(first_sig.symbol, "BANKBEES")
        self.assertGreaterEqual(first_sig.conviction, 0.60)
        self.assertLessEqual(first_sig.conviction, 1.0)
        self.assertGreaterEqual(first_sig.metadata["spread_z"], self.config.entry_z_threshold)

    def test_snap_release_equilibrium(self) -> None:
        """Verify SNAP_EQUILIBRIUM triggers when |current_z| <= exit_z."""
        # Inside equilibrium band [-0.2, 0.2]
        is_exit, reason = check_snap_release(current_z=0.1, entry_z=1.8, is_long=True)
        self.assertTrue(is_exit)
        self.assertEqual(reason, "SNAP_EQUILIBRIUM")

        is_exit, reason = check_snap_release(current_z=-0.15, entry_z=1.8, is_long=True)
        self.assertTrue(is_exit)
        self.assertEqual(reason, "SNAP_EQUILIBRIUM")

        is_exit, reason = check_snap_release(current_z=0.0, entry_z=1.8, is_long=False)
        self.assertTrue(is_exit)
        self.assertEqual(reason, "SNAP_EQUILIBRIUM")

        # Boundary checks
        is_exit, reason = check_snap_release(current_z=0.20, entry_z=1.8, is_long=False)
        self.assertTrue(is_exit)
        self.assertEqual(reason, "SNAP_EQUILIBRIUM")

        is_exit, reason = check_snap_release(current_z=-0.20, entry_z=1.8, is_long=True)
        self.assertTrue(is_exit)
        self.assertEqual(reason, "SNAP_EQUILIBRIUM")

        # Outside equilibrium band
        is_exit, reason = check_snap_release(current_z=0.35, entry_z=1.8, is_long=False)
        self.assertFalse(is_exit)
        self.assertEqual(reason, "")

    def test_snap_release_broken_string_stop(self) -> None:
        """Verify BROKEN_STRING_STOP when dislocation exceeds stop_z."""
        # Long position: stop when current_z <= -stop_z (-3.2)
        is_exit, reason = check_snap_release(current_z=-3.2, entry_z=1.8, is_long=True)
        self.assertTrue(is_exit)
        self.assertEqual(reason, "BROKEN_STRING_STOP")

        is_exit, reason = check_snap_release(current_z=-3.5, entry_z=1.8, is_long=True)
        self.assertTrue(is_exit)
        self.assertEqual(reason, "BROKEN_STRING_STOP")

        # Long position: holding at -1.5 (neither exit nor stop)
        is_exit, reason = check_snap_release(current_z=-1.5, entry_z=1.8, is_long=True)
        self.assertFalse(is_exit)
        self.assertEqual(reason, "")

        # Short position: stop when current_z >= stop_z (+3.2)
        is_exit, reason = check_snap_release(current_z=3.2, entry_z=1.8, is_long=False)
        self.assertTrue(is_exit)
        self.assertEqual(reason, "BROKEN_STRING_STOP")

        is_exit, reason = check_snap_release(current_z=3.6, entry_z=1.8, is_long=False)
        self.assertTrue(is_exit)
        self.assertEqual(reason, "BROKEN_STRING_STOP")

        # Short position: holding at +1.5
        is_exit, reason = check_snap_release(current_z=1.5, entry_z=1.8, is_long=False)
        self.assertFalse(is_exit)
        self.assertEqual(reason, "")

    def test_insufficient_data_safety(self) -> None:
        """Verify empty signal list when dataframe has fewer bars than lookback."""
        short_df = pd.DataFrame({"Close": [100.0] * 5}, index=self.idx[:5])
        signals = self.engine.generate_signals(short_df, short_df, self.config)
        self.assertEqual(signals, [])

        empty_df = pd.DataFrame()
        signals = self.engine.generate_signals(empty_df, empty_df, self.config)
        self.assertEqual(signals, [])


if __name__ == "__main__":
    unittest.main()
