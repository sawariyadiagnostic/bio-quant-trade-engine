"""Comprehensive tests for Muscle 10: Multi-Objective Alpha Fitness Engine."""
import math
import unittest
import numpy as np
import pandas as pd

from ravana.domain.formula_ast import ASTNode
from ravana.fitness.config import FitnessConfig, ObjectiveWeights
from ravana.fitness.deflated_sharpe import (
    compute_dsr,
    compute_psr,
    expected_maximum_sharpe,
)
from ravana.fitness.engine import MultiObjectiveFitnessEngine
from ravana.fitness.fdr import benjamini_hochberg_fdr, ic_to_p_value
from ravana.fitness.metrics import (
    compute_annualized_sharpe,
    compute_ic_and_rank_ic,
    compute_max_drawdown,
    compute_turnover,
)
from ravana.fitness.pareto import dominates, find_pareto_frontier


class TestMultiObjectiveFitness(unittest.TestCase):
    """Test suite verifying econometric, statistical, and optimization metrics."""

    def setUp(self) -> None:
        np.random.seed(42)
        self.n_bars = 100
        self.target = np.random.normal(0.001, 0.01, self.n_bars)
        # Signal positively correlated with target
        self.signal_pos = self.target + np.random.normal(0, 0.005, self.n_bars)
        # Noise signal uncorrelated
        self.signal_noise = np.random.normal(0, 1.0, self.n_bars)

    def test_ic_and_rank_ic_computation(self) -> None:
        """Verify Pearson and Spearman rank IC on correlated and noise vectors."""
        ic_pos, rank_ic_pos = compute_ic_and_rank_ic(self.signal_pos, self.target)
        self.assertGreater(ic_pos, 0.5)
        self.assertGreater(rank_ic_pos, 0.5)

        ic_noise, rank_ic_noise = compute_ic_and_rank_ic(self.signal_noise, self.target)
        self.assertLess(abs(ic_noise), 0.25)
        self.assertLess(abs(rank_ic_noise), 0.25)

    def test_deflated_sharpe_ratio_deflation_property(self) -> None:
        """Verify DSR decreases strictly as the number of tested trials increases (selection bias)."""
        rets = np.random.normal(0.002, 0.01, 252)  # Positive Sharpe ~3.0
        dsr_10 = compute_dsr(rets, num_trials=10, variance_of_trials=0.5)
        dsr_500 = compute_dsr(rets, num_trials=500, variance_of_trials=0.5)
        self.assertGreater(dsr_10, dsr_500)
        self.assertTrue(0.0 <= dsr_500 <= 1.0)

    def test_benjamini_hochberg_fdr_control(self) -> None:
        """Verify Benjamini-Hochberg controls false discovery rate."""
        # 3 strong signals (tiny p-values) and 7 pure noise (large p-values)
        p_vals = [0.0001, 0.001, 0.005, 0.35, 0.50, 0.62, 0.78, 0.81, 0.90, 0.95]
        passes = benjamini_hochberg_fdr(p_vals, q_alpha=0.05)

        # The first 3 should pass FDR
        self.assertTrue(passes[0])
        self.assertTrue(passes[1])
        self.assertTrue(passes[2])
        # The noise tests should fail FDR
        self.assertFalse(passes[3])
        self.assertFalse(passes[-1])

    def test_turnover_and_drawdown(self) -> None:
        """Verify turnover and max drawdown calculations."""
        # Flat signal has zero turnover
        flat_sig = np.ones(50)
        self.assertEqual(compute_turnover(flat_sig), 0.0)

        # Alternating signal has high turnover
        alt_sig = np.array([1.0, -1.0] * 25)
        self.assertGreater(compute_turnover(alt_sig), 0.9)

        # Drop of 20% in returns
        rets = np.array([0.0, 0.1, -0.2, 0.05])
        mdd = compute_max_drawdown(rets)
        self.assertGreater(mdd, 0.1)

    def test_pareto_frontier_selection(self) -> None:
        """Verify non-dominated sorting correctly identifies the Pareto frontier."""
        # Candidate 0: High IC (0.3), High DSR (0.9), Low Turnover (0.1) -> Pareto
        # Candidate 1: Lower IC (0.1), Lower DSR (0.5), Higher Turnover (0.4) -> Dominated by 0
        # Candidate 2: Ultra High IC (0.4), Lower DSR (0.8), Medium Turnover (0.2) -> Tradeoff, Pareto
        objs = [
            (0.3, 0.25, 0.9, 0.1, 0.05, 5.0),
            (0.1, 0.05, 0.5, 0.4, 0.15, 8.0),
            (0.4, 0.35, 0.8, 0.2, 0.08, 6.0),
        ]
        pareto = find_pareto_frontier(objs)
        self.assertIn(0, pareto)
        self.assertIn(2, pareto)
        self.assertNotIn(1, pareto)  # Candidate 1 is dominated!

    def test_multiobjective_fitness_engine_population(self) -> None:
        """Verify end-to-end multi-objective evaluation across a batch of formulas."""
        engine = MultiObjectiveFitnessEngine()

        signals = np.vstack([self.signal_pos, self.signal_noise])
        nodes = [
            ASTNode("FEATURE", val="LOG_RET_1"),
            ASTNode("ADD", children=[ASTNode("FEATURE", val="LOG_RET_1"), ASTNode("FEATURE", val="VWAP_DRIFT")]),
        ]

        metrics = engine.evaluate_population(signals, self.target, nodes=nodes)
        self.assertEqual(len(metrics), 2)

        # First ranked should be the positive correlated signal
        self.assertGreater(metrics[0].ic, metrics[1].ic)
        self.assertGreater(metrics[0].composite_fitness, metrics[1].composite_fitness)
        self.assertTrue(any(m.is_pareto_optimal for m in metrics))


if __name__ == "__main__":
    unittest.main()
