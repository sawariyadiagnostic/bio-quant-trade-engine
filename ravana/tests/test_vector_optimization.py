"""Unit tests for VectorLake, VectorSimulator, and AntiOverfittingOptimizer."""
from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from ravana.optimization.anti_overfitting_suite import AntiOverfittingOptimizer
from ravana.optimization.vector_engine import (
    VectorLake,
    VectorSimulationResult,
    VectorSimulator,
)


class TestVectorOptimization(unittest.TestCase):
    """Test suite for vectorized simulation and anti-overfitting optimizer."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.lake = VectorLake()
        cls.sim = VectorSimulator(cls.lake)
        cls.opt = AntiOverfittingOptimizer(lake=cls.lake)

    def test_lake_dimensions_and_alignment(self) -> None:
        self.assertGreater(self.lake.n_bars, 1000)
        self.assertGreater(self.lake.n_syms, 5)
        self.assertEqual(self.lake.close_mat.shape, (self.lake.n_bars, self.lake.n_syms))
        self.assertEqual(self.lake.high_mat.shape, (self.lake.n_bars, self.lake.n_syms))
        self.assertEqual(self.lake.low_mat.shape, (self.lake.n_bars, self.lake.n_syms))
        self.assertEqual(self.lake.vwap_mat.shape, (self.lake.n_bars, self.lake.n_syms))

    def test_empty_mask_simulation(self) -> None:
        long_m = np.zeros_like(self.lake.close_mat, dtype=bool)
        short_m = np.zeros_like(self.lake.close_mat, dtype=bool)
        res = self.sim.simulate(long_m, short_m)
        self.assertEqual(res.total_trades, 0)
        self.assertEqual(res.net_pnl, 0.0)

    def test_walk_forward_evaluation_structure(self) -> None:
        long_m, short_m = self.opt.build_strategy1_orb_masks()
        res = self.opt.evaluate_strategy_walk_forward(
            strategy_name="Test_ORB",
            long_mask=long_m,
            short_mask=short_m,
        )
        self.assertIn("train", res)
        self.assertIn("test_oos", res)
        self.assertIn("full_60d", res)
        self.assertIn("is_overfit", res)
        self.assertIn("mc_max_dd_pct", res)


if __name__ == "__main__":
    unittest.main()
