"""Unit tests verifying ADHD Batch 1 Quant Math & Numerical Singularity Remediations (ADHD-01 to ADHD-09)."""
import math
import os
import sys
import unittest
import numpy as np
import torch
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.domain.tensor_evaluator import (
    cs_rank_tensor,
    ts_max,
    ts_min,
    ts_rank_20,
    TensorASTEvaluator,
    div_safe_tensor,
)
from src.domain.formula_ast import (
    ASTNode,
    ASTEvaluator,
    cs_rank,
    _fast_ts_rank_20,
)
from src.backtest.matrix_engine import MatrixSimulator, compute_sharpe_and_dsr
from src.domain.models import CircuitDamper, quantize_nse_tick
from src.data.vector_lake import adjust_corporate_splits


class TestADHDBatch1QuantMath(unittest.TestCase):
    """Test suite for ADHD-01 through ADHD-09 mathematical fixes."""

    def test_adhd_01_cs_rank_fast_and_correct(self):
        """ADHD-01: Verify O(B log B * T) sort-based ranking on large cross-section without memory explosion."""
        B, T = 100, 500
        np.random.seed(42)
        mat = np.random.randn(B, T).astype(np.float32)
        mat[0:10, 0:50] = np.nan

        # PyTorch
        t_mat = torch.from_numpy(mat)
        ranks_torch = cs_rank_tensor(t_mat)
        self.assertEqual(ranks_torch.shape, (B, T))
        self.assertTrue(torch.all(ranks_torch >= 0.0) and torch.all(ranks_torch <= 1.0))
        # Masked NaN elements must be neutral 0.5
        self.assertTrue(torch.all(ranks_torch[0:10, 0:50] == 0.5))

        # NumPy
        ranks_np = cs_rank(mat)
        self.assertEqual(ranks_np.shape, (B, T))
        self.assertTrue(np.all(ranks_np >= 0.0) and np.all(ranks_np <= 1.0))
        self.assertTrue(np.all(ranks_np[0:10, 0:50] == 0.5))
        print("[PASS] test_adhd_01_cs_rank_fast_and_correct")

    def test_adhd_02_ts_extrema_no_zero_inversion(self):
        """ADHD-02: Verify ts_max and ts_min do not invert negative maxima or zero-out positive minima."""
        # Negative returns
        neg_x = torch.tensor([[-0.05, -0.02, -0.08, float("nan"), -0.01]], dtype=torch.float32)
        max_res = ts_max(neg_x, window=3)
        # Bar 4 (-0.01) window includes [-0.08, NaN, -0.01], true max is -0.01, NOT 0.0
        self.assertAlmostEqual(max_res[0, 4].item(), -0.01, places=4)

        # Positive stock prices
        pos_prices = torch.tensor([[100.0, 102.0, float("nan"), 105.0]], dtype=torch.float32)
        min_res = ts_min(pos_prices, window=3)
        # Bar 3 (105.0) window includes [102.0, NaN, 105.0], true min is 102.0, NOT 0.0
        self.assertAlmostEqual(min_res[0, 3].item(), 102.0, places=1)
        print("[PASS] test_adhd_02_ts_extrema_no_zero_inversion")

    def test_adhd_03_ts_rank_dynamic_nan_denominator(self):
        """ADHD-03: Verify ts_rank dynamically counts valid observations and avoids percentile deflation."""
        # 10 leading NaNs followed by 10 identical values of 10.0
        arr = np.array([np.nan] * 10 + [10.0] * 10, dtype=np.float64)
        ranks = _fast_ts_rank_20(arr)
        # At index 19 (10 equal valid observations), rank among valid must be 0.5, NOT 0.25
        self.assertAlmostEqual(ranks[19], 0.5, places=2)

        # PyTorch
        t_arr = torch.from_numpy(arr).unsqueeze(0)
        t_ranks = ts_rank_20(t_arr, window=20)
        self.assertAlmostEqual(t_ranks[0, 19].item(), 0.5, places=2)
        print("[PASS] test_adhd_03_ts_rank_dynamic_nan_denominator")

    def test_adhd_04_ast_constant_hash_and_eval(self):
        """ADHD-04: Verify canonical_hash and evaluators distinguish different numeric CONST nodes."""
        n1 = ASTNode("CONST", val="1.0")
        n2 = ASTNode("CONST", val="999.0")
        self.assertNotEqual(n1.canonical_hash(), n2.canonical_hash())
        self.assertEqual(n1.to_formula_str(), "1.0")
        self.assertEqual(n2.to_formula_str(), "999.0")

        # Tensor evaluator
        X = torch.zeros((5, 10), dtype=torch.float32)
        res1 = TensorASTEvaluator.eval_node(n1, X)
        res2 = TensorASTEvaluator.eval_node(n2, X)
        self.assertTrue(torch.all(res1 == 1.0))
        self.assertTrue(torch.all(res2 == 999.0))
        print("[PASS] test_adhd_04_ast_constant_hash_and_eval")

    def test_adhd_05_matrix_simulator_gross_leverage_bound(self):
        """ADHD-05: Verify MatrixSimulator caps 3D multi-asset gross exposure to sum(|w|) <= 1.0."""
        sim = MatrixSimulator(friction_bps=1.2)
        B, N, T = 10, 50, 100
        # 50 assets each with position 0.5 (unnormalized sum = 25.0x leverage)
        raw_positions = np.full((B, N, T), 0.5, dtype=np.float32)
        returns = np.random.normal(0.001, 0.02, (N, T)).astype(np.float32)

        res = sim.simulate_batch(raw_positions, returns, lookback_years=1.0)
        # Portfolio returns must not blow up with 25x drawdowns
        self.assertTrue(np.all(res["max_drawdown"] <= 100.0))
        self.assertTrue(np.all(np.isfinite(res["cagr"])))
        print("[PASS] test_adhd_05_matrix_simulator_gross_leverage_bound")

    def test_adhd_06_circuit_damper_symmetric_and_tick_nan(self):
        """ADHD-06: Verify symmetric circuit bands and safe tick quantization on NaN/Inf."""
        # Symmetric bands around prev_close = 100 with 10% band
        bands_low = CircuitDamper.calculate_bands(current_price=91.0, prev_close=100.0, band_pct=0.10)
        bands_high = CircuitDamper.calculate_bands(current_price=109.0, prev_close=100.0, band_pct=0.10)
        # Both ₹1 away from circuit limit: distances must be identical (1.0 / 100.0 = 0.01)
        self.assertAlmostEqual(bands_low.distance_to_lower_pct, 0.01, places=4)
        self.assertAlmostEqual(bands_high.distance_to_upper_pct, 0.01, places=4)

        # Tick quantization NaN safety
        self.assertEqual(quantize_nse_tick(float("nan")), 0.0)
        self.assertEqual(quantize_nse_tick(float("inf")), 0.0)
        self.assertEqual(quantize_nse_tick(102.345, 0.05), 102.35)
        print("[PASS] test_adhd_06_circuit_damper_symmetric_and_tick_nan")

    def test_adhd_07_vector_lake_dividend_volume_protection(self):
        """ADHD-07: Verify corporate split volume scaling applies only to genuine splits, not cash dividends."""
        df = pd.DataFrame({
            "Open": [100.0, 100.0],
            "High": [105.0, 105.0],
            "Low": [95.0, 95.0],
            "Close": [100.0, 100.0],
            "Close_Adj": [92.0, 92.0],  # 8% cash dividend, k = 0.92
            "Volume": [1000.0, 1000.0],
        })
        c, o, h, l, v = adjust_corporate_splits(df)
        # Volume must NOT be inflated to 1000 / 0.92 = 1086
        self.assertEqual(v[0], 1000.0)

        # Real 2-for-1 split (k = 0.50)
        df_split = pd.DataFrame({
            "Open": [100.0, 100.0],
            "High": [105.0, 105.0],
            "Low": [95.0, 95.0],
            "Close": [100.0, 100.0],
            "Close_Adj": [50.0, 50.0],  # 2:1 stock split
            "Volume": [1000.0, 1000.0],
        })
        _, _, _, _, v_split = adjust_corporate_splits(df_split)
        # Volume doubles on 2:1 split
        self.assertEqual(v_split[0], 2000.0)
        print("[PASS] test_adhd_07_vector_lake_dividend_volume_protection")

    def test_adhd_08_dsr_sparse_alpha_protection(self):
        """ADHD-08: Verify active-conditioned DSR avoids pinning sparse alphas to 0.50."""
        # 1000 bars, in trade only 5% of the time with positive returns
        B, T = 10, 1000
        net_rets = np.zeros((B, T), dtype=np.float64)
        for b in range(B):
            active_idx = np.random.choice(T, size=50, replace=False)
            net_rets[b, active_idx] = np.random.normal(0.01, 0.005, size=50)

        n_active = np.full(B, 50.0)
        sharpe, dsr = compute_sharpe_and_dsr(net_rets, n_active, total_bars=T)
        self.assertEqual(sharpe.shape, (B,))
        self.assertEqual(dsr.shape, (B,))
        self.assertTrue(np.all(sharpe > 0.0))
        # DSR should not be pinned to exactly 0.50
        self.assertFalse(np.all(dsr == 0.50))
        print("[PASS] test_adhd_08_dsr_sparse_alpha_protection")

    def test_adhd_09_div_safe_parity_and_clamping(self):
        """ADHD-09: Verify DIV_SAFE is clamped to [-1e4, 1e4] across both ASTEvaluator and TensorASTEvaluator."""
        # ASTEvaluator
        df = pd.DataFrame({"X": [1.0], "Y": [2e-6]})
        node = ASTNode("DIV_SAFE", children=[ASTNode("FEATURE", "X"), ASTNode("FEATURE", "Y")])
        res_ast = ASTEvaluator.eval_node(node, df)
        self.assertTrue(np.all(res_ast <= 10000.0))

        # TensorASTEvaluator
        t_l = torch.tensor([1.0], dtype=torch.float32)
        t_r = torch.tensor([2e-6], dtype=torch.float32)
        res_tensor = div_safe_tensor(t_l, t_r)
        self.assertTrue(torch.all(res_tensor <= 10000.0))
        print("[PASS] test_adhd_09_div_safe_parity_and_clamping")


if __name__ == "__main__":
    unittest.main()
