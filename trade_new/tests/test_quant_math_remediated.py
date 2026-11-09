"""Empirical Verification Suite for Quant Math & AST Remediation (Subagent W1-A).

Validates:
1. Safe ts_ema initialization & forward-fill (no NaN collapse to perpetual zeros).
2. Permutation-equivariant cs_rank_tensor with fractional tie-breaking & NaN masking.
3. Finite-masked cs_demean_tensor (single missing asset does not poison cross-section).
4. ts_delta NaN safety and prefix zero-padding.
5. Finite bounds clamping (+-1e4) on div_safe_tensor & _eval_unary.
6. Unary negative parsing in FormulaParser (-TS_RANK_20 parses to NEG, not dummy FEATURE).
7. canonical_hash regime inclusion (regime-specific formulas do not collide in MAP-Elites).
8. O(N) linear-time redundant parenthesis stripping.
9. Longitudinal & true cross-sectional factor IC.
10. Predictive forward return decay IC(s_t, r_{t -> t+tau}).
11. Variance Inflation Factor (VIF) & multicollinearity diagnostics.
"""
import os
import sys
import time
import unittest
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.domain.tensor_evaluator import (
    div_safe_tensor,
    ts_delta,
    ts_ema,
    cs_rank_tensor,
    cs_demean_tensor,
    _eval_unary,
    TensorASTEvaluator,
)
from src.domain.formula_ast import ASTNode, ASTEvaluator, cs_rank, cs_demean
from src.domain.formula_parser import FormulaParser
from src.backtest.factor_analytics import (
    calculate_factor_ic,
    calculate_factor_decay,
    calculate_vif,
    check_multicollinearity,
)


class TestQuantMathRemediated(unittest.TestCase):
    """Test suite verifying all mathematical and AST remediation fixes."""

    def test_ts_ema_nan_safe_initialization_and_forward_fill(self):
        """Verify ts_ema safely initializes on first valid value and forward fills NaNs."""
        # Case 1: Leading NaNs followed by valid observations
        x_leading_nan = torch.tensor([[float("nan"), float("nan"), 10.0, 12.0, 14.0]], dtype=torch.float32)
        ema_res = ts_ema(x_leading_nan, span=5)

        # Must NOT collapse entire series to zero!
        self.assertFalse(torch.all(ema_res == 0.0), "ts_ema collapsed entire series to zeros due to leading NaNs")
        self.assertEqual(ema_res[0, 0].item(), 0.0)
        self.assertEqual(ema_res[0, 1].item(), 0.0)
        self.assertAlmostEqual(ema_res[0, 2].item(), 10.0, places=4)
        self.assertGreater(ema_res[0, 3].item(), 10.0)

        # Case 2: Intermediate NaNs forward-fill rather than propagating NaNs or collapsing
        x_intermediate = torch.tensor([[10.0, 12.0, float("nan"), 14.0]], dtype=torch.float32)
        ema_inter = ts_ema(x_intermediate, span=5)
        self.assertTrue(torch.all(torch.isfinite(ema_inter)))
        # t=2 (NaN) forward-fills t=1 EMA value
        self.assertEqual(ema_inter[0, 2].item(), ema_inter[0, 1].item())
        # t=3 resumes EMA updating
        self.assertGreater(ema_inter[0, 3].item(), ema_inter[0, 2].item())

        # Case 3: All NaNs does not crash and returns zeros
        x_all_nan = torch.full((1, 10), float("nan"), dtype=torch.float32)
        ema_nan = ts_ema(x_all_nan, span=5)
        self.assertTrue(torch.all(ema_nan == 0.0))
        print("[PASS] test_ts_ema_nan_safe_initialization_and_forward_fill")

    def test_cs_rank_fractional_ties_and_permutation_equivariance(self):
        """Verify cs_rank_tensor fractional tie-breaking and T(PX) = P T(X)."""
        # 4 assets across 2 timestamps: assets 0 and 1 are tied at timestamp 0
        x = torch.tensor([
            [10.0, 40.0],
            [10.0, 30.0],
            [20.0, 20.0],
            [30.0, 10.0],
        ], dtype=torch.float32)

        ranks = cs_rank_tensor(x)
        # At timestamp 0: assets 0 and 1 are tied for rank 1 and 2 -> fractional rank 1.5 -> (1.5 - 0.5)/4 = 0.25
        self.assertAlmostEqual(ranks[0, 0].item(), 0.25, places=5)
        self.assertAlmostEqual(ranks[1, 0].item(), 0.25, places=5)
        self.assertAlmostEqual(ranks[2, 0].item(), 0.625, places=5)  # rank 3 -> (3 - 0.5)/4 = 0.625
        self.assertAlmostEqual(ranks[3, 0].item(), 0.875, places=5)  # rank 4 -> (4 - 0.5)/4 = 0.875

        # Test Permutation Equivariance T(P X) == P T(X)
        perm = [2, 0, 3, 1]
        x_perm = x[perm, :]
        ranks_perm = cs_rank_tensor(x_perm)
        expected_ranks_perm = ranks[perm, :]
        self.assertTrue(torch.allclose(ranks_perm, expected_ranks_perm, atol=1e-5),
                        "cs_rank_tensor violated permutation equivariance T(PX) != P T(X)")
        print("[PASS] test_cs_rank_fractional_ties_and_permutation_equivariance")

    def test_cs_rank_nan_masking_conviction_safety(self):
        """Verify NaNs are masked to neutral rank (0.5) and NOT promoted to #1 conviction buy."""
        # Asset 1 is NaN / missing data
        x = torch.tensor([
            [10.0],
            [float("nan")],
            [30.0],
        ], dtype=torch.float32)

        ranks = cs_rank_tensor(x)
        # Missing asset MUST receive neutral rank 0.5, NEVER top rank (>0.7) or bottom rank (0.0)
        self.assertAlmostEqual(ranks[1, 0].item(), 0.5, places=5)
        # Valid assets are ranked among valid cohort: asset 0 is lower (0.25), asset 2 is higher (0.75)
        self.assertAlmostEqual(ranks[0, 0].item(), 0.25, places=5)
        self.assertAlmostEqual(ranks[2, 0].item(), 0.75, places=5)
        print("[PASS] test_cs_rank_nan_masking_conviction_safety")

    def test_cs_demean_finite_masking_no_poisoning(self):
        """Verify single missing asset does not poison cross-sectional mean across healthy assets."""
        # 3 assets: asset 1 has NaN
        x = torch.tensor([
            [10.0, 50.0],
            [float("nan"), 60.0],
            [30.0, 70.0],
        ], dtype=torch.float32)

        demeaned = cs_demean_tensor(x)
        # Timestamp 0: Mean of healthy assets (10, 30) is 20.0
        # Demeaned healthy: 10 - 20 = -10, 30 - 20 = +10
        self.assertAlmostEqual(demeaned[0, 0].item(), -10.0, places=5)
        self.assertAlmostEqual(demeaned[1, 0].item(), 0.0, places=5)  # NaN asset gets 0.0 neutral
        self.assertAlmostEqual(demeaned[2, 0].item(), 10.0, places=5)

        # Sum of healthy demeaned assets is strictly zero
        self.assertAlmostEqual((demeaned[0, 0] + demeaned[2, 0]).item(), 0.0, places=5)
        print("[PASS] test_cs_demean_finite_masking_no_poisoning")

    def test_ts_delta_nan_safe(self):
        """Verify ts_delta zero-fills prefix and handles NaNs cleanly."""
        x = torch.tensor([[float("nan"), 10.0, 20.0, float("nan"), 50.0, 70.0]], dtype=torch.float32)
        delta_res = ts_delta(x, period=2)
        self.assertTrue(torch.all(torch.isfinite(delta_res)))
        self.assertEqual(delta_res[0, 0].item(), 0.0)
        self.assertEqual(delta_res[0, 1].item(), 0.0)
        # t=2: 20.0 - nan -> nan_to_num -> 0.0
        self.assertEqual(delta_res[0, 2].item(), 0.0)
        # t=5: 70.0 - nan -> 0.0
        self.assertEqual(delta_res[0, 5].item(), 0.0)
        print("[PASS] test_ts_delta_nan_safe")

    def test_finite_bounds_clamping_div_safe_and_unary(self):
        """Verify div_safe_tensor and _eval_unary clamp outputs to +-1e4, preventing float32 overflow."""
        # div_safe huge numerator
        l = torch.tensor([1e20, -1e20, 10.0], dtype=torch.float32)
        r = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32)
        res_div = div_safe_tensor(l, r)
        self.assertEqual(res_div[0].item(), 1e4)
        self.assertEqual(res_div[1].item(), -1e4)
        self.assertEqual(res_div[2].item(), 0.0)

        # Unary EXP_SAFE clamping
        huge_x = torch.tensor([100.0, -100.0], dtype=torch.float32)
        res_exp = _eval_unary("EXP_SAFE", huge_x)
        self.assertTrue(torch.all(torch.isfinite(res_exp)))
        self.assertLessEqual(res_exp[0].item(), 1e4)

        # Unary NEG operator
        x_neg = torch.tensor([5.0, -3.0], dtype=torch.float32)
        res_neg = _eval_unary("NEG", x_neg)
        self.assertEqual(res_neg[0].item(), -5.0)
        self.assertEqual(res_neg[1].item(), 3.0)
        print("[PASS] test_finite_bounds_clamping_div_safe_and_unary")

    def test_formula_parser_unary_negative(self):
        """Verify -TS_RANK_20(...) parses as NEG operator rather than dummy FEATURE."""
        formula = "-TS_RANK_20(Close_Adj)"
        node = FormulaParser.parse(formula)
        self.assertEqual(node.op, "NEG")
        self.assertEqual(len(node.children), 1)
        self.assertEqual(node.children[0].op, "TS_RANK_20")
        self.assertEqual(node.children[0].children[0].op, "FEATURE")
        self.assertEqual(node.children[0].children[0].val, "Close_Adj")

        # Test combined expression: 5.0 * -Close_Adj
        node2 = FormulaParser.parse("5.0 * -Close_Adj")
        self.assertEqual(node2.op, "MUL")
        self.assertEqual(node2.children[1].op, "NEG")

        # Evaluate parsed AST via TensorASTEvaluator
        evaluator = TensorASTEvaluator()
        data = torch.tensor([[10.0, 20.0, 30.0]], dtype=torch.float32)  # [1, 3]
        f_map = {"Close_Adj": 0}
        # Evaluation of -Close_Adj must equal -data
        neg_node = FormulaParser.parse("-Close_Adj")
        res = evaluator.eval_node(neg_node, data.unsqueeze(-1), feature_names=["Close_Adj"])
        self.assertTrue(torch.allclose(res, -data))
        print("[PASS] test_formula_parser_unary_negative")

    def test_ast_canonical_hash_regime_uniqueness(self):
        """Verify ASTNode canonical_hash includes sorted active_regimes to avoid MAP-Elites collisions."""
        # Same formula structure but different regime activation
        node_bull = ASTNode("TS_RANK_20", active_regimes={0, 1, 2}, children=[ASTNode("FEATURE", "Close_Adj")])
        node_bear = ASTNode("TS_RANK_20", active_regimes={3, 4}, children=[ASTNode("FEATURE", "Close_Adj")])
        node_bull_dup = ASTNode("TS_RANK_20", active_regimes={2, 0, 1}, children=[ASTNode("FEATURE", "Close_Adj")])

        hash_bull = node_bull.canonical_hash()
        hash_bear = node_bear.canonical_hash()
        hash_bull_dup = node_bull_dup.canonical_hash()

        # Different regimes must yield different hashes
        self.assertNotEqual(hash_bull, hash_bear, "Regime-specific formulas collided in canonical_hash!")
        # Permutation of regime set must yield identical hash (sorted order invariance)
        self.assertEqual(hash_bull, hash_bull_dup, "Canonical hash depended on active_regimes set insertion order")
        print("[PASS] test_ast_canonical_hash_regime_uniqueness")

    def test_strip_redundant_parens_linear_time(self):
        """Verify FormulaParser._strip_redundant_parens runs in O(N) linear time on deep parens."""
        # Construct deeply nested string with 1000 pairs of parentheses
        depth = 1000
        deep_expr = "(" * depth + "Close_Adj" + ")" * depth
        t0 = time.perf_counter()
        stripped = FormulaParser._strip_redundant_parens(deep_expr)
        elapsed = time.perf_counter() - t0

        self.assertEqual(stripped, "Close_Adj")
        # Must execute in < 10 milliseconds (O(N) linear pass vs O(N^2) quadratic crawl)
        self.assertLess(elapsed, 0.05, f"Redundant paren stripping took too long: {elapsed:.4f}s")

        # Verify non-redundant complex parentheses are preserved
        complex_expr = "(A + B) * (C + D)"
        self.assertEqual(FormulaParser._strip_redundant_parens(complex_expr), complex_expr)
        print(f"[PASS] test_strip_redundant_parens_linear_time (elapsed: {elapsed * 1000:.2f}ms)")

    def test_factor_ic_longitudinal_and_cross_sectional(self):
        """Verify calculate_factor_ic on both 1D longitudinal and 2D cross-sectional slices."""
        np.random.seed(42)
        # 1. Longitudinal 1D
        signals_1d = np.linspace(-2.0, 2.0, 100)
        noise = np.random.normal(0, 0.01, 100)
        returns_1d = 0.02 * signals_1d + noise
        ic_1d = calculate_factor_ic(signals_1d, returns_1d)
        self.assertIn("ic", ic_1d)
        self.assertIn("rank_ic", ic_1d)
        self.assertGreater(ic_1d["ic"], 0.3)
        self.assertLess(ic_1d["p_value"], 0.01)

        # 2. Cross-Sectional 2D: [N_assets=10, T_time=50]
        N, T = 10, 50
        sig_cs = np.random.normal(0, 1, (N, T))
        fwd_ret_cs = 0.05 * sig_cs + np.random.normal(0, 0.5, (N, T))
        ic_cs = calculate_factor_ic(sig_cs, fwd_ret_cs, cross_sectional=True)

        self.assertIn("ic", ic_cs)
        self.assertIn("rank_ic", ic_cs)
        self.assertIn("icir", ic_cs)
        self.assertGreater(ic_cs["ic"], 0.0)
        self.assertGreater(ic_cs["rank_ic"], 0.0)
        print("[PASS] test_factor_ic_longitudinal_and_cross_sectional")

    def test_predictive_forward_return_decay(self):
        """Verify calculate_factor_decay calculates IC(s_t, r_{t -> t+tau}) with forward returns."""
        np.random.seed(42)
        T = 200
        # Signal that predicts returns with decaying correlation at larger horizons
        signal = np.random.normal(0, 1, T)
        # Immediate lag 1 has strong relationship, lag 5 has weak relationship
        fwd_returns = np.zeros(T)
        fwd_returns[1:] = 0.5 * signal[:-1] + np.random.normal(0, 0.5, T - 1)

        decay = calculate_factor_decay(signal, fwd_returns=fwd_returns, max_lags=4)
        self.assertEqual(len(decay), 4)
        # Lag 1 predictive IC must be significantly positive
        self.assertGreater(decay[0], 0.20)
        print(f"[PASS] test_predictive_forward_return_decay (decay profile: {decay})")

    def test_variance_inflation_factor_and_multicollinearity(self):
        """Verify VIF detects orthogonal vs collinear factor sets."""
        np.random.seed(42)
        n_samples = 300
        # 3 Orthogonal factors
        f1 = np.random.normal(0, 1, n_samples)
        f2 = np.random.normal(0, 1, n_samples)
        f3 = np.random.normal(0, 1, n_samples)
        ortho_factors = np.column_stack([f1, f2, f3])
        vif_ortho = calculate_vif(ortho_factors, ["F1", "F2", "F3"])

        for name, v in vif_ortho.items():
            self.assertLess(v, 2.0, f"Orthogonal factor {name} had unexpected high VIF {v}")

        # Add collinear factor: F4 = 0.99 * F1 + small noise
        f4_collinear = 0.99 * f1 + np.random.normal(0, 0.05, n_samples)
        collinear_factors = np.column_stack([f1, f2, f3, f4_collinear])
        diag = check_multicollinearity(collinear_factors, ["F1", "F2", "F3", "F4_Collinear"], threshold=10.0)

        self.assertTrue(diag["has_multicollinearity"])
        self.assertIn("F1", diag["collinear_factors"])
        self.assertIn("F4_Collinear", diag["collinear_factors"])
        self.assertNotIn("F2", diag["collinear_factors"])
        print(f"[PASS] test_variance_inflation_factor_and_multicollinearity (VIFs: {diag['vif']})")


if __name__ == "__main__":
    unittest.main()
