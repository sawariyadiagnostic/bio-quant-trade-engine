"""Unit Tests for RAVANA Muscle 2: PyTorch Vectorized Tensor AST Evaluator.

Verifies:
1. Numerical parity between PyTorch TensorASTEvaluator and NumPy ASTEvaluator.
2. 2D [T, F] and 3D [B, T, F] tensor dimensional evaluation.
3. Zero-crash numerical bounds under extreme inputs (NaNs, Infs, 0-division).
4. Full operator coverage (arithmetic, rolling, correlation, blend, delays).
"""
import unittest
import numpy as np
import pandas as pd
import torch

from ravana.domain.formula_ast import (
    ASTNode,
    ASTEvaluator,
    STANDARD_FEATURES,
    TreeGenerator,
)
from ravana.domain.formula_parser import FormulaParser
from ravana.compute.tensor_evaluator import TensorASTEvaluator


class TestTensorASTEvaluator(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.T = 120
        self.F = len(STANDARD_FEATURES)
        self.feature_map = {name: i for i, name in enumerate(STANDARD_FEATURES)}

        # Create synthetic realistic market data
        c = np.random.uniform(500.0, 550.0, self.T)
        h = c + np.random.uniform(1.0, 5.0, self.T)
        l = c - np.random.uniform(1.0, 5.0, self.T)
        o = np.random.uniform(l, h)
        v = np.random.uniform(1e5, 1e6, self.T)

        data_dict = {
            "Open": o,
            "High": h,
            "Low": l,
            "Close": c,
            "Close_Adj": c,
            "Volume": v,
            "Spread": (h - l) / c,
            "Vol_ZScore": np.random.uniform(-2.0, 2.0, self.T),
            "Vol_Price_Flow": (c - o) * v / c,
            "True_Range": h - l,
            "TR_Expansion": np.ones(self.T),
            "EMA_Slope": np.zeros(self.T),
            "Dist_EMA20": np.zeros(self.T),
            "Dist_VWAP": np.zeros(self.T),
            "Downside_Vol_Ratio": np.full(self.T, 0.5),
            "RSI_14": np.random.uniform(30.0, 70.0, self.T),
            "Parkinson_Vol": np.random.uniform(0.01, 0.05, self.T),
            "Garman_Klass_Vol": np.random.uniform(0.01, 0.05, self.T),
            "SP500_Ret": np.random.uniform(-0.02, 0.02, self.T),
            "ADR_Ret": np.random.uniform(-0.02, 0.02, self.T),
            "Global_Confluence": np.ones(self.T),
        }
        self.df = pd.DataFrame(data_dict)

        # Convert to 2D PyTorch Tensor [T, F]
        feature_matrix = np.column_stack([self.df[col].to_numpy(dtype=np.float32) for col in STANDARD_FEATURES])
        self.X_2d = torch.from_numpy(feature_matrix)

        # Convert to 3D PyTorch Tensor [B=2, T, F]
        self.X_3d = torch.stack([self.X_2d, self.X_2d * 1.05], dim=0)

    def test_arithmetic_numerical_parity(self):
        """Verify ADD, SUB, MUL, DIV_SAFE parity between PyTorch and NumPy."""
        formulas = [
            "(Close + Open)",
            "(High - Low)",
            "(Close * 1.5)",
            "((High - Low) / (Close + 1e-4))",
        ]
        for f_str in formulas:
            ast = FormulaParser.parse(f_str)
            numpy_res = ASTEvaluator.eval_node(ast, self.df)
            torch_res = TensorASTEvaluator.eval_node(ast, self.X_2d, self.feature_map).numpy()

            np.testing.assert_allclose(
                torch_res, numpy_res, rtol=1e-4, atol=1e-4,
                err_msg=f"Parity mismatch on {f_str}"
            )

    def test_unary_numerical_parity(self):
        """Verify unary operator parity between PyTorch and NumPy."""
        for op in ("ABS", "SIGN", "SIGMOID", "TANH", "RELU", "EXP_SAFE", "LOG_SAFE", "NEG", "SQRT_ABS"):
            ast = ASTNode(op, children=[ASTNode("FEATURE", "Vol_ZScore")])
            numpy_res = ASTEvaluator.eval_node(ast, self.df)
            torch_res = TensorASTEvaluator.eval_node(ast, self.X_2d, self.feature_map).numpy()

            np.testing.assert_allclose(
                torch_res, numpy_res, rtol=1e-4, atol=1e-4,
                err_msg=f"Parity mismatch on unary op {op}"
            )

    def test_rolling_numerical_parity(self):
        """Verify rolling operator parity: DELTA, EMA, DELAY, DECAY, ZSCORE, CORR."""
        formulas = [
            ("DELTA_5", ASTNode("DELTA_5", children=[ASTNode("FEATURE", "Close")])),
            ("TS_DELAY_1", ASTNode("TS_DELAY_1", children=[ASTNode("FEATURE", "Close")])),
            ("TS_DELAY_5", ASTNode("TS_DELAY_5", children=[ASTNode("FEATURE", "Close")])),
            ("EMA_10", ASTNode("EMA_10", children=[ASTNode("FEATURE", "Close")])),
            ("TS_ZSCORE_20", ASTNode("TS_ZSCORE_20", children=[ASTNode("FEATURE", "Close")])),
            ("TS_CORR_20", ASTNode("TS_CORR_20", children=[ASTNode("FEATURE", "Close"), ASTNode("FEATURE", "Volume")])),
            ("TS_DECAY_LINEAR_10", ASTNode("TS_DECAY_LINEAR_10", children=[ASTNode("FEATURE", "Close")])),
        ]
        for name, ast in formulas:
            numpy_res = ASTEvaluator.eval_node(ast, self.df)
            torch_res = TensorASTEvaluator.eval_node(ast, self.X_2d, self.feature_map).numpy()

            # Compare suffix where rolling windows are populated
            np.testing.assert_allclose(
                torch_res[25:], numpy_res[25:], rtol=1e-3, atol=1e-3,
                err_msg=f"Rolling parity mismatch on {name}"
            )

    def test_blend_smooth_parity(self):
        """Verify continuous BLEND parity."""
        ast = ASTNode("BLEND", children=[
            ASTNode("FEATURE", "Vol_ZScore"),
            ASTNode("FEATURE", "Close"),
            ASTNode("FEATURE", "Open")
        ])
        numpy_res = ASTEvaluator.eval_node(ast, self.df)
        torch_res = TensorASTEvaluator.eval_node(ast, self.X_2d, self.feature_map).numpy()

        np.testing.assert_allclose(torch_res, numpy_res, rtol=1e-4, atol=1e-4)

    def test_3d_tensor_batch_evaluation(self):
        """Verify 3D [B=2, T, F] batch evaluation computes parallel asset slices."""
        ast = FormulaParser.parse("TS_CORR_20(Close, Volume) + BLEND(Vol_ZScore, High, Low)")
        res_3d = TensorASTEvaluator.eval_node(ast, self.X_3d, self.feature_map)

        self.assertEqual(res_3d.shape, (2, self.T))
        self.assertTrue(torch.all(torch.isfinite(res_3d)))

    def test_extreme_inputs_zero_crash(self):
        """Verify zero division, extreme values, and NaNs do not cause crash."""
        extreme_X = torch.tensor([
            [1e12, 0.0, -1e12],
            [0.0, 0.0, 0.0],
            [float("nan"), float("inf"), -float("inf")]
        ], dtype=torch.float32)
        f_map = {"High": 0, "Low": 1, "Close": 2}

        div_ast = ASTNode("DIV_SAFE", children=[ASTNode("FEATURE", "High"), ASTNode("FEATURE", "Low")])
        res = TensorASTEvaluator.eval_node(div_ast, extreme_X, f_map)
        self.assertTrue(torch.all(torch.isfinite(res)))

    def test_device_consistency(self):
        """Verify evaluator works across cpu and (if available) cuda with identical results."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_dev = self.X_2d.to(device)
        ast = TreeGenerator.generate(max_depth=3)

        res_cpu = TensorASTEvaluator.eval_node(ast, self.X_2d, self.feature_map)
        res_dev = TensorASTEvaluator.eval_node(ast, X_dev, self.feature_map)

        np.testing.assert_allclose(
            res_cpu.numpy(), res_dev.cpu().numpy(), rtol=1e-4, atol=1e-4
        )

    def test_small_t_windows(self):
        """Verify T < window (T=1, 5, 19 with window=20) evaluates without crash and yields finite values."""
        for t_len in [1, 5, 19]:
            x_small = torch.randn(2, t_len, self.F)
            ast = ASTNode("TS_ZSCORE_20", children=[ASTNode("FEATURE", "Close")])
            out = TensorASTEvaluator.eval_node(ast, x_small, self.feature_map)
            self.assertEqual(out.shape, (2, t_len))
            self.assertTrue(torch.all(torch.isfinite(out)))

    def test_constant_array_zero_variance(self):
        """Verify constant inputs with zero variance yield 0.0 without NaNs."""
        x_const = torch.full((2, 50, self.F), 42.0, dtype=torch.float32)
        ast_zscore = ASTNode("TS_ZSCORE_20", children=[ASTNode("FEATURE", "Close")])
        out_z = TensorASTEvaluator.eval_node(ast_zscore, x_const, self.feature_map)
        self.assertTrue(torch.all(out_z == 0.0))

        ast_corr = ASTNode("TS_CORR_20", children=[ASTNode("FEATURE", "Close"), ASTNode("FEATURE", "Volume")])
        out_c = TensorASTEvaluator.eval_node(ast_corr, x_const, self.feature_map)
        self.assertTrue(torch.all(out_c == 0.0))

    def test_nan_inf_feature_propagation(self):
        """Verify raw feature slices with NaNs and Infs produce strictly finite bounded results."""
        x_bad = torch.randn(20, self.F)
        x_bad[0:5, :] = float("nan")
        x_bad[5:10, :] = float("inf")
        x_bad[10:15, :] = float("-inf")

        for op in ("FEATURE", "DELTA_5", "TS_DELAY_1", "TS_DELAY_5", "EMA_10", "TS_MAX_10", "TS_MIN_10", "TS_DECAY_LINEAR_10"):
            ast = ASTNode(op, children=[ASTNode("FEATURE", "Close")]) if op != "FEATURE" else ASTNode("FEATURE", "Close")
            out = TensorASTEvaluator.eval_node(ast, x_bad, self.feature_map)
            self.assertTrue(torch.all(torch.isfinite(out)), f"Operator {op} produced non-finite values on bad data")

    def test_dispatcher_3d_routing(self):
        """Verify ComputeDispatcher routes 3D tensors on CPU seamlessly without ValueError."""
        from ravana.compute.dispatcher import ComputeDispatcher
        ast = ASTNode("FEATURE", "Close")
        out = ComputeDispatcher.evaluate(ast, self.X_3d, self.feature_map, backend="auto", device="cpu")
        self.assertEqual(out.shape, (2, self.T))
        self.assertTrue(torch.all(torch.isfinite(out)))


if __name__ == "__main__":
    unittest.main()
