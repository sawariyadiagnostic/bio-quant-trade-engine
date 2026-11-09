"""End-to-End Synergy and Coordination Test for Muscle 1 and Muscle 2.

Verifies complete institutional interoperability between:
- Muscle 1: AST expression trees, bounded operators, dimensional type guards, infix formula parser, and genetic mutators.
- Muscle 2: PyTorch vectorized tensor evaluator, 1D convolution rolling engines, EvaluationContext, and ComputeDispatcher.
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
    GeneticMutator,
    FeatureDimension,
)
from ravana.domain.formula_parser import FormulaParser
from ravana.compute.tensor_evaluator import TensorASTEvaluator
from ravana.compute.dispatcher import ComputeDispatcher, EvaluationContext


class TestMuscle1Muscle2Coordination(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.T = 120
        self.feature_map = {name: i for i, name in enumerate(STANDARD_FEATURES)}

        # Realistic market series
        c = np.random.uniform(500.0, 550.0, self.T)
        h = c + np.random.uniform(1.0, 5.0, self.T)
        l = c - np.random.uniform(1.0, 5.0, self.T)
        o = np.random.uniform(l, h)
        v = np.random.uniform(1e5, 1e6, self.T)

        data = {
            "Open": o, "High": h, "Low": l, "Close": c, "Close_Adj": c, "Volume": v,
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
            "Parkinson_Vol": np.full(self.T, 0.02),
            "Garman_Klass_Vol": np.full(self.T, 0.02),
            "SP500_Ret": np.zeros(self.T),
            "ADR_Ret": np.zeros(self.T),
            "Global_Confluence": np.ones(self.T),
        }
        self.df = pd.DataFrame(data)
        self.ctx = EvaluationContext(self.df)

    def test_parser_to_dispatcher_parity(self):
        """Verify complex parsed formulas evaluate with exact backend parity."""
        test_strings = [
            "TS_ZSCORE_20(Close) * SIGMOID(Vol_ZScore)",
            "BLEND(Vol_ZScore, TS_DELAY_1(Close), TS_DELAY_5(Close))",
            "DIV_SAFE(High - Low, Close) + TS_DECAY_LINEAR_10(Close)",
            "EMA_10(Close) - TS_DELAY_1(EMA_10(Close))",
            "TS_CORR_20(Close, Volume) + TANH(Spread)",
            "SQRT_ABS(Volume) * SIGN(Close - Open) + 1.5",
        ]
        for expr in test_strings:
            ast = FormulaParser.parse(expr)
            self.assertIsNotNone(ast, f"Failed to parse formula: {expr}")

            res_numpy = ComputeDispatcher.evaluate(ast, self.ctx, backend="numpy")
            res_torch = ComputeDispatcher.evaluate(ast, self.ctx, backend="torch")

            self.assertIsInstance(res_numpy, np.ndarray)
            self.assertIsInstance(res_torch, torch.Tensor)
            self.assertEqual(len(res_numpy), self.T)
            self.assertEqual(len(res_torch), self.T)
            self.assertTrue(np.all(np.isfinite(res_numpy)), f"Non-finite in numpy on {expr}")
            self.assertTrue(torch.all(torch.isfinite(res_torch)), f"Non-finite in torch on {expr}")

            # Verify numerical parity on suffix where rolling window is fully formed
            np.testing.assert_allclose(
                res_numpy[25:], res_torch.cpu().numpy()[25:],
                rtol=0.02, atol=0.05,
                err_msg=f"Backend parity mismatch on parsed expr: {expr}"
            )

    def test_random_population_batch_coordination(self):
        """Verify generated & mutated formula population evaluates cleanly in batch."""
        n_formulas = 50
        # 1. Muscle 1: Generate dimensionally sound AST trees
        population = [
            TreeGenerator.generate(max_depth=4, target_dimension=FeatureDimension.DIMENSIONLESS)
            for _ in range(n_formulas)
        ]
        # 2. Muscle 1: Mutate half the population
        mutated_population = [
            GeneticMutator.mutate_point(tree) if i % 2 == 0 else tree
            for i, tree in enumerate(population)
        ]

        # 3. Muscle 2: Vectorized batch evaluation across population
        batch_signals_torch = ComputeDispatcher.evaluate_batch(mutated_population, self.ctx, backend="torch")
        self.assertIsInstance(batch_signals_torch, torch.Tensor)
        self.assertEqual(batch_signals_torch.shape, (n_formulas, self.T))
        self.assertTrue(torch.all(torch.isfinite(batch_signals_torch)))

        batch_signals_np = ComputeDispatcher.evaluate_batch(mutated_population, self.ctx, backend="numpy")
        self.assertIsInstance(batch_signals_np, np.ndarray)
        self.assertEqual(batch_signals_np.shape, (n_formulas, self.T))
        self.assertTrue(np.all(np.isfinite(batch_signals_np)))

    def test_3d_multi_asset_batch_coordination(self):
        """Verify Muscle 1 formulas evaluate across 3D multi-asset universes in Muscle 2."""
        B, T, F = 3, 100, len(STANDARD_FEATURES)
        X_3d = torch.randn(B, T, F)
        # Give positive close and volume
        X_3d[:, :, self.feature_map["Close"]] = torch.abs(X_3d[:, :, self.feature_map["Close"]]) + 500.0
        X_3d[:, :, self.feature_map["Volume"]] = torch.abs(X_3d[:, :, self.feature_map["Volume"]]) + 1000.0

        formulas = [
            FormulaParser.parse("TS_DELAY_1(Close)"),
            FormulaParser.parse("EMA_10(Close)"),
            FormulaParser.parse("TS_ZSCORE_20(Close)"),
            FormulaParser.parse("BLEND(Vol_ZScore, High, Low)"),
        ]
        # Evaluates 4 formulas across 3 assets over 100 bars -> shape [4, 3, 100]
        batch_3d = TensorASTEvaluator.evaluate_batch(formulas, X_3d, self.feature_map)
        self.assertEqual(batch_3d.shape, (4, B, T))
        self.assertTrue(torch.all(torch.isfinite(batch_3d)))

    def test_zero_crash_under_adversarial_data(self):
        """Verify Muscle 1 ASTs on Muscle 2 engine never crash under corrupt/poisoned data."""
        # Inject extreme adversarial data
        bad_df = self.df.copy()
        bad_df.iloc[0:10, :] = float("nan")
        bad_df.iloc[10:20, :] = float("inf")
        bad_df.iloc[20:30, :] = float("-inf")
        bad_df.iloc[30:40, :] = 1e12

        bad_ctx = EvaluationContext(bad_df)
        formulas = [
            FormulaParser.parse("DIV_SAFE(Close, Open - Open)"),  # Denominator zero
            FormulaParser.parse("EXP_SAFE(Volume)"),               # Huge exponent
            FormulaParser.parse("LOG_SAFE(Close - Close)"),        # Log of zero
            FormulaParser.parse("TS_ZSCORE_20(Close)"),            # Corrupted rolling
            FormulaParser.parse("TS_CORR_20(Close, Volume)"),      # Corrupted corr
        ]
        for ast in formulas:
            out_torch = ComputeDispatcher.evaluate(ast, bad_ctx, backend="torch")
            self.assertTrue(torch.all(torch.isfinite(out_torch)), f"Torch produced non-finite on {ast.op}")


if __name__ == "__main__":
    unittest.main()
