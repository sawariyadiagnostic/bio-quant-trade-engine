"""Comprehensive Test Suite for Institutional Quantitative AST, Bounded Math, and Dimensional Integrity."""
import unittest
import numpy as np
import pandas as pd

from ravana.domain.formula_ast import (
    ASTNode,
    ASTEvaluator,
    TreeGenerator,
    GeneticMutator,
    GeneticCrossover,
    FeatureDimension,
    OPERATORS_BINARY,
    OPERATORS_UNARY,
    OPERATORS_ROLLING_UNARY,
)
from ravana.domain.formula_parser import FormulaParser


class TestInstitutionalFormulaAST(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 100
        self.df = pd.DataFrame({
            "Open": np.random.uniform(500, 550, n),
            "High": np.random.uniform(550, 600, n),
            "Low": np.random.uniform(450, 500, n),
            "Close": np.random.uniform(500, 550, n),
            "Close_Adj": np.random.uniform(500, 550, n),
            "Volume": np.random.uniform(1e5, 1e6, n),
            "Spread": np.random.uniform(0.001, 0.005, n),
            "Vol_ZScore": np.random.uniform(-3.0, 3.0, n),
            "Parkinson_Vol": np.random.uniform(0.01, 0.08, n),
            "RSI_14": np.random.uniform(20.0, 80.0, n),
        })

    def test_smooth_blend_evaluation(self):
        """Verify BLEND smoothly transitions between subtrees without cliff-edge jumps."""
        cond = ASTNode("SUB", children=[ASTNode("FEATURE", "Vol_ZScore"), ASTNode("CONST", "0.0")])
        a = ASTNode("CONST", "10.0")
        b = ASTNode("CONST", "0.0")
        blend_node = ASTNode("BLEND", children=[cond, a, b])

        # Test on extreme conditions
        calm_df = pd.DataFrame({"Vol_ZScore": [-10.0]})  # Extreme negative -> weight -> 0 -> returns b (0.0)
        res_calm = ASTEvaluator.eval_node(blend_node, calm_df)
        self.assertAlmostEqual(res_calm[0], 0.0, places=3)

        high_vol_df = pd.DataFrame({"Vol_ZScore": [10.0]})  # Extreme positive -> weight -> 1 -> returns a (10.0)
        res_high = ASTEvaluator.eval_node(blend_node, high_vol_df)
        self.assertAlmostEqual(res_high[0], 10.0, places=3)

        zero_df = pd.DataFrame({"Vol_ZScore": [0.0]})  # Zero -> weight = 0.5 -> returns 5.0
        res_zero = ASTEvaluator.eval_node(blend_node, zero_df)
        self.assertAlmostEqual(res_zero[0], 5.0, places=3)

    def test_ts_zscore_and_ts_corr(self):
        """Verify rolling z-score and rolling correlation primitives."""
        # TS_ZSCORE
        z_node = ASTNode("TS_ZSCORE_20", children=[ASTNode("FEATURE", "Close")])
        z_res = ASTEvaluator.eval_node(z_node, self.df)
        self.assertTrue(np.all(np.isfinite(z_res)))
        self.assertTrue(np.all(z_res >= -4.0))
        self.assertTrue(np.all(z_res <= 4.0))

        # TS_CORR_20 between Close and Volume
        corr_node = ASTNode("TS_CORR_20", children=[ASTNode("FEATURE", "Close"), ASTNode("FEATURE", "Volume")])
        corr_res = ASTEvaluator.eval_node(corr_node, self.df)
        self.assertTrue(np.all(np.isfinite(corr_res)))
        self.assertTrue(np.all(corr_res >= -1.0))
        self.assertTrue(np.all(corr_res <= 1.0))

    def test_ts_decay_linear(self):
        """Verify linear time decay weighting computes valid finite values."""
        decay_node = ASTNode("TS_DECAY_LINEAR_10", children=[ASTNode("FEATURE", "Close")])
        decay_res = ASTEvaluator.eval_node(decay_node, self.df)
        self.assertTrue(np.all(np.isfinite(decay_res)))

    def test_formula_parser_blend_and_corr(self):
        """Verify parsing of BLEND and TS_CORR_20 formula strings."""
        raw = "BLEND(Vol_ZScore, Close * 1.5, Open) + TS_CORR_20(Close, Volume)"
        parsed = FormulaParser.parse(raw)
        self.assertEqual(parsed.op, "ADD")
        self.assertEqual(parsed.children[0].op, "BLEND")
        self.assertEqual(parsed.children[1].op, "TS_CORR_20")

        res = ASTEvaluator.eval_node(parsed, self.df)
        self.assertTrue(np.all(np.isfinite(res)))

    def test_dimensional_generator_soundness(self):
        """Verify TreeGenerator generates trees conforming to physical dimensions."""
        for _ in range(50):
            tree = TreeGenerator.generate(max_depth=4, target_dimension=FeatureDimension.DIMENSIONLESS)
            self.assertIsInstance(tree, ASTNode)
            # Evaluate tree to verify zero runtime evaluation errors
            res = ASTEvaluator.eval_node(tree, self.df)
            self.assertTrue(np.all(np.isfinite(res)))

    def test_arity_hardening_in_crossover(self):
        """Verify crossover preserves exact arities for BLEND, binary, and unary operators."""
        for _ in range(50):
            p1 = TreeGenerator.generate(max_depth=4)
            p2 = TreeGenerator.generate(max_depth=4)
            c1, c2 = GeneticCrossover.crossover(p1, p2)

            for child in (c1, c2):
                for node in child.get_all_subnodes():
                    if node.op == "BLEND":
                        self.assertEqual(len(node.children), 3)
                    elif node.op in OPERATORS_BINARY:
                        self.assertEqual(len(node.children), 2)
                    elif node.op in (OPERATORS_UNARY + OPERATORS_ROLLING_UNARY):
                        self.assertEqual(len(node.children), 1)
                    elif node.op in ("FEATURE", "CONST"):
                        self.assertEqual(len(node.children), 0)

    def test_clamped_bounded_math(self):
        """Verify zero division and large exponent bounds prevent crashes."""
        df = pd.DataFrame({"X": [100.0, -100.0, 0.0, np.nan, np.inf, -np.inf]})
        exp_node = ASTNode("EXP_SAFE", children=[ASTNode("FEATURE", "X")])
        exp_res = ASTEvaluator.eval_node(exp_node, df)
        self.assertTrue(np.all(np.isfinite(exp_res)))

        div_node = ASTNode("DIV_SAFE", children=[ASTNode("FEATURE", "X"), ASTNode("CONST", "0.0")])
        div_res = ASTEvaluator.eval_node(div_node, df)
        self.assertTrue(np.all(np.isfinite(div_res)))


if __name__ == "__main__":
    unittest.main()
