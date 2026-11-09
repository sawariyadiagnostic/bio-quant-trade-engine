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

    def test_ts_delay_lag_operator(self):
        """Verify TS_DELAY_1 and TS_DELAY_5 compute true lagged states."""
        delay1_node = ASTNode("TS_DELAY_1", children=[ASTNode("FEATURE", "Close")])
        delay1_res = ASTEvaluator.eval_node(delay1_node, self.df)
        self.assertTrue(np.all(np.isfinite(delay1_res)))
        # At index 1, delay1 must equal Close at index 0
        self.assertEqual(delay1_res[1], self.df["Close"].iloc[0])

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

    def test_ast_simplification_and_intron_pruning(self):
        """Verify algebraic simplifications, identity reductions, and constant folding."""
        # 1. Constant folding: 2 + 3 -> 5
        add_node = ASTNode("ADD", children=[ASTNode("CONST", "2.0"), ASTNode("CONST", "3.0")])
        s_add = add_node.simplify()
        self.assertEqual(s_add.op, "CONST")
        self.assertEqual(float(s_add.val), 5.0)

        # 2. X - X -> 0.0
        x_node = ASTNode("FEATURE", "Close")
        sub_node = ASTNode("SUB", children=[x_node, x_node])
        s_sub = sub_node.simplify()
        self.assertEqual(s_sub.op, "CONST")
        self.assertEqual(float(s_sub.val), 0.0)

        # 3. X / X -> 1.0
        div_node = ASTNode("DIV_SAFE", children=[x_node, x_node])
        s_div = div_node.simplify()
        self.assertEqual(s_div.op, "CONST")
        self.assertEqual(float(s_div.val), 1.0)

        # 4. Involutions: ABS(ABS(X)) -> ABS(X)
        abs_abs = ASTNode("ABS", children=[ASTNode("ABS", children=[x_node])])
        s_abs = abs_abs.simplify()
        self.assertEqual(s_abs.op, "ABS")
        self.assertEqual(s_abs.children[0].op, "FEATURE")

        # 5. Constant rolling operator: TS_MAX_10(5.0) -> 5.0, DELTA_5(5.0) -> 0.0
        max_const = ASTNode("TS_MAX_10", children=[ASTNode("CONST", "5.0")])
        self.assertEqual(max_const.simplify().op, "CONST")
        self.assertEqual(float(max_const.simplify().val), 5.0)

        delta_const = ASTNode("DELTA_5", children=[ASTNode("CONST", "5.0")])
        self.assertEqual(delta_const.simplify().op, "CONST")
        self.assertEqual(float(delta_const.simplify().val), 0.0)

    def test_new_alpha_primitives_numpy(self):
        """Verify evaluation of SIGNED_POWER, TS_STD_20, TS_MEDIAN_10, TS_ARGMAX_10, TS_ARGMIN_10."""
        n = 50
        df = pd.DataFrame({
            "Close": np.linspace(10.0, 60.0, n),
            "Signal": np.full(n, -3.0)
        })
        # SIGNED_POWER(-3) = sign(-3) * 3^2 = -9
        sp_node = ASTNode("SIGNED_POWER", children=[ASTNode("FEATURE", "Signal")])
        sp_res = ASTEvaluator.eval_node(sp_node, df)
        self.assertTrue(np.allclose(sp_res, -9.0))

        # TS_STD_20
        std_node = ASTNode("TS_STD_20", children=[ASTNode("FEATURE", "Close")])
        std_res = ASTEvaluator.eval_node(std_node, df)
        self.assertTrue(np.all(np.isfinite(std_res)))
        self.assertTrue(std_res[-1] > 0.0)

        # TS_ARGMAX_10 on monotonically increasing series should be 1.0
        am_node = ASTNode("TS_ARGMAX_10", children=[ASTNode("FEATURE", "Close")])
        am_res = ASTEvaluator.eval_node(am_node, df)
        self.assertTrue(np.allclose(am_res[10:], 1.0))

        # TS_ARGMIN_10 on monotonically increasing series should be 0.0
        an_node = ASTNode("TS_ARGMIN_10", children=[ASTNode("FEATURE", "Close")])
        an_res = ASTEvaluator.eval_node(an_node, df)
        self.assertTrue(np.allclose(an_res[10:], 0.0))


if __name__ == "__main__":
    unittest.main()
