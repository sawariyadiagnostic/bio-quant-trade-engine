"""Comprehensive Test Suite for RAVANA Formula AST, Bounded Math, MegaEvolution, and Parsing."""
import unittest
import numpy as np
import pandas as pd

from ravana.domain.formula_ast import (
    ASTNode,
    MegaEvolutionNode,
    ASTEvaluator,
    TreeGenerator,
    GeneticMutator,
    GeneticCrossover,
    cs_rank,
    cs_demean,
    OPERATORS_BINARY,
    OPERATORS_UNARY,
    OPERATORS_ROLLING,
    OPERATORS_CROSS_SECTIONAL,
)
from ravana.domain.formula_parser import FormulaParser


class TestFormulaASTComprehensive(unittest.TestCase):
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
            "Vol_ZScore": np.random.uniform(-3.0, 3.0, n),
            "Parkinson_Vol": np.random.uniform(0.01, 0.08, n),
        })

    def test_mega_evolution_ast_evaluation(self):
        """Verify MegaEvolutionNode dynamically switches between base and mega forms."""
        cond = ASTNode("SUB", children=[ASTNode("FEATURE", "Vol_ZScore"), ASTNode("CONST", "2.0")])
        base = ASTNode("FEATURE", "Close_Adj")
        mega = ASTNode("MUL", children=[ASTNode("FEATURE", "Parkinson_Vol"), ASTNode("CONST", "10.0")])

        mega_node = MegaEvolutionNode(condition=cond, base=base, mega=mega)

        df_calm = pd.DataFrame({
            "Vol_ZScore": [0.5, 1.2, 1.8],
            "Close_Adj": [250.0, 252.0, 255.0],
            "Parkinson_Vol": [0.05, 0.06, 0.07]
        })
        # Condition <= 0 -> evaluates base (Close_Adj)
        res_calm = ASTEvaluator.eval_node(mega_node, df_calm)
        np.testing.assert_allclose(res_calm, [250.0, 252.0, 255.0])

        df_crash = pd.DataFrame({
            "Vol_ZScore": [2.5, 3.1, 4.0],
            "Close_Adj": [240.0, 230.0, 220.0],
            "Parkinson_Vol": [0.2, 0.3, 0.4]
        })
        # Condition > 0 -> evaluates mega (Parkinson_Vol * 10)
        res_crash = ASTEvaluator.eval_node(mega_node, df_crash)
        np.testing.assert_allclose(res_crash, [2.0, 3.0, 4.0])

    def test_mega_evolution_formula_string_and_parsing(self):
        """Verify MegaEvolution formula serializes to string and parses back to identical AST."""
        cond = ASTNode("FEATURE", "Vol_ZScore")
        base = ASTNode("FEATURE", "Close_Adj")
        mega = ASTNode("FEATURE", "Parkinson_Vol")
        mega_node = MegaEvolutionNode(condition=cond, base=base, mega=mega)

        formula_str = mega_node.to_formula_str()
        self.assertIn("MEGA_SWITCH", formula_str)

        parsed_node = FormulaParser.parse(formula_str)
        self.assertEqual(parsed_node.op, "MEGA_SWITCH")
        self.assertEqual(len(parsed_node.children), 3)
        self.assertEqual(parsed_node.children[0].val, "Vol_ZScore")
        self.assertEqual(parsed_node.children[1].val, "Close_Adj")
        self.assertEqual(parsed_node.children[2].val, "Parkinson_Vol")

    def test_formula_parser_complex_expressions(self):
        """Test parsing infix arithmetic, nested parentheses, and scientific constants."""
        raw = "((Close - Open) / 1.5e-2) + (High * 2.0)"
        parsed = FormulaParser.parse(raw)
        self.assertEqual(parsed.op, "ADD")
        self.assertEqual(parsed.children[0].op, "DIV_SAFE")
        self.assertEqual(parsed.children[1].op, "MUL")

        # Evaluate parsed expression
        res = ASTEvaluator.eval_node(parsed, self.df)
        self.assertTrue(np.all(np.isfinite(res)))

    def test_ast_canonical_hash_and_bounded_math(self):
        """Verify ASTNode canonical hash commutativity and bounded math clamping."""
        node1 = ASTNode("ADD", children=[ASTNode("FEATURE", "A"), ASTNode("FEATURE", "B")])
        node2 = ASTNode("ADD", children=[ASTNode("FEATURE", "B"), ASTNode("FEATURE", "A")])
        self.assertEqual(node1.canonical_hash(), node2.canonical_hash())

        sub1 = ASTNode("SUB", children=[ASTNode("FEATURE", "A"), ASTNode("FEATURE", "B")])
        sub2 = ASTNode("SUB", children=[ASTNode("FEATURE", "B"), ASTNode("FEATURE", "A")])
        self.assertNotEqual(sub1.canonical_hash(), sub2.canonical_hash())

        # Math clamping
        df = pd.DataFrame({"X": [100.0, -100.0, 0.0, np.nan, np.inf, -np.inf]})
        exp_node = ASTNode("EXP_SAFE", children=[ASTNode("FEATURE", "X")])
        exp_res = ASTEvaluator.eval_node(exp_node, df)
        self.assertTrue(np.all(np.isfinite(exp_res)))
        self.assertTrue(exp_res[0] <= np.exp(20.0) + 1.0)

        log_node = ASTNode("LOG_SAFE", children=[ASTNode("FEATURE", "X")])
        log_res = ASTEvaluator.eval_node(log_node, df)
        self.assertTrue(np.all(np.isfinite(log_res)))

        div_zero_node = ASTNode("DIV_SAFE", children=[ASTNode("FEATURE", "X"), ASTNode("CONST", "0.0")])
        div_res = ASTEvaluator.eval_node(div_zero_node, df)
        self.assertTrue(np.all(np.isfinite(div_res)))

    def test_cross_sectional_operators(self):
        """Verify CS_Rank and CS_Demean compute cross-sectional alpha across multi-asset slices."""
        raw_slice = np.array([
            [10.0, 20.0, 30.0, 40.0],
            [20.0, 10.0, 10.0, 50.0],
            [30.0, 30.0, 20.0, 10.0]
        ], dtype=np.float64)

        ranked = cs_rank(raw_slice)
        self.assertEqual(ranked.shape, (3, 4))
        self.assertAlmostEqual(ranked[0, 0], 0.5 / 3.0)
        self.assertAlmostEqual(ranked[1, 0], 1.5 / 3.0)
        self.assertAlmostEqual(ranked[2, 0], 2.5 / 3.0)

        demeaned = cs_demean(raw_slice)
        self.assertEqual(demeaned.shape, (3, 4))
        cs_sums = np.sum(demeaned, axis=0)
        np.testing.assert_allclose(cs_sums, [0.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_arity_hardening_in_crossover(self):
        """Verify crossover never leaves binary or unary nodes with invalid child counts."""
        for _ in range(50):
            p1 = TreeGenerator.generate(max_depth=4)
            p2 = TreeGenerator.generate(max_depth=4)
            c1, c2 = GeneticCrossover.crossover(p1, p2)

            for child in (c1, c2):
                for node in child.get_all_subnodes():
                    if node.op in OPERATORS_BINARY:
                        self.assertEqual(len(node.children), 2, f"Binary op {node.op} child count != 2")
                    elif node.op in (OPERATORS_UNARY + OPERATORS_ROLLING + OPERATORS_CROSS_SECTIONAL):
                        self.assertEqual(len(node.children), 1, f"Unary op {node.op} child count != 1")
                    elif node.op in ("FEATURE", "CONST"):
                        self.assertEqual(len(node.children), 0, f"Leaf node {node.op} has children")

    def test_epigenetic_methylation(self):
        """Verify epigenetic mutation toggles active regimes correctly."""
        tree = TreeGenerator.generate(max_depth=3)
        mutated = GeneticMutator.mutate_methylation(tree, num_regimes=10)
        self.assertIsInstance(mutated, ASTNode)
        for node in mutated.get_all_subnodes():
            self.assertTrue(len(node.active_regimes) >= 1)

    def test_dict_serialization_round_trip(self):
        """Verify to_dict and from_dict preserve tree topology and parameters."""
        tree = TreeGenerator.generate(max_depth=3)
        d = tree.to_dict()
        restored = ASTNode.from_dict(d)
        self.assertEqual(tree.canonical_hash(), restored.canonical_hash())
        self.assertEqual(tree.to_formula_str(), restored.to_formula_str())


if __name__ == "__main__":
    unittest.main()
