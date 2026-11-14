"""Unit tests for Phase B Cross-Sectional Operators and Indian Asset Universe Expansion."""
import unittest
import numpy as np
import pandas as pd
import torch

from ravana.compute.primitives import cs_rank, cs_zscore
from ravana.domain.formula_ast import (
    ASTNode,
    ASTEvaluator,
    OPERATORS_CROSS_SECTIONAL,
)
from ravana.domain.formula_parser import FormulaParser
from ravana.compute.tensor_evaluator import TensorASTEvaluator
from ravana.compute.evaluator import TensorASTEvaluator as AliasedEvaluator
from ravana.data.universe.registry import AssetRegistry
from ravana.data.universe.models import AssetClass


class TestCrossSectionalPrimitives(unittest.TestCase):
    """Test suite for vectorized PyTorch cross-sectional operators."""

    def test_cs_rank_1d_and_2d(self):
        """1D and 2D single-asset tensors gracefully fall back to ts_rank_20 (non-degenerate)."""
        t_1d = torch.randn(50)
        res_1d = cs_rank(t_1d)
        self.assertEqual(res_1d.shape, t_1d.shape)
        self.assertTrue(torch.all(torch.isfinite(res_1d)))
        self.assertTrue(torch.all(res_1d >= 0.0) and torch.all(res_1d <= 1.0))
        # Non-degenerate variance: should not be flat 0.5
        self.assertGreater(float(torch.std(res_1d)), 0.05)

        t_2d = torch.randn(50, 10)
        res_2d = cs_rank(t_2d)
        self.assertEqual(res_2d.shape, t_2d.shape)
        self.assertTrue(torch.all(torch.isfinite(res_2d)))
        self.assertTrue(torch.all(res_2d >= 0.0) and torch.all(res_2d <= 1.0))
        self.assertGreater(float(torch.std(res_2d)), 0.05)

    def test_cs_rank_3d_single_asset(self):
        """3D tensor with B == 1 (single asset) falls back to ts_rank_20."""
        t_3d_single = torch.randn(1, 40, 5)
        res = cs_rank(t_3d_single)
        self.assertEqual(res.shape, (1, 40, 5))
        self.assertTrue(torch.all(torch.isfinite(res)))
        self.assertTrue(torch.all(res >= 0.0) and torch.all(res <= 1.0))
        self.assertGreater(float(torch.std(res)), 0.05)

    def test_cs_rank_3d_multi_asset(self):
        """3D tensor with B > 1 ranks across dim 0 into [0.0, 1.0]."""
        # B=3, T=1, K=1 with values 10, 30, 20
        t = torch.tensor([[[10.0]], [[30.0]], [[20.0]]], dtype=torch.float32)
        res = cs_rank(t)
        self.assertEqual(res.shape, (3, 1, 1))
        # 10 is min (rank 0 / 2 = 0.0), 30 is max (rank 2 / 2 = 1.0), 20 is mid (rank 1 / 2 = 0.5)
        self.assertAlmostEqual(float(res[0, 0, 0]), 0.0, places=4)
        self.assertAlmostEqual(float(res[1, 0, 0]), 1.0, places=4)
        self.assertAlmostEqual(float(res[2, 0, 0]), 0.5, places=4)

    def test_cs_rank_clamping_and_nan_safety(self):
        """CS_RANK clamps to [0.0, 1.0] and replaces NaNs with 0.5."""
        t = torch.tensor([[[float("nan")]], [[1.0]], [[2.0]]], dtype=torch.float32)
        res = cs_rank(t)
        self.assertTrue(torch.all(torch.isfinite(res)))
        self.assertTrue(torch.all(res >= 0.0))
        self.assertTrue(torch.all(res <= 1.0))

    def test_cs_zscore_1d_and_2d(self):
        """1D and 2D single-asset tensors gracefully fall back to ts_zscore_20 (non-degenerate)."""
        t_1d = torch.randn(50)
        res_1d = cs_zscore(t_1d)
        self.assertEqual(res_1d.shape, t_1d.shape)
        self.assertTrue(torch.all(torch.isfinite(res_1d)))
        self.assertTrue(torch.all(res_1d >= -4.0) and torch.all(res_1d <= 4.0))
        self.assertGreater(float(torch.std(res_1d)), 0.1)

        t_2d = torch.randn(50, 10)
        res_2d = cs_zscore(t_2d)
        self.assertEqual(res_2d.shape, t_2d.shape)
        self.assertTrue(torch.all(torch.isfinite(res_2d)))
        self.assertTrue(torch.all(res_2d >= -4.0) and torch.all(res_2d <= 4.0))
        self.assertGreater(float(torch.std(res_2d)), 0.1)

    def test_cs_zscore_3d_single_asset(self):
        """3D tensor with B == 1 gracefully falls back to ts_zscore_20."""
        t_3d_single = torch.randn(1, 40, 5)
        res = cs_zscore(t_3d_single)
        self.assertEqual(res.shape, (1, 40, 5))
        self.assertTrue(torch.all(torch.isfinite(res)))
        self.assertTrue(torch.all(res >= -4.0) and torch.all(res <= 4.0))
        self.assertGreater(float(torch.std(res)), 0.1)

    def test_cs_zscore_3d_multi_asset(self):
        """3D tensor with B > 1 computes standardized z-score across dim 0."""
        # B=3, T=1, K=1 with values 10, 20, 30: mean=20, std=10
        t = torch.tensor([[[10.0]], [[20.0]], [[30.0]]], dtype=torch.float32)
        res = cs_zscore(t)
        self.assertEqual(res.shape, (3, 1, 1))
        # z for 20 is 0.0
        self.assertAlmostEqual(float(res[1, 0, 0]), 0.0, places=4)
        # z for 10 is -1.0, z for 30 is +1.0
        self.assertAlmostEqual(float(res[0, 0, 0]), -1.0, places=3)
        self.assertAlmostEqual(float(res[2, 0, 0]), 1.0, places=3)

    def test_cs_zscore_clamping_and_nan_safety(self):
        """CS_ZSCORE clamps to [-3.0, 3.0] and replaces NaNs with 0.0."""
        t = torch.tensor([[[float("nan")]], [[1e10]], [[-1e10]]], dtype=torch.float32)
        res = cs_zscore(t)
        self.assertTrue(torch.all(torch.isfinite(res)))
        self.assertTrue(torch.all(res >= -3.0))
        self.assertTrue(torch.all(res <= 3.0))


class TestFormulaASTCrossSectional(unittest.TestCase):
    """Test suite for AST grammar, parsing, and evaluation of CS operators."""

    def test_operators_registered(self):
        """Verify CS_RANK and CS_ZSCORE are in OPERATORS_CROSS_SECTIONAL."""
        self.assertIn("CS_RANK", OPERATORS_CROSS_SECTIONAL)
        self.assertIn("CS_ZSCORE", OPERATORS_CROSS_SECTIONAL)

    def test_parsing_and_string_formatting(self):
        """Verify formula parsing and string representation."""
        p_rank = FormulaParser.parse("CS_RANK(Close)")
        self.assertEqual(p_rank.op, "CS_RANK")
        self.assertEqual(len(p_rank.children), 1)
        self.assertEqual(p_rank.to_formula_str(), "CS_RANK(Close)")

        p_zscore = FormulaParser.parse("CS_ZSCORE(Close)")
        self.assertEqual(p_zscore.op, "CS_ZSCORE")
        self.assertEqual(len(p_zscore.children), 1)
        self.assertEqual(p_zscore.to_formula_str(), "CS_ZSCORE(Close)")

    def test_constant_simplification(self):
        """Verify constant folding for CS operators."""
        node_rank = ASTNode("CS_RANK", children=[ASTNode("CONST", "100.0")]).simplify()
        self.assertEqual(node_rank.op, "CONST")
        self.assertEqual(node_rank.val, "0.5")

        node_zscore = ASTNode("CS_ZSCORE", children=[ASTNode("CONST", "100.0")]).simplify()
        self.assertEqual(node_zscore.op, "CONST")
        self.assertEqual(node_zscore.val, "0.0")

    def test_ast_evaluator_single_asset_df(self):
        """Single-asset DataFrame evaluates CS_RANK to 0.5 and CS_ZSCORE to 0.0."""
        df = pd.DataFrame({"Close": [10.0, 20.0, 30.0, 40.0, 50.0]})
        node_rank = FormulaParser.parse("CS_RANK(Close)")
        node_zscore = FormulaParser.parse("CS_ZSCORE(Close)")

        res_rank = ASTEvaluator.eval_node(node_rank, df)
        self.assertEqual(len(res_rank), 5)
        self.assertTrue(np.all(res_rank == 0.5))

        res_zscore = ASTEvaluator.eval_node(node_zscore, df)
        self.assertEqual(len(res_zscore), 5)
        self.assertTrue(np.all(res_zscore == 0.0))

    def test_ast_evaluator_2d_numpy(self):
        """NumPy 2D array [T, K] (single asset) returns 0.5 for rank and 0.0 for zscore."""
        arr_2d = np.random.randn(50, 6)
        node_rank = FormulaParser.parse("CS_RANK(Close)")
        node_zscore = FormulaParser.parse("CS_ZSCORE(Close)")

        res_rank = ASTEvaluator.eval_node(node_rank, arr_2d)
        self.assertEqual(len(res_rank), 50)
        self.assertTrue(np.all(res_rank == 0.5))

        res_zscore = ASTEvaluator.eval_node(node_zscore, arr_2d)
        self.assertEqual(len(res_zscore), 50)
        self.assertTrue(np.all(res_zscore == 0.0))

    def test_ast_evaluator_multi_asset_panel(self):
        """Multi-asset panel DataFrame calculates cross-sectional rank/zscore across columns."""
        panel_df = pd.DataFrame({
            "ASSET_A": [10.0, 100.0],
            "ASSET_B": [30.0, 200.0],
            "ASSET_C": [20.0, 300.0],
        })
        panel_df.attrs["is_multi_asset"] = True

        node_rank = ASTNode("CS_RANK", children=[ASTNode("FEATURE", "Close")])
        res_rank = ASTEvaluator.eval_node(node_rank, panel_df)
        self.assertEqual(res_rank.shape, (2, 3))
        # Row 0: ASSET_A (10.0) is rank 0.0, ASSET_B (30.0) is rank 1.0, ASSET_C (20.0) is rank 0.5
        np.testing.assert_allclose(res_rank[0], [0.0, 1.0, 0.5], atol=1e-4)

        node_zscore = ASTNode("CS_ZSCORE", children=[ASTNode("FEATURE", "Close")])
        res_zscore = ASTEvaluator.eval_node(node_zscore, panel_df)
        self.assertEqual(res_zscore.shape, (2, 3))
        self.assertTrue(np.all(np.isfinite(res_zscore)))


class TestTensorEvaluatorCrossSectional(unittest.TestCase):
    """Test suite for PyTorch TensorASTEvaluator cross-sectional execution."""

    def setUp(self):
        self.feature_map = {"Close": 0, "Volume": 1}
        self.T = 20
        self.K = 2
        # 2D tensor: [T=20, K=2]
        self.X_2d = torch.randn(self.T, self.K)
        # 3D tensor: [B=4, T=20, K=2]
        self.X_3d = torch.randn(4, self.T, self.K)

    def test_aliased_import(self):
        """TensorASTEvaluator can be imported from ravana.compute.evaluator."""
        self.assertIs(TensorASTEvaluator, AliasedEvaluator)

    def test_2d_single_asset_tensor_evaluation(self):
        """2D tensor evaluates CS_RANK to 0.5 and CS_ZSCORE to 0.0."""
        ast_rank = FormulaParser.parse("CS_RANK(Close)")
        res_rank = TensorASTEvaluator.eval_node(ast_rank, self.X_2d, self.feature_map)
        self.assertEqual(res_rank.shape, (self.T,))
        self.assertTrue(torch.all(res_rank == 0.5))

        ast_zscore = FormulaParser.parse("CS_ZSCORE(Close)")
        res_zscore = TensorASTEvaluator.eval_node(ast_zscore, self.X_2d, self.feature_map)
        self.assertEqual(res_zscore.shape, (self.T,))
        self.assertTrue(torch.all(res_zscore == 0.0))

    def test_3d_multi_asset_tensor_evaluation(self):
        """3D tensor evaluates CS_RANK and CS_ZSCORE across batch dimension."""
        ast_rank = FormulaParser.parse("CS_RANK(Close)")
        res_rank = TensorASTEvaluator.eval_node(ast_rank, self.X_3d, self.feature_map)
        self.assertEqual(res_rank.shape, (4, self.T))
        self.assertTrue(torch.all(torch.isfinite(res_rank)))
        self.assertTrue(torch.all(res_rank >= 0.0))
        self.assertTrue(torch.all(res_rank <= 1.0))

        ast_zscore = FormulaParser.parse("CS_ZSCORE(Close)")
        res_zscore = TensorASTEvaluator.eval_node(ast_zscore, self.X_3d, self.feature_map)
        self.assertEqual(res_zscore.shape, (4, self.T))
        self.assertTrue(torch.all(torch.isfinite(res_zscore)))
        self.assertTrue(torch.all(res_zscore >= -3.0))
        self.assertTrue(torch.all(res_zscore <= 3.0))


class TestIndianAssetCatalog(unittest.TestCase):
    """Test suite verifying the top 10 liquid Indian ETFs in asset catalog."""

    def setUp(self):
        self.registry = AssetRegistry()

    def test_top_10_liquid_indian_etfs_present(self):
        """Verify all 10 liquid Indian ETFs are in the catalog with lot_size=1 and ETF class."""
        expected_etfs = {
            "BANKBEES.NS": ("Banking", 1),
            "NIFTYBEES.NS": ("Broad Market Nifty 50", 1),
            "ITBEES.NS": ("Technology", 1),
            "GOLDBEES.NS": ("Commodity Gold", 1),
            "SILVERBEES.NS": ("Commodity Silver", 1),
            "PHARMABEES.NS": ("Healthcare", 1),
            "AUTOBEES.NS": ("Automobile", 1),
            "JUNIORBEES.NS": ("Next 50", 1),
            "PSUBNKBEES.NS": ("PSU Banks", 1),
            "CPSEETF.NS": ("Public Sector Undertakings", 1),
        }

        for symbol, (expected_sector, expected_lot) in expected_etfs.items():
            inst = self.registry.get(symbol)
            self.assertIsNotNone(inst, f"Symbol {symbol} missing from asset catalog")
            self.assertEqual(inst.asset_class, AssetClass.ETF, f"{symbol} must have asset_class ETF")
            self.assertEqual(inst.lot_size, expected_lot, f"{symbol} lot_size must be {expected_lot}")
            self.assertEqual(inst.sector, expected_sector, f"{symbol} sector must be {expected_sector}")

    def test_all_etfs_basket_contains_all_10(self):
        """Verify all_etfs basket contains all 10 ETFs."""
        all_etfs = self.registry.get_basket("all_etfs")
        symbols = {i.symbol for i in all_etfs}
        required_10 = {
            "BANKBEES.NS", "NIFTYBEES.NS", "ITBEES.NS", "GOLDBEES.NS",
            "SILVERBEES.NS", "PHARMABEES.NS", "AUTOBEES.NS", "JUNIORBEES.NS",
            "PSUBNKBEES.NS", "CPSEETF.NS"
        }
        self.assertTrue(required_10.issubset(symbols), f"Missing from all_etfs: {required_10 - symbols}")


if __name__ == "__main__":
    unittest.main()
