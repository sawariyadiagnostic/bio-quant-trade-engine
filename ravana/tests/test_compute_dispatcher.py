"""Unit Tests for RAVANA ComputeDispatcher."""
import unittest
import numpy as np
import pandas as pd
import torch

from ravana.domain.formula_ast import ASTNode, STANDARD_FEATURES
from ravana.domain.formula_parser import FormulaParser
from ravana.compute.dispatcher import ComputeDispatcher


class TestComputeDispatcher(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.T = 100
        c = np.random.uniform(500.0, 550.0, self.T)
        h = c + 2.0
        l = c - 2.0
        o = c - 0.5
        v = np.random.uniform(1e5, 1e6, self.T)

        self.df = pd.DataFrame({
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
        })

    def test_auto_dispatch_dataframe(self):
        ast = FormulaParser.parse("TS_CORR_20(Close, Volume) + BLEND(Vol_ZScore, High, Low)")
        res = ComputeDispatcher.evaluate(ast, self.df, backend="auto")
        self.assertIsInstance(res, np.ndarray)
        self.assertEqual(len(res), self.T)
        self.assertTrue(np.all(np.isfinite(res)))

    def test_auto_dispatch_tensor(self):
        mat = np.column_stack([self.df[col].to_numpy(dtype=np.float32) for col in STANDARD_FEATURES])
        tensor_data = torch.from_numpy(mat)

        ast = FormulaParser.parse("TS_DECAY_LINEAR_10(Close) - TS_DELAY_1(Close)")
        res_torch = ComputeDispatcher.evaluate(ast, tensor_data, backend="torch")
        self.assertIsInstance(res_torch, torch.Tensor)
        self.assertEqual(len(res_torch), self.T)
        self.assertTrue(torch.all(torch.isfinite(res_torch)))

    def test_backend_parity(self):
        ast = FormulaParser.parse("TS_ZSCORE_20(Close) * SQRT_ABS(Volume)")
        res_numpy = ComputeDispatcher.evaluate(ast, self.df, backend="numpy")
        res_torch = ComputeDispatcher.evaluate(ast, self.df, backend="torch").cpu().numpy()

        np.testing.assert_allclose(res_numpy[25:], res_torch[25:], rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
