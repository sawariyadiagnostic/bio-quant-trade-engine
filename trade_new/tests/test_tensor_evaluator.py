"""Unit tests for Unified PyTorch Tensor AST Evaluator."""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_tensor_evaluator_2d_and_3d_shapes():
    """Verify TensorASTEvaluator evaluates AST trees on both 2D [T, F] and 3D [B, T, F] tensors."""
    from src.domain.formula_ast import ASTNode
    from src.domain.tensor_evaluator import TensorASTEvaluator

    tree = ASTNode("ADD", children=[ASTNode("FEATURE", "Close_Adj"), ASTNode("FEATURE", "2.5")])

    # 2D case: [T=50, F=20]
    T, F_dim = 50, 20
    X_2d = torch.ones((T, F_dim), dtype=torch.float32)
    res_2d = TensorASTEvaluator.eval_node(tree, X_2d)
    assert res_2d.shape == (T,)
    assert torch.allclose(res_2d, torch.full((T,), 3.5))

    # 3D case: [B=4, T=50, F=20]
    B = 4
    X_3d = torch.ones((B, T, F_dim), dtype=torch.float32)
    res_3d = TensorASTEvaluator.eval_node(tree, X_3d)
    assert res_3d.shape == (B, T)
    assert torch.allclose(res_3d, torch.full((B, T), 3.5))
    print("[PASS] test_tensor_evaluator_2d_and_3d_shapes")


def test_tensor_evaluator_arithmetic_and_zero_nan():
    """Verify arithmetic operators and zero NaN/Inf with safe division and sign retention."""
    from src.domain.formula_ast import ASTNode
    from src.domain.tensor_evaluator import TensorASTEvaluator, div_safe_tensor

    l = torch.tensor([10.0, -10.0, 5.0, 0.0])
    r = torch.tensor([2.0, -2.0, 0.0, 0.0])
    div_res = div_safe_tensor(l, r)

    assert torch.isclose(div_res[0], torch.tensor(5.0))
    assert torch.isclose(div_res[1], torch.tensor(5.0))  # (-10) / (-2) = 5
    assert torch.isclose(div_res[2], torch.tensor(0.0))  # Div by 0 clamped to 0
    assert torch.isclose(div_res[3], torch.tensor(0.0))
    assert torch.all(torch.isfinite(div_res))

    div_node = ASTNode("DIV_SAFE", children=[ASTNode("FEATURE", "A"), ASTNode("FEATURE", "B")])
    X = torch.tensor([[100.0, 0.0], [50.0, -0.0000001], [25.0, 5.0]])
    res = TensorASTEvaluator.eval_node(div_node, X, feature_names=["A", "B"])
    assert torch.all(torch.isfinite(res))
    assert not torch.any(torch.isnan(res))
    print("[PASS] test_tensor_evaluator_arithmetic_and_zero_nan")


def test_tensor_evaluator_unary_operators():
    """Verify unary operators clamp extreme inputs and maintain zero NaN/Inf."""
    from src.domain.formula_ast import ASTNode
    from src.domain.tensor_evaluator import TensorASTEvaluator

    X = torch.tensor([[-50.0], [-1.0], [0.0], [1.0], [50.0]], dtype=torch.float32)
    f_names = ["X"]

    for op in ("SIGN", "ABS", "LOG_ABS", "SIGMOID", "RELU", "EXP_SAFE", "LOG_SAFE"):
        node = ASTNode(op, children=[ASTNode("FEATURE", "X")])
        res = TensorASTEvaluator.eval_node(node, X, feature_names=f_names)
        assert torch.all(torch.isfinite(res)), f"Non-finite output in op: {op}"
        assert not torch.any(torch.isnan(res)), f"NaN output in op: {op}"
        assert res.shape == (5,)

    relu_node = ASTNode("RELU", children=[ASTNode("FEATURE", "X")])
    res_relu = TensorASTEvaluator.eval_node(relu_node, X, feature_names=f_names)
    assert res_relu[0].item() == 0.0
    assert res_relu[4].item() == 50.0
    print("[PASS] test_tensor_evaluator_unary_operators")


def test_tensor_evaluator_rolling_operators():
    """Verify rolling time-series operators (DELTA_5, EMA_10, TS_MAX_10, TS_MIN_10, TS_RANK_20)."""
    from src.domain.formula_ast import ASTNode
    from src.domain.tensor_evaluator import TensorASTEvaluator

    # Linear sequence: 1..30
    seq = torch.arange(1.0, 31.0).unsqueeze(-1)  # [30, 1]
    f_names = ["Close"]

    # DELTA_5: x[t] - x[t-5]
    delta_node = ASTNode("DELTA_5", children=[ASTNode("FEATURE", "Close")])
    res_delta = TensorASTEvaluator.eval_node(delta_node, seq, feature_names=f_names)
    assert res_delta[0].item() == 0.0
    assert res_delta[4].item() == 0.0
    assert torch.isclose(res_delta[5], torch.tensor(5.0))
    assert torch.isclose(res_delta[29], torch.tensor(5.0))

    # TS_MAX_10 & TS_MIN_10
    max_node = ASTNode("TS_MAX_10", children=[ASTNode("FEATURE", "Close")])
    min_node = ASTNode("TS_MIN_10", children=[ASTNode("FEATURE", "Close")])
    res_max = TensorASTEvaluator.eval_node(max_node, seq, feature_names=f_names)
    res_min = TensorASTEvaluator.eval_node(min_node, seq, feature_names=f_names)
    assert torch.isclose(res_max[0], torch.tensor(1.0))
    assert torch.isclose(res_max[14], torch.tensor(15.0))
    assert torch.isclose(res_min[14], torch.tensor(6.0))  # 15 - 10 + 1 = 6

    # TS_RANK_20: strictly monotonic sequence should yield rank 1.0 after warmup
    rank_node = ASTNode("TS_RANK_20", children=[ASTNode("FEATURE", "Close")])
    res_rank = TensorASTEvaluator.eval_node(rank_node, seq, feature_names=f_names)
    assert torch.all(res_rank >= 0.0) and torch.all(res_rank <= 1.0)
    assert torch.isclose(res_rank[25], torch.tensor(19.5 / 20.0))

    # EMA_10
    ema_node = ASTNode("EMA_10", children=[ASTNode("FEATURE", "Close")])
    res_ema = TensorASTEvaluator.eval_node(ema_node, seq, feature_names=f_names)
    assert torch.all(torch.isfinite(res_ema))
    assert res_ema[0].item() == 1.0
    assert res_ema[-1].item() < 30.0  # Lags the spot price
    print("[PASS] test_tensor_evaluator_rolling_operators")


def test_tensor_evaluator_cross_sectional():
    """Verify CS_RANK and CS_DEMEAN on multi-asset slices."""
    from src.domain.formula_ast import ASTNode
    from src.domain.tensor_evaluator import TensorASTEvaluator

    # 3 assets (B=3), 4 time bars (T=4), 1 feature (F=1)
    slice_data = torch.tensor([
        [[10.0], [20.0], [30.0], [40.0]],
        [[20.0], [10.0], [10.0], [50.0]],
        [[30.0], [30.0], [20.0], [10.0]],
    ], dtype=torch.float32)

    rank_node = ASTNode("CS_RANK", children=[ASTNode("FEATURE", "Signal")])
    ranked = TensorASTEvaluator.eval_node(rank_node, slice_data, feature_names=["Signal"])
    assert ranked.shape == (3, 4)
    assert torch.isclose(ranked[0, 0], torch.tensor(0.5 / 3.0))
    assert torch.isclose(ranked[1, 0], torch.tensor(1.5 / 3.0))
    assert torch.isclose(ranked[2, 0], torch.tensor(2.5 / 3.0))

    demean_node = ASTNode("CS_DEMEAN", children=[ASTNode("FEATURE", "Signal")])
    demeaned = TensorASTEvaluator.eval_node(demean_node, slice_data, feature_names=["Signal"])
    assert demeaned.shape == (3, 4)
    # Sum across assets (dim 0) must be 0.0 at each time step
    cs_sums = torch.sum(demeaned, dim=0)
    assert torch.allclose(cs_sums, torch.zeros(4), atol=1e-5)
    print("[PASS] test_tensor_evaluator_cross_sectional")


def test_tensor_evaluator_mega_switch_and_device():
    """Verify MEGA_SWITCH dynamic switching and device compatibility."""
    from src.domain.formula_ast import ASTNode
    from src.domain.tensor_evaluator import TensorASTEvaluator

    cond = ASTNode("SUB", children=[ASTNode("FEATURE", "Vol"), ASTNode("FEATURE", "2.0")])
    base = ASTNode("FEATURE", "Close")
    mega = ASTNode("MUL", children=[ASTNode("FEATURE", "Close"), ASTNode("FEATURE", "3.0")])
    switch_node = ASTNode("MEGA_SWITCH", children=[cond, base, mega])

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    for dev_name in devices:
        device = torch.device(dev_name)
        X = torch.tensor([
            [1.0, 100.0],  # Vol <= 2.0 -> base (100.0)
            [3.0, 100.0],  # Vol > 2.0  -> mega (300.0)
        ], dtype=torch.float32, device=device)

        res = TensorASTEvaluator.eval_node(switch_node, X, feature_names=["Vol", "Close"])
        assert res.device.type == dev_name
        assert torch.isclose(res[0], torch.tensor(100.0, device=device))
        assert torch.isclose(res[1], torch.tensor(300.0, device=device))

    print("[PASS] test_tensor_evaluator_mega_switch_and_device")


def load_tests(loader, standard_tests, pattern):
    import inspect
    import unittest
    suite = unittest.TestSuite()
    current_module = sys.modules[__name__]
    for name, func in inspect.getmembers(current_module, inspect.isfunction):
        if name.startswith("test_"):
            suite.addTest(unittest.FunctionTestCase(func))
    return suite


if __name__ == "__main__":
    import unittest
    unittest.main()
