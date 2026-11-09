"""Unified PyTorch Vectorized Tensor AST Evaluator for RAVANA.

Evaluates symbolic AST expression trees directly on 2D [T, F] or 3D [B, T, F] PyTorch tensors.
Supports seamless switching between CPU and CUDA GPU with zero Pandas overhead.
Features vectorized rolling convolutions, unfolds, and bounded zero-crash mathematical operators.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Union
import torch
import torch.nn.functional as F

from ravana.domain.formula_ast import (
    ASTNode,
    OPERATORS_BINARY_ARITHMETIC,
    OPERATORS_BINARY_ROLLING,
    OPERATORS_UNARY,
    OPERATORS_ROLLING_UNARY,
    OPERATORS_TERNARY,
    STANDARD_FEATURES,
)


def div_safe_tensor(l: torch.Tensor, r: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Safe division preserving denominator sign and clamping to [-1e4, 1e4]."""
    r_abs = torch.abs(r)
    denom_sign = torch.sign(r)
    denom_sign = torch.where(denom_sign == 0.0, torch.ones_like(denom_sign), denom_sign)
    safe_denom = denom_sign * torch.clamp(r_abs, min=eps)
    res = torch.where(r_abs > eps, l / safe_denom, torch.zeros_like(l))
    return torch.nan_to_num(torch.clamp(res, min=-1e4, max=1e4), nan=0.0)


def ts_delta(x: torch.Tensor, period: int = 5) -> torch.Tensor:
    """Computes x[t] - x[t - period], zero-padded for prefix t < period."""
    res = torch.zeros_like(x)
    if x.shape[-1] > period:
        res[..., period:] = x[..., period:] - x[..., :-period]
    return res


def ts_delay(x: torch.Tensor, period: int = 1) -> torch.Tensor:
    """Computes lagged series x[t - period] with edge-padded prefix."""
    res = torch.zeros_like(x)
    T = x.shape[-1]
    if T == 0:
        return res
    if T > period:
        res[..., period:] = x[..., :-period]
        res[..., :period] = x[..., 0:1]
    else:
        res[...] = x[..., 0:1]
    return res


def ts_ema(x: torch.Tensor, span: int = 10) -> torch.Tensor:
    """Exponential moving average via 1D depthwise convolution with zero Python loops."""
    T = x.shape[-1]
    if T == 0:
        return x.clone()
    alpha = 2.0 / (span + 1.0)
    w = min(T, max(span * 3, 20))
    weights = torch.tensor([(1.0 - alpha) ** i for i in range(w - 1, -1, -1)], dtype=x.dtype, device=x.device)
    weights = (weights / weights.sum()).view(1, 1, w)
    orig_shape = x.shape
    flat_x = x.reshape(-1, 1, T)
    padded = F.pad(flat_x, (w - 1, 0), mode="replicate")
    conv = F.conv1d(padded, weights)
    return torch.nan_to_num(conv.reshape(orig_shape), nan=0.0)


def ts_max(x: torch.Tensor, window: int = 10) -> torch.Tensor:
    """Rolling maximum via 1D max pooling with -inf padding."""
    T = x.shape[-1]
    if T == 0:
        return x.clone()
    clean_x = torch.nan_to_num(x, nan=0.0)
    orig_shape = clean_x.shape
    flat_x = clean_x.reshape(-1, 1, T)
    padded = F.pad(flat_x, (window - 1, 0), mode="replicate")
    pooled = F.max_pool1d(padded, kernel_size=window, stride=1)
    return torch.nan_to_num(pooled.reshape(orig_shape), nan=0.0)


def ts_min(x: torch.Tensor, window: int = 10) -> torch.Tensor:
    """Rolling minimum via negated 1D max pooling."""
    T = x.shape[-1]
    if T == 0:
        return x.clone()
    clean_x = torch.nan_to_num(x, nan=0.0)
    orig_shape = clean_x.shape
    flat_x = -clean_x.reshape(-1, 1, T)
    padded = F.pad(flat_x, (window - 1, 0), mode="replicate")
    pooled = -F.max_pool1d(padded, kernel_size=window, stride=1)
    return torch.nan_to_num(pooled.reshape(orig_shape), nan=0.0)


def ts_rank_20(x: torch.Tensor, window: int = 20) -> torch.Tensor:
    """Sliding-window percentile rank over 20 periods via PyTorch unfold."""
    T = x.shape[-1]
    if T == 0:
        return x.clone()
    res = torch.full_like(x, 0.5)
    prefix_len = min(T, window)
    for i in range(prefix_len):
        hist = x[..., : i + 1]
        target = x[..., i : i + 1]
        less = (hist < target).sum(dim=-1, keepdim=True).to(x.dtype)
        eq = (hist == target).sum(dim=-1, keepdim=True).to(x.dtype)
        res[..., i : i + 1] = (less + 0.5 * eq) / float(i + 1)
    if T > window:
        windows = x.unfold(dimension=-1, size=window, step=1)
        target = x[..., window - 1 :].unsqueeze(-1)
        less = (windows < target).sum(dim=-1).to(x.dtype)
        eq = (windows == target).sum(dim=-1).to(x.dtype)
        res[..., window - 1 :] = (less + 0.5 * eq) / float(window)
    return torch.nan_to_num(res, nan=0.5)


def ts_zscore_20(x: torch.Tensor, window: int = 20) -> torch.Tensor:
    """Vectorized rolling z-score clamped to [-4.0, 4.0]."""
    T = x.shape[-1]
    if T == 0:
        return x.clone()
    res = torch.zeros_like(x)
    prefix_len = min(T, window)
    for i in range(1, prefix_len):
        hist = x[..., : i + 1]
        mean = hist.mean(dim=-1, keepdim=True)
        std = hist.std(dim=-1, unbiased=True, keepdim=True)
        target = x[..., i : i + 1]
        res[..., i : i + 1] = (target - mean) / (std + 1e-6)
    if T >= window:
        windows = x.unfold(dimension=-1, size=window, step=1)
        mean = windows.mean(dim=-1)
        std = windows.std(dim=-1, unbiased=True)
        target = x[..., window - 1 :]
        res[..., window - 1 :] = (target - mean) / (std + 1e-6)
    return torch.nan_to_num(torch.clamp(res, min=-4.0, max=4.0), nan=0.0)


def ts_decay_linear(x: torch.Tensor, window: int = 10) -> torch.Tensor:
    """Linear time-decay weighted moving average via 1D convolution."""
    T = x.shape[-1]
    if T == 0:
        return x.clone()
    orig_shape = x.shape
    flat_x = x.reshape(-1, 1, T)
    weights = torch.arange(1, window + 1, dtype=x.dtype, device=x.device)
    weights = (weights / weights.sum()).view(1, 1, window)
    padded = F.pad(flat_x, (window - 1, 0), mode="replicate")
    conv = F.conv1d(padded, weights)
    return torch.nan_to_num(conv.reshape(orig_shape), nan=0.0)


def ts_corr_20(x: torch.Tensor, y: torch.Tensor, window: int = 20) -> torch.Tensor:
    """Vectorized rolling Pearson correlation clamped to [-1.0, 1.0]."""
    T = x.shape[-1]
    if T == 0:
        return torch.zeros_like(x)
    res = torch.zeros_like(x)
    prefix_len = min(T, window)
    for i in range(2, prefix_len):
        hx = x[..., : i + 1]
        hy = y[..., : i + 1]
        mx = hx.mean(dim=-1, keepdim=True)
        my = hy.mean(dim=-1, keepdim=True)
        cov = ((hx - mx) * (hy - my)).sum(dim=-1, keepdim=True) / float(i)
        sx = hx.std(dim=-1, unbiased=True, keepdim=True)
        sy = hy.std(dim=-1, unbiased=True, keepdim=True)
        res[..., i : i + 1] = cov / (sx * sy + 1e-6)
    if T >= window:
        wx = x.unfold(dimension=-1, size=window, step=1)
        wy = y.unfold(dimension=-1, size=window, step=1)
        mx = wx.mean(dim=-1, keepdim=True)
        my = wy.mean(dim=-1, keepdim=True)
        cov = ((wx - mx) * (wy - my)).sum(dim=-1) / float(window - 1)
        sx = wx.std(dim=-1, unbiased=True)
        sy = wy.std(dim=-1, unbiased=True)
        res[..., window - 1 :] = cov / (sx * sy + 1e-6)
    return torch.nan_to_num(torch.clamp(res, min=-1.0, max=1.0), nan=0.0)


class TensorASTEvaluator:
    """Evaluates symbolic formula AST nodes directly on 2D or 3D PyTorch tensors."""

    @classmethod
    def eval_node(
        cls,
        node: ASTNode,
        X: torch.Tensor,
        feature_names: Optional[Union[List[str], Dict[str, int]]] = None,
    ) -> torch.Tensor:
        """Evaluates ASTNode over feature tensor X of shape [T, F] or [B, T, F]."""
        is_3d = X.ndim == 3
        shape = (X.shape[0], X.shape[1]) if is_3d else (X.shape[0],)
        f_map = cls._build_feature_map(feature_names)

        if node.op == "CONST":
            return cls._eval_const(node, shape, X)
        if node.op == "FEATURE":
            return cls._eval_feature(node.val, X, f_map, shape)
        if node.op == "BLEND":
            return cls._eval_blend(node, X, f_map, shape)
        if node.op in OPERATORS_BINARY_ROLLING:
            return cls._eval_binary_rolling(node, X, f_map, shape)
        if node.op in OPERATORS_BINARY_ARITHMETIC:
            return cls._eval_binary(node, X, f_map, shape)
        if node.op in OPERATORS_UNARY:
            return cls._eval_unary(node, X, f_map, shape)
        if node.op in OPERATORS_ROLLING_UNARY:
            return cls._eval_rolling(node, X, f_map, shape)

        return torch.zeros(shape, dtype=X.dtype, device=X.device)

    @staticmethod
    def _build_feature_map(
        feature_names: Optional[Union[List[str], Dict[str, int]]]
    ) -> Dict[str, int]:
        if isinstance(feature_names, dict):
            return feature_names
        if isinstance(feature_names, (list, tuple)):
            return {name: i for i, name in enumerate(feature_names)}
        return {name: i for i, name in enumerate(STANDARD_FEATURES)}

    @staticmethod
    def _eval_const(node: ASTNode, shape: tuple, X: torch.Tensor) -> torch.Tensor:
        try:
            return torch.full(shape, float(node.val), dtype=X.dtype, device=X.device)
        except (ValueError, TypeError):
            return torch.zeros(shape, dtype=X.dtype, device=X.device)

    @staticmethod
    def _eval_feature(val: str, X: torch.Tensor, f_map: Dict[str, int], shape: tuple) -> torch.Tensor:
        is_3d = X.ndim == 3
        if val in f_map:
            idx = f_map[val]
            return X[:, :, idx] if is_3d else X[:, idx]
        if val == "Spread" and "High" in f_map and "Low" in f_map:
            h = X[:, :, f_map["High"]] if is_3d else X[:, f_map["High"]]
            l = X[:, :, f_map["Low"]] if is_3d else X[:, f_map["Low"]]
            c_key = "Close_Adj" if "Close_Adj" in f_map else "Close"
            c = X[:, :, f_map[c_key]] if is_3d else X[:, f_map.get(c_key, 0)]
            return torch.nan_to_num((h - l) / (c + 1e-8), nan=0.0)
        try:
            return torch.full(shape, float(val), dtype=X.dtype, device=X.device)
        except (ValueError, TypeError):
            return torch.zeros(shape, dtype=X.dtype, device=X.device)

    @classmethod
    def _eval_blend(cls, node: ASTNode, X: torch.Tensor, f_map: Dict[str, int], shape: tuple) -> torch.Tensor:
        cond = cls.eval_node(node.children[0], X, f_map) if len(node.children) > 0 else torch.zeros(shape, dtype=X.dtype, device=X.device)
        a = cls.eval_node(node.children[1], X, f_map) if len(node.children) > 1 else torch.zeros(shape, dtype=X.dtype, device=X.device)
        b = cls.eval_node(node.children[2], X, f_map) if len(node.children) > 2 else torch.zeros(shape, dtype=X.dtype, device=X.device)
        weight = torch.sigmoid(torch.clamp(cond, min=-10.0, max=10.0))
        res = weight * a + (1.0 - weight) * b
        return torch.nan_to_num(torch.clamp(res, min=-1e4, max=1e4), nan=0.0)

    @classmethod
    def _eval_binary(cls, node: ASTNode, X: torch.Tensor, f_map: Dict[str, int], shape: tuple) -> torch.Tensor:
        l = cls.eval_node(node.children[0], X, f_map) if node.children else torch.zeros(shape, dtype=X.dtype, device=X.device)
        r = cls.eval_node(node.children[1], X, f_map) if len(node.children) > 1 else torch.zeros(shape, dtype=X.dtype, device=X.device)
        if node.op == "ADD":
            res = l + r
        elif node.op == "SUB":
            res = l - r
        elif node.op == "MUL":
            res = l * r
        elif node.op == "DIV_SAFE":
            return div_safe_tensor(l, r)
        else:
            res = torch.zeros_like(l)
        return torch.nan_to_num(torch.clamp(res, min=-1e4, max=1e4), nan=0.0)

    @classmethod
    def _eval_binary_rolling(cls, node: ASTNode, X: torch.Tensor, f_map: Dict[str, int], shape: tuple) -> torch.Tensor:
        l = cls.eval_node(node.children[0], X, f_map) if node.children else torch.zeros(shape, dtype=X.dtype, device=X.device)
        r = cls.eval_node(node.children[1], X, f_map) if len(node.children) > 1 else torch.zeros(shape, dtype=X.dtype, device=X.device)
        if node.op == "TS_CORR_20":
            return ts_corr_20(l, r, window=20)
        return torch.zeros(shape, dtype=X.dtype, device=X.device)

    @classmethod
    def _eval_unary(cls, node: ASTNode, X: torch.Tensor, f_map: Dict[str, int], shape: tuple) -> torch.Tensor:
        child = cls.eval_node(node.children[0], X, f_map) if node.children else torch.zeros(shape, dtype=X.dtype, device=X.device)
        if node.op == "SIGN": res = torch.sign(child)
        elif node.op == "ABS": res = torch.abs(child)
        elif node.op == "LOG_ABS": res = torch.log1p(torch.abs(child))
        elif node.op == "SIGMOID": res = torch.sigmoid(torch.clamp(child, min=-10.0, max=10.0))
        elif node.op == "TANH": res = torch.tanh(torch.clamp(child, min=-10.0, max=10.0))
        elif node.op == "RELU": res = torch.relu(child)
        elif node.op == "EXP_SAFE": res = torch.exp(torch.clamp(child, min=-20.0, max=20.0))
        elif node.op == "LOG_SAFE": res = torch.log(torch.clamp(torch.abs(child), min=1e-8))
        elif node.op == "NEG": res = -child
        elif node.op == "SQRT_ABS": res = torch.sign(child) * torch.sqrt(torch.abs(child))
        else: res = torch.zeros_like(child)
        return torch.nan_to_num(torch.clamp(res, min=-1e4, max=1e4), nan=0.0)

    @classmethod
    def _eval_rolling(cls, node: ASTNode, X: torch.Tensor, f_map: Dict[str, int], shape: tuple) -> torch.Tensor:
        child = cls.eval_node(node.children[0], X, f_map) if node.children else torch.zeros(shape, dtype=X.dtype, device=X.device)
        if node.op == "DELTA_5": return ts_delta(child, period=5)
        if node.op == "TS_DELAY_1": return ts_delay(child, period=1)
        if node.op == "TS_DELAY_5": return ts_delay(child, period=5)
        if node.op == "EMA_10": return ts_ema(child, span=10)
        if node.op == "TS_MAX_10": return ts_max(child, window=10)
        if node.op == "TS_MIN_10": return ts_min(child, window=10)
        if node.op == "TS_RANK_20": return ts_rank_20(child, window=20)
        if node.op == "TS_ZSCORE_20": return ts_zscore_20(child, window=20)
        if node.op == "TS_DECAY_LINEAR_10": return ts_decay_linear(child, window=10)
        return torch.zeros_like(child)

    def evaluate(
        self,
        node: ASTNode,
        X: torch.Tensor,
        feature_names: Optional[Union[List[str], Dict[str, int]]] = None,
    ) -> torch.Tensor:
        """Instance alias for eval_node."""
        return self.eval_node(node, X, feature_names)
