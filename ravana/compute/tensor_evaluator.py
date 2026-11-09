"""Unified PyTorch Vectorized Tensor AST Evaluator for RAVANA.

Evaluates symbolic AST expression trees directly on 2D [T, F] or 3D [B, T, F] PyTorch tensors.
Supports seamless switching between CPU and CUDA GPU with zero Pandas overhead.
Features vectorized rolling convolutions, unfolds, and bounded zero-crash mathematical operators.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F

from ravana.domain.formula_ast import (
    ASTNode,
    OPERATORS_BINARY_ARITHMETIC,
    OPERATORS_BINARY_ROLLING,
    OPERATORS_UNARY,
    OPERATORS_ROLLING_UNARY,
    STANDARD_FEATURES,
)

# Thread-safe pre-computed convolution weight cache to eliminate GC pressure
_WEIGHT_CACHE: Dict[Tuple[str, int, int, torch.dtype, torch.device], torch.Tensor] = {}


def _get_ema_weights(span: int, w: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    key = ("ema", span, w, dtype, device)
    cached = _WEIGHT_CACHE.get(key)
    if cached is None:
        alpha = 2.0 / (span + 1.0)
        weights = torch.tensor([(1.0 - alpha) ** i for i in range(w - 1, -1, -1)], dtype=dtype, device=device)
        cached = (weights / weights.sum()).view(1, 1, w)
        _WEIGHT_CACHE[key] = cached
    return cached


def _get_decay_weights(window: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    key = ("decay", window, 0, dtype, device)
    cached = _WEIGHT_CACHE.get(key)
    if cached is None:
        weights = torch.arange(1, window + 1, dtype=dtype, device=device)
        cached = (weights / weights.sum()).view(1, 1, window)
        _WEIGHT_CACHE[key] = cached
    return cached


def _flatten_for_conv(x: torch.Tensor) -> Tuple[torch.Tensor, int, Tuple[int, ...], int]:
    """Reshapes [T], [B, T], or [B, T, F] to (N, 1, T) along the time axis."""
    orig_shape = x.shape
    if x.ndim == 1:
        return x.reshape(1, 1, -1), x.shape[0], orig_shape, 1
    if x.ndim == 2:
        return x.reshape(x.shape[0], 1, -1), x.shape[-1], orig_shape, 2
    if x.ndim == 3:
        # [B, T, F] -> Convolve along T (dim 1)
        B, T, F_dim = x.shape
        flat = x.permute(0, 2, 1).reshape(B * F_dim, 1, T)
        return flat, T, orig_shape, 3
    flat = x.reshape(-1, 1, x.shape[-1])
    return flat, x.shape[-1], orig_shape, x.ndim


def _restore_from_conv(conv: torch.Tensor, orig_shape: Tuple[int, ...], ndim: int) -> torch.Tensor:
    if ndim == 3:
        B, T, F_dim = orig_shape
        return conv.reshape(B, F_dim, T).permute(0, 2, 1)
    return conv.reshape(orig_shape)


def div_safe_tensor(l: torch.Tensor, r: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Safe division preserving denominator sign and clamping output to [-1e4, 1e4]."""
    l_c = torch.nan_to_num(torch.clamp(l, min=-1e12, max=1e12), nan=0.0, posinf=1e12, neginf=-1e12)
    r_c = torch.nan_to_num(torch.clamp(r, min=-1e12, max=1e12), nan=0.0, posinf=1e12, neginf=-1e12)
    r_abs = torch.abs(r_c)
    denom_sign = torch.sign(r_c)
    denom_sign = torch.where(denom_sign == 0.0, torch.ones_like(denom_sign), denom_sign)
    safe_denom = denom_sign * torch.clamp(r_abs, min=eps)
    res = torch.where(r_abs > eps, l_c / safe_denom, torch.zeros_like(l_c))
    return torch.nan_to_num(torch.clamp(res, min=-1e4, max=1e4), nan=0.0, posinf=1e4, neginf=-1e4)


def ts_delta(x: torch.Tensor, period: int = 5) -> torch.Tensor:
    """Computes x[t] - x[t - period], zero-padded for prefix t < period."""
    clean_x = torch.nan_to_num(torch.clamp(x, min=-1e12, max=1e12), nan=0.0, posinf=1e12, neginf=-1e12)
    res = torch.zeros_like(clean_x)
    if clean_x.ndim == 3:
        T = clean_x.shape[1]
        if T > period:
            res[:, period:, :] = clean_x[:, period:, :] - clean_x[:, :-period, :]
    else:
        T = clean_x.shape[-1]
        if T > period:
            res[..., period:] = clean_x[..., period:] - clean_x[..., :-period]
    return torch.nan_to_num(torch.clamp(res, min=-1e4, max=1e4), nan=0.0, posinf=1e4, neginf=-1e4)


def ts_delay(x: torch.Tensor, period: int = 1) -> torch.Tensor:
    """Computes lagged series x[t - period] with edge-padded prefix."""
    clean_x = torch.nan_to_num(torch.clamp(x, min=-1e12, max=1e12), nan=0.0, posinf=1e12, neginf=-1e12)
    res = torch.zeros_like(clean_x)
    if clean_x.ndim == 3:
        T = clean_x.shape[1]
        if T == 0:
            return res
        if T > period:
            res[:, period:, :] = clean_x[:, :-period, :]
            res[:, :period, :] = clean_x[:, 0:1, :]
        else:
            res[...] = clean_x[:, 0:1, :]
    else:
        T = clean_x.shape[-1]
        if T == 0:
            return res
        if T > period:
            res[..., period:] = clean_x[..., :-period]
            res[..., :period] = clean_x[..., 0:1]
        else:
            res[...] = clean_x[..., 0:1]
    return torch.nan_to_num(torch.clamp(res, min=-1e4, max=1e4), nan=0.0, posinf=1e4, neginf=-1e4)


def ts_ema(x: torch.Tensor, span: int = 10) -> torch.Tensor:
    """Exponential moving average via 1D depthwise convolution with zero Python loops."""
    clean_x = torch.nan_to_num(torch.clamp(x, min=-1e12, max=1e12), nan=0.0, posinf=1e12, neginf=-1e12)
    flat_x, T, orig_shape, ndim = _flatten_for_conv(clean_x)
    if T == 0:
        return clean_x.clone()
    w = min(T, max(span * 3, 20))
    weights = _get_ema_weights(span, w, clean_x.dtype, clean_x.device)
    padded = F.pad(flat_x, (w - 1, 0), mode="replicate")
    conv = F.conv1d(padded, weights)
    res = _restore_from_conv(conv, orig_shape, ndim)
    return torch.nan_to_num(torch.clamp(res, min=-1e4, max=1e4), nan=0.0, posinf=1e4, neginf=-1e4)


def ts_max(x: torch.Tensor, window: int = 10) -> torch.Tensor:
    """Rolling maximum via 1D max pooling with finite bound padding."""
    clean_x = torch.nan_to_num(torch.clamp(x, min=-1e12, max=1e12), nan=-1e12, posinf=1e12, neginf=-1e12)
    flat_x, T, orig_shape, ndim = _flatten_for_conv(clean_x)
    if T == 0:
        return clean_x.clone()
    padded = F.pad(flat_x, (window - 1, 0), mode="replicate")
    pooled = F.max_pool1d(padded, kernel_size=window, stride=1)
    res = _restore_from_conv(pooled, orig_shape, ndim)
    return torch.nan_to_num(torch.clamp(res, min=-1e4, max=1e4), nan=0.0, posinf=1e4, neginf=-1e4)


def ts_min(x: torch.Tensor, window: int = 10) -> torch.Tensor:
    """Rolling minimum via negated 1D max pooling."""
    clean_x = torch.nan_to_num(torch.clamp(x, min=-1e12, max=1e12), nan=1e12, posinf=1e12, neginf=-1e12)
    flat_x, T, orig_shape, ndim = _flatten_for_conv(-clean_x)
    if T == 0:
        return clean_x.clone()
    padded = F.pad(flat_x, (window - 1, 0), mode="replicate")
    pooled = -F.max_pool1d(padded, kernel_size=window, stride=1)
    res = _restore_from_conv(pooled, orig_shape, ndim)
    return torch.nan_to_num(torch.clamp(res, min=-1e4, max=1e4), nan=0.0, posinf=1e4, neginf=-1e4)


def ts_rank_20(x: torch.Tensor, window: int = 20) -> torch.Tensor:
    """Sliding-window percentile rank over 20 periods via PyTorch unfold."""
    clean_x = torch.nan_to_num(torch.clamp(x, min=-1e12, max=1e12), nan=0.0, posinf=1e12, neginf=-1e12)
    T = clean_x.shape[1] if clean_x.ndim == 3 else clean_x.shape[-1]
    if T == 0:
        return clean_x.clone()
    orig_shape = clean_x.shape
    if clean_x.ndim == 3:
        B, T, F_dim = clean_x.shape
        clean_x = clean_x.permute(0, 2, 1).reshape(B * F_dim, T)
    res = torch.full_like(clean_x, 0.5)
    prefix_len = min(T, window)
    for i in range(prefix_len):
        hist = clean_x[..., : i + 1]
        target = clean_x[..., i : i + 1]
        less = (hist < target).sum(dim=-1, keepdim=True).to(clean_x.dtype)
        eq = (hist == target).sum(dim=-1, keepdim=True).to(clean_x.dtype)
        res[..., i : i + 1] = (less + 0.5 * eq) / float(i + 1)
    if T > window:
        windows = clean_x.unfold(dimension=-1, size=window, step=1)
        target = clean_x[..., window - 1 :].unsqueeze(-1)
        less = (windows < target).sum(dim=-1).to(clean_x.dtype)
        eq = (windows == target).sum(dim=-1).to(clean_x.dtype)
        res[..., window - 1 :] = (less + 0.5 * eq) / float(window)
    res = torch.nan_to_num(res, nan=0.5)
    if len(orig_shape) == 3:
        B, T, F_dim = orig_shape
        res = res.reshape(B, F_dim, T).permute(0, 2, 1)
    return res


def ts_zscore_20(x: torch.Tensor, window: int = 20) -> torch.Tensor:
    """Vectorized rolling z-score clamped to [-4.0, 4.0]."""
    clean_x = torch.nan_to_num(torch.clamp(x, min=-1e12, max=1e12), nan=0.0, posinf=1e12, neginf=-1e12)
    T = clean_x.shape[1] if clean_x.ndim == 3 else clean_x.shape[-1]
    if T == 0:
        return clean_x.clone()
    orig_shape = clean_x.shape
    if clean_x.ndim == 3:
        B, T, F_dim = clean_x.shape
        clean_x = clean_x.permute(0, 2, 1).reshape(B * F_dim, T)
    res = torch.zeros_like(clean_x)
    prefix_len = min(T, window)
    for i in range(1, prefix_len):
        hist = clean_x[..., : i + 1]
        mean = hist.mean(dim=-1, keepdim=True)
        std = hist.std(dim=-1, unbiased=True, keepdim=True)
        target = clean_x[..., i : i + 1]
        res[..., i : i + 1] = (target - mean) / (std + 1e-6)
    if T >= window:
        windows = clean_x.unfold(dimension=-1, size=window, step=1)
        mean = windows.mean(dim=-1)
        std = windows.std(dim=-1, unbiased=True)
        target = clean_x[..., window - 1 :]
        res[..., window - 1 :] = (target - mean) / (std + 1e-6)
    res = torch.nan_to_num(torch.clamp(res, min=-4.0, max=4.0), nan=0.0)
    if len(orig_shape) == 3:
        B, T, F_dim = orig_shape
        res = res.reshape(B, F_dim, T).permute(0, 2, 1)
    return res


def ts_decay_linear(x: torch.Tensor, window: int = 10) -> torch.Tensor:
    """Linear time-decay weighted moving average via 1D convolution."""
    clean_x = torch.nan_to_num(torch.clamp(x, min=-1e12, max=1e12), nan=0.0, posinf=1e12, neginf=-1e12)
    flat_x, T, orig_shape, ndim = _flatten_for_conv(clean_x)
    if T == 0:
        return clean_x.clone()
    weights = _get_decay_weights(window, clean_x.dtype, clean_x.device)
    padded = F.pad(flat_x, (window - 1, 0), mode="replicate")
    conv = F.conv1d(padded, weights)
    res = _restore_from_conv(conv, orig_shape, ndim)
    return torch.nan_to_num(torch.clamp(res, min=-1e4, max=1e4), nan=0.0, posinf=1e4, neginf=-1e4)


def ts_corr_20(x: torch.Tensor, y: torch.Tensor, window: int = 20) -> torch.Tensor:
    """Vectorized rolling Pearson correlation clamped to [-1.0, 1.0]."""
    clean_x = torch.nan_to_num(torch.clamp(x, min=-1e12, max=1e12), nan=0.0, posinf=1e12, neginf=-1e12)
    clean_y = torch.nan_to_num(torch.clamp(y, min=-1e12, max=1e12), nan=0.0, posinf=1e12, neginf=-1e12)
    T = clean_x.shape[1] if clean_x.ndim == 3 else clean_x.shape[-1]
    if T == 0:
        return clean_x.clone()
    orig_shape = clean_x.shape
    if clean_x.ndim == 3:
        B, T, F_dim = clean_x.shape
        clean_x = clean_x.permute(0, 2, 1).reshape(B * F_dim, T)
        clean_y = clean_y.permute(0, 2, 1).reshape(B * F_dim, T)
    res = torch.zeros_like(clean_x)
    prefix_len = min(T, window)
    for i in range(2, prefix_len):
        hx = clean_x[..., : i + 1]
        hy = clean_y[..., : i + 1]
        mx = hx.mean(dim=-1, keepdim=True)
        my = hy.mean(dim=-1, keepdim=True)
        cov = ((hx - mx) * (hy - my)).sum(dim=-1, keepdim=True) / float(i)
        sx = hx.std(dim=-1, unbiased=True, keepdim=True)
        sy = hy.std(dim=-1, unbiased=True, keepdim=True)
        res[..., i : i + 1] = cov / (sx * sy + 1e-6)
    if T >= window:
        wx = clean_x.unfold(dimension=-1, size=window, step=1)
        wy = clean_y.unfold(dimension=-1, size=window, step=1)
        mx = wx.mean(dim=-1, keepdim=True)
        my = wy.mean(dim=-1, keepdim=True)
        cov = ((wx - mx) * (wy - my)).sum(dim=-1) / float(window - 1)
        sx = wx.std(dim=-1, unbiased=True)
        sy = wy.std(dim=-1, unbiased=True)
        res[..., window - 1 :] = cov / (sx * sy + 1e-6)
    res = torch.nan_to_num(torch.clamp(res, min=-1.0, max=1.0), nan=0.0)
    if len(orig_shape) == 3:
        B, T, F_dim = orig_shape
        res = res.reshape(B, F_dim, T).permute(0, 2, 1)
    return res


class TensorASTEvaluator:
    """Evaluates symbolic formula AST nodes directly on 2D or 3D PyTorch tensors."""

    @classmethod
    @torch.no_grad()
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

    @classmethod
    @torch.no_grad()
    def evaluate_batch(
        cls,
        nodes: List[ASTNode],
        X: torch.Tensor,
        feature_names: Optional[Union[List[str], Dict[str, int]]] = None,
    ) -> torch.Tensor:
        """Evaluates a batch of ASTNode expression trees, returning [N_formulas, ...]."""
        if not nodes:
            is_3d = X.ndim == 3
            empty_shape = (0, X.shape[0], X.shape[1]) if is_3d else (0, X.shape[0])
            return torch.zeros(empty_shape, dtype=X.dtype, device=X.device)
        return torch.stack([cls.eval_node(node, X, feature_names) for node in nodes], dim=0)

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
            feat = X[:, :, idx] if is_3d else X[:, idx]
            return torch.nan_to_num(torch.clamp(feat, min=-1e12, max=1e12), nan=0.0, posinf=1e12, neginf=-1e12)
        if val == "Spread" and "High" in f_map and "Low" in f_map:
            h = X[:, :, f_map["High"]] if is_3d else X[:, f_map["High"]]
            l = X[:, :, f_map["Low"]] if is_3d else X[:, f_map["Low"]]
            c_key = "Close_Adj" if "Close_Adj" in f_map else ("Close" if "Close" in f_map else None)
            if c_key is not None:
                c = X[:, :, f_map[c_key]] if is_3d else X[:, f_map[c_key]]
            else:
                c = torch.ones_like(h)
            raw_spread = (h - l) / (torch.abs(c) + 1e-8)
            return torch.nan_to_num(torch.clamp(raw_spread, min=-1e4, max=1e4), nan=0.0, posinf=1e4, neginf=-1e4)
        try:
            return torch.full(shape, float(val), dtype=X.dtype, device=X.device)
        except (ValueError, TypeError):
            return torch.zeros(shape, dtype=X.dtype, device=X.device)

    @classmethod
    def _eval_blend(cls, node: ASTNode, X: torch.Tensor, f_map: Dict[str, int], shape: tuple) -> torch.Tensor:
        cond = cls.eval_node(node.children[0], X, f_map) if len(node.children) > 0 else torch.zeros(shape, dtype=X.dtype, device=X.device)
        a = cls.eval_node(node.children[1], X, f_map) if len(node.children) > 1 else torch.zeros(shape, dtype=X.dtype, device=X.device)
        b = cls.eval_node(node.children[2], X, f_map) if len(node.children) > 2 else torch.zeros(shape, dtype=X.dtype, device=X.device)
        a_clean = torch.nan_to_num(torch.clamp(a, min=-1e12, max=1e12), nan=0.0, posinf=1e12, neginf=-1e12)
        b_clean = torch.nan_to_num(torch.clamp(b, min=-1e12, max=1e12), nan=0.0, posinf=1e12, neginf=-1e12)
        weight = torch.sigmoid(torch.clamp(cond, min=-10.0, max=10.0))
        res = weight * a_clean + (1.0 - weight) * b_clean
        return torch.nan_to_num(torch.clamp(res, min=-1e4, max=1e4), nan=0.0, posinf=1e4, neginf=-1e4)

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
        return torch.nan_to_num(torch.clamp(res, min=-1e4, max=1e4), nan=0.0, posinf=1e4, neginf=-1e4)

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
        return torch.nan_to_num(torch.clamp(res, min=-1e4, max=1e4), nan=0.0, posinf=1e4, neginf=-1e4)

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
