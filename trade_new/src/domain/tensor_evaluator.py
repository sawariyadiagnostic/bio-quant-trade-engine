"""Unified PyTorch Tensor AST Evaluator (CPU & GPU device-agnostic).

Replaces the Pandas Series bottleneck with pure vectorized PyTorch operations,
supporting feature tensors of shape [T, F] (single-asset) and [B, T, F] (multi-asset batch).
"""
from typing import Dict, List, Optional, Union
import torch
import torch.nn.functional as F

from src.domain.formula_ast import (
    ASTNode,
    OPERATORS_BINARY,
    OPERATORS_UNARY,
    OPERATORS_ROLLING,
    OPERATORS_CROSS_SECTIONAL,
    STANDARD_FEATURES,
)


def div_safe_tensor(l: torch.Tensor, r: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Safe division with epsilon guard, sign retention, and finite bounds clamping."""
    r_abs = torch.abs(r)
    denom_sign = torch.sign(r)
    denom_sign = torch.where(denom_sign == 0.0, torch.ones_like(denom_sign), denom_sign)
    safe_denom = denom_sign * torch.clamp(r_abs, min=eps)
    res = torch.where(r_abs > eps, l / safe_denom, torch.zeros_like(l))
    res = torch.clamp(res, min=-1e4, max=1e4)
    return torch.nan_to_num(res, nan=0.0, posinf=1e4, neginf=-1e4)


def ts_delta(x: torch.Tensor, period: int = 5) -> torch.Tensor:
    """Computes x[t] - x[t - period], zero-padded for prefix t < period."""
    res = torch.zeros_like(x)
    if x.shape[-1] > period:
        res[..., period:] = x[..., period:] - x[..., :-period]
    return torch.nan_to_num(res, nan=0.0)


def ts_ema(x: torch.Tensor, span: int = 10) -> torch.Tensor:
    """Exponential moving average with forward-fill NaN handling."""
    T = x.shape[-1]
    if T == 0:
        return x.clone()
    alpha = 2.0 / (span + 1.0)
    decay = 1.0 - alpha
    res = torch.zeros_like(x)
    initialized = ~torch.isnan(x[..., 0]) & torch.isfinite(x[..., 0])
    cur_ema = torch.where(initialized, x[..., 0], torch.zeros_like(x[..., 0]))
    res[..., 0] = cur_ema
    for t in range(1, T):
        val = x[..., t]
        is_valid = ~torch.isnan(val) & torch.isfinite(val)
        new_init = ~initialized & is_valid
        cur_ema = torch.where(
            new_init,
            val,
            torch.where(
                initialized & is_valid,
                decay * cur_ema + alpha * val,
                cur_ema,
            ),
        )
        initialized = initialized | is_valid
        res[..., t] = cur_ema
    return torch.nan_to_num(res, nan=0.0)


def ts_max(x: torch.Tensor, window: int = 10) -> torch.Tensor:
    """Rolling maximum via 1D max pooling with -inf padding."""
    T = x.shape[-1]
    if T == 0:
        return x.clone()
    clean_x = torch.nan_to_num(x, nan=0.0)
    flat_x = clean_x.reshape(-1, 1, T)
    padded = F.pad(flat_x, (window - 1, 0), mode="constant", value=-float("inf"))
    pooled = F.max_pool1d(padded, kernel_size=window, stride=1)
    res = pooled.reshape(clean_x.shape)
    return torch.nan_to_num(res, nan=0.0)


def ts_min(x: torch.Tensor, window: int = 10) -> torch.Tensor:
    """Rolling minimum via negated 1D max pooling with -inf padding."""
    T = x.shape[-1]
    if T == 0:
        return x.clone()
    clean_x = torch.nan_to_num(x, nan=0.0)
    flat_x = -clean_x.reshape(-1, 1, T)
    padded = F.pad(flat_x, (window - 1, 0), mode="constant", value=-float("inf"))
    pooled = -F.max_pool1d(padded, kernel_size=window, stride=1)
    res = pooled.reshape(clean_x.shape)
    return torch.nan_to_num(res, nan=0.0)


def ts_rank_20(x: torch.Tensor, window: int = 20) -> torch.Tensor:
    """Sliding-window percentile rank over 20 periods via PyTorch unfold."""
    T = x.shape[-1]
    if T == 0:
        return x.clone()
    res = torch.empty_like(x)
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
    return torch.nan_to_num(res, nan=0.0)


def cs_rank_tensor(x: torch.Tensor) -> torch.Tensor:
    """Cross-sectional rank across asset axis (dim=0) with fractional tie-breaking and NaN masking."""
    if x.ndim <= 1 or x.shape[0] <= 1:
        return torch.full_like(x, 0.5)
    valid = ~torch.isnan(x) & torch.isfinite(x)
    k = valid.sum(dim=0, keepdim=True).to(x.dtype)
    x0 = x.unsqueeze(1)
    x1 = x.unsqueeze(0)
    valid_mask = valid.unsqueeze(1) & valid.unsqueeze(0)
    c_less = ((x0 > x1) & valid_mask).sum(dim=1).to(x.dtype)
    c_eq = ((x0 == x1) & valid_mask).sum(dim=1).to(x.dtype)
    ranks = c_less + 0.5 * (c_eq + 1.0)
    norm_ranks = (ranks - 0.5) / torch.clamp(k, min=1.0)
    res = torch.where(valid & (k > 1), norm_ranks, torch.full_like(x, 0.5))
    return torch.nan_to_num(res, nan=0.5)


def cs_demean_tensor(x: torch.Tensor) -> torch.Tensor:
    """Cross-sectional de-meaning across asset axis (dim=0) using finite masking."""
    if x.ndim <= 1 or x.shape[0] <= 1:
        return torch.zeros_like(x)
    valid = ~torch.isnan(x) & torch.isfinite(x)
    k = valid.sum(dim=0, keepdim=True).to(x.dtype)
    clean_x = torch.where(valid, x, torch.zeros_like(x))
    mean_val = torch.where(
        k > 0,
        clean_x.sum(dim=0, keepdim=True) / torch.clamp(k, min=1.0),
        torch.zeros_like(clean_x),
    )
    res = torch.where(valid & (k > 1), x - mean_val, torch.zeros_like(x))
    return torch.nan_to_num(res, nan=0.0)


def _build_feature_map(
    feature_names: Optional[Union[List[str], Dict[str, int]]]
) -> Dict[str, int]:
    """Constructs dictionary mapping feature names to column indices."""
    if isinstance(feature_names, dict):
        return feature_names
    if isinstance(feature_names, (list, tuple)):
        return {name: i for i, name in enumerate(feature_names)}
    return {name: i for i, name in enumerate(STANDARD_FEATURES)}


def _extract_feature(
    val: str,
    X: torch.Tensor,
    f_map: Dict[str, int],
) -> torch.Tensor:
    """Extracts a feature slice or generates a constant tensor."""
    is_3d = X.ndim == 3
    target_shape = (X.shape[0], X.shape[1]) if is_3d else (X.shape[0],)
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
        const_val = float(val)
        return torch.full(target_shape, const_val, dtype=X.dtype, device=X.device)
    except ValueError:
        return torch.zeros(target_shape, dtype=X.dtype, device=X.device)


def _eval_unary(op: str, child: torch.Tensor) -> torch.Tensor:
    """Evaluates unary operator on PyTorch tensor with finite bounds clamping."""
    if op == "SIGN":
        res = torch.sign(child)
    elif op == "ABS":
        res = torch.abs(child)
    elif op == "LOG_ABS":
        res = torch.log(torch.abs(child) + 1.0)
    elif op == "SIGMOID":
        res = torch.sigmoid(torch.clamp(child, min=-10.0, max=10.0))
    elif op == "RELU":
        res = torch.relu(child)
    elif op == "EXP_SAFE":
        res = torch.exp(torch.clamp(child, min=-20.0, max=20.0))
    elif op == "LOG_SAFE":
        res = torch.log(torch.clamp(torch.abs(child), min=1e-8))
    elif op == "NEG":
        res = -child
    else:
        res = torch.zeros_like(child)
    res = torch.clamp(res, min=-1e4, max=1e4)
    return torch.nan_to_num(res, nan=0.0, posinf=1e4, neginf=-1e4)


def _eval_binary(op: str, l: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Evaluates binary operator on PyTorch tensors."""
    if op == "ADD":
        res = l + r
    elif op == "SUB":
        res = l - r
    elif op == "MUL":
        res = l * r
    elif op == "DIV_SAFE":
        return div_safe_tensor(l, r)
    else:
        res = torch.zeros_like(l)
    return torch.nan_to_num(res, nan=0.0, posinf=1e4, neginf=-1e4)


def _eval_rolling(op: str, child: torch.Tensor) -> torch.Tensor:
    """Evaluates rolling time-series operator along time dimension."""
    if op == "DELTA_5":
        return ts_delta(child, period=5)
    if op == "EMA_10":
        return ts_ema(child, span=10)
    if op == "TS_MAX_10":
        return ts_max(child, window=10)
    if op == "TS_MIN_10":
        return ts_min(child, window=10)
    if op == "TS_RANK_20":
        return ts_rank_20(child, window=20)
    return torch.zeros_like(child)


def _eval_cross_sectional(op: str, child: torch.Tensor) -> torch.Tensor:
    """Evaluates cross-sectional operators across asset dimension."""
    if op == "CS_RANK":
        return cs_rank_tensor(child)
    if op == "CS_DEMEAN":
        return cs_demean_tensor(child)
    return torch.nan_to_num(child, nan=0.0)


class TensorASTEvaluator:
    """Device-agnostic PyTorch AST Evaluator replacing Pandas Series bottleneck."""

    @classmethod
    def eval_node(
        cls,
        node: ASTNode,
        X: torch.Tensor,
        feature_names: Optional[Union[List[str], Dict[str, int]]] = None,
        current_regime: int = 0,
    ) -> torch.Tensor:
        """Evaluates ASTNode over feature tensor X of shape [T, F] or [B, T, F]."""
        is_3d = X.ndim == 3
        shape = (X.shape[0], X.shape[1]) if is_3d else (X.shape[0],)
        if node.is_silenced(current_regime):
            return torch.zeros(shape, dtype=X.dtype, device=X.device)

        f_map = _build_feature_map(feature_names)
        if node.op == "FEATURE":
            return _extract_feature(node.val, X, f_map)

        if node.op == "MEGA_SWITCH":
            cond = cls.eval_node(node.children[0], X, f_map, current_regime) if node.children else torch.zeros(shape, dtype=X.dtype, device=X.device)
            base = cls.eval_node(node.children[1], X, f_map, current_regime) if len(node.children) > 1 else torch.zeros(shape, dtype=X.dtype, device=X.device)
            mega = cls.eval_node(node.children[2], X, f_map, current_regime) if len(node.children) > 2 else torch.zeros(shape, dtype=X.dtype, device=X.device)
            return torch.nan_to_num(torch.where(cond > 0.0, mega, base), nan=0.0)

        if node.op in OPERATORS_BINARY:
            l = cls.eval_node(node.children[0], X, f_map, current_regime) if node.children else torch.zeros(shape, dtype=X.dtype, device=X.device)
            r = cls.eval_node(node.children[1], X, f_map, current_regime) if len(node.children) > 1 else torch.zeros(shape, dtype=X.dtype, device=X.device)
            return _eval_binary(node.op, l, r)

        child = cls.eval_node(node.children[0], X, f_map, current_regime) if node.children else torch.zeros(shape, dtype=X.dtype, device=X.device)
        if node.op in OPERATORS_UNARY:
            return _eval_unary(node.op, child)
        if node.op in OPERATORS_ROLLING:
            return _eval_rolling(node.op, child)
        if node.op in OPERATORS_CROSS_SECTIONAL:
            return _eval_cross_sectional(node.op, child)

        return torch.zeros(shape, dtype=X.dtype, device=X.device)

    def evaluate(
        self,
        node: ASTNode,
        X: torch.Tensor,
        feature_names: Optional[Union[List[str], Dict[str, int]]] = None,
        current_regime: int = 0,
    ) -> torch.Tensor:
        """Instance alias for eval_node."""
        return self.eval_node(node, X, feature_names, current_regime)
