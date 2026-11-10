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
from ravana.compute.primitives import (
    div_safe_tensor,
    ts_argmax,
    ts_argmin,
    ts_corr_20,
    ts_decay_linear,
    ts_delay,
    ts_delta,
    ts_ema,
    ts_max,
    ts_median,
    ts_min,
    ts_rank_20,
    ts_std,
    ts_zscore_20,
)


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
        chunk_size: int = 512,
    ) -> torch.Tensor:
        """Evaluates a batch of ASTNode expression trees, returning [N_formulas, ...]."""
        if not nodes:
            is_3d = X.ndim == 3
            empty_shape = (0, X.shape[0], X.shape[1]) if is_3d else (0, X.shape[0])
            return torch.zeros(empty_shape, dtype=X.dtype, device=X.device)
        if len(nodes) <= chunk_size:
            return torch.stack([cls.eval_node(node, X, feature_names) for node in nodes], dim=0)
        chunks: List[torch.Tensor] = []
        for i in range(0, len(nodes), chunk_size):
            batch = nodes[i : i + chunk_size]
            chunks.append(torch.stack([cls.eval_node(n, X, feature_names) for n in batch], dim=0))
        return torch.cat(chunks, dim=0)

    @classmethod
    def evaluate_batch_streaming(
        cls,
        nodes: Sequence[ASTNode],
        X: torch.Tensor,
        feature_names: Optional[Union[List[str], Dict[str, int]]] = None,
        chunk_size: int = 512,
    ):
        """Yields chunks of evaluated formulas sequentially to prevent GPU VRAM exhaustion."""
        for i in range(0, len(nodes), chunk_size):
            batch = nodes[i : i + chunk_size]
            chunk_tensor = torch.stack([cls.eval_node(n, X, feature_names) for n in batch], dim=0)
            yield chunk_tensor

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
        elif node.op == "SIGNED_POWER": res = torch.sign(child) * (torch.abs(child) ** 2.0)
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
        if node.op == "TS_STD_20": return ts_std(child, window=20)
        if node.op == "TS_MEDIAN_10": return ts_median(child, window=10)
        if node.op == "TS_ARGMAX_10": return ts_argmax(child, window=10)
        if node.op == "TS_ARGMIN_10": return ts_argmin(child, window=10)
        return torch.zeros_like(child)

    def evaluate(
        self,
        node: ASTNode,
        X: torch.Tensor,
        feature_names: Optional[Union[List[str], Dict[str, int]]] = None,
    ) -> torch.Tensor:
        """Instance alias for eval_node."""
        return self.eval_node(node, X, feature_names)
