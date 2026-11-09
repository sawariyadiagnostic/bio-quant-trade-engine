"""Unified Compute Dispatcher for RAVANA.

Provides an institutional-grade unified evaluation gateway that automatically routes:
- Single-asset CPU workloads to the ultra-fast C-compiled NumPy engine (ASTEvaluator).
- Multi-asset (3D [B, T, F]) or GPU workloads to the vectorized PyTorch CUDA engine (TensorASTEvaluator).
Guarantees numerical parity, zero-crash bounds, and maximum throughput across all hardware.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import torch

from ravana.domain.formula_ast import ASTNode, ASTEvaluator, STANDARD_FEATURES
from ravana.compute.tensor_evaluator import TensorASTEvaluator


class EvaluationContext:
    """Pre-materialized dual-backend data container caching CPU arrays and GPU tensors once."""

    def __init__(
        self,
        data: Union[pd.DataFrame, torch.Tensor, np.ndarray],
        feature_names: Optional[Union[List[str], Dict[str, int]]] = None,
        device: str = "auto",
        dtype: torch.dtype = torch.float32,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.is_3d = isinstance(data, (torch.Tensor, np.ndarray)) and data.ndim == 3
        num_cols = (
            data.shape[-1]
            if hasattr(data, "shape")
            else (len(data.columns) if isinstance(data, pd.DataFrame) else 0)
        )
        self.feature_names = ComputeDispatcher._extract_feature_cols(feature_names, num_cols)
        self.f_map = {name: i for i, name in enumerate(self.feature_names)}

        if isinstance(data, pd.DataFrame):
            self.df = data
            mat = np.column_stack([data[c].to_numpy(dtype=np.float32) for c in self.feature_names if c in data])
            self.tensor = torch.tensor(mat, dtype=dtype, device=self.device)
            self.cpu_arrays = {
                c: np.nan_to_num(data[c].to_numpy(dtype=np.float64), nan=0.0)
                for c in self.feature_names
                if c in data
            }
        elif isinstance(data, torch.Tensor):
            self.tensor = data.to(dtype=dtype, device=self.device)
            if not self.is_3d:
                mat = self.tensor.cpu().numpy()
                self.cpu_arrays = {
                    c: mat[:, i].astype(np.float64)
                    for i, c in enumerate(self.feature_names)
                    if i < mat.shape[1]
                }
                self.df = pd.DataFrame(self.cpu_arrays)
            else:
                self.cpu_arrays = {}
                self.df = None
        else:  # np.ndarray
            self.tensor = torch.tensor(data, dtype=dtype, device=self.device)
            if not self.is_3d:
                self.cpu_arrays = {
                    c: data[:, i].astype(np.float64)
                    for i, c in enumerate(self.feature_names)
                    if i < data.shape[1]
                }
                self.df = pd.DataFrame(self.cpu_arrays)
            else:
                self.cpu_arrays = {}
                self.df = None


class ComputeDispatcher:
    """Intelligently routes formula evaluation to optimal hardware backend."""

    @staticmethod
    def _extract_feature_cols(
        feature_names: Optional[Union[List[str], Dict[str, int]]],
        num_cols: int,
    ) -> List[str]:
        if isinstance(feature_names, dict):
            # Sort keys by index value
            sorted_keys = sorted(feature_names.keys(), key=lambda k: feature_names[k])
            return sorted_keys[:num_cols]
        if isinstance(feature_names, (list, tuple)):
            return list(feature_names)[:num_cols]
        return STANDARD_FEATURES[:num_cols]

    @classmethod
    def evaluate(
        cls,
        node: ASTNode,
        data: Union[EvaluationContext, pd.DataFrame, torch.Tensor, np.ndarray],
        feature_names: Optional[Union[List[str], Dict[str, int]]] = None,
        backend: str = "auto",  # 'auto', 'numpy', 'torch'
        device: str = "auto",   # 'auto', 'cpu', 'cuda'
    ) -> Union[np.ndarray, torch.Tensor]:
        """Evaluates an AST formula graph using the optimal compute backend."""
        if isinstance(data, EvaluationContext):
            target_device = str(data.device.type) if device == "auto" else device
            if backend == "auto":
                target_backend = "torch" if (target_device == "cuda" or data.is_3d) else "numpy"
            else:
                target_backend = backend

            if target_backend == "numpy" and not data.is_3d and data.df is not None:
                return ASTEvaluator.eval_node(node, data.df)
            return TensorASTEvaluator.eval_node(node, data.tensor, data.f_map)

        # Determine target device
        if device == "auto":
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            target_device = device

        is_3d = (isinstance(data, (torch.Tensor, np.ndarray)) and data.ndim == 3)

        # Determine target backend: 3D multi-asset tensors MUST use PyTorch
        if backend == "auto":
            target_backend = "torch" if (target_device == "cuda" or is_3d) else "numpy"
        else:
            target_backend = backend

        if target_backend == "numpy":
            if isinstance(data, pd.DataFrame):
                return ASTEvaluator.eval_node(node, data)
            if isinstance(data, torch.Tensor):
                arr = data.cpu().numpy()
                if arr.ndim == 3:
                    f_cols = cls._extract_feature_cols(feature_names, arr.shape[-1])
                    return np.stack([
                        ASTEvaluator.eval_node(node, pd.DataFrame(arr[b], columns=f_cols))
                        for b in range(arr.shape[0])
                    ], axis=0)
                f_cols = cls._extract_feature_cols(feature_names, arr.shape[-1])
                return ASTEvaluator.eval_node(node, pd.DataFrame(arr, columns=f_cols))
            if isinstance(data, np.ndarray):
                if data.ndim == 3:
                    f_cols = cls._extract_feature_cols(feature_names, data.shape[-1])
                    return np.stack([
                        ASTEvaluator.eval_node(node, pd.DataFrame(data[b], columns=f_cols))
                        for b in range(data.shape[0])
                    ], axis=0)
                f_cols = cls._extract_feature_cols(feature_names, data.shape[-1])
                return ASTEvaluator.eval_node(node, pd.DataFrame(data, columns=f_cols))

        # PyTorch Tensor backend
        if isinstance(data, torch.Tensor):
            X_tensor = data.to(target_device)
        elif isinstance(data, pd.DataFrame):
            f_cols = cls._extract_feature_cols(feature_names, len(data.columns))
            mat = np.column_stack([data[c].to_numpy(dtype=np.float32) for c in f_cols if c in data])
            X_tensor = torch.tensor(mat, dtype=torch.float32, device=target_device)
        else:
            X_tensor = torch.tensor(data, dtype=torch.float32, device=target_device)

        return TensorASTEvaluator.eval_node(node, X_tensor, feature_names)

    @classmethod
    def evaluate_batch(
        cls,
        nodes: List[ASTNode],
        data: Union[EvaluationContext, pd.DataFrame, torch.Tensor, np.ndarray],
        feature_names: Optional[Union[List[str], Dict[str, int]]] = None,
        backend: str = "auto",
        device: str = "auto",
    ) -> Union[np.ndarray, torch.Tensor]:
        """Evaluates a batch of AST formulas, returning [N_formulas, ...]."""
        if not nodes:
            return np.empty(0) if backend == "numpy" else torch.empty(0)

        # Fast path via EvaluationContext
        if isinstance(data, EvaluationContext):
            target_device = str(data.device.type) if device == "auto" else device
            use_torch = (target_device == "cuda" or data.is_3d) if backend == "auto" else (backend == "torch")
            if use_torch:
                return TensorASTEvaluator.evaluate_batch(nodes, data.tensor, data.f_map)
            if data.df is not None:
                return np.stack([ASTEvaluator.eval_node(n, data.df) for n in nodes], axis=0)

        # If data is already a torch Tensor
        if isinstance(data, torch.Tensor) and (backend == "torch" or data.ndim == 3 or device == "cuda"):
            target_device = "cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device)
            X_tensor = data.to(target_device)
            return TensorASTEvaluator.evaluate_batch(nodes, X_tensor, feature_names)

        results = [cls.evaluate(n, data, feature_names, backend=backend, device=device) for n in nodes]
        if isinstance(results[0], torch.Tensor):
            return torch.stack(results, dim=0)
        return np.stack(results, axis=0)
