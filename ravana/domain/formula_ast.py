"""Comprehensive Symbolic AST Grammar, Bounded Math Operators, and Genetic Engine for RAVANA.

Includes:
- Bounded zero-crash math operators (DIV_SAFE, EXP_SAFE, LOG_SAFE, TANH, SIGMOID).
- Cross-sectional operators (CS_RANK, CS_DEMEAN) with finite masking.
- MegaEvolutionNode (dual-state conditional regime switcher).
- Arity-hardened genetic crossover and point/subtree/methylation mutation.
- Deterministic canonical hashing invariant to commutative symmetry and active regimes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import random
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
from numpy.lib.stride_tricks import sliding_window_view

OPERATORS_BINARY: Tuple[str, ...] = ("ADD", "SUB", "MUL", "DIV_SAFE")
OPERATORS_UNARY: Tuple[str, ...] = (
    "SIGN", "ABS", "LOG_ABS", "SIGMOID", "TANH", "RELU", "EXP_SAFE", "LOG_SAFE", "NEG"
)
OPERATORS_ROLLING: Tuple[str, ...] = ("DELTA_5", "EMA_10", "TS_MAX_10", "TS_MIN_10", "TS_RANK_20")
OPERATORS_CROSS_SECTIONAL: Tuple[str, ...] = ("CS_RANK", "CS_DEMEAN")

STANDARD_FEATURES: List[str] = [
    "Close_Adj", "Open", "High", "Low", "Close", "Volume", "Spread", "Vol_ZScore", "Vol_Price_Flow",
    "True_Range", "TR_Expansion", "EMA_Slope", "Dist_EMA20", "Dist_VWAP", "Downside_Vol_Ratio",
    "RSI_14", "Parkinson_Vol", "Garman_Klass_Vol", "SP500_Ret", "ADR_Ret", "Global_Confluence"
]


def cs_rank(matrix: np.ndarray) -> np.ndarray:
    """Cross-sectional rank across asset axis (axis=0) normalized to [0, 1] with finite masking."""
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim == 1 or arr.shape[0] <= 1:
        return np.full_like(arr, 0.5)
    valid = np.isfinite(arr)
    k = np.sum(valid, axis=0, keepdims=True)
    ranks = stats.rankdata(arr, axis=0, nan_policy="omit")
    norm_ranks = (ranks - 0.5) / np.maximum(k, 1)
    res = np.where(valid & (k > 1), norm_ranks, 0.5)
    return np.nan_to_num(res, nan=0.5)


def cs_demean(matrix: np.ndarray) -> np.ndarray:
    """Cross-sectional de-meaning across asset axis (axis=0) ensuring sum(alpha)=0."""
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim == 1 or arr.shape[0] <= 1:
        return np.zeros_like(arr)
    valid = np.isfinite(arr)
    k = np.sum(valid, axis=0, keepdims=True)
    clean_arr = np.where(valid, arr, 0.0)
    mean_val = np.where(
        k > 0,
        np.sum(clean_arr, axis=0, keepdims=True) / np.maximum(k, 1),
        0.0,
    )
    res = np.where(valid & (k > 1), arr - mean_val, 0.0)
    return np.nan_to_num(res, nan=0.0)


@dataclass
class ASTNode:
    """Core node of a symbolic trading formula AST.
    
    Attributes:
        op: Operator name ('ADD', 'DIV_SAFE', 'FEATURE', 'CONST', 'MEGA_SWITCH', etc.).
        val: Literal value for features (e.g. 'Close') or constants (e.g. '1.5').
        active_regimes: Set of regime IDs where this node produces signal.
        children: Subnodes for binary, unary, rolling, or switch operations.
    """
    op: str
    val: str = ""
    active_regimes: Set[int] = field(default_factory=lambda: set(range(10)))
    children: List[ASTNode] = field(default_factory=list)

    def is_silenced(self, regime: int) -> bool:
        """Checks if this node is muted under the current market regime."""
        return regime == -1 or regime not in self.active_regimes

    def clone(self) -> ASTNode:
        """Deep copies the node and all of its descendants."""
        return ASTNode(
            op=self.op,
            val=self.val,
            active_regimes=set(self.active_regimes),
            children=[c.clone() for c in self.children]
        )

    def depth(self) -> int:
        """Computes maximum depth of the tree rooted at this node."""
        return 1 + max((c.depth() for c in self.children), default=0)

    def size(self) -> int:
        """Counts total number of nodes in the subtree."""
        return 1 + sum(c.size() for c in self.children)

    def count_nodes(self) -> int:
        """Alias for size() for parsimony calculations."""
        return self.size()

    def get_all_subnodes(self) -> List[ASTNode]:
        """Flattens the entire subtree into a list of nodes."""
        return [self] + [sub for c in self.children for sub in c.get_all_subnodes()]

    def canonical_hash(self) -> str:
        """Deterministic 16-character hex hash invariant to commutative child ordering."""
        regimes_tag = "@R:" + ",".join(str(r) for r in sorted(self.active_regimes))
        if self.op in ("FEATURE", "CONST"):
            raw = f"{self.op}:{self.val}{regimes_tag}"
        elif self.op in ("ADD", "MUL") and len(self.children) == 2:
            h0 = self.children[0].canonical_hash()
            h1 = self.children[1].canonical_hash()
            ordered = sorted([h0, h1])
            raw = f"{self.op}({ordered[0]},{ordered[1]}){regimes_tag}"
        else:
            child_hashes = [c.canonical_hash() for c in self.children]
            raw = f"{self.op}(" + ",".join(child_hashes) + f"){regimes_tag}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def to_formula_str(self) -> str:
        """Converts AST into human-readable mathematical notation."""
        if self.op in ("FEATURE", "CONST"):
            return self.val
        if self.op == "MEGA_SWITCH":
            c = self.children[0].to_formula_str() if len(self.children) > 0 else "0"
            b = self.children[1].to_formula_str() if len(self.children) > 1 else "0"
            m = self.children[2].to_formula_str() if len(self.children) > 2 else "0"
            return f"MEGA_SWITCH({c}, {b}, {m})"
        if self.op in (OPERATORS_ROLLING + OPERATORS_UNARY + OPERATORS_CROSS_SECTIONAL):
            arg = self.children[0].to_formula_str() if self.children else "X"
            return f"{self.op}({arg})"
        sym = {"ADD": "+", "SUB": "-", "MUL": "*", "DIV_SAFE": "/"}.get(self.op, self.op)
        left = self.children[0].to_formula_str() if self.children else "0"
        right = self.children[1].to_formula_str() if len(self.children) > 1 else "0"
        return f"({left} {sym} {right})"

    def to_dict(self) -> Dict[str, Any]:
        """Serializes AST node and all children into a JSON-compatible dictionary."""
        return {
            "op": self.op,
            "val": self.val,
            "active_regimes": sorted(list(self.active_regimes)),
            "children": [c.to_dict() for c in self.children]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ASTNode:
        """Deserializes AST node hierarchy from a dictionary."""
        return cls(
            op=data["op"],
            val=data.get("val", ""),
            active_regimes=set(data.get("active_regimes", range(10))),
            children=[cls.from_dict(c) for c in data.get("children", [])]
        )


class MegaEvolutionNode(ASTNode):
    """Dual-state AST node that dynamically transitions between base and mega forms."""
    def __init__(self, condition: ASTNode, base: ASTNode, mega: ASTNode, active_regimes: Optional[Set[int]] = None):
        regimes = active_regimes if active_regimes is not None else set(range(10))
        super().__init__(op="MEGA_SWITCH", val="", active_regimes=regimes, children=[condition, base, mega])

    def clone(self) -> MegaEvolutionNode:
        return MegaEvolutionNode(
            condition=self.children[0].clone(),
            base=self.children[1].clone(),
            mega=self.children[2].clone(),
            active_regimes=set(self.active_regimes)
        )


def _fast_ts_rank_20(arr: np.ndarray) -> np.ndarray:
    """Sliding-window percentile rank over 20 periods with finite masking."""
    n = len(arr)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    res = np.full(n, 0.5, dtype=np.float64)
    w = 20
    for i in range(min(n, w)):
        prefix = arr[:i + 1]
        valid_prefix = np.isfinite(prefix)
        k = np.sum(valid_prefix)
        target = arr[i]
        if np.isfinite(target) and k > 0:
            less = np.count_nonzero((prefix < target) & valid_prefix)
            eq = np.count_nonzero((prefix == target) & valid_prefix)
            res[i] = (less + 0.5 * eq) / float(max(k, 1))
    if n > w:
        windows = sliding_window_view(arr, window_shape=w)
        target = arr[w - 1:]
        valid_win = np.isfinite(windows)
        valid_tgt = np.isfinite(target)
        k = np.sum(valid_win, axis=1)
        less = np.count_nonzero((windows < target[:, None]) & valid_win, axis=1)
        eq = np.count_nonzero((windows == target[:, None]) & valid_win, axis=1)
        calc_rank = (less + 0.5 * eq) / np.maximum(k, 1).astype(np.float64)
        res[w - 1:] = np.where(valid_tgt & (k > 0), calc_rank, 0.5)
    return np.nan_to_num(res, nan=0.5)


class ASTEvaluator:
    """Evaluates an ASTNode hierarchy against a feature DataFrame safely."""

    @staticmethod
    def eval_node(node: ASTNode, df: pd.DataFrame, current_regime: int = 0) -> np.ndarray:
        n = len(df)
        if node.is_silenced(current_regime):
            return np.zeros(n, dtype=np.float64)

        if node.op == "CONST":
            return ASTEvaluator._eval_const(node, n)
        if node.op == "FEATURE":
            return ASTEvaluator._eval_feature(node, df, n)
        if node.op == "MEGA_SWITCH":
            return ASTEvaluator._eval_mega_switch(node, df, current_regime, n)
        if node.op in OPERATORS_BINARY:
            return ASTEvaluator._eval_binary(node, df, current_regime, n)
        if node.op in OPERATORS_UNARY:
            return ASTEvaluator._eval_unary(node, df, current_regime, n)
        if node.op in OPERATORS_ROLLING:
            return ASTEvaluator._eval_rolling(node, df, current_regime, n)
        if node.op in OPERATORS_CROSS_SECTIONAL:
            return ASTEvaluator._eval_cross_sectional(node, df, current_regime, n)
        return np.zeros(n, dtype=np.float64)

    @staticmethod
    def _eval_const(node: ASTNode, n: int) -> np.ndarray:
        try:
            return np.full(n, float(node.val), dtype=np.float64)
        except (ValueError, TypeError):
            return np.zeros(n, dtype=np.float64)

    @staticmethod
    def _eval_feature(node: ASTNode, df: pd.DataFrame, n: int) -> np.ndarray:
        cache = df.attrs.get("_col_arrays")
        if cache is None:
            cache = {
                c: np.nan_to_num(df[c].to_numpy(dtype=np.float64), nan=0.0, posinf=1.0, neginf=-1.0)
                for c in df.columns
            }
            df.attrs["_col_arrays"] = cache
        if node.val in cache:
            return cache[node.val]
        if node.val == "Spread":
            h = cache.get("High", df["High"].to_numpy(dtype=np.float64) if "High" in df.columns else np.zeros(n, dtype=np.float64))
            l = cache.get("Low", df["Low"].to_numpy(dtype=np.float64) if "Low" in df.columns else np.zeros(n, dtype=np.float64))
            c = cache.get("Close_Adj", cache.get("Close", np.ones(n, dtype=np.float64)))
            return np.nan_to_num((h - l) / (c + 1e-8), nan=0.0)
        try:
            return np.full(n, float(node.val), dtype=np.float64)
        except (ValueError, TypeError):
            return np.zeros(n, dtype=np.float64)

    @staticmethod
    def _eval_mega_switch(node: ASTNode, df: pd.DataFrame, regime: int, n: int) -> np.ndarray:
        cond = ASTEvaluator.eval_node(node.children[0], df, regime) if len(node.children) > 0 else np.zeros(n)
        base = ASTEvaluator.eval_node(node.children[1], df, regime) if len(node.children) > 1 else np.zeros(n)
        mega = ASTEvaluator.eval_node(node.children[2], df, regime) if len(node.children) > 2 else np.zeros(n)
        return np.nan_to_num(np.where(cond > 0.0, mega, base), nan=0.0)

    @staticmethod
    def _eval_binary(node: ASTNode, df: pd.DataFrame, regime: int, n: int) -> np.ndarray:
        l = ASTEvaluator.eval_node(node.children[0], df, regime) if node.children else np.zeros(n)
        r = ASTEvaluator.eval_node(node.children[1], df, regime) if len(node.children) > 1 else np.zeros(n)
        if node.op == "ADD":
            res = l + r
        elif node.op == "SUB":
            res = l - r
        elif node.op == "MUL":
            res = l * r
        elif node.op == "DIV_SAFE":
            r_abs = np.abs(r)
            denom_sign = np.where(r >= 0.0, 1.0, -1.0)
            safe_denom = denom_sign * np.maximum(r_abs, 1e-6)
            res = np.where(r_abs > 1e-6, l / safe_denom, 0.0)
            res = np.clip(res, -1e4, 1e4)
        else:
            res = np.zeros(n)
        return np.nan_to_num(res, nan=0.0, posinf=1e4, neginf=-1e4)

    @staticmethod
    def _eval_unary(node: ASTNode, df: pd.DataFrame, regime: int, n: int) -> np.ndarray:
        child = ASTEvaluator.eval_node(node.children[0], df, regime) if node.children else np.zeros(n)
        if node.op == "SIGN": res = np.sign(child)
        elif node.op == "ABS": res = np.abs(child)
        elif node.op == "LOG_ABS": res = np.log1p(np.abs(child))
        elif node.op == "SIGMOID": res = 1.0 / (1.0 + np.exp(-np.clip(child, -10.0, 10.0)))
        elif node.op == "TANH": res = np.tanh(np.clip(child, -10.0, 10.0))
        elif node.op == "RELU": res = np.maximum(0.0, child)
        elif node.op == "EXP_SAFE": res = np.exp(np.clip(child, -20.0, 20.0))
        elif node.op == "LOG_SAFE": res = np.log(np.maximum(1e-8, np.abs(child)))
        elif node.op == "NEG": res = -child
        else: res = np.zeros(n)
        res = np.clip(res, -1e4, 1e4)
        return np.nan_to_num(res, nan=0.0, posinf=1e4, neginf=-1e4)

    @staticmethod
    def _eval_rolling(node: ASTNode, df: pd.DataFrame, regime: int, n: int) -> np.ndarray:
        child = ASTEvaluator.eval_node(node.children[0], df, regime) if node.children else np.zeros(n)
        if node.op == "TS_RANK_20":
            return _fast_ts_rank_20(child)
        s = pd.Series(child)
        if node.op == "DELTA_5":
            res = s.diff(5).fillna(0.0).to_numpy()
        elif node.op == "EMA_10":
            res = s.ewm(span=10, adjust=False).mean().to_numpy()
        elif node.op == "TS_MAX_10":
            res = s.rolling(10, min_periods=1).max().to_numpy()
        elif node.op == "TS_MIN_10":
            res = s.rolling(10, min_periods=1).min().to_numpy()
        else:
            res = np.zeros(n)
        return np.nan_to_num(res, nan=0.0)

    @staticmethod
    def _eval_cross_sectional(node: ASTNode, df: pd.DataFrame, regime: int, n: int) -> np.ndarray:
        child = ASTEvaluator.eval_node(node.children[0], df, regime) if node.children else np.zeros(n)
        if node.op == "CS_RANK":
            return cs_rank(child)
        elif node.op == "CS_DEMEAN":
            return cs_demean(child)
        return np.nan_to_num(child, nan=0.0)


class TreeGenerator:
    """Randomly generates AST expression trees with bounded depth."""

    @staticmethod
    def generate(
        max_depth: int = 4,
        features: Optional[List[str]] = None,
        active_regimes: Optional[Set[int]] = None,
        method: str = "grow"
    ) -> ASTNode:
        feat_list = features if features is not None else STANDARD_FEATURES
        regimes = active_regimes if active_regimes is not None else set(range(10))
        if max_depth <= 1 or (method == "grow" and random.random() < 0.35):
            if random.random() < 0.15:
                const_val = str(round(random.uniform(-2.0, 2.0), 2))
                return ASTNode("CONST", const_val, regimes)
            return ASTNode("FEATURE", random.choice(feat_list), regimes)

        op = random.choice(OPERATORS_BINARY + OPERATORS_UNARY + OPERATORS_ROLLING)
        num_children = 2 if op in OPERATORS_BINARY else 1
        children = [
            TreeGenerator.generate(max_depth - 1, feat_list, regimes, method)
            for _ in range(num_children)
        ]
        return ASTNode(op, active_regimes=regimes, children=children)


class GeneticMutator:
    """Applies point, subtree, and epigenetic methylation mutations to an AST tree."""

    @staticmethod
    def mutate_subtree(
        tree: ASTNode,
        max_depth: int = 3,
        features: Optional[List[str]] = None
    ) -> ASTNode:
        cloned = tree.clone()
        all_nodes = cloned.get_all_subnodes()
        target = random.choice(all_nodes)
        sub = TreeGenerator.generate(max_depth, features, target.active_regimes)
        target.op, target.val, target.children = sub.op, sub.val, sub.children
        return cloned

    @staticmethod
    def mutate_point(
        tree: ASTNode,
        features: Optional[List[str]] = None
    ) -> ASTNode:
        """Mutates a single node operator or feature without replacing subtree."""
        cloned = tree.clone()
        feat_list = features if features is not None else STANDARD_FEATURES
        target = random.choice(cloned.get_all_subnodes())
        if target.op == "FEATURE":
            target.val = random.choice(feat_list)
        elif target.op == "CONST":
            target.val = str(round(random.uniform(-2.0, 2.0), 2))
        elif target.op in OPERATORS_BINARY:
            target.op = random.choice(OPERATORS_BINARY)
        elif target.op in OPERATORS_UNARY:
            target.op = random.choice(OPERATORS_UNARY)
        elif target.op in OPERATORS_ROLLING:
            target.op = random.choice(OPERATORS_ROLLING)
        return cloned

    @staticmethod
    def mutate_methylation(tree: ASTNode, num_regimes: int = 10) -> ASTNode:
        """Epigenetic mutation toggling regime activation on a random subnode."""
        cloned = tree.clone()
        target = random.choice(cloned.get_all_subnodes())
        r = random.randint(0, num_regimes - 1)
        if r in target.active_regimes and len(target.active_regimes) > 1:
            target.active_regimes.remove(r)
        else:
            target.active_regimes.add(r)
        return cloned


class GeneticCrossover:
    """Performs genetic crossover with arity hardening to guarantee valid child ASTs."""

    @staticmethod
    def crossover(parent1: ASTNode, parent2: ASTNode) -> Tuple[ASTNode, ASTNode]:
        c1, c2 = parent1.clone(), parent2.clone()
        t1, t2 = random.choice(c1.get_all_subnodes()), random.choice(c2.get_all_subnodes())
        t1.op, t2.op = t2.op, t1.op
        t1.val, t2.val = t2.val, t1.val
        t1.active_regimes, t2.active_regimes = set(t2.active_regimes), set(t1.active_regimes)
        t1.children, t2.children = [c.clone() for c in t2.children], [c.clone() for c in t1.children]

        # Arity hardening to prevent deformed AST structures during chimeric swaps
        for node in (t1, t2):
            if node.op in OPERATORS_BINARY:
                while len(node.children) < 2:
                    node.children.append(ASTNode("CONST", val="1.0", active_regimes=set(node.active_regimes)))
                if len(node.children) > 2:
                    node.children = node.children[:2]
            elif node.op in (OPERATORS_UNARY + OPERATORS_ROLLING + OPERATORS_CROSS_SECTIONAL):
                if len(node.children) == 0:
                    node.children.append(ASTNode("CONST", val="1.0", active_regimes=set(node.active_regimes)))
                elif len(node.children) > 1:
                    node.children = node.children[:1]
            elif node.op in ("FEATURE", "CONST"):
                node.children = []
        return c1, c2
