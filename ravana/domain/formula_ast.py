"""Institutional Quantitative AST Grammar & Bounded Math Operators for RAVANA.

Pruned of biological analogies (methylation, fake regimes) and single-asset degeneracies (CS_RANK).
Enriched with essential quantitative primitives (TS_ZSCORE, TS_CORR, TS_DECAY_LINEAR, SIGNED_POWER, BLEND)
and dimensional consistency rules preventing dimensional nonsense (e.g. Price + Volume).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import enum
import hashlib
import random
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# Operator classifications
OPERATORS_BINARY_ARITHMETIC: Tuple[str, ...] = ("ADD", "SUB", "MUL", "DIV_SAFE")
OPERATORS_BINARY_ROLLING: Tuple[str, ...] = ("TS_CORR_20",)
OPERATORS_BINARY: Tuple[str, ...] = OPERATORS_BINARY_ARITHMETIC + OPERATORS_BINARY_ROLLING

OPERATORS_UNARY: Tuple[str, ...] = (
    "SIGN", "ABS", "LOG_ABS", "SIGMOID", "TANH", "RELU", "EXP_SAFE", "LOG_SAFE", "NEG", "SQRT_ABS",
    "SIGNED_POWER"
)
OPERATORS_ROLLING_UNARY: Tuple[str, ...] = (
    "DELTA_5", "EMA_10", "TS_MAX_10", "TS_MIN_10", "TS_RANK_20", "TS_ZSCORE_20", "TS_DECAY_LINEAR_10",
    "TS_DELAY_1", "TS_DELAY_5", "TS_STD_20", "TS_MEDIAN_10", "TS_ARGMAX_10", "TS_ARGMIN_10"
)
OPERATORS_TERNARY: Tuple[str, ...] = ("BLEND",)


class FeatureDimension(enum.Enum):
    """Physical dimensions for financial time-series to enforce dimensional integrity."""
    PRICE = "PRICE"              # In Rupees (Open, High, Low, Close)
    VOLUME = "VOLUME"            # In Shares (Volume)
    DIMENSIONLESS = "DIMENSIONLESS"  # Returns, Ratios, Z-scores, RSI, Indicators


FEATURE_DIMENSIONS: Dict[str, FeatureDimension] = {
    "Open": FeatureDimension.PRICE,
    "High": FeatureDimension.PRICE,
    "Low": FeatureDimension.PRICE,
    "Close": FeatureDimension.PRICE,
    "Close_Adj": FeatureDimension.PRICE,
    "Volume": FeatureDimension.VOLUME,
    "Spread": FeatureDimension.DIMENSIONLESS,
    "Vol_ZScore": FeatureDimension.DIMENSIONLESS,
    "Vol_Price_Flow": FeatureDimension.DIMENSIONLESS,
    "True_Range": FeatureDimension.PRICE,
    "TR_Expansion": FeatureDimension.DIMENSIONLESS,
    "EMA_Slope": FeatureDimension.DIMENSIONLESS,
    "Dist_EMA20": FeatureDimension.DIMENSIONLESS,
    "Dist_VWAP": FeatureDimension.DIMENSIONLESS,
    "Downside_Vol_Ratio": FeatureDimension.DIMENSIONLESS,
    "RSI_14": FeatureDimension.DIMENSIONLESS,
    "Parkinson_Vol": FeatureDimension.DIMENSIONLESS,
    "Garman_Klass_Vol": FeatureDimension.DIMENSIONLESS,
    "SP500_Ret": FeatureDimension.DIMENSIONLESS,
    "ADR_Ret": FeatureDimension.DIMENSIONLESS,
    "Global_Confluence": FeatureDimension.DIMENSIONLESS,
    # Muscle 7 Stationary Vocabulary (21 Features)
    "LOG_RET_1": FeatureDimension.DIMENSIONLESS,
    "LOG_RET_5": FeatureDimension.DIMENSIONLESS,
    "LOG_RET_20": FeatureDimension.DIMENSIONLESS,
    "ATR_14_NORM": FeatureDimension.DIMENSIONLESS,
    "VOLATILITY_20": FeatureDimension.DIMENSIONLESS,
    "PARKINSON_VOL_20": FeatureDimension.DIMENSIONLESS,
    "GARMAN_KLASS_VOL_20": FeatureDimension.DIMENSIONLESS,
    "VOLUME_ZSCORE_LAGGED": FeatureDimension.DIMENSIONLESS,
    "VWAP_DRIFT": FeatureDimension.DIMENSIONLESS,
    "BOLLINGER_BAND_PCT_20": FeatureDimension.DIMENSIONLESS,
    "MACD_DIFF_NORM": FeatureDimension.DIMENSIONLESS,
    "SPREAD_BPS": FeatureDimension.DIMENSIONLESS,
    "SPREAD_ZSCORE_20": FeatureDimension.DIMENSIONLESS,
    "SPREAD_DELTA_5": FeatureDimension.DIMENSIONLESS,
    "ADF_STAT_60": FeatureDimension.DIMENSIONLESS,
    "OU_HALF_LIFE_60": FeatureDimension.DIMENSIONLESS,
    "MACRO_VIX_LEVEL": FeatureDimension.DIMENSIONLESS,
    "MACRO_VIX_RET_1": FeatureDimension.DIMENSIONLESS,
    "MACRO_USDINR_RET_1": FeatureDimension.DIMENSIONLESS,
    "MACRO_SP500_RET_1": FeatureDimension.DIMENSIONLESS,
}

STANDARD_FEATURES: List[str] = [
    "Open", "High", "Low", "Close", "Close_Adj", "Volume",
    "Spread", "Vol_ZScore", "Vol_Price_Flow", "True_Range",
    "TR_Expansion", "EMA_Slope", "Dist_EMA20", "Dist_VWAP",
    "Downside_Vol_Ratio", "RSI_14", "Parkinson_Vol", "Garman_Klass_Vol",
    "SP500_Ret", "ADR_Ret", "Global_Confluence",
]


@dataclass
class ASTNode:
    """Core node of a symbolic trading formula AST.
    
    Attributes:
        op: Operator name ('ADD', 'DIV_SAFE', 'FEATURE', 'CONST', 'BLEND', 'TS_CORR_20', etc.).
        val: Literal value for features (e.g. 'Close') or constants (e.g. '1.5').
        children: Subnodes for binary, unary, rolling, or ternary operations.
    """
    op: str
    val: str = ""
    children: List[ASTNode] = field(default_factory=list)

    def clone(self) -> ASTNode:
        """Deep copies the node and all of its descendants."""
        return ASTNode(
            op=self.op,
            val=self.val,
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
        if self.op in ("FEATURE", "CONST"):
            raw = f"{self.op}:{self.val}"
        elif self.op in ("ADD", "MUL", "TS_CORR_20") and len(self.children) == 2:
            h0 = self.children[0].canonical_hash()
            h1 = self.children[1].canonical_hash()
            ordered = sorted([h0, h1])
            raw = f"{self.op}({ordered[0]},{ordered[1]})"
        else:
            child_hashes = [c.canonical_hash() for c in self.children]
            raw = f"{self.op}(" + ",".join(child_hashes) + ")"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def to_formula_str(self) -> str:
        """Converts AST into human-readable mathematical notation."""
        if self.op in ("FEATURE", "CONST"):
            return self.val
        if self.op == "BLEND" and len(self.children) >= 3:
            c = self.children[0].to_formula_str()
            a = self.children[1].to_formula_str()
            b = self.children[2].to_formula_str()
            return f"BLEND({c}, {a}, {b})"
        if self.op in OPERATORS_BINARY_ROLLING and len(self.children) >= 2:
            a = self.children[0].to_formula_str()
            b = self.children[1].to_formula_str()
            return f"{self.op}({a}, {b})"
        if self.op in (OPERATORS_ROLLING_UNARY + OPERATORS_UNARY):
            arg = self.children[0].to_formula_str() if self.children else "X"
            return f"{self.op}({arg})"
        sym = {"ADD": "+", "SUB": "-", "MUL": "*", "DIV_SAFE": "/"}.get(self.op, self.op)
        left = self.children[0].to_formula_str() if self.children else "0"
        right = self.children[1].to_formula_str() if len(self.children) > 1 else "0"
        return f"({left} {sym} {right})"

    def to_infix(self) -> str:
        """Alias for to_formula_str() conforming to standard compiler AST conventions."""
        return self.to_formula_str()

    def to_dict(self) -> Dict[str, Any]:
        """Serializes AST node and all children into a JSON-compatible dictionary."""
        return {
            "op": self.op,
            "val": self.val,
            "children": [c.to_dict() for c in self.children]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ASTNode:
        """Deserializes AST node hierarchy from a dictionary."""
        return cls(
            op=data["op"],
            val=data.get("val", ""),
            children=[cls.from_dict(c) for c in data.get("children", [])]
        )

    def simplify(self) -> ASTNode:
        """Recursively simplifies algebraic expressions, prunes introns, and folds constants."""
        self.children = [c.simplify() for c in self.children]
        if self.op in OPERATORS_UNARY or self.op in OPERATORS_ROLLING_UNARY:
            return self._simplify_unary()
        if self.op in OPERATORS_BINARY_ARITHMETIC:
            return self._simplify_binary()
        if self.op == "BLEND" and len(self.children) == 3:
            if self.children[1].canonical_hash() == self.children[2].canonical_hash():
                return self.children[1]
        return self

    def _simplify_unary(self) -> ASTNode:
        """Prunes constant rolling operations and unary involutions."""
        if not self.children:
            return self
        child = self.children[0]
        if child.op == "CONST":
            try:
                cv = float(child.val)
                if self.op in ("DELTA_5", "TS_ZSCORE_20", "TS_STD_20"):
                    return ASTNode("CONST", val="0.0")
                if self.op in ("TS_RANK_20", "TS_ARGMAX_10", "TS_ARGMIN_10"):
                    return ASTNode("CONST", val="0.5")
                if self.op in ("TS_DELAY_1", "TS_DELAY_5", "EMA_10", "TS_MAX_10", "TS_MIN_10", "TS_MEDIAN_10", "TS_DECAY_LINEAR_10"):
                    return ASTNode("CONST", val=f"{cv:.4g}")
                if self.op == "SIGN": res = 1.0 if cv > 0 else (-1.0 if cv < 0 else 0.0)
                elif self.op == "ABS": res = abs(cv)
                elif self.op == "NEG": res = -cv
                elif self.op == "RELU": res = max(0.0, cv)
                elif self.op == "SQRT_ABS": res = np.sign(cv) * np.sqrt(abs(cv))
                elif self.op == "SIGNED_POWER": res = np.sign(cv) * (abs(cv) ** 2.0)
                elif self.op == "EXP_SAFE": res = np.exp(np.clip(cv, -20.0, 20.0))
                elif self.op == "LOG_SAFE": res = np.log(max(abs(cv), 1e-8))
                elif self.op == "SIGMOID": res = 1.0 / (1.0 + np.exp(-np.clip(cv, -10.0, 10.0)))
                elif self.op == "TANH": res = np.tanh(np.clip(cv, -10.0, 10.0))
                else: res = cv
                return ASTNode("CONST", val=f"{res:.4g}")
            except Exception:
                pass
        if self.op == "NEG" and child.op == "NEG" and child.children:
            return child.children[0]
        if self.op in ("ABS", "SIGN", "RELU") and child.op == self.op:
            return child
        return self

    def _simplify_binary(self) -> ASTNode:
        """Folds constants and simplifies identities for binary arithmetic."""
        if len(self.children) < 2:
            return self
        l, r = self.children[0], self.children[1]
        if l.op == "CONST" and r.op == "CONST":
            try:
                lv, rv = float(l.val), float(r.val)
                if self.op == "ADD": res = lv + rv
                elif self.op == "SUB": res = lv - rv
                elif self.op == "MUL": res = lv * rv
                elif self.op == "DIV_SAFE": res = lv / rv if abs(rv) > 1e-8 else 0.0
                else: res = 0.0
                return ASTNode("CONST", val=f"{res:.4g}")
            except Exception:
                pass
        if self.op == "SUB" and l.canonical_hash() == r.canonical_hash():
            return ASTNode("CONST", val="0.0")
        if self.op == "DIV_SAFE" and l.canonical_hash() == r.canonical_hash():
            return ASTNode("CONST", val="1.0")

        try:
            if self.op == "ADD":
                if l.op == "CONST" and float(l.val or 0) == 0.0: return r
                if r.op == "CONST" and float(r.val or 0) == 0.0: return l
            if self.op == "SUB" and r.op == "CONST" and float(r.val or 0) == 0.0:
                return l
            if self.op == "MUL":
                if (l.op == "CONST" and float(l.val or 0) == 0.0) or (r.op == "CONST" and float(r.val or 0) == 0.0):
                    return ASTNode("CONST", val="0.0")
                if l.op == "CONST" and float(l.val or 0) == 1.0: return r
                if r.op == "CONST" and float(r.val or 0) == 1.0: return l
            if self.op == "DIV_SAFE" and r.op == "CONST" and float(r.val or 0) == 1.0:
                return l
        except (ValueError, TypeError):
            pass
        return self


# --- Vectorized Fast Math Primitives ---

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


def _fast_ts_zscore(arr: np.ndarray, w: int = 20) -> np.ndarray:
    """Computes rolling z-score (X - mean) / (std + 1e-6) clamped to [-4.0, 4.0]."""
    s = pd.Series(arr)
    mean = s.rolling(w, min_periods=2).mean()
    std = s.rolling(w, min_periods=2).std(ddof=1)
    z = (s - mean) / (std + 1e-6)
    return np.nan_to_num(np.clip(z.fillna(0.0).to_numpy(), -4.0, 4.0), nan=0.0)


def _fast_ts_decay_linear(arr: np.ndarray, w: int = 10) -> np.ndarray:
    """Linear time-decay weighted moving average (recent bars weighted higher)."""
    weights = np.arange(1, w + 1, dtype=np.float64)
    weights /= weights.sum()
    s = pd.Series(arr)
    res = s.rolling(w, min_periods=1).apply(lambda x: np.dot(x[-len(weights):], weights[-len(x):]) / weights[-len(x):].sum(), raw=True)
    return np.nan_to_num(res.fillna(0.0).to_numpy(), nan=0.0)


def _fast_ts_corr(x: np.ndarray, y: np.ndarray, w: int = 20) -> np.ndarray:
    """Rolling Pearson correlation clamped to [-1.0, 1.0]."""
    sx = pd.Series(x)
    sy = pd.Series(y)
    corr = sx.rolling(w, min_periods=3).corr(sy)
    return np.nan_to_num(np.clip(corr.fillna(0.0).to_numpy(), -1.0, 1.0), nan=0.0)


class ASTEvaluator:
    """Evaluates an ASTNode hierarchy against a feature DataFrame safely."""

    @staticmethod
    def eval_node(node: ASTNode, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        if node.op == "CONST":
            return ASTEvaluator._eval_const(node, n)
        if node.op == "FEATURE":
            return ASTEvaluator._eval_feature(node, df, n)
        if node.op == "BLEND":
            return ASTEvaluator._eval_blend(node, df, n)
        if node.op in OPERATORS_BINARY_ROLLING:
            return ASTEvaluator._eval_binary_rolling(node, df, n)
        if node.op in OPERATORS_BINARY_ARITHMETIC:
            return ASTEvaluator._eval_binary(node, df, n)
        if node.op in OPERATORS_UNARY:
            return ASTEvaluator._eval_unary(node, df, n)
        if node.op in OPERATORS_ROLLING_UNARY:
            return ASTEvaluator._eval_rolling(node, df, n)
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
        if cache is None or cache.get("_len") != n or cache.get("_df_id") != id(df):
            cache = {
                c: np.nan_to_num(df[c].to_numpy(dtype=np.float64), nan=0.0, posinf=1.0, neginf=-1.0)
                for c in df.columns
            }
            cache["_len"] = n
            cache["_df_id"] = id(df)
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
    def _eval_blend(node: ASTNode, df: pd.DataFrame, n: int) -> np.ndarray:
        cond = ASTEvaluator.eval_node(node.children[0], df) if len(node.children) > 0 else np.zeros(n)
        a = ASTEvaluator.eval_node(node.children[1], df) if len(node.children) > 1 else np.zeros(n)
        b = ASTEvaluator.eval_node(node.children[2], df) if len(node.children) > 2 else np.zeros(n)
        weight = 1.0 / (1.0 + np.exp(-np.clip(cond, -10.0, 10.0)))
        res = weight * a + (1.0 - weight) * b
        return np.nan_to_num(np.clip(res, -1e4, 1e4), nan=0.0)

    @staticmethod
    def _eval_binary(node: ASTNode, df: pd.DataFrame, n: int) -> np.ndarray:
        l = ASTEvaluator.eval_node(node.children[0], df) if node.children else np.zeros(n)
        r = ASTEvaluator.eval_node(node.children[1], df) if len(node.children) > 1 else np.zeros(n)
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
    def _eval_binary_rolling(node: ASTNode, df: pd.DataFrame, n: int) -> np.ndarray:
        l = ASTEvaluator.eval_node(node.children[0], df) if node.children else np.zeros(n)
        r = ASTEvaluator.eval_node(node.children[1], df) if len(node.children) > 1 else np.zeros(n)
        if node.op == "TS_CORR_20":
            return _fast_ts_corr(l, r, w=20)
        return np.zeros(n, dtype=np.float64)

    @staticmethod
    def _eval_unary(node: ASTNode, df: pd.DataFrame, n: int) -> np.ndarray:
        child = ASTEvaluator.eval_node(node.children[0], df) if node.children else np.zeros(n)
        if node.op == "SIGN": res = np.sign(child)
        elif node.op == "ABS": res = np.abs(child)
        elif node.op == "LOG_ABS": res = np.log1p(np.abs(child))
        elif node.op == "SIGMOID": res = 1.0 / (1.0 + np.exp(-np.clip(child, -10.0, 10.0)))
        elif node.op == "TANH": res = np.tanh(np.clip(child, -10.0, 10.0))
        elif node.op == "RELU": res = np.maximum(0.0, child)
        elif node.op == "EXP_SAFE": res = np.exp(np.clip(child, -20.0, 20.0))
        elif node.op == "LOG_SAFE": res = np.log(np.maximum(1e-8, np.abs(child)))
        elif node.op == "NEG": res = -child
        elif node.op == "SQRT_ABS": res = np.sign(child) * np.sqrt(np.abs(child))
        elif node.op == "SIGNED_POWER": res = np.sign(child) * (np.abs(child) ** 2.0)
        else: res = np.zeros(n)
        res = np.clip(res, -1e4, 1e4)
        return np.nan_to_num(res, nan=0.0, posinf=1e4, neginf=-1e4)

    @staticmethod
    def _eval_rolling(node: ASTNode, df: pd.DataFrame, n: int) -> np.ndarray:
        child = ASTEvaluator.eval_node(node.children[0], df) if node.children else np.zeros(n)
        if node.op == "TS_RANK_20":
            return _fast_ts_rank_20(child)
        if node.op == "TS_ZSCORE_20":
            return _fast_ts_zscore(child, w=20)
        if node.op == "TS_DECAY_LINEAR_10":
            return _fast_ts_decay_linear(child, w=10)
        s = pd.Series(child)
        if node.op == "TS_DELAY_1":
            return s.shift(1).bfill().fillna(0.0).to_numpy()
        elif node.op == "TS_DELAY_5":
            return s.shift(5).bfill().fillna(0.0).to_numpy()
        elif node.op == "DELTA_5":
            res = s.diff(5).fillna(0.0).to_numpy()
        elif node.op == "EMA_10":
            res = s.ewm(span=10, adjust=False).mean().to_numpy()
        elif node.op == "TS_MAX_10":
            res = s.rolling(10, min_periods=1).max().to_numpy()
        elif node.op == "TS_MIN_10":
            res = s.rolling(10, min_periods=1).min().to_numpy()
        elif node.op == "TS_STD_20":
            res = s.rolling(20, min_periods=1).std(ddof=1).fillna(0.0).to_numpy()
        elif node.op == "TS_MEDIAN_10":
            res = s.rolling(10, min_periods=1).median().to_numpy()
        elif node.op == "TS_ARGMAX_10":
            res = s.rolling(10, min_periods=1).apply(lambda w: float(np.argmax(w)) / max(len(w) - 1, 1), raw=True).fillna(0.5).to_numpy()
        elif node.op == "TS_ARGMIN_10":
            res = s.rolling(10, min_periods=1).apply(lambda w: float(np.argmin(w)) / max(len(w) - 1, 1), raw=True).fillna(0.5).to_numpy()
        else:
            res = np.zeros(n)
        return np.nan_to_num(res, nan=0.0)


# Re-export genetic generation and search operators for backward compatibility
from ravana.domain.genetics import TreeGenerator, GeneticMutator, GeneticCrossover

__all__ = [
    "ASTNode",
    "ASTEvaluator",
    "FeatureDimension",
    "FEATURE_DIMENSIONS",
    "OPERATORS_BINARY",
    "OPERATORS_BINARY_ARITHMETIC",
    "OPERATORS_BINARY_ROLLING",
    "OPERATORS_UNARY",
    "OPERATORS_ROLLING_UNARY",
    "OPERATORS_TERNARY",
    "TreeGenerator",
    "GeneticMutator",
    "GeneticCrossover",
]
