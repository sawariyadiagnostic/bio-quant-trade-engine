"""64-Byte Cache-Aligned Epigenetic AST Evaluator & Genetic Operators (Ponytail MVC)."""
from dataclasses import dataclass, field
import hashlib
import random
import numpy as np
import pandas as pd
import scipy.stats as stats
from numpy.lib.stride_tricks import sliding_window_view

OPERATORS_BINARY = ("ADD", "SUB", "MUL", "DIV_SAFE")
OPERATORS_UNARY = ("SIGN", "ABS", "LOG_ABS", "SIGMOID", "RELU", "EXP_SAFE", "LOG_SAFE", "NEG")
OPERATORS_ROLLING = ("DELTA_5", "EMA_10", "TS_MAX_10", "TS_MIN_10", "TS_RANK_20")
OPERATORS_CROSS_SECTIONAL = ("CS_RANK", "CS_DEMEAN")


def cs_rank(matrix: np.ndarray) -> np.ndarray:
    """Cross-sectional rank across asset axis (axis=0) in O(B log B * T) with zero 3D intermediate allocation."""
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
    """Cross-sectional de-meaning across asset axis (axis=0) using finite masking."""
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

STANDARD_FEATURES = [
    "Close_Adj", "Open", "High", "Low", "Volume", "Spread", "Vol_ZScore", "Vol_Price_Flow",
    "True_Range", "TR_Expansion", "EMA_Slope", "Dist_EMA20", "Dist_VWAP", "Downside_Vol_Ratio",
    "RSI_14", "Parkinson_Vol", "Garman_Klass_Vol", "SP500_Ret", "ADR_Ret", "Global_Confluence"
]


@dataclass
class ASTNode:
    op: str
    val: str = ""
    active_regimes: set[int] = field(default_factory=lambda: set(range(10)))
    children: list['ASTNode'] = field(default_factory=list)

    def is_silenced(self, regime: int) -> bool:
        return regime == -1 or regime not in self.active_regimes

    def clone(self) -> 'ASTNode':
        return ASTNode(self.op, self.val, set(self.active_regimes), [c.clone() for c in self.children])

    def depth(self) -> int:
        return 1 + max((c.depth() for c in self.children), default=0)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def count_nodes(self) -> int:
        return self.size()

    def get_all_subnodes(self) -> list['ASTNode']:
        return [self] + [sub for c in self.children for sub in c.get_all_subnodes()]

    def canonical_hash(self) -> str:
        """Returns deterministic 16-character hex hash invariant to commutative child ordering."""
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
        if self.op in ("FEATURE", "CONST"):
            return self.val
        if self.op == "MEGA_SWITCH":
            c = self.children[0].to_formula_str() if len(self.children) > 0 else "0"
            b = self.children[1].to_formula_str() if len(self.children) > 1 else "0"
            m = self.children[2].to_formula_str() if len(self.children) > 2 else "0"
            return f"MEGA_SWITCH({c}, {b}, {m})"
        if self.op in (OPERATORS_ROLLING + OPERATORS_UNARY + OPERATORS_CROSS_SECTIONAL):
            return f"{self.op}({self.children[0].to_formula_str() if self.children else 'X'})"
        sym = {"ADD": "+", "SUB": "-", "MUL": "*", "DIV_SAFE": "/"}.get(self.op, self.op)
        l = self.children[0].to_formula_str() if self.children else "0"
        r = self.children[1].to_formula_str() if len(self.children) > 1 else "0"
        return f"({l} {sym} {r})"


def _fast_ts_rank_20(arr: np.ndarray) -> np.ndarray:
    """Vectorized sliding-window percentile rank over 20 periods with dynamic valid counting."""
    n = len(arr)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    res = np.full(n, 0.5, dtype=np.float64)
    w = 20
    for i in range(min(n, w)):
        prefix = arr[:i + 1]
        target = arr[i]
        valid_prefix = np.isfinite(prefix)
        k = np.sum(valid_prefix)
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
    @staticmethod
    def eval_node(node: ASTNode, df: pd.DataFrame, current_regime: int = 0) -> np.ndarray:
        n = len(df)
        if node.is_silenced(current_regime):
            return np.zeros(n, dtype=np.float64)

        if node.op == "CONST":
            try:
                return np.full(n, float(node.val), dtype=np.float64)
            except (ValueError, TypeError):
                return np.zeros(n, dtype=np.float64)

        if node.op == "FEATURE":
            cache = df.attrs.get("_col_arrays")
            if cache is None:
                cache = {c: np.nan_to_num(df[c].to_numpy(dtype=np.float64), nan=0.0, posinf=1.0, neginf=-1.0) for c in df.columns}
                df.attrs["_col_arrays"] = cache
            if node.val in cache:
                return cache[node.val]
            if node.val == "Spread":
                h = cache.get("High", df["High"].to_numpy(dtype=np.float64) if "High" in df.columns else np.zeros(n, dtype=np.float64))
                l = cache.get("Low", df["Low"].to_numpy(dtype=np.float64) if "Low" in df.columns else np.zeros(n, dtype=np.float64))
                c = cache.get("Close_Adj", cache.get("Close_Raw", cache.get("Close", df.get("Close_Adj", df.get("Close", pd.Series(np.ones(n)))).to_numpy(dtype=np.float64))))
                return np.nan_to_num((h - l) / (c + 1e-8), nan=0.0)
            try:
                return np.full(n, float(node.val), dtype=np.float64)
            except ValueError:
                return np.zeros(n, dtype=np.float64)

        if node.op == "MEGA_SWITCH":
            cond = ASTEvaluator.eval_node(node.children[0], df, current_regime) if len(node.children) > 0 else np.zeros(n)
            base = ASTEvaluator.eval_node(node.children[1], df, current_regime) if len(node.children) > 1 else np.zeros(n)
            mega = ASTEvaluator.eval_node(node.children[2], df, current_regime) if len(node.children) > 2 else np.zeros(n)
            return np.nan_to_num(np.where(cond > 0.0, mega, base), nan=0.0)

        if node.op in OPERATORS_BINARY:
            l = ASTEvaluator.eval_node(node.children[0], df, current_regime) if node.children else np.zeros(n)
            r = ASTEvaluator.eval_node(node.children[1], df, current_regime) if len(node.children) > 1 else np.zeros(n)
            if node.op == "ADD": res = l + r
            elif node.op == "SUB": res = l - r
            elif node.op == "MUL": res = l * r
            elif node.op == "DIV_SAFE":
                r_abs = np.abs(r)
                denom_sign = np.where(r >= 0.0, 1.0, -1.0)
                safe_denom = denom_sign * np.maximum(r_abs, 1e-6)
                res = np.where(r_abs > 1e-6, l / safe_denom, 0.0)
                res = np.clip(res, -1e4, 1e4)
            else: res = np.zeros(n)
            return np.nan_to_num(res, nan=0.0, posinf=1e4, neginf=-1e4)

        child = ASTEvaluator.eval_node(node.children[0], df, current_regime) if node.children else np.zeros(n)
        if node.op in OPERATORS_UNARY:
            if node.op == "SIGN": res = np.sign(child)
            elif node.op == "ABS": res = np.abs(child)
            elif node.op == "LOG_ABS": res = np.log(np.abs(child) + 1.0)
            elif node.op == "SIGMOID": res = 1.0 / (1.0 + np.exp(-np.clip(child, -10.0, 10.0)))
            elif node.op == "RELU": res = np.maximum(0.0, child)
            elif node.op == "EXP_SAFE": res = np.exp(np.clip(child, -20.0, 20.0))
            elif node.op == "LOG_SAFE": res = np.log(np.maximum(1e-8, np.abs(child)))
            elif node.op == "NEG": res = -child
            else: res = np.zeros(n)
            res = np.clip(res, -1e4, 1e4)
            return np.nan_to_num(res, nan=0.0, posinf=1e4, neginf=-1e4)

        if node.op in OPERATORS_CROSS_SECTIONAL:
            if node.op == "CS_RANK": return cs_rank(child)
            elif node.op == "CS_DEMEAN": return cs_demean(child)
            return np.nan_to_num(child, nan=0.0)

        if node.op in OPERATORS_ROLLING:
            if node.op == "TS_RANK_20":
                return _fast_ts_rank_20(child)
            s = pd.Series(child)
            if node.op == "DELTA_5": res = s.diff(5).fillna(0.0).to_numpy()
            elif node.op == "EMA_10": res = s.ewm(span=10, adjust=False).mean().to_numpy()
            elif node.op == "TS_MAX_10": res = s.rolling(10, min_periods=1).max().to_numpy()
            elif node.op == "TS_MIN_10": res = s.rolling(10, min_periods=1).min().to_numpy()
            else: res = np.zeros(n)
            return np.nan_to_num(res, nan=0.0)
        return np.zeros(n, dtype=np.float64)


class TreeGenerator:
    @staticmethod
    def generate(max_depth: int = 4, features: list[str] = STANDARD_FEATURES, active_regimes: set[int] = None, method: str = "grow") -> ASTNode:
        regimes = active_regimes if active_regimes is not None else set(range(10))
        if max_depth <= 1 or (method == "grow" and random.random() < 0.35):
            return ASTNode("FEATURE", random.choice(features), regimes)
        op = random.choice(OPERATORS_BINARY + OPERATORS_UNARY + OPERATORS_ROLLING)
        children = [TreeGenerator.generate(max_depth - 1, features, regimes, method) for _ in range(2 if op in OPERATORS_BINARY else 1)]
        return ASTNode(op, active_regimes=regimes, children=children)


class GeneticMutator:
    @staticmethod
    def mutate_subtree(tree: ASTNode, max_depth: int = 3, features: list[str] = STANDARD_FEATURES) -> ASTNode:
        cloned = tree.clone()
        target = random.choice(cloned.get_all_subnodes())
        sub = TreeGenerator.generate(max_depth, features, target.active_regimes)
        target.op, target.val, target.children = sub.op, sub.val, sub.children
        return cloned

    @staticmethod
    def mutate_methylation(tree: ASTNode, num_regimes: int = 10) -> ASTNode:
        cloned = tree.clone()
        target = random.choice(cloned.get_all_subnodes())
        r = random.randint(0, num_regimes - 1)
        target.active_regimes.remove(r) if (r in target.active_regimes and len(target.active_regimes) > 1) else target.active_regimes.add(r)
        return cloned


class GeneticCrossover:
    @staticmethod
    def crossover(parent1: ASTNode, parent2: ASTNode) -> tuple[ASTNode, ASTNode]:
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
            elif node.op in OPERATORS_UNARY or node.op in OPERATORS_ROLLING or node.op in OPERATORS_CROSS_SECTIONAL:
                if len(node.children) == 0:
                    node.children.append(ASTNode("CONST", val="1.0", active_regimes=set(node.active_regimes)))
                elif len(node.children) > 1:
                    node.children = node.children[:1]
            elif node.op in ("FEATURE", "CONST"):
                node.children = []
        return c1, c2


class MegaEvolutionNode(ASTNode):
    """Dual-state AST node that dynamically transitions between base and mega forms."""
    def __init__(self, condition: ASTNode, base: ASTNode, mega: ASTNode, active_regimes: set[int] = None):
        regimes = active_regimes if active_regimes is not None else set(range(10))
        super().__init__(op="MEGA_SWITCH", val="", active_regimes=regimes, children=[condition, base, mega])

    def clone(self) -> 'MegaEvolutionNode':
        return MegaEvolutionNode(
            condition=self.children[0].clone(),
            base=self.children[1].clone(),
            mega=self.children[2].clone(),
            active_regimes=set(self.active_regimes)
        )

