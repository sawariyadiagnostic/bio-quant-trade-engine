"""Hierarchical Risk Parity (HRP) Capital Allocator (Wave 4: Task 4.2)."""
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
try:
    from sklearn.covariance import ledoit_wolf
except (ImportError, MemoryError):
    ledoit_wolf = None


class HRPAllocator:
    """
    Implements Marcos López de Prado's Hierarchical Risk Parity (HRP).
    Allocates risk budgets across strategy/asset clusters using graph hierarchy
    and recursive bisection, eliminating matrix inversion blowups.
    Upgraded with true dendrogram tree traversal, volatility floors, and box bounds.
    """
    @staticmethod
    def apply_box_bounds(
        weights: np.ndarray,
        min_weight: float = 0.02,
        max_weight: float = 0.30,
        max_iter: int = 100
    ) -> np.ndarray:
        """Enforces box constraints [min_weight, max_weight] on weights summing to 1.0."""
        w = np.copy(weights)
        N = len(w)
        if N <= 0:
            return w
        # Feasibility check: if bounds are infeasible for N assets, relax bounds
        if N * min_weight > 1.0 or N * max_weight < 1.0:
            eff_min = min(min_weight, 1.0 / N)
            eff_max = max(max_weight, 1.0 / N)
        else:
            eff_min = min_weight
            eff_max = max_weight

        for _ in range(max_iter):
            clipped = np.clip(w, eff_min, eff_max)
            excess = 1.0 - np.sum(clipped)
            if abs(excess) < 1e-7:
                return clipped
            free_mask = (clipped > eff_min + 1e-8) & (clipped < eff_max - 1e-8)
            if not np.any(free_mask):
                w = clipped + excess / N
                return np.clip(w, eff_min, eff_max) / np.sum(np.clip(w, eff_min, eff_max))
            w = np.copy(clipped)
            free_sum = np.sum(w[free_mask])
            if free_sum > 0:
                w[free_mask] += excess * (w[free_mask] / free_sum)
            else:
                w[free_mask] += excess / np.sum(free_mask)

        w = np.clip(w, eff_min, eff_max)
        return w / np.sum(w)

    @staticmethod
    def compute_weights(
        returns_matrix: np.ndarray,
        linkage_method: str = "ward",
        use_shrinkage: bool = True,
        min_weight: float = 0.02,
        max_weight: float = 0.30
    ) -> np.ndarray:
        """
        Computes risk-parity weights for N return series using true dendrogram traversal.

        Args:
            returns_matrix: 2D array of shape [T, N].
            linkage_method: Hierarchical clustering linkage ('ward', 'average', 'single').
            use_shrinkage: Whether to apply Ledoit-Wolf analytical covariance shrinkage.
            min_weight: Lower box bound (default 0.02).
            max_weight: Upper box bound (default 0.30).

        Returns:
            1D array of portfolio weights summing to 1.0 with box bounds.
        """
        if returns_matrix.ndim == 1:
            return np.array([1.0], dtype=np.float64)

        T, N = returns_matrix.shape
        if N <= 1:
            return np.ones(N, dtype=np.float64)

        # 1. Covariance Matrix with Optional Ledoit-Wolf Shrinkage
        if use_shrinkage and T >= 4 and ledoit_wolf is not None:
            try:
                cov, _ = ledoit_wolf(returns_matrix)
            except Exception:
                cov = np.cov(returns_matrix, rowvar=False)
        else:
            cov = np.cov(returns_matrix, rowvar=False)

        # Volatility floor and reachability check for zero-variance / non-finite assets
        raw_std = np.sqrt(np.maximum(0.0, np.diag(cov)))
        if np.any(raw_std < 1e-4) or not np.all(np.isfinite(cov)):
            eq_w = np.full(N, 1.0 / N, dtype=np.float64)
            return HRPAllocator.apply_box_bounds(eq_w, min_weight, max_weight)

        std = np.maximum(raw_std, 1e-4)
        inv_std = 1.0 / std
        corr = cov * np.outer(inv_std, inv_std)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)

        # 2. Distance Matrix
        dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
        np.fill_diagonal(dist, 0.0)

        # 3. Hierarchical Linkage and True Dendrogram Tree Traversal
        try:
            condensed = squareform(dist, checks=False)
            link = linkage(condensed, method=linkage_method)
            weights = HRPAllocator._tree_recursive_bisection(cov, link, N)
        except Exception:
            try:
                condensed = squareform(dist, checks=False)
                link = linkage(condensed, method="single")
                weights = HRPAllocator._tree_recursive_bisection(cov, link, N)
            except Exception:
                inv_var = 1.0 / np.maximum(1e-8, np.diag(cov))
                weights = (inv_var / np.sum(inv_var)).astype(np.float64)

        # 4. Box Bounds Enforcement
        bounded_weights = HRPAllocator.apply_box_bounds(weights, min_weight, max_weight)
        return bounded_weights / np.sum(bounded_weights)

    @staticmethod
    def _tree_recursive_bisection(cov: np.ndarray, link: np.ndarray, N: int) -> np.ndarray:
        """Top-down recursive risk budget splitting traversing the true dendrogram tree."""
        weights = np.ones(N, dtype=np.float64)
        node_leaves: dict[int, list[int]] = {i: [i] for i in range(N)}

        for k in range(N - 1):
            parent = N + k
            left_child = int(link[k, 0])
            right_child = int(link[k, 1])
            node_leaves[parent] = node_leaves[left_child] + node_leaves[right_child]

        queue = [2 * N - 2]
        while queue:
            node = queue.pop(0)
            if node < N:
                continue
            k = node - N
            left = int(link[k, 0])
            right = int(link[k, 1])
            c0 = node_leaves[left]
            c1 = node_leaves[right]

            var0 = HRPAllocator._get_cluster_var(cov, c0)
            var1 = HRPAllocator._get_cluster_var(cov, c1)
            alpha = 1.0 - var0 / max(1e-8, (var0 + var1))

            for idx in c0:
                weights[idx] *= alpha
            for idx in c1:
                weights[idx] *= (1.0 - alpha)

            queue.append(left)
            queue.append(right)

        return weights

    @staticmethod
    def _get_quasi_diag(link: np.ndarray) -> List[int]:
        """Hierarchical tree seriation sorting similar items adjacent along diagonal."""
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]

        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df1 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df1]).sort_index()
            sort_ix.index = range(sort_ix.shape[0])

        return [int(x) for x in sort_ix.tolist()]

    @staticmethod
    def _get_cluster_var(cov: np.ndarray, c_items: List[int]) -> float:
        """Computes variance of a sub-cluster using inverse-variance item weighting."""
        sub_cov = cov[np.ix_(c_items, c_items)]
        sub_var = np.maximum(np.diag(sub_cov), 1e-8)  # (1e-4)^2 volatility floor
        inv_diag = 1.0 / sub_var
        w = inv_diag / np.sum(inv_diag)
        return float(np.dot(np.dot(w, sub_cov), w))

    @staticmethod
    def _recursive_bisection(cov: np.ndarray, sort_ix: List[int]) -> List[float]:
        """Legacy midpoint recursive bisection maintained for backward compatibility."""
        w = pd.Series(1.0, index=range(len(sort_ix)))
        c_items = [list(range(len(sort_ix)))]

        while len(c_items) > 0:
            c_items = [
                i[j:k]
                for i in c_items
                for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
                if len(i) > 1
            ]
            for i in range(0, len(c_items), 2):
                c0 = c_items[i]
                c1 = c_items[i + 1]
                var0 = HRPAllocator._get_cluster_var(cov, c0)
                var1 = HRPAllocator._get_cluster_var(cov, c1)
                alpha = 1.0 - var0 / max(1e-8, (var0 + var1))
                w[c0] *= alpha
                w[c1] *= (1.0 - alpha)

        return w.tolist()
