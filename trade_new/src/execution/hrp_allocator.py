"""Hierarchical Risk Parity (HRP) Capital Allocator (Wave 4: Task 4.2)."""
from typing import List
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
try:
    from sklearn.covariance import ledoit_wolf
except ImportError:
    ledoit_wolf = None


class HRPAllocator:
    """
    Implements Marcos López de Prado's Hierarchical Risk Parity (HRP).
    Allocates risk budgets across strategy/asset clusters using graph hierarchy
    and recursive bisection, eliminating matrix inversion blowups.
    Upgraded with Ward/Average linkage and analytical Ledoit-Wolf shrinkage.
    """
    @staticmethod
    def compute_weights(
        returns_matrix: np.ndarray,
        linkage_method: str = "ward",
        use_shrinkage: bool = True
    ) -> np.ndarray:
        """
        Computes risk-parity weights for N return series.

        Args:
            returns_matrix: 2D array of shape [T, N].
            linkage_method: Hierarchical clustering linkage ('ward', 'average', 'single').
            use_shrinkage: Whether to apply Ledoit-Wolf analytical covariance shrinkage.

        Returns:
            1D array of portfolio weights summing to 1.0.
        """
        if returns_matrix.ndim == 1:
            return np.array([1.0], dtype=np.float64)

        T, N = returns_matrix.shape
        if N <= 1:
            return np.ones(N, dtype=np.float64)

        # 1. Covariance & Correlation Matrix with Optional Ledoit-Wolf Shrinkage
        if use_shrinkage and T >= 4 and ledoit_wolf is not None:
            try:
                cov, _ = ledoit_wolf(returns_matrix)
            except Exception:
                cov = np.cov(returns_matrix, rowvar=False)
        else:
            cov = np.cov(returns_matrix, rowvar=False)

        std = np.sqrt(np.maximum(1e-12, np.diag(cov)))
        if np.any(std < 1e-8) or not np.all(np.isfinite(cov)):
            return np.full(N, 1.0 / N, dtype=np.float64)

        inv_std = 1.0 / std
        corr = cov * np.outer(inv_std, inv_std)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)

        # 2. Distance Matrix
        dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
        np.fill_diagonal(dist, 0.0)

        # Condensed distance matrix for hierarchical clustering
        try:
            condensed = squareform(dist, checks=False)
            link = linkage(condensed, method=linkage_method)
            sort_ix = HRPAllocator._get_quasi_diag(link)
        except Exception:
            # Fallback to single linkage or inverse variance weighting
            try:
                condensed = squareform(dist, checks=False)
                link = linkage(condensed, method="single")
                sort_ix = HRPAllocator._get_quasi_diag(link)
            except Exception:
                inv_var = 1.0 / np.maximum(1e-8, np.diag(cov))
                return (inv_var / np.sum(inv_var)).astype(np.float64)

        # 3. Recursive Bisection
        sorted_cov = cov[np.ix_(sort_ix, sort_ix)]
        weights_sorted = HRPAllocator._recursive_bisection(sorted_cov, sort_ix)

        # Re-order weights to match original input indices
        final_weights = np.zeros(N, dtype=np.float64)
        for original_idx, w in zip(sort_ix, weights_sorted):
            final_weights[original_idx] = w

        return final_weights / np.sum(final_weights)

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
        inv_diag = 1.0 / np.maximum(1e-8, np.diag(sub_cov))
        w = inv_diag / np.sum(inv_diag)
        return float(np.dot(np.dot(w, sub_cov), w))

    @staticmethod
    def _recursive_bisection(cov: np.ndarray, sort_ix: List[int]) -> List[float]:
        """Top-down recursive risk budget splitting along the dendrogram."""
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
