"""Hierarchical Risk Parity (HRP) Capital Allocation Engine."""
from __future__ import annotations

from typing import List, Optional
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

try:
    from sklearn.covariance import ledoit_wolf
except ImportError:
    ledoit_wolf = None


class HierarchicalRiskParity:
    """Marcos López de Prado's Hierarchical Risk Parity (HRP) Portfolio Allocator.

    Allocates risk budgets across asset clusters using tree hierarchy and recursive
    bisection, eliminating matrix inversion blowups and extreme corner weights.
    """

    @classmethod
    def compute_weights(
        cls,
        returns_matrix: np.ndarray,
        linkage_method: str = "ward",
        use_shrinkage: bool = True,
        use_denoising: bool = True,
    ) -> np.ndarray:
        """Compute robust risk-parity weights for N asset return series."""
        if returns_matrix.ndim == 1 or returns_matrix.shape[1] <= 1:
            return np.ones(returns_matrix.shape[1] if returns_matrix.ndim > 1 else 1, dtype=np.float64)

        T, N = returns_matrix.shape

        # Raw volatility check on input series before any shrinkage inflation (VULN-01 fix)
        raw_vol = np.std(returns_matrix, axis=0)
        if np.any(raw_vol < 1e-8) or not np.all(np.isfinite(returns_matrix)):
            return np.full(N, 1.0 / N, dtype=np.float64)

        cov = cls._estimate_covariance(returns_matrix, use_shrinkage, T)

        # Raw diagonal check on covariance
        raw_diag = np.diag(cov)
        if np.any(raw_diag < 1e-8) or not np.all(np.isfinite(cov)):
            return np.full(N, 1.0 / N, dtype=np.float64)

        # Optional Marcenko-Pastur RMT covariance denoising
        if use_denoising and T > N:
            cov = cls.denoise_covariance_rmt(cov, T)

        std = np.sqrt(np.maximum(1e-12, np.diag(cov)))
        inv_std = 1.0 / std
        corr = cov * np.outer(inv_std, inv_std)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)

        dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
        np.fill_diagonal(dist, 0.0)

        try:
            condensed = squareform(dist, checks=False)
            link = linkage(condensed, method=linkage_method)
            sort_ix = cls._get_quasi_diag(link)
        except Exception:
            return cls._fallback_inverse_variance(cov)

        sorted_cov = cov[np.ix_(sort_ix, sort_ix)]
        weights_sorted = cls._recursive_bisection(sorted_cov, sort_ix)

        final_weights = np.zeros(N, dtype=np.float64)
        for original_idx, w in zip(sort_ix, weights_sorted):
            final_weights[original_idx] = w

        total_w = np.sum(final_weights)
        return (final_weights / total_w) if total_w > 0 else np.full(N, 1.0 / N, dtype=np.float64)

    @classmethod
    def denoise_covariance_rmt(cls, cov: np.ndarray, T: int, detone: bool = False) -> np.ndarray:
        """Filter sampling noise from covariance matrix via Marcenko-Pastur RMT spectral cutoff."""
        N = cov.shape[0]
        if T <= N:
            return cov

        std = np.sqrt(np.diag(cov))
        inv_std = 1.0 / np.maximum(1e-12, std)
        corr = np.clip(cov * np.outer(inv_std, inv_std), -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)

        e_val, e_vec = np.linalg.eigh(corr)
        idx = np.argsort(e_val)[::-1]
        e_val, e_vec = e_val[idx], e_vec[:, idx]

        q = float(T) / float(N)
        lambda_plus = (1.0 + np.sqrt(1.0 / q)) ** 2

        noise_idx = e_val <= lambda_plus
        if np.any(noise_idx):
            e_val_denoised = e_val.copy()
            e_val_denoised[noise_idx] = np.mean(e_val[noise_idx])
            if detone and len(e_val_denoised) > 2:
                e_val_denoised[0] = np.mean(e_val[noise_idx])

            corr_denoised = np.dot(e_vec, np.dot(np.diag(e_val_denoised), e_vec.T))
            diag_inv = 1.0 / np.sqrt(np.maximum(1e-12, np.diag(corr_denoised)))
            corr_denoised = corr_denoised * np.outer(diag_inv, diag_inv)
            np.fill_diagonal(corr_denoised, 1.0)
            return corr_denoised * np.outer(std, std)
        return cov

    @classmethod
    def _estimate_covariance(cls, mat: np.ndarray, use_shrinkage: bool, T: int) -> np.ndarray:
        """Estimate covariance with optional Ledoit-Wolf analytical shrinkage."""
        if use_shrinkage and T >= 5 and ledoit_wolf is not None:
            try:
                cov, _ = ledoit_wolf(mat)
                return cov
            except Exception:
                pass
        return np.cov(mat, rowvar=False)

    @staticmethod
    def _fallback_inverse_variance(cov: np.ndarray) -> np.ndarray:
        """Fallback to inverse-variance weighting when hierarchical clustering degenerates."""
        inv_var = 1.0 / np.maximum(1e-8, np.diag(cov))
        return (inv_var / np.sum(inv_var)).astype(np.float64)

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

    @classmethod
    def _recursive_bisection(cls, cov: np.ndarray, sort_ix: List[int]) -> List[float]:
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
                var0 = cls._get_cluster_var(cov, c0)
                var1 = cls._get_cluster_var(cov, c1)
                total_var = var0 + var1
                alpha = 0.5 if total_var <= 1e-12 else 1.0 - (var0 / total_var)
                w[c0] *= alpha
                w[c1] *= (1.0 - alpha)

        return w.tolist()
