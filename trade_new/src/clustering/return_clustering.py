"""Pure Return-Space Correlation Clustering and Occam's Pruning.

Eliminates high-dimensional embedding bloat by clustering candidate formulas directly
in return space using empirical Pearson correlation, correlation distance metrics,
and Ward's hierarchical linkage.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
try:
    from sklearn.covariance import ledoit_wolf
except ImportError:
    ledoit_wolf = None


def compute_correlation_matrix(
    returns: np.ndarray,
    use_shrinkage: Optional[bool] = None
) -> np.ndarray:
    """Computes empirical Pearson correlation matrix C [N, N] from returns [N, T].
    Applies Ledoit-Wolf shrinkage when candidate count exceeds validation window (T < N).
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError(f"Expected 2D returns array [N, T], got shape {r.shape}")
    N, T = r.shape
    if N == 0:
        return np.empty((0, 0), dtype=np.float64)
    if N == 1:
        return np.ones((1, 1), dtype=np.float64)
    if T <= 1:
        return np.eye(N, dtype=np.float64)

    # Apply Ledoit-Wolf shrinkage when T < N or explicitly requested
    apply_shrink = use_shrinkage if use_shrinkage is not None else (T < N)
    if apply_shrink and T >= 4 and ledoit_wolf is not None:
        try:
            cov, _ = ledoit_wolf(r.T)
        except Exception:
            cov = np.cov(r, rowvar=True)
    else:
        cov = np.cov(r, rowvar=True)

    if np.isscalar(cov):
        cov = np.array([[cov]], dtype=np.float64)

    stds = np.sqrt(np.maximum(1e-12, np.diag(cov)))
    inv_stds = np.where(stds > 1e-4, 1.0 / stds, 0.0)
    corr = cov * np.outer(inv_stds, inv_stds)
    corr = np.nan_to_num(corr, nan=0.0)
    np.clip(corr, -1.0, 1.0, out=corr)
    np.fill_diagonal(corr, 1.0)
    return corr


def compute_correlation_distance(
    corr: np.ndarray,
    squared_ward: bool = False
) -> np.ndarray:
    """Transforms correlation matrix C into distance metric D = sqrt(0.5 * (1 - C)) or D^2 = 2(1 - C)."""
    if squared_ward:
        dist = np.clip(2.0 * (1.0 - corr), 0.0, 4.0)
    else:
        dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)
    return dist


def compute_ward_distance(corr: np.ndarray, squared: bool = True) -> np.ndarray:
    """Computes Ward-compliant distance metric D^2 = 2(1 - rho) or Euclidean D = sqrt(2(1 - rho))."""
    dist_sq = np.clip(2.0 * (1.0 - corr), 0.0, 4.0)
    np.fill_diagonal(dist_sq, 0.0)
    return dist_sq if squared else np.sqrt(dist_sq)


def threshold_to_distance(rho_threshold: float = 0.85) -> float:
    """Converts a correlation threshold rho to hierarchical distance threshold."""
    return float(np.sqrt(np.clip(0.5 * (1.0 - rho_threshold), 0.0, 1.0)))


def hierarchical_linkage(dist: np.ndarray, method: str = "ward") -> np.ndarray:
    """Computes hierarchical clustering linkage matrix from distance matrix D."""
    N = dist.shape[0]
    if N <= 1:
        return np.empty((0, 4), dtype=np.float64)
    condensed = squareform(dist, checks=False)
    try:
        return linkage(condensed, method=method)
    except Exception:
        try:
            return linkage(condensed, method="average")
        except Exception:
            return linkage(condensed, method="single")


def assign_clusters(
    link: np.ndarray,
    n_samples: int,
    rho_threshold: float = 0.85,
    dist_threshold: Optional[float] = None,
) -> np.ndarray:
    """Assigns cluster IDs (1-indexed) based on correlation/distance threshold."""
    if n_samples <= 0:
        return np.empty(0, dtype=np.int32)
    if n_samples == 1:
        return np.ones(1, dtype=np.int32)
    if dist_threshold is None:
        dist_threshold = threshold_to_distance(rho_threshold)
    labels = fcluster(link, t=dist_threshold, criterion="distance")
    return labels.astype(np.int32)


def prune_clusters(
    cluster_labels: np.ndarray,
    fitness_scores: np.ndarray,
    depths: np.ndarray,
    parsimony_penalty: float = 0.05,
) -> List[int]:
    """Selects the single best formula per cluster via Occam's composite fitness.
    Inert / dead formula groups (label <= 0) are pruned immediately.
    """
    if len(cluster_labels) == 0:
        return []
    fit = np.asarray(fitness_scores, dtype=np.float64)
    dep = np.asarray(depths, dtype=np.float64)
    composite = fit - (dep * parsimony_penalty)
    selected_indices: List[int] = []
    unique_labels = np.unique(cluster_labels)
    for label in unique_labels:
        if label <= 0:
            continue  # Zero-variance / dead formulas pruned immediately
        mask = np.where(cluster_labels == label)[0]
        best_in_cluster = mask[np.argmax(composite[mask])]
        selected_indices.append(int(best_in_cluster))
    selected_indices.sort()
    return selected_indices


@dataclass
class ReturnClusterer:
    """End-to-end Return-Space Correlation Clusterer and Occam's Pruner."""
    rho_threshold: float = 0.85
    parsimony_penalty: float = 0.05
    linkage_method: str = "ward"

    def fit_predict(self, returns: np.ndarray) -> np.ndarray:
        """Computes cluster labels for candidate formula return matrix [N, T].
        Zero-variance / dead formulas cluster together into inert group (-1).
        """
        N = returns.shape[0]
        if N <= 0:
            return np.empty(0, dtype=np.int32)
        if N == 1:
            st = float(np.std(returns[0]))
            return np.array([-1 if st < 1e-4 else 1], dtype=np.int32)

        stds = np.std(returns, axis=1)
        dead_mask = stds < 1e-4

        labels = np.full(N, -1, dtype=np.int32)
        act_idx = np.where(~dead_mask)[0]
        if len(act_idx) == 0:
            return labels

        if len(act_idx) == 1:
            labels[act_idx[0]] = 1
            return labels

        sub_rets = returns[act_idx]
        corr = compute_correlation_matrix(sub_rets)
        dist = compute_correlation_distance(corr)
        link = hierarchical_linkage(dist, method=self.linkage_method)
        act_labels = assign_clusters(link, len(act_idx), rho_threshold=self.rho_threshold)
        labels[act_idx] = act_labels
        return labels

    def cluster_and_prune(
        self,
        returns: np.ndarray,
        fitness_scores: np.ndarray,
        depths: np.ndarray,
    ) -> Tuple[np.ndarray, List[int]]:
        """Clusters candidate formulas and prunes clones, returning (labels, elites)."""
        labels = self.fit_predict(returns)
        elites = prune_clusters(
            labels,
            fitness_scores,
            depths,
            parsimony_penalty=self.parsimony_penalty,
        )
        return labels, elites
