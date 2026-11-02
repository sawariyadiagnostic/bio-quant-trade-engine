"""
Anchored UMAP-HDBSCAN Manifold Classifier & Out-of-Sample Predictor.
Bridges bioinformatic manifold reduction (UMAP 3D) with density clustering (HDBSCAN).
Prevents cluster label-flipping by anchoring reference centroids and core radii in the manifold space.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from loguru import logger
from ..domain.models import RegimeState

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    umap = None
    UMAP_AVAILABLE = False

class AnchoredManifoldClassifier:
    def __init__(
        self, 
        min_cluster_size: int = 6, 
        min_samples: int = 3,
        use_umap: bool = True,
        umap_components: int = 3,
        umap_neighbors: int = 15,
        umap_min_dist: float = 0.1
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.use_umap = use_umap and UMAP_AVAILABLE
        self.umap_components = umap_components
        self.umap_neighbors = umap_neighbors
        self.umap_min_dist = umap_min_dist

        self.reducer = None
        self.clusterer = None
        self.centroids = {}       # cluster_id -> mean vector in manifold space
        self.radii = {}           # cluster_id -> 95th percentile distance to member points
        self.cluster_stats = {}   # cluster_id -> forward expectancy metrics
        self.is_fitted = False

    def fit(self, X: np.ndarray, forward_returns: np.ndarray = None) -> np.ndarray:
        """
        Fits UMAP dimensionality reduction (if enabled) + HDBSCAN on calibration dataset.
        Extracts centroids and radii for all dense clusters (k >= 0) in the manifold space.
        """
        if len(X) < self.min_cluster_size:
            raise ValueError(f"Dataset too small ({len(X)}) for min_cluster_size={self.min_cluster_size}")

        # 1. Non-linear Manifold Projection via UMAP (Bioinformatics paradigm)
        if self.use_umap:
            # We omit random_state to enable multi-threaded execution (n_jobs=-1) for maximum speed
            self.reducer = umap.UMAP(
                n_neighbors=self.umap_neighbors,
                min_dist=self.umap_min_dist,
                n_components=self.umap_components,
                metric="euclidean"
            )
            X_manifold = self.reducer.fit_transform(X)
        else:
            X_manifold = X

        # 2. Density-based clustering on manifold
        self.clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            copy=True
        )
        labels = self.clusterer.fit_predict(X_manifold)

        self.centroids.clear()
        self.radii.clear()
        self.cluster_stats.clear()

        unique_clusters = set(labels)
        for cid in unique_clusters:
            if cid == -1:
                continue
            mask = (labels == cid)
            pts = X_manifold[mask]
            centroid = np.mean(pts, axis=0)
            distances = np.linalg.norm(pts - centroid, axis=1)
            radius = float(np.percentile(distances, 95)) # 95th percentile core boundary

            self.centroids[cid] = centroid
            self.radii[cid] = radius

            # Calculate statistical expectancy if forward returns provided
            if forward_returns is not None and len(forward_returns) == len(labels):
                fwd = forward_returns[mask]
                valid_fwd = fwd[~np.isnan(fwd)]
                if len(valid_fwd) > 0:
                    self.cluster_stats[cid] = {
                        "mean_fwd_5d": float(np.mean(valid_fwd)),
                        "win_rate": float(np.mean(valid_fwd > 0.0)),
                        "count": int(len(valid_fwd))
                    }

        self.is_fitted = True
        noise_ratio = float(np.mean(labels == -1))
        logger.info(
            f"Fitted Manifold (UMAP={self.use_umap}): {len(self.centroids)} dense clusters discovered. "
            f"Noise ratio: {noise_ratio:.1%}"
        )
        return labels

    def predict_single(self, x: np.ndarray) -> RegimeState:
        """
        Deterministic Out-of-Sample prediction.
        Projects point onto the anchored manifold centroids using core radii boundary checks.
        Zero label flipping.
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted on calibration data first.")

        x_vec = np.asarray(x).reshape(1, -1)
        
        # Project into manifold space if UMAP active
        if self.use_umap and self.reducer is not None:
            x_manifold = self.reducer.transform(x_vec)[0]
        else:
            x_manifold = x_vec[0]

        best_cluster = -1
        min_normalized_dist = float("inf")

        for cid, centroid in self.centroids.items():
            dist = np.linalg.norm(x_manifold - centroid)
            radius = self.radii[cid]
            norm_dist = dist / (radius + 1e-6)

            # Inside the 95th percentile core boundary
            if norm_dist <= 1.0 and norm_dist < min_normalized_dist:
                min_normalized_dist = norm_dist
                best_cluster = cid

        is_noise = (best_cluster == -1)
        density_score = float(max(0.0, 1.0 - min_normalized_dist)) if not is_noise else 0.0
        exp = self.cluster_stats.get(best_cluster, {}).get("mean_fwd_5d", 0.0)

        return RegimeState(
            cluster_id=int(best_cluster),
            is_noise=bool(is_noise),
            density_score=float(density_score),
            forward_expectancy=float(exp)
        )

    def predict_batch(self, X: np.ndarray) -> list[RegimeState]:
        """
        Vectorized Out-of-Sample Batch Prediction.
        Projects all rows into UMAP manifold space simultaneously in C-speed,
        then evaluates core centroid boundaries.
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted on calibration data first.")
        if len(X) == 0:
            return []

        X_mat = np.asarray(X)
        if self.use_umap and self.reducer is not None:
            X_manifold = self.reducer.transform(X_mat)
        else:
            X_manifold = X_mat

        results = []
        for x_m in X_manifold:
            best_cluster = -1
            min_norm_dist = float("inf")

            for cid, centroid in self.centroids.items():
                dist = np.linalg.norm(x_m - centroid)
                radius = self.radii[cid]
                norm_dist = dist / (radius + 1e-6)

                if norm_dist <= 1.0 and norm_dist < min_norm_dist:
                    min_norm_dist = norm_dist
                    best_cluster = cid

            is_noise = (best_cluster == -1)
            density_score = float(max(0.0, 1.0 - min_norm_dist)) if not is_noise else 0.0
            exp = self.cluster_stats.get(best_cluster, {}).get("mean_fwd_5d", 0.0)

            results.append(RegimeState(
                cluster_id=int(best_cluster),
                is_noise=bool(is_noise),
                density_score=float(density_score),
                forward_expectancy=float(exp)
            ))
        return results
