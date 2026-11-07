"""3D UMAP Strategy Manifold & Occam's Razor Alpha Pruner (Wave 3: Task 3.2)."""
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


@dataclass
class AlphaManifoldPruner:
    """
    Projects high-dimensional strategy behaviors into a 3D manifold,
    clusters redundant clones, and applies Occam's razor to preserve only
    the simplest, most orthogonal alpha representations.
    """
    min_cluster_size: int = 4

    def cluster_and_prune(
        self,
        phenotype_matrix: np.ndarray,
        ast_lengths: List[int],
        sortinos: List[float],
        use_umap: bool = True
    ) -> Dict[str, Any]:
        """
        Deduplicates redundant strategy clones while preserving orthogonal alpha streams.

        Args:
            phenotype_matrix: 2D array of binary trade decisions (shape: [N, D]).
            ast_lengths: String/depth complexity lengths of candidate formulas.
            sortinos: Annualized Sortino ratios of candidate formulas.
            use_umap: If True and UMAP is available, uses UMAP; otherwise fast SVD.

        Returns:
            Dictionary with surviving indices, orthogonal indices, labels, and 3D coordinates.
        """
        N, D = phenotype_matrix.shape
        if N == 0:
            return {"surviving_indices": [], "orthogonal_indices": [], "compression_ratio": 0.0, "embedding_3d": np.empty((0, 3))}

        # 1. 3D Manifold Dimensionality Reduction (UMAP or Fallback SVD)
        if UMAP_AVAILABLE and N >= 10 and use_umap:
            n_neighbors = max(2, min(15, N - 1))
            reducer = umap.UMAP(
                n_components=3,
                n_neighbors=n_neighbors,
                min_dist=0.1,
                random_state=42
            )
            embedding_3d = reducer.fit_transform(phenotype_matrix)
        else:
            # High-performance SVD fallback for small batches or minimal environments
            centered = phenotype_matrix - np.mean(phenotype_matrix, axis=0)
            u, s, _ = np.linalg.svd(centered, full_matrices=False)
            comps = min(3, u.shape[1])
            embedding_3d = np.zeros((N, 3), dtype=np.float32)
            embedding_3d[:, :comps] = (u[:, :comps] * s[:comps]).astype(np.float32)

        # 2. Density Clustering (HDBSCAN or Distance Threshold Fallback)
        min_c = max(2, min(self.min_cluster_size, N // 2))
        if HDBSCAN_AVAILABLE and N >= 6:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_c, min_samples=1)
            labels = clusterer.fit_predict(embedding_3d)
        else:
            # Deterministic distance-based clustering fallback
            labels = np.full(N, -1, dtype=int)
            current_c = 0
            visited = set()
            for i in range(N):
                if i in visited:
                    continue
                dists = np.mean(np.abs(phenotype_matrix - phenotype_matrix[i]), axis=1)
                matches = np.where(dists < 0.05)[0]
                if len(matches) >= min_c:
                    labels[matches] = current_c
                    visited.update(matches)
                    current_c += 1

        # 3. Occam's Razor Pruning & Orthogonal Alpha Isolation
        surviving_indices: List[int] = []
        orthogonal_indices: List[int] = []

        unique_clusters = set(labels)
        for cluster_id in unique_clusters:
            if cluster_id == -1:
                # HDBSCAN Noise = Pure Idiosyncratic / Orthogonal Alpha!
                noise_points = np.where(labels == -1)[0].tolist()
                orthogonal_indices.extend(noise_points)
                surviving_indices.extend(noise_points)
            else:
                # Dense Cluster = Redundant Clones. Apply Occam's Razor.
                members = np.where(labels == cluster_id)[0].tolist()
                # Sort primarily by lowest AST length (Occam's razor) and secondarily by highest Sortino
                members.sort(key=lambda idx: (ast_lengths[idx], -sortinos[idx]))
                # Keep single best representative
                surviving_indices.append(members[0])

        surviving_indices = sorted(list(set(surviving_indices)))
        orthogonal_indices = sorted(list(set(orthogonal_indices)))
        compression_ratio = round(1.0 - (len(surviving_indices) / max(1, N)), 4)

        return {
            "surviving_indices": surviving_indices,
            "orthogonal_indices": orthogonal_indices,
            "cluster_labels": labels,
            "embedding_3d": embedding_3d.astype(np.float32),
            "compression_ratio": compression_ratio,
            "clusters_count": len([c for c in unique_clusters if c != -1])
        }
