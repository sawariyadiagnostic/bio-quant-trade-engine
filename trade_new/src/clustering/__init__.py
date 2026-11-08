"""Clustering Bounded Context."""
from .return_clustering import (
    compute_correlation_matrix,
    compute_correlation_distance,
    hierarchical_linkage,
    assign_clusters,
    prune_clusters,
    ReturnClusterer,
)

__all__ = [
    "compute_correlation_matrix",
    "compute_correlation_distance",
    "hierarchical_linkage",
    "assign_clusters",
    "prune_clusters",
    "ReturnClusterer",
]
