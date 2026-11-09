"""Unit tests for Pure Return-Space Correlation Clustering and Occam's Pruning."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_correlation_matrix_computation():
    """Verify empirical Pearson correlation matrix and constant return guards."""
    from src.clustering.return_clustering import compute_correlation_matrix

    np.random.seed(42)
    T = 200
    base = np.random.normal(0.001, 0.02, T)
    r0 = base
    r1 = base + np.random.normal(0, 0.001, T)  # rho ~ 0.99
    r2 = -base  # rho = -1.0
    r3 = np.zeros(T)  # Zero variance (no trades)

    R = np.vstack([r0, r1, r2, r3])
    C = compute_correlation_matrix(R)

    assert C.shape == (4, 4)
    assert np.allclose(np.diag(C), 1.0)
    assert C[0, 1] > 0.95
    assert np.isclose(C[0, 2], -1.0, atol=1e-4)
    assert C[0, 3] == 0.0  # Zero-variance handled safely
    assert np.all(np.isfinite(C))
    print("[PASS] test_correlation_matrix_computation")


def test_correlation_distance_metric():
    """Verify correlation distance D = sqrt(0.5 * (1 - C)) bounds and properties."""
    from src.clustering.return_clustering import (
        compute_correlation_distance,
        threshold_to_distance,
    )

    C = np.array([
        [1.0, 1.0, 0.0, -1.0],
        [1.0, 1.0, 0.5, 0.0],
        [0.0, 0.5, 1.0, -0.5],
        [-1.0, 0.0, -0.5, 1.0],
    ])
    D = compute_correlation_distance(C)

    assert np.allclose(np.diag(D), 0.0)
    assert np.isclose(D[0, 1], 0.0)  # Identical series -> distance 0
    assert np.isclose(D[0, 2], np.sqrt(0.5))  # Uncorrelated -> distance ~0.7071
    assert np.isclose(D[0, 3], 1.0)  # Perfectly inverted -> distance 1.0
    assert np.all(D >= 0.0) and np.all(D <= 1.0)

    # Threshold conversion check: rho=0.85 -> distance ~ 0.27386
    expected_dist = np.sqrt(0.5 * (1.0 - 0.85))
    assert np.isclose(threshold_to_distance(0.85), expected_dist)
    print("[PASS] test_correlation_distance_metric")


def test_hierarchical_clustering_and_cluster_assignment():
    """Verify hierarchical clustering groups correlated formulas and separates diverse ones."""
    from src.clustering.return_clustering import (
        compute_correlation_matrix,
        compute_correlation_distance,
        hierarchical_linkage,
        assign_clusters,
    )

    np.random.seed(42)
    T = 300
    # Cluster 1: Momentum clan (Formulas 0, 1, 2)
    m1 = np.random.normal(0.001, 0.02, T)
    f0 = m1
    f1 = m1 + np.random.normal(0, 0.002, T)
    f2 = m1 + np.random.normal(0, 0.003, T)

    # Cluster 2: Mean-reversion clan (Formulas 3, 4)
    m2 = np.random.normal(0.0005, 0.015, T)
    f3 = m2
    f4 = m2 + np.random.normal(0, 0.002, T)

    # Outlier / Orthogonal strategy (Formula 5)
    f5 = np.random.normal(0, 0.03, T)

    R = np.vstack([f0, f1, f2, f3, f4, f5])
    C = compute_correlation_matrix(R)
    D = compute_correlation_distance(C)
    link = hierarchical_linkage(D, method="ward")
    labels = assign_clusters(link, n_samples=6, rho_threshold=0.85)

    assert len(labels) == 6
    # Clan 1 must share the same cluster
    assert labels[0] == labels[1] == labels[2]
    # Clan 2 must share the same cluster
    assert labels[3] == labels[4]
    # Clan 1 and Clan 2 must have distinct clusters
    assert labels[0] != labels[3]
    print("[PASS] test_hierarchical_clustering_and_cluster_assignment")


def test_occam_pruning_and_deduplication():
    """Verify Occam's pruning selects parsimonious formula over slightly higher overfitted clone."""
    from src.clustering.return_clustering import prune_clusters

    # Cluster labels: 3 strategies in Cluster 1, 2 in Cluster 2
    labels = np.array([1, 1, 1, 2, 2])

    # Cluster 1:
    # Index 0: Sortino 2.50, Depth 8 -> Composite = 2.50 - 8*0.05 = 2.10
    # Index 1: Sortino 2.45, Depth 2 -> Composite = 2.45 - 2*0.05 = 2.35 (Winner!)
    # Index 2: Sortino 1.80, Depth 3 -> Composite = 1.80 - 3*0.05 = 1.65
    # Cluster 2:
    # Index 3: Sortino 3.00, Depth 4 -> Composite = 3.00 - 4*0.05 = 2.80 (Winner!)
    # Index 4: Sortino 2.50, Depth 2 -> Composite = 2.50 - 2*0.05 = 2.40
    fitness = np.array([2.50, 2.45, 1.80, 3.00, 2.50])
    depths = np.array([8, 2, 3, 4, 2])

    elites = prune_clusters(labels, fitness, depths, parsimony_penalty=0.05)

    assert elites == [1, 3]
    print("[PASS] test_occam_pruning_and_deduplication")


def test_return_clusterer_end_to_end():
    """Verify ReturnClusterer high-level interface clusters and prunes in single call."""
    from src.clustering.return_clustering import ReturnClusterer

    np.random.seed(42)
    T, N = 250, 4
    base = np.random.normal(0.001, 0.02, T)
    f0 = base
    f1 = base + np.random.normal(0, 0.001, T)  # Clone of 0
    f2 = np.random.normal(0.0005, 0.015, T)
    f3 = np.random.normal(0.0002, 0.010, T)
    R = np.vstack([f0, f1, f2, f3])

    fitness = np.array([2.0, 2.05, 1.5, 1.8])
    depths = np.array([2, 5, 3, 3])

    clusterer = ReturnClusterer(rho_threshold=0.85, parsimony_penalty=0.05)
    labels, elites = clusterer.cluster_and_prune(R, fitness, depths)

    assert len(labels) == 4
    assert labels[0] == labels[1]  # Clones identified
    assert 0 in elites  # Index 0 chosen over 1 due to lower depth (depth 2 vs 5)
    assert 1 not in elites  # Complex clone pruned
    print("[PASS] test_return_clusterer_end_to_end")


def load_tests(loader, standard_tests, pattern):
    import inspect
    import unittest
    suite = unittest.TestSuite()
    current_module = sys.modules[__name__]
    for name, func in inspect.getmembers(current_module, inspect.isfunction):
        if name.startswith("test_"):
            suite.addTest(unittest.FunctionTestCase(func))
    return suite


if __name__ == "__main__":
    import unittest
    unittest.main()
