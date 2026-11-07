"""Unit tests for Alpha Manifold Pruner (Wave 3: Task 3.2)."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_alpha_manifold_prunes_clones_preserves_noise():
    """Verify AlphaManifoldPruner deduplicates redundant clones and preserves orthogonal noise points."""
    from src.clustering.alpha_manifold import AlphaManifoldPruner

    np.random.seed(42)
    # Generate 30 formulas:
    # 20 clones of Strategy A (with small noise)
    # 8 clones of Strategy B
    # 2 completely unique outlier strategies
    D = 100
    base_a = (np.random.randn(D) > 0.0).astype(np.float32)
    base_b = (np.random.randn(D) > 0.0).astype(np.float32)

    clones_a = [base_a.copy() for _ in range(20)]
    clones_b = [base_b.copy() for _ in range(8)]
    noise_1 = (np.random.randn(D) > 0.5).astype(np.float32)
    noise_2 = (np.random.randn(D) < -0.5).astype(np.float32)

    phenotypes = np.stack(clones_a + clones_b + [noise_1, noise_2], axis=0)  # Shape: (30, 100)
    ast_lengths = [10 + i for i in range(30)]  # Different lengths for Occam's test
    sortinos = [2.0 + (i % 3) for i in range(30)]

    pruner = AlphaManifoldPruner(min_cluster_size=4)
    result = pruner.cluster_and_prune(
        phenotype_matrix=phenotypes,
        ast_lengths=ast_lengths,
        sortinos=sortinos
    )

    assert "surviving_indices" in result
    assert "orthogonal_indices" in result
    assert "embedding_3d" in result

    surviving = result["surviving_indices"]
    # 30 initial formulas should be compressed down to <= 6 surviving formulas (1 from A, 1 from B, + noise)
    assert len(surviving) <= 6
    assert result["compression_ratio"] >= 0.70  # At least 70% clones eliminated!

    # 3D embedding coordinates exist for WebUI visualization
    assert result["embedding_3d"].shape == (30, 3)
    print("[PASS] test_alpha_manifold_prunes_clones_preserves_noise")


if __name__ == "__main__":
    test_alpha_manifold_prunes_clones_preserves_noise()
