"""Unit tests for Centroidal Voronoi Tessellation (CVT) Archive (Wave 3: Task 3.3)."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_cvt_archive_insertion_and_competition():
    """Verify CVTArchiveEngine maps continuous 3D coordinates to Voronoi cells and manages elitist competition."""
    from src.evolution.cvt_archive import CVTArchiveEngine

    archive = CVTArchiveEngine(cells=50, dims=3, seed=42)
    assert archive.total_cells == 50
    assert archive.occupied_count() == 0

    # 1. First insertion into a cell
    coords_1 = np.array([0.1, 0.2, -0.3], dtype=np.float32)
    inserted_1 = archive.insert(
        formula_str="F1_ALPHA",
        fitness=50.0,
        behavior_coords=coords_1
    )
    assert inserted_1 is True
    assert archive.occupied_count() == 1
    assert archive.coverage() == 1 / 50

    # 2. Challenger with lower fitness in identical cell: Must be REJECTED
    inserted_low = archive.insert(
        formula_str="F2_WEAK",
        fitness=40.0,
        behavior_coords=coords_1
    )
    assert inserted_low is False
    assert archive.occupied_count() == 1
    assert archive.get_cell_elite(coords_1)["formula_str"] == "F1_ALPHA"

    # 3. Challenger with higher fitness in identical cell: Must OVERWRITE
    inserted_high = archive.insert(
        formula_str="F3_SUPER_ALPHA",
        fitness=95.0,
        behavior_coords=coords_1
    )
    assert inserted_high is True
    assert archive.occupied_count() == 1
    assert archive.get_cell_elite(coords_1)["formula_str"] == "F3_SUPER_ALPHA"
    print("[PASS] test_cvt_archive_insertion_and_competition")


def test_cvt_archive_multi_niche_coverage():
    """Verify diverse coordinates occupy distinct Voronoi cells."""
    from src.evolution.cvt_archive import CVTArchiveEngine

    archive = CVTArchiveEngine(cells=100, dims=3, seed=42)
    np.random.seed(42)

    # Insert 40 random diverse points
    for i in range(40):
        coords = np.random.randn(3).astype(np.float32)
        archive.insert(formula_str=f"F_{i}", fitness=float(i), behavior_coords=coords)

    # Should occupy multiple diverse cells
    assert archive.occupied_count() > 15
    assert archive.coverage() > 0.15
    print("[PASS] test_cvt_archive_multi_niche_coverage")


if __name__ == "__main__":
    test_cvt_archive_insertion_and_competition()
    test_cvt_archive_multi_niche_coverage()
