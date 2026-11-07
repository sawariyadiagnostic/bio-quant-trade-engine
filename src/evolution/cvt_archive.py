"""Centroidal Voronoi Tessellation (CVT) Quality-Diversity Archive (Wave 3: Task 3.3)."""
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class CVTArchiveEngine:
    """
    Centroidal Voronoi Tessellation (CVT) MAP-Elites Archive.
    Replaces rigid discrete grids with unconstrained Voronoi cell tessellations
    mapping continuous 3D UMAP behavioral manifolds.
    """
    cells: int = 500
    dims: int = 3
    seed: int = 42

    def __post_init__(self):
        self.total_cells = self.cells
        rng = np.random.RandomState(self.seed)
        # Initialize centroid centers using standardized Gaussian distribution
        self.centroids = rng.normal(0.0, 1.0, (self.cells, self.dims)).astype(np.float32)
        # Normalize centroids to unit sphere for stable topological distance
        norms = np.linalg.norm(self.centroids, axis=1, keepdims=True)
        self.centroids = self.centroids / np.maximum(norms, 1e-8)

        self.cells_data: Dict[int, Dict[str, Any]] = {}

    def find_cell(self, behavior_coords: np.ndarray) -> int:
        """Finds the index of the nearest Voronoi centroid in continuous manifold space."""
        b = np.asarray(behavior_coords, dtype=np.float32).flatten()
        if len(b) != self.dims:
            raise ValueError(f"Expected {self.dims} behavioral dimensions, got {len(b)}")

        b_norm = b / max(1e-8, float(np.linalg.norm(b)))
        dists = np.sum((self.centroids - b_norm[np.newaxis, :]) ** 2, axis=1)
        return int(np.argmin(dists))

    def insert(
        self,
        formula_str: str,
        fitness: float,
        behavior_coords: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Inserts an individual into the archive if the target cell is unoccupied
        or if challenger fitness strictly dominates the incumbent.
        """
        cell_idx = self.find_cell(behavior_coords)
        incumbent = self.cells_data.get(cell_idx)

        if incumbent is None or fitness > incumbent["fitness"]:
            self.cells_data[cell_idx] = {
                "cell_idx": cell_idx,
                "formula_str": formula_str,
                "fitness": float(fitness),
                "behavior_coords": np.asarray(behavior_coords, dtype=np.float32),
                "metadata": metadata or {}
            }
            return True
        return False

    def occupied_count(self) -> int:
        """Returns the number of non-empty Voronoi cells."""
        return len(self.cells_data)

    def coverage(self) -> float:
        """Returns the behavioral coverage ratio [0.0, 1.0]."""
        return len(self.cells_data) / max(1, self.total_cells)

    def get_cell_elite(self, behavior_coords: np.ndarray) -> Optional[Dict[str, Any]]:
        """Returns the champion formula occupying the Voronoi cell nearest to the coordinates."""
        cell_idx = self.find_cell(behavior_coords)
        return self.cells_data.get(cell_idx)
