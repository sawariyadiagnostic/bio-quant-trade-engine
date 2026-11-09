"""Centroidal Voronoi Tessellation (CVT) Quality-Diversity Archive (Wave 3: Task 3.3)."""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from scipy.spatial import cKDTree

# Standard 4D behavioral manifold boundaries:
# 1. Realized Volatility: [0.01, 0.80] (supports low-vol cash-preservation up to 80% momentum)
# 2. Portfolio Turnover: [0.05, 250.0] (supports patient swing macro up to 250x intraday churn)
# 3. Downside Beta to Market: [-1.5, 2.5] (supports inverse hedging up to 2.5x high-beta)
# 4. Lag-1 Autocorrelation: [-0.8, 0.8] (supports mean-reversion and momentum persistence)
BOUNDS_4D = np.array([
    [0.01, 0.80],
    [0.05, 250.0],
    [-1.5, 2.5],
    [-0.8, 0.8]
], dtype=np.float64)


@dataclass
class CVTArchiveEngine:
    """
    Centroidal Voronoi Tessellation (CVT) MAP-Elites Archive.
    Replaces rigid discrete grids with unconstrained Voronoi cell tessellations
    mapping continuous 4D behavioral manifolds with O(log K) cKDTree queries.
    """
    cells: int = 1000
    dims: int = 4
    seed: int = 42
    bounds: np.ndarray = field(default_factory=lambda: BOUNDS_4D.copy())

    def __post_init__(self) -> None:
        self.total_cells = self.cells
        if self.dims != len(self.bounds):
            if self.dims == 3:
                self.bounds = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]], dtype=np.float64)
            else:
                self.bounds = np.array([[-1.0, 1.0]] * self.dims, dtype=np.float64)

        self._init_centroids()
        self.cells_data: Dict[int, Dict[str, Any]] = {}

    def _init_centroids(self) -> None:
        """Initializes Voronoi centroids and builds spatial KD-tree in normalized [0, 1]^D space."""
        rng = np.random.RandomState(self.seed)
        # Uniform sampling across unit hypercube [0, 1]^D
        self.norm_centroids = rng.uniform(0.0, 1.0, (self.cells, self.dims)).astype(np.float64)
        self.kdtree = cKDTree(self.norm_centroids)

        # Unnormalized centroid coordinates in real behavioral space
        span = self.bounds[:, 1] - self.bounds[:, 0]
        self.centroids = self.norm_centroids * span[np.newaxis, :] + self.bounds[:, 0][np.newaxis, :]

    def normalize_coords(self, coords: np.ndarray) -> np.ndarray:
        """Normalizes arbitrary behavioral coordinates to [0, 1]^D bounded hypercube."""
        c = np.asarray(coords, dtype=np.float64).flatten()
        if len(c) != self.dims:
            raise ValueError(f"Expected {self.dims} behavioral dimensions, got {len(c)}")
        low = self.bounds[:, 0]
        span = np.maximum(self.bounds[:, 1] - low, 1e-8)
        return np.clip((c - low) / span, 0.0, 1.0)

    def update_bounds_dynamically(self, coords: np.ndarray, margin: float = 0.1) -> bool:
        """Dynamically expands Voronoi hypercube bounds if new behavioral coordinates exceed limits."""
        c = np.asarray(coords, dtype=np.float64).flatten()
        expanded = False
        for i in range(self.dims):
            if c[i] < self.bounds[i, 0]:
                self.bounds[i, 0] = c[i] - margin * max(abs(c[i]), 1e-4)
                expanded = True
            elif c[i] > self.bounds[i, 1]:
                self.bounds[i, 1] = c[i] + margin * max(abs(c[i]), 1e-4)
                expanded = True
        if expanded:
            self._init_centroids()
        return expanded

    def find_cell(self, behavior_coords: np.ndarray) -> int:
        """Finds the index of the nearest Voronoi centroid in continuous manifold space in O(log K)."""
        self.update_bounds_dynamically(behavior_coords)
        norm_b = self.normalize_coords(behavior_coords)
        _, cell_idx = self.kdtree.query(norm_b)
        return int(cell_idx)

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

    def get_cell_elite(self, behavior_coords: np.ndarray) -> Optional[Dict[str, Any]]:
        """Returns champion formula occupying Voronoi cell nearest to given coordinates."""
        cell_idx = self.find_cell(behavior_coords)
        return self.cells_data.get(cell_idx)

    def get_elite_by_idx(self, cell_idx: int) -> Optional[Dict[str, Any]]:
        """Returns champion formula occupying specific Voronoi cell index."""
        return self.cells_data.get(cell_idx)

    def occupied_count(self) -> int:
        """Returns count of occupied Voronoi niches."""
        return len(self.cells_data)

    def coverage(self) -> float:
        """Returns behavioral space coverage ratio in range [0.0, 1.0]."""
        return len(self.cells_data) / max(1, self.total_cells)

    def get_all_elites(self) -> List[Dict[str, Any]]:
        """Returns list of all resident elite dictionaries in the archive."""
        return list(self.cells_data.values())

    def get_top_elites(self, limit: int = 10, metric: str = "fitness") -> List[Dict[str, Any]]:
        """Returns top elites sorted by fitness or chosen metric descending."""
        elites = self.get_all_elites()
        return sorted(elites, key=lambda x: float(x.get(metric, x.get("fitness", 0.0))), reverse=True)[:limit]

    def to_dict(self) -> Dict[str, Any]:
        """Serializes CVT archive state to dictionary."""
        serialized_cells = {}
        for idx, item in self.cells_data.items():
            serialized_cells[str(idx)] = {
                "cell_idx": item["cell_idx"],
                "formula_str": item["formula_str"],
                "fitness": item["fitness"],
                "behavior_coords": item["behavior_coords"].tolist(),
                "metadata": item["metadata"]
            }
        return {
            "cells": self.cells,
            "dims": self.dims,
            "occupied_count": self.occupied_count(),
            "coverage": self.coverage(),
            "cells_data": serialized_cells
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Restores CVT archive state from serialized dictionary."""
        self.cells_data.clear()
        for idx_str, item in data.get("cells_data", {}).items():
            idx = int(idx_str)
            self.cells_data[idx] = {
                "cell_idx": item["cell_idx"],
                "formula_str": item["formula_str"],
                "fitness": float(item["fitness"]),
                "behavior_coords": np.asarray(item["behavior_coords"], dtype=np.float32),
                "metadata": item.get("metadata", {})
            }
