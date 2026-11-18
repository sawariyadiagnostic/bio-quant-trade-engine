"""Unit tests for Muscle 9: MAP-Elites Behavioral Archive Engine."""
from __future__ import annotations

import os
import tempfile
import unittest
import numpy as np
import pandas as pd

from ravana.domain.formula_ast import ASTNode
from ravana.evolution.config import EvolutionConfig
from ravana.evolution.engine import GeneticAlphaEngine
from ravana.evolution.map_elites import EliteIndividual, MapElitesArchive


class TestMapElitesArchive(unittest.TestCase):
    """Test suite for 20x20 behavioral grid indexing, CIAS scoring, and SQLite sync."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test_alpha_cockpit.db")
        self.archive = MapElitesArchive(db_path=self.db_path)

    def test_coordinate_mapping_boundaries(self) -> None:
        """Verify continuous (volatility, turnover) correctly projects into [0..19, 0..19]."""
        # Origin
        r, c = self.archive.compute_cell_coordinates(0.0, 0.0)
        self.assertEqual((r, c), (0, 0))

        # Far upper-right corner
        r, c = self.archive.compute_cell_coordinates(0.39, 0.99)
        self.assertEqual((r, c), (19, 19))

        # Beyond bounds clipping
        r, c = self.archive.compute_cell_coordinates(0.55, 1.80)
        self.assertEqual((r, c), (19, 19))

        # Midpoint (~0.20 vol, ~0.50 turnover)
        r, c = self.archive.compute_cell_coordinates(0.20, 0.50)
        self.assertEqual((r, c), (10, 10))

    def test_cias_scoring_monotonicity(self) -> None:
        """Verify CIAS strictly increases with IC, DSR, Sharpe and decreases with MDD."""
        base_cias = self.archive.compute_cias(ic=0.10, rank_ic=0.08, dsr=0.60, sharpe=1.5, max_drawdown=0.05)
        better_ic = self.archive.compute_cias(ic=0.20, rank_ic=0.08, dsr=0.60, sharpe=1.5, max_drawdown=0.05)
        self.assertGreater(better_ic, base_cias)

        higher_mdd = self.archive.compute_cias(ic=0.10, rank_ic=0.08, dsr=0.60, sharpe=1.5, max_drawdown=0.25)
        self.assertLess(higher_mdd, base_cias)

    def test_add_and_replacement_policy(self) -> None:
        """Verify empty niche fills, weaker candidate rejected, superior candidate replaces."""
        # 1. Fill empty cell (row 5, col 5)
        added = self.archive.add(
            formula_str="ADD(LOG_RET_1, VWAP_DRIFT)",
            volatility=0.10,  # row 5
            turnover=0.25,    # col 5
            fitness=1.20,
            ic=0.05,
            rank_ic=0.04,
            dsr=0.50,
            sharpe=1.20,
        )
        self.assertTrue(added)
        self.assertEqual(len(self.archive.get_all_elites()), 1)

        resident = self.archive.get_elite(5, 5)
        self.assertIsNotNone(resident)
        self.assertEqual(resident.formula_str, "ADD(LOG_RET_1, VWAP_DRIFT)")

        # 2. Add weaker candidate to same cell -> Rejected
        added_weak = self.archive.add(
            formula_str="WEAK_CANDIDATE",
            volatility=0.10,
            turnover=0.25,
            fitness=0.40,
            ic=0.01,
            rank_ic=0.01,
            dsr=0.20,
            sharpe=0.40,
        )
        self.assertFalse(added_weak)
        # Resident unchanged
        self.assertEqual(self.archive.get_elite(5, 5).formula_str, "ADD(LOG_RET_1, VWAP_DRIFT)")

        # 3. Add superior candidate to same cell -> Replaced
        added_strong = self.archive.add(
            formula_str="CHAMPION_SUPERIOR",
            volatility=0.10,
            turnover=0.25,
            fitness=2.50,
            ic=0.18,
            rank_ic=0.15,
            dsr=0.85,
            sharpe=2.50,
        )
        self.assertTrue(added_strong)
        self.assertEqual(self.archive.get_elite(5, 5).formula_str, "CHAMPION_SUPERIOR")

    def test_database_persistence_roundtrip(self) -> None:
        """Verify sync_to_db persists elites and load_from_db reconstructs archive accurately."""
        # Add 3 distinct elites
        self.archive.add("ALPHA_A", volatility=0.05, turnover=0.10, fitness=1.5, ic=0.10)
        self.archive.add("ALPHA_B", volatility=0.25, turnover=0.60, fitness=2.0, ic=0.15)
        self.archive.add("ALPHA_C", volatility=0.35, turnover=0.85, fitness=1.8, ic=0.12)

        persisted_count = self.archive.sync_to_db()
        self.assertEqual(persisted_count, 3)

        # Reconstruct in a fresh archive instance
        fresh_archive = MapElitesArchive(db_path=self.db_path)
        loaded_count = fresh_archive.load_from_db()
        self.assertEqual(loaded_count, 3)
        self.assertEqual(len(fresh_archive.get_all_elites()), 3)

    def test_genetic_engine_archive_population(self) -> None:
        """Verify GeneticAlphaEngine populates archive during generational evolution."""
        dates = pd.date_range("2026-01-01 09:15", periods=50, freq="5min", tz="Asia/Kolkata")
        f1 = np.sin(np.linspace(0, 10, 50))
        features_df = pd.DataFrame({"LOG_RET_1": f1, "VWAP_DRIFT": f1 * 0.5}, index=dates)

        targets = f1 * 0.01
        is_entry = np.zeros(50, dtype=np.int32)
        is_entry[5::4] = 1
        targets_df = pd.DataFrame({"net_return": targets, "is_entry_bar": is_entry}, index=dates)

        cfg = EvolutionConfig(
            population_size=8,
            max_generations=2,
            feature_names=["LOG_RET_1", "VWAP_DRIFT"],
            random_seed=42,
        )
        engine = GeneticAlphaEngine(cfg)
        result = engine.evolve(features_df, targets_df, generations=2)

        self.assertIsNotNone(result.archive)
        self.assertGreater(len(result.archive.get_all_elites()), 0)


if __name__ == "__main__":
    unittest.main()
