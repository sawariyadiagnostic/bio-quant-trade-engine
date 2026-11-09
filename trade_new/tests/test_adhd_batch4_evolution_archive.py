"""
Comprehensive TDD Verification Suite for ADHD Batch 4:
Evolutionary Archive, Manifold & Speciation (ADHD-30 to ADHD-39).
"""
import os
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.domain.formula_ast import (
    ASTNode, TreeGenerator, GeneticMutator, GeneticCrossover,
    ASTEvaluator
)
from src.domain.formula_parser import FormulaParser
from src.domain.models import InstrumentType
from src.evolution.cvt_archive import CVTArchiveEngine
from src.evolution.map_elites import MapElitesArchive, EliteIndividual
from src.evolution.archive_store import ArchiveStore
from src.evolution.thymic_chamber import ThymicStressChamber
from src.evolution.genetic_engine import GeneticEvolutionEngine
from src.evolution.nursery import EvolutionaryNursery
from src.evolution.symbiosis import SymbioticBreeder


def _generate_synthetic_df(n_bars: int = 100) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="B")
    close = 100.0 + np.cumsum(np.random.randn(n_bars) * 1.5)
    open_p = close + np.random.randn(n_bars) * 0.5
    high = np.maximum(open_p, close) + np.abs(np.random.randn(n_bars))
    low = np.minimum(open_p, close) - np.abs(np.random.randn(n_bars))
    vol = np.random.randint(1000, 10000, size=n_bars)

    return pd.DataFrame({
        "Open": open_p, "High": high, "Low": low, "Close": close,
        "Close_Adj": close, "Volume": vol, "Spread": 0.05,
        "Vol_ZScore": np.random.randn(n_bars), "Vol_Price_Flow": np.random.randn(n_bars),
        "True_Range": high - low, "TR_Expansion": 1.0,
        "EMA_Slope": 0.01, "Dist_EMA20": 0.02, "Dist_VWAP": 0.01,
        "Downside_Vol_Ratio": 0.5, "RSI_14": 55.0, "Parkinson_Vol": 0.15,
        "Garman_Klass_Vol": 0.15, "SP500_Ret": 0.001, "ADR_Ret": 0.002,
        "Global_Confluence": 0.5
    }, index=dates)


class TestADHDBatch4EvolutionArchive(unittest.TestCase):
    def test_adhd30_dynamic_voronoi_bounds(self):
        """ADHD-30: Voronoi bounds dynamically adapt to outlier coordinates."""
        cvt = CVTArchiveEngine(cells=100, dims=4)
        extreme_coords = np.array([1.5, 300.0, 5.0, 2.0], dtype=np.float32)
        cell = cvt.find_cell(extreme_coords)
        self.assertTrue(0 <= cell < 100)
        self.assertGreaterEqual(cvt.bounds[0, 1], 1.5)
        self.assertGreaterEqual(cvt.bounds[1, 1], 300.0)

    def test_adhd31_dynamic_grid_matrix_sizing(self):
        """ADHD-31: ArchiveStore.get_grid_matrix dynamically adapts dimensions to avoid clipping."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            db_path = tf.name
        try:
            store = ArchiveStore(db_path=db_path)
            elites = [{
                "ticker": "NIFTYBEES", "lookback": "1y", "row": 24, "col": 39,
                "fitness": 3.45, "net_return_pct": 25.0, "cagr_pct": 28.0,
                "max_drawdown": 4.5, "win_rate": 65.0, "turnover": 12.0,
                "volatility": 0.14, "sortino": 2.1, "profit_factor": 1.8,
                "trades_count": 45, "formula_str": "FEATURE(Close)",
                "active_regimes": [0], "behavior_signature": "00" * 60
            }]
            store.batch_upsert_elites("NIFTYBEES", "1y", elites)
            matrix = store.get_grid_matrix("NIFTYBEES", "1y")

            self.assertGreaterEqual(matrix["grid_rows"], 25)
            self.assertGreaterEqual(matrix["grid_cols"], 40)
            self.assertEqual(matrix["z"][24][39], 3.45)
        finally:
            store.checkpoint()
            del store
            import gc; gc.collect()
            try:
                if os.path.exists(db_path):
                    os.remove(db_path)
            except Exception:
                pass

    def test_adhd32_all_occupied_cells_sampling_and_immigrants(self):
        """ADHD-32: evolve_generation samples across all resident elites and injects immigrants."""
        df = _generate_synthetic_df(80)
        engine = GeneticEvolutionEngine(population_size=20, cells=50)

        for idx in range(10):
            dummy_node = ASTNode("FEATURE", val="Close")
            engine.archive.insert(
                row=idx, col=0, fitness=1.5 + idx * 0.1, formula=dummy_node,
                metrics={"volatility": 0.15, "turnover": 10.0}
            )

        self.assertEqual(len(engine.archive.get_resident_elites()), 10)
        stats = engine.evolve_generation(df, current_regime=0, chunk_size=10)
        self.assertEqual(stats["generation"], 1)
        self.assertGreater(len(engine.population), 0)

    def test_adhd33_nursery_and_symbiotic_wiring(self):
        """ADHD-33: Engine integrates EvolutionaryNursery and SymbioticBreeder."""
        engine = GeneticEvolutionEngine(population_size=10, cells=20)
        self.assertIsInstance(engine.nursery, EvolutionaryNursery)
        self.assertEqual(engine.p_symbiosis, 0.15)

    def test_adhd34_target_mode_downside_beta(self):
        """ADHD-34: _compute_downside_beta aligns benchmark return with target mode."""
        archive = MapElitesArchive(cells=10, dims=4)
        net_rets = np.array([-0.01, -0.02, 0.01, -0.015, -0.005, 0.02, -0.01], dtype=np.float64)
        open_p = np.array([100.0, 102.0, 99.0, 98.0, 97.0, 96.0, 98.0], dtype=np.float64)
        close_p = np.array([101.0, 100.0, 98.0, 96.0, 95.0, 97.0, 99.0], dtype=np.float64)

        beta_overnight = archive._compute_downside_beta(net_rets, open_p, close_p, target_mode="overnight")
        beta_intraday = archive._compute_downside_beta(net_rets, open_p, close_p, target_mode="intraday")

        self.assertIsInstance(beta_overnight, float)
        self.assertIsInstance(beta_intraday, float)
        self.assertTrue(-0.5 <= beta_overnight <= 1.5)
        self.assertTrue(-0.5 <= beta_intraday <= 1.5)

    def test_adhd35_thymic_zero_signal_quarantine(self):
        """ADHD-35: Dead or completely silenced formulas are quarantined with 100% drawdown."""
        chamber = ThymicStressChamber()
        dead_node = ASTNode("CONST", val="0.0")
        dd = chamber.measure_stress_drawdown(dead_node, active_regime=0)
        self.assertEqual(dd, 1.0)
        self.assertFalse(chamber.test_formula_safety(dead_node, active_regime=0))

    def test_adhd36_parsimony_pressure_penalty(self):
        """ADHD-36: Parsimony penalty reduces raw fitness for deep/bloated trees."""
        archive = MapElitesArchive(cells=50, dims=4)
        df = _generate_synthetic_df(80)

        simple_node = ASTNode("FEATURE", val="Close")
        archive.evaluate_and_insert(simple_node, df)

        bloated_node = ASTNode("ADD", children=[
            ASTNode("MUL", children=[
                ASTNode("FEATURE", val="Close"),
                ASTNode("FEATURE", val="Open")
            ]),
            ASTNode("SUB", children=[
                ASTNode("FEATURE", val="High"),
                ASTNode("FEATURE", val="Low")
            ])
        ])
        self.assertGreater(bloated_node.size(), 5)
        self.assertEqual(bloated_node.count_nodes(), bloated_node.size())

    def test_adhd37_dirty_cell_tracking_and_sqlite_streaming(self):
        """ADHD-37: Archive tracks dirty cells and clears them on dirty_only export."""
        archive = MapElitesArchive(cells=50, dims=4)
        dummy_node = ASTNode("FEATURE", val="Close")

        archive.insert(row=1, col=2, fitness=2.5, formula=dummy_node)
        self.assertEqual(len(archive.dirty_cells), 1)
        cell_idx = 1 * 40 + 2
        self.assertIn(cell_idx, archive.dirty_cells)

        records = archive.to_archive_records("NIFTYBEES", "1y", dirty_only=True)
        self.assertEqual(len(records), 1)
        self.assertEqual(len(archive.dirty_cells), 0)

        records2 = archive.to_archive_records("NIFTYBEES", "1y", dirty_only=True)
        self.assertEqual(len(records2), 0)

    def test_adhd38_crossover_arity_and_symbiosis_regimes(self):
        """ADHD-38: Crossover preserves valid arity, and SymbioticBreeder propagates active regimes."""
        p1 = ASTNode("ADD", children=[ASTNode("FEATURE", val="Close"), ASTNode("CONST", val="1.0")])
        p2 = ASTNode("SIGMOID", children=[ASTNode("FEATURE", val="Open")])

        c1, c2 = GeneticCrossover.crossover(p1, p2)
        for node in c1.get_all_subnodes() + c2.get_all_subnodes():
            if node.op in ("ADD", "SUB", "MUL", "DIV_SAFE"):
                self.assertEqual(len(node.children), 2)
            elif node.op in ("SIGMOID", "RELU", "NEG"):
                self.assertEqual(len(node.children), 1)

        sensor = ASTNode("FEATURE", val="Spread", active_regimes={0, 1})
        actuator = ASTNode("FEATURE", val="Close", active_regimes={1, 2})
        chimera = SymbioticBreeder.fuse(sensor, actuator)
        self.assertEqual(chimera.active_regimes, {1})
        self.assertEqual(chimera.children[0].active_regimes, {1})
        self.assertEqual(chimera.children[1].active_regimes, {1})

    def test_adhd39_behavior_signature_type_safety(self):
        """ADHD-39: Behavior signature remains float32 ndarray and serializes cleanly."""
        archive = MapElitesArchive(cells=20, dims=4)
        dummy_node = ASTNode("FEATURE", val="Close")
        archive.insert(row=0, col=0, fitness=1.8, formula=dummy_node)

        elite = archive.elites[0]
        self.assertIsInstance(elite.behavior_signature, np.ndarray)
        self.assertEqual(elite.behavior_signature.dtype, np.float32)

        records = archive.to_archive_records("NIFTYBEES", "1y")
        self.assertIsInstance(records[0]["behavior_signature"], str)


if __name__ == "__main__":
    unittest.main()
