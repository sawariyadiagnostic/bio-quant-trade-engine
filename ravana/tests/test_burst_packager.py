"""Unit tests for KaggleBurstPackager in RAVANA v4.3."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from ravana.compute.burst_packager import KaggleBurstPackager
from ravana.compute.kaggle_runner import KaggleBurstConfig, KaggleQuotaInfo
from ravana.evolution.map_elites import MapElitesArchive


class TestKaggleBurstPackager(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "test_elites.db"
        self.archive = MapElitesArchive(db_path=str(self.db_path))

        self.cfg = KaggleBurstConfig(kernel_slug="test-burst")
        self.packager = KaggleBurstPackager(config=self.cfg, archive=self.archive)
        self.packager.dispatcher._api = MagicMock()
        self.packager.dispatcher._username = "testuser"

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_build_burst_bundle(self) -> None:
        bundle_dir = Path(self.tmp_dir.name) / "bundle"
        out_path = self.packager.build_burst_bundle(bundle_dir, pop_size=500, generations=20)
        self.assertTrue((out_path / "run.py").exists())
        self.assertTrue((out_path / "kernel-metadata.json").exists())

    def test_assimilate_champions_into_archive(self) -> None:
        results_file = Path(self.tmp_dir.name) / "champion_results.json"
        data = {
            "population_size": 1000,
            "generations": 50,
            "champion_formula": "TsRank(Return(close, 5), 10)",
            "metrics": {
                "ic": 0.0824,
                "rank_ic": 0.0765,
                "dsr": 1.42,
                "fdr_q": 0.012,
                "turnover": 0.18,
                "mdd": 0.065,
            },
        }
        results_file.write_text(json.dumps(data), encoding="utf-8")

        count = self.packager.assimilate_champions(results_file)
        self.assertEqual(count, 1)

        # Verify champion is saved in archive
        champ = self.archive.get_best_champion(ticker="GLOBAL_BURST")
        self.assertIsNotNone(champ)
        self.assertEqual(champ.formula_str, "TsRank(Return(close, 5), 10)")
        self.assertAlmostEqual(champ.fitness, 0.0765)


if __name__ == "__main__":
    unittest.main()
