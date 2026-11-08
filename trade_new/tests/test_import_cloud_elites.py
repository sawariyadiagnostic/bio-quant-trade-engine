"""
test_import_cloud_elites.py
Institutional unit test suite for Cloud GPU Elite Import & Audit Pipeline.
Validates JSON loading, formula parsing, Thymic Chamber crash filtering,
and WAL SQLite archive store persistence.
"""
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

# Ensure trade_new is first on sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
trade_new_dir = os.path.abspath(os.path.join(script_dir, ".."))
if "" in sys.path:
    sys.path.remove("")
cwd = os.getcwd()
if cwd in sys.path and os.path.abspath(cwd) != trade_new_dir:
    sys.path.remove(cwd)
if trade_new_dir not in sys.path:
    sys.path.insert(0, trade_new_dir)

try:
    from scripts.import_cloud_elites import (
        audit_candidate,
        build_arg_parser,
        format_audit_row,
        load_cloud_export,
        main,
        normalize_candidate,
        persist_survivors,
    )
except ModuleNotFoundError:
    from trade_new.scripts.import_cloud_elites import (
        audit_candidate,
        build_arg_parser,
        format_audit_row,
        load_cloud_export,
        main,
        normalize_candidate,
        persist_survivors,
    )
from src.evolution.archive_store import ArchiveStore
from src.evolution.thymic_chamber import ThymicChamber


class TestImportCloudElites(unittest.TestCase):
    """Test suite covering cloud elite importing, Thymic Chamber crash filtering, and SQLite storage."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_alpha_elites.db")
        self.sample_json_path = os.path.join(self.temp_dir, "test_elites.json")
        self.chamber = ThymicChamber()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_cloud_export_dict_format(self):
        """Validates loading JSON exports structured as a metadata dictionary."""
        payload = {
            "ticker": "BANKBEES",
            "lookback": "5y",
            "exported_at": "2026-11-08 12:00:00 UTC",
            "total_elites": 2,
            "elites": [
                {"formula_str": "SUB(RSI_14, 0.50)", "sortino": 2.5, "net_return_pct": 120.0},
                {"formula_str": "EMA_10(Vol_Price_Flow)", "sortino": 2.1, "net_return_pct": 95.0}
            ]
        }
        with open(self.sample_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        ticker, lookback, candidates = load_cloud_export(self.sample_json_path)
        self.assertEqual(ticker, "BANKBEES")
        self.assertEqual(lookback, "5y")
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0]["formula_str"], "SUB(RSI_14, 0.50)")
        self.assertEqual(candidates[0]["sortino"], 2.5)

    def test_load_cloud_export_list_format(self):
        """Validates loading JSON exports structured as a raw list of candidates."""
        payload = [
            {"formula_str": "RSI_14", "sortino": 2.2},
            "EMA_Slope"
        ]
        with open(self.sample_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        ticker, lookback, candidates = load_cloud_export(self.sample_json_path)
        self.assertIsNone(ticker)
        self.assertIsNone(lookback)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0]["formula_str"], "RSI_14")
        self.assertEqual(candidates[1]["formula_str"], "EMA_Slope")

    def test_load_cloud_export_file_not_found(self):
        """Validates FileNotFoundError is raised when file does not exist."""
        non_existent = os.path.join(self.temp_dir, "missing.json")
        with self.assertRaises(FileNotFoundError):
            load_cloud_export(non_existent)

    def test_normalize_candidate_various_inputs(self):
        """Validates candidate normalization across string, partial, and full dictionaries."""
        norm_str = normalize_candidate("  (Close_Adj - Open)  ")
        self.assertEqual(norm_str["formula_str"], "(Close_Adj - Open)")
        self.assertEqual(norm_str["sortino"], 0.0)

        norm_dict = normalize_candidate({
            "formula": "RSI_14",
            "sortino": "3.14",
            "net_return_pct": "150.5",
            "trades_count": "100"
        })
        self.assertEqual(norm_dict["formula_str"], "RSI_14")
        self.assertAlmostEqual(norm_dict["sortino"], 3.14)
        self.assertAlmostEqual(norm_dict["net_return_pct"], 150.5)
        self.assertEqual(norm_dict["trades_count"], 100)

    def test_audit_candidate_sortino_floor(self):
        """Validates that candidate with Sortino below floor is rejected with BELOW_SORTINO."""
        cand = {"formula_str": "SUB(RSI_14, 0.50)", "sortino": 1.4, "fitness": 1.4}
        res = audit_candidate(cand, self.chamber, min_sortino=2.0, max_dd_tolerance=0.06)
        self.assertEqual(res["status"], "BELOW_SORTINO")
        self.assertFalse(res["passed"])

    def test_audit_candidate_parse_error(self):
        """Validates that malformed formula syntax results in PARSE_ERROR."""
        cand = {"formula_str": "SUB(((RSI_14)", "sortino": 3.0, "fitness": 3.0}
        res = audit_candidate(cand, self.chamber, min_sortino=2.0, max_dd_tolerance=0.06)
        self.assertEqual(res["status"], "PARSE_ERROR")
        self.assertFalse(res["passed"])

    def test_audit_candidate_thymic_quarantine(self):
        """Validates that candidate with high crash drawdown (>6%) is quarantined."""
        cand = {"formula_str": "MUL(Close_Adj, 5.0)", "sortino": 3.5, "fitness": 3.5}
        res = audit_candidate(cand, self.chamber, min_sortino=2.0, max_dd_tolerance=0.06)
        self.assertEqual(res["status"], "QUARANTINED")
        self.assertFalse(res["passed"])
        self.assertGreater(res["stress_dd"], 0.06)

    def test_audit_candidate_thymic_survivor(self):
        """Validates that candidate surviving acute crash stress passes with SURVIVOR status."""
        cand = {"formula_str": "SUB(RSI_14, 0.50)", "sortino": 2.5, "fitness": 2.5}
        res = audit_candidate(cand, self.chamber, min_sortino=2.0, max_dd_tolerance=0.06)
        self.assertEqual(res["status"], "SURVIVOR")
        self.assertTrue(res["passed"])
        self.assertLessEqual(res["stress_dd"], 0.06)

    def test_persist_survivors_dry_run_vs_commit(self):
        """Validates that dry-run skips SQLite insertion while commit inserts rows."""
        store = ArchiveStore(db_path=self.test_db_path)
        survivors = [
            {
                "formula_str": "SUB(RSI_14, 0.50)",
                "sortino": 2.8,
                "fitness": 3.0,
                "net_return_pct": 140.0,
                "cagr_pct": 28.0,
                "max_drawdown": 0.04,
                "win_rate": 62.0,
                "trades_count": 120,
                "row": 2,
                "col": 4
            }
        ]

        # Dry run: count should remain 0
        count_dry = persist_survivors(store, "BANKBEES", "5y", survivors, dry_run=True)
        self.assertEqual(count_dry, 0)

        # Commit: count should become 1
        count_commit = persist_survivors(store, "BANKBEES", "5y", survivors, dry_run=False)
        self.assertEqual(count_commit, 1)

        # Verify row in database
        elites = store.get_top_elites(limit=10, ticker="BANKBEES")
        self.assertEqual(len(elites), 1)
        self.assertEqual(elites[0]["formula_str"], "SUB(RSI_14, 0.50)")
        self.assertAlmostEqual(elites[0]["sortino"], 2.8)

    def test_format_audit_row(self):
        """Validates ASCII row formatting length and structure."""
        item = {
            "formula_str": "SUB(RSI_14, 0.50)",
            "sortino": 2.8,
            "net_return_pct": 145.2,
            "stress_dd": 0.025,
            "status": "SURVIVOR"
        }
        row = format_audit_row(item, 1)
        self.assertIn("SUB(RSI_14, 0.50)", row)
        self.assertIn("2.80", row)
        self.assertIn("SURVIVOR", row)
        self.assertIn("COMMITTED", row)

    def test_cli_main_dry_run_e2e(self):
        """Validates end-to-end execution of main() in dry-run mode."""
        payload = {
            "ticker": "BANKBEES",
            "lookback": "5y",
            "elites": [
                {"formula_str": "SUB(RSI_14, 0.50)", "sortino": 2.5, "net_return_pct": 110.0},
                {"formula_str": "MUL(Close_Adj, 10.0)", "sortino": 3.5, "net_return_pct": 200.0},
                {"formula_str": "RSI_14", "sortino": 1.2, "net_return_pct": 40.0}
            ]
        }
        with open(self.sample_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        args = [
            "--file", self.sample_json_path,
            "--ticker", "BANKBEES",
            "--min-sortino", "2.0",
            "--db-path", self.test_db_path,
            "--dry-run"
        ]
        ret = main(args)
        self.assertEqual(ret, 0)

        # Verify DB remained untouched in dry-run
        store = ArchiveStore(db_path=self.test_db_path)
        self.assertEqual(store.count_champions(ticker="BANKBEES"), 0)

    def test_cli_main_commit_e2e(self):
        """Validates end-to-end execution of main() persisting surviving champions."""
        payload = {
            "ticker": "BANKBEES",
            "lookback": "5y",
            "elites": [
                {"formula_str": "SUB(RSI_14, 0.50)", "sortino": 2.5, "net_return_pct": 110.0},
                {"formula_str": "MUL(Close_Adj, 10.0)", "sortino": 3.5, "net_return_pct": 200.0}
            ]
        }
        with open(self.sample_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        args = [
            "--file", self.sample_json_path,
            "--ticker", "BANKBEES",
            "--min-sortino", "2.0",
            "--db-path", self.test_db_path
        ]
        ret = main(args)
        self.assertEqual(ret, 0)

        # 1 survivor expected (SUB(RSI_14, 0.50)); MUL(Close_Adj, 10.0) is quarantined
        store = ArchiveStore(db_path=self.test_db_path)
        self.assertEqual(store.count_champions(ticker="BANKBEES"), 1)


if __name__ == "__main__":
    unittest.main()
