"""Unit tests for LakeDiffInspector."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from ravana.data.storage.diff import LakeDiffInspector, LakePartitionDiff


class TestLakeDiffInspector(unittest.TestCase):
    """Test verification suite for partition diffing and snapshot comparison."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.tmp_dir.name)
        self.snap_dir = self.base_dir / ".snapshots"
        self.snap_dir.mkdir(parents=True)

        idx = pd.date_range("2026-01-01", periods=10, freq="5min")
        self.df_old = pd.DataFrame({
            "A": np.zeros(10),
            "B": [1.0] * 10,
        }, index=idx)

        self.df_new = pd.DataFrame({
            "A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "B": [2.0] * 10,
            "C": [0.5] * 10,
        }, index=idx)

        self.active_path = self.base_dir / "TEST_features.parquet"
        self.snap_path = self.snap_dir / "TEST_features_1000.parquet"

        self.df_new.to_parquet(self.active_path)
        self.df_old.to_parquet(self.snap_path)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_compare_partition_revives_dead_feature(self) -> None:
        diff = LakeDiffInspector.compare_partition(self.active_path, self.snap_path)
        self.assertIn("C", diff.columns_added)
        self.assertEqual(diff.new_rows, 10)
        self.assertEqual(diff.old_rows, 10)

        stat_a = diff.column_stats["A"]
        self.assertTrue(stat_a.is_dead_old)
        self.assertFalse(stat_a.is_dead_new)
        self.assertEqual(stat_a.old_zeros, 10)
        self.assertEqual(stat_a.new_zeros, 0)

    def test_summary_table_formatting(self) -> None:
        diff = LakeDiffInspector.compare_partition(self.active_path, self.snap_path)
        table = diff.summary_table()
        self.assertIn("=== Lakehouse Partition Diff ===", table)
        self.assertIn("REVIVED", table)


if __name__ == "__main__":
    unittest.main()
