"""Unit tests for Muscle 6: Sovereign Columnar Parquet Data Lake & LakeManager.
Verifies schema validation, incremental upsert deduplication, atomic writes,
zero-copy column projections, time-range queries, and multi-tier routing.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from ravana.data.storage.config import StorageConfig
from ravana.data.storage.manager import LakeManager
from ravana.data.storage.schema import LakeSchemaValidator, SchemaValidationError
from ravana.data.storage.upsert import IncrementalUpsertEngine


class TestDataStorageEngine(unittest.TestCase):
    """Institutional test suite for Muscle 6 Sovereign Parquet Lakehouse."""

    def setUp(self) -> None:
        """Create temporary test directory and synthetic data."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.lake_root = Path(self.temp_dir.name)
        self.tz = "Asia/Kolkata"
        self.config = StorageConfig(lake_root=self.lake_root)
        self.manager = LakeManager(self.config)
        self.upsert_engine = IncrementalUpsertEngine(self.config)

        self.times = pd.date_range("2026-03-09 09:15:00", "2026-03-09 15:25:00", freq="5min", tz=self.tz)
        self.sample_df = pd.DataFrame(
            {
                "Open": np.linspace(100.0, 105.0, len(self.times)),
                "High": np.linspace(101.0, 106.0, len(self.times)),
                "Low": np.linspace(99.0, 104.0, len(self.times)),
                "Close": np.linspace(100.5, 105.5, len(self.times)),
                "Volume": np.full(len(self.times), 1000.0),
                "Close_Adj": np.linspace(100.5, 105.5, len(self.times)),
                "Open_Adj": np.linspace(100.0, 105.0, len(self.times)),
                "High_Adj": np.linspace(101.0, 106.0, len(self.times)),
                "Low_Adj": np.linspace(99.0, 104.0, len(self.times)),
                "Is_Synthetic": [False] * len(self.times),
                "Is_Spike": [False] * len(self.times),
            },
            index=self.times,
        )

    def tearDown(self) -> None:
        """Cleanup temporary files."""
        self.temp_dir.cleanup()

    def test_schema_validator_valid_ohlcv(self) -> None:
        """Verify compliant OHLCV series passes schema validation."""
        report = LakeSchemaValidator.validate(self.sample_df, tier="silver")
        self.assertTrue(report.is_valid)
        self.assertEqual(len(report.errors), 0)

    def test_schema_validator_catches_nans_and_inversions(self) -> None:
        """Verify schema validator catches NaNs, Infs, and geometric inversions."""
        bad_df = self.sample_df.copy()
        bad_df.iloc[5, bad_df.columns.get_loc("Close")] = np.nan
        bad_df.iloc[10, bad_df.columns.get_loc("High")] = 50.0  # High < Low inversion

        report = LakeSchemaValidator.validate(bad_df, tier="silver")
        self.assertFalse(report.is_valid)
        self.assertTrue(any("NaN" in e for e in report.errors))
        self.assertTrue(any("High < Low" in e for e in report.errors))

    def test_incremental_upsert_deduplication_and_merging(self) -> None:
        """Verify upsert merges overlapping bars without duplicates and updates values."""
        target_path = self.lake_root / "m4_cleaned" / "intraday_5m" / "TEST_clean.parquet"
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Initial write of 50 bars (0 to 49)
        initial_df = self.sample_df.iloc[:50].copy()
        self.upsert_engine.upsert(target_path, initial_df)
        self.assertEqual(len(pd.read_parquet(target_path)), 50)

        # 2. Incoming update: bars 40 to 60 (10 overlapping bars with updated Close, 11 new bars)
        incoming_df = self.sample_df.iloc[40:61].copy()
        incoming_df["Close"] = 999.0  # Mark modified value

        stats = self.upsert_engine.upsert(target_path, incoming_df)
        self.assertEqual(stats["initial_bars"], 50)
        self.assertEqual(stats["incoming_bars"], 21)
        self.assertEqual(stats["final_bars"], 61)  # 50 + 11 new bars = 61 total

        # Verify overlapping bar 45 carries updated value
        reloaded = pd.read_parquet(target_path)
        self.assertEqual(reloaded.iloc[45]["Close"], 999.0)
        self.assertEqual(len(reloaded), 61)
        self.assertTrue(reloaded.index.is_monotonic_increasing)

    def test_lake_manager_read_and_column_projection(self) -> None:
        """Verify zero-copy columnar projection reads only requested columns."""
        target_path = self.lake_root / "m5_resampled" / "intraday_15m" / "BANKBEES_15m.parquet"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        self.sample_df.to_parquet(target_path)

        # Projection query: only load Close and Volume
        df_proj = self.manager.read(
            tier="gold",
            timeframe="15m",
            ticker="BANKBEES",
            columns=["Close", "Volume"],
        )
        self.assertEqual(list(df_proj.columns), ["Close", "Volume"])
        self.assertEqual(len(df_proj), len(self.sample_df))

    def test_lake_manager_time_slice_query(self) -> None:
        """Verify point-in-time window filtering returns only bars within bounds."""
        target_path = self.lake_root / "m4_cleaned" / "intraday_5m" / "BANKBEES_clean.parquet"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        self.sample_df.to_parquet(target_path)

        start_ts = self.times[10]
        end_ts = self.times[20]

        df_slice = self.manager.read(
            tier="silver",
            timeframe="5m",
            ticker="BANKBEES",
            start_time=start_ts,
            end_time=end_ts,
        )
        self.assertEqual(len(df_slice), 11)  # bars 10 to 20 inclusive
        self.assertEqual(df_slice.index[0], start_ts)
        self.assertEqual(df_slice.index[-1], end_ts)

    def test_lake_manager_audit_all_partitions(self) -> None:
        """Verify full lake integrity audit checks all tiers."""
        p_silver = self.lake_root / "m4_cleaned" / "intraday_5m" / "BANKBEES_clean.parquet"
        p_silver.parent.mkdir(parents=True, exist_ok=True)
        self.sample_df.to_parquet(p_silver)

        audit = self.manager.audit_lake()
        self.assertTrue(audit["all_passed"])
        self.assertIn("m4_cleaned/intraday_5m/BANKBEES_clean.parquet", audit["partitions"])
        self.assertEqual(audit["partitions"]["m4_cleaned/intraday_5m/BANKBEES_clean.parquet"]["bars"], len(self.sample_df))


if __name__ == "__main__":
    unittest.main()
