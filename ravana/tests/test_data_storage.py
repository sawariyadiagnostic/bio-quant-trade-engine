"""Unit tests for Muscle 6: Sovereign Columnar Parquet Data Lake & LakeManager.
Verifies schema validation, contract dispatches, incremental upsert deduplication,
atomic writes, rollback, zero-copy column projections, time-range queries, and multi-tier routing.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from ravana.data.storage.config import StorageConfig
from ravana.data.storage.manager import LakeManager, PartitionColumnNotFoundError
from ravana.data.storage.schema import (
    LakeSchemaValidator,
    SchemaValidationError,
    SeriesType,
)
from ravana.data.storage.upsert import IncrementalUpsertEngine, SchemaEvolutionError


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
        self.assertEqual(report.series_type, SeriesType.OHLCV)
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

    def test_schema_validator_catches_negative_prices_and_duplicates(self) -> None:
        """Verify schema validator rejects negative prices and duplicate timestamps."""
        neg_df = self.sample_df.copy()
        neg_df.iloc[2, neg_df.columns.get_loc("Low")] = -5.0
        report_neg = LakeSchemaValidator.validate(neg_df, tier="silver")
        self.assertFalse(report_neg.is_valid)
        self.assertTrue(any("non-positive" in e for e in report_neg.errors))

        dup_idx = self.times.copy()
        dup_times = dup_idx[:5].append(dup_idx[:1])  # Duplicate timestamp
        dup_df = pd.DataFrame(np.ones((6, 5)), index=dup_times, columns=["Open", "High", "Low", "Close", "Volume"])
        report_dup = LakeSchemaValidator.validate(dup_df, tier="silver")
        self.assertFalse(report_dup.is_valid)
        self.assertTrue(any("duplicate" in e for e in report_dup.errors))

    def test_schema_validator_spread_analytics_contract(self) -> None:
        """Verify ETF spread analytics contract enforces mathematical relations."""
        etf_close = np.full(10, 500.0)
        idx_close = np.full(10, 50000.0)
        multiplier = np.full(10, 100.0)
        theo_close = idx_close / multiplier  # 500.0
        spread_inr = etf_close - theo_close  # 0.0
        basis_bps = (spread_inr / theo_close) * 10000.0

        spread_df = pd.DataFrame(
            {
                "ETF_Close": etf_close,
                "Index_Close": idx_close,
                "Multiplier": multiplier,
                "Theoretical_Close": theo_close,
                "Spread_INR": spread_inr,
                "Basis_Bps": basis_bps,
                "Rolling_Correlation": np.full(10, 0.99),
            },
            index=self.times[:10],
        )
        report = LakeSchemaValidator.validate(spread_df, tier="silver", ticker="BANKBEES_spread")
        self.assertTrue(report.is_valid)
        self.assertEqual(report.series_type, SeriesType.ETF_SPREAD)

    def test_incremental_upsert_deduplication_and_merging(self) -> None:
        """Verify upsert merges overlapping bars without duplicates and updates values."""
        target_path = self.lake_root / "m4_cleaned" / "intraday_5m" / "TEST_clean.parquet"
        target_path.parent.mkdir(parents=True, exist_ok=True)

        initial_df = self.sample_df.iloc[:50].copy()
        self.upsert_engine.upsert(target_path, initial_df)
        self.assertEqual(len(pd.read_parquet(target_path)), 50)

        incoming_df = self.sample_df.iloc[40:61].copy()
        incoming_df["Close"] = 999.0

        stats = self.upsert_engine.upsert(target_path, incoming_df)
        self.assertEqual(stats["initial_bars"], 50)
        self.assertEqual(stats["incoming_bars"], 21)
        self.assertEqual(stats["final_bars"], 61)

        reloaded = pd.read_parquet(target_path)
        self.assertEqual(reloaded.iloc[45]["Close"], 999.0)
        self.assertEqual(len(reloaded), 61)
        self.assertTrue(reloaded.index.is_monotonic_increasing)

    def test_upsert_schema_drift_rejection(self) -> None:
        """Verify upsert rejects incoming batches with mismatched columns to prevent NaN injection."""
        target_path = self.lake_root / "m4_cleaned" / "intraday_5m" / "DRIFT_clean.parquet"
        target_path.parent.mkdir(parents=True, exist_ok=True)

        self.upsert_engine.upsert(target_path, self.sample_df)

        bad_incoming = self.sample_df.iloc[:10][["Open", "High", "Low", "Close"]].copy()
        with self.assertRaises(SchemaEvolutionError):
            self.upsert_engine.upsert(target_path, bad_incoming)

    def test_lake_manager_read_and_column_projection(self) -> None:
        """Verify zero-copy columnar projection reads only requested columns."""
        target_path = self.lake_root / "m5_resampled" / "intraday_15m" / "BANKBEES_15m.parquet"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        self.sample_df.to_parquet(target_path)

        df_proj = self.manager.read(
            tier="gold",
            timeframe="15m",
            ticker="BANKBEES",
            columns=["Close", "Volume"],
        )
        self.assertEqual(list(df_proj.columns), ["Close", "Volume"])
        self.assertEqual(len(df_proj), len(self.sample_df))

        with self.assertRaises(PartitionColumnNotFoundError):
            self.manager.read(
                tier="gold",
                timeframe="15m",
                ticker="BANKBEES",
                columns=["NonExistentColumn"],
            )

    def test_lake_manager_time_slice_query_and_timezone_coercion(self) -> None:
        """Verify point-in-time window filtering works with naive timestamps via auto-coercion."""
        target_path = self.lake_root / "m4_cleaned" / "intraday_5m" / "BANKBEES_clean.parquet"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        self.sample_df.to_parquet(target_path)

        df_slice = self.manager.read(
            tier="silver",
            timeframe="5m",
            ticker="BANKBEES",
            start_time="2026-03-09 10:00:00",
            end_time="2026-03-09 11:00:00",
        )
        self.assertEqual(len(df_slice), 13)
        self.assertEqual(str(df_slice.index[0]), "2026-03-09 10:00:00+05:30")
        self.assertEqual(str(df_slice.index[-1]), "2026-03-09 11:00:00+05:30")

    def test_lake_manager_rollback_and_catalog(self) -> None:
        """Verify partition snapshot retention, rollback, and zero-scan footer catalog."""
        target_path = self.lake_root / "m4_cleaned" / "intraday_5m" / "BANKBEES_clean.parquet"
        target_path.parent.mkdir(parents=True, exist_ok=True)

        self.manager.write(tier="silver", timeframe="5m", ticker="BANKBEES", df=self.sample_df)

        catalog = self.manager.get_tier_catalog(tier="silver")
        self.assertIn("m4_cleaned/intraday_5m/BANKBEES_clean.parquet", catalog)
        self.assertEqual(catalog["m4_cleaned/intraday_5m/BANKBEES_clean.parquet"]["rows"], len(self.sample_df))

        corrupt_batch = self.sample_df.copy()
        corrupt_batch["Open"] = 103.0
        corrupt_batch["High"] = 104.0
        corrupt_batch["Low"] = 102.0
        corrupt_batch["Close"] = 103.5
        self.manager.write(tier="silver", timeframe="5m", ticker="BANKBEES", df=corrupt_batch)
        self.assertEqual(pd.read_parquet(target_path).iloc[0]["Close"], 103.5)

        rolled_back = self.manager.rollback(tier="silver", timeframe="5m", ticker="BANKBEES")
        self.assertTrue(rolled_back)
        self.assertAlmostEqual(pd.read_parquet(target_path).iloc[0]["Close"], 100.5, places=2)

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
