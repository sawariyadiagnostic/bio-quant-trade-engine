"""Unified Lakehouse Manager for multi-tier Parquet storage with zero-copy queries and rollback."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import pyarrow.parquet as pq

from ravana.data.storage.config import StorageConfig
from ravana.data.storage.schema import LakeSchemaValidator, SchemaValidationError, SchemaValidationReport, SeriesType
from ravana.data.storage.upsert import IncrementalUpsertEngine, SchemaEvolutionError


class PartitionColumnNotFoundError(Exception):
    """Raised when requested columns do not exist in the target partition schema."""

    pass


class LakeManager:
    """Unified interface for reading, writing, auditing, and rolling back Parquet partitions."""

    def __init__(self, config: Optional[StorageConfig] = None) -> None:
        self.config = config or StorageConfig()
        self.upsert_engine = IncrementalUpsertEngine(self.config)
        self.validator = LakeSchemaValidator()

    def _normalize_cutoff(self, cutoff: Any, target_tz: Optional[Any]) -> pd.Timestamp:
        """Normalize string/date/timestamp into timezone-aligned pd.Timestamp."""
        ts = pd.Timestamp(cutoff)
        if target_tz is not None:
            if ts.tz is None:
                return ts.tz_localize(target_tz)
            return ts.tz_convert(target_tz)
        if ts.tz is not None:
            return ts.tz_localize(None)
        return ts

    def read(
        self,
        tier: str,
        timeframe: str,
        ticker: str,
        columns: Optional[List[str]] = None,
        start_time: Optional[Any] = None,
        end_time: Optional[Any] = None,
        closed_only: bool = False,
    ) -> pd.DataFrame:
        """Read partition with column projection, timezone normalization, and causal boundary guards."""
        target_path = self.config.resolve_path(tier, timeframe, ticker)
        if not target_path.exists():
            raise FileNotFoundError(f"Partition not found: {target_path}")

        if columns is not None:
            schema = pq.read_schema(target_path)
            available = set(schema.names)
            invalid = [c for c in columns if c not in available]
            if invalid:
                raise PartitionColumnNotFoundError(
                    f"Requested column(s) {invalid} not found in {target_path}. Available columns: {sorted(available)}"
                )

        df = pd.read_parquet(target_path, columns=columns)
        if not isinstance(df.index, pd.DatetimeIndex) or (start_time is None and end_time is None):
            return df

        target_tz = df.index.tz
        norm_start = self._normalize_cutoff(start_time, target_tz) if start_time is not None else None
        norm_end = self._normalize_cutoff(end_time, target_tz) if end_time is not None else None

        if norm_start is not None and norm_end is not None:
            sliced = df.loc[norm_start:norm_end]
            if closed_only and len(sliced) > 0 and sliced.index[-1] == norm_end:
                return sliced.iloc[:-1]
            return sliced
        if norm_start is not None:
            return df.loc[norm_start:]
        if norm_end is not None:
            sliced = df.loc[:norm_end]
            if closed_only and len(sliced) > 0 and sliced.index[-1] == norm_end:
                return sliced.iloc[:-1]
            return sliced
        return df

    def write(
        self,
        tier: str,
        timeframe: str,
        ticker: str,
        df: pd.DataFrame,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """Validate and incrementally upsert DataFrame into specified partition with post-merge verification."""
        if validate:
            report = self.validator.validate(df, tier=tier, ticker=ticker)
            if not report.is_valid:
                raise SchemaValidationError(f"Validation failed for {ticker} ({timeframe}): {report.errors}")

        target_path = self.config.resolve_path(tier, timeframe, ticker)
        stats = self.upsert_engine.upsert(target_path, df)

        if validate:
            reloaded = pd.read_parquet(target_path)
            post_report = self.validator.validate(reloaded, tier=tier, ticker=ticker)
            if not post_report.is_valid:
                self.rollback(tier, timeframe, ticker)
                raise SchemaValidationError(f"Post-upsert integrity violated for {ticker}: {post_report.errors}")

        return stats

    def rollback(self, tier: str, timeframe: str, ticker: str) -> bool:
        """Roll back partition to its latest pre-upsert snapshot."""
        target_path = self.config.resolve_path(tier, timeframe, ticker)
        snapshot_dir = target_path.parent / ".snapshots"
        if not snapshot_dir.exists():
            return False

        pattern = f"{target_path.stem}_*.parquet"
        snaps = sorted(snapshot_dir.glob(pattern), key=os.path.getmtime)
        if not snaps:
            return False

        latest_snap = snaps[-1]
        shutil.copy2(latest_snap, target_path)
        latest_snap.unlink(missing_ok=True)
        return True

    def list_tiers(self) -> List[str]:
        """List available medallion tiers on disk."""
        if not self.config.lake_root.exists():
            return []
        return sorted([d.name for d in self.config.lake_root.iterdir() if d.is_dir() and not d.name.startswith(".")])

    def list_timeframes(self, tier: str) -> List[str]:
        """List available timeframes within a medallion tier."""
        tier_dir = self.config.get_tier_dir(tier)
        if not tier_dir.exists():
            return []
        return sorted([d.name for d in tier_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])

    def get_tier_catalog(self, tier: str) -> Dict[str, Any]:
        """Catalog all partitions and coverage in a tier via Parquet footers without loading data."""
        tier_dir = self.config.get_tier_dir(tier)
        if not tier_dir.exists():
            return {}

        catalog: Dict[str, Any] = {}
        for p in sorted(tier_dir.glob("**/*.parquet")):
            if ".tmp." in p.name or ".snapshots" in str(p):
                continue
            rel_path = p.relative_to(self.config.lake_root).as_posix()
            try:
                pf = pq.ParquetFile(p)
                meta = pf.metadata
                catalog[rel_path] = {
                    "rows": meta.num_rows,
                    "columns": pf.schema.names,
                    "row_groups": meta.num_row_groups,
                    "size_bytes": p.stat().st_size,
                }
            except Exception as e:
                catalog[rel_path] = {"error": str(e)}
        return catalog

    def audit_lake(self) -> Dict[str, Any]:
        """Perform comprehensive schema and contract audit across all lakehouse partitions."""
        partitions: Dict[str, Dict[str, Any]] = {}
        if not self.config.lake_root.exists():
            return {"all_passed": True, "total_partitions": 0, "partitions": {}}

        parquet_files = sorted(self.config.lake_root.glob("**/*.parquet"))
        for file_path in parquet_files:
            if ".tmp." in file_path.name or ".snapshots" in str(file_path):
                continue

            rel_posix = file_path.relative_to(self.config.lake_root).as_posix()
            tier_name = file_path.relative_to(self.config.lake_root).parts[0]
            ticker_name = file_path.stem.split("_")[0]

            try:
                df = pd.read_parquet(file_path)
                report: SchemaValidationReport = self.validator.validate(df, tier=tier_name, ticker=ticker_name)
                partitions[rel_posix] = {
                    "bars": len(df),
                    "series_type": report.series_type.value,
                    "valid": report.is_valid,
                    "errors": report.errors,
                    "warnings": report.warnings,
                    "columns": list(df.columns),
                }
            except Exception as e:
                partitions[rel_posix] = {
                    "bars": 0,
                    "series_type": "unknown",
                    "valid": False,
                    "errors": [f"Failed to read parquet: {str(e)}"],
                    "warnings": [],
                    "columns": [],
                }

        all_passed = all(p["valid"] for p in partitions.values()) if partitions else True
        return {
            "all_passed": all_passed,
            "total_partitions": len(partitions),
            "partitions": partitions,
        }
