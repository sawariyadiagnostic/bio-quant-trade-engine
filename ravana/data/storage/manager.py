"""Unified Lakehouse Manager for multi-tier Parquet storage."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from ravana.data.storage.config import StorageConfig
from ravana.data.storage.schema import LakeSchemaValidator, SchemaValidationError, SchemaValidationReport
from ravana.data.storage.upsert import IncrementalUpsertEngine


class LakeManager:
    """Unified interface for reading, writing, and auditing columnar market data partitions."""

    def __init__(self, config: Optional[StorageConfig] = None) -> None:
        self.config = config or StorageConfig()
        self.upsert_engine = IncrementalUpsertEngine(self.config)
        self.validator = LakeSchemaValidator()

    def read(
        self,
        tier: str,
        timeframe: str,
        ticker: str,
        columns: Optional[List[str]] = None,
        start_time: Optional[pd.Timestamp] = None,
        end_time: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Read partition with zero-copy column projection and point-in-time window filtering."""
        target_path = self.config.resolve_path(tier, timeframe, ticker)
        if not target_path.exists():
            raise FileNotFoundError(f"Partition not found: {target_path}")

        df = pd.read_parquet(target_path, columns=columns)

        if start_time is not None and end_time is not None:
            return df.loc[start_time:end_time]
        if start_time is not None:
            return df.loc[start_time:]
        if end_time is not None:
            return df.loc[:end_time]
        return df

    def write(
        self,
        tier: str,
        timeframe: str,
        ticker: str,
        df: pd.DataFrame,
        validate: bool = True,
    ) -> Dict[str, int]:
        """Validate and incrementally upsert DataFrame into specified partition."""
        if validate:
            report = self.validator.validate(df, tier=tier)
            if not report.is_valid:
                raise SchemaValidationError(f"Validation failed for {ticker} ({timeframe}): {report.errors}")

        target_path = self.config.resolve_path(tier, timeframe, ticker)
        return self.upsert_engine.upsert(target_path, df)

    def audit_lake(self) -> Dict[str, Any]:
        """Perform comprehensive schema and invariant audit across all lakehouse partitions."""
        partitions: Dict[str, Dict[str, Any]] = {}
        if not self.config.lake_root.exists():
            return {"all_passed": True, "total_partitions": 0, "partitions": {}}

        parquet_files = sorted(self.config.lake_root.glob("**/*.parquet"))
        for file_path in parquet_files:
            if ".tmp." in file_path.name:
                continue

            rel_posix = file_path.relative_to(self.config.lake_root).as_posix()
            tier_name = file_path.relative_to(self.config.lake_root).parts[0]

            try:
                df = pd.read_parquet(file_path)
                report: SchemaValidationReport = self.validator.validate(df, tier=tier_name)
                partitions[rel_posix] = {
                    "bars": len(df),
                    "valid": report.is_valid,
                    "errors": report.errors,
                    "warnings": report.warnings,
                    "columns": list(df.columns),
                }
            except Exception as e:
                partitions[rel_posix] = {
                    "bars": 0,
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
