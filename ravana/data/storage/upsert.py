"""Incremental upsert engine with schema drift protection, snapshots, and Win32 atomic swap."""
from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from ravana.data.storage.config import StorageConfig

logger = logging.getLogger(__name__)


class SchemaEvolutionError(Exception):
    """Raised when incoming dataset columns diverge from existing partition schema."""

    pass


class IncrementalUpsertEngine:
    """Merges new market bars into existing partitions with atomic snapshotting and drift guards."""

    def __init__(
        self,
        config: Optional[StorageConfig] = None,
        enable_snapshots: bool = True,
        max_snapshots: int = 3,
    ) -> None:
        self.config = config or StorageConfig()
        self.enable_snapshots = enable_snapshots
        self.max_snapshots = max_snapshots

    def upsert(
        self,
        target_path: Path | str,
        incoming_df: pd.DataFrame,
        allow_schema_evolution: bool = False,
    ) -> Dict[str, Any]:
        """Perform deterministic atomic upsert with schema guards and snapshot retention."""
        path = Path(target_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if incoming_df.empty:
            existing_count = len(pd.read_parquet(path)) if path.exists() else 0
            return {"initial_bars": existing_count, "incoming_bars": 0, "final_bars": existing_count, "collisions": 0}

        clean_incoming = self._deduplicate_internal(incoming_df)
        initial_bars = 0
        collisions = 0

        if path.exists():
            existing_df = pd.read_parquet(path)
            initial_bars = len(existing_df)
            self._validate_schema_compatibility(existing_df, clean_incoming, allow_schema_evolution)
            clean_incoming = self._harmonize_timezones(existing_df, clean_incoming)

            collisions = len(existing_df.index.intersection(clean_incoming.index))
            combined = pd.concat([existing_df, clean_incoming])
        else:
            combined = clean_incoming.copy()

        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()

        if self.enable_snapshots and path.exists():
            self._create_snapshot(path)

        self._atomic_write_with_retry(path, combined)

        return {
            "initial_bars": initial_bars,
            "incoming_bars": len(incoming_df),
            "final_bars": len(combined),
            "collisions": collisions,
        }

    @staticmethod
    def _deduplicate_internal(df: pd.DataFrame) -> pd.DataFrame:
        """Drop internal batch duplicate timestamps keeping last occurrence."""
        if df.index.duplicated().any():
            return df[~df.index.duplicated(keep="last")]
        return df

    @staticmethod
    def _validate_schema_compatibility(
        existing: pd.DataFrame, incoming: pd.DataFrame, allow_evolution: bool = False
    ) -> None:
        """Enforce strict column schema matching to prevent silent NaN injection."""
        if set(existing.columns) != set(incoming.columns):
            if allow_evolution:
                logger.info(
                    "Schema evolution permitted: existing %s -> incoming %s",
                    sorted(existing.columns),
                    sorted(incoming.columns),
                )
                return
            missing = set(existing.columns) - set(incoming.columns)
            extra = set(incoming.columns) - set(existing.columns)
            raise SchemaEvolutionError(
                f"Schema divergence detected! Existing columns: {sorted(existing.columns)}, "
                f"Incoming missing: {sorted(missing)}, Incoming extra: {sorted(extra)}"
            )

    @staticmethod
    def _harmonize_timezones(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
        """Harmonize incoming index timezone to match existing partition."""
        if existing.index.tz != incoming.index.tz:
            if existing.index.tz is not None and incoming.index.tz is None:
                return incoming.tz_localize(existing.index.tz)
            if existing.index.tz is not None and incoming.index.tz is not None:
                return incoming.tz_convert(existing.index.tz)
            if existing.index.tz is None and incoming.index.tz is not None:
                return incoming.tz_localize(None)
        return incoming

    def _create_snapshot(self, target_path: Path) -> None:
        """Archive existing partition into .snapshots directory prior to replacement."""
        snapshot_dir = target_path.parent / ".snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        ts_tag = int(time.time() * 1000)
        snap_path = snapshot_dir / f"{target_path.stem}_{ts_tag}.parquet"

        try:
            shutil.copy2(target_path, snap_path)
            # Prune older snapshots exceeding retention
            pattern = f"{target_path.stem}_*.parquet"
            all_snaps = sorted(snapshot_dir.glob(pattern), key=os.path.getmtime)
            while len(all_snaps) > self.max_snapshots:
                oldest = all_snaps.pop(0)
                oldest.unlink(missing_ok=True)
        except Exception:
            pass  # Non-fatal snapshot warning

    def _atomic_write_with_retry(self, target_path: Path, df: pd.DataFrame, max_retries: int = 5) -> None:
        """Atomically persist DataFrame with Win32 exponential backoff retry loop."""
        tmp_path = target_path.with_suffix(f".tmp.{os.getpid()}_{int(time.time() * 1000)}")
        try:
            df.to_parquet(tmp_path, compression=self.config.compression)
            for attempt in range(max_retries):
                try:
                    tmp_path.replace(target_path)
                    break
                except PermissionError:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(0.02 * (2**attempt))
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
