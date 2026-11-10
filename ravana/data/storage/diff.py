"""Lakehouse Partition Diff Inspector.

Compares active Lakehouse Parquet partitions against historical snapshots captured in .snapshots/,
auditing schema migrations, row counts, and numerical distribution shifts across pipeline runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ColumnDiff:
    """Statistical summary shift for an individual column."""
    name: str
    old_mean: Optional[float] = None
    new_mean: Optional[float] = None
    old_std: Optional[float] = None
    new_std: Optional[float] = None
    old_zeros: int = 0
    new_zeros: int = 0
    old_nans: int = 0
    new_nans: int = 0
    is_dead_old: bool = False
    is_dead_new: bool = False


@dataclass
class LakePartitionDiff:
    """Comprehensive diff between active Parquet partition and a historical snapshot."""
    target_path: Path
    snapshot_path: Optional[Path]
    old_rows: int = 0
    new_rows: int = 0
    columns_added: List[str] = field(default_factory=list)
    columns_removed: List[str] = field(default_factory=list)
    columns_retained: List[str] = field(default_factory=list)
    column_stats: Dict[str, ColumnDiff] = field(default_factory=dict)

    def summary_table(self) -> str:
        """Formats the diff into a readable terminal summary table."""
        lines = [
            f"=== Lakehouse Partition Diff ===",
            f"Active:   {self.target_path}",
            f"Snapshot: {self.snapshot_path if self.snapshot_path else 'None (First Run)'}",
            f"Rows:     Old={self.old_rows} | New={self.new_rows} (Delta={self.new_rows - self.old_rows:+d})",
            f"Columns Added:   {self.columns_added if self.columns_added else 'None'}",
            f"Columns Removed: {self.columns_removed if self.columns_removed else 'None'}",
            "",
            f"{'Column':<24} | {'Old Mean':<10} | {'New Mean':<10} | {'Old Zeros':<9} | {'New Zeros':<9} | {'Status'}",
            "-" * 80,
        ]
        for col, stat in sorted(self.column_stats.items()):
            old_m = f"{stat.old_mean:10.4f}" if stat.old_mean is not None else f"{'N/A':<10}"
            new_m = f"{stat.new_mean:10.4f}" if stat.new_mean is not None else f"{'N/A':<10}"
            status = "REVIVED" if (stat.is_dead_old and not stat.is_dead_new) else ("DEAD" if stat.is_dead_new else "OK")
            lines.append(
                f"{col:<24} | {old_m} | {new_m} | {stat.old_zeros:<9} | {stat.new_zeros:<9} | {status}"
            )
        return "\n".join(lines)


class LakeDiffInspector:
    """Audits differences between active partitions and their historical snapshots."""

    @staticmethod
    def get_latest_snapshot(target_path: Path | str) -> Optional[Path]:
        """Locate the most recent historical snapshot for a parquet partition."""
        p = Path(target_path)
        snap_dir = p.parent / ".snapshots"
        if not snap_dir.exists():
            return None
        pattern = f"{p.stem}_*.parquet"
        snaps = sorted(snap_dir.glob(pattern), key=os.path.getmtime)
        return snaps[-1] if snaps else None

    @classmethod
    def compare_partition(
        cls,
        target_path: Path | str,
        snapshot_path: Optional[Path | str] = None,
    ) -> LakePartitionDiff:
        """Compare an active lake partition against a snapshot."""
        active_p = Path(target_path)
        if not active_p.exists():
            raise FileNotFoundError(f"Active partition not found: {active_p}")

        snap_p = Path(snapshot_path) if snapshot_path else cls.get_latest_snapshot(active_p)
        df_new = pd.read_parquet(active_p)
        if snap_p is None or not snap_p.exists():
            return LakePartitionDiff(
                target_path=active_p,
                snapshot_path=None,
                new_rows=len(df_new),
                columns_added=list(df_new.columns),
            )

        df_old = pd.read_parquet(snap_p)
        return cls._compute_diff(active_p, snap_p, df_old, df_new)

    @staticmethod
    def _compute_diff(
        active_p: Path, snap_p: Path, df_old: pd.DataFrame, df_new: pd.DataFrame
    ) -> LakePartitionDiff:
        """Computes statistical metrics between old and new DataFrames."""
        old_cols = set(df_old.columns)
        new_cols = set(df_new.columns)
        added = sorted(new_cols - old_cols)
        removed = sorted(old_cols - new_cols)
        retained = sorted(old_cols & new_cols)

        diff = LakePartitionDiff(
            target_path=active_p,
            snapshot_path=snap_p,
            old_rows=len(df_old),
            new_rows=len(df_new),
            columns_added=added,
            columns_removed=removed,
            columns_retained=retained,
        )

        all_cols = sorted(new_cols | old_cols)
        for col in all_cols:
            diff.column_stats[col] = LakeDiffInspector._compute_col_diff(col, df_old, df_new)

        return diff

    @staticmethod
    def _compute_col_diff(col: str, df_old: pd.DataFrame, df_new: pd.DataFrame) -> ColumnDiff:
        """Calculates mean, std, zero count and dead status for a single column."""
        s_old = df_old[col] if col in df_old.columns else None
        s_new = df_new[col] if col in df_new.columns else None

        stat = ColumnDiff(name=col)
        if s_old is not None and pd.api.types.is_numeric_dtype(s_old):
            stat.old_mean = float(s_old.mean()) if not s_old.empty else None
            stat.old_std = float(s_old.std()) if not s_old.empty else None
            stat.old_zeros = int((s_old == 0.0).sum())
            stat.old_nans = int(s_old.isna().sum())
            stat.is_dead_old = bool((s_old == 0.0).all() or stat.old_std == 0.0)

        if s_new is not None and pd.api.types.is_numeric_dtype(s_new):
            stat.new_mean = float(s_new.mean()) if not s_new.empty else None
            stat.new_std = float(s_new.std()) if not s_new.empty else None
            stat.new_zeros = int((s_new == 0.0).sum())
            stat.new_nans = int(s_new.isna().sum())
            stat.is_dead_new = bool((s_new == 0.0).all() or stat.new_std == 0.0)

        return stat


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "ravana/data/lake/m7_features/intraday_5m/BANKBEES_features.parquet"
    res = LakeDiffInspector.compare_partition(path)
    print(res.summary_table())
