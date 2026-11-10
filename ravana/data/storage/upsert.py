"""Incremental upsert engine with atomic swapping for Parquet Lakehouse."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

from ravana.data.storage.config import StorageConfig


class IncrementalUpsertEngine:
    """Merges new market bars into existing partitions without duplicates."""

    def __init__(self, config: Optional[StorageConfig] = None) -> None:
        self.config = config or StorageConfig()

    def upsert(self, target_path: Path | str, incoming_df: pd.DataFrame) -> Dict[str, int]:
        """Perform atomic incremental upsert merging incoming bars with existing partition."""
        path = Path(target_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if incoming_df.empty:
            existing_count = len(pd.read_parquet(path)) if path.exists() else 0
            return {"initial_bars": existing_count, "incoming_bars": 0, "final_bars": existing_count}

        initial_bars = 0
        if path.exists():
            existing_df = pd.read_parquet(path)
            initial_bars = len(existing_df)
            combined = pd.concat([existing_df, incoming_df])
        else:
            combined = incoming_df.copy()

        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()

        self._atomic_write(path, combined)

        return {
            "initial_bars": initial_bars,
            "incoming_bars": len(incoming_df),
            "final_bars": len(combined),
        }

    def _atomic_write(self, target_path: Path, df: pd.DataFrame) -> None:
        """Atomically persist DataFrame by writing to temporary file then replacing target."""
        tmp_path = target_path.with_suffix(f".tmp.{os.getpid()}_{int(time.time() * 1000)}")
        try:
            df.to_parquet(tmp_path, compression=self.config.compression)
            tmp_path.replace(target_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise
