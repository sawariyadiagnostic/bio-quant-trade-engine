"""Institutional IngestedDataset contract connecting Muscle 3 to downstream muscles."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd

from ravana.data.ingestion.config import ExchangeType


@dataclass(frozen=True)
class IngestedDataset:
    """Immutable, strongly-typed market dataset contract connecting Muscle 3 to Muscle 4.
    
    Bundles the primary traded asset, its underlying benchmark index, macro drivers,
    and telemetry metadata into a single unified container.
    """

    ticker: str
    interval: str
    exchange: ExchangeType
    data: pd.DataFrame
    benchmark_ticker: Optional[str] = None
    benchmark_data: Optional[pd.DataFrame] = None
    macro_data: Dict[str, pd.DataFrame] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def bar_count(self) -> int:
        """Return the number of bars in the primary series."""
        return len(self.data)

    @property
    def start_time(self) -> Optional[pd.Timestamp]:
        """Return the earliest timestamp in the dataset."""
        return self.data.index[0] if not self.data.empty else None

    @property
    def end_time(self) -> Optional[pd.Timestamp]:
        """Return the latest timestamp in the dataset."""
        return self.data.index[-1] if not self.data.empty else None

    def summary(self) -> Dict[str, Any]:
        """Return a diagnostic dictionary of dataset metrics."""
        zero_vol = (self.data["Volume"] == 0).sum() if "Volume" in self.data.columns else 0
        return {
            "ticker": self.ticker,
            "interval": self.interval,
            "exchange": self.exchange.value if hasattr(self.exchange, "value") else str(self.exchange),
            "bars": self.bar_count,
            "start": str(self.start_time),
            "end": str(self.end_time),
            "zero_volume_bars": int(zero_vol),
            "has_benchmark": self.benchmark_data is not None and not self.benchmark_data.empty,
            "macro_assets": list(self.macro_data.keys()),
        }
