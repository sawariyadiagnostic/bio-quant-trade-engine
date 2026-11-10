"""Immutable contract for cleaned and regularized datasets (Muscle 4 -> Muscle 5 & 6)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import pandas as pd

from ravana.data.ingestion.config import ExchangeType


@dataclass(frozen=True)
class CleanedDataset:
    """Immutable, strongly-typed dataset contract output by Muscle 4 Data Quality Engine.
    
    Contains time-grid aligned primary asset data, synthetic bar flags, spike sanitization,
    underlying benchmark index data, tracking spread analytics, and quality audit metrics.
    """

    ticker: str
    interval: str
    exchange: ExchangeType
    data: pd.DataFrame
    benchmark_ticker: Optional[str] = None
    benchmark_data: Optional[pd.DataFrame] = None
    spread_data: Optional[pd.DataFrame] = None
    macro_data: Dict[str, pd.DataFrame] = field(default_factory=dict)
    quality_report: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def bar_count(self) -> int:
        """Return total number of bars in cleaned primary series."""
        return len(self.data)

    @property
    def synthetic_bar_count(self) -> int:
        """Return number of forward-filled synthetic gap bars."""
        if "Is_Synthetic" in self.data.columns:
            return int(self.data["Is_Synthetic"].sum())
        return 0

    @property
    def spike_count(self) -> int:
        """Return number of detected and clamped outlier price spikes."""
        if "Is_Spike" in self.data.columns:
            return int(self.data["Is_Spike"].sum())
        return 0

    @property
    def start_time(self) -> Optional[pd.Timestamp]:
        """Return earliest timestamp in the cleaned series."""
        return self.data.index[0] if not self.data.empty else None

    @property
    def end_time(self) -> Optional[pd.Timestamp]:
        """Return latest timestamp in the cleaned series."""
        return self.data.index[-1] if not self.data.empty else None

    def summary(self) -> Dict[str, Any]:
        """Return diagnostic metrics for dataset health and microstructure alignment."""
        zero_vol = (self.data["Volume"] == 0).sum() if "Volume" in self.data.columns else 0
        return {
            "ticker": self.ticker,
            "interval": self.interval,
            "exchange": self.exchange.value if hasattr(self.exchange, "value") else str(self.exchange),
            "bars": self.bar_count,
            "synthetic_bars": self.synthetic_bar_count,
            "spikes_detected": self.spike_count,
            "zero_volume_bars": int(zero_vol),
            "start": str(self.start_time),
            "end": str(self.end_time),
            "has_benchmark": self.benchmark_data is not None and not self.benchmark_data.empty,
            "has_spread": self.spread_data is not None and not self.spread_data.empty,
            "macro_assets": list(self.macro_data.keys()),
        }
