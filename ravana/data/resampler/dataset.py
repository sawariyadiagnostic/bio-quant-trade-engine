"""Immutable multi-timeframe dataset contract (Muscle 5 -> downstream)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd

from ravana.data.ingestion.config import ExchangeType


@dataclass(frozen=True)
class ResampledDataset:
    """Immutable multi-timeframe dataset output by Muscle 5 Resampler.
    
    Contains time-aligned, right-labeled OHLCV series across multiple derived
    timeframes (e.g. 5m, 15m, 30m, 1h, 1wk, 1mo) for primary assets, benchmarks,
    and ETF-to-benchmark tracking spread series.
    """

    ticker: str
    base_interval: str
    exchange: ExchangeType
    timeframes: Dict[str, pd.DataFrame]
    benchmark_ticker: Optional[str] = None
    benchmark_timeframes: Dict[str, pd.DataFrame] = field(default_factory=dict)
    spread_timeframes: Dict[str, pd.DataFrame] = field(default_factory=dict)
    macro_timeframes: Dict[str, Dict[str, pd.DataFrame]] = field(default_factory=dict)
    quality_report: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def available_timeframes(self) -> List[str]:
        """Return list of generated timeframes in primary series."""
        return list(self.timeframes.keys())

    @property
    def bar_counts(self) -> Dict[str, int]:
        """Return mapping of timeframe to total bar count."""
        return {tf: len(df) for tf, df in self.timeframes.items()}

    def get_timeframe(self, interval: str) -> pd.DataFrame:
        """Retrieve primary series DataFrame for a specific timeframe."""
        if interval not in self.timeframes:
            raise KeyError(f"Timeframe '{interval}' not found. Available: {self.available_timeframes}")
        return self.timeframes[interval]

    def get_benchmark_timeframe(self, interval: str) -> Optional[pd.DataFrame]:
        """Retrieve benchmark series DataFrame for a specific timeframe."""
        return self.benchmark_timeframes.get(interval)

    def get_spread_timeframe(self, interval: str) -> Optional[pd.DataFrame]:
        """Retrieve spread series DataFrame for a specific timeframe."""
        return self.spread_timeframes.get(interval)
