"""Configuration specifications for market data ingestion."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class IngestionConfig:
    """Immutable configuration contract for market data ingestion."""

    ticker: str
    interval: str = "1d"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    period: Optional[str] = None
    timezone: str = "Asia/Kolkata"
    market_open_time: str = "09:15:00"
    market_close_time: str = "15:30:00"
    allow_special_sessions: bool = True
    special_session_dates: List[str] = field(default_factory=list)

    def is_intraday(self) -> bool:
        """Return True if the interval represents intraday minutes/hours."""
        return any(unit in self.interval.lower() for unit in ["m", "h", "s"])
