"""Configuration specifications and multi-asset universe definitions for market data ingestion."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

# Standard Institutional Universes
CORE_ETF_UNIVERSE: List[str] = ["BANKBEES.NS", "NIFTYBEES.NS"]
GLOBAL_MACRO_UNIVERSE: List[str] = ["^GSPC", "^VIX", "INR=X", "BZ=F", "^TNX"]
DOMESTIC_MACRO_UNIVERSE: List[str] = ["^NSEBANK", "^NSEI", "^INDIAVIX"]


class ExchangeType(str, Enum):
    """Supported exchange market session profiles."""

    NSE = "NSE"
    US = "US"
    GLOBAL_24H = "GLOBAL_24H"


@dataclass(frozen=True)
class IngestionConfig:
    """Immutable configuration contract for market data ingestion."""

    ticker: str
    interval: str = "1d"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    period: Optional[str] = None
    exchange: ExchangeType = ExchangeType.NSE
    timezone: str = "Asia/Kolkata"
    market_open_time: str = "09:15:00"
    market_close_time: str = "15:30:00"
    allow_special_sessions: bool = True
    special_session_dates: List[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        ticker: str,
        interval: str = "1d",
        period: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> IngestionConfig:
        """Factory constructor that automatically deduces exchange, timezone, and market hours."""
        t_upper = ticker.upper()

        if t_upper in ["^GSPC", "^IXIC", "^DJI", "SPY", "QQQ", "^VIX", "^TNX"]:
            return cls(
                ticker=ticker,
                interval=interval,
                period=period,
                start_date=start_date,
                end_date=end_date,
                exchange=ExchangeType.US,
                timezone="America/New_York",
                market_open_time="09:30:00",
                market_close_time="16:00:00",
            )
        elif any(t_upper.startswith(p) for p in ["INR=", "EUR=", "GBP="]) or t_upper.endswith("=F"):
            return cls(
                ticker=ticker,
                interval=interval,
                period=period,
                start_date=start_date,
                end_date=end_date,
                exchange=ExchangeType.GLOBAL_24H,
                timezone="UTC",
                market_open_time="00:00:00",
                market_close_time="23:59:59",
            )
        else:
            return cls(
                ticker=ticker,
                interval=interval,
                period=period,
                start_date=start_date,
                end_date=end_date,
                exchange=ExchangeType.NSE,
                timezone="Asia/Kolkata",
                market_open_time="09:15:00",
                market_close_time="15:30:00",
            )

    def is_intraday(self) -> bool:
        """Return True if the interval represents intraday minutes/hours/seconds.
        
        Strictly excludes Daily ('1d'), Weekly ('1wk'), and Monthly ('1mo', '3mo').
        """
        val = self.interval.strip().lower()
        if any(val.endswith(unit) for unit in ["d", "wk", "mo", "y", "month", "year"]):
            return False
        return any(unit in val for unit in ["m", "h", "s", "min", "sec", "hour"])
