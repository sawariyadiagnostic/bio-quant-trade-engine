"""NSE Market Session Gatekeeper.
Enforces strict regulatory market hours (09:15 to 15:30 IST) and calendar filtering.
"""
from __future__ import annotations

from typing import List, Optional, Set
import pandas as pd


class NSESessionGatekeeper:
    """Institutional gatekeeper enforcing NSE trading session boundaries."""

    def __init__(
        self,
        market_open: str = "09:15:00",
        market_close: str = "15:30:00",
        target_timezone: str = "Asia/Kolkata",
    ) -> None:
        self.market_open = market_open
        self.market_close = market_close
        self.target_timezone = target_timezone

    def standardize_timezone(
        self, df: pd.DataFrame, target_tz: Optional[str] = None
    ) -> pd.DataFrame:
        """Convert or localize DataFrame DatetimeIndex to target exchange timezone."""
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)

        tz = target_tz or self.target_timezone
        out = df.copy()
        if out.index.tz is None:
            out.index = out.index.tz_localize(tz)
        else:
            out.index = out.index.tz_convert(tz)
        return out

    def filter_weekends(
        self, df: pd.DataFrame, special_whitelist: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Remove Saturday (5) and Sunday (6) bars unless whitelisted (e.g. Muhurat)."""
        if df.empty:
            return df
        whitelist: Set[str] = set(special_whitelist or [])
        mask = (df.index.dayofweek < 5) | (df.index.strftime("%Y-%m-%d").isin(whitelist))
        return df.loc[mask]

    def filter_trading_hours(
        self,
        df: pd.DataFrame,
        is_intraday: bool = True,
        special_whitelist: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Enforce 09:15 to 15:30 IST market hours and purge weekend bars."""
        if df.empty:
            return df

        df_std = self.standardize_timezone(df)
        df_weekdays = self.filter_weekends(df_std, special_whitelist)

        if not is_intraday:
            return df_weekdays

        # Strict intraday market hours: 09:15:00 to 15:30:00 IST
        return df_weekdays.between_time(
            self.market_open, self.market_close, inclusive="both"
        )

    def is_market_open(self, timestamp: pd.Timestamp) -> bool:
        """Check whether a specific timestamp falls inside official market session."""
        ts = timestamp
        if ts.tz is None:
            ts = ts.tz_localize(self.target_timezone)
        else:
            ts = ts.tz_convert(self.target_timezone)

        if ts.dayofweek >= 5:
            return False

        t = ts.time()
        open_time = pd.Timestamp(self.market_open).time()
        close_time = pd.Timestamp(self.market_close).time()
        return open_time <= t <= close_time
