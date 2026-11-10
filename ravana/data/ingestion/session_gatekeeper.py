"""NSE Market Session Gatekeeper.
Enforces institutional market hours, handles special weekend trading sessions
(Union Budget, Diwali Muhurat, DR failovers), and purges post-market synthetic ticks.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple
import pandas as pd

# Registry of official NSE special sessions outside normal Monday-Friday 09:15-15:30
KNOWN_SPECIAL_SESSIONS: Dict[str, List[Tuple[str, str]]] = {
    # Union Budget Live Saturday Sessions
    "2020-02-01": [("09:15:00", "15:30:00")],
    "2025-02-01": [("09:15:00", "15:30:00")],
    # Diwali Muhurat Evening Sessions
    "2019-10-27": [("18:15:00", "19:15:00")],
    "2020-11-14": [("18:15:00", "19:15:00")],
    "2021-11-04": [("18:15:00", "19:15:00")],
    "2022-10-24": [("18:15:00", "19:15:00")],
    "2023-11-12": [("18:15:00", "19:15:00")],
    "2024-11-01": [("18:00:00", "19:00:00")],
    # Disaster Recovery Saturday Live Split Sessions
    "2024-03-02": [("09:15:00", "10:00:00"), ("11:30:00", "12:30:00")],
    "2024-05-18": [("09:15:00", "10:00:00"), ("11:30:00", "12:30:00")],
    # Special Saturday Trading (Ayodhya Consecration Reschedule)
    "2024-01-20": [("09:15:00", "15:30:00")],
}

KNOWN_NSE_HOLIDAYS: Set[str] = {
    "2024-01-22", "2024-01-26", "2024-03-08", "2024-03-25", "2024-03-29",
    "2024-04-11", "2024-04-17", "2024-05-01", "2024-05-20", "2024-06-17",
    "2024-07-17", "2024-08-15", "2024-10-02", "2024-11-15", "2024-12-25",
    "2025-01-26", "2025-02-26", "2025-03-14", "2025-03-31", "2025-04-10",
    "2025-04-14", "2025-04-18", "2025-05-01", "2025-08-15", "2025-08-27",
    "2025-10-02", "2025-10-21", "2025-10-22", "2025-11-05", "2025-12-25",
    "2026-01-26", "2026-03-03", "2026-03-20", "2026-04-03", "2026-04-14",
    "2026-05-01", "2026-08-15", "2026-10-02", "2026-10-20", "2026-11-08", "2026-12-25",
}


class NSESessionGatekeeper:
    """Institutional gatekeeper enforcing NSE trading session boundaries."""

    KNOWN_SPECIAL_SESSIONS = KNOWN_SPECIAL_SESSIONS
    KNOWN_NSE_HOLIDAYS = KNOWN_NSE_HOLIDAYS

    def __init__(
        self,
        market_open: str = "09:15:00",
        market_close: str = "15:30:00",
        target_timezone: str = "Asia/Kolkata",
        label_convention: str = "left",
    ) -> None:
        self.market_open = market_open
        self.market_close = market_close
        self.target_timezone = target_timezone
        self.label_convention = label_convention

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

        # Normalize UTC daily conversion drift (05:30:00 IST -> 00:00:00 IST)
        if len(out) > 0 and (out.index.hour == 5).all() and (out.index.minute == 30).all():
            out.index = out.index.normalize()
        return out

    def get_special_sessions(
        self, custom_whitelist: Optional[List[str]] = None
    ) -> Dict[str, List[Tuple[str, str]]]:
        """Combine built-in registry with user-provided special session dates."""
        sessions = dict(KNOWN_SPECIAL_SESSIONS)
        if custom_whitelist:
            for d in custom_whitelist:
                if d not in sessions:
                    sessions[d] = [(self.market_open, self.market_close)]
        return sessions

    def filter_weekends(
        self, df: pd.DataFrame, special_whitelist: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Purge Saturday and Sunday bars unless registered as a special session."""
        if df.empty:
            return df
        sessions = self.get_special_sessions(special_whitelist)
        date_strs = df.index.strftime("%Y-%m-%d")
        mask = (df.index.dayofweek < 5) | (date_strs.isin(sessions.keys()))
        return df.loc[mask]

    def _filter_intraday_by_date(
        self,
        df: pd.DataFrame,
        sessions: Dict[str, List[Tuple[str, str]]],
        market_open: Optional[str] = None,
        market_close: Optional[str] = None,
    ) -> pd.DataFrame:
        """Filter intraday bars date-by-date adhering to special and regular session windows."""
        cleaned_groups = []
        inclusive_mode = "left" if self.label_convention == "left" else "right"
        m_open = market_open or self.market_open
        m_close = market_close or self.market_close

        for date_val, group in df.groupby(df.index.date):
            date_str = date_val.strftime("%Y-%m-%d")
            if date_str in sessions:
                windows = sessions[date_str]
                slices = [
                    group.between_time(w[0], w[1], inclusive=inclusive_mode) for w in windows
                ]
                if slices:
                    cleaned_groups.append(pd.concat(slices))
                continue

            if date_val.weekday() >= 5:
                continue

            # Standard session: [market_open, market_close) excludes synthetic closing bar
            reg = group.between_time(m_open, m_close, inclusive=inclusive_mode)
            cleaned_groups.append(reg)

        return pd.concat(cleaned_groups).sort_index() if cleaned_groups else pd.DataFrame(columns=df.columns)

    def filter_trading_hours(
        self,
        df: pd.DataFrame,
        is_intraday: bool = True,
        filter_weekends_flag: bool = True,
        special_whitelist: Optional[List[str]] = None,
        target_tz: Optional[str] = None,
        market_open: Optional[str] = None,
        market_close: Optional[str] = None,
    ) -> pd.DataFrame:
        """Enforce market hours, special session profiles, and weekend cleansing."""
        if df.empty:
            return df

        df_std = self.standardize_timezone(df, target_tz=target_tz)
        df_valid_days = self.filter_weekends(df_std, special_whitelist) if filter_weekends_flag else df_std

        if not is_intraday:
            return df_valid_days

        sessions = self.get_special_sessions(special_whitelist)
        return self._filter_intraday_by_date(
            df_valid_days, sessions, market_open=market_open, market_close=market_close
        )



    def is_market_open(self, timestamp: pd.Timestamp) -> bool:
        """Check whether a specific timestamp falls inside official market session."""
        ts = timestamp
        if ts.tz is None:
            ts = ts.tz_localize(self.target_timezone)
        else:
            ts = ts.tz_convert(self.target_timezone)

        date_str = ts.strftime("%Y-%m-%d")
        if date_str in KNOWN_NSE_HOLIDAYS:
            return False

        if date_str in KNOWN_SPECIAL_SESSIONS:
            t = ts.time()
            for open_s, close_s in KNOWN_SPECIAL_SESSIONS[date_str]:
                if pd.Timestamp(open_s).time() <= t < pd.Timestamp(close_s).time():
                    return True
            return False

        if ts.dayofweek >= 5:
            return False

        t = ts.time()
        open_time = pd.Timestamp(self.market_open).time()
        close_time = pd.Timestamp(self.market_close).time()
        return open_time <= t < close_time
