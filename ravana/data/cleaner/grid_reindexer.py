"""Time grid reindexing and causal intraday gap filling for institutional feeds."""
from __future__ import annotations

import re
from typing import List, Optional, Tuple
import pandas as pd

from ravana.data.ingestion.config import ExchangeType, IngestionConfig
from ravana.data.ingestion.session_gatekeeper import (
    KNOWN_SPECIAL_SESSIONS,
    KNOWN_NSE_HOLIDAYS,
    NSESessionGatekeeper,
)


class TimeGridReindexer:
    """Standardizes intraday series to an exact, continuous time grid with zero lookahead.
    
    Fills missing intraday bars using strict causal forward-filling for prices
    and zero-volume for non-trading bars, marking all synthetic entries with `Is_Synthetic`.
    """

    INTERVAL_FREQ_MAP = {
        "1m": "1min",
        "2m": "2min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "60m": "60min",
        "1h": "60min",
    }

    def __init__(self, exchange: ExchangeType = ExchangeType.NSE) -> None:
        self.exchange = exchange

    def reindex(
        self,
        df: pd.DataFrame,
        interval: str,
    ) -> Tuple[pd.DataFrame, int]:
        """Reindex series to the complete expected session grid.
        
        Args:
            df: OHLCV DataFrame with DatetimeIndex.
            interval: Bar duration (e.g. '5m', '1m', '1d').
            
        Returns:
            Tuple of (reindexed_df, count_of_synthetic_bars_added).
        """
        if df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return df.copy(), 0

        if not IngestionConfig.is_interval_intraday(interval):
            return self._reindex_non_intraday(df)

        return self._reindex_intraday(df, interval)

    def _reindex_non_intraday(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Process daily, weekly, or monthly series by ensuring monotonicity and no duplicates."""
        out = df[~df.index.duplicated(keep="last")].sort_index()
        if "Is_Synthetic" not in out.columns:
            out["Is_Synthetic"] = False
        out["Close"] = out["Close"].ffill().bfill()
        out["Open"] = out["Open"].fillna(out["Close"])
        out["High"] = out["High"].fillna(out["Close"])
        out["Low"] = out["Low"].fillna(out["Close"])
        for col in ["Close_Adj", "Open_Adj", "High_Adj", "Low_Adj"]:
            if col in out.columns:
                out[col] = out[col].fillna(out["Close"])
        if "Volume" in out.columns:
            out["Volume"] = out["Volume"].fillna(0.0)
        return out, 0

    def _reindex_intraday(self, df: pd.DataFrame, interval: str) -> Tuple[pd.DataFrame, int]:
        """Reindex intraday series to exact regular session bars with zero lookahead."""
        freq = self._resolve_frequency(interval)
        target_grid = self._build_target_grid(df.index, freq)

        if len(target_grid) == 0:
            return df.copy(), 0

        out = df[~df.index.duplicated(keep="last")].sort_index()
        is_missing_mask = ~target_grid.isin(out.index)
        num_synthetic = int(is_missing_mask.sum())

        reindexed = out.reindex(target_grid)
        reindexed["Is_Synthetic"] = is_missing_mask

        # Causal forward-fill prices only (strictly zero lookahead)
        reindexed["Close"] = reindexed["Close"].ffill()

        # If initial bars on the first session are missing, truncate leading unobserved NaNs
        # rather than backward-filling future prices into the past
        if reindexed["Close"].isna().any():
            valid_start = reindexed["Close"].first_valid_index()
            if valid_start is not None:
                reindexed = reindexed.loc[valid_start:]
            num_synthetic = int(reindexed["Is_Synthetic"].sum())

        synthetic_idx = reindexed["Is_Synthetic"]
        reindexed.loc[synthetic_idx, "Open"] = reindexed.loc[synthetic_idx, "Close"]
        reindexed.loc[synthetic_idx, "High"] = reindexed.loc[synthetic_idx, "Close"]
        reindexed.loc[synthetic_idx, "Low"] = reindexed.loc[synthetic_idx, "Close"]

        # Fill all adjusted columns to prevent NaNs in downstream feature matrices
        for col in ["Close_Adj", "Open_Adj", "High_Adj", "Low_Adj"]:
            if col in reindexed.columns:
                reindexed[col] = reindexed[col].ffill()
                reindexed.loc[synthetic_idx, col] = reindexed.loc[synthetic_idx, "Close"]

        if "Volume" in reindexed.columns:
            reindexed.loc[synthetic_idx, "Volume"] = 0.0

        return reindexed, num_synthetic

    def _resolve_frequency(self, interval: str) -> str:
        """Resolve bar interval string to pandas offset frequency."""
        norm = interval.lower().strip()
        if norm in self.INTERVAL_FREQ_MAP:
            return self.INTERVAL_FREQ_MAP[norm]
        match = re.match(r"^(\d+)([mhd])$", norm)
        if match:
            val, unit = match.groups()
            unit_map = {"m": "min", "h": "h", "d": "D"}
            return f"{val}{unit_map.get(unit, 'min')}"
        return "5min"

    def _build_target_grid(self, index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
        """Build complete continuous session DatetimeIndex for all trading dates present."""
        unique_dates = index.normalize().unique()
        tz = index.tz
        grid_parts: List[pd.DatetimeIndex] = []

        for session_date in unique_dates:
            date_str = session_date.strftime("%Y-%m-%d")

            # Defensive guard: skip weekends/holidays if exchange is NSE and date is not a special session
            if self.exchange == ExchangeType.NSE:
                if session_date.weekday() >= 5 and date_str not in KNOWN_SPECIAL_SESSIONS:
                    continue
                if date_str in KNOWN_NSE_HOLIDAYS and date_str not in KNOWN_SPECIAL_SESSIONS:
                    continue

            windows = self._get_session_windows(date_str)
            for start_hm, end_hm in windows:
                start_ts = pd.Timestamp(f"{date_str} {start_hm}", tz=tz)
                end_ts = pd.Timestamp(f"{date_str} {end_hm}", tz=tz)
                session_range = pd.date_range(
                    start=start_ts,
                    end=end_ts,
                    freq=freq,
                    inclusive="left",
                )
                grid_parts.append(session_range)

        if not grid_parts:
            return pd.DatetimeIndex([], tz=tz)

        full_index = grid_parts[0]
        for part in grid_parts[1:]:
            full_index = full_index.append(part)
        return full_index.unique().sort_values()

    def _get_session_windows(self, date_str: str) -> List[Tuple[str, str]]:
        """Return list of (start_time, end_time) session windows based on exchange profile."""
        if self.exchange == ExchangeType.NSE:
            if date_str in KNOWN_SPECIAL_SESSIONS:
                return KNOWN_SPECIAL_SESSIONS[date_str]
            return [("09:15:00", "15:30:00")]
        elif self.exchange == ExchangeType.US:
            return [("09:30:00", "16:00:00")]
        elif self.exchange == ExchangeType.GLOBAL_24H:
            return [("00:00:00", "23:59:59")]
        return [("09:15:00", "15:30:00")]
