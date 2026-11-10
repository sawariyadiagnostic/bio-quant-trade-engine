"""Session-aware, causal multi-timeframe resampler implementation."""
from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


class SessionAwareResampler:
    """Causal, session-anchored aggregation engine for intraday and macro timeframes."""

    BASE_AGG_RULES: Dict[str, str] = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Open_Adj": "first",
        "High_Adj": "max",
        "Low_Adj": "min",
        "Close_Adj": "last",
        "Volume": "sum",
        "Is_Synthetic": "all",
        "Is_Spike": "any",
    }

    @classmethod
    def resample_intraday(
        cls,
        df: pd.DataFrame,
        target_minutes: int,
        base_bar_minutes: int = 5,
    ) -> pd.DataFrame:
        """Resample intraday bars with session-anchored origin, boundary clamping, and vectorized performance."""
        if df.empty:
            return df.copy()

        df = df.sort_index()
        target_delta = pd.Timedelta(minutes=target_minutes)
        base_delta = pd.Timedelta(minutes=base_bar_minutes)

        index_series = df.index.to_series()
        time_diff = index_series.diff()
        date_diff = df.index.normalize() != pd.Series(df.index.normalize(), index=df.index).shift(1)
        is_new_session = (time_diff > pd.Timedelta(minutes=60)) | date_diff
        is_new_session.iloc[0] = True
        session_id = is_new_session.cumsum()

        session_starts = index_series.groupby(session_id).transform("first")
        session_ends = index_series.groupby(session_id).transform("last") + base_delta

        elapsed = df.index - session_starts
        bin_num = elapsed // target_delta
        bin_right = session_starts + (bin_num + 1) * target_delta
        clamped_right = np.minimum(bin_right, session_ends)

        agg_rules = {col: cls.BASE_AGG_RULES.get(col, "last") for col in df.columns}
        resampled = df.groupby(clamped_right).agg(agg_rules)

        has_ohlcv = all(c in df.columns for c in ["High", "Low", "Close", "Volume"])
        if has_ohlcv:
            typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
            turnover = typical_price * df["Volume"]
            session_to = turnover.groupby(clamped_right).sum()
            vol = resampled["Volume"]
            resampled["VWAP"] = np.where(vol > 0, session_to / vol, resampled["Close"])

        if "Is_Synthetic" in df.columns:
            resampled["Synthetic_Ratio"] = df["Is_Synthetic"].astype(float).groupby(clamped_right).mean()

        resampled.index.name = None
        resampled.index = pd.DatetimeIndex(resampled.index, tz=df.index.tz)

        resampled = resampled.dropna(how="all")
        if resampled.isna().any().any():
            nan_cols = resampled.columns[resampled.isna().any()].tolist()
            raise ValueError(f"Resampling produced NaNs in columns: {nan_cols}")

        return resampled

    @classmethod
    def resample_daily_to_higher(
        cls,
        df: pd.DataFrame,
        freq: str = "W-FRI",
        require_closed_period: bool = True,
    ) -> pd.DataFrame:
        """Resample daily series to weekly or monthly with causal sealing."""
        if df.empty:
            return df.copy()

        df = df.sort_index()
        has_ohlcv = all(c in df.columns for c in ["High", "Low", "Close", "Volume"])
        turnover = (df["High"] + df["Low"] + df["Close"]) / 3.0 * df["Volume"] if has_ohlcv else None

        agg_rules = {col: cls.BASE_AGG_RULES.get(col, "last") for col in df.columns}
        res = df[list(agg_rules.keys())].resample(freq, closed="right", label="right").agg(agg_rules)

        if turnover is not None and "Volume" in res.columns and "Close" in res.columns:
            res_to = turnover.resample(freq, closed="right", label="right").sum()
            vol = res["Volume"]
            res["VWAP"] = np.where(vol > 0, res_to / vol, res["Close"])

        count_col = "Close" if "Close" in df.columns else df.columns[0]
        res["Bar_Count"] = df[count_col].resample(freq, closed="right", label="right").count()
        res["Last_Trading_Date"] = df.index.to_series().resample(freq, closed="right", label="right").max()

        res = res[res["Bar_Count"] > 0]
        if require_closed_period and not res.empty:
            last_date = df.index[-1]
            last_res_date = res.index[-1]
            is_weekly = "W" in freq.upper()
            is_monthly = freq.upper() in ("ME", "M", "1MO")
            if is_weekly:
                is_fri_or_later = last_date.weekday() >= 4
                from ravana.data.ingestion.session_gatekeeper import NSESessionGatekeeper
                rem_days = [last_date + pd.Timedelta(days=d) for d in range(1, 5 - last_date.weekday())]
                all_rem_holidays = bool(rem_days) and all(
                    d.strftime("%Y-%m-%d") in NSESessionGatekeeper.KNOWN_NSE_HOLIDAYS or d.weekday() >= 5
                    for d in rem_days
                )
                is_closed = is_fri_or_later or all_rem_holidays or (last_date >= last_res_date)
            elif is_monthly:
                next_bday = last_date + pd.offsets.BDay(1)
                is_closed = (next_bday.month != last_date.month) or (last_date >= last_res_date)
            else:
                is_closed = last_date >= last_res_date

            if not is_closed:
                res = res.iloc[:-1]

        return res

    @staticmethod
    def point_in_time_merge(
        base_df: pd.DataFrame,
        higher_df: pd.DataFrame,
        higher_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Merge higher-timeframe features onto base series causally via backward as-of join."""
        if base_df.empty or higher_df.empty:
            return base_df.copy()

        target_higher = higher_df.copy()

        # Causal Sealing: If higher series is normalized to midnight in its native timezone,
        # seal it to the end of its native session (23:59:59) BEFORE timezone conversion
        # to prevent leaking US market closes 15 hours early into Indian trading hours.
        if (target_higher.index.time == pd.Timestamp("00:00:00").time()).all():
            target_higher.index = target_higher.index + pd.Timedelta(hours=23, minutes=59, seconds=59)

        if target_higher.index.tz != base_df.index.tz:
            if target_higher.index.tz is None and base_df.index.tz is not None:
                target_higher = target_higher.tz_localize(base_df.index.tz)
            elif target_higher.index.tz is not None and base_df.index.tz is None:
                target_higher = target_higher.tz_localize(None)
            else:
                target_higher = target_higher.tz_convert(base_df.index.tz)

        features = target_higher[higher_cols] if higher_cols else target_higher

        # Guard any remaining midnight-aligned series in base timezone
        if (features.index.time == pd.Timestamp("00:00:00").time()).all() and (base_df.index.time != pd.Timestamp("00:00:00").time()).any():
            features = features.copy()
            features.index = features.index + pd.Timedelta(hours=23, minutes=59, seconds=59)

        return pd.merge_asof(
            base_df.sort_index(),
            features.sort_index(),
            left_index=True,
            right_index=True,
            direction="backward",
        )
