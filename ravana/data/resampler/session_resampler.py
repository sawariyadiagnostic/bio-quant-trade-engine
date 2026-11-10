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
    def _resample_single_session(
        cls,
        session_df: pd.DataFrame,
        turnover_series: Optional[pd.Series],
        rule: str,
        agg_rules: Dict[str, str],
        base_delta: pd.Timedelta,
    ) -> pd.DataFrame:
        """Resample a single session's bars with right-labeling and session clamping."""
        resampled = session_df[list(agg_rules.keys())].resample(
            rule, closed="left", label="right", origin="start"
        ).agg(agg_rules)

        if turnover_series is not None and "Volume" in resampled.columns and "Close" in resampled.columns:
            session_to = turnover_series.loc[session_df.index].resample(
                rule, closed="left", label="right", origin="start"
            ).sum()
            vol = resampled["Volume"]
            resampled["VWAP"] = np.where(vol > 0, session_to / vol, resampled["Close"])

        if "Is_Synthetic" in session_df.columns:
            resampled["Synthetic_Ratio"] = session_df["Is_Synthetic"].astype(float).resample(
                rule, closed="left", label="right", origin="start"
            ).mean()

        session_close_ts = session_df.index[-1] + base_delta
        clamped_idx = [min(ts, session_close_ts) for ts in resampled.index]
        resampled.index = pd.DatetimeIndex(clamped_idx, tz=session_df.index.tz)

        if resampled.isna().any().any():
            nan_cols = resampled.columns[resampled.isna().any()].tolist()
            raise ValueError(f"Resampling produced NaNs in columns: {nan_cols}")

        return resampled

    @classmethod
    def resample_intraday(
        cls,
        df: pd.DataFrame,
        target_minutes: int,
        base_bar_minutes: int = 5,
    ) -> pd.DataFrame:
        """Resample intraday bars with session-anchored origin and boundary clamping."""
        if df.empty:
            return df.copy()

        rule = f"{target_minutes}min"
        base_delta = pd.Timedelta(minutes=base_bar_minutes)
        has_ohlcv = all(c in df.columns for c in ["High", "Low", "Close", "Volume"])
        if has_ohlcv:
            typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
            turnover: Optional[pd.Series] = typical_price * df["Volume"]
        else:
            turnover = None

        agg_rules = {col: cls.BASE_AGG_RULES.get(col, "last") for col in df.columns}
        session_frames: List[pd.DataFrame] = []

        for _, session_df in df.groupby(df.index.normalize()):
            if session_df.empty:
                continue
            frame = cls._resample_single_session(session_df, turnover, rule, agg_rules, base_delta)
            session_frames.append(frame)

        if not session_frames:
            return pd.DataFrame(columns=df.columns)
        return pd.concat(session_frames)

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

        has_ohlcv = all(c in df.columns for c in ["High", "Low", "Close", "Volume"])
        if has_ohlcv:
            typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
            turnover: Optional[pd.Series] = typical_price * df["Volume"]
        else:
            turnover = None

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
            if df.index[-1] < res.index[-1]:
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

        target_higher = higher_df
        if higher_df.index.tz != base_df.index.tz:
            target_higher = higher_df.tz_convert(base_df.index.tz)

        features = target_higher[higher_cols] if higher_cols else target_higher
        return pd.merge_asof(
            base_df.sort_index(),
            features.sort_index(),
            left_index=True,
            right_index=True,
            direction="backward",
        )
