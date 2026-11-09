"""Vectorized Data Lake with Statutory Circuit Band Winsorization (Wave 1: Task 1.1)."""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


def validate_and_clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-flight validation: enforces datetime index, deduplication, monotonicity, and NaN filling."""
    clean_df = df.copy()
    if not isinstance(clean_df.index, pd.DatetimeIndex):
        clean_df.index = pd.to_datetime(clean_df.index)
    if clean_df.index.has_duplicates:
        clean_df = clean_df[~clean_df.index.duplicated(keep="last")]
    if not clean_df.index.is_monotonic_increasing:
        clean_df = clean_df.sort_index()
    # NaN filling: forward-fill then backward-fill, then zero-fill any residual uninitialized cells
    return clean_df.ffill().bfill().infer_objects(copy=False).fillna(0.0)


def adjust_corporate_splits(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Corporate actions split adjustment scaling OHLCV by Close_Adj / Close ratio."""
    has_adj = "Close_Adj" in df.columns
    raw_col = "Close" if "Close" in df.columns else ("Close_Raw" if "Close_Raw" in df.columns else None)

    o = df["Open"].to_numpy(dtype=np.float64)
    h = df["High"].to_numpy(dtype=np.float64)
    l = df["Low"].to_numpy(dtype=np.float64)
    v = df.get("Volume", pd.Series(10000.0, index=df.index)).to_numpy(dtype=np.float64)

    if has_adj and raw_col is not None:
        c_adj = df["Close_Adj"].to_numpy(dtype=np.float64)
        c_raw = df[raw_col].to_numpy(dtype=np.float64)
        k = np.where(c_raw > 1e-8, c_adj / np.maximum(c_raw, 1e-8), 1.0)
        # Apply corporate split ratio if Close_Adj differs from raw Close
        if not np.allclose(k, 1.0, atol=1e-5):
            o = o * k
            h = h * k
            l = l * k
            v = v / np.maximum(k, 1e-8)
        c = c_adj
    elif has_adj:
        c = df["Close_Adj"].to_numpy(dtype=np.float64)
    elif raw_col is not None:
        c = df[raw_col].to_numpy(dtype=np.float64)
    else:
        c = df["Open"].to_numpy(dtype=np.float64)

    return c, o, h, l, v


@dataclass
class VectorDataLake:
    """High-performance vectorized market tensor store."""
    ticker: str
    tensors: Dict[str, np.ndarray]
    dates: pd.DatetimeIndex

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, ticker: str = "NIFTYBEES.NS") -> "VectorDataLake":
        """Builds a vectorized lake from a historical OHLCV DataFrame."""
        df = validate_and_clean_ohlcv(df)
        c, o, h, l, v = adjust_corporate_splits(df)

        n = len(c)
        # 1. Statutory Return Vectors (Overnight, Intraday, Close-to-Close)
        # Forward Overnight Drift: Close[t] to Open[t+1] (for forward horizon simulation)
        overnight = np.zeros(n, dtype=np.float64)
        if n > 1:
            overnight[:-1] = (o[1:] - c[:-1]) / (c[:-1] + 1e-8)

        # Realized Overnight Auction Jump: Close[t-1] to Open[t] (lagged auction return)
        auction_jump = np.zeros(n, dtype=np.float64)
        if n > 1:
            auction_jump[1:] = (o[1:] - c[:-1]) / (c[:-1] + 1e-8)

        # Intraday: Open[t] to Close[t]
        intraday = (c - o) / (o + 1e-8)
        intraday_core = intraday.copy()

        # Close-to-Close: Close[t-1] to Close[t]
        c2c = np.zeros(n, dtype=np.float64)
        if n > 1:
            c2c[1:] = (c[1:] - c[:-1]) / (c[:-1] + 1e-8)

        # Statutory NSE Circuit Band Winsorization (-10% to +10%)
        overnight = np.clip(overnight, -0.10, 0.10)
        auction_jump = np.clip(auction_jump, -0.10, 0.10)
        intraday = np.clip(intraday, -0.10, 0.10)
        intraday_core = np.clip(intraday_core, -0.10, 0.10)
        c2c = np.clip(c2c, -0.10, 0.10)

        # 2. Standard Institutional Factor Computations
        c_series = pd.Series(c, index=df.index)
        v_series = pd.Series(v, index=df.index)
        typical_p = (h + l + c) / 3.0
        pv = typical_p * v
        if isinstance(df.index, pd.DatetimeIndex) and len(np.unique(df.index.date)) < len(df):
            # Intraday bars spanning multiple dates: reset VWAP at session day boundaries
            df_vwap = pd.DataFrame({"pv": pv, "v": v, "date": df.index.date})
            grouped = df_vwap.groupby("date")
            cum_pv = grouped["pv"].cumsum().to_numpy()
            cum_vol = grouped["v"].cumsum().to_numpy()
            vwap = np.where(cum_vol > 0, cum_pv / (cum_vol + 1e-8), typical_p)
        else:
            # Single bar per day or homogeneous daily series
            vwap = typical_p
        dist_vwap = (c - vwap) / (vwap + 1e-8)

        # True Range
        c_prev = np.roll(c, 1)
        c_prev[0] = c[0]
        tr = np.maximum(h - l, np.maximum(np.abs(h - c_prev), np.abs(l - c_prev)))

        # Volume Z-Score (20-day rolling with denominator clamped >= 1.0)
        v_roll_mean = v_series.rolling(20, min_periods=1).mean().to_numpy()
        v_roll_std = v_series.rolling(20, min_periods=1).std(ddof=0).fillna(1.0).to_numpy()
        denom = np.maximum(v_roll_std, 1.0)
        vol_zscore = (v - v_roll_mean) / denom

        # Volatilities
        log_hl = np.log(np.maximum(h / np.maximum(l, 1e-8), 1e-8)) ** 2
        log_co = np.log(np.maximum(c / np.maximum(o, 1e-8), 1e-8)) ** 2
        roll_hl = pd.Series(log_hl, index=df.index).rolling(20, min_periods=1).mean().to_numpy()
        roll_co = pd.Series(log_co, index=df.index).rolling(20, min_periods=1).mean().to_numpy()
        parkinson_vol = np.sqrt(roll_hl / (4.0 * np.log(2.0))) * np.sqrt(252)
        gk_raw = 0.5 * roll_hl - (2.0 * np.log(2.0) - 1.0) * roll_co
        garman_klass_vol = np.sqrt(np.maximum(gk_raw, 0.0)) * np.sqrt(252)

        # Downside Vol Ratio
        rets_s = pd.Series(c2c, index=df.index)
        down_rets = rets_s.where(rets_s < 0, 0.0)
        down_var = down_rets.rolling(20, min_periods=1).var().fillna(0.0).to_numpy()
        tot_var = rets_s.rolling(20, min_periods=1).var().fillna(1e-6).to_numpy()
        downside_vol_ratio = np.sqrt(down_var / (tot_var + 1e-8))

        tensors = {
            "Close_Adj": c.astype(np.float32),
            "Open": o.astype(np.float32),
            "High": h.astype(np.float32),
            "Low": l.astype(np.float32),
            "Volume": v.astype(np.float32),
            "gross_overnight_rets": overnight.astype(np.float32),
            "overnight_auction_jump": auction_jump.astype(np.float32),
            "gross_intraday_rets": intraday.astype(np.float32),
            "intraday_core_return": intraday_core.astype(np.float32),
            "gross_close_to_close_rets": c2c.astype(np.float32),
            "Dist_VWAP": dist_vwap.astype(np.float32),
            "True_Range": tr.astype(np.float32),
            "Vol_ZScore": vol_zscore.astype(np.float32),
            "Parkinson_Vol": parkinson_vol.astype(np.float32),
            "Garman_Klass_Vol": garman_klass_vol.astype(np.float32),
            "Downside_Vol_Ratio": downside_vol_ratio.astype(np.float32),
        }

        return cls(ticker=ticker, tensors=tensors, dates=df.index)

    def to_dataframe(self) -> pd.DataFrame:
        """Converts tensors back to a standard pandas DataFrame."""
        cols = {}
        for k, v in self.tensors.items():
            if hasattr(v, "numpy"):
                cols[k] = v.numpy()
            elif isinstance(v, np.ndarray):
                cols[k] = v
            else:
                cols[k] = np.asarray(v)
        return pd.DataFrame(cols, index=self.dates)

    @classmethod
    def load_mock(cls, periods: int = 100, ticker: str = "NIFTYBEES.NS") -> "VectorDataLake":
        """Generates realistic synthetic OHLCV data lake for testing."""
        np.random.seed(42)
        idx = pd.date_range("2024-01-01", periods=periods, freq="B")
        daily_drift = np.random.normal(0.0005, 0.012, periods)
        p = 250.0 * np.exp(np.cumsum(daily_drift))
        df = pd.DataFrame(index=idx)
        df["Open"] = p * (1.0 + np.random.normal(0.0, 0.003, periods))
        df["Close_Adj"] = p
        df["High"] = np.maximum(df["Open"], df["Close_Adj"]) * (1.0 + np.abs(np.random.normal(0.0, 0.005, periods)))
        df["Low"] = np.minimum(df["Open"], df["Close_Adj"]) * (1.0 - np.abs(np.random.normal(0.0, 0.005, periods)))
        df["Volume"] = np.random.uniform(50000.0, 200000.0, periods)
        return cls.from_dataframe(df, ticker=ticker)
