"""Flash spike detection and causal outlier sanitization for tick/bar feeds."""
from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd


class FlashSpikeFilter:
    """Detects and sanitizes aberrant single-bar price spikes without lookahead bias.
    
    Uses causal rolling statistics (shifted by 1 bar) to identify anomalous prints
    and clamps them back to statistical bounds while preserving valid market volatility.
    """

    def __init__(
        self,
        window: int = 20,
        threshold_std: float = 4.0,
        clamp_spikes: bool = True,
    ) -> None:
        self.window = max(5, window)
        self.threshold_std = float(threshold_std)
        self.clamp_spikes = clamp_spikes

    def filter(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Detect and clamp price spikes in the OHLCV DataFrame.
        
        Args:
            df: DataFrame containing Open, High, Low, Close, Volume.
            
        Returns:
            Tuple of (sanitized_df, count_of_spikes_detected).
        """
        if df.empty or len(df) < 5 or "Close" not in df.columns:
            out = df.copy()
            if "Is_Spike" not in out.columns:
                out["Is_Spike"] = False
            return out, 0

        out = df.copy()
        close = out["Close"]

        # Strictly causal: shift by 1 so bar t is not evaluated against itself
        rolling_med = close.rolling(self.window, min_periods=3).median().shift(1)
        rolling_med = rolling_med.fillna(close.expanding(min_periods=1).median().shift(1)).fillna(close)

        # Robust scale estimator: Causal Median Absolute Deviation (MAD), breakdown point 50%
        abs_dev = (close.shift(1) - rolling_med).abs()
        rolling_mad = abs_dev.rolling(self.window, min_periods=3).median().shift(1)
        rolling_mad = rolling_mad.fillna(abs_dev.expanding(min_periods=1).median().shift(1)).fillna(0.0)
        robust_sigma = 1.4826 * rolling_mad

        # Baseline noise floor (0.2% of price)
        sigma_floor = rolling_med * 0.002
        effective_sigma = np.maximum(robust_sigma.values, sigma_floor.values)

        upper_bound = rolling_med.values + self.threshold_std * effective_sigma
        lower_bound = np.maximum(0.01, rolling_med.values - self.threshold_std * effective_sigma)

        is_close_spike = (close.values > upper_bound) | (close.values < lower_bound)

        # Check aberrant wicks
        max_oc = np.maximum(out["Open"].values, close.values)
        min_oc = np.minimum(out["Open"].values, close.values)
        is_high_spike = (out["High"].values - max_oc) > (self.threshold_std * effective_sigma)
        is_low_spike = (min_oc - out["Low"].values) > (self.threshold_std * effective_sigma)

        total_spike_mask = is_close_spike | is_high_spike | is_low_spike
        spike_count = int(total_spike_mask.sum())

        if self.clamp_spikes and spike_count > 0:
            out = self._apply_clamping(out, is_close_spike, is_high_spike, is_low_spike, upper_bound, lower_bound, effective_sigma)

        out["Is_Spike"] = total_spike_mask
        return out, spike_count

    def _apply_clamping(
        self,
        df: pd.DataFrame,
        is_close_spike: np.ndarray,
        is_high_spike: np.ndarray,
        is_low_spike: np.ndarray,
        upper_bound: np.ndarray,
        lower_bound: np.ndarray,
        effective_sigma: np.ndarray,
    ) -> pd.DataFrame:
        """Clamp anomalous values back to their causal statistical bounds."""
        # Clamp Close
        df.loc[is_close_spike, "Close"] = np.clip(
            df.loc[is_close_spike, "Close"].values,
            lower_bound[is_close_spike],
            upper_bound[is_close_spike],
        )

        max_oc = np.maximum(df["Open"].values, df["Close"].values)
        min_oc = np.minimum(df["Open"].values, df["Close"].values)

        # Clamp High wicks
        df.loc[is_high_spike, "High"] = max_oc[is_high_spike] + (self.threshold_std * effective_sigma[is_high_spike])

        # Clamp Low wicks
        df.loc[is_low_spike, "Low"] = np.maximum(
            0.01,
            min_oc[is_low_spike] - (self.threshold_std * effective_sigma[is_low_spike]),
        )

        # Ensure candlestick physical geometry: High >= max(Open, Close), Low <= min(Open, Close)
        df["High"] = np.maximum(df["High"].values, np.maximum(df["Open"].values, df["Close"].values))
        df["Low"] = np.minimum(df["Low"].values, np.minimum(df["Open"].values, df["Close"].values))

        if "Close_Adj" in df.columns:
            df["Close_Adj"] = df["Close"]

        return df
