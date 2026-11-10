"""Flash spike detection and causal outlier sanitization for tick/bar feeds."""
from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd


class FlashSpikeFilter:
    """Detects and sanitizes aberrant single-bar price spikes without lookahead bias.
    
    Uses causal Median Absolute Deviation (MAD) with 50% breakdown point and an
    institutional volume gate to distinguish bad tick glitches from genuine breakouts.
    """

    def __init__(
        self,
        window: int = 20,
        threshold_std: float = 4.0,
        clamp_spikes: bool = True,
        use_volume_gate: bool = True,
        volume_surge_multiplier: float = 2.0,
        volume_surge_expansion: float = 2.0,
    ) -> None:
        self.window = max(5, window)
        self.threshold_std = float(threshold_std)
        self.clamp_spikes = clamp_spikes
        self.use_volume_gate = use_volume_gate
        self.volume_surge_multiplier = float(volume_surge_multiplier)
        self.volume_surge_expansion = float(volume_surge_expansion)

    def filter(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Detect and clamp price spikes in the OHLCV DataFrame."""
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
        abs_dev = (close - rolling_med).abs()
        rolling_mad = abs_dev.rolling(self.window, min_periods=3).median().shift(1)
        rolling_mad = rolling_mad.fillna(abs_dev.expanding(min_periods=1).median().shift(1)).fillna(0.0)
        robust_sigma = 1.4826 * rolling_mad

        # Baseline noise floor (0.2% of price)
        sigma_floor = rolling_med * 0.002
        effective_sigma = np.maximum(robust_sigma.values, sigma_floor.values)

        # Volume Gate: distinguish anemic bad ticks from high-volume institutional momentum
        threshold_mult = np.full(len(df), self.threshold_std)
        if self.use_volume_gate and "Volume" in out.columns:
            vol = out["Volume"]
            vol_med = vol.rolling(self.window, min_periods=3).median().shift(1).fillna(vol)
            vol_med = np.maximum(1.0, vol_med.values)
            vol_ratio = vol.values / vol_med
            # If volume is surge (e.g. >2x median), expand threshold to protect breakouts
            surge_mask = vol_ratio >= self.volume_surge_multiplier
            threshold_mult[surge_mask] = self.threshold_std * self.volume_surge_expansion

        upper_bound = rolling_med.values + threshold_mult * effective_sigma
        lower_bound = np.maximum(0.01, rolling_med.values - threshold_mult * effective_sigma)

        is_close_spike = (close.values > upper_bound) | (close.values < lower_bound)
        is_open_spike = np.zeros(len(df), dtype=bool)
        if "Open" in out.columns:
            is_open_spike = (out["Open"].values > upper_bound) | (out["Open"].values < lower_bound)

        # Check aberrant wicks
        max_oc = np.maximum(out["Open"].values, close.values)
        min_oc = np.minimum(out["Open"].values, close.values)
        is_high_spike = (out["High"].values - max_oc) > (threshold_mult * effective_sigma)
        is_low_spike = (min_oc - out["Low"].values) > (threshold_mult * effective_sigma)

        total_spike_mask = is_close_spike | is_open_spike | is_high_spike | is_low_spike
        spike_count = int(total_spike_mask.sum())

        if self.clamp_spikes and spike_count > 0:
            out = self._apply_clamping(
                out, is_close_spike, is_open_spike, is_high_spike, is_low_spike,
                upper_bound, lower_bound, effective_sigma, threshold_mult,
            )

        out["Is_Spike"] = total_spike_mask
        return out, spike_count

    def _apply_clamping(
        self,
        df: pd.DataFrame,
        is_close_spike: np.ndarray,
        is_open_spike: np.ndarray,
        is_high_spike: np.ndarray,
        is_low_spike: np.ndarray,
        upper_bound: np.ndarray,
        lower_bound: np.ndarray,
        effective_sigma: np.ndarray,
        threshold_mult: np.ndarray,
    ) -> pd.DataFrame:
        """Clamp anomalous values back to their causal statistical bounds."""
        # Clamp Close & Open symmetrically
        df.loc[is_close_spike, "Close"] = np.clip(
            df.loc[is_close_spike, "Close"].values,
            lower_bound[is_close_spike],
            upper_bound[is_close_spike],
        )
        if "Open" in df.columns:
            df.loc[is_open_spike, "Open"] = np.clip(
                df.loc[is_open_spike, "Open"].values,
                lower_bound[is_open_spike],
                upper_bound[is_open_spike],
            )

        max_oc = np.maximum(df["Open"].values, df["Close"].values)
        min_oc = np.minimum(df["Open"].values, df["Close"].values)

        # Clamp High wicks
        df.loc[is_high_spike, "High"] = max_oc[is_high_spike] + (threshold_mult[is_high_spike] * effective_sigma[is_high_spike])

        # Clamp Low wicks
        df.loc[is_low_spike, "Low"] = np.maximum(
            0.01,
            min_oc[is_low_spike] - (threshold_mult[is_low_spike] * effective_sigma[is_low_spike]),
        )

        # Re-enforce candlestick physical geometry
        df["High"] = np.maximum(df["High"].values, np.maximum(df["Open"].values, df["Close"].values))
        df["Low"] = np.minimum(df["Low"].values, np.minimum(df["Open"].values, df["Close"].values))

        for p_col, a_col in [("Close", "Close_Adj"), ("Open", "Open_Adj"), ("High", "High_Adj"), ("Low", "Low_Adj")]:
            if a_col in df.columns and p_col in df.columns:
                ratio = np.where(df[p_col] > 0, df[a_col] / df[p_col], 1.0)
                df[a_col] = df[p_col] * ratio

        return df
