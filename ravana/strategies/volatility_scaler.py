"""Dynamic Volatility Hurdle and Payoff Scaler for RAVANA v4.3.

Dynamically scales strategy momentum hurdles, profit targets, and stop-loss distances
based on the daily opening ATR ratio (Today's ATR / 20-Day Median ATR) to capture
intraday gains on low-ATR days while maintaining strict Risk-to-Reward symmetry.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AdaptiveThresholdScaler:
    """Calculates volatility-adaptive thresholds for momentum and mean-reversion engines."""

    def __init__(
        self,
        low_vol_cutoff: float = 0.80,
        high_vol_cutoff: float = 1.20,
        min_hurdle_floor: float = 0.85,
    ) -> None:
        self.low_vol_cutoff = float(low_vol_cutoff)
        self.high_vol_cutoff = float(high_vol_cutoff)
        self.min_hurdle_floor = float(min_hurdle_floor)

    def compute_atr_ratio(self, today_atr: float, median_20d_atr: float) -> float:
        """Compute the ratio between today's opening ATR and historical 20-day median ATR."""
        if median_20d_atr <= 0.0 or today_atr <= 0.0:
            return 1.0
        return float(today_atr / median_20d_atr)

    def get_momentum_hurdle(self, base_hurdle: float = 1.20, atr_ratio: float = 1.0) -> float:
        """Dynamically scale the entry momentum Z-score hurdle.

        On low-ATR days, reduce the hurdle down to floor (0.85) to enable setup activation.
        On high-ATR days, expand hurdle to prevent entering extended moves.
        """
        if atr_ratio < self.low_vol_cutoff:
            scaled = base_hurdle * atr_ratio
            return max(self.min_hurdle_floor, round(scaled, 2))
        if atr_ratio >= self.high_vol_cutoff:
            return min(1.35, round(base_hurdle * 1.125, 2))
        return float(base_hurdle)

    def get_target_multiplier(self, base_mult: float = 1.50, atr_ratio: float = 1.0) -> float:
        """Dynamically scale the profit target ATR multiplier.

        On low-ATR days, compress target to 0.75 ATR to bank realistic intraday gains.
        On high-ATR days, expand target to 2.00 ATR for parabolic runner capture.
        """
        if atr_ratio < self.low_vol_cutoff:
            return 0.75
        if atr_ratio >= self.high_vol_cutoff:
            return 2.00
        return float(base_mult)

    def get_stop_multiplier(self, base_mult: float = 0.75, atr_ratio: float = 1.0) -> float:
        """Dynamically scale the initial stop-loss ATR multiplier.

        On low-ATR days, tighten stop to 0.45 ATR to maintain asymmetric 1.67:1 R:R payoff.
        """
        if atr_ratio < self.low_vol_cutoff:
            return 0.45
        return float(base_mult)

    def get_inav_bps_hurdle(self, base_bps: float = 15.0, atr_ratio: float = 1.0) -> float:
        """Dynamically compress the iNAV tracking error hurdle on quiet days."""
        if atr_ratio < self.low_vol_cutoff:
            return 10.0
        return float(base_bps)
