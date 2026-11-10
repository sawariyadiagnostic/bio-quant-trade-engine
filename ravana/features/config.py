"""Configuration and Feature Registry for Muscle 7 Feature Matrix Engine.

Defines the 21 standardized stationary market vocabulary features,
their mathematical descriptions, rolling windows, and normalization rules.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class FeatureName(str, Enum):
    """Canonical registry of the 21 institutional stationary features."""

    # Price Returns (Stationary)
    LOG_RET_1 = "LOG_RET_1"
    LOG_RET_5 = "LOG_RET_5"
    LOG_RET_20 = "LOG_RET_20"

    # Volatility & Range (Scale-Invariant)
    ATR_14_NORM = "ATR_14_NORM"
    VOLATILITY_20 = "VOLATILITY_20"
    PARKINSON_VOL_20 = "PARKINSON_VOL_20"
    GARMAN_KLASS_VOL_20 = "GARMAN_KLASS_VOL_20"

    # Volume & Microstructure (Causally Lagged)
    VOLUME_ZSCORE_LAGGED = "VOLUME_ZSCORE_LAGGED"
    VWAP_DRIFT = "VWAP_DRIFT"

    # Momentum & Technical Oscillators
    RSI_14 = "RSI_14"
    BOLLINGER_BAND_PCT_20 = "BOLLINGER_BAND_PCT_20"
    MACD_DIFF_NORM = "MACD_DIFF_NORM"

    # ETF Tracking Spread & Cointegration Dynamics
    SPREAD_BPS = "SPREAD_BPS"
    SPREAD_ZSCORE_20 = "SPREAD_ZSCORE_20"
    SPREAD_DELTA_5 = "SPREAD_DELTA_5"
    ADF_STAT_60 = "ADF_STAT_60"
    OU_HALF_LIFE_60 = "OU_HALF_LIFE_60"

    # Cross-Asset & Global Macro Drivers
    MACRO_VIX_LEVEL = "MACRO_VIX_LEVEL"
    MACRO_VIX_RET_1 = "MACRO_VIX_RET_1"
    MACRO_USDINR_RET_1 = "MACRO_USDINR_RET_1"
    MACRO_SP500_RET_1 = "MACRO_SP500_RET_1"


class FeatureRegistry:
    """Provides metadata and discovery for all 21 stationary features."""

    _DESCRIPTIONS: Dict[str, str] = {
        FeatureName.LOG_RET_1.value: "1-bar logarithmic return ln(Close_t / Close_{t-1})",
        FeatureName.LOG_RET_5.value: "5-bar logarithmic return ln(Close_t / Close_{t-5})",
        FeatureName.LOG_RET_20.value: "20-bar logarithmic return ln(Close_t / Close_{t-20})",
        FeatureName.ATR_14_NORM.value: "14-bar Normalized Average True Range (ATR / Close)",
        FeatureName.VOLATILITY_20.value: "20-bar realized standard deviation of log returns",
        FeatureName.PARKINSON_VOL_20.value: "20-bar Parkinson high-low scale-invariant volatility",
        FeatureName.GARMAN_KLASS_VOL_20.value: "20-bar Garman-Klass OHLC volatility estimator",
        FeatureName.VOLUME_ZSCORE_LAGGED.value: "20-bar volume z-score lagged by 1 bar (zero lookahead)",
        FeatureName.VWAP_DRIFT.value: "Fractional drift from session VWAP: (Close - VWAP) / VWAP",
        FeatureName.RSI_14.value: "14-bar Relative Strength Index centered at zero in [-1.0, 1.0]",
        FeatureName.BOLLINGER_BAND_PCT_20.value: "20-bar Bollinger Band %b in [0.0, 1.0]",
        FeatureName.MACD_DIFF_NORM.value: "Normalized MACD histogram difference (MACD - Signal) / Close",
        FeatureName.SPREAD_BPS.value: "ETF tracking spread to fair index in basis points",
        FeatureName.SPREAD_ZSCORE_20.value: "20-bar rolling z-score of ETF tracking spread",
        FeatureName.SPREAD_DELTA_5.value: "5-bar change in ETF tracking spread basis points",
        FeatureName.ADF_STAT_60.value: "60-bar rolling Augmented Dickey-Fuller stationarity t-statistic",
        FeatureName.OU_HALF_LIFE_60.value: "60-bar Ornstein-Uhlenbeck mean-reversion half-life (bars)",
        FeatureName.MACRO_VIX_LEVEL.value: "Log-normalized India/US VIX level: ln(VIX / 15.0)",
        FeatureName.MACRO_VIX_RET_1.value: "1-bar percentage change in VIX volatility index",
        FeatureName.MACRO_USDINR_RET_1.value: "1-bar return in USD/INR currency driver",
        FeatureName.MACRO_SP500_RET_1.value: "1-bar return in S&P 500 global benchmark",
    }

    @classmethod
    def all_features(cls) -> List[str]:
        """Return list of all 21 registered feature names."""
        return [f.value for f in FeatureName]

    @classmethod
    def get_description(cls, feature_name: str) -> str:
        """Return human-readable mathematical description of feature."""
        return cls._DESCRIPTIONS.get(feature_name, "Unknown feature")


@dataclass
class FeatureConfig:
    """Configuration parameters for FeatureMatrixEngine computation."""

    # Window lookbacks
    short_return_window: int = 5
    medium_return_window: int = 20
    atr_window: int = 14
    volatility_window: int = 20
    volume_window: int = 20
    volume_lag: int = 1
    rsi_window: int = 14
    bollinger_window: int = 20
    bollinger_std: float = 2.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    spread_z_window: int = 20
    spread_delta_window: int = 5
    adf_window: int = 60
    ou_window: int = 60
    ou_min_half_life: float = 1.0
    ou_max_half_life: float = 252.0

    # Macro baseline level for normalization
    vix_baseline: float = 15.0

    # Finite bounds clipping
    clip_outliers: bool = True
    max_abs_feature_val: float = 1e5

    # Target features list
    active_features: List[str] = field(default_factory=FeatureRegistry.all_features)
