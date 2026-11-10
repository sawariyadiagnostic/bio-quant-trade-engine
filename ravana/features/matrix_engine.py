"""Feature Matrix Engine for Muscle 7.

Computes 21 institutional stationary features across price, volatility,
volume, tracking spread cointegration, and macro drivers. Generates 2D matrices
and 3D tensors compatible with Muscle 2 Tensor Engine.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch

from ravana.features.config import FeatureConfig, FeatureName, FeatureRegistry
from ravana.features.stationarity import RollingADF, RollingOUHalfLife
from ravana.features.technical_primitives import (
    bollinger_pct,
    garman_klass_volatility,
    lagged_volume_zscore,
    log_returns,
    macd_normalized,
    normalized_atr,
    parkinson_volatility,
    realized_volatility,
    relative_strength_index,
    tracking_spread_bps,
    vwap_drift,
)


class FeatureMatrixEngine:
    """Orchestrates feature extraction and tensor transformations."""

    def __init__(self, config: Optional[FeatureConfig] = None) -> None:
        self.config = config or FeatureConfig()
        self.adf_estimator = RollingADF(window=self.config.adf_window)
        self.ou_estimator = RollingOUHalfLife(
            window=self.config.ou_window,
            min_half_life=self.config.ou_min_half_life,
            max_half_life=self.config.ou_max_half_life,
        )

    def compute_feature_matrix(
        self, df: pd.DataFrame, symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """Extract all 21 stationary features from a clean market bar DataFrame."""
        features: Dict[str, pd.Series] = {}

        # 1. Price Returns
        c = df["Close"]
        features[FeatureName.LOG_RET_1.value] = log_returns(c, 1)
        features[FeatureName.LOG_RET_5.value] = log_returns(c, self.config.short_return_window)
        features[FeatureName.LOG_RET_20.value] = log_returns(c, self.config.medium_return_window)

        # 2. Volatility & Range
        h, l, o = df["High"], df["Low"], df["Open"]
        features[FeatureName.ATR_14_NORM.value] = normalized_atr(h, l, c, self.config.atr_window)
        features[FeatureName.VOLATILITY_20.value] = realized_volatility(
            c, self.config.volatility_window
        )
        features[FeatureName.PARKINSON_VOL_20.value] = parkinson_volatility(
            h, l, self.config.volatility_window
        )
        features[FeatureName.GARMAN_KLASS_VOL_20.value] = garman_klass_volatility(
            o, h, l, c, self.config.volatility_window
        )

        # 3. Volume & Microstructure
        v = df["Volume"]
        vwap = df["VWAP"] if "VWAP" in df.columns else self._compute_session_vwap(df)
        features[FeatureName.VOLUME_ZSCORE_LAGGED.value] = lagged_volume_zscore(
            v, self.config.volume_window, self.config.volume_lag
        )
        features[FeatureName.VWAP_DRIFT.value] = vwap_drift(c, vwap)

        # 4. Momentum & Oscillators
        features[FeatureName.RSI_14.value] = relative_strength_index(c, self.config.rsi_window)
        features[FeatureName.BOLLINGER_BAND_PCT_20.value] = bollinger_pct(
            c, self.config.bollinger_window, self.config.bollinger_std
        )
        features[FeatureName.MACD_DIFF_NORM.value] = macd_normalized(
            c, self.config.macd_fast, self.config.macd_slow, self.config.macd_signal
        )

        # 5. Tracking Spread & Cointegration
        self._compute_spread_features(df, features)

        # 6. Global Macro Drivers
        self._compute_macro_features(df, features)

        matrix = pd.DataFrame(features, index=df.index)
        return self._sanitize_matrix(matrix)

    def _compute_spread_features(
        self, df: pd.DataFrame, features: Dict[str, pd.Series]
    ) -> None:
        """Compute ETF tracking spread and stationarity dynamics."""
        c = df["Close"]
        if "Theoretical_Close" in df.columns:
            spread_bps = tracking_spread_bps(c, df["Theoretical_Close"])
        elif "Spread_INR" in df.columns:
            theo = np.maximum(1e-4, c - df["Spread_INR"])
            spread_bps = tracking_spread_bps(c, theo)
        else:
            spread_bps = pd.Series(0.0, index=df.index)

        # Spread z-score and momentum
        mean = spread_bps.rolling(self.config.spread_z_window, min_periods=5).mean()
        std = (
            spread_bps.rolling(self.config.spread_z_window, min_periods=5)
            .std()
            .replace(0.0, np.nan)
        )
        zscore = ((spread_bps - mean) / std).fillna(0.0).clip(-5.0, 5.0)
        delta5 = (spread_bps - spread_bps.shift(self.config.spread_delta_window)).fillna(0.0)

        features[FeatureName.SPREAD_BPS.value] = spread_bps
        features[FeatureName.SPREAD_ZSCORE_20.value] = zscore
        features[FeatureName.SPREAD_DELTA_5.value] = delta5
        features[FeatureName.ADF_STAT_60.value] = self.adf_estimator.compute(spread_bps)
        features[FeatureName.OU_HALF_LIFE_60.value] = self.ou_estimator.compute(spread_bps)

    def _compute_macro_features(
        self, df: pd.DataFrame, features: Dict[str, pd.Series]
    ) -> None:
        """Compute normalized global macro drivers with fallback."""
        # VIX Level
        vix = self._find_column(df, ["Macro_VIX", "^VIX", "VIX", "INDIAVIX"])
        if vix is not None:
            features[FeatureName.MACRO_VIX_LEVEL.value] = np.log(
                np.maximum(1.0, vix) / self.config.vix_baseline
            )
        else:
            features[FeatureName.MACRO_VIX_LEVEL.value] = pd.Series(0.0, index=df.index)

        # VIX Return (prefer causal inter-day return if present)
        vix_ret = self._find_column(df, ["Macro_VIX_RET", "VIX_RET", "Macro_VIX_RET_1D"])
        if vix_ret is not None:
            features[FeatureName.MACRO_VIX_RET_1.value] = vix_ret.fillna(0.0)
        elif vix is not None:
            features[FeatureName.MACRO_VIX_RET_1.value] = log_returns(vix, 1)
        else:
            features[FeatureName.MACRO_VIX_RET_1.value] = pd.Series(0.0, index=df.index)

        # USDINR Return
        usdinr_ret = self._find_column(df, ["Macro_USDINR_RET", "USDINR_RET", "Macro_USDINR_RET_1D"])
        if usdinr_ret is not None:
            features[FeatureName.MACRO_USDINR_RET_1.value] = usdinr_ret.fillna(0.0)
        else:
            usdinr = self._find_column(df, ["Macro_USDINR", "INR=X", "USDINR"])
            features[FeatureName.MACRO_USDINR_RET_1.value] = (
                log_returns(usdinr, 1) if usdinr is not None else pd.Series(0.0, index=df.index)
            )

        # S&P 500 Return
        sp500_ret = self._find_column(df, ["Macro_SP500_RET", "SP500_RET", "Macro_SP500_RET_1D"])
        if sp500_ret is not None:
            features[FeatureName.MACRO_SP500_RET_1.value] = sp500_ret.fillna(0.0)
        else:
            sp500 = self._find_column(df, ["Macro_SP500", "^GSPC", "SP500"])
            features[FeatureName.MACRO_SP500_RET_1.value] = (
                log_returns(sp500, 1) if sp500 is not None else pd.Series(0.0, index=df.index)
            )

    @staticmethod
    def _compute_session_vwap(df: pd.DataFrame) -> pd.Series:
        """Calculate intraday cumulative session VWAP resetting each calendar day."""
        if not isinstance(df.index, pd.DatetimeIndex) or "Volume" not in df.columns:
            return df["Close"]
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
        pv = typical_price * df["Volume"]
        session_dates = df.index.normalize()
        cum_pv = pv.groupby(session_dates).cumsum()
        cum_vol = df["Volume"].groupby(session_dates).cumsum()
        vwap = cum_pv / cum_vol.replace(0.0, np.nan)
        return vwap.fillna(df["Close"])

    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
        """Find first matching column from list of candidate names."""
        for name in candidates:
            if name in df.columns:
                return df[name]
        return None

    def _sanitize_matrix(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """Sanitize feature matrix to guarantee zero NaNs, Infs, and finite bounds."""
        sanitized = matrix.ffill().fillna(0.0)
        if self.config.clip_outliers:
            bound = self.config.max_abs_feature_val
            sanitized = sanitized.clip(-bound, bound)
        return sanitized.astype(np.float32)

    def compute_multi_asset_tensor(
        self, universe: Dict[str, pd.DataFrame]
    ) -> Tuple[torch.Tensor, List[str], List[str], pd.DatetimeIndex]:
        """Align multiple assets and construct a 3D float32 tensor of shape [B, T, K]."""
        if not universe:
            raise ValueError("Universe dictionary cannot be empty")

        symbols = sorted(universe.keys())
        matrices = {s: self.compute_feature_matrix(universe[s], symbol=s) for s in symbols}

        # Find common time index
        common_index = matrices[symbols[0]].index
        for s in symbols[1:]:
            common_index = common_index.intersection(matrices[s].index)

        if len(common_index) == 0:
            raise ValueError("No overlapping timestamps across asset universe")

        # Reindex all matrices to shared continuous index
        feature_names = FeatureRegistry.all_features()
        aligned_arrays: List[np.ndarray] = []

        for s in symbols:
            aligned_df = matrices[s].loc[common_index, feature_names]
            aligned_arrays.append(aligned_df.to_numpy(dtype=np.float32))

        # Stack into [B, T, K]
        stacked = np.stack(aligned_arrays, axis=0)
        tensor = torch.from_numpy(stacked)
        return tensor, symbols, feature_names, common_index
