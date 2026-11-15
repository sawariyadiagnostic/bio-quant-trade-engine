"""Feature Matrix Engine for Muscle 7.

Computes 36 institutional stationary features across price, volatility,
volume, tracking spread cointegration, macro/bond drivers, intraday
session-clock windows (Morning 09:15-10:30 Overflow, Afternoon 14:00-14:40 Bond Window,
and Opening Range Breakout Drift), and advanced information theory / microstructure
dynamics (Variance Ratio, OFI, FVG drift, Stop Sweep Reclaim, VIX Crush Relief). Generates 2D matrices
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
        """Extract all 36 stationary features from a clean market bar DataFrame."""
        features: Dict[str, pd.Series] = {}

        # 1. Price Returns & Scale-Invariant Volatility
        atr_norm = self._compute_price_and_vol_features(df, features)

        # 2. Volume & Microstructure
        self._compute_volume_microstructure_features(df, features)

        # 3. Momentum & Technical Oscillators
        self._compute_momentum_features(df, features)

        # 4. Tracking Spread & Cointegration
        self._compute_spread_features(df, features)

        # 5. Global Macro & Bond Yield Drivers
        self._compute_macro_features(df, features)

        # 6. Intraday Session Clock & Opening Range Windows
        self._compute_intraday_clock_features(df, atr_norm, features)

        # 7. Cross-Market, Index Lead-Lag & Institutional Calendar Dependencies
        self._compute_dependency_features(df, features)

        # 8. Information Theory & Microstructure Dynamics (New 5 Features)
        self._compute_microstructure_information_features(df, atr_norm, features)

        matrix = pd.DataFrame(features, index=df.index)
        active = [col for col in self.config.active_features if col in matrix.columns]
        return self._sanitize_matrix(matrix[active])

    def _compute_price_and_vol_features(
        self, df: pd.DataFrame, features: Dict[str, pd.Series]
    ) -> pd.Series:
        """Compute price returns and scale-invariant volatility features."""
        c = df["Close"]
        features[FeatureName.LOG_RET_1.value] = log_returns(c, 1)
        features[FeatureName.LOG_RET_5.value] = log_returns(c, self.config.short_return_window)
        features[FeatureName.LOG_RET_20.value] = log_returns(c, self.config.medium_return_window)

        h, l, o = df["High"], df["Low"], df["Open"]
        atr_norm = normalized_atr(h, l, c, self.config.atr_window)
        features[FeatureName.ATR_14_NORM.value] = atr_norm
        features[FeatureName.VOLATILITY_20.value] = realized_volatility(
            c, self.config.volatility_window
        )
        features[FeatureName.PARKINSON_VOL_20.value] = parkinson_volatility(
            h, l, self.config.volatility_window
        )
        features[FeatureName.GARMAN_KLASS_VOL_20.value] = garman_klass_volatility(
            o, h, l, c, self.config.volatility_window
        )
        return atr_norm

    def _compute_volume_microstructure_features(
        self, df: pd.DataFrame, features: Dict[str, pd.Series]
    ) -> None:
        """Compute volume and intraday VWAP microstructure features."""
        c = df["Close"]
        v = df["Volume"] if "Volume" in df.columns else pd.Series(1.0, index=df.index)
        vwap = df["VWAP"] if "VWAP" in df.columns else self._compute_session_vwap(df)
        features[FeatureName.VOLUME_ZSCORE_LAGGED.value] = lagged_volume_zscore(
            v, self.config.volume_window, self.config.volume_lag
        )
        features[FeatureName.VWAP_DRIFT.value] = vwap_drift(c, vwap)

    def _compute_momentum_features(
        self, df: pd.DataFrame, features: Dict[str, pd.Series]
    ) -> None:
        """Compute normalized momentum oscillators."""
        c = df["Close"]
        features[FeatureName.RSI_14.value] = relative_strength_index(c, self.config.rsi_window)
        features[FeatureName.BOLLINGER_BAND_PCT_20.value] = bollinger_pct(
            c, self.config.bollinger_window, self.config.bollinger_std
        )
        features[FeatureName.MACD_DIFF_NORM.value] = macd_normalized(
            c, self.config.macd_fast, self.config.macd_slow, self.config.macd_signal
        )

    def _compute_microstructure_information_features(
        self,
        df: pd.DataFrame,
        atr_norm: pd.Series,
        features: Dict[str, pd.Series],
    ) -> None:
        """Compute Lo-MacKinlay VR, OFI, FVG drift, stop sweep, and VIX relief."""
        idx = df.index
        c = df["Close"].astype(np.float64)
        h = df["High"].astype(np.float64)
        l = df["Low"].astype(np.float64)
        v = df["Volume"].astype(np.float64) if "Volume" in df.columns else pd.Series(1.0, index=idx)

        # a) VARIANCE_RATIO_20
        features[FeatureName.VARIANCE_RATIO_20.value] = self._calc_variance_ratio(
            c, self.config.variance_ratio_window, self.config.short_return_window
        )
        # b) ORDER_FLOW_IMBALANCE_5
        ofi_5 = self._calc_order_flow_imbalance(c, h, l, v, self.config.order_flow_window)
        features[FeatureName.ORDER_FLOW_IMBALANCE_5.value] = ofi_5
        # c) FVG_BULLISH_DRIFT
        features[FeatureName.FVG_BULLISH_DRIFT.value] = self._calc_fvg_drift(
            h, l, c, self.config.atr_window, idx
        )
        # d) STOP_SWEEP_RECLAIM
        features[FeatureName.STOP_SWEEP_RECLAIM.value] = self._calc_stop_sweep_reclaim(
            l, c, ofi_5, self.config.sweep_low_window, idx
        )
        # e) VIX_CRUSH_RELIEF
        vix = self._find_column(df, ["Macro_VIX", "^VIX", "VIX", "INDIAVIX"])
        features[FeatureName.VIX_CRUSH_RELIEF.value] = self._calc_vix_crush(
            vix, self.config.vix_crush_threshold, self.config.vix_crush_return_threshold, idx
        )

    @staticmethod
    def _calc_variance_ratio(c: pd.Series, window: int, short_win: int) -> pd.Series:
        """Compute Lo-MacKinlay Variance Ratio: r5_var / (5 * r1_var + 1e-6)."""
        r1 = log_returns(c, 1)
        r5 = log_returns(c, short_win)
        var1 = r1.rolling(window).var()
        var5 = r5.rolling(window).var()
        vr = var5 / (5.0 * var1 + 1e-6)
        flat_mask = (var1.fillna(0.0) < 1e-12) & (var5.fillna(0.0) < 1e-12)
        return vr.where(~flat_mask, 1.0).fillna(1.0).clip(0.10, 3.00)

    @staticmethod
    def _calc_order_flow_imbalance(
        c: pd.Series, h: pd.Series, l: pd.Series, v: pd.Series, window: int
    ) -> pd.Series:
        """Compute volume-weighted Close Location Value over window bars."""
        clv = (2.0 * c - h - l) / (h - l + 1e-5)
        vol_clv_sum = (v * clv).rolling(window).sum()
        vol_sum = v.rolling(window).sum()
        return (vol_clv_sum / (vol_sum + 1e-5)).fillna(0.0).clip(-1.0, 1.0)

    @staticmethod
    def _calc_fvg_drift(
        h: pd.Series, l: pd.Series, c: pd.Series, window: int, idx: pd.Index
    ) -> pd.Series:
        """Compute 3-bar Fair Value Gap bullish drift normalized by ATR."""
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(window, min_periods=1).mean()
        fvg = np.maximum(0.0, (l - h.shift(2)) / (atr + 1e-5))
        return pd.Series(fvg, index=idx).fillna(0.0).clip(0.0, 5.0)

    @staticmethod
    def _calc_stop_sweep_reclaim(
        l: pd.Series, c: pd.Series, ofi: pd.Series, window: int, idx: pd.Index
    ) -> pd.Series:
        """Compute 10-bar swing low sweep and reclaim confirmation."""
        prior_low = l.shift(1).rolling(window).min()
        cond_sweep = (l < prior_low) & (c > prior_low) & (ofi > 0.15)
        return pd.Series(np.where(cond_sweep, 1.0, 0.0), index=idx).fillna(0.0)

    @staticmethod
    def _calc_vix_crush(
        vix: Optional[pd.Series], threshold: float, ret_threshold: float, idx: pd.Index
    ) -> pd.Series:
        """Compute VIX crush relief indicator."""
        if vix is None:
            return pd.Series(0.0, index=idx)
        vix_ret = log_returns(vix, 1)
        cond = (vix > threshold) & (vix_ret < ret_threshold)
        return pd.Series(np.where(cond, 1.0, 0.0), index=idx).fillna(0.0)

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
        idx = df.index
        # VIX Level & Return
        vix = self._find_column(df, ["Macro_VIX", "^VIX", "VIX", "INDIAVIX"])
        features[FeatureName.MACRO_VIX_LEVEL.value] = (
            np.log(np.maximum(1.0, vix) / self.config.vix_baseline)
            if vix is not None
            else pd.Series(0.0, index=idx)
        )
        vix_ret = self._find_column(df, ["Macro_VIX_RET", "VIX_RET", "Macro_VIX_RET_1D"])
        features[FeatureName.MACRO_VIX_RET_1.value] = (
            vix_ret.fillna(0.0)
            if vix_ret is not None
            else (log_returns(vix, 1) if vix is not None else pd.Series(0.0, index=idx))
        )
        # USDINR, S&P 500, Bond Yield
        macro_triplets = [
            (FeatureName.MACRO_USDINR_RET_1.value, ["Macro_USDINR_RET", "USDINR_RET", "Macro_USDINR_RET_1D"], ["Macro_USDINR", "INR=X", "USDINR"]),
            (FeatureName.MACRO_SP500_RET_1.value, ["Macro_SP500_RET", "SP500_RET", "Macro_SP500_RET_1D"], ["Macro_SP500", "^GSPC", "SP500"]),
            (FeatureName.MACRO_YIELD_RET_1.value, ["Macro_YIELD_RET", "Macro_TNX_RET", "TNX_RET", "US10Y_RET"], ["Macro_TNX", "^TNX", "US10Y", "Macro_YIELD"]),
        ]
        for name, ret_cols, raw_cols in macro_triplets:
            ret = self._find_column(df, ret_cols)
            if ret is not None:
                features[name] = ret.fillna(0.0)
            else:
                raw = self._find_column(df, raw_cols)
                features[name] = log_returns(raw, 1) if raw is not None else pd.Series(0.0, index=idx)

    def _compute_intraday_clock_features(
        self,
        df: pd.DataFrame,
        atr_norm: pd.Series,
        features: Dict[str, pd.Series],
    ) -> None:
        """Compute stationary session-clock windows and causal Opening Range breakout drift."""
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            for k in (
                FeatureName.SESSION_CLOCK_NORM.value,
                FeatureName.MORNING_OVERFLOW_WINDOW.value,
                FeatureName.AFTERNOON_BOND_WINDOW.value,
                FeatureName.OPENING_RANGE_DRIFT.value,
            ):
                features[k] = pd.Series(0.0, index=idx)
            return

        mins = idx.hour * 60.0 + idx.minute
        clock_norm = np.clip((mins - 740.0) / 185.0, -1.0, 1.0)
        if (mins == 0).all():
            clock_norm = np.zeros(len(idx), dtype=np.float64)

        morning = np.where((mins >= 555.0) & (mins <= 630.0), 1.0, 0.0)
        afternoon = np.where(
            (mins >= 835.0) & (mins <= 880.0),
            1.0,
            np.where((mins >= 895.0) & (mins <= 915.0), -1.0, 0.0),
        )

        features[FeatureName.SESSION_CLOCK_NORM.value] = pd.Series(clock_norm, index=idx)
        features[FeatureName.MORNING_OVERFLOW_WINDOW.value] = pd.Series(morning, index=idx)
        features[FeatureName.AFTERNOON_BOND_WINDOW.value] = pd.Series(afternoon, index=idx)
        features[FeatureName.OPENING_RANGE_DRIFT.value] = self._calc_opening_range_drift(
            df, atr_norm, idx
        )

    def _calc_opening_range_drift(
        self, df: pd.DataFrame, atr_norm: pd.Series, idx: pd.DatetimeIndex
    ) -> pd.Series:
        """Compute ATR-normalized breakout drift from 09:15-09:45 Opening Range midpoint."""
        session_dates = idx.normalize()
        bar_num = df.groupby(session_dates).cumcount()
        or_bars = max(1, self.config.opening_range_bars)

        or_h = df["High"].where(bar_num < or_bars, np.nan).groupby(session_dates).ffill()
        or_l = df["Low"].where(bar_num < or_bars, np.nan).groupby(session_dates).ffill()
        exp_h = df["High"].groupby(session_dates).cummax()
        exp_l = df["Low"].groupby(session_dates).cummin()

        or_mid = 0.5 * (or_h.fillna(exp_h) + or_l.fillna(exp_l))
        raw_drift = (df["Close"] - or_mid) / np.maximum(1e-4, df["Close"])
        safe_atr = np.maximum(1e-4, atr_norm)
        return (raw_drift / safe_atr).fillna(0.0).clip(-5.0, 5.0)

    def _compute_dependency_features(
        self, df: pd.DataFrame, features: Dict[str, pd.Series]
    ) -> None:
        idx = df.index
        # 1. Brent Crude
        crude_ret = self._find_column(df, ["Macro_CRUDE_RET", "Macro_BRENT_RET", "BZ=F_RET"])
        if crude_ret is not None:
            features[FeatureName.MACRO_CRUDE_RET_1.value] = crude_ret.fillna(0.0)
        else:
            crude = self._find_column(df, ["Macro_CRUDE", "Macro_BRENT", "BZ=F", "BRENT"])
            features[FeatureName.MACRO_CRUDE_RET_1.value] = (
                log_returns(crude, 1) if crude is not None else pd.Series(0.0, index=idx)
            )

        # 2. NASDAQ
        nasdaq_ret = self._find_column(df, ["Macro_NASDAQ_RET", "Macro_IXIC_RET", "IXIC_RET"])
        if nasdaq_ret is not None:
            features[FeatureName.MACRO_NASDAQ_RET_1.value] = nasdaq_ret.fillna(0.0)
        else:
            nasdaq = self._find_column(df, ["Macro_NASDAQ", "Macro_IXIC", "^IXIC", "IXIC"])
            features[FeatureName.MACRO_NASDAQ_RET_1.value] = (
                log_returns(nasdaq, 1) if nasdaq is not None else pd.Series(0.0, index=idx)
            )

        # 3. Heavyweight Constituent / Index Lead-Lag Drift
        c = df["Close"]
        theo_col = self._find_column(
            df, ["Theoretical_Close", "Theoretical_Price", "Index_Close", "Benchmark_Close"]
        )
        if theo_col is not None:
            idx_ret_3 = log_returns(theo_col, 3)
            etf_ret_3 = log_returns(c, 3)
            features[FeatureName.INDEX_LEAD_LAG_DRIFT.value] = (idx_ret_3 - etf_ret_3).clip(-0.05, 0.05)
        else:
            features[FeatureName.INDEX_LEAD_LAG_DRIFT.value] = pd.Series(0.0, index=idx)

        # 4. Turn-of-Month SIP Window & Thursday Expiry Gamma Window (Timezone-Safe!)
        self._compute_calendar_regimes(idx, features)

    @staticmethod
    def _compute_calendar_regimes(idx: pd.Index, features: Dict[str, pd.Series]) -> None:
        """Tag Turn-of-Month SIP inflow and Thursday weekly expiry gamma windows."""
        if isinstance(idx, pd.DatetimeIndex):
            ist_idx = idx.tz_convert("Asia/Kolkata") if idx.tz is not None else idx
            day = ist_idx.day
            sip = np.where((day <= 7) | (day >= 28), 1.0, 0.0)
            features[FeatureName.TURN_OF_MONTH_SIP_WINDOW.value] = pd.Series(sip, index=idx)

            mins = ist_idx.hour * 60.0 + ist_idx.minute
            is_thu = ist_idx.dayofweek == 3
            gamma = np.where(is_thu & (mins >= 855.0) & (mins <= 920.0), 1.0, 0.0)
            features[FeatureName.EXPIRY_GAMMA_WINDOW.value] = pd.Series(gamma, index=idx)
        else:
            features[FeatureName.TURN_OF_MONTH_SIP_WINDOW.value] = pd.Series(0.0, index=idx)
            features[FeatureName.EXPIRY_GAMMA_WINDOW.value] = pd.Series(0.0, index=idx)

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
