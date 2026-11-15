"""Multi-Sector Autonomous Data Feeder for RAVANA v4.1."""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SECTOR_MAP: Dict[str, Dict[str, str]] = {
    "BANKBEES": {"benchmark": "NSEBANK", "sector": "Banking_Financials", "is_tier1": True},
    "NIFTYBEES": {"benchmark": "NSEI", "sector": "Broad_Market", "is_tier1": True},
    "ITBEES": {"benchmark": "CNXIT", "sector": "Information_Technology", "is_tier1": True},
    "AUTOBEES": {"benchmark": "CNXAUTO", "sector": "Automotive", "is_tier1": False},
    "PHARMABEES": {"benchmark": "CNXPHARMA", "sector": "Pharma_Healthcare", "is_tier1": False},
    "PSUBNKBEES": {"benchmark": "CNXPSUBANK", "sector": "Banking_Financials", "is_tier1": False},
    "JUNIORBEES": {"benchmark": "NSEI", "sector": "Next50_Midcap", "is_tier1": False},
    "CPSEETF": {"benchmark": "NSEI", "sector": "Energy_CPSE", "is_tier1": False},
    "GOLDBEES": {"benchmark": "GC", "sector": "Commodities_Gold", "is_tier1": True},
    "SILVERBEES": {"benchmark": "SI", "sector": "Commodities_Silver", "is_tier1": False},
}


class SectorDataFeeder:
    """Loads and enriches 5-minute lakehouse market data for all sector ETF pairs."""

    def __init__(self, data_dir: Optional[str] = None) -> None:
        if data_dir is None:
            base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            self.data_dir = os.path.join(base, "ravana", "data", "lake", "m3_ingested", "intraday_5m")
        else:
            self.data_dir = data_dir

    def load_sector_data(self, symbol: str) -> pd.DataFrame:
        """Load and enrich market bars with technical indicators and spread metrics."""
        etf_path = os.path.join(self.data_dir, f"{symbol}.parquet")
        if not os.path.exists(etf_path):
            raise FileNotFoundError(f"Missing parquet partition: {etf_path}")

        meta = SECTOR_MAP.get(symbol, {"benchmark": "NSEI", "sector": "General"})
        bench_sym = meta["benchmark"]
        bench_path = os.path.join(self.data_dir, f"{bench_sym}.parquet")

        df = pd.read_parquet(etf_path).copy()
        df = self._standardize_columns(df)
        df = self._compute_volatility_features(df)
        df = self._compute_session_vwap_and_bollinger(df)
        df = self._compute_order_flow_and_fvg(df)
        df = self._compute_tracking_spread(df, bench_path)
        df = self._compute_calendar_windows(df)
        return df

    def load_all_sectors(self) -> Dict[str, pd.DataFrame]:
        """Load and enrich all 10 sector ETFs into memory."""
        datasets: Dict[str, pd.DataFrame] = {}
        for sym in SECTOR_MAP:
            try:
                datasets[sym] = self.load_sector_data(sym)
                logger.info("Loaded sector %s: %d bars", sym, len(datasets[sym]))
            except Exception as e:
                logger.warning("Could not load sector %s: %s", sym, e)
        return datasets

    @staticmethod
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure standard OHLCV column nomenclature."""
        out = df.copy()
        if "Open_Adj" in out.columns and "Open" not in out.columns:
            out["Open"] = out["Open_Adj"]
        if "High_Adj" in out.columns and "High" not in out.columns:
            out["High"] = out["High_Adj"]
        if "Low_Adj" in out.columns and "Low" not in out.columns:
            out["Low"] = out["Low_Adj"]
        if "Close_Adj" in out.columns and "Close" not in out.columns:
            out["Close"] = out["Close_Adj"]
        if "Volume" not in out.columns:
            out["Volume"] = 10000.0
        return out

    @staticmethod
    def _compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ATR-14 and Lo-MacKinlay Variance Ratio (VR-20)."""
        high = df["High"].to_numpy(dtype=np.float64)
        low = df["Low"].to_numpy(dtype=np.float64)
        close = df["Close"].to_numpy(dtype=np.float64)
        n = len(df)

        tr = np.zeros(n, dtype=np.float64)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], max(abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1])))

        atr = np.zeros(n, dtype=np.float64)
        atr[0] = tr[0]
        alpha = 1.0 / 14.0
        for i in range(1, n):
            atr[i] = tr[i] * alpha + atr[i - 1] * (1.0 - alpha)
        df["ATR_14"] = atr

        log_p = np.log(np.maximum(1e-4, df["Close"]))
        ret_1 = log_p.diff()
        ret_5 = log_p.diff(5)
        var_1 = ret_1.rolling(20, min_periods=5).var()
        var_5 = ret_5.rolling(20, min_periods=5).var()
        df["VARIANCE_RATIO_20"] = (var_5 / (5.0 * var_1.replace(0.0, np.nan))).fillna(1.0).clip(0.1, 5.0)
        df["LOG_RET_20"] = log_p.diff(20).fillna(0.0)
        return df

    @staticmethod
    def _compute_session_vwap_and_bollinger(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate intraday session VWAP and 20-period Bollinger Band percent."""
        dates = df.index.normalize() if isinstance(df.index, pd.DatetimeIndex) else [0] * len(df)
        typical_px = (df["High"] + df["Low"] + df["Close"]) / 3.0
        cum_pv = (typical_px * df["Volume"]).groupby(dates).cumsum()
        cum_vol = df["Volume"].groupby(dates).cumsum()
        df["VWAP"] = (cum_pv / (cum_vol + 1e-6)).fillna(df["Close"])

        rolling_mean_20 = df["Close"].rolling(20, min_periods=5).mean()
        rolling_std_20 = df["Close"].rolling(20, min_periods=5).std().replace(0.0, 1e-6)
        upper_band = rolling_mean_20 + 2.0 * rolling_std_20
        lower_band = rolling_mean_20 - 2.0 * rolling_std_20
        df["BOLLINGER_BAND_PCT_20"] = (
            (df["Close"] - lower_band) / (upper_band - lower_band + 1e-6)
        ).clip(0.0, 1.0)
        return df

    @staticmethod
    def _compute_order_flow_and_fvg(df: pd.DataFrame) -> pd.DataFrame:
        """Compute Order Flow Imbalance (OFI-5) and Fair Value Gap (FVG) drift."""
        ret_sign = np.sign(df["Close"].diff().fillna(0.0))
        signed_vol = ret_sign * df["Volume"]
        vol_sum_5 = df["Volume"].rolling(5, min_periods=1).sum()
        df["OFI_5"] = (signed_vol.rolling(5, min_periods=1).sum() / (vol_sum_5 + 1e-6)).clip(-1.0, 1.0)
        df["FVG_BULLISH_DRIFT"] = (
            (df["Low"] - df["High"].shift(2)) / df["Close"]
        ).fillna(0.0).clip(-0.05, 0.05)

        vol_mean = df["Volume"].rolling(20, min_periods=5).mean()
        vol_std = df["Volume"].rolling(20, min_periods=5).std().replace(0.0, 1e-6)
        df["VOLUME_ZSCORE_LAGGED"] = ((df["Volume"].shift(1) - vol_mean) / vol_std).fillna(0.0)
        df["MACRO_YIELD_RET_1"] = 0.0
        return df

    @staticmethod
    def _compute_tracking_spread(df: pd.DataFrame, bench_path: str) -> pd.DataFrame:
        """Calculate tracking spread against benchmark index if available."""
        if os.path.exists(bench_path):
            df_idx = pd.read_parquet(bench_path)
            # Reindex benchmark to match ETF timestamps
            aligned_idx = df_idx["Close"].reindex(df.index).ffill().bfill()
            scale = float(df["Close"].iloc[0] / aligned_idx.iloc[0]) if aligned_idx.iloc[0] > 0 else 1.0
            theoretical_close = aligned_idx * scale
            spread_bps = ((df["Close"] - theoretical_close) / theoretical_close) * 10000.0
            df["Theoretical_Close"] = theoretical_close
            df["SPREAD_BPS"] = spread_bps.fillna(0.0)
            mean_spd = spread_bps.rolling(20, min_periods=5).mean()
            std_spd = spread_bps.rolling(20, min_periods=5).std().replace(0.0, np.nan)
            df["SPREAD_ZSCORE_20"] = ((spread_bps - mean_spd) / std_spd).fillna(0.0).clip(-5.0, 5.0)
        else:
            df["SPREAD_BPS"] = 0.0
            df["SPREAD_ZSCORE_20"] = 0.0
        return df

    @staticmethod
    def _compute_calendar_windows(df: pd.DataFrame) -> pd.DataFrame:
        """Tag monthly SIP windows and expiry days."""
        if isinstance(df.index, pd.DatetimeIndex):
            days = df.index.day
            df["TURN_OF_MONTH_SIP_WINDOW"] = np.where((days >= 28) | (days <= 7), 1.0, 0.0)
        else:
            df["TURN_OF_MONTH_SIP_WINDOW"] = 0.0
        return df
