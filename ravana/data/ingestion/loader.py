"""Universal Market Data Loader.
Dual-source data ingestion (yfinance remote API and local Parquet/CSV)
with session gatekeeping and contract normalization.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from ravana.data.ingestion.config import IngestionConfig
from ravana.data.ingestion.session_gatekeeper import NSESessionGatekeeper

logger = logging.getLogger(__name__)

STANDARD_COL_MAP: Dict[str, str] = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
    "adj close": "Close_Adj",
    "close_adj": "Close_Adj",
    "adj_close": "Close_Adj",
    "close_raw": "Close_Raw",
    "dividends": "Dividends",
    "stock splits": "Stock Splits",
}


class MarketDataLoader:
    """Institutional data loader for live feeds and local historical lakes."""

    def __init__(self, gatekeeper: Optional[NSESessionGatekeeper] = None) -> None:
        self.gatekeeper = gatekeeper or NSESessionGatekeeper()

    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize DataFrame columns to standard TitleCase names and cast float64."""
        out = df.copy()
        if isinstance(out.columns, pd.MultiIndex):
            # Flatten yfinance MultiIndex (e.g. ('Close', 'BANKBEES.NS') -> 'Close')
            out.columns = [col[0] for col in out.columns]

        rename_dict = {}
        for col in out.columns:
            cleaned_col = str(col).strip().lower()
            if cleaned_col in STANDARD_COL_MAP:
                rename_dict[col] = STANDARD_COL_MAP[cleaned_col]
        out = out.rename(columns=rename_dict)

        required = ["Open", "High", "Low", "Close", "Volume"]
        for col in required:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").astype(np.float64)
        return out

    def clean_index_and_chronology(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort DatetimeIndex chronologically and drop duplicate timestamps."""
        if df.empty:
            return df
        out = df.copy()
        if not isinstance(out.index, pd.DatetimeIndex):
            date_col = next(
                (c for c in ["date", "datetime", "timestamp", "Date", "Datetime"] if c in out.columns),
                None,
            )
            if date_col is not None:
                out = out.set_index(date_col)
            out.index = pd.to_datetime(out.index)

        out = self.gatekeeper.standardize_timezone(out)
        out = out[~out.index.duplicated(keep="last")]
        return out.sort_index()

    def load_local(
        self,
        file_path: str,
        is_intraday: bool = False,
        special_whitelist: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Load market bars from local CSV or Parquet file with gatekeeping."""
        if file_path.endswith(".parquet"):
            raw = pd.read_parquet(file_path)
        elif file_path.endswith(".csv"):
            raw = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        std = self.standardize_columns(raw)
        indexed = self.clean_index_and_chronology(std)
        return self.gatekeeper.filter_trading_hours(
            indexed, is_intraday=is_intraday, special_whitelist=special_whitelist
        )

    def fetch_yfinance(self, config: IngestionConfig) -> pd.DataFrame:
        """Fetch remote market bars from Yahoo Finance with strict gatekeeping."""
        import yfinance as yf

        kwargs = {"interval": config.interval, "progress": False, "auto_adjust": False}
        if config.period is not None:
            kwargs["period"] = config.period
        else:
            kwargs["start"] = config.start_date
            kwargs["end"] = config.end_date

        raw = yf.download(config.ticker, **kwargs)
        if raw.empty:
            logger.warning("Empty response received from yfinance for %s", config.ticker)
            return pd.DataFrame()

        std = self.standardize_columns(raw)
        indexed = self.clean_index_and_chronology(std)
        return self.gatekeeper.filter_trading_hours(
            indexed,
            is_intraday=config.is_intraday(),
            special_whitelist=config.special_session_dates,
        )

    def ingest(
        self, config: IngestionConfig, local_fallback_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Unified ingestion pipeline: attempts live fetch with local fallback."""
        try:
            df = self.fetch_yfinance(config)
            if not df.empty:
                return df
        except Exception as e:
            logger.error("Remote fetch failed for %s: %s", config.ticker, e)

        if local_fallback_path is not None:
            logger.info("Falling back to local data: %s", local_fallback_path)
            return self.load_local(
                local_fallback_path,
                is_intraday=config.is_intraday(),
                special_whitelist=config.special_session_dates,
            )
        return pd.DataFrame()
