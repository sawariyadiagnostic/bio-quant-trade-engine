"""Universal Market Data Loader.
Dual-source data ingestion (yfinance remote API and local Parquet/CSV)
with session gatekeeping, physical invariant validation, and contract normalization.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from ravana.data.ingestion.config import IngestionConfig, ETF_BENCHMARK_MAP
from ravana.data.ingestion.session_gatekeeper import NSESessionGatekeeper
from ravana.data.ingestion.dataset import IngestedDataset

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


class CandlestickSanityValidator:
    """Enforces physical and mathematical invariants on OHLCV candles."""

    @staticmethod
    def sanitize(df: pd.DataFrame) -> pd.DataFrame:
        """Repair or clamp physically impossible candles (High < Low, Open <= 0, etc.)."""
        if df.empty:
            return df
        out = df.copy()

        # 1. Repair non-positive Open/Close/High/Low prices
        if "Close" in out.columns:
            out["Close"] = out["Close"].replace(0.0, np.nan).ffill().bfill()
        if "Open" in out.columns and "Close" in out.columns:
            out["Open"] = np.where(out["Open"] <= 0.0, out["Close"], out["Open"])
        if "Low" in out.columns and "Close" in out.columns:
            out["Low"] = np.where(out["Low"] <= 0.0, out["Close"], out["Low"])
        if "High" in out.columns and "Close" in out.columns:
            out["High"] = np.where(out["High"] <= 0.0, out["Close"], out["High"])

        # 2. Geometric bounds: High >= max(Open, Close), Low <= min(Open, Close)
        if all(col in out.columns for col in ["Open", "High", "Low", "Close"]):
            max_oc = np.maximum(out["Open"], out["Close"])
            min_oc = np.minimum(out["Open"], out["Close"])
            out["High"] = np.maximum(out["High"], max_oc)
            out["Low"] = np.minimum(out["Low"], min_oc)

        # 3. Volume non-negativity
        if "Volume" in out.columns:
            out["Volume"] = np.maximum(out["Volume"], 0.0)

        # 4. Proportional OHLC scaling across splits if Close_Adj is available
        if "Close_Adj" in out.columns and "Close" in out.columns:
            ratio = np.where(out["Close"] > 0, out["Close_Adj"] / out["Close"], 1.0)
            out["Open_Adj"] = out["Open"] * ratio
            out["High_Adj"] = out["High"] * ratio
            out["Low_Adj"] = out["Low"] * ratio
        return out


class MarketDataLoader:
    """Institutional data loader for live feeds and local historical lakes."""

    def __init__(self, gatekeeper: Optional[NSESessionGatekeeper] = None) -> None:
        self.gatekeeper = gatekeeper or NSESessionGatekeeper()

    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize DataFrame columns to standard TitleCase names and cast float64."""
        out = df.copy()
        if isinstance(out.columns, pd.MultiIndex):
            # Flatten multi-level headers while guarding against duplicate collision
            flattened = []
            for col in out.columns:
                c = col[0] if isinstance(col, tuple) else col
                flattened.append(c)
            out.columns = flattened

        rename_dict = {}
        for col in out.columns:
            cleaned_col = str(col).strip().lower()
            if cleaned_col in STANDARD_COL_MAP:
                rename_dict[col] = STANDARD_COL_MAP[cleaned_col]
        out = out.rename(columns=rename_dict)

        # Eliminate duplicate column names if multi-ticker was downloaded
        out = out.loc[:, ~out.columns.duplicated()]

        required = ["Open", "High", "Low", "Close", "Volume", "Close_Adj"]
        for col in required:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").astype(np.float64)
        return CandlestickSanityValidator.sanitize(out)

    def clean_index_and_chronology(
        self, df: pd.DataFrame, target_tz: Optional[str] = None
    ) -> pd.DataFrame:
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

        out = self.gatekeeper.standardize_timezone(out, target_tz=target_tz)
        out = out[~out.index.duplicated(keep="last")]
        return out.sort_index()

    def _detect_is_intraday(self, df: pd.DataFrame) -> bool:
        """Detect whether a DataFrame's index represents intraday frequency."""
        if len(df) < 2 or not isinstance(df.index, pd.DatetimeIndex):
            return False
        deltas = df.index[1:min(10, len(df))] - df.index[0:min(9, len(df) - 1)]
        return any(d < pd.Timedelta(days=1) for d in deltas)

    def load_local(
        self,
        file_path: str,
        is_intraday: Optional[bool] = None,
        special_whitelist: Optional[List[str]] = None,
        target_tz: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load market bars from local CSV or Parquet file with gatekeeping."""
        if file_path.endswith(".parquet"):
            raw = pd.read_parquet(file_path)
        elif file_path.endswith(".csv"):
            raw = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        std = self.standardize_columns(raw)
        indexed = self.clean_index_and_chronology(std, target_tz=target_tz)
        intraday_flag = is_intraday if is_intraday is not None else self._detect_is_intraday(indexed)
        return self.gatekeeper.filter_trading_hours(
            indexed,
            is_intraday=intraday_flag,
            special_whitelist=special_whitelist,
            target_tz=target_tz,
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
        indexed = self.clean_index_and_chronology(std, target_tz=config.timezone)
        should_filter_weekends = not any(config.interval.lower().endswith(u) for u in ["mo", "wk", "y", "month", "year"])
        return self.gatekeeper.filter_trading_hours(
            indexed,
            is_intraday=config.is_intraday(),
            filter_weekends_flag=should_filter_weekends,
            special_whitelist=config.special_session_dates,
            target_tz=config.timezone,
            market_open=config.market_open_time,
            market_close=config.market_close_time,
        )

    def fetch_universe(
        self,
        tickers: List[str],
        interval: str = "1d",
        period: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch and clean an entire multi-asset universe with automatic exchange profiles."""
        universe_data: Dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            cfg = IngestionConfig.create(ticker=ticker, interval=interval, period=period)
            df = self.fetch_yfinance(cfg)
            if not df.empty:
                universe_data[ticker] = df
        return universe_data

    def ingest_etf(
        self,
        ticker: str = "BANKBEES.NS",
        interval: str = "5m",
        period: Optional[str] = "60d",
        include_macro: bool = False,
        local_fallback_path: Optional[str] = None,
    ) -> IngestedDataset:
        """Ingest primary ETF paired with its underlying benchmark index and macro drivers."""
        cfg_etf = IngestionConfig.create(ticker=ticker, interval=interval, period=period)
        df_etf = self.ingest(cfg_etf, local_fallback_path=local_fallback_path)

        # Pair with underlying index (e.g. ^NSEBANK for BANKBEES.NS)
        benchmark_ticker = ETF_BENCHMARK_MAP.get(ticker.upper())
        df_bench = None
        if benchmark_ticker:
            cfg_bench = IngestionConfig.create(ticker=benchmark_ticker, interval=interval, period=period)
            df_bench = self.ingest(cfg_bench)

        macro_dfs: Dict[str, pd.DataFrame] = {}
        if include_macro:
            macro_tickers = ["^GSPC", "^VIX", "INR=X"]
            macro_dfs = self.fetch_universe(macro_tickers, interval="1d", period=period)

        metadata = {
            "source": "yfinance" if not df_etf.empty else "local",
            "is_intraday": cfg_etf.is_intraday(),
        }

        return IngestedDataset(
            ticker=ticker,
            interval=interval,
            exchange=cfg_etf.exchange,
            data=df_etf,
            benchmark_ticker=benchmark_ticker,
            benchmark_data=df_bench,
            macro_data=macro_dfs,
            metadata=metadata,
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
