"""
Resilient Dual-Stream NSE Data Loader with Sector Batch Chunking & Parquet Lake.
Hardware-friendly: throttles execution, pauses between network calls, caps memory.
"""
import os
import time
import gc
import pandas as pd
import numpy as np
import yfinance as yf
from loguru import logger

# 12-Sector Diversified Liquid Universe (100+ Equities)
NSE_SECTORS = {
    "Banking_Financials": [
        "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS",
        "BANKBARODA.NS", "INDUSINDBK.NS", "PNB.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS"
    ],
    "Information_Technology": [
        "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS",
        "MPHASIS.NS", "PERSISTENT.NS", "COFORGE.NS"
    ],
    "Automotive": [
        "MARUTI.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS",
        "HEROMOTOCO.NS", "BHARATFORG.NS", "MOTHERSON.NS"
    ],
    "Energy_Oil_Gas": [
        "RELIANCE.NS", "ONGC.NS", "BPCL.NS", "IOC.NS", "GAIL.NS", "PETRONET.NS"
    ],
    "Power_Utilities": [
        "NTPC.NS", "POWERGRID.NS", "ADANIGREEN.NS", "TATAPOWER.NS", "JSWENERGY.NS", "NHPC.NS"
    ],
    "Metals_Mining": [
        "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "JINDALSTEL.NS",
        "NMDC.NS", "COALINDIA.NS"
    ],
    "Pharma_Healthcare": [
        "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "APOLLOHOSP.NS", "DIVISLAB.NS",
        "LUPIN.NS", "MANKIND.NS"
    ],
    "FMCG_Consumption": [
        "ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS", "TATACONSUM.NS",
        "DABUR.NS", "MARICO.NS"
    ],
    "Consumer_Durables_Retail": [
        "TITAN.NS", "TRENT.NS", "DMART.NS", "HAVELLS.NS", "VOLTAS.NS", "DIXON.NS"
    ],
    "Capital_Goods_Defense": [
        "LT.NS", "HAL.NS", "BEL.NS", "SIEMENS.NS", "ABB.NS", "BHEL.NS"
    ],
    "Realty_Construction": [
        "ADANIENT.NS", "DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "ULTRACEMCO.NS", "GRASIM.NS"
    ],
    "Chemicals_Fertilizers": [
        "PIDILITIND.NS", "SRF.NS", "GUJGASLTD.NS", "DEEPAKNTR.NS", "TATACHEM.NS", "AARTIIND.NS"
    ]
}

class NSEDataLoader:
    def __init__(self, lake_dir: str = None):
        if lake_dir is None:
            # Default to trade/data/lake
            self.lake_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "lake"))
        else:
            self.lake_dir = lake_dir
        os.makedirs(self.lake_dir, exist_ok=True)

    def _clean_yfinance_df(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Processes raw yfinance download into clean dual-stream DataFrame."""
        if raw_df.empty:
            return pd.DataFrame()
            
        df = raw_df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten multi-index
            df.columns = [col[0] for col in df.columns]

        # Standardize required dual-streams
        # Close_Adj for returns/momentum; Close_Raw for statutory circuit bands & execution
        if "Adj Close" in df.columns:
            df["Close_Adj"] = df["Adj Close"]
            df["Close_Raw"] = df["Close"]
        else:
            df["Close_Adj"] = df["Close"]
            df["Close_Raw"] = df["Close"]

        # Ensure numeric cleanliness
        required_cols = ["Open", "High", "Low", "Close_Raw", "Close_Adj", "Volume"]
        df = df[[c for c in required_cols if c in df.columns]].dropna()
        return df

    def fetch_ticker(self, ticker: str, sector: str = "General", period: str = "5y", force_refresh: bool = False) -> pd.DataFrame:
        """
        Loads a single ticker from local Parquet lake. If missing or forced, downloads and persists.
        """
        sector_dir = os.path.join(self.lake_dir, sector)
        os.makedirs(sector_dir, exist_ok=True)
        parquet_path = os.path.join(sector_dir, f"{ticker.replace('.NS', '')}.parquet")

        if not force_refresh and os.path.exists(parquet_path):
            try:
                df = pd.read_parquet(parquet_path)
                return df
            except Exception as e:
                logger.warning(f"Failed to read Parquet for {ticker}: {e}. Re-fetching from source.")

        # Download from Yahoo Finance with retry
        for attempt in range(3):
            try:
                raw_df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=False)
                df = self._clean_yfinance_df(raw_df)
                if not df.empty:
                    df.to_parquet(parquet_path, compression="snappy")
                    return df
            except Exception as e:
                time.sleep(0.5 * (attempt + 1))

        logger.error(f"Failed to fetch market data for {ticker}")
        return pd.DataFrame()

    def fetch_sector(self, sector_name: str, period: str = "5y", max_tickers: int = None) -> dict:
        """
        Loads all tickers for a sector in a chunked batch with CPU throttling.
        """
        if sector_name not in NSE_SECTORS:
            raise ValueError(f"Unknown sector: {sector_name}. Available: {list(NSE_SECTORS.keys())}")

        tickers = NSE_SECTORS[sector_name]
        if max_tickers:
            tickers = tickers[:max_tickers]

        sector_data = {}
        logger.info(f"Loading Sector '{sector_name}' ({len(tickers)} tickers)...")

        for ticker in tickers:
            df = self.fetch_ticker(ticker, sector=sector_name, period=period)
            if not df.empty:
                sector_data[ticker] = df
            # Thermal throttle: 50ms pause to keep laptop fan quiet
            time.sleep(0.05)

        # Clear memory
        gc.collect()
        return sector_data

    def fetch_universe_sample(self, leaders_per_sector: int = 1, period: str = "3y") -> dict:
        """
        Quickly fetches top N leaders from all 12 sectors with minimal CPU overhead.
        """
        universe_data = {}
        for sector_name, tickers in NSE_SECTORS.items():
            selected = tickers[:leaders_per_sector]
            for ticker in selected:
                df = self.fetch_ticker(ticker, sector=sector_name, period=period)
                if not df.empty:
                    universe_data[ticker] = df
                time.sleep(0.03)
            gc.collect()
        return universe_data

    def resample_candles(self, df: pd.DataFrame, rule: str = "3h") -> pd.DataFrame:
        """
        Resamples high-resolution intraday candles into multi-hour session bars (e.g. 3h/5h).
        Preserves dual-stream OHLCV column integrity.
        """
        if df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return df

        agg_dict = {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Volume": "sum"
        }
        if "Close_Raw" in df.columns:
            agg_dict["Close_Raw"] = "last"
        if "Close_Adj" in df.columns:
            agg_dict["Close_Adj"] = "last"
        if "Close" in df.columns and "Close_Raw" not in df.columns:
            agg_dict["Close"] = "last"

        resampled = df.resample(rule).agg(agg_dict).dropna()
        if "Volume" in resampled.columns:
            resampled = resampled[resampled["Volume"] > 0]
        return resampled

    def fetch_timeframe_ticker(
        self, 
        ticker: str, 
        sector: str = "General", 
        timeframe: str = "1d", 
        period: str = "1y",
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Loads ticker data for specific timeframes: '1d', '1h', or 'session' (~3h).
        Persists to local Parquet cache with timeframe suffixes.
        """
        if timeframe == "1d":
            return self.fetch_ticker(ticker, sector=sector, period=period, force_refresh=force_refresh)

        clean_sym = ticker.replace(".NS", "")
        sector_dir = os.path.join(self.lake_dir, sector)
        os.makedirs(sector_dir, exist_ok=True)
        parquet_path = os.path.join(sector_dir, f"{clean_sym}_{timeframe}.parquet")

        if not force_refresh and os.path.exists(parquet_path):
            try:
                return pd.read_parquet(parquet_path)
            except Exception as e:
                logger.warning(f"Failed to read cached timeframe {timeframe} for {ticker}: {e}")

        # Fetch 1h from yfinance
        try:
            raw_df = yf.download(ticker, period=period, interval="1h", progress=False, auto_adjust=False)
            df_1h = self._clean_yfinance_df(raw_df)
            if df_1h.empty:
                return pd.DataFrame()

            if timeframe == "1h":
                df_1h.to_parquet(parquet_path, compression="snappy")
                return df_1h
            elif timeframe in ["session", "3h", "5h"]:
                rule = "3h" if timeframe in ["session", "3h"] else "5h"
                resampled = self.resample_candles(df_1h, rule=rule)
                resampled.to_parquet(parquet_path, compression="snappy")
                return resampled
            else:
                return df_1h
        except Exception as e:
            logger.error(f"Error fetching {timeframe} for {ticker}: {e}")
            return pd.DataFrame()
