"""Adversarial Real-Data Stress-Tester for RAVANA Muscle 3."""
import sys
import os
import time
import tracemalloc
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from ravana.data.ingestion.config import IngestionConfig
from ravana.data.ingestion.session_gatekeeper import NSESessionGatekeeper
from ravana.data.ingestion.loader import MarketDataLoader

def print_separator(title):
    print("\n" + "=" * 80)
    print(f" >>> {title}")
    print("=" * 80)

def inspect_df(name: str, df: pd.DataFrame, is_intraday: bool = False):
    print(f"--- Inspection: {name} ---")
    print(f"Shape: {df.shape}")
    if df.empty:
        print("DataFrame is EMPTY!")
        return

    print(f"Index Type: {type(df.index)}")
    print(f"Timezone: {df.index.tz}")
    print(f"Date Range: {df.index.min()} -> {df.index.max()}")
    print(f"Columns & Dtypes:\n{df.dtypes}")
    print(f"NaN Counts per Column:\n{df.isna().sum()}")
    print(f"Duplicate Index Count: {df.index.duplicated().sum()}")
    print(f"Is Monotonic Increasing: {df.index.is_monotonic_increasing}")
    
    # Zero volume check
    if "Volume" in df.columns:
        zero_vol = (df["Volume"] == 0).sum()
        print(f"Zero Volume Bars: {zero_vol} ({zero_vol / len(df) * 100:.2f}%)")

    # Day of week check
    weekends = (df.index.dayofweek >= 5).sum()
    print(f"Weekend bars (Sat/Sun): {weekends}")

    # For intraday, check daily bar distribution and boundary timestamps
    if is_intraday:
        # Group by date
        dates = df.index.normalize().unique()
        print(f"Total Unique Trading Days: {len(dates)}")
        
        # Check first and last bar times per day
        first_times = df.groupby(df.index.date).apply(lambda g: g.index[0].time())
        last_times = df.groupby(df.index.date).apply(lambda g: g.index[-1].time())
        bars_per_day = df.groupby(df.index.date).size()

        print(f"First Bar Times value counts:\n{first_times.value_counts()}")
        print(f"Last Bar Times value counts:\n{last_times.value_counts()}")
        print(f"Bars per Day Summary: min={bars_per_day.min()}, max={bars_per_day.max()}, median={bars_per_day.median()}")
        
        # Check if there are days with abnormal bar counts
        abnormal_days = bars_per_day[bars_per_day < bars_per_day.median()]
        if not abnormal_days.empty:
            print(f"Sample days with fewer bars than median ({len(abnormal_days)} days):")
            print(abnormal_days.head(5))

    print(f"Sample 3 First Rows:\n{df.head(3)}")
    print(f"Sample 3 Last Rows:\n{df.tail(3)}")

def run_stress_test():
    loader = MarketDataLoader()
    
    # Task 1: Ingest BANKBEES.NS on Daily (5y and max)
    print_separator("TASK 1: BANKBEES.NS Daily (5y)")
    t0 = time.perf_counter()
    tracemalloc.start()
    cfg_daily_5y = IngestionConfig(ticker="BANKBEES.NS", interval="1d", period="5y")
    df_bankbees_daily_5y = loader.fetch_yfinance(cfg_daily_5y)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    t1 = time.perf_counter()
    print(f"Latency: {t1 - t0:.2f}s | Peak Memory: {peak / (1024*1024):.2f} MB")
    inspect_df("BANKBEES.NS Daily 5y", df_bankbees_daily_5y, is_intraday=False)

    print_separator("TASK 1 (cont): BANKBEES.NS Daily (max)")
    t0 = time.perf_counter()
    tracemalloc.start()
    cfg_daily_max = IngestionConfig(ticker="BANKBEES.NS", interval="1d", period="max")
    df_bankbees_daily_max = loader.fetch_yfinance(cfg_daily_max)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    t1 = time.perf_counter()
    print(f"Latency: {t1 - t0:.2f}s | Peak Memory: {peak / (1024*1024):.2f} MB")
    inspect_df("BANKBEES.NS Daily max", df_bankbees_daily_max, is_intraday=False)

    # Task 2: Ingest BANKBEES.NS on 5-minute (60d)
    print_separator("TASK 2: BANKBEES.NS 5m (60d)")
    t0 = time.perf_counter()
    tracemalloc.start()
    cfg_5m = IngestionConfig(ticker="BANKBEES.NS", interval="5m", period="60d")
    df_bankbees_5m = loader.fetch_yfinance(cfg_5m)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    t1 = time.perf_counter()
    print(f"Latency: {t1 - t0:.2f}s | Peak Memory: {peak / (1024*1024):.2f} MB")
    inspect_df("BANKBEES.NS 5m 60d", df_bankbees_5m, is_intraday=True)

    # Task 3: Ingest BANKBEES.NS on 1-hour (730d)
    print_separator("TASK 3: BANKBEES.NS 1h (730d)")
    t0 = time.perf_counter()
    tracemalloc.start()
    cfg_1h = IngestionConfig(ticker="BANKBEES.NS", interval="1h", period="730d")
    df_bankbees_1h = loader.fetch_yfinance(cfg_1h)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    t1 = time.perf_counter()
    print(f"Latency: {t1 - t0:.2f}s | Peak Memory: {peak / (1024*1024):.2f} MB")
    inspect_df("BANKBEES.NS 1h 730d", df_bankbees_1h, is_intraday=True)

    # Task 4: Ingest NIFTYBEES.NS on Daily and 5m
    print_separator("TASK 4A: NIFTYBEES.NS Daily (5y)")
    cfg_nifty_daily = IngestionConfig(ticker="NIFTYBEES.NS", interval="1d", period="5y")
    df_nifty_daily = loader.fetch_yfinance(cfg_nifty_daily)
    inspect_df("NIFTYBEES.NS Daily 5y", df_nifty_daily, is_intraday=False)

    print_separator("TASK 4B: NIFTYBEES.NS 5m (60d)")
    cfg_nifty_5m = IngestionConfig(ticker="NIFTYBEES.NS", interval="5m", period="60d")
    df_nifty_5m = loader.fetch_yfinance(cfg_nifty_5m)
    inspect_df("NIFTYBEES.NS 5m 60d", df_nifty_5m, is_intraday=True)

    # Task 5: Existing parquet files from data/lake/Indices/BANKBEES.parquet
    print_separator("TASK 5: Load local data/lake/Indices/BANKBEES.parquet")
    local_path = "data/lake/Indices/BANKBEES.parquet"
    if os.path.exists(local_path):
        df_local_raw = pd.read_parquet(local_path)
        print("Raw parquet inspection:")
        print(f"Raw shape: {df_local_raw.shape}, raw columns: {df_local_raw.columns.tolist()}")
        print(f"Raw index type: {type(df_local_raw.index)}")
        print(f"Raw head:\n{df_local_raw.head(2)}")
        
        try:
            df_local_loaded = loader.load_local(local_path, is_intraday=False)
            inspect_df("Local BANKBEES.parquet (via loader)", df_local_loaded, is_intraday=False)
        except Exception as e:
            print(f"FAILED to load local parquet via loader: {type(e).__name__}: {e}")
    else:
        print(f"File not found: {local_path}")

    # Also inspect NIFTYBEES.parquet
    local_nifty = "data/lake/Indices/NIFTYBEES.parquet"
    if os.path.exists(local_nifty):
        print_separator("TASK 5 (cont): Load local data/lake/Indices/NIFTYBEES.parquet")
        try:
            df_local_nifty = loader.load_local(local_nifty, is_intraday=False)
            inspect_df("Local NIFTYBEES.parquet (via loader)", df_local_nifty, is_intraday=False)
        except Exception as e:
            print(f"FAILED to load local NIFTYBEES parquet via loader: {type(e).__name__}: {e}")

if __name__ == "__main__":
    run_stress_test()
