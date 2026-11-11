"""Adversarial Microstructure & Edge-Case Deep Dive for RAVANA Muscle 3."""
import sys
import os
import time
import tracemalloc
import pandas as pd
import numpy as np
import yfinance as yf

sys.path.insert(0, os.path.abspath("."))

from ravana.data.ingestion.config import IngestionConfig
from ravana.data.ingestion.session_gatekeeper import NSESessionGatekeeper
from ravana.data.ingestion.loader import MarketDataLoader

def print_hdr(title):
    print("\n" + "=" * 80)
    print(f" >>> {title}")
    print("=" * 80)

def test_raw_yfinance_structure():
    print_hdr("1. Raw yfinance Output Columns & Levels")
    raw_daily = yf.download("BANKBEES.NS", period="5d", interval="1d", progress=False, auto_adjust=False)
    print("Single ticker daily columns:", raw_daily.columns)
    print("Is MultiIndex?", isinstance(raw_daily.columns, pd.MultiIndex))
    if isinstance(raw_daily.columns, pd.MultiIndex):
        print("Levels:", raw_daily.columns.levels)
        print("Column tuples:", raw_daily.columns.tolist())

    raw_intraday = yf.download("BANKBEES.NS", period="5d", interval="5m", progress=False, auto_adjust=False)
    print("Single ticker 5m columns:", raw_intraday.columns)
    print("5m Index tz:", raw_intraday.index.tz)
    print("5m Head index:", raw_intraday.index[:2])

    print("\n--- Multi-Ticker Download Test ---")
    raw_multi = yf.download(["BANKBEES.NS", "NIFTYBEES.NS"], period="5d", interval="1d", progress=False, auto_adjust=False)
    print("Multi-ticker columns:", raw_multi.columns)
    
    loader = MarketDataLoader()
    try:
        std_multi = loader.standardize_columns(raw_multi)
        print("std_multi columns after loader:", std_multi.columns.tolist())
        print("Are columns unique?:", std_multi.columns.is_unique)
        if not std_multi.columns.is_unique:
            print("ALERT: Duplicate columns found! Column counts:\n", pd.Series(std_multi.columns).value_counts())
            # Try accessing 'Close'
            print("std_multi['Close'] shape:", std_multi["Close"].shape)
    except Exception as e:
        print(f"standardize_columns exploded on multi-ticker: {type(e).__name__}: {e}")

def test_nan_and_zero_price_behavior():
    print_hdr("2. NaN and Zero-Price Handling in Loader & Gatekeeper")
    loader = MarketDataLoader()
    
    # Create DataFrame with adversarial values: NaNs, 0.0 prices, negative prices, inf
    adv_df = pd.DataFrame({
        "Date": pd.date_range("2026-11-02", periods=5, freq="D", tz="Asia/Kolkata"),
        "open": [100.0, 0.0, np.nan, -50.0, 105.0],
        "high": [105.0, 102.0, np.nan, 100.0, np.inf],
        "low": [95.0, 0.0, np.nan, -60.0, 95.0],
        "close": [102.0, 101.0, np.nan, -55.0, 104.0],
        "volume": [1000, 0, 500, -100, 0]
    }).set_index("Date")

    print("Input Adversarial DataFrame:\n", adv_df)
    std = loader.standardize_columns(adv_df)
    print("After standardize_columns:\n", std)
    print("Does loader raise an error or warn? No! Let's check NaNs:\n", std.isna().sum())
    print("Are negative/zero prices caught?:", (std[["Open", "High", "Low", "Close"]] <= 0).sum().to_dict())

def test_timezone_dst_and_naive_conversion():
    print_hdr("3. Timezone, UTC and Daylight Savings Analysis")
    gatekeeper = NSESessionGatekeeper()
    
    # What if a daily feed has UTC timestamps at 00:00:00 UTC?
    utc_daily = pd.DataFrame(
        {"Close": [100.0, 101.0]},
        index=pd.to_datetime(["2026-11-09 00:00:00+00:00", "2026-11-10 00:00:00+00:00"])
    )
    conv_utc = gatekeeper.standardize_timezone(utc_daily)
    print("UTC Daily original index:\n", utc_daily.index)
    print("Converted to Asia/Kolkata:\n", conv_utc.index)
    # Notice: 00:00 UTC converted to Asia/Kolkata becomes 05:30:00 IST!
    print("Notice the time component:", conv_utc.index[0].time())

    # What if a US data vendor or yfinance returns tz-naive vs tz-aware?
    naive_daily = pd.DataFrame(
        {"Close": [100.0, 101.0]},
        index=pd.to_datetime(["2026-11-09", "2026-11-10"])
    )
    conv_naive = gatekeeper.standardize_timezone(naive_daily)
    print("Naive Daily original index:\n", naive_daily.index)
    print("Localized to Asia/Kolkata:\n", conv_naive.index)

def test_intraday_gap_and_auction_boundary():
    print_hdr("4. Intraday Gap and 15:30 Bar Behavior")
    # Fetch high-res 1m and 5m for recent trading days
    loader = MarketDataLoader()
    cfg_1m = IngestionConfig(ticker="BANKBEES.NS", interval="1m", period="5d")
    df_1m = loader.fetch_yfinance(cfg_1m)
    if not df_1m.empty:
        print(f"1m Shape: {df_1m.shape}, Date Range: {df_1m.index.min()} to {df_1m.index.max()}")
        last_bars_1m = df_1m.groupby(df_1m.index.date).apply(lambda g: g.index[-1].time())
        first_bars_1m = df_1m.groupby(df_1m.index.date).apply(lambda g: g.index[0].time())
        print("1m First bar times:\n", first_bars_1m.value_counts())
        print("1m Last bar times:\n", last_bars_1m.value_counts())
        print("1m Sample last 5 rows of last day:\n", df_1m.tail(5))
        # Is there any 15:30:00 bar?
        bars_at_1530 = df_1m.between_time("15:30:00", "15:30:00")
        print(f"Total bars exactly at 15:30:00 in 1m data: {len(bars_at_1530)}")
        if not bars_at_1530.empty:
            print("15:30 bars sample:\n", bars_at_1530)

def test_missing_bars_and_microstructure():
    print_hdr("5. Missing Bars & Zero Volume in BANKBEES 5m")
    loader = MarketDataLoader()
    cfg_5m = IngestionConfig(ticker="BANKBEES.NS", interval="5m", period="60d")
    df = loader.fetch_yfinance(cfg_5m)
    
    # Check days with missing bars
    bars_per_day = df.groupby(df.index.date).size()
    abnormal = bars_per_day[bars_per_day != 75]
    print("Days with != 75 bars:\n", abnormal)
    
    for dt, cnt in abnormal.items():
        day_df = df[df.index.date == dt]
        # Generate expected 75 bars
        expected_times = pd.date_range(
            f"{dt} 09:15:00", f"{dt} 15:25:00", freq="5min", tz="Asia/Kolkata"
        )
        missing = expected_times.difference(day_df.index)
        print(f"Date {dt} (has {cnt} bars) is missing: {[t.strftime('%H:%M') for t in missing]}")

if __name__ == "__main__":
    test_raw_yfinance_structure()
    test_nan_and_zero_price_behavior()
    test_timezone_dst_and_naive_conversion()
    test_intraday_gap_and_auction_boundary()
    test_missing_bars_and_microstructure()
