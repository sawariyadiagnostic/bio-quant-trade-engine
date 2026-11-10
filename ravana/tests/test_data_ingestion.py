"""Unit tests for Muscle 3: Market Data Ingestion & Session Gatekeeper.
Strictly verifies:
- NSE Trading Hours Gatekeeping (09:15 to 15:25 IST for left-closed 5m).
- Pre-market, post-market, and weekend filtration.
- Special session preservation (Union Budget Saturdays, Diwali Muhurat evenings).
- Timezone standardization to Asia/Kolkata (UTC+05:30).
- Candlestick physical sanity validation and proportional split adjustment.
- Monthly / Daily interval preservation (no accidental wipeout).
"""
from __future__ import annotations

import os
import tempfile
import unittest
import numpy as np
import pandas as pd

from ravana.data.ingestion.config import IngestionConfig
from ravana.data.ingestion.session_gatekeeper import NSESessionGatekeeper
from ravana.data.ingestion.loader import MarketDataLoader, CandlestickSanityValidator


class TestNSESessionGatekeeper(unittest.TestCase):
    """Test suite for the institutional NSE Market Session Gatekeeper."""

    def setUp(self) -> None:
        self.gatekeeper = NSESessionGatekeeper()

    def test_filter_intraday_hours_left_closed(self) -> None:
        """Verify left-closed intervals admit [09:15, 15:30), excluding 15:30:00 closing bar."""
        timestamps = pd.date_range(
            start="2026-11-09 09:15:00",
            end="2026-11-09 15:30:00",
            freq="5min",
            tz="Asia/Kolkata",
        )
        df = pd.DataFrame(
            {
                "Open": np.linspace(100, 110, len(timestamps)),
                "High": np.linspace(101, 111, len(timestamps)),
                "Low": np.linspace(99, 109, len(timestamps)),
                "Close": np.linspace(100.5, 110.5, len(timestamps)),
                "Volume": np.full(len(timestamps), 1000.0),
            },
            index=timestamps,
        )

        filtered = self.gatekeeper.filter_trading_hours(df, is_intraday=True)

        # In left-closed 5m data, there are exactly 75 bars: 09:15:00 to 15:25:00
        self.assertEqual(len(filtered), 75)
        self.assertEqual(filtered.index[0].time(), pd.Timestamp("09:15:00").time())
        self.assertEqual(filtered.index[-1].time(), pd.Timestamp("15:25:00").time())
        self.assertNotIn(pd.Timestamp("2026-11-09 15:30:00", tz="Asia/Kolkata"), filtered.index)

    def test_union_budget_saturday_retained(self) -> None:
        """Verify Saturday 2025-02-01 (Union Budget) is retained by default."""
        dates = pd.date_range("2025-01-31", "2025-02-03", freq="D", tz="Asia/Kolkata")
        df = pd.DataFrame({"Close": [100.0, 101.0, 102.0, 103.0]}, index=dates)

        cleaned = self.gatekeeper.filter_weekends(df)
        # Jan 31 (Fri), Feb 01 (Budget Sat), Feb 03 (Mon) -> 3 days retained (Feb 02 Sun dropped)
        self.assertEqual(len(cleaned), 3)
        self.assertIn(pd.Timestamp("2025-02-01", tz="Asia/Kolkata"), cleaned.index)

    def test_diwali_muhurat_evening_session_retained(self) -> None:
        """Verify Diwali Muhurat evening hours (18:15 to 19:15) are strictly retained."""
        timestamps = pd.date_range(
            start="2023-11-12 18:00:00",
            end="2023-11-12 19:30:00",
            freq="5min",
            tz="Asia/Kolkata",
        )
        df = pd.DataFrame({"Close": np.linspace(100, 105, len(timestamps))}, index=timestamps)

        filtered = self.gatekeeper.filter_trading_hours(df, is_intraday=True)
        self.assertFalse(filtered.empty)
        self.assertEqual(filtered.index[0].time(), pd.Timestamp("18:15:00").time())
        self.assertLess(filtered.index[-1].time(), pd.Timestamp("19:15:00").time())


class TestMarketDataLoader(unittest.TestCase):
    """Test suite for universal market data loader."""

    def setUp(self) -> None:
        self.loader = MarketDataLoader()

    def test_monthly_interval_not_intraday(self) -> None:
        """Verify 1mo interval is correctly marked non-intraday, preserving data."""
        cfg = IngestionConfig(ticker="BANKBEES.NS", interval="1mo")
        self.assertFalse(cfg.is_intraday())

        monthly_idx = pd.date_range("2025-01-01", periods=6, freq="MS", tz="Asia/Kolkata")
        df_monthly = pd.DataFrame(
            {
                "Open": [100.0] * 6,
                "High": [105.0] * 6,
                "Low": [95.0] * 6,
                "Close": [102.0] * 6,
                "Volume": [10000.0] * 6,
            },
            index=monthly_idx,
        )
        res = self.loader.gatekeeper.filter_trading_hours(
            df_monthly, is_intraday=cfg.is_intraday(), filter_weekends_flag=False
        )
        self.assertEqual(len(res), 6)

    def test_candlestick_sanity_validator(self) -> None:
        """Verify zero prices and High/Low geometric inversions are repaired."""
        df_corrupt = pd.DataFrame(
            {
                "Open": [0.0, 100.0],       # 0.0 Open price
                "High": [90.0, 102.0],      # High < Open
                "Low": [110.0, 98.0],       # Low > High
                "Close": [100.0, 101.0],
                "Volume": [-500.0, 1000.0], # Negative volume
            }
        )
        cleaned = CandlestickSanityValidator.sanitize(df_corrupt)
        self.assertGreater(cleaned["Open"].iloc[0], 0.0)
        self.assertGreaterEqual(cleaned["High"].iloc[0], cleaned["Low"].iloc[0])
        self.assertGreaterEqual(cleaned["High"].iloc[0], cleaned["Open"].iloc[0])
        self.assertGreaterEqual(cleaned["Volume"].iloc[0], 0.0)

    def test_proportional_split_adjustment(self) -> None:
        """Verify Close_Adj scales Open/High/Low proportionally preserving candle geometry."""
        df_split = pd.DataFrame(
            {
                "Open": [1000.0],
                "High": [1050.0],
                "Low": [980.0],
                "Close": [1000.0],
                "Close_Adj": [100.0],  # 10:1 split
                "Volume": [1000.0],
            }
        )
        scaled = CandlestickSanityValidator.sanitize(df_split)
        self.assertEqual(scaled["Open_Adj"].iloc[0], 100.0)
        self.assertEqual(scaled["High_Adj"].iloc[0], 105.0)
        self.assertEqual(scaled["Low_Adj"].iloc[0], 98.0)
        self.assertGreaterEqual(scaled["High_Adj"].iloc[0], scaled["Close_Adj"].iloc[0])
        self.assertLessEqual(scaled["Low_Adj"].iloc[0], scaled["Close_Adj"].iloc[0])

    def test_load_local_csv_and_standardize_columns(self) -> None:
        """Verify local CSV ingestion standardizes column names and dtypes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test_raw.csv")
            df_mock = pd.DataFrame(
                {
                    "date": ["2026-11-09 09:15:00", "2026-11-09 09:20:00", "2026-11-09 09:25:00"],
                    "open": ["100.2", "101.5", "102.1"],
                    "high": ["101.0", "102.0", "103.0"],
                    "low": ["100.0", "101.0", "101.5"],
                    "close": ["100.8", "101.8", "102.5"],
                    "volume": [5000, 6200, 4800],
                }
            )
            df_mock.to_csv(csv_path, index=False)

            loaded = self.loader.load_local(csv_path, is_intraday=True)
            expected_cols = ["Open", "High", "Low", "Close", "Volume"]
            for col in expected_cols:
                self.assertIn(col, loaded.columns)
                self.assertEqual(loaded[col].dtype, np.float64)

            self.assertEqual(len(loaded), 3)
            self.assertEqual(str(loaded.index.tz), "Asia/Kolkata")

    def test_macro_universe_auto_deduction(self) -> None:
        """Verify IngestionConfig.create automatically deduces exchange profiles and timezones."""
        cfg_bank = IngestionConfig.create("BANKBEES.NS")
        self.assertEqual(cfg_bank.timezone, "Asia/Kolkata")
        self.assertEqual(cfg_bank.market_open_time, "09:15:00")

        cfg_sp500 = IngestionConfig.create("^GSPC")
        self.assertEqual(cfg_sp500.timezone, "America/New_York")
        self.assertEqual(cfg_sp500.market_open_time, "09:30:00")

        cfg_fx = IngestionConfig.create("INR=X")
        self.assertEqual(cfg_fx.timezone, "UTC")
        self.assertEqual(cfg_fx.market_open_time, "00:00:00")

    def test_ingest_etf_contract(self) -> None:
        """Verify ingest_etf bundles ETF data, benchmark pair, and contract metadata."""
        # Use local parquet to verify offline pairing
        local_path = "data/lake/Indices/BANKBEES.parquet"
        if os.path.exists(local_path):
            dataset = self.loader.ingest_etf(
                ticker="BANKBEES.NS",
                interval="1d",
                period="5y",
                include_macro=False,
                local_fallback_path=local_path,
            )
            self.assertEqual(dataset.ticker, "BANKBEES.NS")
            self.assertEqual(dataset.benchmark_ticker, "^NSEBANK")
            self.assertGreater(dataset.bar_count, 1000)
            summary = dataset.summary()
            self.assertEqual(summary["ticker"], "BANKBEES.NS")
            self.assertIn("bars", summary)


if __name__ == "__main__":
    unittest.main()


