"""Unit tests for Muscle 3: Market Data Ingestion & Session Gatekeeper.
Strictly verifies:
- NSE Trading Hours Gatekeeping (09:15 to 15:30 IST).
- Pre-market, post-market, and weekend filtration.
- Timezone standardization to Asia/Kolkata (UTC+05:30).
- Column normalization and dtype casting.
- Local CSV/Parquet ingestion and deduplication.
"""
from __future__ import annotations

import os
import tempfile
import unittest
import numpy as np
import pandas as pd

from ravana.data.ingestion.config import IngestionConfig
from ravana.data.ingestion.session_gatekeeper import NSESessionGatekeeper
from ravana.data.ingestion.loader import MarketDataLoader


class TestNSESessionGatekeeper(unittest.TestCase):
    """Test suite for the institutional NSE Market Session Gatekeeper."""

    def setUp(self) -> None:
        self.gatekeeper = NSESessionGatekeeper()

    def test_filter_intraday_hours(self) -> None:
        """Verify only 09:15 to 15:30 IST bars are retained; pre/post market dropped."""
        timestamps = pd.date_range(
            start="2026-11-09 08:30:00",
            end="2026-11-09 16:30:00",
            freq="15min",
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

        # Check earliest bar is >= 09:15 and latest bar is <= 15:30
        self.assertFalse(filtered.empty)
        first_time = filtered.index[0].time()
        last_time = filtered.index[-1].time()
        self.assertGreaterEqual(first_time, pd.Timestamp("09:15:00").time())
        self.assertLessEqual(last_time, pd.Timestamp("15:30:00").time())

        # Ensure 09:00 pre-market and 16:00 post-market were dropped
        pre_market = filtered.index.indexer_at_time("09:00:00")
        post_market = filtered.index.indexer_at_time("16:00:00")
        self.assertEqual(len(pre_market), 0)
        self.assertEqual(len(post_market), 0)

    def test_weekend_filtration(self) -> None:
        """Verify Saturday and Sunday bars are purged unless explicitly whitelisted."""
        dates = pd.date_range("2026-11-06", "2026-11-10", freq="D", tz="Asia/Kolkata")
        df = pd.DataFrame({"Close": [100.0, 101.0, 102.0, 103.0, 104.0]}, index=dates)

        # 2026-11-06 is Friday, 07 is Sat, 08 is Sun, 09 is Mon, 10 is Tue
        cleaned = self.gatekeeper.filter_weekends(df)
        self.assertEqual(len(cleaned), 3)  # Fri, Mon, Tue
        self.assertNotIn(pd.Timestamp("2026-11-07", tz="Asia/Kolkata"), cleaned.index)
        self.assertNotIn(pd.Timestamp("2026-11-08", tz="Asia/Kolkata"), cleaned.index)

    def test_timezone_standardization(self) -> None:
        """Verify naive timestamps and UTC timestamps are standardized to Asia/Kolkata."""
        # Naive index
        naive_idx = pd.date_range("2026-11-09 09:15", periods=5, freq="5min")
        df_naive = pd.DataFrame({"Close": [100, 101, 102, 103, 104]}, index=naive_idx)
        std_naive = self.gatekeeper.standardize_timezone(df_naive)
        self.assertEqual(str(std_naive.index.tz), "Asia/Kolkata")

        # UTC index
        utc_idx = pd.date_range("2026-11-09 03:45", periods=5, freq="5min", tz="UTC")
        df_utc = pd.DataFrame({"Close": [100, 101, 102, 103, 104]}, index=utc_idx)
        std_utc = self.gatekeeper.standardize_timezone(df_utc)
        self.assertEqual(str(std_utc.index.tz), "Asia/Kolkata")
        # 03:45 UTC is 09:15 IST
        self.assertEqual(std_utc.index[0].hour, 9)
        self.assertEqual(std_utc.index[0].minute, 15)


class TestMarketDataLoader(unittest.TestCase):
    """Test suite for universal market data loader."""

    def setUp(self) -> None:
        self.loader = MarketDataLoader()

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

            # Check standard columns
            expected_cols = ["Open", "High", "Low", "Close", "Volume"]
            for col in expected_cols:
                self.assertIn(col, loaded.columns)
                self.assertEqual(loaded[col].dtype, np.float64)

            self.assertEqual(len(loaded), 3)
            self.assertEqual(str(loaded.index.tz), "Asia/Kolkata")

    def test_deduplication_and_sorting(self) -> None:
        """Verify duplicate rows and out-of-order timestamps are resolved."""
        timestamps = [
            pd.Timestamp("2026-11-09 09:25:00", tz="Asia/Kolkata"),
            pd.Timestamp("2026-11-09 09:15:00", tz="Asia/Kolkata"),
            pd.Timestamp("2026-11-09 09:20:00", tz="Asia/Kolkata"),
            pd.Timestamp("2026-11-09 09:15:00", tz="Asia/Kolkata"),  # Duplicate!
        ]
        df_disordered = pd.DataFrame(
            {
                "Open": [102.0, 100.0, 101.0, 100.0],
                "High": [103.0, 101.0, 102.0, 101.0],
                "Low": [101.5, 99.5, 100.5, 99.5],
                "Close": [102.5, 100.5, 101.5, 100.5],
                "Volume": [1000, 1000, 1000, 1000],
            },
            index=timestamps,
        )

        cleaned = self.loader.clean_index_and_chronology(df_disordered)
        self.assertEqual(len(cleaned), 3)
        self.assertTrue(cleaned.index.is_monotonic_increasing)
        self.assertEqual(cleaned.index[0], pd.Timestamp("2026-11-09 09:15:00", tz="Asia/Kolkata"))

    def test_fetch_yfinance_and_fallback(self) -> None:
        """Verify yfinance fetch or fallback gracefully operates under IngestionConfig."""
        cfg = IngestionConfig(ticker="BANKBEES.NS", interval="1d", period="5d")
        df = self.loader.fetch_yfinance(cfg)
        if not df.empty:
            self.assertIn("Close", df.columns)
            self.assertEqual(str(df.index.tz), "Asia/Kolkata")
            self.assertGreaterEqual(len(df), 1)

        # Test local fallback
        with tempfile.TemporaryDirectory() as tmpdir:
            fallback_csv = os.path.join(tmpdir, "fallback.csv")
            pd.DataFrame(
                {
                    "date": ["2026-11-09"],
                    "open": [560.0],
                    "high": [565.0],
                    "low": [558.0],
                    "close": [562.0],
                    "volume": [100000.0],
                }
            ).to_csv(fallback_csv, index=False)

            invalid_cfg = IngestionConfig(ticker="INVALID_TICKER_9999.NS", interval="1d")
            res = self.loader.ingest(invalid_cfg, local_fallback_path=fallback_csv)
            self.assertEqual(len(res), 1)
            self.assertEqual(res["Close"].iloc[0], 562.0)



if __name__ == "__main__":
    unittest.main()
