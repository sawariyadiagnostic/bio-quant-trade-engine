"""Unit tests for Muscle 4: Data Quality & Cleaning Engine.
Verifies intraday grid reindexing, causal gap-filling, flash spike sanitization,
and ETF-to-benchmark tracking spread alignment.
"""
from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from ravana.data.cleaner.config import CleanerConfig
from ravana.data.cleaner.dataset import CleanedDataset
from ravana.data.cleaner.grid_reindexer import TimeGridReindexer
from ravana.data.cleaner.spike_filter import FlashSpikeFilter
from ravana.data.cleaner.spread_aligner import ETFBUSpreadAligner
from ravana.data.cleaner.cleaner import DataCleaningEngine
from ravana.data.ingestion.config import ExchangeType
from ravana.data.ingestion.dataset import IngestedDataset


class TestDataQualityEngine(unittest.TestCase):
    """Test suite for Muscle 4 data cleaning, grid reindexing, and spread alignment."""

    def setUp(self) -> None:
        """Create reproducible test datasets."""
        self.tz = "Asia/Kolkata"
        # Full 75-bar 5m grid for 2026-03-10
        self.full_times = pd.date_range("2026-03-10 09:15:00", "2026-03-10 15:25:00", freq="5min", tz=self.tz)

    def _make_intraday_bar(self, times: pd.DatetimeIndex, base_price: float = 500.0) -> pd.DataFrame:
        """Helper to create valid OHLCV series."""
        n = len(times)
        np.random.seed(42)
        close = base_price + np.cumsum(np.random.normal(0, 0.5, n))
        high = close + np.random.uniform(0.1, 0.8, n)
        low = close - np.random.uniform(0.1, 0.8, n)
        open_ = (high + low) / 2.0
        volume = np.random.uniform(1000, 50000, n)
        return pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
            index=times,
        )

    def test_time_grid_reindexer_intraday_gap_filling(self) -> None:
        """Verify missing intraday bars are causally filled to complete the 75-bar grid."""
        df_full = self._make_intraday_bar(self.full_times)
        # Drop bars 20 and 21 (e.g. 10:55 and 11:00) simulating data feed holes (73 bars)
        df_gappy = df_full.drop(df_full.index[[20, 21]])
        self.assertEqual(len(df_gappy), 73)

        reindexer = TimeGridReindexer(exchange=ExchangeType.NSE)
        df_clean, synthetic_count = reindexer.reindex(df_gappy, interval="5m")

        # Must restore exact 75 bars
        self.assertEqual(len(df_clean), 75)
        self.assertEqual(synthetic_count, 2)
        self.assertTrue("Is_Synthetic" in df_clean.columns)

        # Check synthetic bar properties
        synthetic_rows = df_clean[df_clean["Is_Synthetic"]]
        self.assertEqual(len(synthetic_rows), 2)
        self.assertTrue((synthetic_rows["Volume"] == 0.0).all())

        # Prices must be causally forward-filled from bar 19
        prev_close = df_full["Close"].iloc[19]
        for _, row in synthetic_rows.iterrows():
            self.assertAlmostEqual(row["Close"], prev_close, places=4)
            self.assertAlmostEqual(row["Open"], prev_close, places=4)
            self.assertAlmostEqual(row["High"], prev_close, places=4)
            self.assertAlmostEqual(row["Low"], prev_close, places=4)

    def test_time_grid_reindexer_non_intraday(self) -> None:
        """Verify daily series are monotonically sorted with duplicate timestamps removed."""
        dates = pd.date_range("2026-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {"Open": 100.0, "High": 105.0, "Low": 95.0, "Close": 102.0, "Volume": 1000.0},
            index=dates,
        )
        # Add duplicate row
        df_dup = pd.concat([df, df.iloc[[3]]])
        reindexer = TimeGridReindexer()
        df_clean, synth = reindexer.reindex(df_dup, interval="1d")

        self.assertEqual(len(df_clean), 10)
        self.assertEqual(synth, 0)
        self.assertTrue(df_clean.index.is_monotonic_increasing)

    def test_flash_spike_filter_adversarial_spikes(self) -> None:
        """Verify single-bar price spikes and aberrant wicks are detected and clamped."""
        df = self._make_intraday_bar(self.full_times, base_price=500.0)

        # Adversarial spike 1: Massive Close jump at bar 40
        df.iloc[40, df.columns.get_loc("Close")] = 750.0  # +50% jump
        df.iloc[40, df.columns.get_loc("High")] = 760.0

        # Adversarial spike 2: Massive High wick at bar 60
        df.iloc[60, df.columns.get_loc("High")] = 900.0

        spike_filter = FlashSpikeFilter(window=20, threshold_std=4.0, clamp_spikes=True)
        df_clean, spike_count = spike_filter.filter(df)

        self.assertGreaterEqual(spike_count, 2)
        self.assertTrue(df_clean["Is_Spike"].iloc[40])
        self.assertTrue(df_clean["Is_Spike"].iloc[60])

        # Clamped Close at bar 40 must be significantly below 750.0 (near statistical bound)
        self.assertLess(df_clean["Close"].iloc[40], 550.0)
        # Clamped High at bar 60 must be significantly below 900.0
        self.assertLess(df_clean["High"].iloc[60], 550.0)

        # Candlestick physical geometry must hold everywhere
        self.assertTrue((df_clean["High"] >= df_clean["Close"]).all())
        self.assertTrue((df_clean["Low"] <= df_clean["Close"]).all())

    def test_flash_spike_filter_normal_volatility_unclamped(self) -> None:
        """Verify normal market volatility is preserved without false-positive spike clamping."""
        df = self._make_intraday_bar(self.full_times, base_price=500.0)
        spike_filter = FlashSpikeFilter(window=20, threshold_std=4.0, clamp_spikes=True)
        df_clean, spike_count = spike_filter.filter(df)

        self.assertEqual(spike_count, 0)
        np.testing.assert_allclose(df_clean["Close"].values, df["Close"].values)

    def test_etf_spread_aligner(self) -> None:
        """Verify ETF to benchmark alignment, multiplier deduction, basis, and rolling signals."""
        etf_df = self._make_intraday_bar(self.full_times, base_price=510.0)
        # Benchmark index is ~100x the ETF (51,000 INR)
        idx_df = self._make_intraday_bar(self.full_times, base_price=51000.0)

        aligner = ETFBUSpreadAligner(rolling_window=30)
        spread_df = aligner.align(etf_df, idx_df)

        self.assertIsNotNone(spread_df)
        self.assertEqual(len(spread_df), 75)
        self.assertIn("Multiplier", spread_df.columns)
        self.assertIn("Basis_Bps", spread_df.columns)
        self.assertIn("Spread_ZScore", spread_df.columns)
        self.assertIn("Rolling_Correlation", spread_df.columns)
        self.assertIn("Rolling_Beta", spread_df.columns)

        # Deduced multiplier should be approximately 100.0
        self.assertAlmostEqual(spread_df["Multiplier"].iloc[0], 100.0, delta=2.0)
        # All rolling metrics must be finite
        self.assertTrue(np.isfinite(spread_df["Basis_Bps"].values).all())
        self.assertTrue(np.isfinite(spread_df["Spread_ZScore"].values).all())
        self.assertTrue(np.isfinite(spread_df["Rolling_Correlation"].values).all())

    def test_data_cleaning_engine_full_pipeline(self) -> None:
        """Verify DataCleaningEngine transforms IngestedDataset into CleanedDataset contract."""
        df_etf = self._make_intraday_bar(self.full_times, base_price=510.0)
        # Introduce a gap
        df_gappy = df_etf.drop(df_etf.index[[10, 11]])

        df_bench = self._make_intraday_bar(self.full_times, base_price=51000.0)

        ingested = IngestedDataset(
            ticker="BANKBEES.NS",
            interval="5m",
            exchange=ExchangeType.NSE,
            data=df_gappy,
            benchmark_ticker="^NSEBANK",
            benchmark_data=df_bench,
        )

        cleaner = DataCleaningEngine(CleanerConfig.standard())
        cleaned = cleaner.clean(ingested)

        self.assertIsInstance(cleaned, CleanedDataset)
        self.assertEqual(cleaned.bar_count, 75)
        self.assertEqual(cleaned.synthetic_bar_count, 2)
        self.assertIsNotNone(cleaned.spread_data)
        self.assertEqual(len(cleaned.spread_data), 75)
        self.assertTrue(cleaned.quality_report["gap_integrity_passed"])

        summary = cleaned.summary()
        self.assertEqual(summary["ticker"], "BANKBEES.NS")
        self.assertEqual(summary["bars"], 75)
        self.assertEqual(summary["synthetic_bars"], 2)
        self.assertTrue(summary["has_benchmark"])
        self.assertTrue(summary["has_spread"])


if __name__ == "__main__":
    unittest.main()
