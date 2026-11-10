"""Unit tests for Muscle 5: Causal Multi-Timeframe Resampler.
Verifies session-anchored intraday binning, zero cross-session bleeding,
right-labeled causal timestamps, true VWAP calculation, incomplete period sealing,
and the ResampledDataset contract.
"""
from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from ravana.data.cleaner.dataset import CleanedDataset
from ravana.data.ingestion.config import ExchangeType
from ravana.data.resampler.config import ResamplerConfig
from ravana.data.resampler.dataset import ResampledDataset
from ravana.data.resampler.resampler import ResamplingEngine
from ravana.data.resampler.session_resampler import SessionAwareResampler


class TestDataResamplingEngine(unittest.TestCase):
    """Institutional test suite for Muscle 5 Causal Resampler."""

    def setUp(self) -> None:
        """Create multi-day 5m test data for NSE."""
        self.tz = "Asia/Kolkata"
        # 2 regular NSE trading days (2026-03-09, 2026-03-10) -> 150 bars total
        day1 = pd.date_range("2026-03-09 09:15:00", "2026-03-09 15:25:00", freq="5min", tz=self.tz)
        day2 = pd.date_range("2026-03-10 09:15:00", "2026-03-10 15:25:00", freq="5min", tz=self.tz)
        self.two_day_times = day1.union(day2)

    def _make_sample_df(self, times: pd.DatetimeIndex, base_price: float = 500.0) -> pd.DataFrame:
        """Helper to create valid OHLCV series."""
        n = len(times)
        np.random.seed(42)
        close = base_price + np.cumsum(np.random.normal(0, 0.5, n))
        high = close + np.abs(np.random.normal(0.5, 0.2, n))
        low = close - np.abs(np.random.normal(0.5, 0.2, n))
        open_ = (high + low) / 2.0
        volume = np.random.uniform(1000, 10000, n)
        return pd.DataFrame(
            {
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
                "Close_Adj": close,
                "Open_Adj": open_,
                "High_Adj": high,
                "Low_Adj": low,
                "Is_Synthetic": [False] * n,
                "Is_Spike": [False] * n,
            },
            index=times,
        )

    def test_resample_intraday_exact_divisor_15m(self) -> None:
        """Verify 5m -> 15m produces exactly 25 bars per 75-bar session with right-labels."""
        df = self._make_sample_df(self.two_day_times[:75])  # Single 75-bar session
        resampled = SessionAwareResampler.resample_intraday(df, target_minutes=15, base_bar_minutes=5)

        self.assertEqual(len(resampled), 25, "15m on a 375-min session must produce exactly 25 bars")
        self.assertEqual(str(resampled.index[0].time()), "09:30:00", "First 15m bar must close at 09:30:00")
        self.assertEqual(str(resampled.index[-1].time()), "15:30:00", "Last 15m bar must close at 15:30:00")
        self.assertFalse(resampled.isna().any().any(), "Resampled data must contain zero NaNs")

        # Invariant checks
        self.assertTrue((resampled["High"] >= resampled["Low"]).all())
        self.assertTrue((resampled["High"] >= resampled["Open"]).all())
        self.assertTrue((resampled["High"] >= resampled["Close"]).all())
        self.assertTrue((resampled["Low"] <= resampled["Open"]).all())
        self.assertTrue((resampled["Low"] <= resampled["Close"]).all())

    def test_resample_intraday_inexact_divisors_30m_and_60m(self) -> None:
        """Verify 30m produces 13 bars and 60m produces 7 bars with boundary clamping at 15:30."""
        df = self._make_sample_df(self.two_day_times[:75])

        # 30m: 12 x 30m + 1 x 15m stub = 13 bars
        res_30 = SessionAwareResampler.resample_intraday(df, target_minutes=30, base_bar_minutes=5)
        self.assertEqual(len(res_30), 13, "30m on 375-min session must produce 13 bars")
        self.assertEqual(str(res_30.index[0].time()), "09:45:00")
        self.assertEqual(str(res_30.index[-1].time()), "15:30:00", "Last bar must be clamped to 15:30:00")

        # 60m: 6 x 60m + 1 x 15m stub = 7 bars
        res_60 = SessionAwareResampler.resample_intraday(df, target_minutes=60, base_bar_minutes=5)
        self.assertEqual(len(res_60), 7, "60m on 375-min session must produce 7 bars")
        self.assertEqual(str(res_60.index[0].time()), "10:15:00")
        self.assertEqual(str(res_60.index[-1].time()), "15:30:00", "Last 60m bar must be clamped to 15:30:00")

    def test_harmonic_exact_divisors_75m_and_125m(self) -> None:
        """Verify harmonic session divisors produce zero partial stubs."""
        df = self._make_sample_df(self.two_day_times[:75])

        # 75m: exactly 5 bars
        res_75 = SessionAwareResampler.resample_intraday(df, target_minutes=75, base_bar_minutes=5)
        self.assertEqual(len(res_75), 5)
        self.assertEqual(str(res_75.index[-1].time()), "15:30:00")

        # 125m: exactly 3 bars
        res_125 = SessionAwareResampler.resample_intraday(df, target_minutes=125, base_bar_minutes=5)
        self.assertEqual(len(res_125), 3)
        self.assertEqual(str(res_125.index[-1].time()), "15:30:00")

    def test_zero_cross_session_midnight_bleeding(self) -> None:
        """Verify multi-day data never produces ghost bars during non-trading hours."""
        df = self._make_sample_df(self.two_day_times)  # 2 days = 150 bars
        res_15 = SessionAwareResampler.resample_intraday(df, target_minutes=15, base_bar_minutes=5)

        self.assertEqual(len(res_15), 50, "2 days of 15m must equal exactly 50 bars")
        # Check no timestamps exist between 15:30:01 and 09:29:59
        for ts in res_15.index:
            t = ts.time()
            self.assertTrue(pd.Timestamp("09:30:00").time() <= t <= pd.Timestamp("15:30:00").time())

    def test_vwap_turnover_weighting_and_fallback(self) -> None:
        """Verify VWAP is turnover-weighted and falls back cleanly on zero volume."""
        df = self._make_sample_df(self.two_day_times[:3])  # 3 bars (15 mins)
        # Bar 1 has 10,000 volume, Bar 2 and 3 have 0 volume
        df.iloc[0, df.columns.get_loc("Volume")] = 10000.0
        df.iloc[1, df.columns.get_loc("Volume")] = 0.0
        df.iloc[2, df.columns.get_loc("Volume")] = 0.0

        res = SessionAwareResampler.resample_intraday(df, target_minutes=15, base_bar_minutes=5)
        expected_vwap = (df.iloc[0]["High"] + df.iloc[0]["Low"] + df.iloc[0]["Close"]) / 3.0
        self.assertAlmostEqual(res.iloc[0]["VWAP"], expected_vwap, places=4)

        # Zero volume total fallback
        df["Volume"] = 0.0
        res_zero = SessionAwareResampler.resample_intraday(df, target_minutes=15, base_bar_minutes=5)
        self.assertAlmostEqual(res_zero.iloc[0]["VWAP"], res_zero.iloc[0]["Close"], places=4)

    def test_quality_flags_aggregation(self) -> None:
        """Verify Is_Synthetic requires ALL constituent bars and Is_Spike requires ANY."""
        df = self._make_sample_df(self.two_day_times[:3])
        # Bar 1 is synthetic, Bar 2 and 3 are real
        df.iloc[0, df.columns.get_loc("Is_Synthetic")] = True
        # Bar 2 is a spike
        df.iloc[1, df.columns.get_loc("Is_Spike")] = True

        res = SessionAwareResampler.resample_intraday(df, target_minutes=15, base_bar_minutes=5)
        # Not all are synthetic -> Is_Synthetic should be False
        self.assertFalse(res.iloc[0]["Is_Synthetic"])
        # Synthetic ratio should be 1/3
        self.assertAlmostEqual(res.iloc[0]["Synthetic_Ratio"], 1.0 / 3.0, places=4)
        # At least one spike -> Is_Spike should be True
        self.assertTrue(res.iloc[0]["Is_Spike"])

    def test_daily_to_weekly_with_causal_sealing(self) -> None:
        """Verify daily to weekly drops active unclosed week when require_closed_period is True."""
        dates = pd.date_range("2026-03-02", "2026-03-11", freq="B", tz=self.tz)  # Mon 2nd to Wed 11th
        df = self._make_sample_df(dates, base_price=100.0)

        # With require_closed_period=True, Wed Mar 11 is in-progress and must be dropped
        res_sealed = SessionAwareResampler.resample_daily_to_higher(
            df, freq="W-FRI", require_closed_period=True
        )
        self.assertEqual(len(res_sealed), 1, "Only Week 1 (ended Mar 6) is sealed")
        self.assertEqual(res_sealed.iloc[0]["Bar_Count"], 5)

        # With require_closed_period=False, active week is retained
        res_unsealed = SessionAwareResampler.resample_daily_to_higher(
            df, freq="W-FRI", require_closed_period=False
        )
        self.assertEqual(len(res_unsealed), 2)
        self.assertEqual(res_unsealed.iloc[1]["Bar_Count"], 3)

    def test_point_in_time_asof_merge(self) -> None:
        """Verify backward as-of join prevents lookahead into unclosed higher-timeframe bars."""
        daily_dates = pd.date_range("2026-03-02", "2026-03-11", freq="B", tz=self.tz)
        daily_df = self._make_sample_df(daily_dates, base_price=100.0)

        weekly_df = SessionAwareResampler.resample_daily_to_higher(
            daily_df, freq="W-FRI", require_closed_period=False
        )
        weekly_df["Weekly_SMA"] = weekly_df["Close"].rolling(1).mean()

        merged = SessionAwareResampler.point_in_time_merge(
            base_df=daily_df,
            higher_df=weekly_df,
            higher_cols=["Weekly_SMA"],
        )
        self.assertIn("Weekly_SMA", merged.columns)
        # Week 1 closed on Friday 2026-03-06.
        # Monday 2026-03-09 must observe Friday 2026-03-06 weekly close.
        fri_weekly = weekly_df.loc["2026-03-06"]["Weekly_SMA"]
        mon_merged = merged.loc["2026-03-09"]["Weekly_SMA"]
        self.assertEqual(fri_weekly, mon_merged)

    def test_resampling_engine_full_contract(self) -> None:
        """Verify ResamplingEngine produces immutable ResampledDataset contract with all timeframes."""
        df_5m = self._make_sample_df(self.two_day_times)
        cleaned = CleanedDataset(
            ticker="BANKBEES.NS",
            interval="5m",
            exchange=ExchangeType.NSE,
            data=df_5m,
            benchmark_ticker="^NSEBANK",
            benchmark_data=df_5m.copy(),
            spread_data=df_5m.copy(),
        )

        config = ResamplerConfig(intraday_intervals=("15m", "30m", "1h"))
        engine = ResamplingEngine(config)
        dataset = engine.resample_dataset(cleaned)

        self.assertIsInstance(dataset, ResampledDataset)
        self.assertEqual(dataset.ticker, "BANKBEES.NS")
        self.assertIn("5m", dataset.timeframes)
        self.assertIn("15m", dataset.timeframes)
        self.assertIn("30m", dataset.timeframes)
        self.assertIn("1h", dataset.timeframes)
        self.assertEqual(len(dataset.timeframes["15m"]), 50)
        self.assertEqual(len(dataset.timeframes["30m"]), 26)
        self.assertEqual(len(dataset.timeframes["1h"]), 14)

    def test_muhurat_trading_session_resampling(self) -> None:
        """Verify Diwali Muhurat session (18:15-19:15) resamples without NaN or clamping bugs."""
        times = pd.date_range("2025-11-01 18:15:00", "2025-11-01 19:10:00", freq="5min", tz=self.tz)
        df = self._make_sample_df(times)

        for target in [15, 30, 60, 75, 125]:
            res = SessionAwareResampler.resample_intraday(df, target_minutes=target, base_bar_minutes=5)
            self.assertFalse(res.isna().any().any())
            self.assertEqual(res.index[-1].strftime("%H:%M:%S"), "19:15:00")

    def test_disaster_recovery_two_session_day_no_crash(self) -> None:
        """Verify multi-session day (DR switchover 09:15-10:00 and 11:30-12:30) does not crash on NaNs."""
        t1 = pd.date_range("2024-05-18 09:15", "2024-05-18 09:55", freq="5min", tz=self.tz)
        t2 = pd.date_range("2024-05-18 11:30", "2024-05-18 12:25", freq="5min", tz=self.tz)
        df = self._make_sample_df(t1.union(t2))

        res = SessionAwareResampler.resample_intraday(df, target_minutes=15, base_bar_minutes=5)
        self.assertFalse(res.isna().any().any())
        self.assertEqual(len(res), 7)

    def test_good_friday_and_weekend_month_end_causal_sealing(self) -> None:
        """Verify holiday 4-day weeks and weekend month-ends are NOT dropped as unclosed."""
        dates_thurs = pd.date_range("2026-03-30", "2026-04-02", freq="B", tz=self.tz)
        df_thurs = self._make_sample_df(dates_thurs)
        res_wk = SessionAwareResampler.resample_daily_to_higher(df_thurs, freq="W-FRI", require_closed_period=True)
        self.assertEqual(len(res_wk), 1)

        dates_may = pd.bdate_range("2026-05-01", "2026-05-29", tz=self.tz)
        df_may = self._make_sample_df(dates_may)
        res_mo = SessionAwareResampler.resample_daily_to_higher(df_may, freq="ME", require_closed_period=True)
        self.assertEqual(len(res_mo), 1)

    def test_point_in_time_merge_zero_intraday_lookahead(self) -> None:
        """Verify intraday bars on Friday morning NEVER observe Friday's weekly close."""
        daily_dates = pd.date_range("2026-03-02", "2026-03-06", freq="B", tz=self.tz)
        daily_df = self._make_sample_df(daily_dates, base_price=100.0)
        daily_df.iloc[-1, daily_df.columns.get_loc("Close")] = 999.0

        weekly_df = SessionAwareResampler.resample_daily_to_higher(daily_df, freq="W-FRI", require_closed_period=False)

        intraday_times = pd.date_range("2026-03-06 09:15:00", "2026-03-06 10:00:00", freq="5min", tz=self.tz)
        intraday_df = self._make_sample_df(intraday_times, base_price=100.0)

        merged = SessionAwareResampler.point_in_time_merge(intraday_df, weekly_df[["Close"]])
        self.assertNotEqual(merged.iloc[0]["Close_y"], 999.0)

    def test_spread_aligner_adaptive_window_scaling(self) -> None:
        """Verify ResamplingEngine adapts spread aligner window across timeframes."""
        engine = ResamplingEngine()
        aligner_5m = engine._get_spread_aligner(5)
        aligner_15m = engine._get_spread_aligner(15)
        aligner_60m = engine._get_spread_aligner(60)

        self.assertEqual(aligner_5m.rolling_window, 60)
        self.assertEqual(aligner_15m.rolling_window, 20)
        self.assertEqual(aligner_60m.rolling_window, 10)

    def test_vectorized_resampling_performance(self) -> None:
        """Verify 100 sessions resample in under 1.5 seconds."""
        import time
        days = pd.bdate_range("2025-01-01", periods=100)
        all_times = [pd.date_range(d.strftime("%Y-%m-%d") + " 09:15:00", d.strftime("%Y-%m-%d") + " 15:25:00", freq="5min", tz=self.tz) for d in days]
        idx = pd.DatetimeIndex(np.concatenate(all_times))
        df = self._make_sample_df(idx)

        t0 = time.perf_counter()
        res = SessionAwareResampler.resample_intraday(df, target_minutes=15)
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 1.5, f"Vectorized resample must complete <1.5s, took {elapsed:.3f}s")
        self.assertEqual(len(res), 2500)


if __name__ == "__main__":
    unittest.main()
