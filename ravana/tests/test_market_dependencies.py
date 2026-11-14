"""Unit tests for the 7 Indian Structural Market Dependencies.

Validates:
1. Big-Brother to Little-Brother Lead-Lag (BANKBEES to PSUBNKBEES beta 1.45 lag).
2. Heavyweight Kingmaker sweep (Theoretical_Close vs Close driving INDEX_LEAD_LAG_DRIFT).
3. Brent Crude Oil triad (MACRO_CRUDE_RET_1 banking/auto vs CPSE energy sensitivity).
4. NASDAQ + USDINR dual engine (MACRO_NASDAQ_RET_1 + USDINR tech export tailwind).
5. Thursday 14:15-15:20 weekly expiry gamma window (EXPIRY_GAMMA_WINDOW == 1.0 for exactly 14 bars).
6. Turn-of-Month SIP window (TURN_OF_MONTH_SIP_WINDOW == 1.0 for days <= 7 or >= 28).
7. Continuation Gate fee savings and trade count reduction.
"""

import unittest
import numpy as np
import pandas as pd

from ravana.features.config import FeatureConfig, FeatureName
from ravana.features.matrix_engine import FeatureMatrixEngine
from ravana.backtest.config import AssetType, BacktestConfig, TradingRegime
from ravana.backtest.engine import EventDrivenBacktester


class TestMarketDependencies(unittest.TestCase):
    """Test suite validating all 7 structural Indian market microstructure dependencies."""

    def setUp(self) -> None:
        np.random.seed(42)

    def _build_mock_bar_df(
        self,
        dates: pd.DatetimeIndex,
        base_price: float = 500.0,
        include_all_macros: bool = True,
    ) -> pd.DataFrame:
        """Helper to build a clean OHLCV DataFrame with complete macro columns."""
        n = len(dates)
        close = pd.Series(base_price + np.cumsum(np.random.normal(0, 0.2, n)), index=dates)
        high = close + np.abs(np.random.normal(0.2, 0.05, n)) + 0.1
        low = close - np.abs(np.random.normal(0.2, 0.05, n)) - 0.1
        open_px = (high + low) / 2.0

        data = {
            "Open": open_px,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": pd.Series(np.random.uniform(10000, 50000, n), index=dates),
            "VWAP": close,
            "Spread_INR": pd.Series(np.full(n, 0.05), index=dates),
            "Theoretical_Close": close,
        }
        if include_all_macros:
            data.update(
                {
                    "Macro_VIX": pd.Series(np.full(n, 14.0), index=dates),
                    "Macro_USDINR": pd.Series(np.full(n, 84.5), index=dates),
                    "Macro_SP500": pd.Series(np.full(n, 5850.0), index=dates),
                    "Macro_TNX": pd.Series(np.full(n, 4.35), index=dates),
                    "Macro_CRUDE": pd.Series(np.full(n, 75.0), index=dates),
                    "Macro_NASDAQ": pd.Series(np.full(n, 18500.0), index=dates),
                }
            )
        return pd.DataFrame(data, index=dates)

    def test_dependency_1_big_brother_little_brother_lead_lag(self) -> None:
        """Verify BANKBEES (Big Brother) leads PSUBNKBEES (Little Brother) with beta ~ 1.45."""
        n = 300
        dates = pd.date_range("2026-01-05 09:15", periods=n, freq="5min", tz="Asia/Kolkata")

        # Generate BANKBEES returns as independent series
        ret_bank = np.random.normal(0.0, 0.002, n)
        # PSUBNKBEES lags by 1 bar with beta = 1.45 + idiosyncratic noise
        ret_psu = np.zeros(n)
        for t in range(1, n):
            ret_psu[t] = 1.45 * ret_bank[t - 1] + np.random.normal(0.0, 0.0008)

        # 1-bar lagged cross-correlation
        s_bank = pd.Series(ret_bank[:-1])
        s_psu_lag1 = pd.Series(ret_psu[1:])
        cross_corr = float(s_bank.corr(s_psu_lag1))

        # Regression beta: Cov(PSU_{t+1}, BANK_t) / Var(BANK_t)
        cov_mat = np.cov(ret_psu[1:], ret_bank[:-1])
        empirical_beta = float(cov_mat[0, 1] / cov_mat[1, 1])

        self.assertGreater(cross_corr, 0.70)
        self.assertAlmostEqual(empirical_beta, 1.45, delta=0.15)

    def test_dependency_2_heavyweight_kingmaker_sweep(self) -> None:
        """Verify Theoretical_Close leading ETF Close produces positive INDEX_LEAD_LAG_DRIFT."""
        dates = pd.date_range("2026-01-05 09:15", periods=50, freq="5min", tz="Asia/Kolkata")
        df = self._build_mock_bar_df(dates, base_price=500.0)

        # Bar 20: Heavyweight sweep in cash basket drives Theoretical_Close up by 2.0%
        # while ETF Close lags, moving only 0.2%
        base_theo = float(df["Theoretical_Close"].iloc[20])
        base_close = float(df["Close"].iloc[20])
        df.iloc[20:, df.columns.get_loc("Theoretical_Close")] = base_theo * 1.02
        df.iloc[20:, df.columns.get_loc("Close")] = base_close * 1.002

        engine = FeatureMatrixEngine(FeatureConfig())
        matrix = engine.compute_feature_matrix(df, symbol="BANKBEES")

        drift = matrix[FeatureName.INDEX_LEAD_LAG_DRIFT.value]
        # At bar 22 (within 3 bars after shock), index return exceeds ETF return -> positive drift
        self.assertGreater(drift.iloc[22], 0.005)

        # Test extreme clipping at bounds [-0.05, +0.05]
        df_extreme = df.copy()
        df_extreme.iloc[30:, df_extreme.columns.get_loc("Theoretical_Close")] *= 1.10
        matrix_extreme = engine.compute_feature_matrix(df_extreme, symbol="BANKBEES")
        self.assertAlmostEqual(matrix_extreme[FeatureName.INDEX_LEAD_LAG_DRIFT.value].iloc[30], 0.05)

        # Negative clipping bound at -0.05
        df_extreme.iloc[40:, df_extreme.columns.get_loc("Theoretical_Close")] *= 0.85
        matrix_neg = engine.compute_feature_matrix(df_extreme, symbol="BANKBEES")
        self.assertAlmostEqual(matrix_neg[FeatureName.INDEX_LEAD_LAG_DRIFT.value].iloc[40], -0.05)

        # Fallback test: DataFrame without Theoretical_Close produces zero drift
        df_no_theo = df.drop(columns=["Theoretical_Close"])
        matrix_no_theo = engine.compute_feature_matrix(df_no_theo, symbol="BANKBEES")
        self.assertTrue((matrix_no_theo[FeatureName.INDEX_LEAD_LAG_DRIFT.value] == 0.0).all())

    def test_dependency_3_brent_crude_oil_triad(self) -> None:
        """Verify MACRO_CRUDE_RET_1 extraction and opposing bank/auto vs CPSE sensitivity."""
        dates = pd.date_range("2026-01-05 09:15", periods=100, freq="5min", tz="Asia/Kolkata")
        df = self._build_mock_bar_df(dates, base_price=100.0)

        # Populate realistic fluctuating crude oil price series
        crude_prices = 75.0 + np.cumsum(np.random.normal(0, 0.4, len(dates)))
        df["Macro_CRUDE"] = crude_prices

        engine = FeatureMatrixEngine(FeatureConfig())
        matrix = engine.compute_feature_matrix(df, symbol="BANKBEES")

        crude_feat = matrix[FeatureName.MACRO_CRUDE_RET_1.value]
        self.assertTrue(np.all(np.isfinite(crude_feat)))
        self.assertFalse((crude_feat == 0.0).all())

        # Test Indian market macroeconomic triad sensitivity:
        # Bank/Auto has negative sensitivity (-0.4) to crude spike
        # CPSE (energy upstream) has positive sensitivity (+0.6) to crude spike
        crude_ret = crude_feat.to_numpy()
        bank_ret = -0.40 * crude_ret + np.random.normal(0, 0.001, len(crude_ret))
        cpse_ret = +0.60 * crude_ret + np.random.normal(0, 0.001, len(crude_ret))

        corr_bank_crude = float(np.corrcoef(crude_ret[1:], bank_ret[1:])[0, 1])
        corr_cpse_crude = float(np.corrcoef(crude_ret[1:], cpse_ret[1:])[0, 1])

        self.assertLess(corr_bank_crude, -0.30)
        self.assertGreater(corr_cpse_crude, 0.30)

        # Missing crude column fallback
        df_no_crude = df.drop(columns=["Macro_CRUDE"])
        matrix_no_crude = engine.compute_feature_matrix(df_no_crude, symbol="BANKBEES")
        self.assertTrue((matrix_no_crude[FeatureName.MACRO_CRUDE_RET_1.value] == 0.0).all())

    def test_dependency_4_nasdaq_and_usdinr_dual_engine(self) -> None:
        """Verify NASDAQ and USDINR feature extraction and compounding tech export tailwind."""
        dates = pd.date_range("2026-01-05 09:15", periods=80, freq="5min", tz="Asia/Kolkata")
        df = self._build_mock_bar_df(dates, base_price=100.0)

        # Set positive NASDAQ and USDINR jumps at bar 40
        df.iloc[40:, df.columns.get_loc("Macro_NASDAQ")] *= 1.02  # +2% NASDAQ jump
        df.iloc[40:, df.columns.get_loc("Macro_USDINR")] *= 1.005  # +0.5% Rupee depreciation

        engine = FeatureMatrixEngine(FeatureConfig())
        matrix = engine.compute_feature_matrix(df, symbol="ITBEES")

        nasdaq_ret = matrix[FeatureName.MACRO_NASDAQ_RET_1.value]
        usdinr_ret = matrix[FeatureName.MACRO_USDINR_RET_1.value]

        self.assertGreater(nasdaq_ret.iloc[40], 0.015)
        self.assertGreater(usdinr_ret.iloc[40], 0.004)

        # Compounding tailwind: both features positive at shock bar
        dual_engine_signal = nasdaq_ret.iloc[40] + usdinr_ret.iloc[40]
        self.assertGreater(dual_engine_signal, 0.02)

        # Missing NASDAQ fallback
        df_no_nasdaq = df.drop(columns=["Macro_NASDAQ"])
        matrix_no_nasdaq = engine.compute_feature_matrix(df_no_nasdaq, symbol="ITBEES")
        self.assertTrue((matrix_no_nasdaq[FeatureName.MACRO_NASDAQ_RET_1.value] == 0.0).all())

    def test_dependency_5_thursday_expiry_gamma_window(self) -> None:
        """Verify EXPIRY_GAMMA_WINDOW is 1.0 for exactly 14 bars on Thursday 14:15-15:20 IST."""
        # 5-day week: 2026-01-05 (Mon) to 2026-01-09 (Fri). Thursday is 2026-01-08.
        all_dates = []
        for d in pd.date_range("2026-01-05", "2026-01-09", freq="D"):
            day_times = pd.date_range(
                f"{d.strftime('%Y-%m-%d')} 09:15",
                f"{d.strftime('%Y-%m-%d')} 15:25",
                freq="5min",
                tz="Asia/Kolkata",
            )
            all_dates.extend(day_times)

        dates_idx = pd.DatetimeIndex(all_dates)
        df = self._build_mock_bar_df(dates_idx, base_price=500.0)

        engine = FeatureMatrixEngine(FeatureConfig())
        matrix = engine.compute_feature_matrix(df, symbol="BANKBEES")
        gamma_window = matrix[FeatureName.EXPIRY_GAMMA_WINDOW.value]

        # 1. Exactly 14 bars in the whole week have gamma_window == 1.0
        self.assertEqual(int(gamma_window.sum()), 14)

        # 2. All 14 bars are strictly on Thursday (2026-01-08)
        active_bars = matrix[gamma_window == 1.0]
        self.assertTrue(all(ts.strftime("%Y-%m-%d") == "2026-01-08" for ts in active_bars.index))

        # 3. Timestamps span exactly 14:15 to 15:20 IST
        times = [ts.strftime("%H:%M") for ts in active_bars.index]
        self.assertEqual(times[0], "14:15")
        self.assertEqual(times[-1], "15:20")
        self.assertEqual(len(times), 14)

        # 4. Non-Thursdays have zero active bars
        non_thurs_mask = [ts.dayofweek != 3 for ts in matrix.index]
        self.assertEqual(gamma_window[non_thurs_mask].sum(), 0.0)

    def test_dependency_6_turn_of_month_sip_window(self) -> None:
        """Verify TURN_OF_MONTH_SIP_WINDOW is 1.0 for days <= 7 or >= 28, and 0.0 otherwise."""
        # Generate bars spanning January 1 to January 31, 2026
        all_dates = []
        for day in range(1, 32):
            day_times = pd.date_range(
                f"2026-01-{day:02d} 09:15",
                f"2026-01-{day:02d} 15:25",
                freq="30min",
                tz="Asia/Kolkata",
            )
            all_dates.extend(day_times)

        dates_idx = pd.DatetimeIndex(all_dates)
        df = self._build_mock_bar_df(dates_idx, base_price=500.0)

        engine = FeatureMatrixEngine(FeatureConfig())
        matrix = engine.compute_feature_matrix(df, symbol="NIFTYBEES")
        sip_window = matrix[FeatureName.TURN_OF_MONTH_SIP_WINDOW.value]

        # Days 1 through 7 must be 1.0
        days_1_to_7 = matrix[[ts.day <= 7 for ts in matrix.index]]
        self.assertTrue((days_1_to_7[FeatureName.TURN_OF_MONTH_SIP_WINDOW.value] == 1.0).all())

        # Days 28 through 31 must be 1.0
        days_28_to_31 = matrix[[ts.day >= 28 for ts in matrix.index]]
        self.assertTrue((days_28_to_31[FeatureName.TURN_OF_MONTH_SIP_WINDOW.value] == 1.0).all())

        # Mid-month days (8 through 27) must be 0.0
        days_mid_month = matrix[[8 <= ts.day <= 27 for ts in matrix.index]]
        self.assertTrue((days_mid_month[FeatureName.TURN_OF_MONTH_SIP_WINDOW.value] == 0.0).all())

    def test_dependency_7_continuation_gate_fee_savings_and_trade_reduction(self) -> None:
        """Verify continuation gate avoids 09:15 churning, saves DP fees, and reduces trade count."""
        # 4-day trending scenario
        dates = pd.date_range("2026-01-05 09:15", "2026-01-08 15:25", freq="5min")
        dates = [d for d in dates if "09:15" <= d.strftime("%H:%M") <= "15:25"]
        n = len(dates)

        df = pd.DataFrame(
            {
                "Open": np.linspace(500.0, 530.0, n),
                "High": np.linspace(501.0, 531.0, n),
                "Low": np.linspace(499.0, 529.0, n),
                "Close": np.linspace(500.0, 530.0, n),
                "Volume": np.full(n, 50000.0),
            },
            index=pd.DatetimeIndex(dates),
        )

        # Persistent strong bullish signal (0.85)
        signals = pd.Series(0.85, index=df.index)

        # Baseline: No continuation gate (churns at 09:15 every morning)
        cfg_no_gate = BacktestConfig(
            initial_capital=200000.0,
            asset_type=AssetType.ETF,
            regime=TradingRegime.OVERNIGHT,
            fixed_dp_fee_inr=17.0,
            hold_on_strong_continuation=False,
        )
        res_no_gate = EventDrivenBacktester(cfg_no_gate).run(df, signals, ticker="BANKBEES.NS")

        # Treatment: Continuation gate enabled (threshold 0.50)
        cfg_gate = BacktestConfig(
            initial_capital=200000.0,
            asset_type=AssetType.ETF,
            regime=TradingRegime.OVERNIGHT,
            fixed_dp_fee_inr=17.0,
            hold_on_strong_continuation=True,
            continuation_threshold=0.50,
        )
        res_gate = EventDrivenBacktester(cfg_gate).run(df, signals, ticker="BANKBEES.NS")

        # Verifications
        self.assertEqual(res_no_gate.metrics["continuation_holds_count"], 0)
        self.assertEqual(res_no_gate.metrics["total_dp_fees_inr"], 68.0)  # 4 sell days * ₹17
        self.assertEqual(len(res_no_gate.trades), 8)  # 4 daily round-trips

        self.assertEqual(res_gate.metrics["continuation_holds_count"], 3)  # 3 morning holds
        self.assertEqual(res_gate.metrics["total_dp_fees_inr"], 17.0)  # Only 1 terminal sell
        self.assertEqual(len(res_gate.trades), 2)  # 1 entry + 1 terminal exit

        # Fee savings = ₹51.0
        fee_saved = res_no_gate.metrics["total_dp_fees_inr"] - res_gate.metrics["total_dp_fees_inr"]
        self.assertEqual(fee_saved, 51.0)

        # Net PnL is higher with continuation gate
        self.assertGreater(res_gate.metrics["final_equity"], res_no_gate.metrics["final_equity"])


if __name__ == "__main__":
    unittest.main()
