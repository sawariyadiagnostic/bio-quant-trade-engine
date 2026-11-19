"""Tests for Muscle 7: Market Vocabulary Feature Matrix Engine.

Verifies:
1. Rolling ADF Stationarity and OU Half-Life estimation.
2. Causal technical and microstructure feature primitives.
3. FeatureMatrixEngine full 21-feature pipeline on single and multi-asset universes.
4. Strict zero-lookahead causality and finite numerical bounds.
5. Integration with Muscle 6 sovereign Parquet lake.
"""

import unittest
import numpy as np
import pandas as pd
import torch

from ravana.features.config import FeatureConfig, FeatureRegistry
from ravana.features.stationarity import FastFractionalDiff, RollingADF, RollingOUHalfLife
from ravana.features.technical_primitives import (
    log_returns,
    realized_volatility,
    parkinson_volatility,
    garman_klass_volatility,
    normalized_atr,
    vwap_drift,
    lagged_volume_zscore,
    relative_strength_index,
    bollinger_pct,
    macd_normalized,
    tracking_spread_bps,
)
from ravana.features.matrix_engine import FeatureMatrixEngine


class TestFeatureStationarity(unittest.TestCase):
    """Test ADF and Ornstein-Uhlenbeck estimators on synthetic series."""

    def setUp(self) -> None:
        np.random.seed(42)
        self.n_bars = 200

        # Stationary AR(1) process: x_t = 0.5 * x_{t-1} + e_t (Half life = -ln(2)/ln(0.5) = 1.0 bar)
        e = np.random.normal(0, 1, self.n_bars)
        x = np.zeros(self.n_bars)
        for t in range(1, self.n_bars):
            x[t] = 0.5 * x[t - 1] + e[t]
        self.mean_reverting_series = pd.Series(x)

        # Random walk: w_t = w_{t-1} + e_t
        self.random_walk = pd.Series(np.cumsum(e))

    def test_rolling_adf_statistic_sensitivity(self) -> None:
        """Verify ADF t-stat is significantly negative on stationary series."""
        adf_calc = RollingADF(window=60, max_lags=1)
        stat_stat = adf_calc.compute(self.mean_reverting_series)
        stat_rw = adf_calc.compute(self.random_walk)

        self.assertEqual(len(stat_stat), self.n_bars)
        self.assertTrue(np.all(np.isfinite(stat_stat)))
        # Mean reverting series t-stat should be more negative than random walk on average
        valid_stat = stat_stat.iloc[60:]
        valid_rw = stat_rw.iloc[60:]
        self.assertLess(valid_stat.mean(), valid_rw.mean())

    def test_rolling_ou_half_life(self) -> None:
        """Verify OU half-life estimates mean-reversion speed within bounded limits."""
        ou_calc = RollingOUHalfLife(window=60, min_half_life=1.0, max_half_life=252.0)
        hl_stat = ou_calc.compute(self.mean_reverting_series)
        hl_rw = ou_calc.compute(self.random_walk)

        self.assertTrue(np.all(np.isfinite(hl_stat)))
        self.assertTrue(np.all(hl_stat >= 1.0))
        self.assertTrue(np.all(hl_stat <= 252.0))
        # AR(1) with phi=0.5 has theoretical half-life = 1.0. Rolling estimate should be small (< 10)
        self.assertLess(hl_stat.iloc[60:].median(), 10.0)
        # Random walk half-life should hit upper bounds
        self.assertGreater(hl_rw.iloc[60:].median(), hl_stat.iloc[60:].median())

    def test_constant_series_zero_crash(self) -> None:
        """Verify zero variance does not raise ZeroDivisionError or produce NaNs."""
        const_series = pd.Series(np.full(100, 50.0))
        adf_calc = RollingADF(window=30)
        ou_calc = RollingOUHalfLife(window=30)

        adf_res = adf_calc.compute(const_series)
        ou_res = ou_calc.compute(const_series)

        self.assertTrue(np.all(np.isfinite(adf_res)))
        self.assertTrue(np.all(np.isfinite(ou_res)))

    def test_fractional_diff_weights_and_memory(self) -> None:
        """Verify FastFractionalDiff maintains high memory correlation while differencing."""
        diff_0 = FastFractionalDiff(d=0.0)
        diff_1 = FastFractionalDiff(d=1.0)
        diff_40 = FastFractionalDiff(d=0.40, threshold=1e-4)

        # d=0 should have single weight 1.0
        self.assertEqual(len(diff_0.weights), 1)
        self.assertAlmostEqual(diff_0.weights[0], 1.0)

        # d=1 should have weights [1.0, -1.0]
        self.assertEqual(len(diff_1.weights), 2)
        self.assertAlmostEqual(diff_1.weights[0], 1.0)
        self.assertAlmostEqual(diff_1.weights[1], -1.0)

        # d=0.25 on random walk preserves memory correlation >= 0.75
        diff_25 = FastFractionalDiff(d=0.25, threshold=1e-4)
        frac_series_25 = diff_25.compute(self.random_walk)
        corr_25 = diff_25.memory_correlation(self.random_walk.iloc[30:], frac_series_25.iloc[30:])
        self.assertGreater(corr_25, 0.75)

        # d=0.40 preserves substantial memory (>0.55), far higher than d=1.0 (~0.0)
        frac_series_40 = diff_40.compute(self.random_walk)
        corr_40 = diff_40.memory_correlation(self.random_walk.iloc[30:], frac_series_40.iloc[30:])
        self.assertGreater(corr_40, 0.55)
        self.assertTrue(np.all(np.isfinite(frac_series_40)))

    def test_fractional_diff_stationarity_gain(self) -> None:
        """Verify frac diff improves stationarity t-statistic relative to raw random walk."""
        diff_40 = FastFractionalDiff(d=0.40)
        frac_rw = diff_40.compute(self.random_walk)
        adf_calc = RollingADF(window=60)
        raw_adf = adf_calc.compute(self.random_walk)
        frac_adf = adf_calc.compute(frac_rw)

        # Frac diff ADF t-stat should be more negative than raw random walk
        self.assertLess(frac_adf.iloc[60:].mean(), raw_adf.iloc[60:].mean())



class TestTechnicalPrimitives(unittest.TestCase):
    """Test microstructural and volatility feature calculations."""

    def setUp(self) -> None:
        np.random.seed(42)
        dates = pd.date_range("2026-01-01 09:15", periods=100, freq="5min")
        self.close = pd.Series(100.0 + np.cumsum(np.random.normal(0, 0.5, 100)), index=dates)
        self.high = self.close + np.abs(np.random.normal(0, 0.3, 100)) + 0.1
        self.low = self.close - np.abs(np.random.normal(0, 0.3, 100)) - 0.1
        self.open = (self.high + self.low) / 2.0
        self.volume = pd.Series(np.random.uniform(1000, 5000, 100), index=dates)
        self.vwap = self.close.rolling(10, min_periods=1).mean()

    def test_parkinson_and_garman_klass_volatility(self) -> None:
        """Verify Parkinson and Garman-Klass volatility estimators are non-negative and finite."""
        p_vol = parkinson_volatility(self.high, self.low, window=20)
        gk_vol = garman_klass_volatility(self.open, self.high, self.low, self.close, window=20)

        self.assertTrue(np.all(np.isfinite(p_vol)))
        self.assertTrue(np.all(np.isfinite(gk_vol)))
        self.assertTrue(np.all(p_vol >= 0.0))
        self.assertTrue(np.all(gk_vol >= 0.0))

    def test_lagged_volume_zscore_no_lookahead(self) -> None:
        """Verify lagged volume z-score at t uses volume only up to t-1."""
        vol_z = lagged_volume_zscore(self.volume, window=20, lag=1)
        self.assertTrue(np.all(np.isfinite(vol_z)))

        # Mutate bar t=50 volume drastically; bar t=50 vol_z must be UNCHANGED
        vol_mutated = self.volume.copy()
        vol_mutated.iloc[50] = 999999.0
        vol_z_mutated = lagged_volume_zscore(vol_mutated, window=20, lag=1)

        self.assertAlmostEqual(vol_z.iloc[50], vol_z_mutated.iloc[50], places=5)
        # Bar 51 must reflect the shock
        self.assertNotEqual(vol_z.iloc[51], vol_z_mutated.iloc[51])

    def test_vwap_drift_scaling(self) -> None:
        """Verify vwap drift measures fractional distance from benchmark price."""
        drift = vwap_drift(self.close, self.vwap)
        self.assertTrue(np.all(np.isfinite(drift)))
        # Fractional drift for 0.5 step price changes should stay within +/- 10%
        self.assertTrue(np.all(np.abs(drift) < 0.10))


class TestFeatureMatrixEngine(unittest.TestCase):
    """Test full FeatureMatrixEngine pipeline, matrix contracts, and tensor outputs."""

    def setUp(self) -> None:
        np.random.seed(42)
        n = 150
        dates = pd.date_range("2026-01-05 09:15", periods=n, freq="5min", tz="Asia/Kolkata")
        base_price = 50.0 + np.cumsum(np.random.normal(0, 0.2, n))

        # Synthetic ETF dataset with all required upstream columns
        self.df = pd.DataFrame(
            {
                "Open": base_price + np.random.normal(0, 0.05, n),
                "High": base_price + np.abs(np.random.normal(0.2, 0.05, n)),
                "Low": base_price - np.abs(np.random.normal(0.2, 0.05, n)),
                "Close": base_price,
                "Volume": np.random.uniform(5000, 20000, n),
                "VWAP": base_price + np.random.normal(0, 0.05, n),
                "Spread_INR": np.random.normal(0.05, 0.02, n),
                "Theoretical_Close": base_price - np.random.normal(0.05, 0.02, n),
                "Macro_VIX": np.full(n, 14.5) + np.random.normal(0, 0.2, n),
                "Macro_USDINR": np.full(n, 84.0) + np.random.normal(0, 0.1, n),
                "Macro_SP500": np.full(n, 5800.0) + np.random.normal(0, 5.0, n),
                "Macro_TNX": np.full(n, 4.40) + np.random.normal(0, 0.02, n),
                "Macro_CRUDE": np.full(n, 75.0) + np.random.normal(0, 0.5, n),
                "Macro_NASDAQ": np.full(n, 18000.0) + np.random.normal(0, 20.0, n),
            },
            index=dates,
        )

    def test_36_stationary_features_presence(self) -> None:
        """Verify engine computes all 36 registered stationary features without NaNs."""
        engine = FeatureMatrixEngine(FeatureConfig())
        matrix = engine.compute_feature_matrix(self.df, symbol="BANKBEES")

        expected_features = FeatureRegistry.all_features()
        self.assertEqual(len(expected_features), 36)

        for feat in expected_features:
            self.assertIn(feat, matrix.columns, f"Feature {feat} missing from matrix output")

        # Zero NaNs or Infs allowed in institutional contract
        self.assertFalse(matrix.isna().any().any(), "Matrix contains NaN values")
        self.assertTrue(np.all(np.isfinite(matrix.values)), "Matrix contains non-finite values")
        self.assertEqual(len(matrix), len(self.df))

    def test_multi_asset_3d_tensor_dispatch(self) -> None:
        """Verify compute_multi_asset_tensor produces [B, T, K] float32 tensor with 36 features."""
        engine = FeatureMatrixEngine(FeatureConfig())
        df2 = self.df.copy()
        df2["Close"] = df2["Close"] * 1.5

        universe = {"BANKBEES": self.df, "NIFTYBEES": df2}
        tensor, symbols, features, time_index = engine.compute_multi_asset_tensor(universe)

        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tensor.shape, (2, len(self.df), 36))
        self.assertEqual(symbols, ["BANKBEES", "NIFTYBEES"])
        self.assertEqual(len(features), 36)
        self.assertEqual(len(time_index), len(self.df))
        self.assertTrue(torch.isfinite(tensor).all())

    def test_intraday_clock_windows_time_alignment(self) -> None:
        """Verify session-clock windows accurately tag morning overflow, afternoon bond, and MIS dump."""
        engine = FeatureMatrixEngine(FeatureConfig())
        matrix = engine.compute_feature_matrix(self.df, symbol="BANKBEES")

        # 09:15 bar: clock ~ -1.0, morning_window == 1.0, bond_window == 0.0
        self.assertAlmostEqual(matrix["SESSION_CLOCK_NORM"].iloc[0], -1.0, delta=0.05)
        self.assertEqual(matrix["MORNING_OVERFLOW_WINDOW"].iloc[0], 1.0)
        self.assertEqual(matrix["AFTERNOON_BOND_WINDOW"].iloc[0], 0.0)

        # Check bounds
        self.assertTrue((matrix["SESSION_CLOCK_NORM"] >= -1.0).all())
        self.assertTrue((matrix["SESSION_CLOCK_NORM"] <= 1.0).all())
        self.assertTrue(set(matrix["MORNING_OVERFLOW_WINDOW"].unique()).issubset({0.0, 1.0}))
        self.assertTrue(set(matrix["AFTERNOON_BOND_WINDOW"].unique()).issubset({-1.0, 0.0, 1.0}))
        self.assertTrue((matrix["OPENING_RANGE_DRIFT"] >= -5.0).all())
        self.assertTrue((matrix["OPENING_RANGE_DRIFT"] <= 5.0).all())

    def test_macro_yield_return_fallback(self) -> None:
        """Verify MACRO_YIELD_RET_1 computes properly and falls back to zeros when missing."""
        engine = FeatureMatrixEngine(FeatureConfig())
        matrix_with_yield = engine.compute_feature_matrix(self.df, symbol="BANKBEES")
        self.assertTrue(np.all(np.isfinite(matrix_with_yield["MACRO_YIELD_RET_1"])))

        # Remove yield column
        df_no_yield = self.df.drop(columns=["Macro_TNX"])
        matrix_no_yield = engine.compute_feature_matrix(df_no_yield, symbol="BANKBEES")
        self.assertTrue((matrix_no_yield["MACRO_YIELD_RET_1"] == 0.0).all())

    def test_causality_zero_lookahead_invariant(self) -> None:
        """Verify modifying bar t > 80 has zero impact on feature values at t <= 80."""
        engine = FeatureMatrixEngine(FeatureConfig())
        orig_matrix = engine.compute_feature_matrix(self.df, symbol="BANKBEES")

        # Shock data at t=100
        shocked_df = self.df.copy()
        shocked_df.iloc[100:, shocked_df.columns.get_loc("Close")] *= 2.0
        shocked_df.iloc[100:, shocked_df.columns.get_loc("High")] *= 2.0
        shocked_df.iloc[100:, shocked_df.columns.get_loc("Low")] *= 2.0

        shocked_matrix = engine.compute_feature_matrix(shocked_df, symbol="BANKBEES")

        # Every feature at or before t=99 must match identically
        for col in orig_matrix.columns:
            np.testing.assert_allclose(
                orig_matrix[col].iloc[:100].values,
                shocked_matrix[col].iloc[:100].values,
                err_msg=f"Lookahead leak detected in feature {col}",
                atol=1e-5,
            )

    def test_variance_ratio_20(self) -> None:
        """Verify Lo-MacKinlay Variance Ratio is bounded in [0.10, 3.00] and defaults to 1.0."""
        engine = FeatureMatrixEngine(FeatureConfig())
        matrix = engine.compute_feature_matrix(self.df, symbol="BANKBEES")
        vr = matrix["variance_ratio_20"]

        self.assertTrue(np.all(np.isfinite(vr)))
        self.assertTrue((vr >= 0.10).all())
        self.assertTrue((vr <= 3.00).all())

        # Constant price series must default to 1.0
        const_df = self.df.copy()
        const_df["Close"] = 100.0
        const_df["High"] = 100.0
        const_df["Low"] = 100.0
        const_matrix = engine.compute_feature_matrix(const_df, symbol="BANKBEES")
        self.assertTrue((const_matrix["variance_ratio_20"] == 1.0).all())

    def test_order_flow_imbalance_5(self) -> None:
        """Verify Order Flow Imbalance is bounded in [-1.0, 1.0] and responds to CLV."""
        engine = FeatureMatrixEngine(FeatureConfig())
        matrix = engine.compute_feature_matrix(self.df, symbol="BANKBEES")
        ofi = matrix["order_flow_imbalance_5"]

        self.assertTrue(np.all(np.isfinite(ofi)))
        self.assertTrue((ofi >= -1.0).all())
        self.assertTrue((ofi <= 1.0).all())

        # Test pure buying pressure (Close == High): OFI must be positive
        buy_df = self.df.copy()
        buy_df["High"] = buy_df["Close"] + 1.0
        buy_df["Low"] = buy_df["Close"] - 1.0
        buy_df["Close"] = buy_df["High"]  # Close at High -> CLV = +1.0
        buy_matrix = engine.compute_feature_matrix(buy_df, symbol="BANKBEES")
        self.assertTrue((buy_matrix["order_flow_imbalance_5"].iloc[5:] > 0.9).all())

    def test_fvg_bullish_drift(self) -> None:
        """Verify Fair Value Gap bullish drift detects gaps and stays in [0.0, 5.0]."""
        engine = FeatureMatrixEngine(FeatureConfig())
        matrix = engine.compute_feature_matrix(self.df, symbol="BANKBEES")
        fvg = matrix["fvg_bullish_drift"]

        self.assertTrue(np.all(np.isfinite(fvg)))
        self.assertTrue((fvg >= 0.0).all())
        self.assertTrue((fvg <= 5.0).all())

        # Synthesize a clear gap at bar 30: Low[30] > High[28]
        gap_df = self.df.copy()
        gap_df.iloc[30, gap_df.columns.get_loc("Low")] = gap_df.iloc[28]["High"] + 2.0
        gap_df.iloc[30, gap_df.columns.get_loc("Close")] = gap_df.iloc[30]["Low"] + 0.5
        gap_df.iloc[30, gap_df.columns.get_loc("High")] = gap_df.iloc[30]["Close"] + 0.5
        gap_matrix = engine.compute_feature_matrix(gap_df, symbol="BANKBEES")
        self.assertGreater(gap_matrix["fvg_bullish_drift"].iloc[30], 0.0)

    def test_stop_sweep_reclaim(self) -> None:
        """Verify 10-bar swing low sweep and reclaim triggers only under valid conditions."""
        engine = FeatureMatrixEngine(FeatureConfig())
        sweep_df = self.df.copy()

        # Prior 10-bar low before bar 25
        prior_10_low = sweep_df["Low"].iloc[15:25].min()

        # Bar 25 sweeps prior low (Low < prior_10_low), reclaims (Close > prior_10_low), with high volume buying
        sweep_df.iloc[25, sweep_df.columns.get_loc("Low")] = prior_10_low - 0.5
        sweep_df.iloc[25, sweep_df.columns.get_loc("High")] = prior_10_low + 1.0
        sweep_df.iloc[25, sweep_df.columns.get_loc("Close")] = prior_10_low + 1.0
        sweep_df.iloc[25, sweep_df.columns.get_loc("Volume")] = 1000000.0  # Strong reclaim volume -> OFI > 0.15

        matrix = engine.compute_feature_matrix(sweep_df, symbol="BANKBEES")
        self.assertEqual(matrix["stop_sweep_reclaim"].iloc[25], 1.0)

    def test_vix_crush_relief(self) -> None:
        """Verify VIX crush relief triggers when Macro_VIX > 18.0 and return < -2%."""
        engine = FeatureMatrixEngine(FeatureConfig())
        vix_df = self.df.copy()

        # Bar 40: VIX = 20.0, Bar 41: VIX = 19.0 (return ~ -5.1% < -2%, VIX > 18.0)
        vix_df.iloc[40, vix_df.columns.get_loc("Macro_VIX")] = 20.0
        vix_df.iloc[41, vix_df.columns.get_loc("Macro_VIX")] = 19.0

        matrix = engine.compute_feature_matrix(vix_df, symbol="BANKBEES")
        self.assertEqual(matrix["vix_crush_relief"].iloc[41], 1.0)
        self.assertEqual(matrix["vix_crush_relief"].iloc[40], 0.0)

        # Drop Macro_VIX -> should return zeros
        no_vix_df = self.df.drop(columns=["Macro_VIX"])
        no_vix_matrix = engine.compute_feature_matrix(no_vix_df, symbol="BANKBEES")
        self.assertTrue((no_vix_matrix["vix_crush_relief"] == 0.0).all())

    def test_lakehouse_live_integration(self) -> None:
        """Verify feature engine computes cleanly on real Lakehouse Silver/Gold partitions."""
        from ravana.data.storage import LakeManager, StorageConfig
        import os

        config = StorageConfig()
        lake_mgr = LakeManager(config)

        if not os.path.exists(config.lake_root / "m4_cleaned"):
            self.skipTest("Lakehouse partitions not found, skipping live integration test.")

        # Read Silver cleaned partition
        silver_df = lake_mgr.read(tier="m4_cleaned", timeframe="5m", ticker="BANKBEES")
        self.assertFalse(silver_df.empty)

        engine = FeatureMatrixEngine(FeatureConfig())
        matrix = engine.compute_feature_matrix(silver_df, symbol="BANKBEES")

        self.assertEqual(len(matrix), len(silver_df))
        self.assertEqual(len(matrix.columns), 36)
        self.assertFalse(matrix.isna().any().any())
        self.assertTrue(np.all(np.isfinite(matrix.values)))


if __name__ == "__main__":
    unittest.main()
