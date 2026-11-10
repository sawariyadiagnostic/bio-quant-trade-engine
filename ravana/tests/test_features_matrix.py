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
from ravana.features.stationarity import RollingADF, RollingOUHalfLife
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
            },
            index=dates,
        )

    def test_21_stationary_features_presence(self) -> None:
        """Verify engine computes all 21 registered stationary features without NaNs."""
        engine = FeatureMatrixEngine(FeatureConfig())
        matrix = engine.compute_feature_matrix(self.df, symbol="BANKBEES")

        expected_features = FeatureRegistry.all_features()
        self.assertEqual(len(expected_features), 21)

        for feat in expected_features:
            self.assertIn(feat, matrix.columns, f"Feature {feat} missing from matrix output")

        # Zero NaNs or Infs allowed in institutional contract
        self.assertFalse(matrix.isna().any().any(), "Matrix contains NaN values")
        self.assertTrue(np.all(np.isfinite(matrix.values)), "Matrix contains non-finite values")
        self.assertEqual(len(matrix), len(self.df))

    def test_multi_asset_3d_tensor_dispatch(self) -> None:
        """Verify compute_multi_asset_tensor produces [B, T, K] float32 tensor."""
        engine = FeatureMatrixEngine(FeatureConfig())
        df2 = self.df.copy()
        df2["Close"] = df2["Close"] * 1.5

        universe = {"BANKBEES": self.df, "NIFTYBEES": df2}
        tensor, symbols, features, time_index = engine.compute_multi_asset_tensor(universe)

        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tensor.shape, (2, len(self.df), 21))
        self.assertEqual(symbols, ["BANKBEES", "NIFTYBEES"])
        self.assertEqual(len(features), 21)
        self.assertEqual(len(time_index), len(self.df))
        self.assertTrue(torch.isfinite(tensor).all())

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
        self.assertEqual(len(matrix.columns), 21)
        self.assertFalse(matrix.isna().any().any())
        self.assertTrue(np.all(np.isfinite(matrix.values)))


if __name__ == "__main__":
    unittest.main()
