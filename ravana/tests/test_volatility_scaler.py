"""Unit tests for Dynamic Volatility Hurdle Scaler (AdaptiveThresholdScaler)."""
import unittest
from ravana.strategies.volatility_scaler import AdaptiveThresholdScaler


class TestAdaptiveThresholdScaler(unittest.TestCase):
    """Test suite for AdaptiveThresholdScaler."""

    def setUp(self) -> None:
        self.scaler = AdaptiveThresholdScaler()

    def test_normal_regime(self) -> None:
        """Normal volatility (ATR ratio = 1.0) retains base parameters."""
        hurdle = self.scaler.get_momentum_hurdle(base_hurdle=1.20, atr_ratio=1.0)
        target = self.scaler.get_target_multiplier(base_mult=1.50, atr_ratio=1.0)
        stop = self.scaler.get_stop_multiplier(base_mult=0.75, atr_ratio=1.0)
        inav_bps = self.scaler.get_inav_bps_hurdle(base_bps=15.0, atr_ratio=1.0)

        self.assertAlmostEqual(hurdle, 1.20)
        self.assertAlmostEqual(target, 1.50)
        self.assertAlmostEqual(stop, 0.75)
        self.assertAlmostEqual(inav_bps, 15.0)

    def test_low_volatility_compression(self) -> None:
        """Low volatility (ATR ratio = 0.60) scales down hurdles and targets."""
        hurdle = self.scaler.get_momentum_hurdle(base_hurdle=1.20, atr_ratio=0.60)
        target = self.scaler.get_target_multiplier(base_mult=1.50, atr_ratio=0.60)
        stop = self.scaler.get_stop_multiplier(base_mult=0.75, atr_ratio=0.60)
        inav_bps = self.scaler.get_inav_bps_hurdle(base_bps=15.0, atr_ratio=0.60)

        # Hurdle floor at 0.85
        self.assertAlmostEqual(hurdle, 0.85)
        # Tightened micro target and stop
        self.assertAlmostEqual(target, 0.75)
        self.assertAlmostEqual(stop, 0.45)
        # iNAV compressed to 10 bps
        self.assertAlmostEqual(inav_bps, 10.0)

    def test_moderate_low_volatility(self) -> None:
        """Moderate low volatility (ATR ratio = 0.75) scales linearly above 0.85 floor."""
        hurdle = self.scaler.get_momentum_hurdle(base_hurdle=1.20, atr_ratio=0.75)
        expected = max(0.85, 1.20 * 0.75)  # 0.90
        self.assertAlmostEqual(hurdle, expected)

    def test_high_volatility_expansion(self) -> None:
        """High volatility expansion (ATR ratio = 1.30) expands hurdle and target."""
        hurdle = self.scaler.get_momentum_hurdle(base_hurdle=1.20, atr_ratio=1.30)
        target = self.scaler.get_target_multiplier(base_mult=1.50, atr_ratio=1.30)
        stop = self.scaler.get_stop_multiplier(base_mult=0.75, atr_ratio=1.30)

        self.assertAlmostEqual(hurdle, 1.35)
        self.assertAlmostEqual(target, 2.00)
        self.assertAlmostEqual(stop, 0.75)

    def test_compute_atr_ratio(self) -> None:
        """Calculates ratio of today opening ATR to 20-day median ATR."""
        ratio = self.scaler.compute_atr_ratio(today_atr=0.25, median_20d_atr=0.50)
        self.assertAlmostEqual(ratio, 0.50)

        # Zero division guard
        ratio_zero = self.scaler.compute_atr_ratio(today_atr=0.25, median_20d_atr=0.0)
        self.assertAlmostEqual(ratio_zero, 1.0)


if __name__ == "__main__":
    unittest.main()
