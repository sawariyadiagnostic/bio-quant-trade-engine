"""Unit and Integration Tests for Worker 3: BoundedASTModulator, StrategyBanditAllocator & TriggerCompositeStrategy."""

import unittest
import numpy as np
import pandas as pd

from ravana.domain.formula_ast import ASTNode
from ravana.features.config import FeatureName
from ravana.strategies.base import TriggerAction, TriggerSignal
from ravana.strategies.composite import (
    ALL_TRIGGER_NAMES,
    BUCKET_1_NAMES,
    BUCKET_2_NAMES,
    BUCKET_3_NAMES,
    StrategyBanditAllocator,
    TriggerCompositeStrategy,
)
from ravana.strategies.modulator import BoundedASTModulator


class TestBoundedASTModulator(unittest.TestCase):
    """Test suite for BoundedASTModulator."""

    def setUp(self) -> None:
        self.modulator = BoundedASTModulator()
        # Build mock dataframe
        n = 100
        dates = pd.date_range("2026-01-08 09:15", periods=n, freq="5min", tz="Asia/Kolkata")
        self.df = pd.DataFrame(
            {
                "Close": np.linspace(500.0, 520.0, n),
                "High": np.linspace(502.0, 522.0, n),
                "Low": np.linspace(498.0, 518.0, n),
                "Volume": np.full(n, 10000.0),
                "RSI_14": np.sin(np.linspace(0, 10, n)) * 0.5,
                "LOG_RET_5": np.random.RandomState(42).normal(0.001, 0.002, n),
                "ATR_14": np.full(n, 2.5),
            },
            index=dates,
        )

    def test_input_guard_accepts_valid_stationary_features(self) -> None:
        """Verify Input Guard accepts all valid FeatureName members."""
        node1 = ASTNode(op="FEATURE", val=FeatureName.RSI_14.value)
        self.assertTrue(self.modulator.is_valid_ast(node1))

        node2 = ASTNode(
            op="ADD",
            children=[
                ASTNode(op="FEATURE", val=FeatureName.LOG_RET_5.value),
                ASTNode(op="CONST", val="0.05"),
            ],
        )
        self.assertTrue(self.modulator.is_valid_ast(node2))

    def test_input_guard_rejects_non_stationary_features(self) -> None:
        """Verify Input Guard raises ValueError on non-stationary features like raw Close."""
        invalid_node = ASTNode(op="FEATURE", val="Close_Unadjusted_Raw")
        self.assertFalse(self.modulator.is_valid_ast(invalid_node))
        with self.assertRaises(ValueError):
            self.modulator.validate_ast(invalid_node)

    def test_sigmoid_multiplier_bounded_in_half_to_one_and_half(self) -> None:
        """Verify multiplier outputs are strictly bounded in [0.5x, 1.5x]."""
        node = ASTNode(op="FEATURE", val="RSI_14")
        multipliers, alert = self.modulator.compute_multipliers(node, self.df)
        self.assertFalse(alert)
        self.assertTrue((multipliers >= 0.5).all())
        self.assertTrue((multipliers <= 1.5).all())

    def test_rolling_variance_sentinel_stagnation_alert(self) -> None:
        """Verify variance sentinel triggers stagnation_alert and defaults to 1.0x on flat AST."""
        flat_df = self.df.copy()
        flat_df["RSI_14"] = 0.0  # perfectly zero variance
        node = ASTNode(op="FEATURE", val="RSI_14")

        multipliers, alert = self.modulator.compute_multipliers(node, flat_df)
        self.assertTrue(alert)
        # Check that after bar 60 (the limit), multipliers are set strictly to 1.0
        self.assertEqual(multipliers.iloc[-1], 1.0)
        self.assertTrue(self.modulator.stagnation_alert)

    def test_zero_ghost_trades_preservation(self) -> None:
        """Verify Zero Ghost Trades: 0.0 base conviction strictly yields 0.0 modulated conviction."""
        # Single value
        self.assertEqual(self.modulator.modulate_conviction(0.0, 1.5), 0.0)
        self.assertEqual(self.modulator.modulate_conviction(0.0, 0.5), 0.0)

        # Series
        base_conv = pd.Series([0.0, 0.8, 0.0, 0.5, 0.0], index=self.df.index[:5])
        node = ASTNode(op="FEATURE", val="RSI_14")
        modulated = self.modulator.modulate_series(base_conv, node, self.df.iloc[:5])

        self.assertEqual(modulated.iloc[0], 0.0)
        self.assertGreater(modulated.iloc[1], 0.0)
        self.assertEqual(modulated.iloc[2], 0.0)
        self.assertGreater(modulated.iloc[3], 0.0)
        self.assertEqual(modulated.iloc[4], 0.0)


class TestStrategyBanditAllocator(unittest.TestCase):
    """Test suite for StrategyBanditAllocator."""

    def setUp(self) -> None:
        self.bandit = StrategyBanditAllocator(burn_in_trades=15, gamma=0.98)

    def test_burn_in_gate_enforces_one_point_zero(self) -> None:
        """Verify that before 15 closed trades, multiplier is strictly 1.0x."""
        trigger = "OpeningRangeBreakoutTrigger"
        for trade_idx in range(14):
            self.bandit.update(trigger, net_pnl=500.0)
            self.assertEqual(self.bandit.get_multiplier(trigger), 1.0)

        # On trade 15, burn-in gate opens
        self.bandit.update(trigger, net_pnl=500.0)
        mult = self.bandit.get_multiplier(trigger)
        self.assertGreater(mult, 1.0)
        self.assertLessEqual(mult, 1.30)

    def test_clamping_bounds_point_seven_to_one_point_three(self) -> None:
        """Verify that multipliers never exceed [0.70x, 1.30x] regardless of win/loss streaks."""
        trig_win = "ThursdayExpiryGammaTrigger"
        trig_loss = "INAVSpreadMeanReversionTrigger"

        # 40 consecutive wins
        for _ in range(40):
            self.bandit.update(trig_win, net_pnl=100.0)
        self.assertAlmostEqual(self.bandit.get_multiplier(trig_win), 1.30, places=2)

        # 40 consecutive losses
        for _ in range(40):
            self.bandit.update(trig_loss, net_pnl=-100.0)
        self.assertAlmostEqual(self.bandit.get_multiplier(trig_loss), 0.70, places=2)

    def test_exponential_decay_gamma(self) -> None:
        """Verify exponential posterior decay gamma = 0.98."""
        trigger = "CrossSectorLeadLagTrigger"
        # Initial: alpha=1.0, beta=1.0
        self.bandit.update(trigger, net_pnl=100.0)
        # alpha = 0.98 * 1.0 + 1.0 = 1.98; beta = 0.98 * 1.0 + 0 = 0.98
        self.assertAlmostEqual(self.bandit.alpha[trigger], 1.98, places=4)
        self.assertAlmostEqual(self.bandit.beta[trigger], 0.98, places=4)


class TestTriggerCompositeStrategy(unittest.TestCase):
    """Test suite for TriggerCompositeStrategy."""

    def setUp(self) -> None:
        self.strategy = TriggerCompositeStrategy(max_concurrent_positions=2, vr_threshold=1.05)

    def test_regime_bucket_dispatch(self) -> None:
        """Verify Variance Ratio gates Bucket 1 (>1.05) vs Bucket 2 (<=1.05)."""
        # VR = 1.25 -> Bucket 1 Trend Active
        self.assertEqual(self.strategy.get_active_bucket(1.25), "BUCKET_1")
        self.assertTrue(self.strategy.is_trigger_active_in_regime("OpeningRangeBreakoutTrigger", 1.25))
        self.assertFalse(self.strategy.is_trigger_active_in_regime("INAVSpreadMeanReversionTrigger", 1.25))

        # VR = 0.85 -> Bucket 2 Mean Reversion Active
        self.assertEqual(self.strategy.get_active_bucket(0.85), "BUCKET_2")
        self.assertFalse(self.strategy.is_trigger_active_in_regime("OpeningRangeBreakoutTrigger", 0.85))
        self.assertTrue(self.strategy.is_trigger_active_in_regime("INAVSpreadMeanReversionTrigger", 0.85))

        # Bucket 3 Macro/Overnight always active
        self.assertTrue(self.strategy.is_trigger_active_in_regime("OvernightContinuationTrigger", 1.25))
        self.assertTrue(self.strategy.is_trigger_active_in_regime("OvernightContinuationTrigger", 0.85))

    def test_capital_contention_limits_to_two_slots(self) -> None:
        """Verify capital contention limits concurrent positions to max_concurrent_positions = 2."""
        ts = pd.Timestamp("2026-01-08 10:00", tz="Asia/Kolkata")
        # 3 candidate BUY signals
        s1 = TriggerSignal(ts, "BANKBEES", TriggerAction.BUY, 0.90, "OpeningRangeBreakoutTrigger")
        s2 = TriggerSignal(ts, "NIFTYBEES", TriggerAction.BUY, 0.85, "OpeningRangeBreakoutTrigger")
        s3 = TriggerSignal(ts, "ITBEES", TriggerAction.BUY, 0.70, "OpeningRangeBreakoutTrigger")

        # Mock market data dictionary
        df_dummy = pd.DataFrame({"Close": [100.0], "High": [101.0], "Low": [99.0], "Volume": [1000]})
        market_data = {"BANKBEES": df_dummy, "NIFTYBEES": df_dummy, "ITBEES": df_dummy}

        # Override generate_signals on instance for controlled testing
        self.strategy.generate_signals = lambda df, sym, ast_node=None: [
            s1 if sym == "BANKBEES" else (s2 if sym == "NIFTYBEES" else s3)
        ]

        portfolio_sigs = self.strategy.generate_portfolio_signals(market_data, current_positions=set())
        # Exactly 2 BUY signals should be accepted
        self.assertEqual(len(portfolio_sigs), 2)
        symbols = [s.symbol for s in portfolio_sigs]
        self.assertIn("BANKBEES", symbols)
        self.assertIn("NIFTYBEES", symbols)
        self.assertNotIn("ITBEES", symbols)  # Dropped due to capital contention!


if __name__ == "__main__":
    unittest.main()
