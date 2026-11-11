"""Unit tests for Muscle 12 Microstructure Portfolio & Risk Allocation Engine."""
import unittest
import numpy as np
import pandas as pd

from ravana.risk.config import AllocationMethod, DrawdownState, RiskConfig
from ravana.risk.engine import PortfolioRiskEngine
from ravana.risk.gatekeeper import RiskGatekeeper
from ravana.risk.hrp import HierarchicalRiskParity


class TestRiskEngine(unittest.TestCase):
    """Test suite for Muscle 12 risk budgeting, HRP, and capital gatekeeping."""

    def test_capital_viability_gate(self):
        """Verify strict rejection of accounts below ₹2,00,000 capital viability gate."""
        cfg = RiskConfig(min_capital_gate=200000.0)
        gatekeeper = RiskGatekeeper(cfg)

        valid, _ = gatekeeper.validate_account_capital(250000.0)
        self.assertTrue(valid)

        invalid, msg = gatekeeper.validate_account_capital(150000.0)
        self.assertFalse(invalid)
        self.assertIn("below the ₹200,000.00 viability gate", msg)

    def test_drawdown_circuit_breaker(self):
        """Verify 4% drawdown halves sizing and 8% drawdown halts entries."""
        cfg = RiskConfig(drawdown_halve_threshold=0.04, drawdown_halt_threshold=0.08)
        gatekeeper = RiskGatekeeper(cfg)

        # Normal regime (peak 200k, current 198k = 1% DD)
        state, mult = gatekeeper.evaluate_drawdown_breaker(200000.0, 198000.0)
        self.assertEqual(state, DrawdownState.NORMAL)
        self.assertEqual(mult, 1.0)

        # Caution regime (peak 200k, current 190k = 5% DD)
        state, mult = gatekeeper.evaluate_drawdown_breaker(200000.0, 190000.0)
        self.assertEqual(state, DrawdownState.CAUTION_HALVED)
        self.assertEqual(mult, 0.5)

        # Halt regime (peak 200k, current 180k = 10% DD)
        state, mult = gatekeeper.evaluate_drawdown_breaker(200000.0, 180000.0)
        self.assertEqual(state, DrawdownState.HALTED)
        self.assertEqual(mult, 0.0)

    def test_trade_viability_floor(self):
        """Verify sub-₹17,000 trades are rejected to prevent DP fee erosion."""
        cfg = RiskConfig(min_trade_notional_floor=17000.0)
        gatekeeper = RiskGatekeeper(cfg)

        # 10 shares @ ₹500 = ₹5,000 (< ₹17,000 floor)
        valid, _, msg = gatekeeper.filter_trade_viability(price=500.0, shares=10)
        self.assertFalse(valid)
        self.assertIn("below ₹17,000.00 floor", msg)

        # 40 shares @ ₹500 = ₹20,000 (>= ₹17,000 floor)
        valid, shares, _ = gatekeeper.filter_trade_viability(price=500.0, shares=40, expected_alpha_bps=15.0)
        self.assertTrue(valid)
        self.assertEqual(shares, 40)

    def test_hrp_weight_computation(self):
        """Verify Hierarchical Risk Parity generates valid, non-negative, normalized weights."""
        np.random.seed(42)
        # Generate 100 periods of synthetic returns for 4 correlated assets
        base = np.random.normal(0, 0.01, size=(100, 1))
        noise = np.random.normal(0, 0.005, size=(100, 4))
        returns = base + noise

        weights = HierarchicalRiskParity.compute_weights(returns, linkage_method="average")

        self.assertEqual(len(weights), 4)
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=4)
        self.assertTrue(np.all(weights >= 0.0))
        self.assertTrue(np.all(weights <= 1.0))

    def test_portfolio_risk_engine_end_to_end(self):
        """Verify end-to-end allocation across BANKBEES and NIFTYBEES with alpha blending."""
        cfg = RiskConfig(
            min_capital_gate=200000.0,
            max_asset_allocation=0.50,
            allocation_method=AllocationMethod.HRP,
        )
        engine = PortfolioRiskEngine(cfg)

        # 60 periods of returns for 2 assets
        returns_df = pd.DataFrame({
            "BANKBEES": np.random.normal(0.0005, 0.012, 60),
            "NIFTYBEES": np.random.normal(0.0003, 0.008, 60),
        })
        current_prices = {"BANKBEES": 520.0, "NIFTYBEES": 260.0}
        alpha_signals = {"BANKBEES": 0.08, "NIFTYBEES": 0.04}

        res = engine.allocate(
            symbols=["BANKBEES", "NIFTYBEES"],
            returns_df=returns_df,
            current_prices=current_prices,
            current_equity=250000.0,
            peak_equity=250000.0,
            alpha_signals=alpha_signals,
        )

        self.assertEqual(res.drawdown_state, DrawdownState.NORMAL)
        self.assertEqual(res.drawdown_multiplier, 1.0)
        self.assertGreater(res.total_allocated_inr, 0.0)
        self.assertGreater(res.cash_reserve_inr, 0.0)
        self.assertIn("BANKBEES", res.target_shares)
        self.assertIn("NIFTYBEES", res.target_shares)
        # Ensure individual weights obey 50% cap
        for sym, w in res.target_weights.items():
            self.assertLessEqual(w, 0.50)
