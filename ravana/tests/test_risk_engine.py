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
        self.assertIn("below ₹200,000.00 viability gate", msg)

    def test_nan_equity_rejection(self):
        """Verify NaN or Inf equity is strictly rejected rather than bypassing gates."""
        cfg = RiskConfig(min_capital_gate=200000.0)
        gatekeeper = RiskGatekeeper(cfg)

        valid_nan, _ = gatekeeper.validate_account_capital(float("nan"))
        self.assertFalse(valid_nan)

        valid_inf, _ = gatekeeper.validate_account_capital(float("inf"))
        self.assertFalse(valid_inf)

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

    def test_drawdown_hysteresis_and_unhalt(self):
        """Verify hysteresis deadband prevents chattering and allows controlled unhalt."""
        cfg = RiskConfig(drawdown_halve_threshold=0.04, drawdown_halt_threshold=0.08)
        gatekeeper = RiskGatekeeper(cfg)

        # Enter caution at 5% DD
        state, mult = gatekeeper.evaluate_drawdown_breaker(200000.0, 190000.0)
        self.assertEqual(state, DrawdownState.CAUTION_HALVED)

        # Small recovery to 3.5% DD does not prematurely flip back to NORMAL (hysteresis)
        state, mult = gatekeeper.evaluate_drawdown_breaker(200000.0, 193000.0)
        self.assertEqual(state, DrawdownState.CAUTION_HALVED)

        # Full recovery to 1.5% DD (< 2.5% deadband) restores NORMAL
        state, mult = gatekeeper.evaluate_drawdown_breaker(200000.0, 197000.0)
        self.assertEqual(state, DrawdownState.NORMAL)

        # Enter HALT at 9% DD
        state, mult = gatekeeper.evaluate_drawdown_breaker(200000.0, 182000.0)
        self.assertEqual(state, DrawdownState.HALTED)

        # Force unhalt restores CAUTION_HALVED
        state, mult = gatekeeper.evaluate_drawdown_breaker(200000.0, 182000.0, force_unhalt=True)
        self.assertEqual(state, DrawdownState.CAUTION_HALVED)
        self.assertEqual(mult, 0.5)

    def test_water_filling_position_limits(self):
        """Verify water-filling cap redistributes excess weight to uncapped assets."""
        cfg = RiskConfig(max_asset_allocation=0.50, max_gross_exposure=1.0)
        gatekeeper = RiskGatekeeper(cfg)

        weights = np.array([0.80, 0.20])
        capped = gatekeeper.enforce_position_limits(weights)

        # Asset 0 is capped at 0.50, and excess 0.30 is redistributed to Asset 1
        self.assertAlmostEqual(float(capped[0]), 0.50, places=3)
        self.assertAlmostEqual(float(capped[1]), 0.50, places=3)
        self.assertAlmostEqual(float(np.sum(capped)), 1.0, places=3)

    def test_trade_viability_floor(self):
        """Verify sub-₹17,000 trades are rejected to prevent DP fee erosion."""
        cfg = RiskConfig(min_trade_notional_floor=17000.0)
        gatekeeper = RiskGatekeeper(cfg)

        # 10 shares @ ₹500 = ₹5,000 (< ₹17,000 floor)
        valid, _, msg = gatekeeper.filter_trade_viability(price=500.0, shares=10)
        self.assertFalse(valid)
        self.assertIn("below ₹17,000.00 floor", msg)

        # 40 shares @ ₹500 = ₹20,000 (>= ₹17,000 floor)
        valid, shares, _ = gatekeeper.filter_trade_viability(price=500.0, shares=40, expected_alpha_bps=20.0)
        self.assertTrue(valid)
        self.assertEqual(shares, 40)

    def test_hrp_weight_computation(self):
        """Verify Hierarchical Risk Parity generates valid, non-negative, normalized weights."""
        np.random.seed(42)
        base = np.random.normal(0, 0.01, size=(100, 1))
        noise = np.random.normal(0, 0.005, size=(100, 4))
        returns = base + noise

        weights = HierarchicalRiskParity.compute_weights(returns, linkage_method="average")

        self.assertEqual(len(weights), 4)
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=4)
        self.assertTrue(np.all(weights >= 0.0))
        self.assertTrue(np.all(weights <= 1.0))

    def test_hrp_zero_volatility_safety(self):
        """Verify suspended or flat assets do not hijack 100% of allocation (VULN-01 fix)."""
        ret = np.zeros((50, 2))
        ret[:, 0] = np.random.randn(50) * 0.01
        ret[:, 1] = 0.0  # Suspended / flat asset

        weights = HierarchicalRiskParity.compute_weights(ret)
        # Should gracefully return equal weights 1/N rather than giving 97%-100% to suspended asset
        self.assertEqual(len(weights), 2)
        self.assertAlmostEqual(float(weights[0]), 0.5, places=2)
        self.assertAlmostEqual(float(weights[1]), 0.5, places=2)

    def test_marcenko_pastur_denoising(self):
        """Verify RMT spectral cutoff removes sampling noise and preserves positive definiteness."""
        cov = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.03, 0.008],
            [0.005, 0.008, 0.02],
        ])
        denoised = HierarchicalRiskParity.denoise_covariance_rmt(cov, T=50)
        self.assertEqual(denoised.shape, (3, 3))
        # Positive definite eigenvalues
        evals = np.linalg.eigvalsh(denoised)
        self.assertTrue(np.all(evals > 0.0))

    def test_portfolio_risk_engine_end_to_end(self):
        """Verify end-to-end allocation across BANKBEES and NIFTYBEES with alpha blending."""
        cfg = RiskConfig(
            min_capital_gate=200000.0,
            max_asset_allocation=0.50,
            allocation_method=AllocationMethod.HRP,
        )
        engine = PortfolioRiskEngine(cfg)

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

        self.assertIn("BANKBEES", res.target_shares)
        self.assertIn("NIFTYBEES", res.target_shares)
        self.assertGreater(res.total_allocated_inr, 200000.0)
        self.assertLessEqual(res.total_allocated_inr, 250000.0)

    def test_10_asset_portfolio_allocation(self):
        """Verify 10-asset portfolio allocates capital with ₹17,000 floor (VULN-02 fix)."""
        engine = PortfolioRiskEngine(RiskConfig(min_capital_gate=200000.0, min_trade_notional_floor=17000.0))
        symbols = [f"ETF_{i}" for i in range(10)]
        prices = {s: 100.0 for s in symbols}
        # 10 assets on ₹2,00,000 = ₹20,000 per asset >= ₹17,000 floor
        res = engine.allocate(symbols, pd.DataFrame(), prices, 200000.0, 200000.0)
        self.assertGreater(res.total_allocated_inr, 150000.0)
        self.assertEqual(len(res.target_shares), 10)
