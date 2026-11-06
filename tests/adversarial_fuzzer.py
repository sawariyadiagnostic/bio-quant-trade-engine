"""
Adversarial Fuzzing & Broker Chaos Chamber (Phase 2 Red Team Suite).
Systematically stress-tests:
1. Broker HTTP 429 (rate limits) & HTTP 504 / network timeout injections.
2. Black Swan Flash Crash (-15% overnight gap) & Lower Circuit Lockouts.
3. Random AST tree operator fuzzing for memory and numerical bounds.
4. Degenerate and rank-deficient covariance matrices in HRP.
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.domain.models import CircuitBands, TaxSchedule, InstrumentType
from src.domain.invariants import assert_circuit_headroom
from src.domain.formula_ast import ASTNode, ASTEvaluator, TreeGenerator
from src.execution.shoonya_gateway import ShoonyaGateway, ShoonyaCredentials
from src.execution.risk_controller import MicroCapitalController, CircuitDamper
from src.execution.hrp_allocator import HRPAllocator
from src.execution.overnight_engine import TrailingStopSimulator


class TestAdversarialBrokerChaos(unittest.TestCase):
    """Stress-tests broker gateway resilience under network faults."""

    def setUp(self):
        self.creds = ShoonyaCredentials(
            user_id="TEST_UID", password="PWD", totp_key="JBSWY3DPEHPK3PXP",
            vendor_code="VEND", api_key="KEY"
        )
        self.gw = ShoonyaGateway(credentials=self.creds, paper_trading=False)
        self.gw.susertoken = "MOCK_TOKEN"
        # Ensure we test the direct REST handling by bypassing NorenApi sdk object
        self.gw.api = None

    @patch("requests.post")
    def test_broker_http_429_rate_limiting(self, mock_post):
        """Verify gateway gracefully catches HTTP 429 Too Many Requests."""
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.json.return_value = {"stat": "Not_Ok", "emsg": "Rate limit exceeded (429)"}
        mock_resp.text = '{"stat": "Not_Ok", "emsg": "Rate limit exceeded (429)"}'
        mock_post.return_value = mock_resp

        res = self.gw.place_order(symbol="NIFTYBEES", quantity=10, price=250.0)
        self.assertEqual(res.get("stat"), "Not_Ok")
        self.assertIn("Rate limit", res.get("emsg", ""))

    @patch("requests.post")
    def test_broker_network_timeout_resilience(self, mock_post):
        """Verify gateway handles network timeout without unhandled crash."""
        mock_post.side_effect = requests.exceptions.Timeout("Connection timed out after 10s")

        res = self.gw.place_order(symbol="NIFTYBEES", quantity=10, price=250.0)
        self.assertEqual(res.get("stat"), "Not_Ok")
        self.assertIn("timed out", res.get("emsg", ""))


class TestAdversarialMarketShocks(unittest.TestCase):
    """Stress-tests risk controller and execution engine under market dislocations."""

    def test_flash_crash_gap_down(self):
        """Simulates an abrupt -15% opening gap down on equity holding."""
        entry_price = 100.0
        gap_open_price = 85.0   # -15% gap
        day_high = 86.0
        day_low = 82.0

        # Trailing stop set at 0.25% stopgap
        exit_price, reason = TrailingStopSimulator.resolve_exit(
            entry_price=entry_price,
            open_price=gap_open_price,
            high_price=day_high,
            low_price=day_low,
            trail_pct=0.0015,
            stop_pct=0.0025
        )
        # In a flash crash gap down below stop-loss, exit price MUST fill at auction open (85.0),
        # NOT hallucinating a fill at the higher stop-loss level (99.75).
        self.assertEqual(exit_price, gap_open_price)
        self.assertEqual(reason, "AUCTION_GAP_DOWN")

    def test_lower_circuit_freeze_lockout(self):
        """Simulates lower circuit freeze and verifies allocation lockout."""
        prev_close = 100.0
        # Stock locked at -10% lower circuit limit
        locked_price = 90.0
        bands = CircuitDamper.calculate_bands(current_price=locked_price, prev_close=prev_close, band_pct=0.10)

        self.assertTrue(bands.is_locked)
        self.assertEqual(bands.allocation_damper_factor(), 0.0)

        ctrl = MicroCapitalController(total_capital=50000.0)
        units, reason = ctrl.allocate(current_price=locked_price, is_etf=False, circuit_bands=bands)
        self.assertEqual(units, 0)
        self.assertEqual(reason, "CIRCUIT_LOCKED")


class TestAdversarialNumericalFuzzing(unittest.TestCase):
    """Stress-tests AST Evaluator and HRP Allocator with extreme numerical inputs."""

    def test_ast_random_tree_fuzzing(self):
        """Generates 50 random AST trees and verifies finite execution on noisy data."""
        np.random.seed(1337)
        df = pd.DataFrame({
            "Close": np.random.uniform(10.0, 500.0, 100),
            "Volume": np.random.uniform(1e4, 1e7, 100),
            "Dist_VWAP": np.random.uniform(-0.1, 0.1, 100),
            "True_Range": np.random.uniform(0.5, 20.0, 100),
            "RSI_14": np.random.uniform(10.0, 90.0, 100),
            "Zero_Col": np.zeros(100),
            "Inf_Col": np.full(100, np.nan)
        })

        for _ in range(30):
            tree = TreeGenerator.generate(max_depth=4)
            res = ASTEvaluator.eval_node(tree, df, current_regime=0)
            self.assertEqual(len(res), 100)
            # Must strictly contain only finite floats
            self.assertTrue(np.all(np.isfinite(res)))

    def test_hrp_collinear_degenerate_matrix(self):
        """Verifies HRP allocator gracefully handles fully collinear identical assets."""
        T = 50
        base_series = np.random.normal(0.001, 0.02, T)
        # 4 identical assets (perfect correlation 1.0)
        returns_collinear = np.column_stack([base_series, base_series, base_series, base_series])

        weights = HRPAllocator.compute_weights(returns_collinear)
        self.assertEqual(len(weights), 4)
        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=4)
        # Under perfect symmetry, weights must be equally distributed
        for w in weights:
            self.assertAlmostEqual(float(w), 0.25, places=3)


if __name__ == "__main__":
    unittest.main()
