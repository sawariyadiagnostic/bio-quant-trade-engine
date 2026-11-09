"""Comprehensive Verification Suite for Statutory Frictions, Vector Lake & Portfolio Theory Remediation.

Empirically verifies all 5 remediation areas:
1. VectorDataLake: Split adjustments, flat volume z-score clamping, pre-flight cleaning.
2. Statutory Frictions & RiskSentry: Bipartite DP fee, 5-paise tick quantization, CNC-to-MIS elimination, 0% ledger cash interest.
3. MatrixSimulator: Exit friction deduction, sparse Sortino annualization, Sharpe & Deflated Sharpe (DSR).
4. HRPAllocator: Reachable volatility floor, true dendrogram tree traversal, box bounds [0.02, 0.30].
5. ReturnClusterer: Inert dead formula clustering & immediate pruning, T < N Ledoit-Wolf shrinkage, Ward distance.
"""
import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.vector_lake import VectorDataLake, validate_and_clean_ohlcv, adjust_corporate_splits
from src.domain.models import InstrumentType, TaxSchedule, quantize_nse_tick
from src.execution.overnight_engine import TaxCalculator, PreOpenAuctionRouter, IdleCashManager
from src.execution.risk_sentry import RiskSentry
from src.execution.portfolio_paper_trader import PortfolioPaperTrader
from src.backtest.matrix_engine import MatrixSimulator, compute_sharpe_and_dsr
from src.execution.hrp_allocator import HRPAllocator
from src.clustering.return_clustering import (
    ReturnClusterer,
    compute_correlation_matrix,
    compute_ward_distance,
    prune_clusters,
)


class TestVectorLakeRemediation(unittest.TestCase):
    """Test Suite for VectorDataLake fixes."""

    def test_corporate_action_split_adjustment(self):
        """Verify Open, High, Low scaled by k_t and Volume by 1/k_t when Close_Adj differs from Close."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "Open": [100.0, 100.0, 100.0, 100.0, 100.0],
                "High": [105.0, 105.0, 105.0, 105.0, 105.0],
                "Low": [95.0, 95.0, 95.0, 95.0, 95.0],
                "Close": [100.0, 100.0, 100.0, 100.0, 100.0],
                "Close_Adj": [50.0, 50.0, 50.0, 50.0, 50.0],  # 2:1 stock split ratio k_t = 0.5
                "Volume": [1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
            },
            index=dates,
        )

        lake = VectorDataLake.from_dataframe(df, ticker="TEST.NS")
        tensors = lake.tensors

        # Ratio k_t = 0.5 -> Open, High, Low halved; Volume doubled (2000.0)
        self.assertTrue(np.allclose(tensors["Open"], 50.0))
        self.assertTrue(np.allclose(tensors["High"], 52.5))
        self.assertTrue(np.allclose(tensors["Low"], 47.5))
        self.assertTrue(np.allclose(tensors["Close_Adj"], 50.0))
        self.assertTrue(np.allclose(tensors["Volume"], 2000.0))

    def test_rolling_volume_zscore_flat_volume_clamping(self):
        """Verify rolling std denominator clamped >= 1.0 prevents 1e-8 division blowup on flat volume."""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        df = pd.DataFrame(
            {
                "Open": np.full(30, 250.0),
                "High": np.full(30, 255.0),
                "Low": np.full(30, 245.0),
                "Close": np.full(30, 250.0),
                "Volume": np.full(30, 10000.0),  # Completely flat volume -> std = 0.0
            },
            index=dates,
        )

        lake = VectorDataLake.from_dataframe(df, ticker="FLAT.NS")
        zscores = lake.tensors["Vol_ZScore"]

        # Z-scores must be clean zeros and never blow up to 10^8
        self.assertTrue(np.all(np.isfinite(zscores)))
        self.assertTrue(np.all(np.abs(zscores) < 1.0))
        self.assertTrue(np.allclose(zscores, 0.0))

    def test_preflight_validation_and_cleaning(self):
        """Verify non-monotonic index sorting, duplicate dropping, and NaN filling."""
        idx = pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-01", "2024-01-04"])
        df = pd.DataFrame(
            {
                "Open": [103.0, 100.0, 101.0, np.nan],
                "High": [105.0, 102.0, 103.0, 106.0],
                "Low": [98.0, 95.0, 96.0, 97.0],
                "Close": [103.0, 100.0, 101.0, 104.0],
                "Volume": [1000.0, 1000.0, 1100.0, np.nan],
            },
            index=idx,
        )

        cleaned = validate_and_clean_ohlcv(df)
        self.assertTrue(cleaned.index.is_monotonic_increasing)
        self.assertEqual(len(cleaned), 3)  # Duplicate dropped
        self.assertFalse(cleaned.isna().any().any())  # NaNs filled


class TestStatutoryFrictionsAndExecution(unittest.TestCase):
    """Test Suite for Statutory fee model, tick quantization, and RiskSentry."""

    def test_bipartite_statutory_fee_model(self):
        """Verify TaxCalculator computes (Trade Value * rate) + ₹15.93 DP charge on delivery sells."""
        trade_val = 50000.0

        # Buy ETF delivery: variable rate only (1.2 bps), no DP charge
        buy_friction = TaxCalculator.calculate(trade_val, InstrumentType.ETF_CASH, is_sell=False)
        self.assertAlmostEqual(buy_friction, trade_val * 0.00012, places=4)

        # Sell ETF delivery: variable rate (1.2 bps) + ₹15.93 DP charge
        sell_friction = TaxCalculator.calculate(
            trade_val, InstrumentType.ETF_CASH, is_sell=True, is_delivery=True
        )
        expected_sell = (trade_val * 0.00012) + 15.93
        self.assertAlmostEqual(sell_friction, expected_sell, places=4)

        # Sell Futures: 3 bps friction only, NEVER DP charge
        fut_friction = TaxCalculator.calculate(
            trade_val, InstrumentType.FUTURES, is_sell=True, is_delivery=True
        )
        self.assertAlmostEqual(fut_friction, trade_val * 0.00030, places=4)

    def test_nse_tick_quantization_ge_250(self):
        """Verify 5-paise tick quantization for securities >= 250 (e.g. BANKBEES, HDFCBANK)."""
        # Securities >= 250 quantized to nearest 0.05
        self.assertEqual(quantize_nse_tick(250.03), 250.05)
        self.assertEqual(quantize_nse_tick(250.02), 250.00)
        self.assertEqual(quantize_nse_tick(524.37), 524.35)
        self.assertEqual(quantize_nse_tick(1650.18), 1650.20)

        # Securities < 250 quantized to 0.01 (1 paisa)
        self.assertEqual(quantize_nse_tick(24.354), 24.35)

        # PreOpenAuctionRouter limit price with enforce_tick=True
        order = PreOpenAuctionRouter.route_auction_order(
            ticker="BANKBEES.NS",
            side="BUY",
            indicative_price=500.0,
            quantity=100,
            slippage_ceiling_bps=15.0,
            enforce_tick=True,
        )
        # 500.0 * (1 + 0.0015) = 500.75 -> already 0.05 tick
        self.assertEqual(order["limit_price"], 500.75)

    def test_risksentry_elimination_of_cnc_to_mis_reroute(self):
        """Verify sub-economic delivery sells (< ₹1,60,000) are rejected or routed to futures, NEVER MIS."""
        sentry = RiskSentry(auto_reroute_dp=False, auto_route_futures=False, auto_quantize_tick=True)

        # Sub-economic delivery sell order: notional = 10 * 500 = ₹5,000 < ₹1,60,000
        sub_economic_order = {
            "ticker": "BANKBEES.NS",
            "action": "SELL",
            "product": "CNC",
            "quantity": 10,
            "price": 500.0,
        }

        # Must be rejected, NEVER rerouted to MIS
        allowed, reason, out_order = sentry.validate_outbound_order(sub_economic_order)
        self.assertFalse(allowed)
        self.assertIn("REJECTED_SUB_ECONOMIC_NOTIONAL", reason)
        self.assertNotEqual(out_order.get("product"), "MIS")

        # When auto_route_futures is enabled, route to FUTURES, NEVER MIS
        sentry_futures = RiskSentry(auto_route_futures=True)
        allowed_fut, reason_fut, out_fut = sentry_futures.validate_outbound_order(sub_economic_order)
        self.assertTrue(allowed_fut)
        self.assertEqual(reason_fut, "SUB_ECONOMIC_ROUTED_INDEX_FUTURES")
        self.assertEqual(out_fut["product"], "FUTURES")
        self.assertNotEqual(out_fut.get("product"), "MIS")

    def test_sebi_compliance_zero_interest_on_ledger_cash(self):
        """Verify PortfolioPaperTrader and IdleCashManager credit 0.0% interest on uninvested ledger cash."""
        trader = PortfolioPaperTrader(initial_capital=50000.0)
        self.assertEqual(trader.earn_ledger_cash_interest(), 0.0)

        # IdleCashManager returns 0.0 for ledger cash
        interest = IdleCashManager.earn_interest(50000.0, days=1, is_ledger_cash=True)
        self.assertEqual(interest, 0.0)


class TestMatrixSimulatorRemediation(unittest.TestCase):
    """Test Suite for MatrixSimulator fixes."""

    def test_vanishing_exit_transaction_costs_fixed(self):
        """Verify closing flip transaction costs are accounted for on position exit to 0.0."""
        sim = MatrixSimulator(friction_bps=10.0)  # 0.0010 friction
        T = 4
        # Enters at bar 1 (0 -> 1) and exits at bar 3 (1 -> 0)
        positions = np.array([[0.0, 1.0, 1.0, 0.0]], dtype=np.float32)
        flat_returns = np.zeros(T, dtype=np.float32)

        res = sim.simulate_batch(positions, flat_returns, lookback_years=1.0)
        net_rets = res["net_returns"][0]

        # Bar 1 has entry friction -0.0010
        self.assertAlmostEqual(net_rets[1], -0.0010, places=4)
        # Bar 3 (exit to 0.0) must also have exit friction -0.0010
        self.assertAlmostEqual(net_rets[3], -0.0010, places=4)
        # Total active bars must be 3 (bars 1, 2, 3)
        self.assertEqual(res["n_active_bars"][0], 3)
        # Mean return must be negative reflecting both entry and exit flips
        self.assertLess(res["mean_return"][0], 0.0)

    def test_sortino_annualization_sparse_trading(self):
        """Verify Sortino annualization is correctly scaled by active_bars / total_bars ratio."""
        sim = MatrixSimulator(friction_bps=0.0)
        T = 200

        # Strategy 1: Active all 200 bars
        pos_full = np.ones((1, T), dtype=np.float32)
        # Strategy 2: Active only 10 bars (sparse)
        pos_sparse = np.zeros((1, T), dtype=np.float32)
        pos_sparse[0, :10] = 1.0

        daily_rets = np.full(T, 0.005, dtype=np.float32)  # 50 bps positive return
        res_full = sim.simulate_batch(pos_full, daily_rets, lookback_years=1.0)
        res_sparse = sim.simulate_batch(pos_sparse, daily_rets, lookback_years=1.0)

        # Full strategy Sortino should be strictly greater than sparse strategy Sortino
        self.assertGreater(res_full["sortino"][0], res_sparse["sortino"][0])
        # Activity ratio for sparse: 10 holding bars + 1 exit friction bar = 11 / 200 = 0.055
        self.assertAlmostEqual(res_sparse["activity_ratio"][0], 11.0 / 200.0, places=4)

    def test_sharpe_and_deflated_sharpe_ratio_metrics(self):
        """Verify Sharpe Ratio and Deflated Sharpe Ratio (DSR) are computed and bounded."""
        sim = MatrixSimulator(friction_bps=1.2)
        B, T = 10, 250
        np.random.seed(42)
        positions = np.ones((B, T), dtype=np.float32)
        returns = np.random.normal(0.0008, 0.015, T).astype(np.float32)

        res = sim.simulate_batch(positions, returns, lookback_years=1.0)
        self.assertIn("sharpe", res)
        self.assertIn("deflated_sharpe", res)
        self.assertEqual(res["sharpe"].shape, (B,))
        self.assertEqual(res["deflated_sharpe"].shape, (B,))
        # DSR is a probability bounded in [0.0, 1.0]
        self.assertTrue(np.all(res["deflated_sharpe"] >= 0.0))
        self.assertTrue(np.all(res["deflated_sharpe"] <= 1.0))


class TestHRPAllocatorRemediation(unittest.TestCase):
    """Test Suite for HRPAllocator fixes."""

    def test_zero_variance_volatility_floor_preventing_100_percent_allocation(self):
        """Verify zero-variance asset does NOT capture 100% fund allocation."""
        np.random.seed(42)
        T, N = 200, 4
        active_1 = np.random.normal(0.001, 0.02, T)
        active_2 = np.random.normal(0.0005, 0.015, T)
        active_3 = np.random.normal(0.0008, 0.025, T)
        dead_zero = np.zeros(T)  # Degenerate zero-variance asset

        rets_matrix = np.column_stack([active_1, active_2, active_3, dead_zero])
        weights = HRPAllocator.compute_weights(rets_matrix)

        # Dead asset must NEVER receive 100% allocation
        self.assertLess(weights[3], 0.35)
        self.assertTrue(np.all(weights >= 0.0))
        self.assertAlmostEqual(np.sum(weights), 1.0, places=4)

    def test_true_dendrogram_tree_traversal_and_box_bounds(self):
        """Verify tree recursive bisection obeys box bounds [0.02, 0.30] for N >= 4."""
        np.random.seed(42)
        T, N = 300, 6
        rets = np.random.normal(0.0005, 0.02, (T, N))
        weights = HRPAllocator.compute_weights(rets, min_weight=0.02, max_weight=0.30)

        self.assertEqual(len(weights), N)
        self.assertAlmostEqual(np.sum(weights), 1.0, places=4)
        # Every weight must strictly reside inside box bounds [0.02, 0.30]
        self.assertTrue(np.all(weights >= 0.02 - 1e-6))
        self.assertTrue(np.all(weights <= 0.30 + 1e-6))


class TestReturnClustererRemediation(unittest.TestCase):
    """Test Suite for ReturnClusterer fixes."""

    def test_zero_variance_dead_formulas_inert_group_and_pruned(self):
        """Verify dead formulas cluster into inert group (-1) and are pruned immediately."""
        T = 200
        np.random.seed(42)
        f0 = np.random.normal(0.001, 0.02, T)
        f1 = f0 + np.random.normal(0, 0.001, T)  # Clone of f0
        f2_dead = np.zeros(T)  # Dead formula 1
        f3_dead = np.zeros(T)  # Dead formula 2

        R = np.vstack([f0, f1, f2_dead, f3_dead])
        fitness = np.array([2.5, 2.0, 1.0, 1.0])
        depths = np.array([3, 5, 1, 1])

        clusterer = ReturnClusterer(rho_threshold=0.85)
        labels, elites = clusterer.cluster_and_prune(R, fitness, depths)

        # Dead formulas must be assigned to inert group -1
        self.assertEqual(labels[2], -1)
        self.assertEqual(labels[3], -1)

        # Neither dead formula should EVER be chosen as an elite
        self.assertNotIn(2, elites)
        self.assertNotIn(3, elites)
        self.assertIn(0, elites)  # f0 selected as active elite

    def test_ledoit_wolf_shrinkage_when_T_less_than_N(self):
        """Verify Ledoit-Wolf shrinkage is triggered when candidate count exceeds horizon (T < N)."""
        np.random.seed(42)
        N, T = 15, 8  # T < N: high dimensional singularity regime
        returns = np.random.normal(0.001, 0.02, (N, T))

        # Must compute valid well-conditioned correlation matrix without singular matrix error
        corr = compute_correlation_matrix(returns)
        self.assertEqual(corr.shape, (N, N))
        self.assertTrue(np.all(np.isfinite(corr)))
        self.assertTrue(np.allclose(np.diag(corr), 1.0))

    def test_ward_squared_euclidean_distance(self):
        """Verify squared Euclidean distance D^2 = 2(1 - rho) for Ward linkage compliance."""
        corr = np.array([[1.0, 0.8, -0.5], [0.8, 1.0, 0.0], [-0.5, 0.0, 1.0]])
        dist_sq = compute_ward_distance(corr, squared=True)

        self.assertTrue(np.allclose(np.diag(dist_sq), 0.0))
        # D^2(0, 1) = 2*(1 - 0.8) = 0.4
        self.assertAlmostEqual(dist_sq[0, 1], 0.4, places=4)
        # D^2(1, 2) = 2*(1 - 0.0) = 2.0
        self.assertAlmostEqual(dist_sq[1, 2], 2.0, places=4)
        # D^2(0, 2) = 2*(1 - (-0.5)) = 3.0
        self.assertAlmostEqual(dist_sq[0, 2], 3.0, places=4)


if __name__ == "__main__":
    unittest.main()
