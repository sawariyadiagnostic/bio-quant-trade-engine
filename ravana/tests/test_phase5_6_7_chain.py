"""Integration tests for Phases 5, 6, and 7 (Muscles 13 through 20).

Validates the full institutional quant chain:
- Muscle 17: Dynamic Volatility Sizing (Inverse-vol & DSR Kelly scaling)
- Muscle 18: Circuit Breaker & Emergency Kill Switch
- Muscle 19: Broker Order Router (Idempotent state machine, asymmetric liquidation)
- Muscle 20: Real-Time Observability & Concept Drift Dashboard (PSI, KS, automated kill)
"""
import unittest
import numpy as np
import pandas as pd

from ravana.backtest.orders import OrderSide, RoundTripTrade
from ravana.domain.formula_ast import ASTNode
from ravana.execution.router import (
    BrokerOrderRouter,
    PaperBrokerGateway,
)
from ravana.monitor.drift_dashboard import ObservabilityDriftMonitor
from ravana.risk.circuit_breaker import CircuitBreakerKillSwitch
from ravana.risk.config import DrawdownState, RiskConfig
from ravana.risk.engine import AllocationResult
from ravana.risk.sizing import DynamicVolatilitySizer, VolatilitySizedAllocation
from ravana.validation.config import (
    CPCVResult,
    DSRGuardReport,
    MonteCarloReport,
    ValidationCertificate,
    WalkForwardResult,
)


class TestPhase567Chain(unittest.TestCase):
    """Test suite covering Muscles 17 through 20 within the integrated chain."""

    def setUp(self) -> None:
        np.random.seed(42)
        self.risk_cfg = RiskConfig(
            drawdown_halve_threshold=0.04,
            drawdown_halt_threshold=0.08,
            min_capital_gate=10_000.0,
        )
        self.breaker = CircuitBreakerKillSwitch(
            risk_config=self.risk_cfg,
            max_daily_loss_pct=0.02,
            max_consecutive_losses=4,
        )
        self.gateway = PaperBrokerGateway()
        self.router = BrokerOrderRouter(
            gateway=self.gateway,
            kill_switch=self.breaker,
            live_mode=False,
            allow_shorting=False,
        )
        self.sizer = DynamicVolatilitySizer(
            risk_config=self.risk_cfg,
            kelly_fraction=0.50,
            min_cash_reserve_pct=0.10,
        )

    def test_dynamic_volatility_sizer_inverse_vol_and_kelly(self) -> None:
        """Verify Muscle 17 scales allocations by inverse volatility and DSR Kelly modulation."""
        equity = 1_000_000.0
        base_alloc = AllocationResult(
            target_weights={"INFY": 0.5, "TCS": 0.5},
            target_shares={"INFY": 300, "TCS": 120},
            target_notionals={"INFY": 450_000.0, "TCS": 420_000.0},
            drawdown_state=DrawdownState.NORMAL,
            drawdown_multiplier=1.0,
            cash_reserve_inr=130_000.0,
            total_allocated_inr=870_000.0,
        )
        prices = {"INFY": 1500.0, "TCS": 3500.0}
        # INFY is half as volatile as TCS -> should receive higher weight after inverse-vol scaling
        vols = {"INFY": 0.15, "TCS": 0.30}

        wf_res = WalkForwardResult(
            folds=[],
            stitched_oos_returns=np.array([]),
            stitched_oos_equity=np.array([]),
            aggregate_oos_ic=0.05,
            aggregate_oos_rank_ic=0.04,
            aggregate_oos_sharpe=1.8,
            aggregate_oos_mdd=-0.05,
            walk_forward_efficiency=0.8,
            passed_validation=True,
        )
        cpcv_res = CPCVResult(
            num_combinations=15,
            num_backtest_paths=5,
            pbo=0.08,
            oos_sharpe_distribution=[1.5, 1.8, 1.6],
            mean_oos_sharpe=1.63,
            median_oos_sharpe=1.60,
            logit_distribution=[],
            passed_pbo_gate=True,
        )
        dsr_res = DSRGuardReport(
            observed_sharpe_annual=1.8,
            observed_sharpe_per_period=0.113,
            expected_max_sharpe_per_period=0.05,
            expected_max_sharpe_annual=0.8,
            empirical_trial_std_per_period=0.03,
            num_independent_trials=100,
            sample_length_bars=500,
            min_track_record_length=45.0,
            deflated_sharpe_ratio=0.96,
            skewness=-0.1,
            kurtosis=3.2,
            has_sufficient_track_record=True,
            passed_guard=True,
        )
        mc_res = MonteCarloReport(
            num_simulations=1000,
            block_size=10,
            median_cagr=0.25,
            var_95=-0.015,
            var_99=-0.025,
            cvar_95=-0.022,
            median_max_drawdown=0.04,
            dd_95_percentile=0.06,
            dd_99_percentile=0.075,
            probability_of_caution=0.02,
            probability_of_ruin=0.005,
            passed_stress_test=True,
        )

        cert = ValidationCertificate(
            ticker="INFY",
            champion_formula="TsDelay(LOG_RET_1, 1)",
            walk_forward=wf_res,
            cpcv=cpcv_res,
            dsr_guard=dsr_res,
            monte_carlo=mc_res,
            is_certified_for_live=True,
        )

        sized = self.sizer.size_allocation(
            allocation=base_alloc,
            current_prices=prices,
            realized_annual_vols=vols,
            current_equity=equity,
            certificate=cert,
        )

        self.assertGreater(sized.kelly_fraction_applied, 0.35)
        self.assertGreaterEqual(sized.cash_reserve_inr, equity * 0.10)
        self.assertLessEqual(sized.total_deployed_inr + sized.cash_reserve_inr, equity + 1.0)
        # Inverse vol: INFY weight should exceed TCS weight
        self.assertGreater(sized.sized_weights["INFY"], sized.sized_weights["TCS"])
        self.assertGreater(sized.sized_shares["INFY"], 0)
        self.assertGreater(sized.sized_shares["TCS"], 0)

    def test_circuit_breaker_intraday_loss_and_recovery(self) -> None:
        """Verify Muscle 18 trips on 2% daily loss and recovers on next day start."""
        self.breaker.start_new_day(1_000_000.0)

        # Normal small drawdown (-0.5%)
        s1 = self.breaker.evaluate(current_equity=995_000.0)
        self.assertTrue(s1.is_trading_allowed)
        self.assertFalse(s1.in_cooldown)

        # Severe drop past -2.0% -> (-2.5%)
        s2 = self.breaker.evaluate(current_equity=975_000.0)
        self.assertFalse(s2.is_trading_allowed)
        self.assertTrue(s2.in_cooldown)
        self.assertIn("Daily loss", s2.reason)

        # Start of next day with reset: trailing DD from 1M is only -2.5% (< 4% caution)
        self.breaker.start_new_day(975_000.0)
        s3 = self.breaker.evaluate(current_equity=975_000.0)
        self.assertTrue(s3.is_trading_allowed)

    def test_circuit_breaker_trailing_drawdown_and_consecutive_losses(self) -> None:
        """Verify Muscle 18 triggers caution and halt on trailing drawdowns and losing streaks."""
        # Trailing halt across days: day loss is small (-0.54%), but cumulative drawdown from peak is 8.5%
        s_halt = self.breaker.evaluate(
            current_equity=915_000.0,
            day_start_equity=920_000.0,
            peak_equity=1_000_000.0,
        )
        self.assertFalse(s_halt.is_trading_allowed)
        self.assertEqual(s_halt.drawdown_state, DrawdownState.HALTED)
        self.assertIn("HALT gate", s_halt.reason)

        # Consecutive losing trades test
        now = pd.Timestamp("2025-01-02 15:30", tz="Asia/Kolkata")
        loss_trades = [
            RoundTripTrade(
                entry_time=now,
                exit_time=now,
                ticker="RELIANCE",
                shares=10,
                entry_price=2500.0,
                exit_price=2450.0,
                gross_pnl=-500.0,
                net_pnl=-520.0,
                total_friction=20.0,
            )
            for _ in range(4)
        ]
        s_streak = self.breaker.evaluate(current_equity=1_000_000.0, closed_trades=loss_trades)
        self.assertFalse(s_streak.is_trading_allowed)
        self.assertEqual(s_streak.consecutive_losses, 4)
        self.assertIn("Consecutive loss streak", s_streak.reason)

    def test_broker_order_router_execution_and_asymmetric_kill_switch(self) -> None:
        """Verify Muscle 19 blocks BUY orders on kill switch trip but permits SELL liquidations."""
        base_alloc = AllocationResult(
            target_weights={"RELIANCE": 0.5},
            target_shares={"RELIANCE": 100},
            target_notionals={"RELIANCE": 250_000.0},
            drawdown_state=DrawdownState.NORMAL,
            drawdown_multiplier=1.0,
            cash_reserve_inr=50_000.0,
            total_allocated_inr=250_000.0,
        )
        sized_alloc = VolatilitySizedAllocation(
            base_allocation=base_alloc,
            sized_weights={"RELIANCE": 0.5},
            sized_shares={"RELIANCE": 100},
            sized_notionals={"RELIANCE": 250_000.0},
            vol_scalars={"RELIANCE": 1.0},
            kelly_fraction_applied=0.5,
            cash_reserve_inr=50_000.0,
            total_deployed_inr=250_000.0,
        )

        # 1. Normal BUY order execution
        b_status = self.breaker.evaluate(1_000_000.0)
        receipt = self.router.route_target_allocation(
            sized_allocation=sized_alloc,
            current_holdings={"RELIANCE": 0},
            current_prices={"RELIANCE": 2500.0},
            breaker_status=b_status,
        )
        self.assertEqual(len(receipt.submitted_orders), 1)
        self.assertEqual(len(receipt.executed_trades), 1)
        self.assertEqual(receipt.executed_trades[0].shares, 100)

        # 2. Trip manual kill switch
        self.breaker.trigger_kill_switch("EMERGENCY_RISK_HALT")
        killed_status = self.breaker.evaluate(1_000_000.0)

        # 3. New BUY entry must be rejected
        tcs_base = AllocationResult(
            target_weights={"TCS": 0.5},
            target_shares={"TCS": 50},
            target_notionals={"TCS": 175_000.0},
            drawdown_state=DrawdownState.HALTED,
            drawdown_multiplier=0.0,
            cash_reserve_inr=1_000_000.0,
            total_allocated_inr=0.0,
        )
        tcs_sized = VolatilitySizedAllocation(
            base_allocation=tcs_base,
            sized_weights={"TCS": 0.5},
            sized_shares={"TCS": 50},
            sized_notionals={"TCS": 175_000.0},
            vol_scalars={"TCS": 1.0},
            kelly_fraction_applied=0.0,
            cash_reserve_inr=1_000_000.0,
            total_deployed_inr=0.0,
        )
        receipt_blocked = self.router.route_target_allocation(
            sized_allocation=tcs_sized,
            current_holdings={"TCS": 0},
            current_prices={"TCS": 3500.0},
            breaker_status=killed_status,
        )
        self.assertEqual(len(receipt_blocked.rejected_orders), 1)
        self.assertEqual(receipt_blocked.rejected_orders[0].side, OrderSide.BUY)

        # 4. SELL liquidation must be permitted for portfolio safety
        empty_alloc = VolatilitySizedAllocation(
            base_allocation=base_alloc,
            sized_weights={},
            sized_shares={},
            sized_notionals={},
            vol_scalars={},
            kelly_fraction_applied=0.0,
            cash_reserve_inr=1_000_000.0,
            total_deployed_inr=0.0,
        )
        receipt_liquidate = self.router.route_target_allocation(
            sized_allocation=empty_alloc,
            current_holdings={"RELIANCE": 100},
            current_prices={"RELIANCE": 2500.0},
            breaker_status=killed_status,
        )
        self.assertEqual(len(receipt_liquidate.executed_trades), 1)
        self.assertEqual(receipt_liquidate.executed_trades[0].side, OrderSide.SELL)
        self.assertEqual(receipt_liquidate.executed_trades[0].shares, 100)

    def test_broker_order_router_naked_shorting_guard(self) -> None:
        """Verify Muscle 19 strictly prevents unauthorized naked shorting for cash equities."""
        base_alloc = AllocationResult(
            target_weights={},
            target_shares={},
            target_notionals={},
            drawdown_state=DrawdownState.NORMAL,
            drawdown_multiplier=1.0,
            cash_reserve_inr=1_000_000.0,
            total_allocated_inr=0.0,
        )
        # Target 0 when holding 40 shares -> sell exactly 40
        empty_alloc = VolatilitySizedAllocation(
            base_allocation=base_alloc,
            sized_weights={},
            sized_shares={},
            sized_notionals={},
            vol_scalars={},
            kelly_fraction_applied=0.0,
            cash_reserve_inr=1_000_000.0,
            total_deployed_inr=0.0,
        )
        b_status = self.breaker.evaluate(1_000_000.0)
        rec_sell = self.router.route_target_allocation(
            sized_allocation=empty_alloc,
            current_holdings={"INFY": 40},
            current_prices={"INFY": 1500.0},
            breaker_status=b_status,
        )
        self.assertEqual(len(rec_sell.executed_trades), 1)
        self.assertEqual(rec_sell.executed_trades[0].shares, 40)

        # Target -20 when holding 0 shares -> clamped to 0 orders
        short_alloc = VolatilitySizedAllocation(
            base_allocation=base_alloc,
            sized_weights={"INFY": -0.2},
            sized_shares={"INFY": -20},
            sized_notionals={"INFY": -30_000.0},
            vol_scalars={},
            kelly_fraction_applied=0.0,
            cash_reserve_inr=1_000_000.0,
            total_deployed_inr=0.0,
        )
        rec_naked = self.router.route_target_allocation(
            sized_allocation=short_alloc,
            current_holdings={"INFY": 0},
            current_prices={"INFY": 1500.0},
            breaker_status=b_status,
        )
        self.assertEqual(len(rec_naked.submitted_orders), 0)
        self.assertEqual(len(rec_naked.executed_trades), 0)

    def test_observability_drift_monitor_psi_and_ks(self) -> None:
        """Verify Muscle 20 computes PSI and KS drift and trips circuit breaker on critical shift."""
        monitor = ObservabilityDriftMonitor(
            psi_threshold=0.25,
            severe_psi_kill_threshold=0.50,
            min_sample_size=30,
        )

        ref_series = np.random.normal(0.0, 1.0, size=500)
        ref_df = pd.DataFrame({"LOG_RET_1": ref_series})

        # 1. Warm-up protection: < 30 samples returns 0.0 PSI
        tiny_cur = np.random.normal(5.0, 1.0, size=20)
        psi_tiny = monitor.compute_psi(ref_series, tiny_cur, min_samples=30)
        self.assertEqual(psi_tiny, 0.0)

        # 2. Healthy window with matching distribution
        healthy_cur = np.random.normal(0.02, 1.05, size=200)
        psi_healthy = monitor.compute_psi(ref_series, healthy_cur, min_samples=30)
        self.assertLess(psi_healthy, 0.25)

        # 3. Severe concept drift window
        drift_cur = np.random.normal(2.5, 2.0, size=300)
        psi_drift = monitor.compute_psi(ref_series, drift_cur, min_samples=30)
        self.assertGreater(psi_drift, 0.25)

        # 4. System health evaluation trips circuit breaker
        live_df = pd.DataFrame({"LOG_RET_1": drift_cur})
        report = monitor.evaluate_system_health(
            reference_features_df=ref_df,
            live_features_df=live_df,
            kill_switch=self.breaker,
        )
        self.assertTrue(report.kill_switch_triggered)
        self.assertIn("LOG_RET_1", report.drifted_features)
        self.assertTrue(self.breaker.is_killed)


if __name__ == "__main__":
    unittest.main()
