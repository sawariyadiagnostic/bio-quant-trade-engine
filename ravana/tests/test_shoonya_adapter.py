"""Comprehensive unit and integration tests for ShoonyaBrokerGateway adapter."""
from __future__ import annotations

import concurrent.futures
import time
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd

from ravana.backtest.config import AssetType, TradingRegime
from ravana.backtest.orders import Order, OrderSide, OrderStatus, Trade
from ravana.execution.router import (
    BrokerOrderRouter,
    ExecutionAuditRecord,
    EXIT_TRANCHE_1,
    RoutingReceipt,
)
from ravana.execution.shoonya_adapter import (
    ShoonyaBrokerGateway,
    ShoonyaGatewayConfig,
    TokenBucketRateLimiter,
)
from ravana.risk.circuit_breaker import CircuitBreakerKillSwitch, CircuitBreakerStatus
from ravana.risk.config import DrawdownState
from ravana.risk.engine import AllocationResult
from ravana.risk.sizing import VolatilitySizedAllocation
from src.execution.shoonya_gateway import ShoonyaCredentials


def _create_sample_sized_allocation() -> VolatilitySizedAllocation:
    """Helper to build a realistic VolatilitySizedAllocation for routing tests."""
    base = AllocationResult(
        target_weights={"NIFTYBEES": 0.6, "GOLDBEES": 0.4},
        target_shares={"NIFTYBEES": 200, "GOLDBEES": 100},
        target_notionals={"NIFTYBEES": 50000.0, "GOLDBEES": 6000.0},
        drawdown_state=DrawdownState.NORMAL,
        drawdown_multiplier=1.0,
        cash_reserve_inr=44000.0,
        total_allocated_inr=56000.0,
    )
    return VolatilitySizedAllocation(
        base_allocation=base,
        sized_weights={"NIFTYBEES": 0.6, "GOLDBEES": 0.4},
        sized_shares={"NIFTYBEES": 200, "GOLDBEES": 50},
        sized_notionals={"NIFTYBEES": 50000.0, "GOLDBEES": 3000.0},
        vol_scalars={"NIFTYBEES": 1.0, "GOLDBEES": 0.8},
        kelly_fraction_applied=0.5,
        cash_reserve_inr=47000.0,
        total_deployed_inr=53000.0,
    )


class TestShoonyaBrokerGateway(unittest.TestCase):
    """Test suite for ShoonyaBrokerGateway, rate limiter, and circuit breaker interlock."""

    def test_shoonya_gateway_config_defaults(self) -> None:
        """Verify default parameters of ShoonyaGatewayConfig match requirements."""
        cfg = ShoonyaGatewayConfig()
        self.assertTrue(cfg.paper_trading)
        self.assertEqual(cfg.exchange, "NSE")
        self.assertEqual(cfg.default_product_type, "C")
        self.assertEqual(cfg.order_timeout_seconds, 3.0)
        self.assertEqual(cfg.max_requests_per_second, 8.0)
        self.assertEqual(cfg.inter_order_delay_seconds, 0.125)
        self.assertEqual(cfg.asset_type, AssetType.ETF)
        self.assertEqual(cfg.regime, TradingRegime.OVERNIGHT)
        self.assertEqual(cfg.statutory_dp_fee_inr, 15.93)
        self.assertEqual(cfg.brokerage_per_order_inr, 0.0)
        self.assertIsNone(cfg.audit_file_path)

    def test_token_bucket_rate_limiter_burst_and_throttle(self) -> None:
        """Verify rate limiter allows capacity burst and throttles excess calls."""
        limiter = TokenBucketRateLimiter(max_rate=10.0, capacity=2.0)
        w1 = limiter.acquire(1.0)
        w2 = limiter.acquire(1.0)
        self.assertEqual(w1, 0.0)
        self.assertEqual(w2, 0.0)

        start = time.monotonic()
        w3 = limiter.acquire(1.0)
        elapsed = time.monotonic() - start
        self.assertGreater(w3, 0.0)
        self.assertGreaterEqual(elapsed, 0.08)

    def test_token_bucket_burst_ten_rapid_orders(self) -> None:
        """Fire a burst of 10 rapid orders and assert clean spacing without exceptions."""
        limiter = TokenBucketRateLimiter(max_rate=100.0, capacity=4.0)
        delays = []
        for _ in range(10):
            wait_time = limiter.acquire(1.0)
            delays.append(wait_time)

        self.assertEqual(len(delays), 10)
        self.assertEqual(delays[0], 0.0)
        self.assertEqual(delays[1], 0.0)
        self.assertEqual(delays[2], 0.0)
        self.assertEqual(delays[3], 0.0)
        self.assertTrue(all(d >= 0.0 for d in delays))
        self.assertGreater(sum(delays), 0.0)

        # Integration: 10 rapid orders placed on ShoonyaBrokerGateway in paper mode
        cfg = ShoonyaGatewayConfig(paper_trading=True, max_requests_per_second=200.0)
        gateway = ShoonyaBrokerGateway(cfg)
        now = pd.Timestamp.now(tz="Asia/Kolkata")
        trades = []
        for i in range(10):
            order = Order(f"BURST_ORD_{i}", now, "NIFTYBEES", OrderSide.BUY, 10)
            trades.append(gateway.place_order(order, reference_price=250.0))

        self.assertEqual(len(trades), 10)
        self.assertTrue(all(t is not None for t in trades))
        self.assertEqual(len(gateway.clordid_ledger), 10)

    def test_token_bucket_concurrent_threads(self) -> None:
        """Verify thread-safety of TokenBucketRateLimiter across concurrent worker threads."""
        limiter = TokenBucketRateLimiter(max_rate=100.0, capacity=10.0)

        def worker() -> float:
            return limiter.acquire(1.0)

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker) for _ in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        self.assertEqual(len(results), 10)
        self.assertTrue(all(r >= 0.0 for r in results))

    def test_totp_epoch_buffer_guard(self) -> None:
        """Verify 4-second epoch buffer guard pauses near 30s epoch boundary."""
        with patch("time.time", return_value=58.0):
            with patch("time.sleep") as mock_sleep:
                waited = ShoonyaBrokerGateway.wait_for_totp_epoch_buffer(min_buffer_seconds=4.0)
                self.assertGreater(waited, 2.0)
                mock_sleep.assert_called_once()

        with patch("time.time", return_value=40.0):
            with patch("time.sleep") as mock_sleep:
                waited = ShoonyaBrokerGateway.wait_for_totp_epoch_buffer(min_buffer_seconds=4.0)
                self.assertEqual(waited, 0.0)
                mock_sleep.assert_not_called()

    def test_totp_epoch_buffer_guard_mock_time_conditions(self) -> None:
        """Test wait_for_totp_epoch_buffer under mock time conditions (t >= 27s)."""
        test_cases = [
            (27.0, 3.15),   # 30.0 - 27.0 = 3.0s < 4.0s -> wait = 3.15s
            (28.5, 1.65),   # 30.0 - 28.5 = 1.5s < 4.0s -> wait = 1.65s
            (29.9, 0.25),   # 30.0 - 29.9 = 0.1s < 4.0s -> wait = 0.25s
            (58.0, 2.15),   # 60.0 - 58.0 = 2.0s < 4.0s -> wait = 2.15s
        ]
        for mock_t, expected_wait in test_cases:
            with patch("time.time", return_value=mock_t):
                with patch("time.sleep") as mock_sleep:
                    waited = ShoonyaBrokerGateway.wait_for_totp_epoch_buffer(min_buffer_seconds=4.0)
                    self.assertAlmostEqual(waited, expected_wait, places=2)
                    mock_sleep.assert_called_once()
                    self.assertAlmostEqual(mock_sleep.call_args[0][0], expected_wait, places=2)

        safe_cases = [10.0, 15.0, 25.0, 26.0, 40.0, 56.0]
        for mock_t in safe_cases:
            with patch("time.time", return_value=mock_t):
                with patch("time.sleep") as mock_sleep:
                    waited = ShoonyaBrokerGateway.wait_for_totp_epoch_buffer(min_buffer_seconds=4.0)
                    self.assertEqual(waited, 0.0)
                    mock_sleep.assert_not_called()

    def test_paper_mode_buy_and_sell_execution(self) -> None:
        """Test realistic paper fill, friction calculation, and clordid ledger."""
        cfg = ShoonyaGatewayConfig(paper_trading=True, statutory_dp_fee_inr=15.93)
        gateway = ShoonyaBrokerGateway(cfg)
        now = pd.Timestamp.now(tz="Asia/Kolkata")

        # 1. Buy Order
        buy_order = Order("ORD_BUY_001", now, "NIFTYBEES", OrderSide.BUY, 100)
        buy_trade = gateway.place_order(buy_order, reference_price=250.0, bar_volume=1e6)

        self.assertIsNotNone(buy_trade)
        self.assertEqual(buy_order.status, OrderStatus.FILLED)
        self.assertEqual(buy_trade.shares, 100)
        self.assertGreater(buy_trade.price, 250.0)
        self.assertEqual(buy_trade.friction.dp_charges_inr, 0.0)
        self.assertEqual(buy_trade.friction.brokerage_inr, 0.0)

        # Idempotent replay
        replay_trade = gateway.place_order(buy_order, reference_price=250.0)
        self.assertIs(replay_trade, buy_trade)

        # 2. Sell Order
        sell_order = Order("ORD_SELL_001", now, "NIFTYBEES", OrderSide.SELL, 50)
        sell_trade = gateway.place_order(sell_order, reference_price=252.0, bar_volume=1e6)

        self.assertIsNotNone(sell_trade)
        self.assertEqual(sell_order.status, OrderStatus.FILLED)
        self.assertEqual(sell_trade.shares, 50)
        self.assertEqual(sell_trade.friction.dp_charges_inr, 15.93)

    def test_paper_mode_rejection_conditions(self) -> None:
        """Verify invalid parameters or illiquid market bars correctly reject orders."""
        gateway = ShoonyaBrokerGateway(ShoonyaGatewayConfig(paper_trading=True))
        now = pd.Timestamp.now()

        o1 = Order("ORD_REJ_1", now, "NIFTYBEES", OrderSide.BUY, 0)
        self.assertIsNone(gateway.place_order(o1, reference_price=250.0))
        self.assertEqual(o1.status, OrderStatus.REJECTED)

        o2 = Order("ORD_REJ_2", now, "NIFTYBEES", OrderSide.BUY, 10)
        self.assertIsNone(gateway.place_order(o2, reference_price=float("nan")))
        self.assertEqual(o2.status, OrderStatus.REJECTED)

        o3 = Order("ORD_REJ_3", now, "NIFTYBEES", OrderSide.BUY, 10)
        self.assertIsNone(gateway.place_order(o3, reference_price=250.0, bar_volume=0.0))
        self.assertEqual(o3.status, OrderStatus.REJECTED)

    def test_live_mode_order_fill_and_retry(self) -> None:
        """Verify live order routing, session expiry handling, and fill polling."""
        cfg = ShoonyaGatewayConfig(
            paper_trading=False,
            credentials=ShoonyaCredentials(user_id="USER1", password="P", totp_key="KEY"),
            order_timeout_seconds=2.0,
        )
        gateway = ShoonyaBrokerGateway(cfg)
        now = pd.Timestamp.now()
        order = Order("LIVE_ORD_1", now, "BANKBEES", OrderSide.BUY, 50, limit_price=450.0)

        mock_place = MagicMock(side_effect=[
            {"stat": "Not_Ok", "emsg": "Session expired"},
            {"stat": "Ok", "norenordno": "261114999"},
        ])
        gateway.gateway.place_order = mock_place
        gateway.gateway.authenticate = MagicMock(return_value=True)

        mock_hist = [
            {"norenordno": "261114999", "status": "COMPLETE", "flqty": "50", "flprc": "450.25"}
        ]
        with patch.object(gateway, "_query_order_history", return_value=mock_hist):
            trade = gateway.place_order(order, reference_price=450.0)

        self.assertIsNotNone(trade)
        self.assertEqual(trade.shares, 50)
        self.assertEqual(trade.price, 450.25)
        self.assertEqual(order.status, OrderStatus.FILLED)
        self.assertGreaterEqual(gateway.gateway.authenticate.call_count, 1)
        self.assertEqual(mock_place.call_count, 2)

    def test_session_expiry_fault_injection_and_reauth_retry(self) -> None:
        """Mock Finvasia Session Expired on initial order and verify auto re-auth and fill."""
        cfg = ShoonyaGatewayConfig(
            paper_trading=False,
            credentials=ShoonyaCredentials(user_id="U1234", password="pwd", totp_key="KEY123"),
            order_timeout_seconds=2.0,
            inter_order_delay_seconds=0.0,
        )
        gateway = ShoonyaBrokerGateway(cfg)
        gateway.gateway.susertoken = "EXPIRED_TOKEN_XYZ"
        now = pd.Timestamp.now(tz="Asia/Kolkata")
        order = Order("FAULT_INJ_ORD_1", now, "NIFTYBEES", OrderSide.BUY, 100, limit_price=250.0)

        mock_place = MagicMock(side_effect=[
            {"stat": "Not_Ok", "emsg": "Session Expired"},
            {"stat": "Ok", "norenordno": "261114001"},
        ])
        gateway.gateway.place_order = mock_place

        reauth_called = []

        def mock_auth() -> bool:
            reauth_called.append(True)
            gateway.gateway.susertoken = "FRESH_TOKEN_ABC"
            return True

        gateway._authenticate_live = MagicMock(side_effect=mock_auth)

        mock_hist = [
            {"norenordno": "261114001", "status": "COMPLETE", "flqty": "100", "flprc": "250.25"}
        ]
        with patch.object(gateway, "_query_order_history", return_value=mock_hist):
            trade = gateway.place_order(order, reference_price=250.0)

        self.assertIsNotNone(trade)
        self.assertEqual(order.status, OrderStatus.FILLED)
        self.assertEqual(trade.shares, 100)
        self.assertEqual(trade.price, 250.25)
        self.assertEqual(len(reauth_called), 1)
        self.assertEqual(mock_place.call_count, 2)
        self.assertEqual(gateway.gateway.susertoken, "FRESH_TOKEN_ABC")
        self.assertIn("FAULT_INJ_ORD_1", gateway.clordid_ledger)

    def test_session_expiry_reauth_failure_leads_to_rejection(self) -> None:
        """Verify order is rejected if session expires and re-authentication fails."""
        cfg = ShoonyaGatewayConfig(
            paper_trading=False,
            credentials=ShoonyaCredentials(user_id="U1234", password="pwd", totp_key="KEY123"),
            inter_order_delay_seconds=0.0,
        )
        gateway = ShoonyaBrokerGateway(cfg)
        gateway.gateway.susertoken = "OLD_TOKEN"
        now = pd.Timestamp.now(tz="Asia/Kolkata")
        order = Order("FAULT_REAUTH_FAIL", now, "NIFTYBEES", OrderSide.BUY, 50)

        gateway.gateway.place_order = MagicMock(return_value={"stat": "Not_Ok", "emsg": "Session Expired"})
        gateway._authenticate_live = MagicMock(return_value=False)

        trade = gateway.place_order(order, reference_price=250.0)
        self.assertIsNone(trade)
        self.assertEqual(order.status, OrderStatus.REJECTED)
        self.assertNotIn("FAULT_REAUTH_FAIL", gateway.clordid_ledger)

    def test_order_fill_polling_timeout_rejects_order_cleanly(self) -> None:
        """Verify order fill timeout results in OrderStatus.REJECTED and no ghost trade."""
        cfg = ShoonyaGatewayConfig(
            paper_trading=False,
            credentials=ShoonyaCredentials(user_id="U1234", password="pwd", totp_key="KEY123"),
            order_timeout_seconds=0.05,
            inter_order_delay_seconds=0.0,
        )
        gateway = ShoonyaBrokerGateway(cfg)
        gateway.gateway.susertoken = "VALID_TOKEN"
        now = pd.Timestamp.now(tz="Asia/Kolkata")
        order = Order("ORD_TIMEOUT_TEST", now, "NIFTYBEES", OrderSide.BUY, 50)

        gateway.gateway.place_order = MagicMock(return_value={"stat": "Ok", "norenordno": "ORD_TO_999"})
        pending_hist = [{"norenordno": "ORD_TO_999", "status": "PENDING", "flqty": "0"}]
        with patch.object(gateway, "_query_order_history", return_value=pending_hist):
            trade = gateway.place_order(order, reference_price=250.0)

        self.assertIsNone(trade)
        self.assertEqual(order.status, OrderStatus.REJECTED)
        self.assertNotIn("ORD_TIMEOUT_TEST", gateway.clordid_ledger)

    def test_order_fill_terminal_rejection_and_cancellation(self) -> None:
        """Verify broker CANCELLED and REJECTED states transition order to REJECTED."""
        cfg = ShoonyaGatewayConfig(
            paper_trading=False,
            credentials=ShoonyaCredentials(user_id="U1234", password="pwd", totp_key="KEY123"),
            order_timeout_seconds=1.0,
            inter_order_delay_seconds=0.0,
        )
        gateway = ShoonyaBrokerGateway(cfg)
        gateway.gateway.susertoken = "VALID_TOKEN"
        now = pd.Timestamp.now(tz="Asia/Kolkata")

        # Case A: Exchange returns REJECTED
        ord_rej = Order("ORD_EXCH_REJ", now, "BANKBEES", OrderSide.BUY, 50)
        gateway.gateway.place_order = MagicMock(return_value={"stat": "Ok", "norenordno": "REJ_111"})
        hist_rej = [{"status": "REJECTED", "rejreason": "RMS: Margin Insufficient"}]
        with patch.object(gateway, "_query_order_history", return_value=hist_rej):
            trade_rej = gateway.place_order(ord_rej, reference_price=450.0)

        self.assertIsNone(trade_rej)
        self.assertEqual(ord_rej.status, OrderStatus.REJECTED)
        self.assertNotIn("ORD_EXCH_REJ", gateway.clordid_ledger)

        # Case B: Exchange returns CANCELLED
        ord_cnl = Order("ORD_EXCH_CNL", now, "BANKBEES", OrderSide.BUY, 50)
        gateway.gateway.place_order = MagicMock(return_value={"stat": "Ok", "norenordno": "CNL_222"})
        hist_cnl = [{"status": "CANCELLED", "rejreason": "Cancelled by user"}]
        with patch.object(gateway, "_query_order_history", return_value=hist_cnl):
            trade_cnl = gateway.place_order(ord_cnl, reference_price=450.0)

        self.assertIsNone(trade_cnl)
        self.assertEqual(ord_cnl.status, OrderStatus.REJECTED)
        self.assertNotIn("ORD_EXCH_CNL", gateway.clordid_ledger)

    def test_circuit_breaker_interlock(self) -> None:
        """Verify Muscle 18 circuit breaker trip triggers cancel and liquidation."""
        gateway = ShoonyaBrokerGateway(ShoonyaGatewayConfig(paper_trading=True))
        now = pd.Timestamp.now()

        buy_order = Order("BUY_FOR_CB", now, "NIFTYBEES", OrderSide.BUY, 20)
        gateway.place_order(buy_order, reference_price=250.0)

        status = CircuitBreakerStatus(
            is_trading_allowed=False,
            kill_switch_engaged=True,
            drawdown_state=DrawdownState.HALTED,
            daily_return_pct=-0.025,
            trailing_drawdown_pct=0.035,
            consecutive_losses=4,
            in_cooldown=True,
            reason="Max daily loss breach 2.5%",
        )

        result = gateway.on_circuit_breaker_tripped(
            status, cancel_pending=True, liquidate_all=True
        )
        self.assertEqual(result["status"], "CIRCUIT_BREAKER_HANDLED")
        self.assertEqual(result["reason"], "Max daily loss breach 2.5%")
        self.assertEqual(result["cancel_result"]["stat"], "Ok")
        self.assertEqual(result["liquidation_result"]["stat"], "Ok")

    def test_muscle18_kill_switch_panic_interlock(self) -> None:
        """Verify KillSwitch triggers panic cancel/liquidation and blocks BUY while allowing SELL."""
        gateway = ShoonyaBrokerGateway(ShoonyaGatewayConfig(paper_trading=True))
        kill_switch = CircuitBreakerKillSwitch()
        router = BrokerOrderRouter(gateway=gateway, kill_switch=kill_switch, live_mode=False)

        now = pd.Timestamp.now(tz="Asia/Kolkata")
        buy_setup = Order("PRE_SEED_BUY", now, "NIFTYBEES", OrderSide.BUY, 50)
        gateway.place_order(buy_setup, reference_price=250.0)
        self.assertIn("NIFTYBEES-EQ", gateway.gateway.paper_positions)

        # 1. Trigger Emergency Kill Switch
        kill_switch.trigger_kill_switch("CRITICAL_DRAWDOWN")
        self.assertTrue(kill_switch.is_killed)

        breaker_status = kill_switch.evaluate(current_equity=80000.0)
        self.assertFalse(breaker_status.is_trading_allowed)
        self.assertTrue(breaker_status.kill_switch_engaged)
        self.assertIn("CRITICAL_DRAWDOWN", breaker_status.reason)

        # 2. Verify gateway on_circuit_breaker_tripped cancels and liquidates
        cb_res = gateway.on_circuit_breaker_tripped(
            breaker_status, cancel_pending=True, liquidate_all=True
        )
        self.assertEqual(cb_res["status"], "CIRCUIT_BREAKER_HANDLED")
        self.assertEqual(cb_res["cancel_result"]["stat"], "Ok")
        self.assertEqual(cb_res["liquidation_result"]["stat"], "Ok")
        self.assertEqual(len(gateway.gateway.paper_positions), 0)

        # 3. Verify BrokerOrderRouter blocks BUY while allowing SELL liquidation
        orders = [
            Order("BLOCKED_BUY", now, "BANKBEES", OrderSide.BUY, 20),
            Order("ALLOWED_SELL_LIQ", now, "NIFTYBEES", OrderSide.SELL, 50),
        ]
        prices = {"BANKBEES": 450.0, "NIFTYBEES": 250.0}
        receipt = router.route_orders(orders, prices, breaker_status)

        self.assertEqual(len(receipt.submitted_orders), 1)
        self.assertEqual(len(receipt.rejected_orders), 1)
        self.assertEqual(receipt.rejected_orders[0].order_id, "BLOCKED_BUY")
        self.assertEqual(receipt.rejected_orders[0].status, OrderStatus.REJECTED)
        self.assertIn("BLOCKED_BUY", receipt.rejection_reasons)
        self.assertEqual(receipt.executed_trades[0].ticker, "NIFTYBEES")
        self.assertEqual(receipt.executed_trades[0].side, OrderSide.SELL)

    def test_integration_with_broker_order_router(self) -> None:
        """Verify ShoonyaBrokerGateway seamlessly routes orders via BrokerOrderRouter."""
        gateway = ShoonyaBrokerGateway(ShoonyaGatewayConfig(paper_trading=True))
        kill_switch = CircuitBreakerKillSwitch()
        router = BrokerOrderRouter(gateway=gateway, kill_switch=kill_switch, live_mode=False)

        now = pd.Timestamp.now(tz="Asia/Kolkata")
        orders = [
            Order("ROUTE_1", now, "NIFTYBEES", OrderSide.BUY, 100),
            Order("ROUTE_2", now, "BANKBEES", OrderSide.BUY, 50),
        ]
        prices = {"NIFTYBEES": 250.0, "BANKBEES": 500.0}
        breaker_status = kill_switch.evaluate(current_equity=100000.0)

        receipt = router.route_orders(orders, prices, breaker_status)
        self.assertEqual(len(receipt.submitted_orders), 2)
        self.assertEqual(len(receipt.executed_trades), 2)
        self.assertEqual(len(receipt.rejected_orders), 0)
        self.assertEqual(len(receipt.audit_records), 2)

    def test_muscle19_router_volatility_sized_allocation_end_to_end(self) -> None:
        """Wire ShoonyaBrokerGateway into BrokerOrderRouter with VolatilitySizedAllocation."""
        gateway = ShoonyaBrokerGateway(ShoonyaGatewayConfig(paper_trading=True, statutory_dp_fee_inr=15.93))
        kill_switch = CircuitBreakerKillSwitch()
        router = BrokerOrderRouter(gateway=gateway, kill_switch=kill_switch, live_mode=False)

        now = pd.Timestamp.now(tz="Asia/Kolkata")
        sized_alloc = _create_sample_sized_allocation()
        current_holdings = {"NIFTYBEES": 100, "GOLDBEES": 80}
        current_prices = {"NIFTYBEES": 250.0, "GOLDBEES": 60.0}
        breaker_status = kill_switch.evaluate(current_equity=100000.0)

        receipt = router.route_target_allocation(
            sized_allocation=sized_alloc,
            current_holdings=current_holdings,
            current_prices=current_prices,
            breaker_status=breaker_status,
            timestamp=now,
        )

        self.assertEqual(len(receipt.submitted_orders), 2)
        self.assertEqual(len(receipt.executed_trades), 2)
        self.assertEqual(len(receipt.rejected_orders), 0)
        self.assertEqual(len(receipt.audit_records), 2)

        for audit in receipt.audit_records:
            self.assertIsInstance(audit, ExecutionAuditRecord)
            self.assertGreater(audit.filled_shares, 0)
            self.assertTrue(np.isfinite(audit.realized_slippage_bps))
            self.assertTrue(np.isfinite(audit.modeled_slippage_bps))

        sell_trade = [t for t in receipt.executed_trades if t.side == OrderSide.SELL][0]
        self.assertEqual(sell_trade.friction.dp_charges_inr, 15.93)
        self.assertGreater(sell_trade.friction.stt_inr, 0.0)

    def test_shoonya_helper_methods(self) -> None:
        """Verify ShoonyaBrokerGateway helper methods for continuation, Stoikov micro-price, and stagger."""
        gateway = ShoonyaBrokerGateway(ShoonyaGatewayConfig(paper_trading=True))
        self.assertEqual(gateway.continuation_holds_count, 0)

        # 1. should_hold_continuation
        self.assertTrue(gateway.should_hold_continuation("NIFTYBEES", 0.65, 0.50))
        self.assertFalse(gateway.should_hold_continuation("NIFTYBEES", 0.35, 0.50))
        self.assertTrue(gateway.should_hold_continuation("NIFTYBEES", 0.50, 0.50))

        # 2. compute_stoikov_micro_price
        p_sym = gateway.compute_stoikov_micro_price(
            bid_price=100.0, ask_price=101.0, bid_qty=500, ask_qty=500
        )
        self.assertAlmostEqual(p_sym, 100.5, places=2)

        p_skew = gateway.compute_stoikov_micro_price(
            bid_price=100.0, ask_price=101.0, bid_qty=900, ask_qty=100
        )
        self.assertAlmostEqual(p_skew, 100.9, places=2)

        # 3. get_staggered_open_timestamp
        t_open = pd.Timestamp("2026-01-05 09:15:00", tz="Asia/Kolkata")
        t_stag = gateway.get_staggered_open_timestamp(t_open)
        self.assertEqual(t_stag, pd.Timestamp("2026-01-05 09:15:45", tz="Asia/Kolkata"))

        t_mid = pd.Timestamp("2026-01-05 09:20:00", tz="Asia/Kolkata")
        self.assertEqual(gateway.get_staggered_open_timestamp(t_mid), t_mid)

    def test_router_continuation_gate_and_two_tranche_exits(self) -> None:
        """Verify BrokerOrderRouter morning continuation hold and EXIT_TRANCHE_1 half-exit."""
        gateway = ShoonyaBrokerGateway(ShoonyaGatewayConfig(paper_trading=True))
        router = BrokerOrderRouter(gateway=gateway, live_mode=False)

        # 1. Continuation Hold: 09:15 morning liquidation skipped when signal >= 0.50
        t_morning = pd.Timestamp("2026-01-05 09:15:00", tz="Asia/Kolkata")
        empty_alloc = VolatilitySizedAllocation(
            base_allocation=AllocationResult(
                target_weights={}, target_shares={}, target_notionals={},
                drawdown_state=DrawdownState.NORMAL, drawdown_multiplier=1.0,
                cash_reserve_inr=100000.0, total_allocated_inr=0.0,
            ),
            sized_weights={}, sized_shares={}, sized_notionals={},
            vol_scalars={}, kelly_fraction_applied=0.5,
            cash_reserve_inr=100000.0, total_deployed_inr=0.0,
        )
        receipt_hold = router.route_target_allocation(
            sized_allocation=empty_alloc,
            current_holdings={"NIFTYBEES": 100},
            current_prices={"NIFTYBEES": 250.0},
            timestamp=t_morning,
            hold_on_strong_continuation=True,
            continuation_threshold=0.50,
            continuation_signals={"NIFTYBEES": 0.75},
        )
        self.assertEqual(len(receipt_hold.submitted_orders), 0)
        self.assertEqual(len(receipt_hold.executed_trades), 0)
        self.assertEqual(router.continuation_holds_count, 1)
        self.assertEqual(gateway.continuation_holds_count, 1)

        # 2. Two-tranche exit: EXIT_TRANCHE_1 routes exactly 50% shares
        receipt_t1 = router.route_target_allocation(
            sized_allocation=EXIT_TRANCHE_1,
            current_holdings={"NIFTYBEES": 100, "GOLDBEES": 40},
            current_prices={"NIFTYBEES": 250.0, "GOLDBEES": 60.0},
            timestamp=pd.Timestamp("2026-01-05 11:30:00", tz="Asia/Kolkata"),
        )
        self.assertEqual(len(receipt_t1.submitted_orders), 2)
        shares_map = {t.ticker: t.shares for t in receipt_t1.executed_trades}
        self.assertEqual(shares_map["NIFTYBEES"], 50)
        self.assertEqual(shares_map["GOLDBEES"], 20)


if __name__ == "__main__":
    unittest.main()
