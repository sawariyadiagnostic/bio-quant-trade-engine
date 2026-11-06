"""Unit and Isolation Tests for Two-Stage Risk Sentry & Market Microstructure Protection.

Validates:
1. Stage 1 In-Process Atomic Freeze latency (<0.5ms) and SHM abort signal.
2. In-memory order validation rejection (<50us) during freeze.
3. Stage 2 Parallel Async Socket Dispatch for order cancellation and liquidation.
4. Lower-Circuit Evaporation Trap and Synthetic Delta Hedging (NIFTYBEES).
5. Statutory CDSL DP Charge floor (Rs 15,930 notional threshold and MIS rerouting).
6. Headless QuickAuth TOTP Session Watchdog.
"""
import asyncio
import time
import unittest
from unittest.mock import MagicMock, AsyncMock, patch

from src.execution.risk_sentry import RiskSentry
from src.execution.synthetic_hedge import SyntheticDeltaHedgeManager
from src.execution.shoonya_watchdog import ShoonyaSessionWatchdog
from src.pipeline.shared_telemetry import SharedMemoryBus


class TestRiskSentryIsolation(unittest.TestCase):
    def setUp(self):
        self.mock_gateway = MagicMock()
        self.mock_gateway.paper_trading = True
        self.mock_gateway.paper_positions = {
            "TCS": {"quantity": 10, "avg_price": 3500.0},
            "INFY": {"quantity": 20, "avg_price": 1500.0},
        }
        self.mock_gateway.balance = 50000.0

    def test_stage1_atomic_freeze_latency_and_rejection(self):
        """Stage 1 freeze must execute in <0.5ms (500us) and reject outbounds in <50us."""
        bus = SharedMemoryBus(create=True)
        try:
            sentry = RiskSentry(gateway=self.mock_gateway, shm_bus=bus)
            self.assertFalse(sentry.is_frozen)
            self.assertEqual(bus.header.abort_requested, 0)

            # Stage 1 Trigger Timing
            t0 = time.perf_counter_ns()
            report = sentry.freeze_stage1()
            elapsed_us = (time.perf_counter_ns() - t0) / 1000.0

            # Empirical Verification
            self.assertTrue(sentry.is_frozen)
            self.assertEqual(bus.header.abort_requested, 1)
            self.assertLess(report["elapsed_us"], 500.0, f"Stage 1 core freeze took {report['elapsed_us']:.2f}us, expected <500us")
            self.assertLess(elapsed_us, 2500.0, f"Stage 1 freeze + log took {elapsed_us:.2f}us, expected <2500us")
            self.assertEqual(report["status"], "FROZEN")

            # Outbound order rejection latency check
            t_rej0 = time.perf_counter_ns()
            permitted, reason, _ = sentry.validate_outbound_order({
                "symbol": "RELIANCE",
                "action": "BUY",
                "quantity": 10,
                "price": 2500.0,
                "product": "CNC"
            })
            rej_us = (time.perf_counter_ns() - t_rej0) / 1000.0

            self.assertFalse(permitted)
            self.assertIn("FREEZE_ACTIVE", reason)
            self.assertLess(rej_us, 50.0, f"Order rejection took {rej_us:.2f}us, expected <50us")
        finally:
            bus.close()

    def test_stage2_parallel_async_socket_dispatch(self):
        """Stage 2 must cancel orders and liquidate positions concurrently via async gather."""
        sentry = RiskSentry(gateway=self.mock_gateway)
        
        # Setup mock open orders and positions
        mock_orders = [
            {"orderno": f"ORD_{i}", "status": "OPEN", "symbol": f"SYM_{i}"}
            for i in range(5)
        ]
        self.mock_gateway.get_open_orders = MagicMock(return_value=mock_orders)
        self.mock_gateway.cancel_order_async = AsyncMock(return_value={"stat": "Ok"})
        self.mock_gateway.place_order_async = AsyncMock(return_value={"stat": "Ok"})
        
        # Run async Stage 2 liquidation
        loop = asyncio.new_event_loop()
        try:
            t0 = time.perf_counter()
            report = loop.run_until_complete(sentry.execute_stage2_liquidate())
            elapsed = time.perf_counter() - t0

            self.assertEqual(report["status"], "LIQUIDATION_COMPLETE")
            self.assertEqual(report["orders_cancelled_count"], 5)
            self.assertEqual(report["positions_liquidated_count"], 2)
            self.assertIn("TCS", report["liquidated_symbols"])
            self.assertIn("INFY", report["liquidated_symbols"])
            # Ensure liquidation proceeded directly to unencumbered cash (no LIQUIDBEES purchase)
            self.assertNotIn("LIQUIDBEES", report["liquidated_symbols"])
        finally:
            loop.close()

    def test_synthetic_delta_hedge_lower_circuit(self):
        """Lower-circuit evaporation trap intercepts sell and executes synthetic delta hedge on NIFTYBEES."""
        hedge_mgr = SyntheticDeltaHedgeManager()

        # Circuit proximity check: 100.10 with lower circuit at 100.00 is 10 bps (0.10% <= 0.15%)
        is_locked_proximity = hedge_mgr.is_circuit_locked(
            current_price=100.10,
            lower_circuit=100.00,
            bid_depth=1000
        )
        self.assertTrue(is_locked_proximity)

        # Bid depth evaporation: price is fine but bid depth is 0
        is_locked_evap = hedge_mgr.is_circuit_locked(
            current_price=105.00,
            lower_circuit=100.00,
            bid_depth=0
        )
        self.assertTrue(is_locked_evap)

        # Normal condition: price well above circuit, healthy bids
        is_safe = hedge_mgr.is_circuit_locked(
            current_price=108.00,
            lower_circuit=100.00,
            bid_depth=5000
        )
        self.assertFalse(is_safe)

        # Calculate synthetic delta hedge quantity
        # TCS: 10 shares @ 3500 (Rs 35,000), beta = 0.8 -> Rs 28,000 beta-weighted
        # INFY: 20 shares @ 1500 (Rs 30,000), beta = 1.1 -> Rs 33,000 beta-weighted
        # Total beta-weighted notional = Rs 61,000
        # Hedge asset NIFTYBEES @ Rs 250 -> 61,000 / 250 = 244 units short
        portfolio = [
            {"symbol": "TCS", "quantity": 10, "price": 3500.0, "beta": 0.8},
            {"symbol": "INFY", "quantity": 20, "price": 1500.0, "beta": 1.1},
        ]
        hedge_qty = hedge_mgr.calculate_hedge_quantity(
            portfolio_positions=portfolio,
            hedge_price=250.0,
            hedge_beta=1.0
        )
        self.assertEqual(hedge_qty, 244)

        order = hedge_mgr.create_hedge_order(total_qty=hedge_qty, hedge_symbol="NIFTYBEES")
        self.assertEqual(order["symbol"], "NIFTYBEES")
        self.assertEqual(order["action"], "SELL")
        self.assertEqual(order["quantity"], 244)
        self.assertEqual(order["product"], "MIS")

    def test_statutory_dp_charge_floor(self):
        """Delivery (CNC) sells under Rs 15,930 must be rerouted to MIS or rejected to cap DP drag."""
        sentry = RiskSentry(gateway=self.mock_gateway, auto_reroute_dp=True)

        # Sub-economic delivery sell: 5 shares @ Rs 1,000 = Rs 5,000 notional (DP fee Rs 15.93 = 0.32% drag)
        small_order = {
            "symbol": "TATASTEEL",
            "action": "SELL",
            "quantity": 5,
            "price": 1000.0,
            "product": "CNC"
        }
        permitted, reason, validated_order = sentry.validate_outbound_order(small_order)
        self.assertTrue(permitted)
        self.assertEqual(validated_order["product"], "MIS")
        self.assertIn("DP_FLOOR_AUTO_REROUTE_MIS", reason)

        # Non-reroute mode: should reject directly
        strict_sentry = RiskSentry(gateway=self.mock_gateway, auto_reroute_dp=False)
        permitted, reason, _ = strict_sentry.validate_outbound_order(small_order)
        self.assertFalse(permitted)
        self.assertIn("DP_FLOOR_BREACH", reason)

        # Economic delivery sell: 10 shares @ Rs 2,000 = Rs 20,000 notional (> Rs 15,930)
        large_order = {
            "symbol": "RELIANCE",
            "action": "SELL",
            "quantity": 10,
            "price": 2000.0,
            "product": "CNC"
        }
        permitted, reason, validated_order = strict_sentry.validate_outbound_order(large_order)
        self.assertTrue(permitted)
        self.assertEqual(validated_order["product"], "CNC")

        # Buy order: exempt from DP charge floor
        buy_order = {
            "symbol": "TATASTEEL",
            "action": "BUY",
            "quantity": 5,
            "price": 1000.0,
            "product": "CNC"
        }
        permitted, reason, validated_order = strict_sentry.validate_outbound_order(buy_order)
        self.assertTrue(permitted)
        self.assertEqual(validated_order["product"], "CNC")

    def test_headless_quickauth_session_watchdog(self):
        """Session watchdog monitors session health and refreshes authentication via TOTP."""
        mock_gw = MagicMock()
        mock_gw.susertoken = "OLD_VALID_TOKEN"
        mock_gw.get_limits.return_value = {"stat": "Ok", "cash": "50000.00"}
        mock_gw.authenticate.return_value = True

        # Provide a standard base32 test secret for TOTP
        test_secret = "JBSWY3DPEHPK3PXP"
        watchdog = ShoonyaSessionWatchdog(
            gateway=mock_gw,
            totp_secret=test_secret,
            interval_seconds=60.0
        )

        # Test TOTP generation
        totp_code = watchdog.generate_totp()
        self.assertTrue(totp_code.isdigit())
        self.assertEqual(len(totp_code), 6)

        # Test healthy session check
        self.assertTrue(watchdog.check_session_alive())

        # Simulate expired session
        mock_gw.get_limits.return_value = {"stat": "Not_Ok", "emsg": "Session Expired"}
        self.assertFalse(watchdog.check_session_alive())

        # Refresh session
        success = watchdog.refresh_session()
        self.assertTrue(success)
        mock_gw.authenticate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
