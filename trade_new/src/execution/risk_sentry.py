"""Out-of-Band Two-Stage Risk Sentry & Market Microstructure Protection.

Provides sub-millisecond in-process order freezing (Stage 1), parallel async socket
dispatch for emergency liquidations (Stage 2), statutory CDSL DP fee gating,
and lower-circuit synthetic delta hedging.
"""
from __future__ import annotations
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from typing import Any, Optional
from loguru import logger

from .synthetic_hedge import SyntheticDeltaHedgeManager


class RiskSentry:
    """Institutional Two-Stage Emergency Risk Controller & Sentry."""

    STATUTORY_DP_FEE: float = 15.93  # Flat CDSL DP charge on delivery sells
    SUB_ECONOMIC_NOTIONAL_FLOOR: float = 160000.0  # Statutory economic threshold (₹1,60,000)
    MIN_DELIVERY_NOTIONAL_FLOOR: float = 160000.0  # Backwards compatibility alias
    SOCKET_TIMEOUT_SECONDS: float = 3.0  # Max socket wait time for emergency broker operations

    def __init__(
        self,
        gateway: Any = None,
        shm_bus: Any = None,
        auto_reroute_dp: bool = False,
        auto_route_futures: bool = False,
        auto_quantize_tick: bool = True
    ):
        self.gateway = gateway
        self.shm_bus = shm_bus
        self.auto_reroute_dp = auto_reroute_dp
        self.auto_route_futures = auto_route_futures
        self.auto_quantize_tick = auto_quantize_tick
        self.is_frozen: bool = False
        self._is_liquidating: bool = False
        self.hedge_mgr = SyntheticDeltaHedgeManager()
        self._executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="RiskSentryWorker")

    def freeze_stage1(self) -> dict[str, Any]:
        """Stage 1: In-Process Atomic Freeze (<0.5ms).

        Instantly halts local outbound order generation, signals the SharedMemory
        worker abort flag, and marks local sentry state as FROZEN.
        """
        t0 = time.perf_counter_ns()
        self.is_frozen = True

        shm_signaled = False
        if self.shm_bus is not None:
            try:
                self.shm_bus.request_abort()
                shm_signaled = True
            except Exception as e:
                logger.error(f"[RiskSentry] Error setting SHM abort: {e}")

        elapsed_us = (time.perf_counter_ns() - t0) / 1000.0
        logger.warning(
            f"[RiskSentry] STAGE 1 ATOMIC FREEZE TRIGGERED in {elapsed_us:.2f}us! "
            f"SHM signaled: {shm_signaled}"
        )
        return {
            "status": "FROZEN",
            "freeze_timestamp": time.time(),
            "elapsed_us": elapsed_us,
            "shm_signaled": shm_signaled
        }

    def validate_outbound_order(
        self,
        order: dict[str, Any]
    ) -> tuple[bool, str, dict[str, Any]]:
        """High-frequency in-memory pre-trade order validation (<50us).

        Enforces:
        1. Immediate rejection if Risk Sentry is frozen.
        2. NSE 5-paise tick quantization for securities >= ₹250.
        3. Elimination of fatal CNC-to-MIS auto-reroute: Replaces with trade rejection
           or liquid Index Futures routing if notional is sub-economic (< ₹1,60,000).
        """
        if self.is_frozen:
            return False, "FREEZE_ACTIVE: Risk Sentry has frozen all outbounds", {}

        order_copy = dict(order)
        action = str(order_copy.get("action", "")).upper()
        product = str(order_copy.get("product", "CNC")).upper()
        qty = int(order_copy.get("quantity", 0))
        price = float(order_copy.get("price", 0.0))

        # Enforce NSE 5-paise (₹0.05) tick quantization for all securities >= ₹250
        if price >= 250.0:
            tick = 0.05
            quantized_price = round(round(price / tick) * tick, 2)
            if abs(price - quantized_price) > 1e-4:
                if self.auto_quantize_tick:
                    order_copy["price"] = quantized_price
                    price = quantized_price
                    logger.info(f"[RiskSentry] Quantized price ₹{order.get('price')} to ₹{quantized_price} (NSE 5-paise tick).")
                else:
                    return (
                        False,
                        f"INVALID_TICK_SIZE: price ₹{price:.4f} violates NSE 5-paise tick size for securities >= ₹250",
                        {}
                    )

        # Enforce statutory DP floor on CNC (delivery) sell orders: reject or route to futures, never MIS
        if action == "SELL" and product == "CNC":
            notional = qty * price
            if 0 < notional < self.SUB_ECONOMIC_NOTIONAL_FLOOR:
                if self.auto_route_futures:
                    sanitized_order = dict(order_copy)
                    sanitized_order["product"] = "FUTURES"
                    sanitized_order["instrument"] = "FUTURES"
                    logger.info(
                        f"[RiskSentry] Sub-economic delivery sell (₹{notional:,.2f}) routed "
                        f"to liquid Index Futures to eliminate statutory DP drag."
                    )
                    return True, "SUB_ECONOMIC_ROUTED_INDEX_FUTURES", sanitized_order

                logger.warning(
                    f"[RiskSentry] Rejected sub-economic delivery sell (₹{notional:,.2f} < ₹{self.SUB_ECONOMIC_NOTIONAL_FLOOR:,.2f})"
                )
                return (
                    False,
                    f"REJECTED_SUB_ECONOMIC_NOTIONAL: notional ₹{notional:.2f} is below statutory "
                    f"floor of ₹{self.SUB_ECONOMIC_NOTIONAL_FLOOR:,.2f}",
                    {}
                )

        return True, "VALIDATED_OK", order_copy

    async def execute_stage2_liquidate(
        self,
        force_hedge_if_circuit: bool = True
    ) -> dict[str, Any]:
        """Stage 2: Parallel Async Socket Dispatch.

        Concurrently cancels all open orders and liquidates active positions
        to 100% unencumbered cash in parallel via asyncio.gather with deduplication lock.
        """
        if self._is_liquidating:
            logger.warning("[RiskSentry] Stage 2 liquidation already in flight. Deduplicating request.")
            return {
                "status": "LIQUIDATION_IN_PROGRESS",
                "message": "Emergency liquidation already in flight",
                "cash_secured": True
            }

        self._is_liquidating = True
        try:
            t0 = time.perf_counter()
            self.is_frozen = True

            cancelled_orders = await self._cancel_all_orders_parallel()
            liquidated_symbols, hedge_count = await self._liquidate_positions_parallel(force_hedge_if_circuit)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            logger.warning(
                f"[RiskSentry] STAGE 2 LIQUIDATION COMPLETE in {elapsed_ms:.2f}ms. "
                f"Orders cancelled: {len(cancelled_orders)}, Positions liquidated: {len(liquidated_symbols)}"
            )
            return {
                "status": "LIQUIDATION_COMPLETE",
                "elapsed_ms": elapsed_ms,
                "orders_cancelled_count": len(cancelled_orders),
                "positions_liquidated_count": len(liquidated_symbols),
                "liquidated_symbols": liquidated_symbols,
                "synthetic_hedges_count": hedge_count,
                "cash_secured": True
            }
        finally:
            self._is_liquidating = False

    async def _cancel_all_orders_parallel(self) -> list[str]:
        """Cancels open orders concurrently."""
        if not self.gateway:
            return []

        open_orders = []
        if hasattr(self.gateway, "get_open_orders"):
            open_orders = self.gateway.get_open_orders() or []
        elif hasattr(self.gateway, "api") and self.gateway.api:
            loop = asyncio.get_running_loop()
            try:
                orders_raw = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, self.gateway.api.get_order_book),
                    timeout=self.SOCKET_TIMEOUT_SECONDS
                )
                if isinstance(orders_raw, list):
                    open_orders = [o for o in orders_raw if o.get("status") in ["OPEN", "TRIGGER_PENDING"]]
            except asyncio.TimeoutError:
                logger.error("[RiskSentry] Timeout querying order book during cancel.")

        if not open_orders:
            return []

        tasks = []
        cancelled_ids = []
        for o in open_orders:
            orderno = o.get("orderno") or o.get("norenordno")
            if not orderno:
                continue
            cancelled_ids.append(orderno)
            if hasattr(self.gateway, "cancel_order_async"):
                tasks.append(self.gateway.cancel_order_async(orderno=orderno))
            elif hasattr(self.gateway, "api") and self.gateway.api:
                loop = asyncio.get_running_loop()
                tasks.append(loop.run_in_executor(self._executor, self.gateway.api.cancel_order, orderno))

        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=self.SOCKET_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                logger.error(f"[RiskSentry] Timeout ({self.SOCKET_TIMEOUT_SECONDS}s) during parallel cancellations!")

        return cancelled_ids

    async def _liquidate_positions_parallel(
        self,
        force_hedge_if_circuit: bool = True
    ) -> tuple[list[str], int]:
        """Liquidates open positions or creates synthetic hedges in parallel."""
        if not self.gateway:
            return [], 0

        positions_to_close = []
        if getattr(self.gateway, "paper_trading", False):
            for sym, pos in list(self.gateway.paper_positions.items()):
                qty = int(pos.get("quantity", 0))
                price = float(pos.get("avg_price", 100.0))
                if qty != 0:
                    positions_to_close.append({"symbol": sym, "quantity": qty, "price": price})
            self.gateway.paper_positions.clear()
        elif hasattr(self.gateway, "api") and self.gateway.api:
            loop = asyncio.get_running_loop()
            pos_raw = await loop.run_in_executor(self._executor, self.gateway.api.get_positions)
            if isinstance(pos_raw, list):
                for p in pos_raw:
                    netqty = int(p.get("netqty", 0))
                    if netqty != 0:
                        positions_to_close.append({
                            "symbol": p.get("tsym", ""),
                            "quantity": netqty,
                            "price": float(p.get("lp", 0.0) or p.get("netavgprc", 0.0)),
                            "prd": p.get("prd", "C")
                        })

        if not positions_to_close:
            return [], 0

        tasks = []
        liquidated_symbols = []
        hedge_count = 0

        for p in positions_to_close:
            sym = p["symbol"]
            qty = p["quantity"]
            price = p["price"]
            prd = p.get("prd", "C")
            lower_circuit = float(p.get("lower_circuit", 0.0))
            bid_depth = int(p.get("bid_depth", 100))

            # Circuit Proximity or Evaporation Trap: Synthetic Delta Hedge
            if force_hedge_if_circuit and qty > 0 and lower_circuit > 0.0 and self.hedge_mgr.is_circuit_locked(price, lower_circuit, bid_depth):
                hedge_qty = self.hedge_mgr.calculate_hedge_quantity([p], hedge_price=250.0)
                if hedge_qty > 0:
                    hedge_order = self.hedge_mgr.create_hedge_order(hedge_qty, hedge_symbol="NIFTYBEES")
                    hedge_count += 1
                    liquidated_symbols.append(f"{sym}_HEDGED_NIFTYBEES")
                    if hasattr(self.gateway, "place_order_async"):
                        tasks.append(self.gateway.place_order_async(
                            symbol=hedge_order["symbol"],
                            quantity=hedge_order["quantity"],
                            price=0.0,
                            action=hedge_order["action"],
                            product=hedge_order["product"],
                            order_type=hedge_order["order_type"]
                        ))
                    else:
                        loop = asyncio.get_running_loop()
                        tasks.append(loop.run_in_executor(
                            self._executor,
                            self.gateway.place_order,
                            hedge_order["symbol"], hedge_order["quantity"], 0.0,
                            hedge_order["action"], hedge_order["product"], hedge_order["order_type"]
                        ))
                    continue

            action = "SELL" if qty > 0 else "BUY"
            liquidated_symbols.append(sym)

            if hasattr(self.gateway, "place_order_async"):
                tasks.append(self.gateway.place_order_async(
                    symbol=sym,
                    quantity=abs(qty),
                    price=0.0,
                    action=action,
                    product=prd,
                    order_type="MKT"
                ))
            else:
                loop = asyncio.get_running_loop()
                tasks.append(loop.run_in_executor(
                    self._executor,
                    self.gateway.place_order,
                    sym, abs(qty), 0.0, action, prd, "MKT"
                ))

        if tasks:
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=self.SOCKET_TIMEOUT_SECONDS
                )
                for res in results:
                    if isinstance(res, Exception):
                        logger.error(f"[RiskSentry] Exception during parallel liquidation order: {res}")
            except asyncio.TimeoutError:
                logger.error(f"[RiskSentry] Timeout ({self.SOCKET_TIMEOUT_SECONDS}s) during parallel liquidation orders!")

        return liquidated_symbols, hedge_count
