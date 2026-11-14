"""Finvasia Shoonya Execution Adapter for RAVANA Quantitative Engine.

Subclasses BrokerGateway to route orders to Finvasia Shoonya exchange
or realistic microstructure paper simulation, fully integrated with Muscle 11
(Friction & Slippage), Muscle 18 (Circuit Breaker Interlock), and Headless TOTP 2FA.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import threading
import time
from typing import Any, Dict, List, Optional
from loguru import logger
import numpy as np
import pandas as pd
import requests

from ravana.backtest.config import AssetType, TradingRegime
from ravana.backtest.friction import IndianFrictionModel
from ravana.backtest.orders import Order, OrderSide, OrderStatus, Trade
from ravana.backtest.slippage import MicrostructureSlippageModel
from ravana.execution.router import BrokerGateway
from ravana.risk.circuit_breaker import CircuitBreakerStatus
from src.execution.shoonya_gateway import ShoonyaCredentials, ShoonyaGateway
from src.execution.shoonya_watchdog import ShoonyaSessionWatchdog


@dataclass
class ShoonyaGatewayConfig:
    """Configuration container for Shoonya Finvasia broker gateway."""

    credentials: Optional[ShoonyaCredentials] = None
    paper_trading: bool = True
    exchange: str = "NSE"
    default_product_type: str = "C"  # "C" for CNC (Cash & Carry), "M" for MIS (Intraday)
    order_timeout_seconds: float = 3.0
    max_requests_per_second: float = 8.0
    inter_order_delay_seconds: float = 0.125
    asset_type: AssetType = AssetType.ETF
    regime: TradingRegime = TradingRegime.OVERNIGHT
    statutory_dp_fee_inr: float = 15.93
    brokerage_per_order_inr: float = 0.0
    audit_file_path: Optional[str] = None


class TokenBucketRateLimiter:
    """Thread-safe Token Bucket Rate Limiter to guard broker API throughput."""

    def __init__(self, max_rate: float = 8.0, capacity: float = 8.0) -> None:
        self.max_rate = float(max_rate)
        self.capacity = float(capacity)
        self._tokens = float(capacity)
        self._last_time = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens_needed: float = 1.0) -> float:
        """Acquire tokens, blocking if insufficient tokens available. Returns wait duration."""
        wait_time = 0.0
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_time
            self._tokens = min(self.capacity, self._tokens + (elapsed * self.max_rate))
            if self._tokens >= tokens_needed:
                self._tokens -= tokens_needed
                self._last_time = now
                return 0.0
            wait_time = (tokens_needed - self._tokens) / self.max_rate
            self._tokens = 0.0
            self._last_time = now + wait_time

        if wait_time > 0:
            time.sleep(wait_time)
        return wait_time


class ShoonyaBrokerGateway(BrokerGateway):
    """Institutional adapter connecting RAVANA execution engine to Finvasia Shoonya."""

    def __init__(self, config: Optional[ShoonyaGatewayConfig] = None) -> None:
        self.config = config or ShoonyaGatewayConfig()
        self.slippage_model = MicrostructureSlippageModel()
        self.rate_limiter = TokenBucketRateLimiter(
            max_rate=self.config.max_requests_per_second,
            capacity=self.config.max_requests_per_second,
        )
        self.clordid_ledger: Dict[str, Trade] = {}
        self._auth_lock = threading.Lock()

        creds = self.config.credentials or ShoonyaCredentials.from_env()
        self.gateway = ShoonyaGateway(
            credentials=creds,
            paper_trading=self.config.paper_trading,
            audit_file=self.config.audit_file_path,
        )
        totp_sec = creds.totp_key if creds else ""
        self.watchdog = ShoonyaSessionWatchdog(
            gateway=self.gateway,
            totp_secret=totp_sec,
        )

    @staticmethod
    def wait_for_totp_epoch_buffer(min_buffer_seconds: float = 4.0) -> float:
        """Pause execution if remaining seconds in 30s TOTP window < min_buffer_seconds."""
        now = time.time()
        remaining = 30.0 - (now % 30.0)
        if remaining < min_buffer_seconds:
            wait_dur = remaining + 0.15
            logger.info(
                f"TOTP epoch edge detected ({remaining:.2f}s remaining). "
                f"Pausing {wait_dur:.2f}s for next 30s epoch."
            )
            time.sleep(wait_dur)
            return wait_dur
        return 0.0

    def _authenticate_live(self) -> bool:
        """Atomically authenticate with Shoonya using 4-second TOTP epoch buffer guard."""
        with self._auth_lock:
            self.wait_for_totp_epoch_buffer(4.0)
            if self.watchdog and self.watchdog.totp_secret:
                return bool(self.watchdog.refresh_session())
            return bool(self.gateway.authenticate())

    def place_order(
        self, order: Order, reference_price: float, bar_volume: float = 1e5
    ) -> Optional[Trade]:
        """Transmit order to Finvasia Shoonya or simulate fill in Paper Mode."""
        if order.order_id in self.clordid_ledger:
            logger.info(f"ShoonyaBrokerGateway: Idempotent replay for order {order.order_id}")
            return self.clordid_ledger[order.order_id]

        if order.shares <= 0 or reference_price <= 0 or not np.isfinite(reference_price):
            order.status = OrderStatus.REJECTED
            return None

        self.rate_limiter.acquire(1.0)
        if self.config.paper_trading:
            return self._place_paper_order(order, reference_price, bar_volume)
        return self._place_live_order(order, reference_price)

    def _place_paper_order(
        self, order: Order, reference_price: float, bar_volume: float
    ) -> Optional[Trade]:
        """Execute realistic microstructure paper order simulation."""
        fill = self.slippage_model.simulate_fill(
            side=order.side.value,
            desired_shares=order.shares,
            nominal_price=reference_price,
            bar_volume=bar_volume,
            bar_high=reference_price * 1.01,
            bar_low=reference_price * 0.99,
        )
        if fill.filled_shares <= 0 or fill.is_circuit_locked:
            order.status = OrderStatus.REJECTED
            return None

        friction = IndianFrictionModel.calculate_friction(
            side=order.side.value,
            price=fill.executed_price,
            shares=fill.filled_shares,
            asset_type=self.config.asset_type,
            regime=self.config.regime,
            fixed_dp_fee_inr=self.config.statutory_dp_fee_inr,
            brokerage_per_order_inr=self.config.brokerage_per_order_inr,
        )
        order.status = (
            OrderStatus.FILLED if fill.filled_shares == order.shares else OrderStatus.PARTIALLY_FILLED
        )
        trade = Trade(
            trade_id=f"SHOONYA_PAPER_{order.order_id}",
            timestamp=order.timestamp,
            ticker=order.ticker,
            side=order.side,
            shares=fill.filled_shares,
            price=fill.executed_price,
            slippage_bps=fill.slippage_bps,
            friction=friction,
        )
        self.clordid_ledger[order.order_id] = trade
        self._update_paper_positions(order, fill.filled_shares, fill.executed_price)
        self._record_audit_log(order, trade, mode="PAPER")
        return trade

    def _place_live_order(self, order: Order, reference_price: float) -> Optional[Trade]:
        """Execute live order on Finvasia exchange and poll for reconciled fill."""
        tsym = ShoonyaGateway.format_trading_symbol(order.ticker)
        action = "BUY" if order.side == OrderSide.BUY else "SELL"
        order_type = "LMT" if (order.limit_price is not None and order.limit_price > 0) else "MKT"
        price = float(order.limit_price) if order_type == "LMT" else float(reference_price)

        send_res = self._send_with_retry(
            tsym=tsym,
            quantity=order.shares,
            price=price,
            action=action,
            order_type=order_type,
            client_order_id=order.order_id,
        )
        if send_res is None:
            order.status = OrderStatus.REJECTED
            return None

        norenordno = str(send_res.get("norenordno") or send_res.get("result", ""))
        fill_info = self._poll_order_fill(
            norenordno, timeout_seconds=self.config.order_timeout_seconds
        )
        if fill_info is None or fill_info.get("status") != "COMPLETE":
            order.status = OrderStatus.REJECTED
            return None

        filled_shares = int(fill_info.get("filled_shares", 0))
        exec_price = float(fill_info.get("executed_price", reference_price))
        if filled_shares <= 0 or exec_price <= 0:
            order.status = OrderStatus.REJECTED
            return None

        return self._reconcile_and_record_trade(
            order, norenordno, filled_shares, exec_price, reference_price
        )

    def _send_with_retry(
        self,
        tsym: str,
        quantity: int,
        price: float,
        action: str,
        order_type: str,
        client_order_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Place live order with session expiry detection and atomic re-auth retry."""
        self.rate_limiter.acquire(1.0)
        if self.config.inter_order_delay_seconds > 0:
            time.sleep(self.config.inter_order_delay_seconds)

        if not self.gateway.susertoken:
            self._authenticate_live()

        res = self.gateway.place_order(
            symbol=tsym,
            quantity=quantity,
            price=price,
            action=action,
            product=self.config.default_product_type,
            order_type=order_type,
            is_etf=(self.config.asset_type == AssetType.ETF),
            client_order_id=client_order_id,
        )

        if isinstance(res, dict) and self.gateway._is_session_expired(res):
            logger.warning("Shoonya session expired during order placement. Re-authenticating...")
            if self._authenticate_live():
                self.rate_limiter.acquire(1.0)
                res = self.gateway.place_order(
                    symbol=tsym,
                    quantity=quantity,
                    price=price,
                    action=action,
                    product=self.config.default_product_type,
                    order_type=order_type,
                    is_etf=(self.config.asset_type == AssetType.ETF),
                    client_order_id=client_order_id,
                )

        if not isinstance(res, dict) or res.get("stat") != "Ok":
            logger.error(f"Shoonya live order placement failed: {res}")
            return None
        return res

    def _poll_order_fill(
        self, norenordno: str, timeout_seconds: float
    ) -> Optional[Dict[str, Any]]:
        """Poll order history with exponential backoff until filled or timeout."""
        start_time = time.monotonic()
        backoff = 0.05
        max_backoff = 0.5

        while time.monotonic() - start_time < timeout_seconds:
            self.rate_limiter.acquire(1.0)
            history = self._query_order_history(norenordno)
            status_info = self._parse_order_history(history)
            if status_info is not None:
                status = status_info.get("status")
                if status == "COMPLETE":
                    return status_info
                if status in ("REJECTED", "CANCELED", "CANCELLED"):
                    logger.warning(f"Shoonya order {norenordno} ended with status: {status}")
                    return status_info

            time.sleep(backoff)
            backoff = min(max_backoff, backoff * 1.5)

        logger.warning(f"Shoonya order {norenordno} polling timed out after {timeout_seconds}s")
        return None

    def _query_order_history(self, norenordno: str) -> Optional[List[Dict[str, Any]]]:
        """Query order history from Shoonya API or REST fallback."""
        try:
            if self.gateway.api and self.gateway.susertoken:
                res = self.gateway.api.single_order_history(orderno=norenordno)
                if isinstance(res, list):
                    return res
                orders = self.gateway.api.get_order_book()
                if isinstance(orders, list):
                    matching = [
                        o for o in orders
                        if (o.get("norenordno") == norenordno or o.get("orderno") == norenordno)
                    ]
                    if matching:
                        return matching

            if self.gateway.susertoken and self.gateway.creds.user_id:
                j_data = {"uid": self.gateway.creds.user_id, "norenordno": norenordno}
                headers = {
                    "Authorization": f"Bearer {self.gateway.susertoken}",
                    "Content-Type": "application/json",
                }
                resp = requests.post(
                    f"{self.gateway.BASE_URL}/SingleOrderHist",
                    data="jData=" + json.dumps(j_data),
                    headers=headers,
                    timeout=5,
                )
                data = resp.json()
                if isinstance(data, list):
                    return data
        except Exception as exc:
            logger.warning(f"Shoonya error querying order history for {norenordno}: {exc}")
        return None

    def _parse_order_history(
        self, history: Optional[List[Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        """Parse Shoonya order history list into status, filled shares, and executed price."""
        if not history or not isinstance(history, list):
            return None

        for record in history:
            status = str(record.get("status", "")).upper()
            if status == "COMPLETE":
                flqty = int(float(record.get("flqty", record.get("qty", 0))))
                flprc = float(record.get("flprc", record.get("avgprc", record.get("prc", 0.0))))
                return {"status": "COMPLETE", "filled_shares": flqty, "executed_price": flprc}
            if status in ("REJECTED", "CANCELED", "CANCELLED"):
                return {
                    "status": status,
                    "filled_shares": 0,
                    "executed_price": 0.0,
                    "reason": record.get("rejreason", record.get("emsg", "Rejected")),
                }

        latest = history[0] if history else {}
        status = str(latest.get("status", "PENDING")).upper()
        flqty = int(float(latest.get("flqty", 0)))
        flprc = float(latest.get("flprc", latest.get("avgprc", 0.0)))
        return {"status": status, "filled_shares": flqty, "executed_price": flprc}

    def _reconcile_and_record_trade(
        self,
        order: Order,
        order_number: str,
        filled_shares: int,
        executed_price: float,
        reference_price: float,
    ) -> Trade:
        """Calculate slippage, statutory friction, update order status, and ledger."""
        sign = 1.0 if order.side == OrderSide.BUY else -1.0
        slippage_bps = sign * ((executed_price - reference_price) / max(1e-4, reference_price)) * 10000.0

        friction = IndianFrictionModel.calculate_friction(
            side=order.side.value,
            price=executed_price,
            shares=filled_shares,
            asset_type=self.config.asset_type,
            regime=self.config.regime,
            fixed_dp_fee_inr=self.config.statutory_dp_fee_inr,
            brokerage_per_order_inr=self.config.brokerage_per_order_inr,
        )
        order.status = (
            OrderStatus.FILLED if filled_shares == order.shares else OrderStatus.PARTIALLY_FILLED
        )
        trade = Trade(
            trade_id=f"SHOONYA_{order_number}",
            timestamp=order.timestamp,
            ticker=order.ticker,
            side=order.side,
            shares=filled_shares,
            price=round(executed_price, 4),
            slippage_bps=round(slippage_bps, 2),
            friction=friction,
        )
        self.clordid_ledger[order.order_id] = trade
        self._record_audit_log(order, trade, mode="LIVE")
        return trade

    def _update_paper_positions(
        self, order: Order, filled_shares: int, executed_price: float
    ) -> None:
        """Synchronize simulated positions on underlying gateway for liquidation readiness."""
        tsym = ShoonyaGateway.format_trading_symbol(order.ticker)
        pos = self.gateway.paper_positions.get(tsym, {"quantity": 0, "avg_price": 0.0})
        if order.side == OrderSide.BUY:
            new_qty = pos["quantity"] + filled_shares
            new_avg = (
                ((pos["quantity"] * pos["avg_price"]) + (filled_shares * executed_price)) / new_qty
                if new_qty > 0 else 0.0
            )
            self.gateway.paper_positions[tsym] = {"quantity": new_qty, "avg_price": round(new_avg, 2)}
        elif order.side == OrderSide.SELL:
            new_qty = max(0, pos["quantity"] - filled_shares)
            if new_qty == 0:
                self.gateway.paper_positions.pop(tsym, None)
            else:
                self.gateway.paper_positions[tsym]["quantity"] = new_qty

    def _record_audit_log(self, order: Order, trade: Trade, mode: str) -> None:
        """Record execution record into persistent audit file."""
        if not self.config.audit_file_path and not getattr(self.gateway, "audit_file", None):
            return
        record = {
            "order_id": order.order_id,
            "ticker": order.ticker,
            "side": order.side.value,
            "shares": trade.shares,
            "price": trade.price,
            "slippage_bps": trade.slippage_bps,
            "total_friction_inr": trade.friction.total_friction_inr,
            "mode": mode,
            "timestamp": (
                trade.timestamp.isoformat()
                if hasattr(trade.timestamp, "isoformat") else str(trade.timestamp)
            ),
        }
        self.gateway._record_audit(record)

    def on_circuit_breaker_tripped(
        self,
        status: CircuitBreakerStatus,
        cancel_pending: bool = True,
        liquidate_all: bool = False,
    ) -> Dict[str, Any]:
        """Interlock with Muscle 18 circuit breaker to freeze routing, cancel orders, or liquidate."""
        results: Dict[str, Any] = {
            "status": "CIRCUIT_BREAKER_HANDLED",
            "reason": status.reason,
            "is_trading_allowed": status.is_trading_allowed,
            "kill_switch_engaged": status.kill_switch_engaged,
            "drawdown_state": str(status.drawdown_state),
            "trailing_drawdown_pct": status.trailing_drawdown_pct,
            "cancel_pending": cancel_pending,
            "liquidate_all": liquidate_all,
            "cancel_result": None,
            "liquidation_result": None,
        }
        if cancel_pending:
            results["cancel_result"] = self.cancel_all_orders()
        if liquidate_all:
            results["liquidation_result"] = self.liquidate_positions()
        return results

    def cancel_all_orders(self) -> Dict[str, Any]:
        """Emergency order cancellation for all pending orders."""
        self.rate_limiter.acquire(1.0)
        return self.gateway.cancel_all_orders()

    def liquidate_positions(self) -> Dict[str, Any]:
        """Emergency panic liquidation of all active positions to 100% cash."""
        self.rate_limiter.acquire(1.0)
        return self.gateway.liquidate_positions()

    def get_limits(self) -> Dict[str, Any]:
        """Retrieve account balance and available margin limits from broker."""
        self.rate_limiter.acquire(1.0)
        return self.gateway.get_limits()

    def probe_and_heal_session(self) -> Dict[str, Any]:
        """Execute session health probe and automatically refresh expired session."""
        return self.watchdog.probe_and_heal()
