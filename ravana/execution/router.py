"""Muscle 19: Live Broker Order Routing Bridge.

Chains Muscle 17 (VolatilitySizedAllocation), Muscle 18 (CircuitBreakerKillSwitch),
Phase 5 (ValidationCertificate), and Muscle 11 (Order, Trade, IndianFrictionModel,
MicrostructureSlippageModel) into an idempotent execution state machine.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from loguru import logger
import numpy as np
import pandas as pd

from ravana.backtest.config import AssetType, TradingRegime
from ravana.backtest.friction import IndianFrictionModel
from ravana.backtest.orders import Order, OrderSide, OrderStatus, Trade
from ravana.backtest.slippage import MicrostructureSlippageModel
from ravana.risk.circuit_breaker import CircuitBreakerKillSwitch, CircuitBreakerStatus
from ravana.risk.sizing import VolatilitySizedAllocation
from ravana.validation.config import ValidationCertificate


@dataclass(frozen=True)
class ExecutionAuditRecord:
    """Reconciles actual broker fill slippage against Muscle 11 predicted slippage."""

    order_id: str
    ticker: str
    side: OrderSide
    filled_shares: int
    expected_price: float
    executed_price: float
    realized_slippage_bps: float
    modeled_slippage_bps: float
    slippage_anomaly_bps: float


@dataclass(frozen=True)
class RoutingReceipt:
    """Output contract of Muscle 19 consumed by Muscle 20 Observability Monitor."""

    submitted_orders: List[Order]
    executed_trades: List[Trade]
    rejected_orders: List[Order]
    audit_records: List[ExecutionAuditRecord]
    breaker_status: CircuitBreakerStatus
    rejection_reasons: Dict[str, str] = field(default_factory=dict)


class BrokerGateway(ABC):
    """Abstract broker adapter interface (implemented by Paper, Shoonya, or Zerodha)."""

    @abstractmethod
    def place_order(self, order: Order, reference_price: float, bar_volume: float = 1e5) -> Optional[Trade]:
        """Transmit order to exchange and return reconciled Trade fill."""
        pass


class PaperBrokerGateway(BrokerGateway):
    """Realistic microstructure paper-trading broker powered by Muscle 11 models."""

    def __init__(
        self,
        asset_type: AssetType = AssetType.ETF,
        regime: TradingRegime = TradingRegime.OVERNIGHT,
    ) -> None:
        self.asset_type = asset_type
        self.regime = regime
        self.slippage_model = MicrostructureSlippageModel()

    def place_order(self, order: Order, reference_price: float, bar_volume: float = 1e5) -> Optional[Trade]:
        if order.shares <= 0 or reference_price <= 0 or not np.isfinite(reference_price):
            order.status = OrderStatus.REJECTED
            return None

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
            asset_type=self.asset_type,
            regime=self.regime,
        )
        order.status = OrderStatus.FILLED if fill.filled_shares == order.shares else OrderStatus.PARTIALLY_FILLED
        return Trade(
            trade_id=f"PAPER_{order.order_id}",
            timestamp=order.timestamp,
            ticker=order.ticker,
            side=order.side,
            shares=fill.filled_shares,
            price=fill.executed_price,
            slippage_bps=fill.slippage_bps,
            friction=friction,
        )


EXIT_TRANCHE_1 = "EXIT_TRANCHE_1"


class BrokerOrderRouter:
    """Routes sized allocations to the broker while enforcing Muscle 18 and Phase 5 gates."""

    def __init__(
        self,
        gateway: Optional[BrokerGateway] = None,
        kill_switch: Optional[CircuitBreakerKillSwitch] = None,
        live_mode: bool = False,
        allow_shorting: bool = False,
    ) -> None:
        self.gateway = gateway or PaperBrokerGateway()
        self.kill_switch = kill_switch or CircuitBreakerKillSwitch()
        self.live_mode = live_mode
        self.allow_shorting = allow_shorting
        self.slippage_estimator = MicrostructureSlippageModel()
        self.continuation_holds_count: int = 0

    @staticmethod
    def _is_morning_session_open(ts: pd.Timestamp) -> bool:
        """Return True if timestamp is within morning session open (09:15 to 09:25 IST)."""
        if ts is None:
            return False
        local_ts = ts.tz_convert("Asia/Kolkata") if ts.tzinfo is not None else ts
        t_str = local_ts.strftime("%H:%M")
        return "09:15" <= t_str <= "09:25"

    def _resolve_signal(
        self,
        sym: str,
        continuation_signals: Optional[Union[Dict[str, float], float]],
        continuation_signal: Optional[float],
        kwargs: Dict[str, Any],
    ) -> float:
        """Extract symbol-specific continuation signal from multiple input formats."""
        if isinstance(continuation_signals, dict):
            return float(continuation_signals.get(sym, 0.0))
        if isinstance(continuation_signals, (int, float)):
            return float(continuation_signals)
        if continuation_signal is not None:
            return float(continuation_signal)
        if "signal" in kwargs:
            return float(kwargs["signal"])
        if "signals" in kwargs:
            sigs = kwargs["signals"]
            return float(sigs.get(sym, 0.0) if isinstance(sigs, dict) else sigs)
        return 0.0

    def route_exit_tranche(
        self,
        tranche: str,
        current_holdings: Dict[str, int],
        current_prices: Dict[str, float],
        breaker_status: CircuitBreakerStatus,
        certificate: Optional[ValidationCertificate] = None,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> RoutingReceipt:
        """Route partial multi-tranche exit orders (e.g. 50% for EXIT_TRANCHE_1)."""
        ts = timestamp or pd.Timestamp.now(tz="Asia/Kolkata")
        orders: List[Order] = []
        if tranche == EXIT_TRANCHE_1:
            for sym, curr_qty in current_holdings.items():
                if curr_qty > 0:
                    exit_shares = curr_qty // 2 if curr_qty > 1 else curr_qty
                    if exit_shares > 0:
                        orders.append(
                            Order(f"SELL_T1_{sym}_{int(ts.timestamp())}", ts, sym, OrderSide.SELL, exit_shares)
                        )
        return self.route_orders(orders, current_prices, breaker_status, certificate)

    def route_target_allocation(
        self,
        sized_allocation: Optional[Union[VolatilitySizedAllocation, str]] = None,
        current_holdings: Optional[Dict[str, int]] = None,
        current_prices: Optional[Dict[str, float]] = None,
        breaker_status: Optional[CircuitBreakerStatus] = None,
        certificate: Optional[ValidationCertificate] = None,
        timestamp: Optional[pd.Timestamp] = None,
        hold_on_strong_continuation: bool = True,
        continuation_threshold: float = 0.50,
        continuation_signals: Optional[Union[Dict[str, float], float]] = None,
        continuation_signal: Optional[float] = None,
        exit_tranche: Optional[str] = None,
        **kwargs: Any,
    ) -> RoutingReceipt:
        """Diff target shares against current holdings and route delta orders safely."""
        holdings = current_holdings or {}
        prices = current_prices or {}
        ts = timestamp or pd.Timestamp.now(tz="Asia/Kolkata")

        # Two-tranche exit handling
        tranche = exit_tranche or (sized_allocation if isinstance(sized_allocation, str) else None)
        if tranche == EXIT_TRANCHE_1:
            status = breaker_status or self.kill_switch.evaluate(0.0)
            return self.route_exit_tranche(EXIT_TRANCHE_1, holdings, prices, status, certificate, ts)

        orders: List[Order] = []
        target_shares = sized_allocation.sized_shares if isinstance(sized_allocation, VolatilitySizedAllocation) else {}
        all_syms = sorted(set(holdings.keys()) | set(target_shares.keys()))

        for sym in all_syms:
            curr_qty = holdings.get(sym, 0)
            targ_qty = target_shares.get(sym, 0)
            delta = targ_qty - curr_qty
            if delta > 0:
                orders.append(Order(f"BUY_{sym}_{int(ts.timestamp())}", ts, sym, OrderSide.BUY, delta))
            elif delta < 0:
                sell_shares = min(curr_qty, abs(delta)) if not self.allow_shorting else abs(delta)
                if sell_shares <= 0:
                    continue

                sig = self._resolve_signal(sym, continuation_signals, continuation_signal, kwargs)
                if (
                    hold_on_strong_continuation
                    and self._is_morning_session_open(ts)
                    and sig >= continuation_threshold
                ):
                    self.continuation_holds_count += 1
                    if hasattr(self.gateway, "continuation_holds_count"):
                        self.gateway.continuation_holds_count += 1
                    logger.info(
                        f"Continuation hold active for {sym}: signal {sig:.2f} >= {continuation_threshold:.2f} "
                        f"at {ts.strftime('%H:%M:%S')}. Skipping morning sell order and preserving ₹17 DP fee."
                    )
                    continue

                orders.append(Order(f"SELL_{sym}_{int(ts.timestamp())}", ts, sym, OrderSide.SELL, sell_shares))

        status = breaker_status or self.kill_switch.evaluate(0.0)
        return self.route_orders(orders, prices, status, certificate)

    def route_orders(
        self,
        orders: List[Order],
        current_prices: Dict[str, float],
        breaker_status: CircuitBreakerStatus,
        certificate: Optional[ValidationCertificate] = None,
    ) -> RoutingReceipt:
        """Execute order list with pre-trade Kill Switch and ValidationCertificate verification."""
        submitted: List[Order] = []
        executed: List[Trade] = []
        rejected: List[Order] = []
        audits: List[ExecutionAuditRecord] = []
        reasons: Dict[str, str] = {}

        live_certified = (certificate is not None and certificate.is_certified_for_live) if self.live_mode else True

        for ord in orders:
            # Risk-increasing BUY orders require all circuit breakers green AND live certification
            if ord.side == OrderSide.BUY:
                if not breaker_status.is_trading_allowed or self.kill_switch.is_killed:
                    ord.status = OrderStatus.REJECTED
                    rejected.append(ord)
                    reasons[ord.order_id] = f"Blocked by Circuit Breaker / Kill Switch: {breaker_status.reason}"
                    continue
                if not live_certified:
                    ord.status = OrderStatus.REJECTED
                    rejected.append(ord)
                    reasons[ord.order_id] = "Strategy not certified for live capital by Phase 5 Validation"
                    continue

            ref_px = current_prices.get(ord.ticker, 0.0)
            if ref_px <= 0 or not np.isfinite(ref_px):
                ord.status = OrderStatus.REJECTED
                rejected.append(ord)
                reasons[ord.order_id] = f"Invalid or non-positive reference price: {ref_px}"
                continue

            trade = self.gateway.place_order(ord, reference_price=ref_px)
            if trade is None:
                ord.status = OrderStatus.REJECTED
                rejected.append(ord)
                reasons[ord.order_id] = "Broker gateway rejected order (illiquid or circuit locked)"
                continue

            submitted.append(ord)
            executed.append(trade)
            audits.append(self._audit_fill(ord, trade, ref_px))

        return RoutingReceipt(
            submitted_orders=submitted,
            executed_trades=executed,
            rejected_orders=rejected,
            audit_records=audits,
            breaker_status=breaker_status,
            rejection_reasons=reasons,
        )

    def _audit_fill(self, order: Order, trade: Trade, ref_price: float) -> ExecutionAuditRecord:
        """Compare realized fill slippage against Muscle 11 modeled expectation."""
        expected_fill = self.slippage_estimator.simulate_fill(
            side=order.side.value,
            desired_shares=trade.shares,
            nominal_price=ref_price,
            bar_volume=100000.0,
            bar_high=ref_price * 1.02,
            bar_low=ref_price * 0.98,
        )
        sign = 1.0 if order.side == OrderSide.BUY else -1.0
        realized_bps = sign * ((trade.price - ref_price) / max(1e-4, ref_price)) * 10000.0
        anomaly_bps = realized_bps - expected_fill.slippage_bps

        return ExecutionAuditRecord(
            order_id=order.order_id,
            ticker=order.ticker,
            side=order.side,
            filled_shares=trade.shares,
            expected_price=round(ref_price, 4),
            executed_price=round(trade.price, 4),
            realized_slippage_bps=round(realized_bps, 2),
            modeled_slippage_bps=round(expected_fill.slippage_bps, 2),
            slippage_anomaly_bps=round(anomaly_bps, 2),
        )
