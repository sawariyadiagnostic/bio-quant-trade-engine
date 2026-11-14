"""Muscle 19: Live Broker Order Routing Bridge.

Chains Muscle 17 (VolatilitySizedAllocation), Muscle 18 (CircuitBreakerKillSwitch),
Phase 5 (ValidationCertificate), and Muscle 11 (Order, Trade, IndianFrictionModel,
MicrostructureSlippageModel) into an idempotent execution state machine.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional
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

    def route_target_allocation(
        self,
        sized_allocation: VolatilitySizedAllocation,
        current_holdings: Dict[str, int],
        current_prices: Dict[str, float],
        breaker_status: CircuitBreakerStatus,
        certificate: Optional[ValidationCertificate] = None,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> RoutingReceipt:
        """Diff target shares against current holdings and route delta orders safely."""
        ts = timestamp or pd.Timestamp.now(tz="Asia/Kolkata")
        orders: List[Order] = []
        all_syms = sorted(set(current_holdings.keys()) | set(sized_allocation.sized_shares.keys()))

        for sym in all_syms:
            curr_qty = current_holdings.get(sym, 0)
            targ_qty = sized_allocation.sized_shares.get(sym, 0)
            delta = targ_qty - curr_qty
            if delta > 0:
                orders.append(Order(f"BUY_{sym}_{int(ts.timestamp())}", ts, sym, OrderSide.BUY, delta))
            elif delta < 0:
                # Strictly prevent unauthorized naked shorting for cash delivery equities
                sell_shares = min(curr_qty, abs(delta)) if not self.allow_shorting else abs(delta)
                if sell_shares > 0:
                    orders.append(Order(f"SELL_{sym}_{int(ts.timestamp())}", ts, sym, OrderSide.SELL, sell_shares))

        return self.route_orders(orders, current_prices, breaker_status, certificate)

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
