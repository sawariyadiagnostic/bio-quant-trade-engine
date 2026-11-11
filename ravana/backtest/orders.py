"""Order, Trade, and Position entities for the causal backtesting ledger."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import pandas as pd

from ravana.backtest.friction import FrictionBreakdown


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    """Internal order ticket emitted by an alpha strategy."""
    order_id: str
    timestamp: pd.Timestamp
    ticker: str
    side: OrderSide
    shares: int
    status: OrderStatus = OrderStatus.PENDING
    limit_price: Optional[float] = None


@dataclass(frozen=True)
class Trade:
    """Executed trade fill with exact statutory and slippage attribution."""
    trade_id: str
    timestamp: pd.Timestamp
    ticker: str
    side: OrderSide
    shares: int
    price: float
    slippage_bps: float
    friction: FrictionBreakdown


@dataclass
class FifoLot:
    """Unrealized inventory tranche for rigorous FIFO round-trip reconciliation."""
    shares: int
    price: float
    timestamp: pd.Timestamp
    friction_per_share: float


@dataclass(frozen=True)
class RoundTripTrade:
    """Matched round-trip trade with exact gross and net PnL."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    ticker: str
    shares: int
    entry_price: float
    exit_price: float
    gross_pnl: float
    net_pnl: float
    total_friction: float

    @property
    def is_win(self) -> bool:
        return self.net_pnl > 0.0


class Position:
    """Tracks inventory, realized gains, and causal holding duration for a symbol."""

    def __init__(self, ticker: str, allow_shorting: bool = False):
        self.ticker = ticker
        self.allow_shorting = allow_shorting
        self.shares = 0
        self.avg_entry_price = 0.0
        self.realized_pnl = 0.0
        self.realized_net_pnl = 0.0
        self.entry_timestamp: Optional[pd.Timestamp] = None
        self.open_lots: List[FifoLot] = []
        self.closed_trades: List[RoundTripTrade] = []

    def update_on_trade(self, trade: Trade) -> float:
        """Update position state causal to a new trade fill and return realized gross PnL."""
        realized = 0.0
        if trade.side == OrderSide.BUY:
            total_cost = (self.shares * self.avg_entry_price) + (trade.shares * trade.price)
            self.shares += trade.shares
            self.avg_entry_price = total_cost / self.shares if self.shares > 0 else 0.0
            if self.entry_timestamp is None:
                self.entry_timestamp = trade.timestamp
            fee_per_share = trade.friction.total_friction_inr / max(1, trade.shares)
            self.open_lots.append(FifoLot(trade.shares, trade.price, trade.timestamp, fee_per_share))
        elif trade.side == OrderSide.SELL:
            closing_shares = min(self.shares, trade.shares)
            if closing_shares > 0:
                realized = (trade.price - self.avg_entry_price) * closing_shares
                self.realized_pnl += realized

                # Reconcile FIFO lots for net PnL
                rem_to_close = closing_shares
                sell_fee_per_share = trade.friction.total_friction_inr / max(1, trade.shares)
                while rem_to_close > 0 and self.open_lots:
                    lot = self.open_lots[0]
                    matched_qty = min(rem_to_close, lot.shares)
                    buy_fric = lot.friction_per_share * matched_qty
                    sell_fric = sell_fee_per_share * matched_qty
                    rt_gross = (trade.price - lot.price) * matched_qty
                    rt_net = rt_gross - (buy_fric + sell_fric)
                    self.realized_net_pnl += rt_net
                    self.closed_trades.append(
                        RoundTripTrade(
                            entry_time=lot.timestamp,
                            exit_time=trade.timestamp,
                            ticker=self.ticker,
                            shares=matched_qty,
                            entry_price=lot.price,
                            exit_price=trade.price,
                            gross_pnl=round(rt_gross, 2),
                            net_pnl=round(rt_net, 2),
                            total_friction=round(buy_fric + sell_fric, 2),
                        )
                    )
                    lot.shares -= matched_qty
                    rem_to_close -= matched_qty
                    if lot.shares <= 0:
                        self.open_lots.pop(0)

                self.shares -= closing_shares
                if self.shares <= 0:
                    self.shares = 0
                    self.avg_entry_price = 0.0
                    self.entry_timestamp = None
                    self.open_lots.clear()
            elif not self.allow_shorting:
                return 0.0

        return realized

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate marked-to-market unrealized gain or loss."""
        if self.shares <= 0:
            return 0.0
        return (current_price - self.avg_entry_price) * self.shares
