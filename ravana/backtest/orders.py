"""Order, Trade, and Position entities for the causal backtesting ledger."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
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


class Position:
    """Tracks inventory, realized gains, and causal holding duration for a symbol."""

    def __init__(self, ticker: str):
        self.ticker = ticker
        self.shares = 0
        self.avg_entry_price = 0.0
        self.realized_pnl = 0.0
        self.entry_timestamp: Optional[pd.Timestamp] = None

    def update_on_trade(self, trade: Trade) -> float:
        """Update position state causal to a new trade fill and return realized gross PnL."""
        realized = 0.0
        if trade.side == OrderSide.BUY:
            total_cost = (self.shares * self.avg_entry_price) + (trade.shares * trade.price)
            self.shares += trade.shares
            self.avg_entry_price = total_cost / self.shares if self.shares > 0 else 0.0
            if self.entry_timestamp is None:
                self.entry_timestamp = trade.timestamp
        elif trade.side == OrderSide.SELL:
            closing_shares = min(self.shares, trade.shares)
            realized = (trade.price - self.avg_entry_price) * closing_shares
            self.realized_pnl += realized
            self.shares -= trade.shares
            if self.shares <= 0:
                self.shares = 0
                self.avg_entry_price = 0.0
                self.entry_timestamp = None
        return realized

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate marked-to-market unrealized gain or loss."""
        if self.shares <= 0:
            return 0.0
        return (current_price - self.avg_entry_price) * self.shares
