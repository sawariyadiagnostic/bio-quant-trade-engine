"""Configuration schemas for Muscle 11 Event-Driven Backtesting Engine."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AssetType(str, Enum):
    """Institutional asset classification dictating statutory taxation."""
    ETF = "ETF"
    EQUITY_STOCK = "EQUITY_STOCK"
    INDEX_FUTURE = "INDEX_FUTURE"


class TradingRegime(str, Enum):
    """Trading strategy operational window and holding rules."""
    OVERNIGHT = "OVERNIGHT"        # 15:20-15:25 entry -> 09:15 next day exit
    INTRADAY_MIS = "INTRADAY_MIS"  # Squared off before 15:15 IST, permits shorting
    SWING = "SWING"                # Multi-day hold governed by trailing stops or signals


class ExecutionTiming(str, Enum):
    """Causal fill timing protocol."""
    NEXT_BAR_OPEN = "NEXT_BAR_OPEN"    # Standard non-anticipating fill at t+1 Open
    AUCTION_OPEN = "AUCTION_OPEN"      # 09:15:00 IST official opening batch auction


@dataclass(frozen=True)
class BacktestConfig:
    """Institutional parameters for causal order execution and portfolio accounting."""
    initial_capital: float = 200000.0  # ₹2,00,000 capital viability gate
    asset_type: AssetType = AssetType.ETF
    regime: TradingRegime = TradingRegime.OVERNIGHT
    execution_timing: ExecutionTiming = ExecutionTiming.NEXT_BAR_OPEN
    max_volume_participation: float = 0.05  # Max 5% of bar volume to limit impact
    default_half_spread_bps: float = 1.5    # Half-spread baseline in basis points
    fixed_dp_fee_inr: float = 17.0          # Flat ₹17 CDSL debit fee per sell on delivery
    brokerage_per_order_inr: float = 0.0    # Discount broker zero delivery brokerage
    allow_shorting: bool = False            # Long-only by default for Indian cash delivery
    impact_gamma: float = 0.1               # Almgren-Chriss square-root impact coefficient
    risk_free_rate: float = 0.065           # 6.5% statutory benchmark hurdle (RBI 91-day T-Bill)
    entry_cutoff_time: str = "15:20"        # Entry cutoff for overnight strategies
    exit_cutoff_time: str = "09:15"         # Exit time for overnight strategies

    @property
    def initial_capital_inr(self) -> float:
        return self.initial_capital
