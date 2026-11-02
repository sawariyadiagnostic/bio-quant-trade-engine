"""
Domain Core: Immutable Value Objects and Domain Entities.
Pure logic, zero third-party framework dependencies.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum

@dataclass(frozen=True)
class Ticker:
    """Represents a validated NSE financial instrument."""
    symbol: str
    sector: str

    def clean_name(self) -> str:
        return self.symbol.replace(".NS", "")

@dataclass(frozen=True)
class RegimeState:
    """Represents the continuous structural manifold regime for a session."""
    cluster_id: int
    is_noise: bool
    density_score: float
    forward_expectancy: float = 0.0

    @property
    def allows_trading(self) -> bool:
        """Noise states (-1) force 100% Cash / Liquid BeES capital preservation."""
        return not self.is_noise and self.cluster_id >= 0

@dataclass(frozen=True)
class CircuitBands:
    """Statutory NSE Exchange Price Bands & Proximity Damper Metrics."""
    lower_circuit: float
    upper_circuit: float
    distance_to_lower_pct: float
    distance_to_upper_pct: float

    @property
    def is_locked(self) -> bool:
        """True if price is within 1.0% of either circuit freeze limit."""
        return self.distance_to_lower_pct < 0.01 or self.distance_to_upper_pct < 0.01

    def allocation_damper_factor(self) -> float:
        """
        Calculates an exponential dampening factor [0.0, 1.0].
        If distance < 1.0%: 0.0 (Absolute lockout).
        If distance < 2.5%: quadratic dampening.
        """
        min_dist = min(self.distance_to_lower_pct, self.distance_to_upper_pct)
        if min_dist < 0.01:
            return 0.0
        elif min_dist < 0.025:
            return float((min_dist / 0.025) ** 2)
        return 1.0

@dataclass(frozen=True)
class OrderSignal:
    """Structured simulated paper order signal."""
    ticker: str
    action: str  # 'BUY', 'SELL', 'HOLD_CASH'
    raw_price: float
    stop_loss: float
    target: float
    quantity: int
    regime_id: int
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def risk_reward_ratio(self) -> float:
        risk = abs(self.raw_price - self.stop_loss)
        reward = abs(self.target - self.raw_price)
        return round(reward / risk, 2) if risk > 0 else 0.0

class ExecutionTier(str, Enum):
    """Execution tier determining lead-lag gating and compounding rules."""
    TIER1_DAILY_ETF = "TIER1_DAILY_ETF"             # Continuous daily compounding, no foreign gate
    TIER2_ADR_LEAD_EQUITY = "TIER2_ADR_LEAD_EQUITY" # Foreign ADR green-gated single equity

class InstrumentType(str, Enum):
    """Statutory instrument class determining tax and leverage constraints."""
    ETF_CASH = "ETF_CASH"         # 0.001% STT delivery (1.2 bps total friction)
    EQUITY_CASH = "EQUITY_CASH"   # 0.10% buy + 0.10% sell STT (25 bps total friction)
    MTF_LEVERAGED = "MTF_LEVERAGED" # 2x-4x broker margin + ~12% annual interest
    FUTURES = "FUTURES"           # 0.02% sell STT only (3 bps friction, 5x-6x leverage)

@dataclass(frozen=True)
class ADRLeadSignal:
    """Overnight predictive telemetry from foreign listing / ADR."""
    foreign_ticker: str
    domestic_ticker: str
    foreign_return_pct: float
    is_bullish: bool
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass(frozen=True)
class OvernightExecutionRecord:
    """Audit record for overnight trade execution and trailing stop fill."""
    session_date: str
    ticker: str
    tier: ExecutionTier
    instrument: InstrumentType
    leverage: float
    entry_price: float
    open_price: float
    high_price: float
    low_price: float
    exit_price: float
    exit_reason: str
    gross_return_pct: float
    net_return_pct: float
    statutory_friction: float
    starting_balance: float
    ending_balance: float
