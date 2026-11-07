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
    ETF_CASH = "ETF_CASH"         # 0.001% STT delivery (1.2 bps default baseline)
    EQUITY_CASH = "EQUITY_CASH"   # 0.10% buy + 0.10% sell STT (25 bps total friction)
    MTF_LEVERAGED = "MTF_LEVERAGED" # 2x-4x broker margin + ~12% annual interest
    FUTURES = "FUTURES"           # 0.02% sell STT only (3 bps friction baseline)
    INTRADAY_EQUITY = "INTRADAY_EQUITY" # 0.025% sell STT + turnover (~3.5 bps)


@dataclass(frozen=True)
class TaxSchedule:
    """
    Configurable dynamic statutory tax and friction schedule (in basis points).
    Enables custom and dynamic parameterization across Indian Equities, ETFs, and Futures.
    1 bps = 0.0001 (0.01%).
    """
    etf_cash_bps: float = 1.2          # 0.00012 (0.012%) for ETF delivery
    futures_bps: float = 3.0           # 0.00030 (0.030%) for Single-Stock Futures
    equity_cash_bps: float = 25.0      # 0.00250 (0.250%) for Equity Delivery
    intraday_bps: float = 3.5          # 0.00035 (0.035%) for Intraday cash
    custom_slippage_bps: float = 0.0   # Optional market impact / slippage buffer

    def get_friction_rate(self, instrument: InstrumentType | str) -> float:
        """Returns the decimal friction rate (e.g., 0.00012 for 1.2 bps)."""
        inst = instrument.value if isinstance(instrument, InstrumentType) else str(instrument).upper()
        if inst in ("ETF_CASH", "ETF"):
            base = self.etf_cash_bps
        elif inst in ("FUTURES", "FUT"):
            base = self.futures_bps
        elif inst in ("INTRADAY_EQUITY", "INTRADAY"):
            base = self.intraday_bps
        else:
            base = self.equity_cash_bps
        return float((base + self.custom_slippage_bps) / 10000.0)

    def calculate_tax(self, trade_value: float, instrument: InstrumentType | str) -> float:
        """Calculates total friction in currency units."""
        return float(trade_value * self.get_friction_rate(instrument))

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
