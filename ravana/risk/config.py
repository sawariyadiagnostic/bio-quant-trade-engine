"""Configuration schemas for Muscle 12 Microstructure Portfolio & Risk Allocation Engine."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AllocationMethod(str, Enum):
    """Portfolio weighting algorithm."""
    HRP = "HRP"                      # Hierarchical Risk Parity (López de Prado)
    INVERSE_VOL = "INVERSE_VOL"      # Volatility Parity (1 / sigma)
    EQUAL_WEIGHT = "EQUAL_WEIGHT"    # 1 / N Naive benchmark


class DrawdownState(str, Enum):
    """Drawdown circuit breaker state."""
    NORMAL = "NORMAL"                # Normal operation (100% position sizing)
    CAUTION_HALVED = "CAUTION_HALVED" # Drawdown > 4% (50% position sizing)
    HALTED = "HALTED"                # Drawdown > 8% (Emergency 100% cash halt)


@dataclass(frozen=True)
class RiskConfig:
    """Institutional portfolio sizing, diversification, and safety boundaries."""
    initial_capital: float = 200000.0           # ₹2,00,000 Capital Viability Gate
    min_capital_gate: float = 200000.0          # Minimum account equity required
    min_trade_notional_floor: float = 17000.0   # Floor ensuring ₹17 DP fee <= 10 bps
    max_asset_allocation: float = 0.50          # Max 50% in any single ETF/Stock
    max_gross_exposure: float = 1.0             # 100% gross exposure (Cash delivery, no leverage)
    drawdown_halve_threshold: float = 0.04      # 4% peak-to-trough DD halves sizing
    drawdown_halt_threshold: float = 0.08       # 8% peak-to-trough DD halts new entries
    allocation_method: AllocationMethod = AllocationMethod.HRP
    hrp_linkage: str = "ward"                   # Hierarchical clustering linkage
    target_annual_vol: float = 0.15             # 15% annual target volatility
    max_turnover_per_rebalance: float = 0.30    # Max 30% portfolio turnover per rebalance
