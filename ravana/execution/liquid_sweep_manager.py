"""Overnight LIQUIDBEES Cash Yield and Collateral Sweep Manager (Muscle 7).

Sweeps unallocated broker margin into LIQUIDBEES (Nippon Nifty 1D Rate Liquid ETF)
at 15:22 IST to earn 6.50% annualized risk-free TREPS repo yield, neutralizing statutory
DP delivery fees and preventing cash drag. Pledging provides 90% intraday collateral margin.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SweepConfig:
    """Configuration contract for automated LIQUIDBEES repo yield sweeping."""

    liquid_ticker: str = "LIQUIDBEES"
    nav_unit_inr: float = 1000.0
    annual_yield_pct: float = 0.065  # 6.50% annualized repo rate
    min_sweep_threshold_inr: float = 25000.0  # Min cash required to trigger sweep
    operational_cash_buffer_inr: float = 10000.0  # Reserve for morning gap/margin buffers
    sweep_time: str = "15:22"
    morning_release_time: str = "09:12"


@dataclass(frozen=True)
class SweepOrderIntent:
    """Intent record for cash deployment into or release from LIQUIDBEES."""

    action: str  # "BUY_SWEEP" or "SELL_RELEASE"
    ticker: str
    units: int
    nav_price: float
    notional_inr: float
    timestamp: pd.Timestamp
    reason: str


class LiquidSweepManager:
    """Manages 15:22 IST cash sweeps and 09:12 IST collateral releases."""

    def __init__(self, config: Optional[SweepConfig] = None) -> None:
        self.config = config or SweepConfig()

    def evaluate_eod_sweep(
        self,
        current_time: pd.Timestamp,
        free_cash_inr: float,
    ) -> Optional[SweepOrderIntent]:
        """Evaluate if unallocated cash exceeds threshold and generate EOD sweep buy order."""
        time_str = current_time.strftime("%H:%M")
        if time_str < self.config.sweep_time:
            return None

        if free_cash_inr < self.config.min_sweep_threshold_inr:
            return None

        sweepable_cash = free_cash_inr - self.config.operational_cash_buffer_inr
        units = int(sweepable_cash // self.config.nav_unit_inr)
        if units <= 0:
            return None

        notional = float(units * self.config.nav_unit_inr)
        return SweepOrderIntent(
            action="BUY_SWEEP",
            ticker=self.config.liquid_ticker,
            units=units,
            nav_price=self.config.nav_unit_inr,
            notional_inr=notional,
            timestamp=current_time,
            reason=f"EOD_CASH_SWEEP_{units}_UNITS",
        )

    def evaluate_morning_release(
        self,
        current_time: pd.Timestamp,
        required_capital_inr: float,
        current_free_cash_inr: float,
        held_liquid_units: int,
    ) -> Optional[SweepOrderIntent]:
        """Evaluate if morning trading requires cash beyond current free balance."""
        if held_liquid_units <= 0:
            return None

        deficit = required_capital_inr - current_free_cash_inr
        if deficit <= 0:
            return None

        units_to_release = int(np.ceil(deficit / self.config.nav_unit_inr))
        units_to_release = min(units_to_release, held_liquid_units)
        if units_to_release <= 0:
            return None

        notional = float(units_to_release * self.config.nav_unit_inr)
        return SweepOrderIntent(
            action="SELL_RELEASE",
            ticker=self.config.liquid_ticker,
            units=units_to_release,
            nav_price=self.config.nav_unit_inr,
            notional_inr=notional,
            timestamp=current_time,
            reason=f"MORNING_MARGIN_RELEASE_{units_to_release}_UNITS",
        )

    def compute_daily_accrued_yield(
        self,
        held_units: int,
        calendar_days: int = 1,
    ) -> float:
        """Compute accrued fractional dividend interest earned over holding period."""
        if held_units <= 0:
            return 0.0
        notional = held_units * self.config.nav_unit_inr
        daily_rate = (1.0 + self.config.annual_yield_pct) ** (calendar_days / 252.0) - 1.0
        return float(notional * daily_rate)
