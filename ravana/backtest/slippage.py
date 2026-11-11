"""Microstructure slippage, volume participation cap, and circuit breaker models."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionFill:
    """Microstructure result of attempting to fill an order on a market bar."""
    filled_shares: int
    executed_price: float
    slippage_bps: float
    is_circuit_locked: bool
    is_volume_capped: bool


class MicrostructureSlippageModel:
    """Simulates realistic Indian order book crossing, spread drag, and volume impact."""

    def __init__(
        self,
        max_volume_participation: float = 0.05,
        default_half_spread_bps: float = 1.5,
        impact_gamma: float = 0.1,
    ) -> None:
        self.max_volume_participation = max_volume_participation
        self.default_half_spread_bps = default_half_spread_bps
        self.impact_gamma = impact_gamma

    def simulate_fill(
        self,
        side: str,
        desired_shares: int,
        nominal_price: float,
        bar_volume: float,
        bar_high: float,
        bar_low: float,
        half_spread_bps: float | None = None,
    ) -> ExecutionFill:
        """Simulate realistic fill price, impact, and liquidity bounds for a single bar."""
        if desired_shares <= 0 or nominal_price <= 0:
            return ExecutionFill(0, 0.0, 0.0, False, False)

        # 1. Circuit Breaker / Freeze Detection (Zero liquidity)
        if bar_high > 0 and bar_high == bar_low and (bar_volume is None or bar_volume <= 0 or math.isnan(bar_volume)):
            return ExecutionFill(0, 0.0, 0.0, is_circuit_locked=True, is_volume_capped=False)

        # 2. Volume Participation Cap
        if bar_volume is None or math.isnan(bar_volume) or bar_volume <= 0:
            return ExecutionFill(0, 0.0, 0.0, is_circuit_locked=False, is_volume_capped=True)

        max_fillable = max(0, int(bar_volume * self.max_volume_participation))
        filled_shares = min(desired_shares, max_fillable)
        is_volume_capped = filled_shares < desired_shares

        if filled_shares <= 0:
            return ExecutionFill(0, 0.0, 0.0, is_circuit_locked=False, is_volume_capped=True)

        # 3. Spread and Almgren-Chriss Market Impact
        h_spread = half_spread_bps if half_spread_bps is not None else self.default_half_spread_bps
        participation_ratio = filled_shares / max(1.0, bar_volume)
        impact_bps = self.impact_gamma * math.sqrt(participation_ratio) * 100.0
        total_slippage_bps = h_spread + impact_bps

        # 4. Slipped Execution Price (clamped to physical bar high/low)
        if side.upper() == "BUY":
            raw_exec_price = nominal_price * (1.0 + total_slippage_bps / 10000.0)
        else:
            raw_exec_price = nominal_price * (1.0 - total_slippage_bps / 10000.0)

        if bar_high > 0 and bar_low > 0 and bar_high >= bar_low:
            exec_price = max(bar_low, min(bar_high, raw_exec_price))
        else:
            exec_price = raw_exec_price

        return ExecutionFill(
            filled_shares=filled_shares,
            executed_price=round(exec_price, 4),
            slippage_bps=round(total_slippage_bps, 2),
            is_circuit_locked=False,
            is_volume_capped=is_volume_capped,
        )
