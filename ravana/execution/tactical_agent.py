"""
Tactical Execution Agent for Timeframe Decoupling in RAVANA v4.2.
Decouples slower Strategic Alpha (5m/15m/1h) from faster Tactical Execution (1m/Tick).
Works passive limit bids at micro-VWAP / iNAV discount to eliminate spread drag and avoid wicks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from ravana.backtest.orders import Order, OrderSide, Trade


@dataclass(frozen=True)
class StrategicIntent:
    """Trading intent emitted by the slower Alpha Clock."""

    ticker: str
    side: OrderSide
    shares: int
    alpha_timestamp: pd.Timestamp
    alpha_reference_price: float
    trigger_name: str
    conviction: float
    max_wait_bars: int = 3
    inav_max_premium_bps: float = 15.0


@dataclass(frozen=True)
class TacticalExecutionResult:
    """Execution receipt from the tactical limit-order agent."""

    is_filled: bool
    filled_shares: int
    executed_price: float
    spread_savings_bps: float
    bars_waited: int
    fill_timestamp: pd.Timestamp
    reason: str


class TacticalExecutionAgent:
    """Manages passive limit execution at intrinsic value / micro-VWAP pullbacks."""

    def __init__(self, default_spread_bps: float = 5.0) -> None:
        self.default_spread_bps = float(default_spread_bps)

    def calculate_passive_limit_price(
        self,
        intent: StrategicIntent,
        current_bar: pd.Series,
    ) -> float:
        """Calculate passive limit bid/ask price at micro-VWAP or inside touch."""
        curr_close = float(current_bar["Close"])
        curr_low = float(current_bar["Low"])
        curr_high = float(current_bar["High"])
        curr_vwap = float(current_bar.get("VWAP", curr_close))
        spread_half = curr_close * (self.default_spread_bps / 10000.0)

        if intent.side == OrderSide.BUY:
            best_bid = curr_close - spread_half
            target_limit = min(best_bid, curr_vwap, intent.alpha_reference_price)
            return float(min(target_limit, curr_close))

        # SELL order: post passive ask above close/micro-VWAP
        best_ask = curr_close + spread_half
        target_limit = max(best_ask, curr_vwap, intent.alpha_reference_price)
        return float(max(target_limit, curr_close))

    def evaluate_tactical_fill(
        self,
        intent: StrategicIntent,
        bar: pd.Series,
        bar_ts: pd.Timestamp,
        bars_waited: int,
    ) -> TacticalExecutionResult:
        """Evaluate if tactical limit order is filled on incoming execution bar."""
        if bars_waited > intent.max_wait_bars:
            return TacticalExecutionResult(
                is_filled=False,
                filled_shares=0,
                executed_price=0.0,
                spread_savings_bps=0.0,
                bars_waited=bars_waited,
                fill_timestamp=bar_ts,
                reason="TACTICAL_TIMEOUT",
            )

        # Check iNAV spread gating
        spread_bps = float(bar.get("SPREAD_BPS", bar.get("Basis_Bps", 0.0)))
        if intent.side == OrderSide.BUY and spread_bps > intent.inav_max_premium_bps:
            return TacticalExecutionResult(
                is_filled=False,
                filled_shares=0,
                executed_price=0.0,
                spread_savings_bps=0.0,
                bars_waited=bars_waited,
                fill_timestamp=bar_ts,
                reason="INAV_PREMIUM_BLOCK",
            )

        limit_price = self.calculate_passive_limit_price(intent, bar)
        bar_low = float(bar["Low"])
        bar_high = float(bar["High"])
        curr_close = float(bar["Close"])

        # Passive Limit Fill checks
        if intent.side == OrderSide.BUY and bar_low <= limit_price:
            savings = ((intent.alpha_reference_price - limit_price) / intent.alpha_reference_price) * 10000.0
            return TacticalExecutionResult(
                is_filled=True,
                filled_shares=intent.shares,
                executed_price=limit_price,
                spread_savings_bps=max(0.0, float(savings)),
                bars_waited=bars_waited,
                fill_timestamp=bar_ts,
                reason="LIMIT_DIP_FILLED",
            )
        if intent.side == OrderSide.SELL and bar_high >= limit_price:
            savings = ((limit_price - intent.alpha_reference_price) / intent.alpha_reference_price) * 10000.0
            return TacticalExecutionResult(
                is_filled=True,
                filled_shares=intent.shares,
                executed_price=limit_price,
                spread_savings_bps=max(0.0, float(savings)),
                bars_waited=bars_waited,
                fill_timestamp=bar_ts,
                reason="LIMIT_LIFT_FILLED",
            )

        # 3-Bar Escalation Rule: Sweep inside touch if conviction >= 0.70 or exiting
        if bars_waited == intent.max_wait_bars and (intent.conviction >= 0.70 or intent.side == OrderSide.SELL):
            return TacticalExecutionResult(
                is_filled=True,
                filled_shares=intent.shares,
                executed_price=curr_close,
                spread_savings_bps=0.0,
                bars_waited=bars_waited,
                fill_timestamp=bar_ts,
                reason="ESCALATION_SWEEP_FILLED",
            )

        wait_reason = "WAITING_FOR_DIP" if intent.side == OrderSide.BUY else "WAITING_FOR_LIFT"
        return TacticalExecutionResult(
            is_filled=False,
            filled_shares=0,
            executed_price=0.0,
            spread_savings_bps=0.0,
            bars_waited=bars_waited,
            fill_timestamp=bar_ts,
            reason=wait_reason,
        )
