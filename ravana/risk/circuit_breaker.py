"""Muscle 18: Circuit Breakers & Emergency Kill Switch.

Enforces daily maximum loss limits, trailing drawdown halts (chained with Muscle 12),
consecutive losing trade cooling periods (chained with Muscle 11 RoundTripTrades),
and an atomic manual/automated Kill Switch blocking all risk-increasing orders.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence
import numpy as np

from ravana.backtest.orders import RoundTripTrade
from ravana.risk.config import DrawdownState, RiskConfig


@dataclass(frozen=True)
class CircuitBreakerStatus:
    """Immutable telemetry emitted by Muscle 18 before every order dispatch."""

    is_trading_allowed: bool
    kill_switch_engaged: bool
    drawdown_state: DrawdownState
    daily_return_pct: float
    trailing_drawdown_pct: float
    consecutive_losses: int
    in_cooldown: bool
    reason: str


class CircuitBreakerKillSwitch:
    """Sovereign capital protection sentry and emergency kill switch."""

    def __init__(
        self,
        risk_config: Optional[RiskConfig] = None,
        max_daily_loss_pct: float = 0.02,       # 2.0% maximum intraday/daily loss halt
        max_consecutive_losses: int = 4,        # 4 consecutive losing round-trips triggers cooldown
    ) -> None:
        self.config = risk_config or RiskConfig()
        self.max_daily_loss_pct = abs(float(max_daily_loss_pct))
        self.max_consecutive_losses = max(1, int(max_consecutive_losses))
        self._kill_switch_engaged: bool = False
        self._kill_reason: str = ""
        self._active_day_start_equity: Optional[float] = None

    def trigger_kill_switch(self, reason: str = "Manual Emergency Halt") -> None:
        """Engage atomic emergency kill switch halting all new entries."""
        self._kill_switch_engaged = True
        self._kill_reason = reason

    def reset_kill_switch(self) -> None:
        """Disengage emergency kill switch after operator verification."""
        self._kill_switch_engaged = False
        self._kill_reason = ""

    def start_new_day(self, opening_equity: float) -> None:
        """Set baseline equity for the new calendar trading session."""
        if opening_equity > 0 and np.isfinite(opening_equity):
            self._active_day_start_equity = float(opening_equity)

    @property
    def is_killed(self) -> bool:
        return self._kill_switch_engaged

    @staticmethod
    def count_consecutive_losses(closed_trades: Sequence[RoundTripTrade]) -> int:
        """Count trailing streak of consecutive losing round-trip trades from Muscle 11."""
        streak = 0
        for trade in reversed(closed_trades):
            if not trade.is_win:
                streak += 1
            else:
                break
        return streak

    def evaluate(
        self,
        current_equity: float,
        day_start_equity: Optional[float] = None,
        peak_equity: Optional[float] = None,
        closed_trades: Optional[Sequence[RoundTripTrade]] = None,
    ) -> CircuitBreakerStatus:
        """Evaluate all capital protection gates prior to order execution."""
        if not np.isfinite(current_equity) or current_equity <= 0:
            self.trigger_kill_switch("Non-positive or NaN account equity detected.")

        base_start = day_start_equity if day_start_equity is not None else (self._active_day_start_equity or current_equity)
        daily_ret = (
            (current_equity - base_start) / base_start
            if base_start > 0 else 0.0
        )
        peak = peak_equity if peak_equity is not None else current_equity
        eff_peak = max(peak, current_equity, base_start)
        trailing_dd = abs(min(0.0, (current_equity - eff_peak) / eff_peak)) if eff_peak > 0 else 0.0
        loss_streak = self.count_consecutive_losses(closed_trades or [])

        if self._kill_switch_engaged:
            return CircuitBreakerStatus(
                is_trading_allowed=False,
                kill_switch_engaged=True,
                drawdown_state=DrawdownState.HALTED,
                daily_return_pct=round(daily_ret * 100.0, 3),
                trailing_drawdown_pct=round(trailing_dd * 100.0, 3),
                consecutive_losses=loss_streak,
                in_cooldown=False,
                reason=f"KILL SWITCH ACTIVE: {self._kill_reason}",
            )

        if daily_ret <= -self.max_daily_loss_pct:
            return CircuitBreakerStatus(
                is_trading_allowed=False,
                kill_switch_engaged=False,
                drawdown_state=DrawdownState.HALTED,
                daily_return_pct=round(daily_ret * 100.0, 3),
                trailing_drawdown_pct=round(trailing_dd * 100.0, 3),
                consecutive_losses=loss_streak,
                in_cooldown=True,
                reason=f"Daily loss ({daily_ret*100:.2f}%) breached -{self.max_daily_loss_pct*100:.2f}% limit.",
            )

        if trailing_dd >= self.config.drawdown_halt_threshold:
            return CircuitBreakerStatus(
                is_trading_allowed=False,
                kill_switch_engaged=False,
                drawdown_state=DrawdownState.HALTED,
                daily_return_pct=round(daily_ret * 100.0, 3),
                trailing_drawdown_pct=round(trailing_dd * 100.0, 3),
                consecutive_losses=loss_streak,
                in_cooldown=True,
                reason=f"Trailing drawdown ({trailing_dd*100:.2f}%) breached {self.config.drawdown_halt_threshold*100:.2f}% HALT gate.",
            )

        if loss_streak >= self.max_consecutive_losses:
            return CircuitBreakerStatus(
                is_trading_allowed=False,
                kill_switch_engaged=False,
                drawdown_state=DrawdownState.CAUTION_HALVED,
                daily_return_pct=round(daily_ret * 100.0, 3),
                trailing_drawdown_pct=round(trailing_dd * 100.0, 3),
                consecutive_losses=loss_streak,
                in_cooldown=True,
                reason=f"Consecutive loss streak ({loss_streak}) reached cooldown threshold ({self.max_consecutive_losses}).",
            )

        dd_state = (
            DrawdownState.CAUTION_HALVED
            if trailing_dd >= self.config.drawdown_halve_threshold
            else DrawdownState.NORMAL
        )
        return CircuitBreakerStatus(
            is_trading_allowed=True,
            kill_switch_engaged=False,
            drawdown_state=dd_state,
            daily_return_pct=round(daily_ret * 100.0, 3),
            trailing_drawdown_pct=round(trailing_dd * 100.0, 3),
            consecutive_losses=loss_streak,
            in_cooldown=False,
            reason="All circuit breakers green.",
        )
