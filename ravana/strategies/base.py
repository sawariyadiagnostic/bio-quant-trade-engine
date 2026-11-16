"""Base abstractions and Zero-Giveback stop engine for RAVANA structural triggers.

Defines:
- TriggerAction: Discrete action states for trade lifecycle.
- StopLayerState: State tracker across the 5 zero-giveback stop layers.
- TriggerSignal: Immutable event object emitted by structural triggers.
- StructuralTrigger: Abstract base class for all sovereign quant triggers.
- ZeroGivebackStopEngine: Institutional 5-layer stop & scale-out state machine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


class TriggerAction(str, Enum):
    """Discrete execution actions emitted by triggers and stop engines."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    EXIT_ALL = "EXIT_ALL"
    EXIT_TRANCHE_1 = "EXIT_TRANCHE_1"
    EXIT_TIME_STOP = "EXIT_TIME_STOP"


# Alias for backward compatibility
Action = TriggerAction


@dataclass
class StopLayerState:
    """State tracker for the 5-Layer Zero-Giveback stop and scale-out engine."""

    initial_stop: float
    breakeven_stop: float
    is_breakeven_triggered: bool = False
    tranche_1_sold: bool = False
    chandelier_stop: float = 0.0
    active_stop: float = 0.0
    peak_price: float = 0.0
    bars_held: int = 0
    is_long: bool = True


@dataclass(frozen=True)
class TriggerSignal:
    """Immutable discrete execution event emitted by a structural trigger."""

    timestamp: pd.Timestamp
    symbol: str
    action: TriggerAction
    conviction: float
    trigger_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Enforce conviction boundary [0.0, 1.0]."""
        val = float(self.conviction)
        bounded = max(0.0, min(1.0, val))
        object.__setattr__(self, "conviction", bounded)


class StructuralTrigger(ABC):
    """Abstract base class for all institutional structural triggers."""

    name: str = "StructuralTrigger"

    @abstractmethod
    def evaluate(self, df: pd.DataFrame, symbol: str) -> pd.Series:
        """Evaluate market data and return signed conviction series in [-1.0, 1.0]."""
        pass

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[TriggerSignal]:
        """Generate discrete execution signals from market data."""
        pass


class ZeroGivebackStopEngine:
    """Institutional 5-Layer 'Zero-Giveback' Stop & Two-Tranche Scale-Out Engine.

    Layer 1: Thesis Invalidation Stop (capped at -30 to -35 bps max loss).
    Layer 2: +0.45 ATR Breakeven Ratchet (locks entry + 2.0 bps statutory friction).
    Layer 3: Two-Tranche Scale-Out at +0.75 ATR (sells 50% shares, snaps stop to +0.25 ATR, MIS only).
    Layer 4: Chandelier Ratchet at +1.0 ATR (locks peak - 0.35 ATR).
    Layer 5: 4-Bar Dead-Money Time Stop (exits flat after 20 mins if peak < +0.20 ATR).
    """

    def __init__(
        self,
        max_invalidation_bps: float = 35.0,
        breakeven_friction_bps: float = 2.0,
        breakeven_atr_threshold: float = 0.45,
        tranche1_atr_threshold: float = 0.75,
        runner_snap_atr: float = 0.25,
        chandelier_activation_atr: float = 1.0,
        chandelier_distance_atr: float = 0.35,
        dead_money_bars: int = 4,
        dead_money_peak_atr: float = 0.20,
    ) -> None:
        self.max_invalidation_bps = max_invalidation_bps
        self.breakeven_friction_bps = breakeven_friction_bps
        self.breakeven_atr_threshold = breakeven_atr_threshold
        self.tranche1_atr_threshold = tranche1_atr_threshold
        self.runner_snap_atr = runner_snap_atr
        self.chandelier_activation_atr = chandelier_activation_atr
        self.chandelier_distance_atr = chandelier_distance_atr
        self.dead_money_bars = dead_money_bars
        self.dead_money_peak_atr = dead_money_peak_atr

    def initialize_position(
        self,
        entry_price: float,
        atr: float,
        is_long: bool = True,
        custom_stop: Optional[float] = None,
    ) -> StopLayerState:
        """Initialize Layer 1 Thesis Invalidation Stop and initial state."""
        if is_long:
            floor_stop = entry_price * (1.0 - self.max_invalidation_bps / 10000.0)
            init_stop = max(custom_stop, floor_stop) if custom_stop is not None else floor_stop
            be_stop = entry_price * (1.0 + self.breakeven_friction_bps / 10000.0)
        else:
            ceil_stop = entry_price * (1.0 + self.max_invalidation_bps / 10000.0)
            init_stop = min(custom_stop, ceil_stop) if custom_stop is not None else ceil_stop
            be_stop = entry_price * (1.0 - self.breakeven_friction_bps / 10000.0)

        return StopLayerState(
            initial_stop=init_stop,
            breakeven_stop=be_stop,
            is_breakeven_triggered=False,
            tranche_1_sold=False,
            chandelier_stop=0.0,
            active_stop=init_stop,
            peak_price=entry_price,
            bars_held=0,
            is_long=is_long,
        )

    def update(
        self,
        state: StopLayerState,
        entry_price: float,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        atr: float,
        is_long: Optional[bool] = None,
        is_mis: bool = True,
    ) -> Tuple[StopLayerState, TriggerAction]:
        """Evaluate the 5 layers against the current bar and return updated state and action."""
        if is_long is None:
            is_long = getattr(state, "is_long", True)
        state.bars_held += 1

        if is_long:
            state.peak_price = max(state.peak_price, bar_high)
            if bar_low <= state.active_stop:
                return state, TriggerAction.EXIT_ALL
            if state.bars_held >= self.dead_money_bars and state.peak_price < entry_price + self.dead_money_peak_atr * atr:
                return state, TriggerAction.EXIT_TIME_STOP
            if not state.is_breakeven_triggered and bar_high >= entry_price + self.breakeven_atr_threshold * atr:
                state.is_breakeven_triggered = True
                state.active_stop = max(state.active_stop, state.breakeven_stop)
            action = TriggerAction.HOLD
            if is_mis and not state.tranche_1_sold and bar_high >= entry_price + self.tranche1_atr_threshold * atr:
                state.tranche_1_sold = True
                state.active_stop = max(state.active_stop, entry_price + self.runner_snap_atr * atr)
                action = TriggerAction.EXIT_TRANCHE_1
            if state.peak_price >= entry_price + self.chandelier_activation_atr * atr:
                dist = 0.15 if state.peak_price >= entry_price + 1.8 * atr else self.chandelier_distance_atr
                chan_stop = state.peak_price - dist * atr
                state.chandelier_stop = chan_stop
                state.active_stop = max(state.active_stop, chan_stop)
            return state, action
        else:
            state.peak_price = min(state.peak_price, bar_low) if state.peak_price > 0 else bar_low
            if bar_high >= state.active_stop:
                return state, TriggerAction.EXIT_ALL
            if state.bars_held >= self.dead_money_bars and state.peak_price > entry_price - self.dead_money_peak_atr * atr:
                return state, TriggerAction.EXIT_TIME_STOP
            if not state.is_breakeven_triggered and bar_low <= entry_price - self.breakeven_atr_threshold * atr:
                state.is_breakeven_triggered = True
                state.active_stop = min(state.active_stop, state.breakeven_stop)
            action = TriggerAction.HOLD
            if is_mis and not state.tranche_1_sold and bar_low <= entry_price - self.tranche1_atr_threshold * atr:
                state.tranche_1_sold = True
                state.active_stop = min(state.active_stop, entry_price - self.runner_snap_atr * atr)
                action = TriggerAction.EXIT_TRANCHE_1
            if state.peak_price <= entry_price - self.chandelier_activation_atr * atr:
                dist = 0.15 if state.peak_price <= entry_price - 1.8 * atr else self.chandelier_distance_atr
                chan_stop = state.peak_price + dist * atr
                state.chandelier_stop = chan_stop
                state.active_stop = min(state.active_stop, chan_stop)
            return state, action
