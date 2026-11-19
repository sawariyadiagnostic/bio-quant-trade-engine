"""Composite Strategy and Online Bayesian Bandit Allocator for RAVANA.

Implements:
1. StrategyBanditAllocator:
   Online Bounded Bayesian Bandit over the 10 structural triggers.
   - Exponentially decayed Beta posterior:
     alpha_k <- gamma * alpha_k + I(net_pnl > 0)
     beta_k  <- gamma * beta_k  + I(net_pnl <= 0)  (gamma = 0.98)
   - Scales conviction by posterior mean mu_k = alpha_k / (alpha_k + beta_k),
     clamped strictly in [0.70x, 1.30x].
   - Burn-in gate: Enforces equal 1.0x weights until a trigger has accumulated
     >= 15 closed trades.

2. TriggerCompositeStrategy:
   - Dispatches across the 3 regime buckets:
     Bucket 1: Trend & Impulse (VR > 1.05)
     Bucket 2: Mean Reversion & Absorption (VR <= 1.05)
     Bucket 3: Macro & Overnight (Always active in session windows)
   - Resolves capital contention: Limits max active positions to max_concurrent_positions = 2.
   - Integrates with BoundedASTModulator and StrategyBanditAllocator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from ravana.domain.formula_ast import ASTNode
from ravana.strategies.base import (
    StructuralTrigger,
    TriggerAction,
    TriggerSignal,
)
from ravana.strategies.modulator import BoundedASTModulator
from ravana.strategies.triggers import (
    AfternoonBondEuropeTWAPTrigger,
    ClosingVWAPSnapbackTrigger,
    CrossSectorLeadLagTrigger,
    DIISIPVWAPAbsorptionTrigger,
    ICTUnicornStopSweepTrigger,
    INAVSpreadMeanReversionTrigger,
    OpeningRangeBreakoutTrigger,
    OvernightContinuationTrigger,
    SectorPEADInstitutionalDriftTrigger,
    ThursdayExpiryGammaTrigger,
    extract_variance_ratio,
)

logger = logging.getLogger(__name__)

ALL_TRIGGER_NAMES: Tuple[str, ...] = (
    "OpeningRangeBreakoutTrigger",
    "ThursdayExpiryGammaTrigger",
    "CrossSectorLeadLagTrigger",
    "SectorPEADInstitutionalDriftTrigger",
    "INAVSpreadMeanReversionTrigger",
    "ClosingVWAPSnapbackTrigger",
    "DIISIPVWAPAbsorptionTrigger",
    "ICTUnicornStopSweepTrigger",
    "AfternoonBondEuropeTWAPTrigger",
    "OvernightContinuationTrigger",
)

BUCKET_1_NAMES: Set[str] = {
    "OpeningRangeBreakoutTrigger",
    "ThursdayExpiryGammaTrigger",
    "CrossSectorLeadLagTrigger",
    "SectorPEADInstitutionalDriftTrigger",
}

BUCKET_2_NAMES: Set[str] = {
    "INAVSpreadMeanReversionTrigger",
    "ClosingVWAPSnapbackTrigger",
    "DIISIPVWAPAbsorptionTrigger",
    "ICTUnicornStopSweepTrigger",
}

BUCKET_3_NAMES: Set[str] = {
    "AfternoonBondEuropeTWAPTrigger",
    "OvernightContinuationTrigger",
}


@dataclass
class BanditArmStats:
    """Statistics for a single trigger bandit arm."""

    trigger_name: str
    alpha: float
    beta: float
    trade_count: int
    posterior_mean: float
    multiplier: float


class StrategyBanditAllocator:
    """Online Bounded Bayesian Bandit Allocator with exponential decay and burn-in gate."""

    def __init__(
        self,
        trigger_names: Optional[List[str]] = None,
        gamma: float = 0.98,
        burn_in_trades: int = 15,
        min_multiplier: float = 0.70,
        max_multiplier: float = 1.30,
        scaling_factor: float = 2.0,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ) -> None:
        """Initialize Bayesian Bandit state trackers."""
        self.trigger_names = list(trigger_names or ALL_TRIGGER_NAMES)
        self.gamma = gamma
        self.burn_in_trades = burn_in_trades
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier
        self.scaling_factor = scaling_factor
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

        self.alpha: Dict[str, float] = {k: prior_alpha for k in self.trigger_names}
        self.beta: Dict[str, float] = {k: prior_beta for k in self.trigger_names}
        self.trade_counts: Dict[str, int] = {k: 0 for k in self.trigger_names}

    def _ensure_trigger(self, trigger_name: str) -> None:
        """Ensure trigger exists in posterior state registries."""
        if trigger_name not in self.alpha:
            self.alpha[trigger_name] = self.prior_alpha
            self.beta[trigger_name] = self.prior_beta
            self.trade_counts[trigger_name] = 0

    def get_posterior_mean(self, trigger_name: str) -> float:
        """Calculate Beta posterior mean: mu = alpha / (alpha + beta)."""
        self._ensure_trigger(trigger_name)
        a = self.alpha[trigger_name]
        b = self.beta[trigger_name]
        total = a + b
        if total <= 0.0:
            return 0.50
        return float(a / total)

    def get_multiplier(self, trigger_name: str) -> float:
        """Compute conviction multiplier bounded in [0.70x, 1.30x] with burn-in gate."""
        self._ensure_trigger(trigger_name)
        # Burn-in Gate: Enforce equal 1.0x weights until >= 15 closed trades
        if self.trade_counts[trigger_name] < self.burn_in_trades:
            return 1.00

        mu_k = self.get_posterior_mean(trigger_name)
        raw_mult = self.scaling_factor * mu_k
        return float(max(self.min_multiplier, min(self.max_multiplier, raw_mult)))

    def scale_conviction(self, trigger_name: str, base_conviction: float) -> float:
        """Scale trigger conviction by bounded Bayesian Bandit multiplier."""
        if base_conviction <= 0.0:
            return 0.0
        mult = self.get_multiplier(trigger_name)
        scaled = base_conviction * mult
        return float(max(0.0, min(1.0, scaled)))

    def update(self, trigger_name: str, net_pnl: float) -> None:
        """Update Beta posterior using exponential decay gamma: alpha*gamma + I(pnl>0)."""
        self._ensure_trigger(trigger_name)
        self.trade_counts[trigger_name] += 1

        is_win = 1.0 if net_pnl > 0.0 else 0.0
        is_loss = 1.0 if net_pnl <= 0.0 else 0.0

        self.alpha[trigger_name] = self.gamma * self.alpha[trigger_name] + is_win
        self.beta[trigger_name] = self.gamma * self.beta[trigger_name] + is_loss

    def record_trade(self, trigger_name: str, net_pnl: float) -> None:
        """Alias for update() to record trade outcome."""
        self.update(trigger_name, net_pnl)

    def get_state(self) -> Dict[str, Any]:
        """Export serialized posterior state dictionary."""
        return {
            "gamma": self.gamma,
            "burn_in_trades": self.burn_in_trades,
            "alpha": dict(self.alpha),
            "beta": dict(self.beta),
            "trade_counts": dict(self.trade_counts),
            "posterior_means": {k: self.get_posterior_mean(k) for k in self.alpha},
            "multipliers": {k: self.get_multiplier(k) for k in self.alpha},
        }

    def get_arm_stats(self, trigger_name: str) -> BanditArmStats:
        """Return structured statistics for a specific trigger arm."""
        self._ensure_trigger(trigger_name)
        return BanditArmStats(
            trigger_name=trigger_name,
            alpha=self.alpha[trigger_name],
            beta=self.beta[trigger_name],
            trade_count=self.trade_counts[trigger_name],
            posterior_mean=self.get_posterior_mean(trigger_name),
            multiplier=self.get_multiplier(trigger_name),
        )


class TriggerCompositeStrategy:
    """Dispatches across 3 regime buckets with capital contention limits and bandit tuning."""

    def __init__(
        self,
        triggers: Optional[List[StructuralTrigger]] = None,
        modulator: Optional[BoundedASTModulator] = None,
        bandit: Optional[StrategyBanditAllocator] = None,
        max_concurrent_positions: int = 4,
        vr_threshold: float = 1.05,
    ) -> None:
        """Initialize composite strategy with all 10 structural triggers."""
        self.triggers = triggers or self._create_default_triggers()
        self.modulator = modulator or BoundedASTModulator()
        self.bandit = bandit or StrategyBanditAllocator()
        self.max_concurrent_positions = max_concurrent_positions
        self.vr_threshold = vr_threshold

        # Active position tracking
        self.active_positions: Set[str] = set()

    @staticmethod
    def _create_default_triggers() -> List[StructuralTrigger]:
        """Instantiate canonical 10 structural triggers."""
        return [
            OpeningRangeBreakoutTrigger(),
            ThursdayExpiryGammaTrigger(),
            CrossSectorLeadLagTrigger(),
            SectorPEADInstitutionalDriftTrigger(),
            INAVSpreadMeanReversionTrigger(),
            ClosingVWAPSnapbackTrigger(),
            DIISIPVWAPAbsorptionTrigger(),
            ICTUnicornStopSweepTrigger(),
            AfternoonBondEuropeTWAPTrigger(),
            OvernightContinuationTrigger(),
        ]

    def get_active_bucket(self, vr: float) -> str:
        """Return active regime bucket name based on Variance Ratio."""
        if vr > self.vr_threshold:
            return "BUCKET_1"
        return "BUCKET_2"

    def is_trigger_active_in_regime(self, trigger_name: str, vr: float) -> bool:
        """Determine whether trigger is active under current Variance Ratio regime."""
        if trigger_name in BUCKET_3_NAMES:
            return True
        if vr > self.vr_threshold:
            return trigger_name in BUCKET_1_NAMES
        return trigger_name in BUCKET_2_NAMES

    def evaluate(
        self,
        df: pd.DataFrame,
        symbol: str,
        ast_node: Optional[ASTNode] = None,
    ) -> pd.Series:
        """Evaluate market data and return combined modulated signed conviction series."""
        if len(df) == 0:
            return pd.Series(dtype=float)

        vr_series = extract_variance_ratio(df)
        combined = pd.Series(0.0, index=df.index)

        for trig in self.triggers:
            series = trig.evaluate(df, symbol)
            # Apply regime gating mask per bar
            if trig.name in BUCKET_1_NAMES:
                active_mask = vr_series > self.vr_threshold
                series = series.where(active_mask, 0.0)
            elif trig.name in BUCKET_2_NAMES:
                active_mask = vr_series <= self.vr_threshold
                series = series.where(active_mask, 0.0)

            # Apply Bandit posterior scaling
            bandit_mult = self.bandit.get_multiplier(trig.name)
            series = series * bandit_mult

            # Combine: choose signal with maximum absolute conviction
            combined = np.where(series.abs() > combined.abs(), series, combined)
            combined = pd.Series(combined, index=df.index)

        # Modulate through AST formula (Safe Epsilon + Sigmoid + Stagnation Sentinel)
        modulated = self.modulator.modulate_series(combined.abs(), ast_node, df)
        signed_result = np.sign(combined) * modulated
        return signed_result.clip(-1.0, 1.0)

    def generate_signals(
        self,
        df: pd.DataFrame,
        symbol: str,
        ast_node: Optional[ASTNode] = None,
    ) -> List[TriggerSignal]:
        """Generate discrete signals for single symbol applying regime gate and modulators."""
        if len(df) == 0:
            return []

        vr_series = extract_variance_ratio(df)
        current_vr = float(vr_series.iloc[-1]) if len(vr_series) > 0 else 1.0

        raw_signals: List[TriggerSignal] = []
        for trig in self.triggers:
            if not self.is_trigger_active_in_regime(trig.name, current_vr):
                continue
            sigs = trig.generate_signals(df, symbol)
            raw_signals.extend(sigs)

        # Apply AST Modulator
        modulated_signals = self.modulator.modulate_signals(raw_signals, ast_node, df)

        # Apply Bayesian Bandit conviction scaling
        final_signals: List[TriggerSignal] = []
        for sig in modulated_signals:
            scaled_conv = self.bandit.scale_conviction(sig.trigger_name, sig.conviction)
            meta: Dict[str, Any] = dict(sig.metadata)
            meta["bandit_multiplier"] = self.bandit.get_multiplier(sig.trigger_name)
            meta["regime_vr"] = current_vr
            meta["regime_bucket"] = self.get_active_bucket(current_vr)

            final_signals.append(
                TriggerSignal(
                    timestamp=sig.timestamp,
                    symbol=sig.symbol,
                    action=sig.action,
                    conviction=scaled_conv,
                    trigger_name=sig.trigger_name,
                    metadata=meta,
                )
            )

        return final_signals

    def _partition_signals(
        self,
        signals: List[TriggerSignal],
        positions: Set[str],
    ) -> Tuple[List[TriggerSignal], List[TriggerSignal]]:
        """Separate signals into priority exits and candidate entries, updating open slots."""
        exit_signals: List[TriggerSignal] = []
        entry_signals: List[TriggerSignal] = []
        for sig in signals:
            if sig.action in (TriggerAction.EXIT_ALL, TriggerAction.EXIT_TIME_STOP, TriggerAction.SELL):
                exit_signals.append(sig)
                positions.discard(sig.symbol)
            elif sig.action == TriggerAction.EXIT_TRANCHE_1:
                exit_signals.append(sig)
            elif sig.action == TriggerAction.BUY:
                entry_signals.append(sig)
        return exit_signals, entry_signals

    def _select_entries_under_contention(
        self,
        entry_signals: List[TriggerSignal],
        positions: Set[str],
    ) -> List[TriggerSignal]:
        """Sort BUY candidates by conviction and accept up to max_concurrent_positions."""
        entry_signals.sort(key=lambda s: s.conviction, reverse=True)
        accepted: List[TriggerSignal] = []
        for sig in entry_signals:
            if sig.conviction <= 0.0 or sig.symbol in positions:
                continue
            if len(positions) < self.max_concurrent_positions:
                accepted.append(sig)
                positions.add(sig.symbol)
            else:
                logger.info(
                    "Capital Contention: Dropped BUY for %s (conv=%.2f), limit %d reached.",
                    sig.symbol,
                    sig.conviction,
                    self.max_concurrent_positions,
                )
        return accepted

    def generate_portfolio_signals(
        self,
        market_data: Dict[str, pd.DataFrame],
        ast_nodes: Optional[Dict[str, ASTNode]] = None,
        current_positions: Optional[Set[str]] = None,
    ) -> List[TriggerSignal]:
        """Resolve capital contention across universe capping active slots at max_concurrent."""
        positions = set(current_positions) if current_positions is not None else set(self.active_positions)
        nodes = ast_nodes or {}

        all_signals: List[TriggerSignal] = []
        for symbol, df in market_data.items():
            sigs = self.generate_signals(df, symbol, ast_node=nodes.get(symbol))
            all_signals.extend(sigs)

        exit_sigs, entry_sigs = self._partition_signals(all_signals, positions)
        accepted_entries = self._select_entries_under_contention(entry_sigs, positions)

        self.active_positions = positions
        return exit_sigs + accepted_entries

    def on_trade_closed(self, trigger_name: str, net_pnl: float) -> None:
        """Feed closed trade PnL back into the Bayesian Bandit allocator."""
        self.bandit.update(trigger_name, net_pnl)
