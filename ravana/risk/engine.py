"""Institutional Portfolio Risk and Microstructure Allocation Engine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from ravana.risk.config import AllocationMethod, DrawdownState, RiskConfig
from ravana.risk.gatekeeper import RiskGatekeeper
from ravana.risk.hrp import HierarchicalRiskParity


@dataclass
class AllocationResult:
    """Target risk-adjusted portfolio weights, notionals, and share orders."""
    target_weights: Dict[str, float]
    target_shares: Dict[str, int]
    target_notionals: Dict[str, float]
    drawdown_state: DrawdownState
    drawdown_multiplier: float
    cash_reserve_inr: float
    total_allocated_inr: float


class PortfolioRiskEngine:
    """Computes risk-budgeted target weights and enforces Indian microstructure gates."""

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        self.gatekeeper = RiskGatekeeper(self.config)

    def allocate(
        self,
        symbols: List[str],
        returns_df: pd.DataFrame,
        current_prices: Dict[str, float],
        current_equity: float,
        peak_equity: float,
        alpha_signals: Optional[Dict[str, float]] = None,
    ) -> AllocationResult:
        """Compute institutional target weights and share allocations across assets."""
        n_assets = len(symbols)
        if n_assets == 0:
            return self._empty_result(current_equity, DrawdownState.NORMAL, 1.0)

        # 1. Capital Viability Gate (Min ₹2,00,000)
        is_viable, msg = self.gatekeeper.validate_account_capital(current_equity)
        if not is_viable:
            return self._empty_result(current_equity, DrawdownState.HALTED, 0.0)

        # 2. Drawdown Circuit Breaker
        dd_state, dd_mult = self.gatekeeper.evaluate_drawdown_breaker(peak_equity, current_equity)
        if dd_mult == 0.0:
            return self._empty_result(current_equity, dd_state, 0.0)

        # 3. Base Asset Weighting (HRP or Volatility Parity)
        raw_weights = self._compute_base_weights(symbols, returns_df)

        # 4. Alpha Signal Blending
        if alpha_signals is not None:
            raw_weights = self._blend_with_alphas(symbols, raw_weights, alpha_signals)

        # 5. Position Limit & Gross Exposure Clamping
        constrained_weights = self.gatekeeper.enforce_position_limits(raw_weights)
        scaled_weights = constrained_weights * dd_mult

        # 6. Sizing Target Shares & Viability Filtering
        target_weights: Dict[str, float] = {}
        target_shares: Dict[str, int] = {}
        target_notionals: Dict[str, float] = {}
        total_allocated = 0.0

        for i, sym in enumerate(symbols):
            px = current_prices.get(sym, 0.0)
            w = float(scaled_weights[i])
            if px <= 0 or w <= 0:
                continue

            notional = current_equity * w
            shares = int(notional // px)
            is_valid, final_shares, _ = self.gatekeeper.filter_trade_viability(
                price=px, shares=shares, expected_alpha_bps=10.0
            )

            if is_valid and final_shares > 0:
                final_notional = final_shares * px
                target_weights[sym] = round(final_notional / current_equity, 4)
                target_shares[sym] = final_shares
                target_notionals[sym] = round(final_notional, 2)
                total_allocated += final_notional

        cash_reserve = max(0.0, current_equity - total_allocated)

        return AllocationResult(
            target_weights=target_weights,
            target_shares=target_shares,
            target_notionals=target_notionals,
            drawdown_state=dd_state,
            drawdown_multiplier=dd_mult,
            cash_reserve_inr=round(cash_reserve, 2),
            total_allocated_inr=round(total_allocated, 2),
        )

    def _compute_base_weights(self, symbols: List[str], returns_df: pd.DataFrame) -> np.ndarray:
        """Compute base risk-budgeted weights according to configured allocation method."""
        if len(symbols) <= 1:
            return np.array([1.0], dtype=np.float64)

        if self.config.allocation_method == AllocationMethod.HRP and not returns_df.empty:
            valid_cols = [c for c in symbols if c in returns_df.columns]
            if len(valid_cols) == len(symbols):
                return HierarchicalRiskParity.compute_weights(
                    returns_df[valid_cols].to_numpy(),
                    linkage_method=self.config.hrp_linkage,
                )

        if self.config.allocation_method == AllocationMethod.INVERSE_VOL and not returns_df.empty:
            vol = returns_df[symbols].std().to_numpy()
            inv_vol = 1.0 / np.maximum(1e-8, vol)
            return inv_vol / np.sum(inv_vol)

        return np.full(len(symbols), 1.0 / len(symbols), dtype=np.float64)

    @staticmethod
    def _blend_with_alphas(
        symbols: List[str], base_weights: np.ndarray, alpha_signals: Dict[str, float]
    ) -> np.ndarray:
        """Modulate base risk weights by directional positive alpha forecasts."""
        multipliers = np.array([max(0.0, alpha_signals.get(s, 0.0)) for s in symbols], dtype=np.float64)
        if np.sum(multipliers) > 0:
            multipliers = multipliers / np.mean(multipliers)
            blended = base_weights * multipliers
            total = np.sum(blended)
            return (blended / total) if total > 0 else base_weights
        return base_weights

    @staticmethod
    def _empty_result(equity: float, state: DrawdownState, mult: float) -> AllocationResult:
        return AllocationResult(
            target_weights={}, target_shares={}, target_notionals={},
            drawdown_state=state, drawdown_multiplier=mult,
            cash_reserve_inr=round(equity, 2), total_allocated_inr=0.0,
        )
