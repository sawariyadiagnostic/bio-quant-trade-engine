"""Microstructure capital gatekeeper, exposure limits, and adaptive drawdown sentry."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np

from ravana.risk.config import DrawdownState, RiskConfig


class RiskGatekeeper:
    """Enforces institutional capital thresholds, statutory fee buffers, and drawdown circuits."""

    def __init__(self, config: RiskConfig):
        self.config = config
        self.peak_equity = 0.0
        self.state = DrawdownState.NORMAL

    def validate_account_capital(self, current_equity: float) -> Tuple[bool, str]:
        """Enforce ₹2,00,000 minimum capital viability gate to prevent statutory fee erosion."""
        if not np.isfinite(current_equity) or current_equity < self.config.min_capital_gate:
            return (
                False,
                f"Equity ₹{current_equity} is invalid or below ₹{self.config.min_capital_gate:,.2f} viability gate.",
            )
        return True, "Capital viability gate cleared."

    def evaluate_drawdown_breaker(
        self, peak_equity: float, current_equity: float, force_unhalt: bool = False
    ) -> Tuple[DrawdownState, float]:
        """Compute portfolio drawdown with hysteresis deadband to prevent whipsaws and deadlocks."""
        if not np.isfinite(current_equity) or current_equity <= 0:
            return DrawdownState.HALTED, 0.0

        if peak_equity > self.peak_equity:
            self.peak_equity = peak_equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        effective_peak = max(self.peak_equity, peak_equity)
        if effective_peak <= 0:
            return DrawdownState.NORMAL, 1.0

        drawdown = (current_equity - effective_peak) / effective_peak
        abs_dd = abs(min(0.0, drawdown))

        # Handle unhalt override
        if force_unhalt and self.state == DrawdownState.HALTED:
            self.peak_equity = current_equity
            self.state = DrawdownState.CAUTION_HALVED
            return self.state, 0.5

        if self.state == DrawdownState.HALTED:
            return DrawdownState.HALTED, 0.0

        if self.state == DrawdownState.CAUTION_HALVED:
            # Hysteresis deadband: recover to NORMAL only when drawdown recovers below 2.5%
            if abs_dd < 0.025:
                self.state = DrawdownState.NORMAL
            elif abs_dd >= self.config.drawdown_halt_threshold:
                self.state = DrawdownState.HALTED
        else:
            if abs_dd >= self.config.drawdown_halt_threshold:
                self.state = DrawdownState.HALTED
            elif abs_dd >= self.config.drawdown_halve_threshold:
                self.state = DrawdownState.CAUTION_HALVED

        mult = 1.0 if self.state == DrawdownState.NORMAL else (0.5 if self.state == DrawdownState.CAUTION_HALVED else 0.0)
        return self.state, mult

    def enforce_position_limits(self, weights: np.ndarray) -> np.ndarray:
        """Water-filling cap enforcement: redistributes excess weights to uncapped assets."""
        if len(weights) == 0:
            return weights

        target_gross = min(float(np.sum(weights)), self.config.max_gross_exposure)
        w = np.copy(weights)
        max_cap = self.config.max_asset_allocation

        for _ in range(10):
            capped = w >= max_cap
            if not np.any(capped) or np.all(capped):
                break
            w[capped] = max_cap
            uncapped_sum = float(np.sum(w[~capped]))
            excess = target_gross - float(np.sum(capped) * max_cap)
            if uncapped_sum > 0 and excess > 0:
                w[~capped] = (w[~capped] / uncapped_sum) * excess
            else:
                break

        gross = float(np.sum(w))
        return (w / gross) * target_gross if gross > 0 else w

    def filter_trade_viability(
        self,
        price: float,
        shares: int,
        expected_alpha_bps: float = 15.0,
    ) -> Tuple[bool, int, str]:
        """Reject micro-trades where fixed ₹17 DP fee or statutory drag erodes expected alpha."""
        if not np.isfinite(price) or not np.isfinite(shares) or price <= 0 or shares <= 0:
            return False, 0, "Non-positive price or shares."

        notional = float(price * shares)
        if notional < self.config.min_trade_notional_floor:
            return (
                False,
                0,
                f"Notional ₹{notional:,.2f} below ₹{self.config.min_trade_notional_floor:,.2f} floor "
                f"(₹17 DP fee would exceed 10 bps drag).",
            )

        dp_drag_bps = (17.0 / notional) * 10000.0
        total_drag_bps = dp_drag_bps + 2.0  # 2.0 bps exchange & statutory drag
        if expected_alpha_bps < total_drag_bps:
            return (
                False,
                0,
                f"Expected alpha {expected_alpha_bps:.1f} bps is below statutory hurdle {total_drag_bps:.1f} bps.",
            )

        return True, shares, "Trade cleared capital viability and fee drag hurdles."
