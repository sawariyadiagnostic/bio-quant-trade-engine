"""Microstructure capital gatekeeper, exposure limits, and adaptive drawdown sentry."""
from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np

from ravana.risk.config import DrawdownState, RiskConfig


class RiskGatekeeper:
    """Enforces institutional capital thresholds, statutory fee buffers, and drawdown circuits."""

    def __init__(self, config: RiskConfig):
        self.config = config

    def validate_account_capital(self, current_equity: float) -> Tuple[bool, str]:
        """Enforce ₹2,00,000 minimum capital viability gate to prevent statutory fee erosion."""
        if current_equity < self.config.min_capital_gate:
            return (
                False,
                f"Equity ₹{current_equity:,.2f} is below the ₹{self.config.min_capital_gate:,.2f} viability gate.",
            )
        return True, "Capital viability gate cleared."

    def evaluate_drawdown_breaker(
        self, peak_equity: float, current_equity: float
    ) -> Tuple[DrawdownState, float]:
        """Compute portfolio peak-to-trough drawdown and return position scaling multiplier."""
        if peak_equity <= 0:
            return DrawdownState.NORMAL, 1.0

        drawdown = (current_equity - peak_equity) / peak_equity
        abs_dd = abs(min(0.0, drawdown))

        if abs_dd >= self.config.drawdown_halt_threshold:
            return DrawdownState.HALTED, 0.0
        if abs_dd >= self.config.drawdown_halve_threshold:
            return DrawdownState.CAUTION_HALVED, 0.5
        return DrawdownState.NORMAL, 1.0

    def enforce_position_limits(self, weights: np.ndarray) -> np.ndarray:
        """Clamp asset weights to max allocation cap and normalize to gross exposure bound."""
        w = np.clip(weights, 0.0, self.config.max_asset_allocation)
        gross = np.sum(w)
        if gross > self.config.max_gross_exposure:
            w = (w / gross) * self.config.max_gross_exposure
        return w

    def filter_trade_viability(
        self,
        price: float,
        shares: int,
        expected_alpha_bps: float = 10.0,
    ) -> Tuple[bool, int, str]:
        """Reject micro-trades where fixed ₹17 DP fee or statutory drag erodes expected alpha."""
        if price <= 0 or shares <= 0:
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
        total_drag_bps = dp_drag_bps + 2.0  # 2.0 bps statutory drag
        if expected_alpha_bps <= total_drag_bps:
            return (
                False,
                0,
                f"Expected alpha {expected_alpha_bps:.1f} bps is below statutory hurdle {total_drag_bps:.1f} bps.",
            )

        return True, shares, "Trade cleared capital viability and fee drag hurdles."
