"""Muscle 17: Dynamic Volatility Sizing Engine.

Chains Muscle 12 (AllocationResult), Muscle 7 (realized volatility features),
and Phase 5 (ValidationCertificate) to apply inverse-volatility targeting,
DSR-adjusted fractional Kelly scaling, lot-size rounding, and cash reserve enforcement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

from ravana.risk.config import RiskConfig
from ravana.risk.engine import AllocationResult
from ravana.validation.config import ValidationCertificate


@dataclass(frozen=True)
class VolatilitySizedAllocation:
    """Output contract of Muscle 17 feeding into Muscle 18 and Muscle 19."""

    base_allocation: AllocationResult
    sized_weights: Dict[str, float]
    sized_shares: Dict[str, int]
    sized_notionals: Dict[str, float]
    vol_scalars: Dict[str, float]
    kelly_fraction_applied: float
    cash_reserve_inr: float
    total_deployed_inr: float


class DynamicVolatilitySizer:
    """Applies target-volatility scaling and fractional Kelly sizing to HRP allocations."""

    def __init__(
        self,
        risk_config: Optional[RiskConfig] = None,
        kelly_fraction: float = 0.50,  # Half-Kelly institutional ceiling
        min_cash_reserve_pct: float = 0.05,
    ) -> None:
        self.config = risk_config or RiskConfig()
        self.kelly_fraction = float(np.clip(kelly_fraction, 0.05, 1.0))
        self.min_cash_reserve_pct = float(np.clip(min_cash_reserve_pct, 0.0, 0.50))

    def size_allocation(
        self,
        allocation: AllocationResult,
        current_prices: Dict[str, float],
        realized_annual_vols: Dict[str, float],
        current_equity: float,
        certificate: Optional[ValidationCertificate] = None,
        lot_sizes: Optional[Dict[str, int]] = None,
    ) -> VolatilitySizedAllocation:
        """Scale Muscle 12 base weights by asset volatility, DSR confidence, and lot size constraints."""
        lot_sizes = lot_sizes or {}
        if not allocation.target_weights or current_equity < self.config.min_capital_gate:
            return VolatilitySizedAllocation(
                base_allocation=allocation,
                sized_weights={},
                sized_shares={},
                sized_notionals={},
                vol_scalars={},
                kelly_fraction_applied=0.0,
                cash_reserve_inr=round(max(0.0, current_equity), 2),
                total_deployed_inr=0.0,
            )

        kelly_mult = self._compute_kelly_multiplier(certificate)
        max_deployable_equity = current_equity * (1.0 - self.min_cash_reserve_pct)
        target_vol = max(0.01, self.config.target_annual_vol)

        vol_scalars: Dict[str, float] = {}
        raw_scaled_weights: Dict[str, float] = {}

        for sym, base_w in allocation.target_weights.items():
            asset_vol = max(0.02, realized_annual_vols.get(sym, target_vol))
            v_scalar = float(np.clip(target_vol / asset_vol, 0.25, 1.50))
            vol_scalars[sym] = round(v_scalar, 4)
            scaled_w = min(self.config.max_asset_allocation, base_w * v_scalar * kelly_mult)
            raw_scaled_weights[sym] = scaled_w

        total_w = sum(raw_scaled_weights.values())
        max_gross = min(self.config.max_gross_exposure, 1.0 - self.min_cash_reserve_pct)
        if total_w > max_gross and total_w > 0:
            scale_down = max_gross / total_w
            for sym in raw_scaled_weights:
                raw_scaled_weights[sym] *= scale_down

        sized_weights: Dict[str, float] = {}
        sized_shares: Dict[str, int] = {}
        sized_notionals: Dict[str, float] = {}
        total_deployed = 0.0

        for sym, w in raw_scaled_weights.items():
            px = current_prices.get(sym, 0.0)
            if px <= 0 or not np.isfinite(px):
                continue
            notional = min(current_equity * w, max_deployable_equity - total_deployed)
            lot = max(1, lot_sizes.get(sym, 1))
            raw_shares = int(notional // px)
            shares = (raw_shares // lot) * lot
            final_notional = shares * px

            if final_notional >= self.config.min_trade_notional_floor and shares > 0:
                sized_weights[sym] = round(final_notional / current_equity, 4)
                sized_shares[sym] = shares
                sized_notionals[sym] = round(final_notional, 2)
                total_deployed += final_notional

        cash_reserve = max(0.0, current_equity - total_deployed)
        return VolatilitySizedAllocation(
            base_allocation=allocation,
            sized_weights=sized_weights,
            sized_shares=sized_shares,
            sized_notionals=sized_notionals,
            vol_scalars=vol_scalars,
            kelly_fraction_applied=round(kelly_mult, 4),
            cash_reserve_inr=round(cash_reserve, 2),
            total_deployed_inr=round(total_deployed, 2),
        )

    def _compute_kelly_multiplier(self, cert: Optional[ValidationCertificate]) -> float:
        """Compute conviction scalar from Phase 5 DSR and Walk-Forward Sharpe."""
        if cert is None:
            return self.kelly_fraction
        dsr_conf = float(np.clip(cert.dsr_guard.deflated_sharpe_ratio, 0.0, 1.0))
        oos_sr = max(0.0, cert.walk_forward.aggregate_oos_sharpe)
        sr_factor = float(np.clip(oos_sr / 2.0, 0.25, 1.25))
        return float(np.clip(self.kelly_fraction * dsr_conf * sr_factor, 0.10, 1.0))
