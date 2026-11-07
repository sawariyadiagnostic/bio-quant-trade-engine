"""Synthetic Delta Hedge Manager for NSE Lower-Circuit Evaporation Protection.

When an asset is locked in lower circuit, regular SELL MKT orders are rejected by
the exchange because there are zero buyers. This module detects circuit proximity
or bid evaporation and calculates a beta-neutralizing synthetic delta short on
a liquid index proxy (e.g., NIFTYBEES or NIFTY Futures) to protect against further downside.
"""
from __future__ import annotations
import math
from typing import Any
from loguru import logger


class SyntheticDeltaHedgeManager:
    """Manages lower-circuit detection and synthetic delta hedging."""

    CIRCUIT_PROXIMITY_THRESHOLD: float = 0.0015  # 15 bps (0.15%) to circuit band

    def is_circuit_locked(
        self,
        current_price: float,
        lower_circuit: float,
        bid_depth: int = 100
    ) -> bool:
        """Determines if the asset is effectively locked at lower circuit.

        Trigger conditions:
        1. Current price is within 15 bps of the lower circuit band.
        2. Top-of-book bid depth has evaporated to zero.
        """
        if bid_depth <= 0:
            logger.warning(f"[SyntheticHedge] Evaporation detected: bid depth={bid_depth}")
            return True

        if lower_circuit > 0.0 and current_price > 0.0:
            distance = (current_price - lower_circuit) / lower_circuit
            if distance <= self.CIRCUIT_PROXIMITY_THRESHOLD:
                logger.warning(
                    f"[SyntheticHedge] Lower circuit proximity lock: dist={distance:.4%}, "
                    f"threshold={self.CIRCUIT_PROXIMITY_THRESHOLD:.4%}"
                )
                return True

        return False

    def calculate_hedge_quantity(
        self,
        portfolio_positions: list[dict[str, Any]],
        hedge_price: float,
        hedge_beta: float = 1.0
    ) -> int:
        """Calculates beta-weighted hedge quantity for portfolio immunization.

        Formula:
            Total Beta Notional = sum(qty_k * price_k * beta_k)
            Hedge Units = int(Total Beta Notional / (hedge_price * hedge_beta))
        """
        if hedge_price <= 0.0 or hedge_beta <= 0.0:
            return 0

        total_beta_notional = 0.0
        for pos in portfolio_positions:
            qty = pos.get("quantity", 0)
            price = pos.get("price", 0.0)
            beta = pos.get("beta", 1.0)
            if qty > 0 and price > 0.0:
                total_beta_notional += float(qty * price * beta)

        if total_beta_notional <= 0.0:
            return 0

        hedge_qty = int(math.floor(total_beta_notional / (hedge_price * hedge_beta)))
        logger.info(
            f"[SyntheticHedge] Hedging beta notional ₹{total_beta_notional:,.2f} "
            f"via {hedge_qty} units @ ₹{hedge_price:.2f}"
        )
        return max(0, hedge_qty)

    def create_hedge_order(
        self,
        total_qty: int,
        hedge_symbol: str = "NIFTYBEES"
    ) -> dict[str, Any]:
        """Constructs the short market order for delta neutralization."""
        return {
            "symbol": hedge_symbol,
            "action": "SELL",
            "quantity": int(total_qty),
            "order_type": "MKT",
            "price": 0.0,
            "product": "MIS",
            "is_etf": "BEES" in hedge_symbol.upper(),
            "reason": "SYNTHETIC_DELTA_HEDGE_LOWER_CIRCUIT"
        }
