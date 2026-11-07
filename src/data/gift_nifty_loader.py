"""
GIFT NIFTY Overnight Lead-Lag Ingestion Engine (Wave 6 / P2: Task 10).
Ingests early-morning GIFT City IFSC Nifty 50 Futures (trading 06:30–08:30 IST)
to determine the implied opening gap and order bias for Indian cash ETFs (NIFTYBEES).
"""
from typing import Dict, Any, Optional
import numpy as np


class GiftNiftyLoader:
    """Pre-market GIFT NIFTY lead-lag gap calculator."""
    def __init__(self, circuit_limit_pct: float = 0.10, threshold_bps: float = 20.0):
        self.circuit_limit_pct = circuit_limit_pct
        self.threshold_pct = threshold_bps / 10000.0

    def calculate_implied_gap(
        self,
        gift_nifty_price: float,
        nse_prev_close: float
    ) -> Dict[str, Any]:
        """
        Computes statutory implied opening gap between GIFT Nifty and prior NSE close.
        Returns signal direction and circuit clipping tripwire.
        """
        if nse_prev_close <= 0.0 or not np.isfinite(nse_prev_close):
            raise ValueError(f"Invalid nse_prev_close: {nse_prev_close}")
        if gift_nifty_price <= 0.0 or not np.isfinite(gift_nifty_price):
            raise ValueError(f"Invalid gift_nifty_price: {gift_nifty_price}")

        raw_gap = (gift_nifty_price - nse_prev_close) / nse_prev_close
        clipped_gap = float(np.clip(raw_gap, -self.circuit_limit_pct, self.circuit_limit_pct))
        circuit_clipped = bool(abs(raw_gap - clipped_gap) > 1e-6)

        if clipped_gap > self.threshold_pct:
            signal = "BULLISH_GAP"
        elif clipped_gap < -self.threshold_pct:
            signal = "BEARISH_GAP"
        else:
            signal = "NEUTRAL"

        return {
            "gift_nifty_price": float(gift_nifty_price),
            "nse_prev_close": float(nse_prev_close),
            "raw_gap_pct": float(raw_gap),
            "implied_gap_pct": clipped_gap,
            "signal": signal,
            "circuit_clipped": circuit_clipped
        }
