"""
All-Weather Overnight Bear Market Inverse & Flight-to-Safety Allocator for RAVANA v4.3.
Provides:
1. Macro Regime Detection: Bull vs Bear Regime based on NIFTY 50 EMA & Variance Ratio.
2. Bull Allocation: Top 2-3 Equity Sector ETF Champions (capped at 35%).
3. Bear Allocation (The Option to Sell):
   - Flight-to-Safety: GOLDBEES Allocation (up to 35%).
   - Synthetic Inverse Short Index Hedge: Delta hedge capturing profit when Nifty drops.
   - LIQUIDBEES: Risk-Free Overnight Cash Yield on unallocated capital.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OvernightAllocationTarget:
    """Target asset allocation recommended for overnight hold."""

    ticker: str
    target_weight_pct: float
    side: str  # 'LONG' or 'INVERSE_SHORT'
    asset_class: str  # 'EQUITY_ETF', 'COMMODITY', 'INVERSE_HEDGE', 'LIQUID_CASH'
    rationale: str


class AllWeatherOvernightAllocator:
    """Allocates capital dynamically across Bull Momentum, Gold Flight-to-Safety, and Inverse Hedges."""

    def __init__(
        self,
        max_equity_cap_pct: float = 0.35,
        max_gold_cap_pct: float = 0.35,
        max_inverse_hedge_pct: float = 0.35,
    ) -> None:
        self.max_equity_cap_pct = float(max_equity_cap_pct)
        self.max_gold_cap_pct = float(max_gold_cap_pct)
        self.max_inverse_hedge_pct = float(max_inverse_hedge_pct)

    def determine_regime(self, bench_df: pd.DataFrame, current_date: Any) -> Dict[str, Any]:
        """Classify current macro regime based on NIFTY 50-day EMA and Variance Ratio."""
        sub = bench_df[bench_df.index.date <= current_date]
        if sub.empty or sub.index[-1].date() != current_date:
            return {"regime": "NEUTRAL", "is_bear": False, "is_bull": False}

        bar = sub.iloc[-1]
        close = float(bar["Close"])
        ema_50 = float(bar.get("EMA_50", close))
        vr_20 = float(bar.get("VR_20", 1.0))

        # Bear Market Markdown: Below 50 EMA and Variance Ratio < 0.95
        is_bear = (close < ema_50 * 0.99) and (vr_20 < 0.95)
        # Bull Market Markup: Above 50 EMA and Variance Ratio >= 0.95
        is_bull = (close >= ema_50) and (vr_20 >= 0.95)

        regime_str = "BEAR_MARKDOWN" if is_bear else ("BULL_MARKUP" if is_bull else "CHOPPY_SIDEWAYS")
        return {
            "regime": regime_str,
            "is_bear": is_bear,
            "is_bull": is_bull,
            "close": close,
            "ema_50": ema_50,
            "vr_20": vr_20,
        }

    def compute_allocations(
        self,
        regime_info: Dict[str, Any],
        ranked_sectors: List[Tuple[str, float, float]],
    ) -> List[OvernightAllocationTarget]:
        """Compute all-weather target weights based on active regime."""
        targets: List[OvernightAllocationTarget] = []

        if regime_info["is_bear"]:
            # -------------------------------------------------------------
            # BEAR REGIME: ACTIVATE THE OPTION TO SELL / INVERSE PROFIT
            # -------------------------------------------------------------
            # 1. Flight-to-Safety: GOLDBEES (rallies when equities drop)
            targets.append(
                OvernightAllocationTarget(
                    ticker="GOLDBEES",
                    target_weight_pct=self.max_gold_cap_pct,
                    side="LONG",
                    asset_class="COMMODITY",
                    rationale="Flight-to-Safety commodity momentum during equity markdown",
                )
            )
            # 2. Synthetic Inverse Short Hedge (Captures profit from falling index)
            targets.append(
                OvernightAllocationTarget(
                    ticker="INVERSE_NIFTY",
                    target_weight_pct=self.max_inverse_hedge_pct,
                    side="INVERSE_SHORT",
                    asset_class="INVERSE_HEDGE",
                    rationale="Synthetic inverse short delta hedge capturing market drop profits",
                )
            )
            # 3. Residual in LIQUIDBEES
            cash_weight = max(0.0, 1.0 - (self.max_gold_cap_pct + self.max_inverse_hedge_pct))
            targets.append(
                OvernightAllocationTarget(
                    ticker="LIQUIDBEES",
                    target_weight_pct=round(cash_weight, 2),
                    side="LONG",
                    asset_class="LIQUID_CASH",
                    rationale="Risk-free overnight cash yield on unallocated margin",
                )
            )
        else:
            # -------------------------------------------------------------
            # BULL / NEUTRAL REGIME: ALLOCATE TO TOP EQUITY SECTORS
            # -------------------------------------------------------------
            equity_candidates = [
                (sym, px, score)
                for sym, px, score in ranked_sectors
                if sym not in ("GOLDBEES", "SILVERBEES")
            ]
            top_champions = equity_candidates[:2]
            allocated_weight = 0.0

            for sym, px, score in top_champions:
                targets.append(
                    OvernightAllocationTarget(
                        ticker=sym,
                        target_weight_pct=self.max_equity_cap_pct,
                        side="LONG",
                        asset_class="EQUITY_ETF",
                        rationale=f"Bullish sector momentum champion (score={score:.2f})",
                    )
                )
                allocated_weight += self.max_equity_cap_pct

            cash_weight = max(0.0, 1.0 - allocated_weight)
            if cash_weight > 0.05:
                targets.append(
                    OvernightAllocationTarget(
                        ticker="LIQUIDBEES",
                        target_weight_pct=round(cash_weight, 2),
                        side="LONG",
                        asset_class="LIQUID_CASH",
                        rationale="Overnight cash yield buffer",
                    )
                )

        return targets
