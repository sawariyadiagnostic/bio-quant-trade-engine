"""Two-Way ETF iNAV Tracking Spread Mean-Reversion Trigger for RAVANA v4.3.

Captures statistical arbitrage convergence between ETF market price and intrinsic iNAV:
1. Long Discount: Buys when ETF trades at discount (Spread Z <= -1.5, BPS <= -15.0)
   AND underlying index basket is stable (not making 3-bar lower lows).
2. Short Premium (The Option to Sell): Sells short (MIS) when ETF trades at premium
   (Spread Z >= +1.5, BPS >= +15.0) AND underlying index basket is not breaking 3-bar highs.
3. Strict Trend Regime: Only active when VR_20 < 0.95 (mean-reverting chop).
4. High-Liquidity Morning Window: 09:35 to 10:45 IST with mandatory 11:30 AM lock.
5. Strict Point-in-Time Causality (Zero Lookahead / No n+1 Leakage).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set
import numpy as np
import pandas as pd

from ravana.strategies.base import (
    StructuralTrigger,
    TriggerAction,
    TriggerSignal,
)
from ravana.strategies.triggers import (
    extract_ofi_proxy,
    extract_variance_ratio,
)


class TwoWayINAVSpreadTrigger(StructuralTrigger):
    """Trigger: Two-Way ETF iNAV Tracking Spread Mean-Reversion (Discount Buy & Premium Fade)."""

    name: str = "TwoWayINAVSpreadTrigger"

    def __init__(
        self,
        z_threshold: float = 1.5,
        bps_threshold: float = 15.0,
        vr_ceiling: float = 0.95,
    ) -> None:
        self.z_threshold = float(z_threshold)
        self.bps_threshold = float(bps_threshold)
        self.vr_ceiling = float(vr_ceiling)

    def evaluate(self, df: pd.DataFrame, symbol: str) -> pd.Series:
        """Legacy 1D series evaluator returning buy conviction."""
        sigs = self.generate_signals(df, symbol)
        conviction = pd.Series(0.0, index=df.index)
        for s in sigs:
            if s.action == TriggerAction.BUY:
                conviction.loc[s.timestamp] = s.conviction
        return conviction

    # Restrict to liquid Tier-1 ETFs with tight tracking spreads (spread std <= 15 bps)
    # Excludes JUNIORBEES (std 212 bps) and CPSEETF (std 262 bps) due to AP creation friction/dividends
    ELIGIBLE_SYMBOLS: Set[str] = {"NIFTYBEES", "PHARMABEES", "ITBEES", "AUTOBEES", "BANKBEES"}

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[TriggerSignal]:
        """Generate causal BUY and SELL signals for ETF iNAV tracking spread dislocations."""
        signals: List[TriggerSignal] = []
        if df.empty or len(df) < 20 or symbol not in self.ELIGIBLE_SYMBOLS:
            return signals

        if "Theoretical_Close" not in df.columns or "SPREAD_BPS" not in df.columns:
            return signals

        vr = extract_variance_ratio(df)
        ofi = extract_ofi_proxy(df)

        daily_entered: Set[Any] = set()

        for i in range(3, len(df)):
            ts = df.index[i]
            t_str = ts.strftime("%H:%M")
            d = ts.normalize()

            # Mandatory morning exit at 11:30 IST
            if t_str == "11:30":
                signals.append(
                    TriggerSignal(
                        timestamp=ts,
                        symbol=symbol,
                        action=TriggerAction.EXIT_ALL,
                        conviction=1.0,
                        trigger_name=self.name,
                        metadata={"reason": "MANDATORY_1130_MORNING_LOCK"},
                    )
                )
                continue

            # Active strictly in high-liquidity morning window: 09:35 to 10:45 IST
            if not ("09:35" <= t_str <= "10:45") or d in daily_entered:
                continue

            curr_vr = float(vr.iloc[i])
            # Only trade spread mean reversion when regime is mean-reverting
            if curr_vr >= self.vr_ceiling:
                continue

            c_px = float(df["Close"].iloc[i])
            theo_px = float(df["Theoretical_Close"].iloc[i])
            spread_bps = float(df["SPREAD_BPS"].iloc[i])
            spread_z = float(df.get("SPREAD_ZSCORE_20", pd.Series(0.0, index=df.index)).iloc[i])
            curr_ofi = float(ofi.iloc[i])
            raw_atr = float(df.get("ATR_14", pd.Series(c_px * 0.01, index=df.index)).iloc[i])
            atr_val = max(raw_atr, c_px * 0.0035)

            # Underlying benchmark index stability check over last 3 bars
            theo_low_3 = float(df["Theoretical_Close"].iloc[i - 3 : i].min())
            theo_high_3 = float(df["Theoretical_Close"].iloc[i - 3 : i].max())

            # 1. LONG: Discount Buy (Spread Z <= -1.5, BPS <= -15), underlying stable
            if (spread_z <= -self.z_threshold) and (spread_bps <= -self.bps_threshold) and (theo_px >= theo_low_3) and (curr_ofi >= 0.0):
                daily_entered.add(d)
                target = theo_px
                stop = c_px - (1.0 * atr_val)
                signals.append(
                    TriggerSignal(
                        timestamp=ts,
                        symbol=symbol,
                        action=TriggerAction.BUY,
                        conviction=0.80,
                        trigger_name=f"{self.name}_DiscountBuy",
                        metadata={"side": "LONG", "target_price": round(target, 2), "stop_loss": round(stop, 2), "spread_bps": spread_bps},
                    )
                )

            # 2. SHORT: Premium Fade (Spread Z >= +1.5, BPS >= +15), underlying stable
            elif (spread_z >= self.z_threshold) and (spread_bps >= self.bps_threshold) and (theo_px <= theo_high_3) and (curr_ofi <= 0.0):
                daily_entered.add(d)
                target = theo_px
                stop = c_px + (1.0 * atr_val)
                signals.append(
                    TriggerSignal(
                        timestamp=ts,
                        symbol=symbol,
                        action=TriggerAction.SELL,
                        conviction=0.80,
                        trigger_name=f"{self.name}_PremiumFade",
                        metadata={"side": "SHORT", "target_price": round(target, 2), "stop_loss": round(stop, 2), "spread_bps": spread_bps},
                    )
                )

        return signals
