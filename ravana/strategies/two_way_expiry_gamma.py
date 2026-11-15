"""RAVANA v4.3: Two-Way Thursday Weekly Expiry Gamma Unpinning Structural Trigger.

Exploits gamma unpinning and delta-hedging squeezes on Thursday index options expiry:
1. Schedule: Active strictly on weekly expiry sessions (Thursday or preceding Wednesday holiday).
2. Time Window: 13:45 to 14:45 IST (Post-lunch option dealer gamma unpinning / squeeze window).
3. Terminal Exit: Mandatory exit at 15:00 IST before 15:15 retail broker auto-liquidation.
4. Two-Way Symmetric Execution:
   - Bullish Unpinning: Close > 6-bar high with Vol >= 1.2x VolMean_20 and OFI > 0.0 -> BUY.
   - Bearish Unpinning: Close < 6-bar low with Vol >= 1.2x VolMean_20 and OFI < 0.0 -> SELL / SHORT.
5. Sizing & Risk: 1.0 ATR stop loss, 1.5 ATR target, 0.25% risk-at-stop capital cap.
"""
from __future__ import annotations

import os
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
    is_weekly_expiry,
)


class TwoWayExpiryGammaTrigger(StructuralTrigger):
    """Trigger: Two-Way Thursday Expiry Gamma Unpinning Breakout (13:45-14:45 IST)."""

    name: str = "TwoWayExpiryGammaTrigger"

    def __init__(
        self,
        lookback_bars: int = 6,
        vol_surge_mult: float = 1.2,
        stop_atr_mult: float = 1.0,
        target_atr_mult: float = 1.5,
    ) -> None:
        self.lookback_bars = int(lookback_bars)
        self.vol_surge_mult = float(vol_surge_mult)
        self.stop_atr_mult = float(stop_atr_mult)
        self.target_atr_mult = float(target_atr_mult)

    def evaluate(self, df: pd.DataFrame, symbol: str) -> pd.Series:
        """Legacy 1D series evaluator."""
        sigs = self.generate_signals(df, symbol)
        conviction = pd.Series(0.0, index=df.index)
        for s in sigs:
            if s.action == TriggerAction.BUY:
                conviction.loc[s.timestamp] = s.conviction
        return conviction

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[TriggerSignal]:
        """Generate causal BUY and SELL signals for Thursday Expiry Gamma unpinning."""
        signals: List[TriggerSignal] = []
        if df.empty or len(df) < 25 or symbol not in ("BANKBEES", "NIFTYBEES", "BANKBEES.NS", "NIFTYBEES.NS"):
            return signals

        r_high = df["High"].shift(1).rolling(self.lookback_bars, min_periods=self.lookback_bars).max()
        r_low = df["Low"].shift(1).rolling(self.lookback_bars, min_periods=self.lookback_bars).min()
        vol_mean = df["Volume"].rolling(20, min_periods=5).mean()
        ofi = extract_ofi_proxy(df)

        daily_entered: Set[Any] = set()

        for i in range(20, len(df)):
            ts = df.index[i]
            t_str = ts.strftime("%H:%M")
            d = ts.date()

            # Pre-15:15 retail dump exit at 15:00 IST
            if t_str == "15:00":
                signals.append(
                    TriggerSignal(
                        timestamp=ts,
                        symbol=symbol,
                        action=TriggerAction.EXIT_ALL,
                        conviction=1.0,
                        trigger_name=self.name,
                        metadata={"reason": "PRE_1515_MIS_SQUARE_OFF"},
                    )
                )
                continue

            # Active strictly on weekly expiry in 13:45 to 14:45 window
            if not is_weekly_expiry(ts) or not ("13:45" <= t_str <= "14:45") or d in daily_entered:
                continue

            curr_close = float(df["Close"].iloc[i])
            curr_vol = float(df["Volume"].iloc[i])
            v_avg = float(vol_mean.iloc[i])
            curr_ofi = float(ofi.iloc[i])

            if np.isnan(r_high.iloc[i]) or np.isnan(r_low.iloc[i]):
                continue

            is_vol_surge = curr_vol >= self.vol_surge_mult * v_avg
            raw_atr = float(df.get("ATR_14", pd.Series(curr_close * 0.01, index=df.index)).iloc[i])
            atr_val = max(raw_atr, curr_close * 0.0035)

            # 1. Bullish Gamma Unpinning -> BUY
            if curr_close > float(r_high.iloc[i]) and is_vol_surge and curr_ofi >= 0.0:
                daily_entered.add(d)
                target = curr_close + (self.target_atr_mult * atr_val)
                stop = curr_close - (self.stop_atr_mult * atr_val)
                signals.append(
                    TriggerSignal(
                        timestamp=ts,
                        symbol=symbol,
                        action=TriggerAction.BUY,
                        conviction=0.80,
                        trigger_name=f"{self.name}_Long",
                        metadata={
                            "side": "LONG",
                            "regime": "EXPIRY_GAMMA_UNPINNING",
                            "target_price": round(target, 2),
                            "stop_loss": round(stop, 2),
                        },
                    )
                )

            # 2. Bearish Gamma Unpinning -> SELL / SHORT
            elif curr_close < float(r_low.iloc[i]) and is_vol_surge and curr_ofi <= 0.0:
                daily_entered.add(d)
                target = curr_close - (self.target_atr_mult * atr_val)
                stop = curr_close + (self.stop_atr_mult * atr_val)
                signals.append(
                    TriggerSignal(
                        timestamp=ts,
                        symbol=symbol,
                        action=TriggerAction.SELL,
                        conviction=0.80,
                        trigger_name=f"{self.name}_Short",
                        metadata={
                            "side": "SHORT",
                            "regime": "EXPIRY_GAMMA_UNPINNING",
                            "target_price": round(target, 2),
                            "stop_loss": round(stop, 2),
                        },
                    )
                )

        return signals
