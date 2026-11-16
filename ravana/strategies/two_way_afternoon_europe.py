"""RAVANA v4.3: Two-Way Afternoon European Open Momentum Structural Trigger.

Exploits European Market Open (DAX/FTSE) liquidity and directional impulse on Indian ETFs:
1. Operational Window: 13:00 to 13:50 IST (Captures European open price discovery).
2. Terminal Exit: Mandatory exit at 14:45 IST strictly ahead of the 15:00-15:15 retail MIS margin panic.
3. Two-Way Symmetric Execution:
   - Bullish Impulse: 5-bar momentum >= +0.15% with OFI > 0.10 -> BUY, target 1.5 ATR, stop 1.0 ATR.
   - Bearish Impulse: 5-bar momentum <= -0.15% with OFI < -0.10 -> SELL / SHORT, target 1.5 ATR, stop 1.0 ATR.
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
from ravana.strategies.triggers import extract_ofi_proxy, extract_session_vwap


class TwoWayAfternoonEuropeTrigger(StructuralTrigger):
    """Trigger: Two-Way European Open Momentum Impulse (13:00-13:50 IST)."""

    name: str = "TwoWayAfternoonEuropeTrigger"

    def __init__(
        self,
        momentum_threshold: float = 0.0015,
        ofi_threshold: float = 0.10,
        stop_atr_mult: float = 1.0,
        target_atr_mult: float = 1.5,
    ) -> None:
        self.momentum_threshold = float(momentum_threshold)
        self.ofi_threshold = float(ofi_threshold)
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
        """Generate causal BUY and SELL signals for European Open momentum."""
        signals: List[TriggerSignal] = []
        if df.empty or len(df) < 20:
            return signals

        log_ret_5 = np.log(df["Close"] / df["Close"].shift(5).replace(0.0, np.nan)).fillna(0.0)
        ofi = extract_ofi_proxy(df)
        session_vwap = extract_session_vwap(df)
        daily_entered: Set[Any] = set()

        for i in range(5, len(df)):
            ts = df.index[i]
            t_str = ts.strftime("%H:%M")
            d = ts.date()

            # Terminal pre-panic exit at 14:45 IST
            if t_str == "14:45":
                signals.append(
                    TriggerSignal(
                        timestamp=ts,
                        symbol=symbol,
                        action=TriggerAction.EXIT_ALL,
                        conviction=1.0,
                        trigger_name=self.name,
                        metadata={"reason": "PRE_1500_RETAIL_DUMP_EXIT"},
                    )
                )
                continue

            # Active entry window: 13:00 to 13:50 IST
            if not ("13:00" <= t_str <= "13:50") or d in daily_entered:
                continue

            ret5 = float(log_ret_5.iloc[i])
            curr_ofi = float(ofi.iloc[i])
            curr_close = float(df["Close"].iloc[i])
            raw_atr = float(df.get("ATR_14", pd.Series(curr_close * 0.01, index=df.index)).iloc[i])
            atr_val = max(raw_atr, curr_close * 0.0035)

            curr_vwap = float(session_vwap.iloc[i])

            # 1. Bullish EU Open Momentum -> BUY (Must be above VWAP)
            if ret5 >= self.momentum_threshold and curr_ofi >= self.ofi_threshold and curr_close > curr_vwap:
                daily_entered.add(d)
                target = curr_close + (self.target_atr_mult * atr_val)
                stop = curr_close - (self.stop_atr_mult * atr_val)
                signals.append(
                    TriggerSignal(
                        timestamp=ts,
                        symbol=symbol,
                        action=TriggerAction.BUY,
                        conviction=0.75,
                        trigger_name=f"{self.name}_Long",
                        metadata={
                            "side": "LONG",
                            "target_price": round(target, 2),
                            "stop_loss": round(stop, 2),
                        },
                    )
                )

            # 2. Bearish EU Open Momentum -> SELL / SHORT (Must be below VWAP)
            elif ret5 <= -self.momentum_threshold and curr_ofi <= -self.ofi_threshold and curr_close < curr_vwap:
                daily_entered.add(d)
                target = curr_close - (self.target_atr_mult * atr_val)
                stop = curr_close + (self.stop_atr_mult * atr_val)
                signals.append(
                    TriggerSignal(
                        timestamp=ts,
                        symbol=symbol,
                        action=TriggerAction.SELL,
                        conviction=0.75,
                        trigger_name=f"{self.name}_Short",
                        metadata={
                            "side": "SHORT",
                            "target_price": round(target, 2),
                            "stop_loss": round(stop, 2),
                        },
                    )
                )

        return signals
