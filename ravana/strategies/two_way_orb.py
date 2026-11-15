"""
Two-Way Morning Opening Range Breakout (ORB) Trigger for RAVANA v4.3.
Supports both Long (BUY) and Short (SELL MIS) trades in the 09:35–11:00 IST morning window,
with order flow confirmation, dynamic conviction scoring, and a mandatory 11:30 AM lock.
"""
from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from ravana.strategies.base import (
    StructuralTrigger,
    TriggerAction,
    TriggerSignal,
)
from ravana.strategies.triggers import (
    extract_ofi_proxy,
    extract_session_vwap,
    extract_variance_ratio,
)


class TwoWayMorningORBTrigger(StructuralTrigger):
    """Trigger: Two-Way Opening Range Breakout (Long & Short) in 09:35–11:00 IST morning window.
    
    Upgrades:
    1. Fresh-cross only: triggers only on the exact 5m candle that first crosses OR_high or OR_low.
    2. Zero chasing: prevents late re-entries on already extended moves.
    3. Strict Trend Regime: requires VR_20 >= 0.95 (filters out AP mean-reverting chop).
    4. Commodity Shield: ignores commodity ETFs (Gold/Silver) which do not follow equity ORB.
    5. Symmetric 1:1.25 Risk-to-Reward: Target at 1.5 ATR, Stop at 1.2 ATR.
    """

    name: str = "TwoWayMorningORBTrigger"

    def __init__(self, ofi_threshold: float = 0.15, vr_threshold: float = 0.95) -> None:
        self.ofi_threshold = float(ofi_threshold)
        self.vr_threshold = float(vr_threshold)

    def evaluate(self, df: pd.DataFrame, symbol: str) -> pd.Series:
        """Legacy 1D series evaluator returning buy conviction."""
        sigs = self.generate_two_way_signals(df, symbol)
        conviction = pd.Series(0.0, index=df.index)
        for s in sigs:
            if s.action == TriggerAction.BUY:
                conviction.loc[s.timestamp] = s.conviction
        return conviction

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[TriggerSignal]:
        """Implement StructuralTrigger abstract contract."""
        return self.generate_two_way_signals(df, symbol)

    def generate_two_way_signals(self, df: pd.DataFrame, symbol: str) -> List[TriggerSignal]:
        """Generate point-in-time causal BUY and SELL signals for morning 5m bars."""
        signals: List[TriggerSignal] = []
        if df.empty or len(df) < 5:
            return signals

        # Exclude commodities from equity ORB
        if symbol in ("GOLDBEES", "SILVERBEES"):
            return signals

        vr = extract_variance_ratio(df)
        ofi = extract_ofi_proxy(df)
        vwap = extract_session_vwap(df)

        daily_or: Dict[Any, Dict[str, float]] = {}
        dates = df.index.normalize()

        for d in set(dates):
            day_bars = df[dates == d]
            # 15-minute Opening Range: 09:15, 09:20, 09:25 bars
            or_bars = day_bars[(day_bars.index.strftime("%H:%M") >= "09:15") & (day_bars.index.strftime("%H:%M") <= "09:25")]
            if len(or_bars) >= 2:
                daily_or[d] = {
                    "high": float(or_bars["High"].max()),
                    "low": float(or_bars["Low"].min()),
                }

        # Track daily trigger execution to prevent multiple re-entries per day per symbol
        daily_entered: Set[Any] = set()

        for i in range(len(df)):
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

            # Only enter in high-liquidity morning window: 09:35 to 10:45 IST
            if not ("09:35" <= t_str <= "10:45"):
                continue

            if d not in daily_or or d in daily_entered:
                continue

            or_high = daily_or[d]["high"]
            or_low = daily_or[d]["low"]
            curr_vr = float(vr.iloc[i])
            curr_ofi = float(ofi.iloc[i])
            curr_close = float(df["Close"].iloc[i])
            prev_close = float(df["Close"].iloc[i - 1]) if i > 0 else curr_close
            curr_vwap = float(vwap.iloc[i]) if vwap is not None else curr_close
            raw_atr = float(df.get("ATR_14", pd.Series(curr_close * 0.01, index=df.index)).iloc[i])
            atr_val = max(raw_atr, curr_close * 0.0035)

            # Strict trend regime filter: only trade breakouts when VR indicates momentum
            if curr_vr < self.vr_threshold:
                continue

            # 1. LONG / BUY BREAKOUT (Fresh Cross Only)
            # Previous bar was inside/below OR_high, current bar closes cleanly above OR_high
            if (prev_close <= or_high) and (curr_close > or_high * 1.0002) and (curr_ofi >= self.ofi_threshold) and (curr_close >= curr_vwap):
                daily_entered.add(d)
                base_conv = 0.70 + 0.15 * min(1.0, (curr_vr - 0.95) / 0.35) + 0.15 * min(1.0, curr_ofi)
                target = curr_close + (1.5 * atr_val)
                stop = curr_close - (1.2 * atr_val)
                signals.append(
                    TriggerSignal(
                        timestamp=ts,
                        symbol=symbol,
                        action=TriggerAction.BUY,
                        conviction=round(min(0.95, base_conv), 3),
                        trigger_name=f"{self.name}_Long",
                        metadata={"or_high": or_high, "ofi": curr_ofi, "side": "LONG", "target_price": round(target, 2), "stop_loss": round(stop, 2)},
                    )
                )

            # 2. SHORT / SELL BREAKDOWN (Fresh Cross Only)
            # Previous bar was inside/above OR_low, current bar closes cleanly below OR_low
            elif (prev_close >= or_low) and (curr_close < or_low * 0.9998) and (curr_ofi <= -self.ofi_threshold) and (curr_close <= curr_vwap):
                daily_entered.add(d)
                base_conv = 0.70 + 0.15 * min(1.0, (curr_vr - 0.95) / 0.35) + 0.15 * min(1.0, abs(curr_ofi))
                target = curr_close - (1.5 * atr_val)
                stop = curr_close + (1.2 * atr_val)
                signals.append(
                    TriggerSignal(
                        timestamp=ts,
                        symbol=symbol,
                        action=TriggerAction.SELL,
                        conviction=round(min(0.95, base_conv), 3),
                        trigger_name=f"{self.name}_Short",
                        metadata={"or_low": or_low, "ofi": curr_ofi, "side": "SHORT", "target_price": round(target, 2), "stop_loss": round(stop, 2)},
                    )
                )

        return signals
