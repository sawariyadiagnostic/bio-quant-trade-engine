"""Two-Way Absorption and Liquidity Sweep Triggers for RAVANA v4.3.

Includes:
1. TwoWayVWAPAbsorptionTrigger: Fades extreme Bollinger Band extensions back to VWAP
   with strict Variance Ratio (VR < 0.90) gating and asymmetric 1:1.5 Risk-to-Reward.
2. TwoWayLiquidityStopSweepTrigger: Capitalizes on retail stop-hunts at 10-bar swing
   highs/lows with wick rejection and order flow confirmation.
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
    extract_session_vwap,
    extract_variance_ratio,
)


class TwoWayVWAPAbsorptionTrigger(StructuralTrigger):
    """Trigger: Fades Bollinger Band extremes back to VWAP in mean-reverting regimes (VR < 0.90)."""

    name: str = "TwoWayVWAPAbsorptionTrigger"

    def __init__(
        self,
        bb_lower: float = 0.18,
        bb_upper: float = 0.82,
        ofi_threshold: float = 0.12,
        vr_ceiling: float = 0.90,
    ) -> None:
        self.bb_lower = float(bb_lower)
        self.bb_upper = float(bb_upper)
        self.ofi_threshold = float(ofi_threshold)
        self.vr_ceiling = float(vr_ceiling)

    def evaluate(self, df: pd.DataFrame, symbol: str) -> pd.Series:
        """Legacy 1D series evaluator returning buy conviction."""
        sigs = self.generate_signals(df, symbol)
        conviction = pd.Series(0.0, index=df.index)
        for s in sigs:
            if s.action == TriggerAction.BUY:
                conviction.loc[s.timestamp] = s.conviction
        return conviction

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[TriggerSignal]:
        """Generate causal BUY and SELL signals for intraday Bollinger absorption."""
        signals: List[TriggerSignal] = []
        if df.empty or len(df) < 20 or symbol in ("GOLDBEES", "SILVERBEES"):
            return signals

        vr = extract_variance_ratio(df)
        ofi = extract_ofi_proxy(df)
        vwap = extract_session_vwap(df)

        # Compute Bollinger Bands
        mid = df["Close"].rolling(20, min_periods=5).mean()
        std = df["Close"].rolling(20, min_periods=5).std().replace(0.0, np.nan)
        lower = mid - 2.0 * std
        upper = mid + 2.0 * std
        bb_pct = ((df["Close"] - lower) / (upper - lower + 1e-6)).fillna(0.5)

        daily_entered: Set[Any] = set()

        for i in range(1, len(df)):
            ts = df.index[i]
            t_str = ts.strftime("%H:%M")
            d = ts.normalize()

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

            # Active strictly in mid-day liquidity window: 10:00 to 14:15 IST
            if not ("10:00" <= t_str <= "14:15") or d in daily_entered:
                continue

            curr_vr = float(vr.iloc[i])
            # Filter: ONLY trade mean-reversion when Variance Ratio confirms chop
            if curr_vr >= self.vr_ceiling:
                continue

            curr_close = float(df["Close"].iloc[i])
            curr_open = float(df["Open"].iloc[i])
            curr_ofi = float(ofi.iloc[i])
            curr_vwap = float(vwap.iloc[i]) if vwap is not None else curr_close
            curr_bb = float(bb_pct.iloc[i])
            raw_atr = float(df.get("ATR_14", pd.Series(curr_close * 0.01, index=df.index)).iloc[i])
            atr_val = max(raw_atr, curr_close * 0.0035)

            # 1. LONG BOUNCE: Lower Band touch + Bullish reversal tick
            if (curr_bb <= self.bb_lower) and (curr_close > curr_open) and (curr_ofi >= self.ofi_threshold) and (curr_close < curr_vwap):
                daily_entered.add(d)
                target = curr_vwap
                stop = curr_close - (0.8 * atr_val)
                signals.append(
                    TriggerSignal(
                        timestamp=ts,
                        symbol=symbol,
                        action=TriggerAction.BUY,
                        conviction=0.75,
                        trigger_name=f"{self.name}_Long",
                        metadata={"side": "LONG", "target_price": round(target, 2), "stop_loss": round(stop, 2)},
                    )
                )

            # 2. SHORT FADE: Upper Band touch + Bearish rejection tick
            elif (curr_bb >= self.bb_upper) and (curr_close < curr_open) and (curr_ofi <= -self.ofi_threshold) and (curr_close > curr_vwap):
                daily_entered.add(d)
                target = curr_vwap
                stop = curr_close + (0.8 * atr_val)
                signals.append(
                    TriggerSignal(
                        timestamp=ts,
                        symbol=symbol,
                        action=TriggerAction.SELL,
                        conviction=0.75,
                        trigger_name=f"{self.name}_Short",
                        metadata={"side": "SHORT", "target_price": round(target, 2), "stop_loss": round(stop, 2)},
                    )
                )

        return signals


class TwoWayLiquidityStopSweepTrigger(StructuralTrigger):
    """Trigger: Capitalizes on retail stop-sweeps at 10-bar swing highs/lows."""

    name: str = "TwoWayLiquidityStopSweepTrigger"

    def __init__(self, sweep_window: int = 10, ofi_threshold: float = 0.15) -> None:
        self.sweep_window = int(sweep_window)
        self.ofi_threshold = float(ofi_threshold)

    def evaluate(self, df: pd.DataFrame, symbol: str) -> pd.Series:
        """Legacy 1D series evaluator returning buy conviction."""
        sigs = self.generate_signals(df, symbol)
        conviction = pd.Series(0.0, index=df.index)
        for s in sigs:
            if s.action == TriggerAction.BUY:
                conviction.loc[s.timestamp] = s.conviction
        return conviction

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[TriggerSignal]:
        """Generate signals when 10-bar swing high/low is pierced and rejected."""
        signals: List[TriggerSignal] = []
        if df.empty or len(df) < self.sweep_window + 5 or symbol in ("GOLDBEES", "SILVERBEES"):
            return signals

        ofi = extract_ofi_proxy(df)
        daily_entered: Set[Any] = set()

        for i in range(self.sweep_window, len(df)):
            ts = df.index[i]
            t_str = ts.strftime("%H:%M")
            d = ts.normalize()

            if not ("09:40" <= t_str <= "14:15") or d in daily_entered:
                continue

            lookback = df.iloc[i - self.sweep_window : i]
            swing_high = float(lookback["High"].max())
            swing_low = float(lookback["Low"].min())

            curr_high = float(df["High"].iloc[i])
            curr_low = float(df["Low"].iloc[i])
            curr_close = float(df["Close"].iloc[i])
            curr_open = float(df["Open"].iloc[i])
            curr_ofi = float(ofi.iloc[i])
            raw_atr = float(df.get("ATR_14", pd.Series(curr_close * 0.01, index=df.index)).iloc[i])
            atr_val = max(raw_atr, curr_close * 0.0035)

            bar_range = max(1e-4, curr_high - curr_low)
            lower_wick = min(curr_open, curr_close) - curr_low
            upper_wick = curr_high - max(curr_open, curr_close)

            # BULLISH SWEEP: Pierced swing low, closed back above with lower wick >= 25% of bar
            if (curr_low < swing_low) and (curr_close > swing_low) and (lower_wick / bar_range >= 0.25) and (curr_ofi >= self.ofi_threshold):
                daily_entered.add(d)
                target = curr_close + (1.5 * atr_val)
                stop = curr_low - (0.2 * atr_val)
                signals.append(
                    TriggerSignal(
                        timestamp=ts,
                        symbol=symbol,
                        action=TriggerAction.BUY,
                        conviction=0.80,
                        trigger_name=f"{self.name}_BullishSweep",
                        metadata={"side": "LONG", "target_price": round(target, 2), "stop_loss": round(stop, 2)},
                    )
                )

            # BEARISH SWEEP: Pierced swing high, closed back below with upper wick >= 25% of bar
            elif (curr_high > swing_high) and (curr_close < swing_high) and (upper_wick / bar_range >= 0.25) and (curr_ofi <= -self.ofi_threshold):
                daily_entered.add(d)
                target = curr_close - (1.5 * atr_val)
                stop = curr_high + (0.2 * atr_val)
                signals.append(
                    TriggerSignal(
                        timestamp=ts,
                        symbol=symbol,
                        action=TriggerAction.SELL,
                        conviction=0.80,
                        trigger_name=f"{self.name}_BearishSweep",
                        metadata={"side": "SHORT", "target_price": round(target, 2), "stop_loss": round(stop, 2)},
                    )
                )

        return signals
