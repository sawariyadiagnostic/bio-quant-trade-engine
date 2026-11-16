"""Two-Way Chop Mean-Reversion Trigger for Low-Volatility Intraday Regimes in RAVANA v4.3.

Capitalizes on mean-reverting stationary noise regimes (Variance Ratio VR_20 < 0.75):
1. Fades extreme Bollinger Band extensions (Lower Band <= 0.15, Upper Band >= 0.85).
2. Requires order flow imbalance confirmation (OFI_1 / OFI_5 >= +0.05 for longs, <= -0.05 for shorts).
3. Executes micro Risk-to-Reward trades (0.40 ATR stop, 0.50 ATR target to VWAP).
4. Strictly active only during morning (09:35-11:15) and afternoon (13:00-14:30) liquidity windows.
"""
from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)

DEFAULT_CHOP_SYMBOLS: Set[str] = {
    "AUTOBEES",
    "BANKBEES",
    "CPSEETF",
    "GOLDBEES",
    "ITBEES",
    "JUNIORBEES",
    "NIFTYBEES",
    "PHARMABEES",
    "PSUBNKBEES",
    "SILVERBEES",
}


class TwoWayChopMeanReversionTrigger(StructuralTrigger):
    """Trigger: Fades Bollinger Band extremes in mean-reverting stationary chop regimes (VR < 0.75)."""

    name: str = "TwoWayChopMeanReversionTrigger"

    def __init__(
        self,
        bb_lower: float = 0.15,
        bb_upper: float = 0.85,
        ofi_threshold: float = 0.05,
        vr_ceiling: float = 0.75,
        micro_stop_mult: float = 0.40,
        micro_target_mult: float = 0.50,
        allowed_symbols: Optional[Set[str]] = None,
    ) -> None:
        self.bb_lower = float(bb_lower)
        self.bb_upper = float(bb_upper)
        self.ofi_threshold = float(ofi_threshold)
        self.vr_ceiling = float(vr_ceiling)
        self.micro_stop_mult = float(micro_stop_mult)
        self.micro_target_mult = float(micro_target_mult)
        self.allowed_symbols = allowed_symbols if allowed_symbols is not None else DEFAULT_CHOP_SYMBOLS

    def evaluate(self, df: pd.DataFrame, symbol: str) -> pd.Series:
        """Legacy 1D series evaluator returning buy conviction."""
        sigs = self.generate_signals(df, symbol)
        conviction = pd.Series(0.0, index=df.index)
        for s in sigs:
            if s.action == TriggerAction.BUY:
                conviction.loc[s.timestamp] = s.conviction
        return conviction

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[TriggerSignal]:
        """Generate causal BUY and SELL signals for chop mean-reversion."""
        signals: List[TriggerSignal] = []
        if df.empty or len(df) < 20 or symbol not in self.allowed_symbols:
            return signals

        if "VR_20" in df.columns:
            vr = df["VR_20"].fillna(1.0)
        elif "VARIANCE_RATIO_20" in df.columns:
            vr = df["VARIANCE_RATIO_20"].fillna(1.0)
        else:
            vr = extract_variance_ratio(df)

        ofi = extract_ofi_proxy(df)
        vwap = extract_session_vwap(df)
        bb_pct = self._compute_bb_pct(df)

        daily_entered: Set[Any] = set()

        for i in range(1, len(df)):
            ts = df.index[i]
            t_str = ts.strftime("%H:%M")
            d = ts.date()

            if t_str in ("11:30", "14:45", "15:00"):
                signals.append(self._make_exit_signal(ts, symbol))
                continue

            in_morning = ("09:35" <= t_str <= "11:15")
            in_afternoon = ("13:00" <= t_str <= "14:30")
            if not (in_morning or in_afternoon) or d in daily_entered:
                continue

            curr_vr = float(vr.iloc[i])
            if curr_vr >= self.vr_ceiling:
                continue

            sig = self._evaluate_bar(df, i, ts, symbol, ofi, vwap, bb_pct)
            if sig is not None:
                daily_entered.add(d)
                signals.append(sig)

        return signals

    def _compute_bb_pct(self, df: pd.DataFrame) -> pd.Series:
        """Compute Bollinger Band percentile position (%B)."""
        mid = df["Close"].rolling(20, min_periods=5).mean()
        std = df["Close"].rolling(20, min_periods=5).std().replace(0.0, np.nan)
        lower = mid - 2.0 * std
        upper = mid + 2.0 * std
        return ((df["Close"] - lower) / (upper - lower + 1e-6)).fillna(0.5)

    def _evaluate_bar(
        self,
        df: pd.DataFrame,
        i: int,
        ts: pd.Timestamp,
        symbol: str,
        ofi: pd.Series,
        vwap: Optional[pd.Series],
        bb_pct: pd.Series,
    ) -> Optional[TriggerSignal]:
        """Evaluate single-bar micro conditions for Bollinger Band absorption."""
        curr_close = float(df["Close"].iloc[i])
        curr_open = float(df["Open"].iloc[i])
        curr_ofi = float(ofi.iloc[i])
        curr_vwap = float(vwap.iloc[i]) if vwap is not None else curr_close
        curr_bb = float(bb_pct.iloc[i])

        raw_atr = float(df.get("ATR_14", pd.Series(curr_close * 0.0050, index=df.index)).iloc[i])
        atr_val = max(raw_atr, curr_close * 0.0020)

        # 1. LONG MEAN-REVERSION: Lower Band touch + Bullish tick + OFI positive
        if (curr_bb <= self.bb_lower) and (curr_close > curr_open) and (curr_ofi >= self.ofi_threshold):
            target = max(curr_vwap, curr_close + (self.micro_target_mult * atr_val))
            stop = curr_close - (self.micro_stop_mult * atr_val)
            return TriggerSignal(
                timestamp=ts,
                symbol=symbol,
                action=TriggerAction.BUY,
                conviction=0.80,
                trigger_name=f"{self.name}_Long",
                metadata={"side": "LONG", "target_price": round(target, 2), "stop_loss": round(stop, 2), "atr": atr_val},
            )

        # 2. SHORT MEAN-REVERSION: Upper Band touch + Bearish tick + OFI negative
        if (curr_bb >= self.bb_upper) and (curr_close < curr_open) and (curr_ofi <= -self.ofi_threshold):
            target = min(curr_vwap, curr_close - (self.micro_target_mult * atr_val))
            stop = curr_close + (self.micro_stop_mult * atr_val)
            return TriggerSignal(
                timestamp=ts,
                symbol=symbol,
                action=TriggerAction.SELL,
                conviction=0.80,
                trigger_name=f"{self.name}_Short",
                metadata={"side": "SHORT", "target_price": round(target, 2), "stop_loss": round(stop, 2), "atr": atr_val},
            )

        return None

    def _make_exit_signal(self, ts: pd.Timestamp, symbol: str) -> TriggerSignal:
        """Helper to create session lock exit signal."""
        return TriggerSignal(
            timestamp=ts,
            symbol=symbol,
            action=TriggerAction.EXIT_ALL,
            conviction=1.0,
            trigger_name=self.name,
            metadata={"reason": "CHOP_SESSION_LOCK_EXIT"},
        )
