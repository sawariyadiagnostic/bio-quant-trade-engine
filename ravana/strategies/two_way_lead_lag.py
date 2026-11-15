"""RAVANA v4.3: Two-Way Cross-Asset Lead-Lag Structural Trigger.

Exploits microstructural lead-lag divergence across highly correlated ETF pairs:
1. BANKBEES (Leader) -> PSUBNKBEES (Follower)
2. NIFTYBEES (Leader) -> JUNIORBEES (Follower)
3. GOLDBEES (Leader) -> SILVERBEES (Follower)

Mechanics:
- Causal Evaluation: Evaluates Leader 3-bar (15m) momentum impulse and OFI at bar t-1.
- Bullish Divergence: Leader r_15m >= +0.22% with positive OFI while Follower lags (r_15m <= +0.08%).
  Signals BUY Follower at bar t.
- Bearish Divergence: Leader r_15m <= -0.22% with negative OFI while Follower lags (r_15m >= -0.08%).
  Signals SELL / SHORT Follower at bar t.
- Self-Loading Feeder: Loads and caches leader dataset from data_dir directly.
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
    extract_session_vwap,
)

PAIR_MAP: Dict[str, str] = {
    "PSUBNKBEES": "BANKBEES",
    "JUNIORBEES": "NIFTYBEES",
    "SILVERBEES": "GOLDBEES",
}


class TwoWayLeadLagTrigger(StructuralTrigger):
    """Trigger: Cross-Asset Two-Way Lead-Lag Momentum Divergence."""

    name: str = "TwoWayLeadLagTrigger"

    def __init__(
        self,
        leader_ret_threshold: float = 0.0022,
        follower_lag_threshold: float = 0.0008,
        lookback_bars: int = 3,
        ofi_threshold: float = 0.10,
        data_dir: Optional[str] = None,
    ) -> None:
        self.leader_ret_threshold = float(leader_ret_threshold)
        self.follower_lag_threshold = float(follower_lag_threshold)
        self.lookback_bars = int(lookback_bars)
        self.ofi_threshold = float(ofi_threshold)
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "lake",
            "m3_ingested",
            "intraday_5m",
        )
        self._cached_leaders: Dict[str, pd.DataFrame] = {}

    def _ensure_leader_loaded(self, leader_symbol: str) -> Optional[pd.DataFrame]:
        """Load and cache leader ETF dataset."""
        if leader_symbol in self._cached_leaders:
            return self._cached_leaders[leader_symbol]
        leader_path = os.path.join(self.data_dir, f"{leader_symbol}.parquet")
        if os.path.exists(leader_path):
            try:
                df = pd.read_parquet(leader_path)
                self._cached_leaders[leader_symbol] = df
                return df
            except Exception:
                return None
        return None

    def evaluate(self, df: pd.DataFrame, symbol: str) -> pd.Series:
        """Legacy 1D series evaluator."""
        sigs = self.generate_signals(df, symbol)
        conviction = pd.Series(0.0, index=df.index)
        for s in sigs:
            if s.action == TriggerAction.BUY:
                conviction.loc[s.timestamp] = s.conviction
        return conviction

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[TriggerSignal]:
        """Generate causal BUY and SELL signals based on leader lead-lag."""
        signals: List[TriggerSignal] = []
        clean_symbol = symbol.replace(".NS", "")
        if clean_symbol not in PAIR_MAP or df.empty or len(df) < 20:
            return signals

        leader_symbol = PAIR_MAP[clean_symbol]
        leader_df = self._ensure_leader_loaded(leader_symbol)
        if leader_df is None or leader_df.empty:
            return signals

        # Causal t-1 indicator computation
        l_ret = leader_df["Close"].pct_change(self.lookback_bars)
        l_ofi = extract_ofi_proxy(leader_df)
        f_ret = df["Close"].pct_change(self.lookback_bars)

        daily_entered: Set[Any] = set()

        for i in range(self.lookback_bars + 1, len(df)):
            ts = df.index[i]
            t_str = ts.strftime("%H:%M")
            d = ts.date()

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

            if not ("09:35" <= t_str <= "14:15") or d in daily_entered:
                continue

            # Strict t-1 causal index on leader
            t_prev = df.index[i - 1]
            if t_prev not in leader_df.index:
                continue

            prev_l_ret = float(l_ret.loc[t_prev])
            prev_l_ofi = float(l_ofi.loc[t_prev])
            prev_f_ret = float(f_ret.loc[t_prev])

            curr_close = float(df["Close"].iloc[i])
            raw_atr = float(df.get("ATR_14", pd.Series(curr_close * 0.01, index=df.index)).iloc[i])
            atr_val = max(raw_atr, curr_close * 0.0035)

            # 1. BULLISH LEAD-LAG: Leader pumped, Follower lagged -> BUY Follower
            if (
                prev_l_ret >= self.leader_ret_threshold
                and prev_l_ofi >= self.ofi_threshold
                and prev_f_ret <= self.follower_lag_threshold
            ):
                daily_entered.add(d)
                target = curr_close + (1.5 * atr_val)
                stop = curr_close - (1.0 * atr_val)
                signals.append(
                    TriggerSignal(
                        timestamp=ts,
                        symbol=symbol,
                        action=TriggerAction.BUY,
                        conviction=0.80,
                        trigger_name=f"{self.name}_Long",
                        metadata={
                            "side": "LONG",
                            "leader": leader_symbol,
                            "target_price": round(target, 2),
                            "stop_loss": round(stop, 2),
                        },
                    )
                )

            # 2. BEARISH LEAD-LAG: Leader dumped, Follower lagged -> SHORT Follower
            elif (
                prev_l_ret <= -self.leader_ret_threshold
                and prev_l_ofi <= -self.ofi_threshold
                and prev_f_ret >= -self.follower_lag_threshold
            ):
                daily_entered.add(d)
                target = curr_close - (1.5 * atr_val)
                stop = curr_close + (1.0 * atr_val)
                signals.append(
                    TriggerSignal(
                        timestamp=ts,
                        symbol=symbol,
                        action=TriggerAction.SELL,
                        conviction=0.80,
                        trigger_name=f"{self.name}_Short",
                        metadata={
                            "side": "SHORT",
                            "leader": leader_symbol,
                            "target_price": round(target, 2),
                            "stop_loss": round(stop, 2),
                        },
                    )
                )

        return signals
