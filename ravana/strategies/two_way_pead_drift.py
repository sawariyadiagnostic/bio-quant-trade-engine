"""Two-Way Sector PEAD / Macro Announcement Institutional Drift Trigger for RAVANA v4.3.

Captures institutional trend continuation following morning macro/earnings momentum:
1. Benchmark Momentum Confirmation: Measures underlying sector index (CNXIT, CNXAUTO, CNXPHARMA)
   session return (|Ret_session| >= 0.50%) with trend persistence (VR_20 >= 0.95).
2. Pullback / Bounce Execution at Session VWAP:
   - Bullish Drift (BUY): Pulls back to touch rising VWAP with positive OFI confirmation.
   - Bearish Drift (SELL MIS): Bounces up to touch falling VWAP with negative OFI confirmation.
3. High-Liquidity Morning Window: 09:40 to 10:45 IST with mandatory 11:30 AM lock.
4. Strict Point-in-Time Causality (Zero Lookahead / No n+1 Leakage).
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
    extract_variance_ratio,
)

# Underlying benchmark index mapping
BENCHMARK_MAP: Dict[str, str] = {
    "ITBEES": "CNXIT.parquet",
    "AUTOBEES": "CNXAUTO.parquet",
    "PHARMABEES": "CNXPHARMA.parquet",
}


class TwoWaySectorDriftTrigger(StructuralTrigger):
    """Trigger: Two-Way Sector Macro Momentum Drift / VWAP Continuation."""

    name: str = "TwoWaySectorDriftTrigger"

    def __init__(
        self,
        min_benchmark_momentum: float = 0.0050,  # 0.50% session move
        vr_threshold: float = 0.95,
        lakehouse_dir: Optional[str] = None,
    ) -> None:
        self.min_benchmark_momentum = float(min_benchmark_momentum)
        self.vr_threshold = float(vr_threshold)
        if lakehouse_dir is None:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            self.lakehouse_dir = os.path.join(base_dir, "ravana", "data", "lake", "m3_ingested", "intraday_5m")
        else:
            self.lakehouse_dir = str(lakehouse_dir)

        self._bench_cache: Dict[str, pd.DataFrame] = {}

    def _get_benchmark_df(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load and cache benchmark index DataFrame."""
        bench_file = BENCHMARK_MAP.get(symbol)
        if not bench_file:
            return None
        if bench_file not in self._bench_cache:
            p = os.path.join(self.lakehouse_dir, bench_file)
            if os.path.exists(p):
                self._bench_cache[bench_file] = pd.read_parquet(p)
            else:
                return None
        return self._bench_cache[bench_file]

    def evaluate(self, df: pd.DataFrame, symbol: str) -> pd.Series:
        """Legacy 1D series evaluator returning buy conviction."""
        sigs = self.generate_signals(df, symbol)
        conviction = pd.Series(0.0, index=df.index)
        for s in sigs:
            if s.action == TriggerAction.BUY:
                conviction.loc[s.timestamp] = s.conviction
        return conviction

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[TriggerSignal]:
        """Generate causal BUY and SELL signals for sector institutional drift."""
        signals: List[TriggerSignal] = []
        if df.empty or len(df) < 10 or symbol not in BENCHMARK_MAP:
            return signals

        bench_df = self._get_benchmark_df(symbol)
        if bench_df is None or bench_df.empty:
            return signals

        vr = extract_variance_ratio(df)
        ofi = extract_ofi_proxy(df)
        vwap = extract_session_vwap(df)

        daily_entered: Set[Any] = set()

        for i in range(5, len(df)):
            ts = df.index[i]
            t_str = ts.strftime("%H:%M")
            d = ts.normalize()

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

            if not ("09:40" <= t_str <= "10:45") or d in daily_entered:
                continue

            curr_vr = float(vr.iloc[i])
            if curr_vr < self.vr_threshold:
                continue

            if ts not in bench_df.index:
                continue

            # Benchmark session momentum (open of current day to ts)
            day_bench = bench_df[bench_df.index.normalize() == d]
            if len(day_bench) < 2 or ts not in day_bench.index:
                continue

            bench_open = float(day_bench["Open"].iloc[0])
            bench_curr = float(day_bench.loc[ts, "Close"])
            bench_ret = (bench_curr - bench_open) / (bench_open + 1e-6)

            curr_close = float(df["Close"].iloc[i])
            curr_low = float(df["Low"].iloc[i])
            curr_high = float(df["High"].iloc[i])
            curr_vwap = float(vwap.iloc[i]) if vwap is not None else curr_close
            curr_ofi = float(ofi.iloc[i])
            raw_atr = float(df.get("ATR_14", pd.Series(curr_close * 0.01, index=df.index)).iloc[i])
            atr_val = max(raw_atr, curr_close * 0.0035)

            # 1. BULLISH DRIFT CONTINUATION (BUY): Benchmark >= +0.50% + Pullback to VWAP
            if (bench_ret >= self.min_benchmark_momentum) and (curr_low <= curr_vwap * 1.001) and (curr_close >= curr_vwap) and (curr_ofi >= 0.05):
                daily_entered.add(d)
                target = curr_close + (1.5 * atr_val)
                stop = curr_close - (1.0 * atr_val)
                signals.append(
                    TriggerSignal(
                        timestamp=ts,
                        symbol=symbol,
                        action=TriggerAction.BUY,
                        conviction=0.85,
                        trigger_name=f"{self.name}_BullishDrift",
                        metadata={"side": "LONG", "target_price": round(target, 2), "stop_loss": round(stop, 2), "bench_ret": bench_ret},
                    )
                )

            # 2. BEARISH DRIFT CONTINUATION (SELL MIS): Benchmark <= -0.50% + Bounce to VWAP
            elif (bench_ret <= -self.min_benchmark_momentum) and (curr_high >= curr_vwap * 0.999) and (curr_close <= curr_vwap) and (curr_ofi <= -0.05):
                daily_entered.add(d)
                target = curr_close - (1.5 * atr_val)
                stop = curr_close + (1.0 * atr_val)
                signals.append(
                    TriggerSignal(
                        timestamp=ts,
                        symbol=symbol,
                        action=TriggerAction.SELL,
                        conviction=0.85,
                        trigger_name=f"{self.name}_BearishDrift",
                        metadata={"side": "SHORT", "target_price": round(target, 2), "stop_loss": round(stop, 2), "bench_ret": bench_ret},
                    )
                )

        return signals
