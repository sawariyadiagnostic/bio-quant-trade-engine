"""Two-Way Sector PEAD / Macro Announcement Institutional Drift Trigger for RAVANA v4.3.

Refined with Institutional Formula AST Engine & Broad Market Regime Conditioning:
1. Benchmark Momentum Impulse via Formula AST:
   Evaluates TS_ZSCORE_20(TS_DECAY_LINEAR_10(LOG_RET_5)) on underlying sector indices
   (CNXIT, CNXAUTO, CNXPHARMA) for scale-free, decay-weighted momentum conviction (|Z| >= 1.2).
2. Broad Market Filter:
   - Long Drift: NIFTY 50 >= Session Open (broad market is supportive or green).
   - Short Drift: NIFTY 50 <= Session Open (broad market selling pressure confirms breakdown).
3. Scale-Free Microstructural Reversion at Session VWAP:
   - Bullish Drift (BUY): Pulls back to touch VWAP with non-negative OFI confirmation.
   - Bearish Drift (SELL MIS): Bounces up to touch VWAP with non-positive OFI confirmation.
4. Volatility-Protected ATR Floor:
   atr_val = max(raw_atr, 0.0050 * price) to prevent tick noise stop-outs on low-priced ETFs.
5. Strict Point-in-Time Causality (Zero Lookahead / No n+1 Leakage).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Set
import numpy as np
import pandas as pd

from ravana.domain.formula_ast import ASTEvaluator
from ravana.domain.formula_parser import FormulaParser
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
    """Trigger: Formula AST-Driven Two-Way Sector Macro Momentum Drift."""

    name: str = "TwoWaySectorDriftTrigger"

    def __init__(
        self,
        z_mom_threshold: float = 1.20,
        vr_threshold: float = 0.95,
        lakehouse_dir: Optional[str] = None,
        min_benchmark_momentum: Optional[float] = None,
    ) -> None:
        self.z_mom_threshold = float(z_mom_threshold)
        self.vr_threshold = float(vr_threshold)
        self.min_benchmark_momentum = min_benchmark_momentum
        if lakehouse_dir is None:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            self.lakehouse_dir = os.path.join(base_dir, "ravana", "data", "lake", "m3_ingested", "intraday_5m")
        else:
            self.lakehouse_dir = str(lakehouse_dir)

        self._formula_node = FormulaParser.parse("TS_ZSCORE_20(TS_DECAY_LINEAR_10(LOG_RET_5))")
        self._bench_cache: Dict[str, pd.DataFrame] = {}
        self._nifty_df: Optional[pd.DataFrame] = None
        self._nifty_open_map: Dict[Any, float] = {}

    def _get_benchmark_df(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load, compute Formula AST decay momentum, and cache benchmark index."""
        bench_file = BENCHMARK_MAP.get(symbol)
        if not bench_file:
            return None
        if bench_file not in self._bench_cache:
            p = os.path.join(self.lakehouse_dir, bench_file)
            if not os.path.exists(p):
                return None
            b_df = pd.read_parquet(p)
            if "LOG_RET_5" not in b_df.columns:
                c = b_df["Close"]
                b_df["LOG_RET_5"] = np.log(c / c.shift(5)).fillna(0.0)
            b_df["Z_DECAY_MOM"] = ASTEvaluator.eval_node(self._formula_node, b_df)
            self._bench_cache[bench_file] = b_df
        return self._bench_cache[bench_file]

    def _get_nifty_context(self) -> Tuple[Optional[pd.DataFrame], Dict[Any, float]]:
        """Load and cache NIFTY 50 reference data and daily session opens."""
        if self._nifty_df is None:
            p = os.path.join(self.lakehouse_dir, "NSEI.parquet")
            if os.path.exists(p):
                self._nifty_df = pd.read_parquet(p)
                self._nifty_open_map = (
                    self._nifty_df.groupby(self._nifty_df.index.date)["Open"].first().to_dict()
                )
        return self._nifty_df, self._nifty_open_map

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
        if df.empty or len(df) < 20 or symbol not in BENCHMARK_MAP:
            return signals

        bench_df = self._get_benchmark_df(symbol)
        if bench_df is None or bench_df.empty:
            return signals

        nifty_df, nifty_open_map = self._get_nifty_context()
        vr = extract_variance_ratio(df)
        ofi = extract_ofi_proxy(df)
        vwap = extract_session_vwap(df)
        daily_entered: Set[Any] = set()

        for i in range(20, len(df)):
            ts = df.index[i]
            t_str = ts.strftime("%H:%M")
            d = ts.date()

            if t_str == "11:30":
                signals.append(self._make_exit_signal(ts, symbol))
                continue

            if not ("09:40" <= t_str <= "10:45") or d in daily_entered:
                continue

            if float(vr.iloc[i]) < self.vr_threshold:
                continue

            if ts not in bench_df.index:
                continue

            sig = self._evaluate_bar(df, i, ts, symbol, bench_df, nifty_df, nifty_open_map, vwap, ofi)
            if sig is not None:
                daily_entered.add(d)
                signals.append(sig)

        return signals

    def _evaluate_bar(
        self,
        df: pd.DataFrame,
        i: int,
        ts: pd.Timestamp,
        symbol: str,
        bench_df: pd.DataFrame,
        nifty_df: Optional[pd.DataFrame],
        nifty_open_map: Dict[Any, float],
        vwap: pd.Series,
        ofi: pd.Series,
    ) -> Optional[TriggerSignal]:
        """Evaluate single bar causal conditions for bullish or bearish drift."""
        b_z_mom = float(bench_df.loc[ts, "Z_DECAY_MOM"])
        curr_close = float(df["Close"].iloc[i])
        curr_low = float(df["Low"].iloc[i])
        curr_high = float(df["High"].iloc[i])
        curr_vwap = float(vwap.iloc[i]) if vwap is not None else curr_close
        curr_ofi = float(ofi.iloc[i])
        raw_atr = float(df.get("ATR_14", pd.Series(curr_close * 0.01, index=df.index)).iloc[i])
        atr_val = max(raw_atr, curr_close * 0.0050)

        # Broad market regime check
        d = ts.date()
        n_open = nifty_open_map.get(d)
        n_close = float(nifty_df.loc[ts, "Close"]) if (nifty_df is not None and ts in nifty_df.index) else None

        # 1. BULLISH DRIFT (BUY): Z_mom >= threshold, NIFTY >= Open, VWAP retest
        if (b_z_mom >= self.z_mom_threshold) and (n_open is None or n_close is None or n_close >= n_open):
            if (curr_low <= curr_vwap * 1.002) and (curr_close >= curr_vwap * 0.999) and (curr_ofi >= 0.0):
                target = curr_close + (1.8 * atr_val)
                stop = curr_close - (1.2 * atr_val)
                return TriggerSignal(
                    timestamp=ts,
                    symbol=symbol,
                    action=TriggerAction.BUY,
                    conviction=0.88,
                    trigger_name=f"{self.name}_BullishDrift",
                    metadata={"side": "LONG", "target_price": round(target, 2), "stop_loss": round(stop, 2), "z_mom": b_z_mom},
                )

        # 2. BEARISH DRIFT (SELL MIS): Z_mom <= -threshold, NIFTY <= Open, VWAP bounce
        if (b_z_mom <= -self.z_mom_threshold) and (n_open is None or n_close is None or n_close <= n_open):
            if (curr_high >= curr_vwap * 0.998) and (curr_close <= curr_vwap * 1.001) and (curr_ofi <= 0.0):
                target = curr_close - (1.8 * atr_val)
                stop = curr_close + (1.2 * atr_val)
                return TriggerSignal(
                    timestamp=ts,
                    symbol=symbol,
                    action=TriggerAction.SELL,
                    conviction=0.88,
                    trigger_name=f"{self.name}_BearishDrift",
                    metadata={"side": "SHORT", "target_price": round(target, 2), "stop_loss": round(stop, 2), "z_mom": b_z_mom},
                )

        return None

    def _make_exit_signal(self, ts: pd.Timestamp, symbol: str) -> TriggerSignal:
        """Construct mandatory 11:30 AM morning lock exit signal."""
        return TriggerSignal(
            timestamp=ts,
            symbol=symbol,
            action=TriggerAction.EXIT_ALL,
            conviction=1.0,
            trigger_name=self.name,
            metadata={"reason": "MANDATORY_1130_MORNING_LOCK"},
        )
