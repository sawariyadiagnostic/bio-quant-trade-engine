"""RAVANA v4.3: Anti-Overfitting Walk-Forward Optimization & Diagnostic Suite.

Implements strict out-of-sample discipline to eliminate curve-fitting ("tailored keys to a specific lock"):
1. In-Sample Training Window (Days 1 to 42: ~67% of dataset).
2. Blind Out-of-Sample Window (Days 43 to 63: ~33% of dataset).
3. Evaluates all 9 strategies on both windows.
4. Monte Carlo Trade Permutation Tests (1,000 paths).
5. Microstructural Asset Archetype Diagnostics.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from ravana.optimization.vector_engine import (
    VectorLake,
    VectorSimulationResult,
    VectorSimulator,
)


class AntiOverfittingOptimizer:
    """Manages In-Sample vs. Out-of-Sample evaluation and microstructural diagnostics."""

    def __init__(self, lake: Optional[VectorLake] = None) -> None:
        self.lake = lake or VectorLake()
        self.sim = VectorSimulator(self.lake)

        # Split dates into In-Sample (~67%) and Out-of-Sample (~33%)
        all_dates = sorted(set(self.lake.dates))
        split_idx = int(len(all_dates) * 0.67)
        self.train_dates = all_dates[:split_idx]
        self.test_dates = all_dates[split_idx:]

        self.train_indices = np.where(np.isin(self.lake.dates, self.train_dates))[0]
        self.test_indices = np.where(np.isin(self.lake.dates, self.test_dates))[0]

    def evaluate_strategy_walk_forward(
        self,
        strategy_name: str,
        long_mask: np.ndarray,
        short_mask: np.ndarray,
        stop_atr_mult: float = 1.0,
        target_atr_mult: float = 1.5,
        exit_time_str: str = "15:00",
    ) -> Dict[str, Any]:
        """Evaluate a strategy across In-Sample and Out-of-Sample windows."""
        train_res = self.sim.simulate(
            long_mask=long_mask,
            short_mask=short_mask,
            time_indices=self.train_indices,
            stop_atr_mult=stop_atr_mult,
            target_atr_mult=target_atr_mult,
            exit_time_str=exit_time_str,
        )

        test_res = self.sim.simulate(
            long_mask=long_mask,
            short_mask=short_mask,
            time_indices=self.test_indices,
            stop_atr_mult=stop_atr_mult,
            target_atr_mult=target_atr_mult,
            exit_time_str=exit_time_str,
        )

        full_res = self.sim.simulate(
            long_mask=long_mask,
            short_mask=short_mask,
            time_indices=None,
            stop_atr_mult=stop_atr_mult,
            target_atr_mult=target_atr_mult,
            exit_time_str=exit_time_str,
        )

        # Anti-Overfitting Assessment
        is_overfit = (
            test_res.net_pnl < 0.0
            or test_res.profit_factor < 1.0
            or (train_res.profit_factor > 2.0 and test_res.profit_factor < 0.8)
        )

        mc_drawdown_95 = self._monte_carlo_drawdown(full_res.trades)

        return {
            "strategy": strategy_name,
            "train": train_res.to_dict(),
            "test_oos": test_res.to_dict(),
            "full_60d": full_res.to_dict(),
            "is_overfit": is_overfit,
            "mc_max_dd_pct": round(mc_drawdown_95, 3),
            "status": "APPROVED (ROBUST)" if not is_overfit and test_res.net_pnl > 0 else "REJECTED (OVERFIT / UNSTABLE)",
        }

    def _monte_carlo_drawdown(self, trades: List[Dict[str, Any]], iterations: int = 1000) -> float:
        """Run 1,000-path Monte Carlo trade shuffling to calculate 95th percentile drawdown."""
        if len(trades) < 3:
            return 0.0
        pnls = np.array([tr["pnl"] for tr in trades])
        max_dds: List[float] = []

        for _ in range(iterations):
            shuffled = np.random.permutation(pnls)
            equity = 200000.0 + np.cumsum(shuffled)
            peak = np.maximum.accumulate(equity)
            dd = (peak - equity) / peak
            max_dds.append(float(np.max(dd)))

        return float(np.percentile(max_dds, 95) * 100.0)

    # -----------------------------------------------------------------------
    # Microstructural Signal Mask Builders
    # -----------------------------------------------------------------------

    def build_strategy1_orb_masks(self) -> Tuple[np.ndarray, np.ndarray]:
        """Strategy 1: Morning ORB (09:35 to 10:30 IST) on Trending Sectors."""
        long_m = np.zeros_like(self.lake.close_mat, dtype=bool)
        short_m = np.zeros_like(self.lake.close_mat, dtype=bool)
        target_syms = {"ITBEES", "AUTOBEES", "NIFTYBEES"}

        time_window = (self.lake.time_strings >= "09:35") & (self.lake.time_strings <= "10:30")
        for s, sym in enumerate(self.lake.symbols):
            if sym not in target_syms:
                continue
            cl = self.lake.close_mat[:, s]
            op = self.lake.open_mat[:, s]
            vr = self.lake.vr_mat[:, s]
            ofi = self.lake.ofi_mat[:, s]

            # High/Low break of first 15 mins (approx 3 bars)
            r_high = pd.Series(self.lake.high_mat[:, s]).shift(1).rolling(3, min_periods=3).max().to_numpy()
            r_low = pd.Series(self.lake.low_mat[:, s]).shift(1).rolling(3, min_periods=3).min().to_numpy()

            long_m[:, s] = (cl > r_high) & (vr >= 0.90) & (ofi > 0.15) & time_window
            short_m[:, s] = (cl < r_low) & (vr >= 0.90) & (ofi < -0.15) & time_window

        return long_m, short_m

    def build_strategy2_europe_masks(self) -> Tuple[np.ndarray, np.ndarray]:
        """Strategy 2: Europe Open Momentum (13:00 to 13:45 IST) on IT & NIFTY."""
        long_m = np.zeros_like(self.lake.close_mat, dtype=bool)
        short_m = np.zeros_like(self.lake.close_mat, dtype=bool)
        target_syms = {"ITBEES", "NIFTYBEES"}  # Exclude domestically pinned BANKBEES

        time_window = (self.lake.time_strings >= "13:00") & (self.lake.time_strings <= "13:45")
        for s, sym in enumerate(self.lake.symbols):
            if sym not in target_syms:
                continue
            ret5 = (self.lake.close_mat[:, s] - self.lake.close_mat[max(0, 0):, s])  # handled below
            ret5 = pd.Series(self.lake.close_mat[:, s]).pct_change(5).fillna(0.0).to_numpy()
            ofi = self.lake.ofi_mat[:, s]

            long_m[:, s] = (ret5 >= 0.0015) & (ofi >= 0.10) & time_window
            short_m[:, s] = (ret5 <= -0.0015) & (ofi <= -0.10) & time_window

        return long_m, short_m

    def build_strategy3_absorption_masks(self) -> Tuple[np.ndarray, np.ndarray]:
        """Strategy 3: VWAP Bollinger Absorption with NIFTY Trend Gate."""
        long_m = np.zeros_like(self.lake.close_mat, dtype=bool)
        short_m = np.zeros_like(self.lake.close_mat, dtype=bool)
        target_syms = {"AUTOBEES", "BANKBEES", "ITBEES", "NIFTYBEES", "PHARMABEES"}

        time_window = (self.lake.time_strings >= "10:00") & (self.lake.time_strings <= "13:30")
        n_bull = self.lake.nifty_close >= self.lake.nifty_open
        n_bear = self.lake.nifty_close <= self.lake.nifty_open

        for s, sym in enumerate(self.lake.symbols):
            if sym not in target_syms:
                continue
            cl = self.lake.close_mat[:, s]
            op = self.lake.open_mat[:, s]
            bb = self.lake.bb_pct_mat[:, s]
            vw = self.lake.vwap_mat[:, s]
            vr = self.lake.vr_mat[:, s]
            ofi = self.lake.ofi_mat[:, s]
            atr = self.lake.atr_mat[:, s]

            # Long lower band with RR check
            long_m[:, s] = (
                (bb <= 0.18)
                & (cl > op)
                & (ofi >= 0.12)
                & (vr < 0.88)
                & (vw - cl >= 0.85 * atr)
                & n_bull
                & time_window
            )
            # Short upper band with RR check
            short_m[:, s] = (
                (bb >= 0.82)
                & (cl < op)
                & (ofi <= -0.12)
                & (vr < 0.88)
                & (cl - vw >= 0.85 * atr)
                & n_bear
                & time_window
            )

        return long_m, short_m

    def build_strategy6_stopsweep_masks(self) -> Tuple[np.ndarray, np.ndarray]:
        """Strategy 6: Two-Way Stop Sweep on Mean-Reverting Assets (GOLDBEES, SILVERBEES, PHARMABEES)."""
        long_m = np.zeros_like(self.lake.close_mat, dtype=bool)
        short_m = np.zeros_like(self.lake.close_mat, dtype=bool)
        target_syms = {"GOLDBEES", "SILVERBEES", "PHARMABEES"}

        time_window = (self.lake.time_strings >= "09:40") & (self.lake.time_strings <= "10:45")
        for s, sym in enumerate(self.lake.symbols):
            if sym not in target_syms:
                continue
            hi = self.lake.high_mat[:, s]
            lo = self.lake.low_mat[:, s]
            cl = self.lake.close_mat[:, s]
            op = self.lake.open_mat[:, s]
            vr = self.lake.vr_mat[:, s]
            ofi = self.lake.ofi_mat[:, s]

            r_high_10 = pd.Series(hi).shift(1).rolling(10, min_periods=10).max().to_numpy()
            r_low_10 = pd.Series(lo).shift(1).rolling(10, min_periods=10).min().to_numpy()

            candle_range = np.maximum(hi - lo, 1e-4)
            lower_wick = np.maximum(0.0, np.minimum(cl, op) - lo) / candle_range
            upper_wick = np.maximum(0.0, hi - np.maximum(cl, op)) / candle_range

            long_m[:, s] = (lo < r_low_10) & (cl > r_low_10) & (lower_wick >= 0.35) & (ofi > 0.05) & (vr <= 0.92) & time_window
            short_m[:, s] = (hi > r_high_10) & (cl < r_high_10) & (upper_wick >= 0.35) & (ofi < -0.05) & (vr <= 0.92) & time_window

        return long_m, short_m

    def build_strategy8_expiry_masks(self) -> Tuple[np.ndarray, np.ndarray]:
        """Strategy 8: Thursday Expiry Gamma Unpinning on BANKBEES and NIFTYBEES."""
        long_m = np.zeros_like(self.lake.close_mat, dtype=bool)
        short_m = np.zeros_like(self.lake.close_mat, dtype=bool)
        target_syms = {"BANKBEES", "NIFTYBEES"}

        time_window = (self.lake.time_strings >= "13:45") & (self.lake.time_strings <= "14:45")
        is_thursday = np.array([ts.dayofweek == 3 for ts in self.lake.timestamps])

        for s, sym in enumerate(self.lake.symbols):
            if sym not in target_syms:
                continue
            cl = self.lake.close_mat[:, s]
            vol = self.lake.vol_mat[:, s]
            v_avg = pd.Series(vol).rolling(20, min_periods=5).mean().to_numpy()
            r_hi6 = pd.Series(self.lake.high_mat[:, s]).shift(1).rolling(6, min_periods=6).max().to_numpy()
            r_lo6 = pd.Series(self.lake.low_mat[:, s]).shift(1).rolling(6, min_periods=6).min().to_numpy()
            ofi = self.lake.ofi_mat[:, s]

            is_vol_surge = vol >= 1.20 * v_avg

            long_m[:, s] = (cl > r_hi6) & is_vol_surge & (ofi >= 0.0) & is_thursday & time_window
            short_m[:, s] = (cl < r_lo6) & is_vol_surge & (ofi <= 0.0) & is_thursday & time_window

        return long_m, short_m

    def build_strategy9_leadlag_masks(self) -> Tuple[np.ndarray, np.ndarray]:
        """Strategy 9: Cross-Asset Lead-Lag Divergence (Follower orders)."""
        long_m = np.zeros_like(self.lake.close_mat, dtype=bool)
        short_m = np.zeros_like(self.lake.close_mat, dtype=bool)

        pairs = [
            ("PSUBNKBEES", "BANKBEES"),
            ("SILVERBEES", "GOLDBEES"),
        ]
        time_window = (self.lake.time_strings >= "09:35") & (self.lake.time_strings <= "14:15")

        for follower, leader in pairs:
            if follower not in self.lake.symbol_to_idx or leader not in self.lake.symbol_to_idx:
                continue
            f_idx = self.lake.symbol_to_idx[follower]
            l_idx = self.lake.symbol_to_idx[leader]

            l_ret3 = pd.Series(self.lake.close_mat[:, l_idx]).pct_change(3).shift(1).fillna(0.0).to_numpy()
            f_ret3 = pd.Series(self.lake.close_mat[:, f_idx]).pct_change(3).shift(1).fillna(0.0).to_numpy()
            l_ofi = pd.Series(self.lake.ofi_mat[:, l_idx]).shift(1).fillna(0.0).to_numpy()

            long_m[:, f_idx] = (l_ret3 >= 0.0022) & (l_ofi >= 0.10) & (f_ret3 <= 0.0008) & time_window
            short_m[:, f_idx] = (l_ret3 <= -0.0022) & (l_ofi <= -0.10) & (f_ret3 >= -0.0008) & time_window

        return long_m, short_m
