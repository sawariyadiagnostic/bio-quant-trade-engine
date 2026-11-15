"""RAVANA v4.3: Anti-Overfitting Walk-Forward Optimization Runner.

Executes the Anti-Memory Out-of-Sample Protocol across all strategies:
1. In-Sample Training (Days 1 to 42: 2026-08-18 to ~2026-10-15).
2. Out-of-Sample Blind Test (Days 43 to 63: ~2026-10-16 to 2026-11-13).
3. Evaluates Overfitting Ratio and 1,000-Path Monte Carlo Drawdowns.
4. Generates Microstructural Root-Cause Diagnostics.
"""
from __future__ import annotations

import os
import sys
import time
import pandas as pd

# UTF-8 Console Safety
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ravana.optimization.anti_overfitting_suite import AntiOverfittingOptimizer
from ravana.optimization.vector_engine import VectorLake


def main() -> None:
    print("=" * 90)
    print("  RAVANA v4.3 ANTI-OVERFITTING WALK-FORWARD OPTIMIZATION & DIAGNOSTICS")
    print("  Protocol: In-Sample (Days 1-42) vs. Out-of-Sample Blind Test (Days 43-63)")
    print("  Anti-Memory Guard: Zero Parameter Curve-Fitting | 1,000-Path Monte Carlo Validation")
    print("=" * 90)

    t0 = time.time()
    lake = VectorLake()
    optimizer = AntiOverfittingOptimizer(lake=lake)
    t_load = time.time() - t0
    print(f"Loaded 3D Vector Lake in {t_load:.2f} seconds across {lake.n_syms} symbols & {lake.n_bars} bars.\n")

    strategies_to_test = [
        ("Strategy 1: Morning ORB (Trending Sectors)", optimizer.build_strategy1_orb_masks, 1.2, 1.5, "11:30"),
        ("Strategy 2: Europe Open (IT & NIFTY Only)", optimizer.build_strategy2_europe_masks, 1.0, 1.5, "14:45"),
        ("Strategy 3: VWAP Absorption (Trend-Gated)", optimizer.build_strategy3_absorption_masks, 1.0, 1.5, "15:00"),
        ("Strategy 6: Stop Sweep (Mean-Reverting Assets)", optimizer.build_strategy6_stopsweep_masks, 1.0, 1.8, "11:30"),
        ("Strategy 8: Expiry Gamma (Thursdays Index Pins)", optimizer.build_strategy8_expiry_masks, 1.0, 1.5, "15:00"),
        ("Strategy 9: Cross Lead-Lag (Follower Divergence)", optimizer.build_strategy9_leadlag_masks, 1.0, 1.5, "15:00"),
    ]

    results = []
    for name, mask_builder, stop_mult, tgt_mult, exit_time in strategies_to_test:
        long_m, short_m = mask_builder()
        eval_res = optimizer.evaluate_strategy_walk_forward(
            strategy_name=name,
            long_mask=long_m,
            short_mask=short_m,
            stop_atr_mult=stop_mult,
            target_atr_mult=tgt_mult,
            exit_time_str=exit_time,
        )
        results.append(eval_res)

    print("-" * 90)
    print(f"{'Strategy Name':<42} | {'Train PnL':>11} | {'OOS PnL':>11} | {'OOS PF':>7} | {'95% MC DD':>9} | {'Status':<12}")
    print("-" * 90)

    for r in results:
        train_pnl = f"₹{r['train']['net_pnl']:>8.2f}"
        oos_pnl = f"₹{r['test_oos']['net_pnl']:>8.2f}"
        oos_pf = f"{r['test_oos']['profit_factor']:>6.2f}"
        mc_dd = f"{r['mc_max_dd_pct']:>6.2f}%"
        status = r['status'].split()[0]
        print(f"{r['strategy']:<42} | {train_pnl} | {oos_pnl} | {oos_pf} | {mc_dd} | {status:<12}")

    print("-" * 90)
    print("\nDetailed Out-of-Sample Performance Audit:")
    for r in results:
        print(f"\n--- {r['strategy']} ---")
        print(f"  In-Sample  (Train): {r['train']['total_trades']} trades | WR: {r['train']['win_rate']:.1f}% | Net PnL: ₹{r['train']['net_pnl']:,.2f} | PF: {r['train']['profit_factor']:.2f}")
        print(f"  Blind Test (OOS)  : {r['test_oos']['total_trades']} trades | WR: {r['test_oos']['win_rate']:.1f}% | Net PnL: ₹{r['test_oos']['net_pnl']:,.2f} | PF: {r['test_oos']['profit_factor']:.2f}")
        print(f"  Full 60-Day Replay: {r['full_60d']['total_trades']} trades | WR: {r['full_60d']['win_rate']:.1f}% | Net PnL: ₹{r['full_60d']['net_pnl']:,.2f} | PF: {r['full_60d']['profit_factor']:.2f}")
        print(f"  95% Monte Carlo Max DD: {r['mc_max_dd_pct']:.2f}% | Evaluation: {r['status']}")

    t_total = time.time() - t0
    print("\n" + "=" * 90)
    print(f"  ALL 6 STRATEGIES OPTIMIZED & OUT-OF-SAMPLE VALIDATED IN {t_total:.2f} SECONDS TOTAL!")
    print("=" * 90)


if __name__ == "__main__":
    main()
