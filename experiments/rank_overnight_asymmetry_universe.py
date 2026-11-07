"""
Institutional Overnight Asymmetry & Downside Risk Screening Engine.
Screens all 95+ assets across the entire data lake (Equities, ETFs, Indices, Global Titans)
to identify the single most optimal asset where:
  - Overnight upside jump is MAXIMIZED
  - Overnight downside loss is MINIMIZED
  - Profit Factor, Asymmetry Ratio, and Net Expectancy are HIGHEST.
Evaluates both 1-Year (250 bars) and Multi-Year full historical lake.
"""
import os
import glob
import json
import numpy as np
import pandas as pd
from loguru import logger

def _np_serial(obj):
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.ndarray, pd.Series)):
        return obj.tolist()
    return str(obj)

def get_asset_friction(path: str, sym: str) -> float:
    """Returns statutory round-trip friction based on asset category."""
    if "Indices" in path or "BEES" in sym:
        return 0.00012  # 1.2 bps (ETF statutory STT is 0.001%)
    elif "Global" in path or sym in ["NVDA", "AAPL", "TSLA", "KO"]:
        return 0.00005  # 0.5 bps (US Equities, zero STT/stamp)
    else:
        return 0.00025  # 2.5 bps (Single-Stock Futures / MTF equivalent)

def evaluate_asset_overnight_asymmetry(
    sym: str, 
    path: str, 
    bars: int = 250
) -> dict | None:
    """Computes exact overnight asymmetry metrics over the last `bars` sessions."""
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        logger.warning(f"Failed to read {path}: {e}")
        return None

    if len(df) < min(60, bars):
        return None

    work_df = df.iloc[-bars:].copy() if len(df) >= bars else df.copy()
    c = work_df["Close_Adj"] if "Close_Adj" in work_df.columns else work_df["Close"]
    o = work_df["Open"]

    c_today = c.iloc[:-1].to_numpy()
    o_next = o.iloc[1:].to_numpy()

    # Discard non-finite values
    valid_mask = np.isfinite(c_today) & np.isfinite(o_next) & (c_today > 0)
    c_today = c_today[valid_mask]
    o_next = o_next[valid_mask]

    if len(c_today) < 40:
        return None

    gaps = (o_next - c_today) / c_today
    pos_gaps = gaps[gaps > 0]
    neg_gaps = gaps[gaps < 0]

    n_total = len(gaps)
    n_win = len(pos_gaps)
    n_loss = len(neg_gaps)

    win_rate = (n_win / n_total) * 100.0 if n_total > 0 else 0.0
    avg_pos_gap = float(np.mean(pos_gaps)) * 100.0 if len(pos_gaps) > 0 else 0.0
    avg_neg_gap = float(np.mean(neg_gaps)) * 100.0 if len(neg_gaps) > 0 else 0.0
    worst_gap_down = float(np.min(gaps)) * 100.0 if len(gaps) > 0 else 0.0
    best_gap_up = float(np.max(gaps)) * 100.0 if len(gaps) > 0 else 0.0

    sum_gains = float(np.sum(pos_gaps)) if len(pos_gaps) > 0 else 0.0
    sum_losses = float(np.sum(np.abs(neg_gaps))) if len(neg_gaps) > 0 else 1e-6
    profit_factor = sum_gains / sum_losses

    asymmetry_ratio = avg_pos_gap / (abs(avg_neg_gap) + 1e-6)

    friction = get_asset_friction(path, sym)
    # Net Expectancy in bps
    p_w = win_rate / 100.0
    net_exp_bps = ((p_w * avg_pos_gap) - ((1.0 - p_w) * abs(avg_neg_gap)) - (friction * 100.0)) * 100.0

    # Compounding net 1-year return (INR 10,000 start)
    net_daily_rets = (1.0 + gaps) * (1.0 - friction) - 1.0
    cum_curve = np.cumprod(1.0 + net_daily_rets)
    final_cap_10k = 10000.0 * float(cum_curve[-1]) if len(cum_curve) > 0 else 10000.0
    net_compounded_ret = ((final_cap_10k - 10000.0) / 10000.0) * 100.0

    # Downside Sortino of gaps
    down_std = float(np.std(neg_gaps)) if len(neg_gaps) > 0 else 0.01
    overnight_sortino = (float(np.mean(gaps)) * np.sqrt(252)) / (down_std + 1e-6)

    # Composite Institutional Optimality Score:
    # High win rate, high profit factor, high net expectancy, low worst-case gap down
    optimality_score = (profit_factor * 0.35) + (asymmetry_ratio * 0.25) + (max(0.0, net_exp_bps) / 10.0 * 0.25) + (min(5.0, 50.0 / (abs(worst_gap_down) + 1e-3)) * 0.15)

    return {
        "symbol": sym,
        "path": path,
        "bars_tested": n_total,
        "win_rate_pct": round(win_rate, 2),
        "avg_jump_bps": round(avg_pos_gap * 100.0, 1),
        "avg_drop_bps": round(abs(avg_neg_gap) * 100.0, 1),
        "worst_gap_down_pct": round(worst_gap_down, 2),
        "best_gap_up_pct": round(best_gap_up, 2),
        "asymmetry_ratio": round(asymmetry_ratio, 2),
        "profit_factor": round(profit_factor, 2),
        "net_exp_bps_per_night": round(net_exp_bps, 1),
        "overnight_sortino": round(overnight_sortino, 2),
        "final_capital_10k": round(final_cap_10k, 2),
        "net_compounded_1y_pct": round(net_compounded_ret, 2),
        "optimality_score": round(optimality_score, 2)
    }

def scan_entire_universe(lake_dir: str = "data/lake", bars: int = 250) -> list[dict]:
    """Scans every single parquet file in the lake."""
    all_files = glob.glob(f"{lake_dir}/**/*.parquet", recursive=True)
    logger.info(f"Found {len(all_files)} total parquet datasets in {lake_dir}.")

    results: list[dict] = []
    for f in all_files:
        sym = os.path.splitext(os.path.basename(f))[0]
        metrics = evaluate_asset_overnight_asymmetry(sym, f, bars=bars)
        if metrics is not None:
            results.append(metrics)

    # Sort descending by composite optimality score
    results.sort(key=lambda x: x["optimality_score"], reverse=True)
    return results

def run_overnight_ranking_experiment():
    # 1. Screen over 1 Year (250 bars)
    ranked_1y = scan_entire_universe("data/lake", bars=250)
    
    # 2. Screen over Full 5-Year History (1235 bars)
    ranked_5y = scan_entire_universe("data/lake", bars=1235)

    payload = {
        "ranked_1year_top25": ranked_1y[:25],
        "ranked_5year_top25": ranked_5y[:25],
        "total_assets_scanned": len(ranked_1y)
    }

    os.makedirs("audit", exist_ok=True)
    out_file = "audit/overnight_asymmetry_ranking.json"
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2, default=_np_serial)
    logger.info(f"Persisted ranking audit to {out_file}")
    return payload

if __name__ == "__main__":
    data = run_overnight_ranking_experiment()
    
    print("\n" + "="*95)
    print("TOP 15 OPTIMAL OVERNIGHT ASYMMETRY ASSETS (1-YEAR SCREENING - 250 SESSIONS)")
    print("="*95)
    print(f"{'Rank':4} | {'Symbol':12} | {'Win Rate':9} | {'Avg Jump':10} | {'Avg Drop':10} | {'Worst Gap':10} | {'Profit Factor':13} | {'1Y INR 10k':14} | {'Score':6}")
    print("-"*95)
    for i, r in enumerate(data["ranked_1year_top25"][:15], 1):
        print(f"#{i:2d}  | {r['symbol']:12} | {r['win_rate_pct']:6.1f}%  | +{r['avg_jump_bps']:5.1f} bps | -{r['avg_drop_bps']:5.1f} bps | {r['worst_gap_down_pct']:6.2f}%   | {r['profit_factor']:13.2f} | INR {r['final_capital_10k']:10,.2f} | {r['optimality_score']:5.2f}")

    print("\n" + "="*95)
    print("TOP 10 MULTI-YEAR ROBUST OVERNIGHT ASYMMETRY ASSETS (5-YEAR SCREENING)")
    print("="*95)
    print(f"{'Rank':4} | {'Symbol':12} | {'Win Rate':9} | {'Avg Jump':10} | {'Avg Drop':10} | {'Worst Gap':10} | {'Profit Factor':13} | {'5Y Return':12} | {'Sortino':7}")
    print("-"*95)
    for i, r in enumerate(data["ranked_5year_top25"][:10], 1):
        print(f"#{i:2d}  | {r['symbol']:12} | {r['win_rate_pct']:6.1f}%  | +{r['avg_jump_bps']:5.1f} bps | -{r['avg_drop_bps']:5.1f} bps | {r['worst_gap_down_pct']:6.2f}%   | {r['profit_factor']:13.2f} | {r['net_compounded_1y_pct']:+10.1f}% | {r['overnight_sortino']:7.2f}")
