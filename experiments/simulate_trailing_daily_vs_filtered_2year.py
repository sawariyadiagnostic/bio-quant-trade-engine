"""
2-Year High-Resolution Trailing Stop (0.15% Trail / 0.25% Stop) Simulation Engine.
Evaluates NIFTYBEES and INFY.NS over 2 full years (475 sessions, Oct 2024 to Oct 2026).
Compares:
  1. Daily Unfiltered Trade + Trailing Stop (Trade Every Day)
  2. Global Lead-Lag Filtered Trade + Trailing Stop (Trade Only on Green Global Signal)
Across 1x, 2x, 3x, and 5x Leverage Tiers starting from INR 10,000.
Measures:
  - Exact Final Capital & Return %
  - Win Rate %
  - Maximum Drawdown %
  - Worst Single-Day Morning Loss at 5x Leverage (Checking user's ~1.25% target)
"""
import os
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

def load_aligned_data(
    india_path: str, 
    lead_path: str
) -> pd.DataFrame:
    """Loads and aligns Indian daily OHLC with global lead return."""
    df_ind = pd.read_parquet(india_path)
    df_lead = pd.read_parquet(lead_path)

    df_ind["dt"] = df_ind.index.strftime("%Y-%m-%d")
    df_lead["dt"] = df_lead.index.strftime("%Y-%m-%d")

    c_ind = df_ind["Close_Adj"] if "Close_Adj" in df_ind.columns else df_ind["Close"]
    o_ind = df_ind["Open"]
    h_ind = df_ind["High"]
    l_ind = df_ind["Low"]
    c_lead = df_lead["Close_Adj"] if "Close_Adj" in df_lead.columns else df_lead["Close"]

    lead_rets = (c_lead / c_lead.shift(1) - 1.0)
    lead_map = dict(zip(df_lead["dt"], lead_rets))

    records = []
    for t in range(len(df_ind) - 1):
        dt_today = df_ind["dt"].iloc[t]
        dt_next = df_ind["dt"].iloc[t + 1]

        if dt_today in lead_map and np.isfinite(lead_map[dt_today]):
            lead_r = float(lead_map[dt_today])
            c_today = float(c_ind.iloc[t])
            o_next = float(o_ind.iloc[t + 1])
            h_next = float(h_ind.iloc[t + 1])
            l_next = float(l_ind.iloc[t + 1])
            c_next = float(c_ind.iloc[t + 1])

            records.append({
                "date_t": dt_today,
                "date_next": dt_next,
                "c_today": c_today,
                "o_next": o_next,
                "h_next": h_next,
                "l_next": l_next,
                "c_next": c_next,
                "lead_ret": lead_r,
                "lead_green": lead_r > 0.0
            })

    return pd.DataFrame(records)

def simulate_trailing_run(
    df: pd.DataFrame,
    initial_cap: float = 10000.0,
    leverage: float = 1.0,
    friction: float = 0.00012,
    mode: str = "DAILY_UNFILTERED",
    trail_pct: float = 0.0015,   # 0.15% trailing buffer
    stop_pct: float = 0.0025     # 0.25% hard stop
) -> dict:
    """Executes trailing stop simulation for a given leverage tier and mode."""
    cap = initial_cap
    curve = [cap]
    trades = []
    skipped = 0
    worst_single_day_pnl_pct = 0.0

    for _, row in df.iterrows():
        lead_is_green = bool(row["lead_green"])
        if mode == "LEAD_FILTERED" and not lead_is_green:
            skipped += 1
            curve.append(cap)
            continue

        c_today = row["c_today"]
        o_next = row["o_next"]
        h_next = row["h_next"]
        l_next = row["l_next"]
        c_next = row["c_next"]

        gap = (o_next - c_today) / c_today
        max_h = (h_next - c_today) / c_today
        min_l = (l_next - c_today) / c_today
        day_c = (c_next - c_today) / c_today

        # Execution logic with 0.15% trail & 0.25% stop:
        if gap <= -stop_pct:
            # Gapped down below stop: filled at Opening Auction
            ret = gap
            exit_type = "AUCTION_GAP_CUT"
        elif gap < 0:
            # Opened slightly down: if dips below stop, cut at -stop_pct
            if min_l <= -stop_pct:
                ret = -stop_pct
                exit_type = "INTRADAY_STOP_CUT"
            else:
                ret = day_c
                exit_type = "CLOSE_EXIT"
        else:
            # Gapped up! Trail with 0.15% buffer
            bar_retrace = (h_next - l_next) / c_today
            trailing_exit = max_h - trail_pct
            if bar_retrace >= trail_pct and trailing_exit >= 0.0:
                ret = max(gap * 0.5, trailing_exit)
                exit_type = "TRAILING_LOCKED"
            elif bar_retrace >= trail_pct and trailing_exit < 0.0:
                ret = 0.0  # breakeven stop protected
                exit_type = "BREAKEVEN_CUT"
            else:
                ret = day_c
                exit_type = "RUN_TO_CLOSE"

        # Apply leverage, transaction friction, and margin interest
        net_ret = (ret * leverage) - (friction * leverage)
        if leverage > 1.0:
            margin_interest = (leverage - 1.0) * (0.09 / 365.0)
            net_ret -= margin_interest

        pnl = cap * net_ret
        cap = max(0.0, cap + pnl)
        curve.append(cap)

        worst_single_day_pnl_pct = min(worst_single_day_pnl_pct, net_ret * 100.0)

        trades.append({
            "date": row["date_next"],
            "ret_pct": round(ret * 100.0, 2),
            "net_ret_pct": round(net_ret * 100.0, 2),
            "exit_type": exit_type,
            "pnl": round(pnl, 2),
            "balance": round(cap, 2),
            "win": ret > 0.0
        })

    eq = np.array(curve)
    peaks = np.maximum.accumulate(eq)
    mdd = float(np.min((eq - peaks) / (peaks + 1e-6))) * 100.0
    wins = sum(1 for tr in trades if tr["win"])
    win_rate = (wins / len(trades)) * 100.0 if trades else 0.0

    return {
        "mode": mode,
        "leverage": leverage,
        "total_trades": len(trades),
        "skipped_sessions": skipped,
        "win_rate_pct": round(win_rate, 2),
        "final_capital": round(cap, 2),
        "net_profit": round(cap - initial_cap, 2),
        "return_pct": round(((cap - initial_cap) / initial_cap) * 100.0, 2),
        "max_drawdown_pct": round(mdd, 2),
        "worst_single_day_loss_pct": round(worst_single_day_pnl_pct, 2)
    }

def run_comparative_trailing_suite():
    # 1. NIFTYBEES with S&P 500 lead
    logger.info("Aligning NIFTYBEES with S&P 500...")
    df_nifty = load_aligned_data(
        india_path="data/lake/Indices/NIFTYBEES.parquet",
        lead_path="data/lake/Global/SP500.parquet"
    )

    # 2. INFY.NS with INFY ADR lead
    logger.info("Aligning INFY.NS with INFY ADR...")
    df_infy = load_aligned_data(
        india_path="data/lake/Information_Technology/INFY.parquet",
        lead_path="data/lake/Global/INFY_ADR.parquet"
    )

    assets = [
        {"name": "NIFTYBEES", "df": df_nifty, "friction": 0.00012},
        {"name": "INFY.NS", "df": df_infy, "friction": 0.00025}
    ]

    all_results = {}
    for a in assets:
        sym = a["name"]
        df_a = a["df"]
        fric = a["friction"]
        all_results[sym] = {"daily_unfiltered": {}, "lead_filtered": {}}

        for lev in [1.0, 2.0, 3.0, 5.0]:
            # Mode A: Daily Unfiltered + Trailing Stop
            res_daily = simulate_trailing_run(
                df_a, initial_cap=10000.0, leverage=lev, friction=fric,
                mode="DAILY_UNFILTERED", trail_pct=0.0015, stop_pct=0.0025
            )
            all_results[sym]["daily_unfiltered"][f"{lev:.0f}x"] = res_daily

            # Mode B: Lead Filtered + Trailing Stop
            res_filt = simulate_trailing_run(
                df_a, initial_cap=10000.0, leverage=lev, friction=fric,
                mode="LEAD_FILTERED", trail_pct=0.0015, stop_pct=0.0025
            )
            all_results[sym]["lead_filtered"][f"{lev:.0f}x"] = res_filt

    os.makedirs("audit", exist_ok=True)
    out_file = "audit/trailing_015_025_2year_audit.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, default=_np_serial)
    logger.info(f"Saved trailing stop audit to {out_file}")
    return all_results

if __name__ == "__main__":
    results = run_comparative_trailing_suite()
    
    print("\n" + "="*95)
    print("2-YEAR COMPARATIVE TRAILING STOP AUDIT (0.15% TRAIL / 0.25% STOP)")
    print("Comparing: DAILY UNFILTERED vs GLOBAL LEAD FILTERED (475 Sessions, Oct 2024 - Oct 2026)")
    print("="*95)

    for sym, modes in results.items():
        print(f"\n{'='*40} ASSET: {sym} {'='*40}")
        print(f"{'Mode & Leverage':25} | {'Trades':9} | {'Win Rate':9} | {'Final Capital':16} | {'Return %':11} | {'MDD':8} | {'Worst Day Hit':14}")
        print("-"*100)
        
        # Daily Unfiltered
        for lev_key, d in modes["daily_unfiltered"].items():
            lbl = f"Daily Unfiltered {lev_key}"
            print(f"{lbl:25} | {d['total_trades']:3d}/475  | {d['win_rate_pct']:6.1f}%  | INR {d['final_capital']:11,.2f} | {d['return_pct']:+9.1f}% | {d['max_drawdown_pct']:6.2f}% | {d['worst_single_day_loss_pct']:+11.2f}%")

        print("."*100)
        # Lead Filtered
        for lev_key, d in modes["lead_filtered"].items():
            lbl = f"Lead-Filtered {lev_key}"
            print(f"{lbl:25} | {d['total_trades']:3d}/475  | {d['win_rate_pct']:6.1f}%  | INR {d['final_capital']:11,.2f} | {d['return_pct']:+9.1f}% | {d['max_drawdown_pct']:6.2f}% | {d['worst_single_day_loss_pct']:+11.2f}%")
