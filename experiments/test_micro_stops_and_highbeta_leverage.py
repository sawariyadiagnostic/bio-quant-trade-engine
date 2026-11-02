"""
Empirical Micro-Stop Loss & High-Beta 5x Leverage Stress Testing Engine.
Answers two critical quantitative questions with real 1-year market data:
1. What happens if Stop-Loss & Trailing Gap are set ultra-tight (0.05%, 0.10%, 0.15%, 0.25%, 0.50%)?
   - Evaluates bid-ask noise whipsaw rate, gap slippage, and net P&L.
2. Can we apply 5x leverage on high-beta jumpers (HAL, MOTHERSON, DIXON) with trailing stops?
   - Measures maximum gap-down shock, drawdown, compounding P&L, and liquidation risk.
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

def load_1year_df(path: str, bars: int = 250) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df.iloc[-bars:].copy()

def test_stop_loss_sweep_niftybees(bars: int = 250) -> dict:
    """
    Sweeps stop-loss and trailing gap thresholds on NIFTYBEES across:
    [0.05%, 0.10%, 0.15%, 0.20%, 0.25%, 0.50%, 1.00%]
    Evaluates whipsaw frequency, real auction gap execution, and final capital.
    """
    df = load_1year_df("data/lake/Indices/NIFTYBEES.parquet", bars=bars)
    c = df["Close_Adj"] if "Close_Adj" in df.columns else df["Close"]
    o = df["Open"]
    h = df["High"]
    l = df["Low"]

    thresholds = [0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0050, 0.0100]
    sweep_results = []
    friction = 0.00012  # 1.2 bps ETF friction

    for th in thresholds:
        th_pct = th * 100.0
        cap = 10000.0
        trades = []
        whipsaws = 0

        for t in range(len(df) - 1):
            c_buy = float(c.iloc[t])
            o_sell = float(o.iloc[t+1])
            h_sell = float(h.iloc[t+1])
            l_sell = float(l.iloc[t+1])
            c_sell = float(c.iloc[t+1])

            gap = (o_sell - c_buy) / c_buy
            max_h = (h_sell - c_buy) / c_buy
            min_l = (l_sell - c_buy) / c_buy

            # Execution logic:
            if gap <= -th:
                # Gapped down below stop: Real execution happens at OPEN price!
                ret = gap
                exit_reason = "GAP_BELOW_STOP (AUCTION_FILL)"
            elif gap < 0:
                # Opened slightly down: if price dipped further to stop, stopped out
                if min_l <= -th:
                    ret = -th
                    exit_reason = "INTRADAY_STOPPED_OUT"
                    # Did price rebound later to positive? If so, this is a WHIPSAW!
                    if max_h > 0.002:
                        whipsaws += 1
                else:
                    ret = (c_sell - c_buy) / c_buy
                    exit_reason = "CLOSE_EXIT"
            else:
                # Gapped up! Trail with buffer = th
                bar_retrace = (h_sell - l_sell) / c_buy
                peak_ret = max_h
                trail_trigger = peak_ret - th
                if bar_retrace >= th:
                    # Stopped out by trailing retrace
                    ret = max(gap * 0.5, trail_trigger)
                    exit_reason = "TRAILING_STOP_HIT"
                    if max_h > ret + 0.003:
                        whipsaws += 1
                else:
                    ret = (c_sell - c_buy) / c_buy
                    exit_reason = "RUN_TO_CLOSE"

            net_ret = (1.0 + ret) * (1.0 - friction) - 1.0
            pnl = cap * net_ret
            cap += pnl

            trades.append({
                "ret": ret,
                "net_ret": net_ret,
                "exit_reason": exit_reason,
                "win": ret > 0.0
            })

        wins = sum(1 for tr in trades if tr["win"])
        win_rate = (wins / len(trades)) * 100.0
        whipsaw_rate = (whipsaws / len(trades)) * 100.0

        sweep_results.append({
            "threshold_pct": round(th_pct, 2),
            "threshold_bps": round(th * 10000, 1),
            "final_capital": round(cap, 2),
            "net_return_pct": round(((cap - 10000.0) / 10000.0) * 100.0, 2),
            "win_rate_pct": round(win_rate, 2),
            "whipsaw_count": whipsaws,
            "whipsaw_rate_pct": round(whipsaw_rate, 2)
        })

    return {
        "asset": "NIFTYBEES",
        "sweep": sweep_results
    }

def simulate_highbeta_leverage(
    symbol: str, 
    parquet_path: str, 
    friction: float = 0.00025,
    bars: int = 250
) -> dict:
    """
    Evaluates a volatile stock under 1x, 2x, 3x, and 5x leverage.
    Compares:
      1. Pure Open Autoseller
      2. Dynamic Trailing Stop (0.3% Trail, 0.5% Hard Stop)
    """
    df = load_1year_df(parquet_path, bars=bars)
    c = df["Close_Adj"] if "Close_Adj" in df.columns else df["Close"]
    o = df["Open"]
    h = df["High"]
    l = df["Low"]

    # Calculate worst single overnight gap down
    all_gaps = [(float(o.iloc[i+1]) - float(c.iloc[i])) / float(c.iloc[i]) for i in range(len(df)-1)]
    worst_gap_down = float(np.min(all_gaps)) * 100.0
    avg_gap = float(np.mean(all_gaps)) * 100.0

    bnh_ret = float((c.iloc[-1] / c.iloc[0]) - 1.0) * 100.0

    leverage_tiers = [1.0, 2.0, 3.0, 5.0]
    results_by_lev = {}

    for lev in leverage_tiers:
        # Mode A: Pure Open Autoseller
        cap_open = 10000.0
        curve_open = [cap_open]
        max_single_day_loss_open = 0.0

        # Mode B: Dynamic Trailing Stop (0.3% Trail / 0.5% Hard Stop)
        cap_trail = 10000.0
        curve_trail = [cap_trail]
        max_single_day_loss_trail = 0.0

        for t in range(len(df) - 1):
            c_buy = float(c.iloc[t])
            o_sell = float(o.iloc[t+1])
            h_sell = float(h.iloc[t+1])
            l_sell = float(l.iloc[t+1])
            c_sell = float(c.iloc[t+1])

            gap = (o_sell - c_buy) / c_buy
            max_h = (h_sell - c_buy) / c_buy
            min_l = (l_sell - c_buy) / c_buy

            # Mode A: Pure Open
            ret_open = gap
            net_ret_open = (ret_open * lev) - (friction * lev)
            if lev > 1.0:
                net_ret_open -= (lev - 1.0) * (0.09 / 365.0) # margin interest
            cap_open = max(0.0, cap_open * (1.0 + net_ret_open))
            curve_open.append(cap_open)
            max_single_day_loss_open = min(max_single_day_loss_open, net_ret_open * 100.0)

            # Mode B: Dynamic Trailing (Trail: 0.3%, Stop: 0.5%)
            trail_buf = 0.003
            hard_stop = 0.005

            if gap <= -hard_stop:
                ret_tr = gap  # filled at auction
            elif gap < 0:
                ret_tr = -hard_stop if min_l <= -hard_stop else (c_sell - c_buy) / c_buy
            else:
                bar_range = (h_sell - l_sell) / c_buy
                if bar_range >= trail_buf:
                    ret_tr = max(gap * 0.5, max_h - trail_buf)
                else:
                    ret_tr = (c_sell - c_buy) / c_buy

            net_ret_tr = (ret_tr * lev) - (friction * lev)
            if lev > 1.0:
                net_ret_tr -= (lev - 1.0) * (0.09 / 365.0)
            cap_trail = max(0.0, cap_trail * (1.0 + net_ret_tr))
            curve_trail.append(cap_trail)
            max_single_day_loss_trail = min(max_single_day_loss_trail, net_ret_tr * 100.0)

        # Max drawdowns
        eq_open = np.array(curve_open)
        mdd_open = float(np.min((eq_open - np.maximum.accumulate(eq_open)) / (np.maximum.accumulate(eq_open) + 1e-6))) * 100.0

        eq_tr = np.array(curve_trail)
        mdd_tr = float(np.min((eq_tr - np.maximum.accumulate(eq_tr)) / (np.maximum.accumulate(eq_tr) + 1e-6))) * 100.0

        results_by_lev[f"{lev:.0f}x"] = {
            "pure_open": {
                "final_capital": round(cap_open, 2),
                "return_pct": round(((cap_open - 10000.0) / 10000.0) * 100.0, 2),
                "max_drawdown_pct": round(mdd_open, 2),
                "worst_single_day_loss_pct": round(max_single_day_loss_open, 2)
            },
            "dynamic_trail": {
                "final_capital": round(cap_trail, 2),
                "return_pct": round(((cap_trail - 10000.0) / 10000.0) * 100.0, 2),
                "max_drawdown_pct": round(mdd_tr, 2),
                "worst_single_day_loss_pct": round(max_single_day_loss_trail, 2)
            }
        }

    return {
        "symbol": symbol,
        "bars": len(df),
        "start_date": df.index[0].strftime("%Y-%m-%d"),
        "end_date": df.index[-1].strftime("%Y-%m-%d"),
        "buy_and_hold_return_pct": round(bnh_ret, 2),
        "average_overnight_gap_pct": round(avg_gap, 2),
        "worst_overnight_gap_down_pct": round(worst_gap_down, 2),
        "max_safe_leverage_ceiling": round(50.0 / abs(worst_gap_down), 1),
        "leverage_results": results_by_lev
    }

def run_all_stress_tests():
    logger.info("Running Micro-Stop Loss Sweep on NIFTYBEES...")
    sweep_nifty = test_stop_loss_sweep_niftybees(bars=250)

    logger.info("Running 5x Leverage Stress Tests on Volatile Jumpers...")
    stocks = [
        {"name": "HAL", "path": "data/lake/Capital_Goods_Defense/HAL.parquet", "friction": 0.00025},
        {"name": "MOTHERSON", "path": "data/lake/Automotive/MOTHERSON.parquet", "friction": 0.00025},
        {"name": "DIXON", "path": "data/lake/Consumer_Durables_Retail/DIXON.parquet", "friction": 0.00025},
        {"name": "NIFTYBEES", "path": "data/lake/Indices/NIFTYBEES.parquet", "friction": 0.00012}
    ]

    stock_results = {}
    for s in stocks:
        stock_results[s["name"]] = simulate_highbeta_leverage(s["name"], s["path"], s["friction"], bars=250)

    final_payload = {
        "micro_stop_sweep": sweep_nifty,
        "highbeta_leverage_stress": stock_results
    }

    os.makedirs("audit", exist_ok=True)
    out_file = "audit/micro_stops_and_highbeta_5x_audit.json"
    with open(out_file, "w") as f:
        json.dump(final_payload, f, indent=2, default=_np_serial)
    logger.info(f"Saved stress test audit to {out_file}")
    return final_payload

if __name__ == "__main__":
    data = run_all_stress_tests()
    
    print("\n" + "="*80)
    print("PART 1: MICRO-STOP LOSS THRESHOLD SWEEP (NIFTYBEES - 1 YEAR)")
    print("="*80)
    print(f"{'Threshold':15} | {'Final Capital':16} | {'Net Return':12} | {'Win Rate':10} | {'Whipsaws (Premature Stop)':25}")
    print("-"*85)
    for row in data["micro_stop_sweep"]["sweep"]:
        print(f"{row['threshold_pct']:5.2f}% ({row['threshold_bps']:4.1f} bps)  | INR {row['final_capital']:11,.2f} | {row['net_return_pct']:+9.2f}% | {row['win_rate_pct']:7.2f}% | {row['whipsaw_count']:3d} trades ({row['whipsaw_rate_pct']:5.1f}%)")

    print("\n" + "="*80)
    print("PART 2: HIGH-BETA VOLATILE STOCKS UNDER 1x, 2x, 3x, 5x LEVERAGE")
    print("="*80)
    for sym, res in data["highbeta_leverage_stress"].items():
        print(f"\n>>> ASSET: {sym} (Buy & Hold 1-Year: {res['buy_and_hold_return_pct']}%)")
        print(f"    Worst Single Overnight Gap Down: {res['worst_overnight_gap_down_pct']}% | Max Safe Leverage: {res['max_safe_leverage_ceiling']}x")
        print(f"    {'Tier':6} | {'Pure Open Final':18} | {'Pure MDD':10} | {'Worst Day Hit':14} | {'Trail Final':18} | {'Trail MDD':10}")
        print("    " + "-"*80)
        for lev_key, ldata in res["leverage_results"].items():
            po = ldata["pure_open"]
            tr = ldata["dynamic_trail"]
            print(f"    {lev_key:6} | INR {po['final_capital']:12,.2f} ({po['return_pct']:+7.1f}%) | {po['max_drawdown_pct']:7.2f}% | {po['worst_single_day_loss_pct']:+11.2f}% | INR {tr['final_capital']:12,.2f} ({tr['return_pct']:+7.1f}%) | {tr['max_drawdown_pct']:7.2f}%")
