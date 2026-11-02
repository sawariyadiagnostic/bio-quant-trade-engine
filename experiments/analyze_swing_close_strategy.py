"""
Comprehensive Analysis of Close-to-Next-Day & Swing Holding Strategies across 81 NSE Equities.
Compares:
  1. Daily Close-to-Close (Every Day Turnover: Buy Close[t], Sell Close[t+1] every day with 0.25% daily friction)
  2. Overnight Only (Buy Close[t], Sell Open[t+1])
  3. Swing Trend Holding (Buy Close[t] when in Uptrend/Active Regime, Hold multi-day, Sell Close when trend ends)
     - Captures overnight gaps + daytime drift with friction paid ONLY ONCE per multi-day swing cycle!
  4. Passive Buy & Hold Benchmark
"""
import os
import sys
import numpy as np
import pandas as pd
from tabulate import tabulate
from loguru import logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.manifold.data_loader import NSE_SECTORS

def analyze_swing_holding(friction_pct: float = 0.0025):
    lake_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "lake"))
    
    all_results = []
    sector_summary = {}

    for sector, tickers in NSE_SECTORS.items():
        sector_results = []
        for ticker in tickers:
            sym = ticker.replace(".NS", "")
            parquet_path = os.path.join(lake_dir, sector, f"{sym}.parquet")
            if not os.path.exists(parquet_path):
                continue
            
            df = pd.read_parquet(parquet_path)
            if len(df) < 60:
                continue

            c = df["Close_Adj"] if "Close_Adj" in df.columns else df["Close_Raw"]
            o = df["Open"]
            h = df["High"]
            l = df["Low"]

            # Daily Close-to-Close returns
            daily_rets = c.pct_change().dropna().to_numpy()
            n_days = len(daily_rets)

            # 1. Unconditional Daily Close-to-Close (Turnover every day)
            # Gross is identical to Buy & Hold
            gross_bnh = float((c.iloc[-1] / c.iloc[0]) - 1.0)
            # Net daily turnover: each day loses friction
            net_daily_turnover_curve = np.cumprod(np.maximum(0.0, (1.0 + daily_rets) * (1.0 - friction_pct)))
            net_daily_turnover_ret = float(net_daily_turnover_curve[-1] - 1.0) if len(net_daily_turnover_curve) > 0 else -1.0

            # 2. Overnight Only: Buy Close[t], Sell Open[t+1]
            c_today = c.iloc[:-1].to_numpy()
            o_tomorrow = o.iloc[1:].to_numpy()
            on_rets = (o_tomorrow - c_today) / c_today
            gross_on_ret = float(np.prod(1.0 + on_rets) - 1.0)

            # 3. Active Swing Holding (Close-to-Close with Structural Trend Filter)
            # Condition: Price > EMA20 and EMA10 > EMA20 (Structural Swing Regime)
            ema_10 = c.ewm(span=10, adjust=False).mean()
            ema_20 = c.ewm(span=20, adjust=False).mean()
            swing_signal = (c > ema_20) & (ema_10 > ema_20)

            # Simulate Multi-Day Swing Execution:
            # Enter at Close[t] when signal turns True (pay half friction: buy STT 0.1% + stamp duty 0.015%)
            # Hold while signal is True
            # Exit at Close[t+k] when signal turns False (pay half friction: sell STT 0.1% + exchange fee)
            in_trade = False
            trade_count = 0
            swing_equity = [1.0]
            entry_price = 0.0
            swing_trade_pnls = []
            trade_durations = []
            current_duration = 0

            for t in range(20, len(c)):
                sig_today = swing_signal.iloc[t]
                price_today = c.iloc[t]

                if not in_trade and sig_today:
                    # Enter swing trade at Close
                    in_trade = True
                    entry_price = price_today
                    trade_count += 1
                    current_duration = 1
                    # Deduct buy friction (~0.12%)
                    swing_equity.append(swing_equity[-1] * (1.0 - friction_pct / 2.0))
                elif in_trade and sig_today:
                    # Continue holding swing: accumulate daily Close-to-Close return (NO TAX DEDUCTED)
                    ret_today = (price_today - c.iloc[t-1]) / c.iloc[t-1]
                    swing_equity.append(swing_equity[-1] * (1.0 + ret_today))
                    current_duration += 1
                elif in_trade and not sig_today:
                    # Exit swing trade at Close
                    ret_today = (price_today - c.iloc[t-1]) / c.iloc[t-1]
                    # Accrue final day return minus sell friction (~0.13%)
                    final_eq = swing_equity[-1] * (1.0 + ret_today) * (1.0 - friction_pct / 2.0)
                    swing_equity.append(final_eq)
                    trade_pnl = (price_today - entry_price) / entry_price - friction_pct
                    swing_trade_pnls.append(trade_pnl)
                    trade_durations.append(current_duration)
                    in_trade = False
                    current_duration = 0
                else:
                    # In cash: 0% return
                    swing_equity.append(swing_equity[-1])

            net_swing_ret = float(swing_equity[-1] - 1.0)
            swing_win_rate = float(np.mean(np.array(swing_trade_pnls) > 0.0) * 100) if swing_trade_pnls else 0.0
            avg_swing_days = float(np.mean(trade_durations)) if trade_durations else 0.0
            
            # Max Drawdown of Swing Holding
            sw_arr = np.array(swing_equity)
            sw_peaks = np.maximum.accumulate(sw_arr)
            sw_dds = (sw_arr - sw_peaks) / sw_peaks
            sw_mdd = float(np.min(sw_dds)) if len(sw_dds) > 0 else 0.0

            res = {
                "ticker": sym,
                "sector": sector,
                "bnh_ret": gross_bnh,
                "daily_turnover_net": net_daily_turnover_ret,
                "gross_on_ret": gross_on_ret,
                "net_swing_ret": net_swing_ret,
                "swing_trades": trade_count,
                "avg_swing_days": avg_swing_days,
                "swing_win_rate": swing_win_rate,
                "swing_mdd": sw_mdd
            }
            all_results.append(res)
            sector_results.append(res)

        if sector_results:
            sector_summary[sector] = {
                "count": len(sector_results),
                "avg_bnh": float(np.mean([r["bnh_ret"] for r in sector_results])),
                "avg_daily_turnover_net": float(np.mean([r["daily_turnover_net"] for r in sector_results])),
                "avg_gross_on": float(np.mean([r["gross_on_ret"] for r in sector_results])),
                "avg_net_swing": float(np.mean([r["net_swing_ret"] for r in sector_results])),
                "avg_trades": float(np.mean([r["swing_trades"] for r in sector_results])),
                "avg_days": float(np.mean([r["avg_swing_days"] for r in sector_results])),
                "avg_win_rate": float(np.mean([r["swing_win_rate"] for r in sector_results])),
                "avg_mdd": float(np.mean([r["swing_mdd"] for r in sector_results]))
            }

    # 1. Print Sector Level Comparison
    print("\n" + "=" * 125)
    print("   STRATEGY COMPARISON: DAILY CLOSE TURNOVER vs OVERNIGHT vs SWING HOLDING (12 SECTORS)")
    print("=" * 125)
    sec_rows = []
    for sec, data in sorted(sector_summary.items(), key=lambda x: x[1]["avg_net_swing"], reverse=True):
        sec_rows.append([
            sec,
            data["count"],
            f"{data['avg_bnh']*100:+8.1f}%",
            f"{data['avg_daily_turnover_net']*100:+8.1f}%",
            f"{data['avg_gross_on']*100:+8.1f}%",
            f"{data['avg_net_swing']*100:+8.1f}%",
            f"{data['avg_trades']:.1f}",
            f"{data['avg_days']:.1f}d",
            f"{data['avg_win_rate']:5.1f}%",
            f"{data['avg_mdd']*100:5.1f}%"
        ])
    sec_headers = ["Sector", "Stocks", "Buy & Hold", "Daily Turnover Net", "Gross Overnight", "Net Swing Holding", "Swings", "Avg Hold", "Win %", "Swing MDD"]
    print(tabulate(sec_rows, headers=sec_headers, tablefmt="grid"))

    # 2. Print Top 20 Best Equities for Swing Holding Strategy
    print("\n" + "=" * 125)
    print("   TOP 20 BEST HIGH-NET RETURN EQUITIES FOR SWING HOLDING (CLOSE-TO-CLOSE REGIME SWING)")
    print("=" * 125)
    top_swing = sorted(all_results, key=lambda x: x["net_swing_ret"], reverse=True)[:20]
    top_rows = []
    for r in top_swing:
        top_rows.append([
            r["ticker"],
            r["sector"],
            f"{r['bnh_ret']*100:+8.1f}%",
            f"{r['daily_turnover_net']*100:+8.1f}%",
            f"{r['net_swing_ret']*100:+8.1f}%",
            f"{r['swing_trades']}",
            f"{r['avg_swing_days']:.1f} days",
            f"{r['swing_win_rate']:5.1f}%",
            f"{r['swing_mdd']*100:5.1f}%"
        ])
    top_headers = ["Ticker", "Sector", "Buy & Hold", "Daily Turnover Net", "Net Swing Holding", "Swings", "Avg Hold", "Win %", "Swing Max DD"]
    print(tabulate(top_rows, headers=top_headers, tablefmt="grid"))

if __name__ == "__main__":
    analyze_swing_holding()
