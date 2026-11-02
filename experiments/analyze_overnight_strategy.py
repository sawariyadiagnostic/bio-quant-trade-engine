"""
Comprehensive Overnight Strategy Evaluation:
Buys at Close of Day t, Sells at Open of Day t+1 across all 81 NSE Equities in 12 Sectors.
Compares:
  1. Gross Overnight Return (Theoretical zero friction)
  2. Net Overnight Return (Indian Statutory Friction: STT 0.1% x 2, Stamp Duty 0.015%, Exchange + GST = ~0.25% / round trip)
  3. Intraday Return (Open to Close of same day)
  4. Standard Buy & Hold Return
"""
import os
import sys
import numpy as np
import pandas as pd
from tabulate import tabulate
from loguru import logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.manifold.data_loader import NSE_SECTORS

def evaluate_overnight_strategy(friction_pct: float = 0.0025):
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

            # Align overnight: Buy at C[t], sell at O[t+1]
            c_today = c.iloc[:-1].to_numpy()
            o_tomorrow = o.iloc[1:].to_numpy()

            # Overnight returns
            overnight_rets = (o_tomorrow - c_today) / c_today
            overnight_gross_curve = np.cumprod(1.0 + overnight_rets)
            gross_overnight_ret = float(overnight_gross_curve[-1] - 1.0)

            # Net overnight returns with friction (STT, Stamp Duty, GST, exchange)
            net_overnight_rets = ((1.0 + overnight_rets) * (1.0 - friction_pct)) - 1.0
            # Clip at -1.0 to prevent negative portfolio
            net_overnight_curve = np.cumprod(np.maximum(0.0, 1.0 + net_overnight_rets))
            net_overnight_ret = float(net_overnight_curve[-1] - 1.0) if len(net_overnight_curve) > 0 else -1.0

            # Intraday returns: Buy at O[t], sell at C[t]
            intraday_rets = (c - o) / o
            intraday_gross_curve = np.cumprod(1.0 + intraday_rets.dropna())
            intraday_ret = float(intraday_gross_curve[-1] - 1.0)

            # Buy and Hold return: C[last] / C[first] - 1
            bnh_ret = float((c.iloc[-1] / c.iloc[0]) - 1.0)

            # Win rate & metrics
            win_rate = float(np.mean(overnight_rets > 0.0) * 100)
            avg_bps = float(np.mean(overnight_rets) * 10000) # basis points
            
            # Overnight Drawdown
            peaks = np.maximum.accumulate(overnight_gross_curve)
            dds = (overnight_gross_curve - peaks) / peaks
            mdd = float(np.min(dds)) if len(dds) > 0 else 0.0

            res = {
                "ticker": sym,
                "sector": sector,
                "days": len(overnight_rets),
                "gross_overnight_ret": gross_overnight_ret,
                "net_overnight_ret": net_overnight_ret,
                "intraday_ret": intraday_ret,
                "bnh_ret": bnh_ret,
                "win_rate": win_rate,
                "avg_bps": avg_bps,
                "overnight_mdd": mdd
            }
            all_results.append(res)
            sector_results.append(res)

        if sector_results:
            sector_summary[sector] = {
                "count": len(sector_results),
                "avg_gross_on": float(np.mean([r["gross_overnight_ret"] for r in sector_results])),
                "avg_net_on": float(np.mean([r["net_overnight_ret"] for r in sector_results])),
                "avg_intraday": float(np.mean([r["intraday_ret"] for r in sector_results])),
                "avg_bnh": float(np.mean([r["bnh_ret"] for r in sector_results])),
                "avg_win_rate": float(np.mean([r["win_rate"] for r in sector_results])),
                "avg_bps": float(np.mean([r["avg_bps"] for r in sector_results]))
            }

    # 1. Print Sector Summary Table
    print("\n" + "=" * 115)
    print("   OVERNIGHT STRATEGY (BUY AT CLOSE, SELL AT NEXT OPEN) - 12 SECTOR COMPARISON")
    print("=" * 115)
    sec_rows = []
    for sec, data in sorted(sector_summary.items(), key=lambda x: x[1]["avg_gross_on"], reverse=True):
        sec_rows.append([
            sec,
            data["count"],
            f"{data['avg_gross_on']*100:+8.1f}%",
            f"{data['avg_net_on']*100:+8.1f}%",
            f"{data['avg_intraday']*100:+8.1f}%",
            f"{data['avg_bnh']*100:+8.1f}%",
            f"{data['avg_win_rate']:5.1f}%",
            f"{data['avg_bps']:+5.1f} bps"
        ])
    sec_headers = ["Sector", "Stocks", "Gross Overnight", "Net Overnight (Taxes)", "Intraday Only", "Buy & Hold", "ON Win %", "Avg Trade (bps)"]
    print(tabulate(sec_rows, headers=sec_headers, tablefmt="grid"))

    # 2. Print Top 15 Best Stocks for Overnight Drift
    print("\n" + "=" * 115)
    print("   TOP 15 HIGHEST GROSS OVERNIGHT RETURN EQUITIES (CLOSE -> NEXT OPEN)")
    print("=" * 115)
    top_on = sorted(all_results, key=lambda x: x["gross_overnight_ret"], reverse=True)[:15]
    top_rows = []
    for r in top_on:
        top_rows.append([
            r["ticker"],
            r["sector"],
            f"{r['gross_overnight_ret']*100:+8.1f}%",
            f"{r['net_overnight_ret']*100:+8.1f}%",
            f"{r['intraday_ret']*100:+8.1f}%",
            f"{r['bnh_ret']*100:+8.1f}%",
            f"{r['win_rate']:5.1f}%",
            f"{r['avg_bps']:+5.1f} bps",
            f"{r['overnight_mdd']*100:5.1f}%"
        ])
    top_headers = ["Ticker", "Sector", "Gross Overnight", "Net Overnight", "Intraday Only", "Buy & Hold", "Win %", "Avg bps", "Max DD"]
    print(tabulate(top_rows, headers=top_headers, tablefmt="grid"))

    # 3. Print Bottom 10 Worst Overnight Stocks
    print("\n" + "=" * 115)
    print("   BOTTOM 10 WORST OVERNIGHT RETURN EQUITIES (NEGATIVE OVERNIGHT DRIFT)")
    print("=" * 115)
    bot_on = sorted(all_results, key=lambda x: x["gross_overnight_ret"])[:10]
    bot_rows = []
    for r in bot_on:
        bot_rows.append([
            r["ticker"],
            r["sector"],
            f"{r['gross_overnight_ret']*100:+8.1f}%",
            f"{r['net_overnight_ret']*100:+8.1f}%",
            f"{r['intraday_ret']*100:+8.1f}%",
            f"{r['bnh_ret']*100:+8.1f}%",
            f"{r['win_rate']:5.1f}%",
            f"{r['avg_bps']:+5.1f} bps",
            f"{r['overnight_mdd']*100:5.1f}%"
        ])
    print(tabulate(bot_rows, headers=top_headers, tablefmt="grid"))

if __name__ == "__main__":
    evaluate_overnight_strategy()
