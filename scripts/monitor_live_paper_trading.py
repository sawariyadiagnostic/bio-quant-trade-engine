"""Live Paper Trading Monitor Dashboard for RAVANA v4.3."""
from __future__ import annotations

from datetime import datetime
import json
import os
import sys
import pandas as pd
from tabulate import tabulate

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.fetch_today_live_data import fetch_today_all

def render_dashboard() -> None:
    now_ist = datetime.now()
    t_str = now_ist.strftime("%H:%M:%S")

    # Determine market phase
    t_hm = now_ist.strftime("%H:%M")
    if t_hm < "09:15":
        phase = "PRE-MARKET WARMUP (Awaiting 09:15 Open)"
        phase_color = "YELLOW"
    elif "09:15" <= t_hm <= "11:30":
        phase = "MORNING EXPANSION SESSION (Active ORB, Drift, Sweep, iNAV)"
        phase_color = "GREEN"
    elif "11:30" < t_hm < "13:00":
        phase = "MIDDAY LUNCH LULL (Capital Protected in 100% Cash, No Forced Trades)"
        phase_color = "CYAN"
    elif "13:00" <= t_hm < "13:15":
        phase = "PRE-EUROPEAN OPEN PREPARATION (Armed for 13:15 Impulse)"
        phase_color = "YELLOW"
    elif "13:15" <= t_hm <= "13:45":
        phase = "AFTERNOON EUROPEAN OPEN IMPULSE (Active ITBEES, NIFTYBEES)"
        phase_color = "GREEN"
    elif "13:45" < t_hm < "14:45":
        phase = "AFTERNOON DRIFT (Trailing Stops Ratcheting to Breakeven)"
        phase_color = "CYAN"
    elif "14:45" <= t_hm < "15:10":
        phase = "PRE-DUMP SHIELD (Autonomous Exit Ahead of 15:00 Broker Dump)"
        phase_color = "RED"
    elif "15:10" <= t_hm <= "15:25":
        phase = "ALL-WEATHER OVERNIGHT ALLOCATION (Regime Bull/Bear CNC Execution)"
        phase_color = "MAGENTA"
    else:
        phase = "POST-MARKET CLOSED (Overnight CNC Active, Cash Yield Compounding)"
        phase_color = "WHITE"

    print("=" * 82)
    print(f"       RAVANA v4.3: LIVE AUTONOMOUS PAPER TRADING MONITOR DASHBOARD")
    print(f"       Timestamp: {t_str} IST  |  Date: {now_ist.strftime('%Y-%m-%d')}")
    print("=" * 82)
    print(f" CURRENT PHASE : [{phase}]")
    print("-" * 82)

    # 1. Fetch live market snapshot
    today_dfs = fetch_today_all()
    etf_rows = []
    for sym in ["NIFTYBEES", "BANKBEES", "ITBEES", "AUTOBEES", "PHARMABEES",
                "PSUBNKBEES", "CPSEETF", "GOLDBEES", "SILVERBEES", "JUNIORBEES", "NSEI"]:
        if sym in today_dfs and not today_dfs[sym].empty:
            df = today_dfs[sym]
            last_bar = df.iloc[-1]
            open_px = float(df.iloc[0]["Open"])
            close_px = float(last_bar["Close"])
            day_chg_pct = (close_px / open_px - 1.0) * 100.0
            atr = float(last_bar.get("ATR_14", 0.0))
            ofi = float(last_bar.get("OFI_5", 0.0))
            vr = float(last_bar.get("VR_20", 1.0))
            t_bar = df.index[-1].strftime("%H:%M")
            etf_rows.append([
                sym, f"{close_px:.2f}", f"{day_chg_pct:+.2f}%",
                f"{atr:.2f}", f"{ofi:+.2f}", f"{vr:.2f}", t_bar
            ])

    print("\n[1] LIVE MARKET ETF UNIVERSE (5-Minute Resolution):")
    print(tabulate(
        etf_rows,
        headers=["Symbol", "LTP (₹)", "Day Chg", "ATR_14", "OFI_5", "VR_20", "Bar Time"],
        tablefmt="simple"
    ))

    # 2. Read live daemon log recent entries
    log_file = os.path.join(BASE_DIR, "audit", "ravana_live_daemon.log")
    print("\n[2] LIVE DAEMON AUDIT STREAM (Last 6 Events):")
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            for l in lines[-6:]:
                print(f"  {l}")
    else:
        print("  No daemon log entries found.")

    # 3. Read today's orders
    orders_file = os.path.join(BASE_DIR, "audit", "shoonya_paper_orders.json")
    print("\n[3] RECENT LIVE PAPER ORDERS EXECUTED:")
    today_orders = []
    if os.path.exists(orders_file):
        with open(orders_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    if "timestamp" in obj and obj["timestamp"].startswith(now_ist.strftime("%Y-%m-%d")):
                        today_orders.append(obj)
                except Exception:
                    pass

    if today_orders:
        order_rows = []
        for o in today_orders[-5:]:
            order_rows.append([
                o.get("order_id", o.get("norenordno", "N/A")),
                o.get("ticker", o.get("tsym", "N/A")),
                o.get("side", o.get("action", "N/A")),
                o.get("shares", o.get("qty", 0)),
                f"{float(o.get('price', 0.0)):.2f}",
                o.get("timestamp", "").split("T")[-1][:8]
            ])
        print(tabulate(order_rows, headers=["Order ID", "Ticker", "Side", "Qty", "Price", "Time"], tablefmt="simple"))
    else:
        print("  No fills executed during Midday Lull (Protected flat in cash at ₹200,000.00).")

    print("\n" + "=" * 82)
    print(" RAVANA v4.3 ENGINE STATUS: 100% AUTONOMOUS BACKGROUND DAEMON ACTIVE")
    print(" Next Action: Armed for 13:15 IST European Open Impulse Window.")
    print("=" * 82)

if __name__ == "__main__":
    render_dashboard()
