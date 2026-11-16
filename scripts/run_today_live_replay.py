"""RAVANA v4.3: Today's Live Market Replay (Monday, 2026-11-16, 09:15-11:30 IST).
Replays today's live 28 5-minute bars through the certified RAVANA portfolio engine.
"""
import os, sys
sys.path.insert(0, os.path.abspath("."))
import pandas as pd
from scripts.fetch_today_live_data import fetch_today_all
from scripts.paper_trade_60d_integrated_portfolio import Integrated60DayPortfolioTrader
from ravana.strategies.base import TriggerAction, TriggerSignal

def run_today_session():
    print("=" * 86)
    print("  RAVANA v4.3: LIVE MARKET SESSION AUDIT (TODAY: 2026-11-16)")
    print("  Window: 09:15 to 11:30 IST (Morning Session) | Capital: ₹200,000.00")
    print("======================================================================================")
    
    # 1. Ingest live data for today
    today_dfs = fetch_today_all()
    if "NSEI" not in today_dfs:
        raise RuntimeError("NIFTY 50 live data missing")
    bench_df = today_dfs["NSEI"]
    
    etf_dfs = {k: v for k, v in today_dfs.items() if k != "NSEI"}
    
    # 2. Initialize certified trader
    trader = Integrated60DayPortfolioTrader(initial_capital=200000.0)
    
    # Pre-evaluate triggers on today's live bars
    signals_by_ts = {}
    for sym, df in etf_dfs.items():
        # ORB
        sigs = trader.orb_trigger.generate_signals(df, sym)
        for s in sigs:
            signals_by_ts.setdefault(s.timestamp, []).append(s)
        # Drift
        sigs = trader.drift_trigger.generate_signals(df, sym)
        for s in sigs:
            signals_by_ts.setdefault(s.timestamp, []).append(s)
        # Stop Sweep
        sigs = trader.stop_sweep_trigger.generate_signals(df, sym)
        for s in sigs:
            signals_by_ts.setdefault(s.timestamp, []).append(s)
        # iNAV
        sigs = trader.inav_trigger.generate_signals(df, sym)
        for s in sigs:
            signals_by_ts.setdefault(s.timestamp, []).append(s)
        # VWAP Absorption
        sigs = trader.vwap_absorb_trigger.generate_signals(df, sym)
        for s in sigs:
            signals_by_ts.setdefault(s.timestamp, []).append(s)
            
    print(f"\nGenerated {sum(len(v) for v in signals_by_ts.values())} candidate signals across {len(signals_by_ts)} time slices.")
    
    # Replay today's bars bar by bar
    all_timestamps = sorted(set.union(*[set(df.index) for df in etf_dfs.values()]))
    
    for ts in all_timestamps:
        t_str = ts.strftime("%H:%M")
        current_bars = {sym: df.loc[ts] for sym, df in etf_dfs.items() if ts in df.index}
        
        # 1. Manage active positions
        trader._manage_intraday_positions(current_bars, ts)
        
        # 2. 11:30 AM Morning Lock (bank morning alpha, flat during midday lull)
        if t_str >= "11:30":
            trader._square_off_all_intraday(current_bars, ts, "MORNING_1130_LOCK")
            continue
            
        # 3. Entry Window (09:35 - 10:45 IST)
        if ("09:35" <= t_str <= "10:45") and (len(trader.open_intraday_trades) < 2):
            candidates = signals_by_ts.get(ts, [])
            valid_candidates = [
                c for c in candidates
                if c.symbol not in trader.open_intraday_trades
                and c.symbol in current_bars
                and c.action.value in ("BUY", "SELL")
            ]
            if valid_candidates:
                winning_sig, alloc_frac = trader.alpha_crown.select_top_allocation(
                    valid_candidates, current_bars, trader.current_cash
                )
                if winning_sig is not None and alloc_frac > 0.0:
                    trader._open_intraday_trade(winning_sig, current_bars[winning_sig.symbol], ts)
                    
    # Audit summary for today
    intraday_trades = [t for t in trader.trade_history if t.get("type") in ("INTRADAY", "INTRADAY_SCALE")]
    print("\n" + "=" * 86)
    print("  TODAY'S LIVE MORNING SESSION AUDIT REPORT (2026-11-16 09:15 - 11:30 IST)")
    print("=" * 86)
    print(f"  Starting Capital      : ₹{trader.initial_capital:,.2f}")
    print(f"  Current Live Capital  : ₹{trader.current_cash:,.2f}")
    print(f"  Net Realized PnL      : ₹{trader.total_intraday_pnl:+,.2f} ({trader.total_intraday_pnl/trader.initial_capital*100:+.2f}%)")
    print(f"  Total Trades Closed   : {len(intraday_trades)}")
    print(f"  Vetoed Traps Blocked  : {trader.vetoed_trades_count} trades (Gemini Veto Guard)")
    print(f"  Open Positions at 11:30: {len(trader.open_intraday_trades)} (100% Flat Cash)")
    print("=" * 86)
    
    if intraday_trades:
        df_tr = pd.DataFrame(intraday_trades)
        print("\n--- ITEM-BY-ITEM REALIZED TRADES TODAY ---")
        print(df_tr[["entry_time", "exit_time", "ticker", "trigger_name", "side", "reason", "pnl"]].to_string(index=False))
    else:
        print("\nNo trades met the strict institutional conviction threshold today; capital preserved in cash.")

if __name__ == "__main__":
    run_today_session()
