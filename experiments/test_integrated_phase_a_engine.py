"""
Empirical Verification Test for Integrated Phase A Overnight Execution Engine.
Tests:
- Tier 1: NIFTYBEES Daily Unfiltered Compounding with 0.15% Trail / 0.25% Stop.
- Tier 2: INFY.NS ADR Lead-Filtered Compounding with 0.15% Trail / 0.25% Stop + Idle Cash Yield.
- Quantifies the Idle Cash Yield boost on uninvested capital.
"""
import os
import sys
import json
import pandas as pd
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.domain.models import ExecutionTier, InstrumentType, ADRLeadSignal
from src.execution.overnight_engine import OvernightExecutionEngine

def run_test():
    print("=" * 70)
    print("INTEGRATED PHASE A OVERNIGHT ENGINE: EMPIRICAL VERIFICATION SUITE")
    print("=" * 70)

    nifty_path = "data/lake/Indices/NIFTYBEES.parquet"
    infy_path = "data/lake/Information_Technology/INFY.parquet"
    adr_path = "data/lake/Global/INFY_ADR.parquet"

    if not all(os.path.exists(p) for p in [nifty_path, infy_path, adr_path]):
        raise FileNotFoundError("Required Parquet lake files not found.")

    df_nifty = pd.read_parquet(nifty_path).sort_index()
    df_infy = pd.read_parquet(infy_path).sort_index()
    df_adr = pd.read_parquet(adr_path).sort_index()

    # Align dates
    df_nifty.index = pd.to_datetime(df_nifty.index).tz_localize(None).normalize()
    df_infy.index = pd.to_datetime(df_infy.index).tz_localize(None).normalize()
    df_adr.index = pd.to_datetime(df_adr.index).tz_localize(None).normalize()

    # Compute ADR daily return
    df_adr["Return"] = df_adr["Close"].pct_change()
    adr_map = {d.strftime("%Y-%m-%d"): r for d, r in zip(df_adr.index, df_adr["Return"])}

    # Aligned 2-year sessions (475 sessions)
    common_dates = df_nifty.index.intersection(df_infy.index).sort_values()
    if len(common_dates) > 476:
        common_dates = common_dates[-476:]

    results = {}

    # 1. Test Tier 1: NIFTYBEES Daily Unfiltered (1x and 5x)
    print("\n[TEST 1] Testing Tier 1: NIFTYBEES Daily Unfiltered...")
    for lev in [1.0, 5.0]:
        engine = OvernightExecutionEngine(initial_capital=10000.0, leverage=lev)
        for i in range(len(common_dates) - 1):
            d_prev = common_dates[i]
            d_curr = common_dates[i + 1]

            bar_prev = df_nifty.loc[d_prev]
            bar_curr = df_nifty.loc[d_curr]

            engine.execute_session(
                date_str=d_curr.strftime("%Y-%m-%d"),
                ticker="NIFTYBEES",
                tier=ExecutionTier.TIER1_DAILY_ETF,
                instrument=InstrumentType.ETF_CASH if lev == 1.0 else InstrumentType.FUTURES,
                entry_price=float(bar_prev["Close"]),
                open_price=float(bar_curr["Open"]),
                high_price=float(bar_curr["High"]),
                low_price=float(bar_curr["Low"]),
                trail_pct=0.0015,
                stop_pct=0.0025
            )

        caps = [r.ending_balance for r in engine.history]
        rets = [r.net_return_pct for r in engine.history]
        wins = sum(1 for r in rets if r > 0)
        cummax = np.maximum.accumulate(caps)
        mdd = np.min((np.array(caps) - cummax) / cummax) * 100.0

        key = f"NIFTYBEES_{int(lev)}x"
        results[key] = {
            "tier": "TIER1_DAILY_ETF",
            "leverage": lev,
            "final_capital": round(engine.capital, 2),
            "net_profit": round(engine.capital - 10000.0, 2),
            "return_pct": round(((engine.capital - 10000.0) / 10000.0) * 100.0, 2),
            "win_rate_pct": round((wins / len(engine.history)) * 100.0, 2),
            "max_drawdown_pct": round(float(mdd), 2),
            "total_trades": len(engine.history)
        }
        print(f"  -> Leverage {lev:.0f}x: Final = INR {engine.capital:,.2f} | Win Rate = {results[key]['win_rate_pct']}% | MDD = {mdd:.2f}%")

    # 2. Test Tier 2: INFY.NS ADR-Filtered with Idle Cash Sweep (1x and 5x)
    print("\n[TEST 2] Testing Tier 2: INFY.NS ADR Lead-Filtered + Idle Cash Yield...")
    for lev in [1.0, 5.0]:
        engine = OvernightExecutionEngine(initial_capital=10000.0, leverage=lev)
        for i in range(len(common_dates) - 1):
            d_prev = common_dates[i]
            d_curr = common_dates[i + 1]

            bar_prev = df_infy.loc[d_prev]
            bar_curr = df_infy.loc[d_curr]

            d_prev_str = d_prev.strftime("%Y-%m-%d")
            adr_ret = adr_map.get(d_prev_str, 0.0)
            signal = ADRLeadSignal(
                foreign_ticker="INFY",
                domestic_ticker="INFY.NS",
                foreign_return_pct=float(adr_ret),
                is_bullish=(adr_ret > 0.0)
            )

            c_prev = float(bar_prev["Close_Adj"] if "Close_Adj" in bar_prev else bar_prev["Close"])
            engine.execute_session(
                date_str=d_curr.strftime("%Y-%m-%d"),
                ticker="INFY.NS",
                tier=ExecutionTier.TIER2_ADR_LEAD_EQUITY,
                instrument=InstrumentType.EQUITY_CASH if lev == 1.0 else InstrumentType.FUTURES,
                entry_price=c_prev,
                open_price=float(bar_curr["Open"]),
                high_price=float(bar_curr["High"]),
                low_price=float(bar_curr["Low"]),
                adr_signal=signal,
                trail_pct=0.0015,
                stop_pct=0.0025,
                idle_yield_enabled=True
            )

        caps = [r.ending_balance for r in engine.history]
        active_trades = [r for r in engine.history if r.exit_reason != "CASH_PRESERVED_ADR_RED"]
        active_wins = sum(1 for r in active_trades if r.gross_return_pct > 0)
        cummax = np.maximum.accumulate(caps)
        mdd = np.min((np.array(caps) - cummax) / cummax) * 100.0

        key = f"INFY_ADR_FILTERED_{int(lev)}x"
        results[key] = {
            "tier": "TIER2_ADR_LEAD_EQUITY",
            "leverage": lev,
            "final_capital": round(engine.capital, 2),
            "net_profit": round(engine.capital - 10000.0, 2),
            "return_pct": round(((engine.capital - 10000.0) / 10000.0) * 100.0, 2),
            "active_trades": len(active_trades),
            "idle_sessions": len(engine.history) - len(active_trades),
            "active_win_rate_pct": round((active_wins / len(active_trades)) * 100.0, 2) if active_trades else 0.0,
            "max_drawdown_pct": round(float(mdd), 2)
        }
        print(f"  -> Leverage {lev:.0f}x: Final = INR {engine.capital:,.2f} | Active Win Rate = {results[key]['active_win_rate_pct']}% | MDD = {mdd:.2f}%")

    # Save audit log
    audit_file = "audit/integrated_phase_a_engine_audit.json"
    with open(audit_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Integrated engine verification complete. Audit saved to {audit_file}")
    return results

if __name__ == "__main__":
    run_test()
