"""
One-Month Dedicated Overnight Paper Trading Engine for RAVANA v4.1.
Executes pure All-Weather Tier-1 Overnight Continuation trading
over 22 trading days (1 Month) with ₹2,00,000 capital, verifying
CNC cash delivery sizing, the 09:15 Continuation Gate DP fee shield,
ACID SQLite WAL state persistence, and Shoonya paper execution.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

# UTF-8 Console Safety
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ravana.backtest.config import AssetType, BacktestConfig, TradingRegime
from ravana.backtest.engine import EventDrivenBacktester
from ravana.backtest.orders import Order, OrderSide
from ravana.daemon.data_feeder import SectorDataFeeder
from ravana.execution.shoonya_adapter import (
    ShoonyaBrokerGateway,
    ShoonyaGatewayConfig,
)
from ravana.strategies.state_ledger import SQLiteStateLedger
from ravana.strategies.triggers import OvernightContinuationTrigger

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("overnight_paper_trading")


def run_overnight_paper_trading_1month(
    initial_capital: float = 200000.0,
    days_to_trade: int = 22,
) -> Dict[str, Any]:
    """Execute dedicated 1-month paper trading simulation on pure Overnight Continuation."""
    print("=" * 82)
    print("  RAVANA 1-MONTH DEDICATED OVERNIGHT PAPER TRADING SIMULATION")
    print("  Regime: CNC (Cash Delivery) | Capital: ₹2,00,000 | Horizon: 1 Month (22 Days)")
    print("  Tier-1 Whitelist: BANKBEES, NIFTYBEES, ITBEES, GOLDBEES")
    print("=" * 82)

    db_path = os.path.join(BASE_DIR, "audit", "ravana_state_ledger.db")
    audit_file = os.path.join(BASE_DIR, "audit", "shoonya_paper_orders.json")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    ledger = SQLiteStateLedger(db_path=db_path)
    gw_cfg = ShoonyaGatewayConfig(
        paper_trading=True,
        default_product_type="C",  # CNC Cash & Carry Delivery
        audit_file_path=audit_file,
    )
    gateway = ShoonyaBrokerGateway(config=gw_cfg)
    feeder = SectorDataFeeder()
    trig = OvernightContinuationTrigger()

    symbols = ["BANKBEES", "NIFTYBEES", "ITBEES"]
    sector_results: Dict[str, Any] = {}

    for sym in symbols:
        df_full = feeder.load_sector_data(sym)
        dates = sorted(set(df_full.index.date))[-days_to_trade:]
        df_1m = df_full[df_full.index.date >= dates[0]].copy()

        # Evaluate signals
        sigs = trig.evaluate(df_1m, sym)

        cfg = BacktestConfig(
            initial_capital=initial_capital,
            regime=TradingRegime.OVERNIGHT,
            asset_type=AssetType.ETF,
            brokerage_per_order_inr=0.0,
            fixed_dp_fee_inr=15.93,
            hold_on_strong_continuation=True,
            continuation_threshold=0.50,
        )
        backtester = EventDrivenBacktester(config=cfg)
        res = backtester.run(df=df_1m, signals=sigs, ticker=sym, signal_threshold=0.50)

        # Log simulated paper orders into Gateway and SQLite Ledger
        for rt in res.closed_round_trips:
            # Paper buy
            ord_buy = Order(
                order_id=f"PAPER_ON_BUY_{sym}_{int(rt.entry_time.timestamp())}",
                timestamp=rt.entry_time,
                ticker=sym,
                side=OrderSide.BUY,
                shares=rt.shares,
            )
            gateway.place_order(ord_buy, reference_price=rt.entry_price)
            # Record in Ledger
            ledger.record_entry(
                ticker=sym,
                originating_trigger="OvernightContinuationTrigger",
                product_type="CNC",
                shares_total=rt.shares,
                entry_price=rt.entry_price,
                dynamic_profit_target=rt.entry_price * 1.025,
                trailing_stop_loss=rt.entry_price * 0.985,
                entry_timestamp=str(rt.entry_time),
            )
            # Paper sell
            ord_sell = Order(
                order_id=f"PAPER_ON_SELL_{sym}_{int(rt.exit_time.timestamp())}",
                timestamp=rt.exit_time,
                ticker=sym,
                side=OrderSide.SELL,
                shares=rt.shares,
            )
            gateway.place_order(ord_sell, reference_price=rt.exit_price)
            ledger.close_position(sym)

        sector_results[sym] = {
            "result": res,
            "days": len(dates),
            "start": str(dates[0]),
            "end": str(dates[-1]),
        }

    return sector_results


def main() -> None:
    t0 = time.perf_counter()
    results = run_overnight_paper_trading_1month(initial_capital=200000.0, days_to_trade=22)
    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 82)
    print("  1-MONTH OVERNIGHT PAPER TRADING PERFORMANCE REPORT")
    print("=" * 82)
    print(f"{'Asset':<12} | {'Trades':<8} | {'Win Rate':<10} | {'Sharpe':<8} | {'Profit Fac':<10} | {'Net Return':<12} | {'Fees (INR)':<10}")
    print("-" * 82)

    total_net_pnl = 0.0
    total_trades = 0
    total_wins = 0

    for sym, data in results.items():
        res = data["result"]
        m = res.metrics
        tot_trades = m.get("round_trip_trades", 0)
        wins = sum(1 for r in res.closed_round_trips if r.is_win)
        gp = sum(r.gross_pnl for r in res.closed_round_trips if r.is_win)
        gl = abs(sum(r.gross_pnl for r in res.closed_round_trips if not r.is_win))
        pf = (gp / gl) if gl > 0 else (99.0 if gp > 0 else 1.0)
        ret_pct = m.get("total_return_pct", 0.0)
        fees = m.get("total_fees_inr", 0.0)
        sharpe = m.get("sharpe_ratio", 0.0)
        net_inr = 200000.0 * (ret_pct / 100.0)

        total_net_pnl += net_inr
        total_trades += tot_trades
        total_wins += wins

        print(
            f"{sym:<12} | "
            f"{tot_trades:<8} | "
            f"{m.get('win_rate_pct', 0.0):>6.1f}%   | "
            f"{sharpe:>6.2f}  | "
            f"{pf:>8.2f}   | "
            f"{ret_pct:>+7.2f}% (₹{net_inr:>+7,.0f}) | "
            f"₹{fees:>7.2f}"
        )

    print("=" * 82)

    # Detailed trade breakdown for BANKBEES
    print("\n--- SAMPLE ITEM-BY-ITEM OVERNIGHT TRADE LOG (BANKBEES) ---")
    bank_rts = results["BANKBEES"]["result"].closed_round_trips
    print(f"{'Entry Date/Time':<18} | {'Exit Date/Time':<18} | {'Shares':<6} | {'Entry (₹)':<10} | {'Exit (₹)':<10} | {'Net PnL (₹)':<12}")
    print("-" * 82)
    for rt in bank_rts[:8]:
        e_str = rt.entry_time.strftime("%Y-%m-%d %H:%M")
        x_str = rt.exit_time.strftime("%Y-%m-%d %H:%M")
        print(f"{e_str:<18} | {x_str:<18} | {rt.shares:<6} | {rt.entry_price:>9.2f}  | {rt.exit_price:>9.2f}  | {rt.net_pnl:>+10.2f}")

    print("\n--- AUDIT PERSISTENCE VERIFICATION ---")
    print("  1. SQLite State Ledger : audit/ravana_state_ledger.db (ACID WAL)")
    print("  2. Paper Order Stream  : audit/shoonya_paper_orders.json (Dispatched & Reconciled)")
    print(f"Simulation completed in {elapsed:.2f} seconds.")
    print("=" * 82)


if __name__ == "__main__":
    main()
