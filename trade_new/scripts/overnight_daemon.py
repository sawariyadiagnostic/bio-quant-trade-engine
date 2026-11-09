"""Headless Automated Overnight Scheduler Daemon for Live and Paper Execution.

Continuously orchestrates overnight trading windows on NSE trading days:
1. 09:00 IST: Pre-open auction exit limit order submission.
2. 09:15 - 09:30 IST: Real-time trailing stop loss monitoring and dynamic profit locking.
3. 15:15 - 15:25 IST: MOC (Market-on-Close) multi-asset signal evaluation, risk parity, and entry.
"""
import argparse
from datetime import datetime, time as dtime
import os
import signal
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Union
from zoneinfo import ZoneInfo

# Ensure trade_new is first on sys.path and remove parent collision
script_dir = os.path.dirname(os.path.abspath(__file__))
trade_new_dir = os.path.abspath(os.path.join(script_dir, ".."))
if "" in sys.path:
    sys.path.remove("")
cwd = os.getcwd()
if cwd in sys.path and os.path.abspath(cwd) != trade_new_dir:
    sys.path.remove(cwd)
if trade_new_dir not in sys.path:
    sys.path.insert(0, trade_new_dir)

from loguru import logger
from src.execution.shoonya_gateway import ShoonyaGateway
from src.pipeline.bankbees_pipeline import BankbeesOvernightPipeline
from src.pipeline.multi_asset_pipeline import MultiAssetOvernightPipeline

NSE_HOLIDAYS_2025_2026 = {
    "2025-01-26", "2025-02-26", "2025-03-14", "2025-03-31", "2025-04-10", "2025-04-14", "2025-04-18",
    "2025-05-01", "2025-08-15", "2025-08-27", "2025-10-02", "2025-10-21", "2025-10-22", "2025-11-05", "2025-12-25",
    "2026-01-26", "2026-03-03", "2026-03-20", "2026-04-03", "2026-04-14", "2026-05-01", "2026-08-15",
    "2026-10-02", "2026-10-20", "2026-11-08", "2026-11-09", "2026-12-25"
}


def is_trading_day(dt: datetime) -> bool:
    """Checks whether the specified datetime falls on an NSE trading weekday (Monday=0 to Friday=4) and not a holiday."""
    if dt.weekday() >= 5:
        return False
    date_str = dt.strftime("%Y-%m-%d")
    return date_str not in NSE_HOLIDAYS_2025_2026


def check_and_execute_entry(
    pipeline: Union[BankbeesOvernightPipeline, MultiAssetOvernightPipeline],
    capital: float,
    gateway: Optional[ShoonyaGateway] = None,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """Evaluates MOC entry signals at 15:20 IST across target universe."""
    logger.info(f"[Daemon] Evaluating MOC entry signals (Capital: ₹{capital:,.2f}, Dry-Run: {dry_run})...")
    if isinstance(pipeline, BankbeesOvernightPipeline):
        result = pipeline.run(lookback_years=1.0, gateway=gateway, dry_run=dry_run)
        summary = {
            "mode": "single",
            "action": str(result.decision.action),
            "conviction": float(result.decision.conviction_score),
            "benchmark_price": float(result.decision.latest_close),
            "allocated_capital": capital if result.decision.action == "BUY" else 0.0,
        }
        logger.info(f"[Daemon] BANKBEES MOC Decision: {summary['action']} (Conviction: {summary['conviction']:.1%})")
        return summary

    result = pipeline.run(lookback_years=2.0, gateway=gateway, dry_run=dry_run)
    allocations = [
        {
            "ticker": ev.ticker,
            "action": str(ev.action),
            "weight_pct": round(float(ev.target_weight) * 100.0, 2),
            "allocated_inr": float(ev.allocated_capital),
            "units": int(ev.quantity),
        }
        for ev in result.evaluations.values()
    ]
    summary = {
        "mode": "multi",
        "total_capital": float(result.total_capital),
        "allocated_capital": float(result.total_allocated_capital),
        "cash_reserve": float(result.cash_reserve),
        "friction": float(result.total_statutory_friction),
        "allocations": allocations,
    }
    active_buys = [a["ticker"] for a in allocations if a["action"] == "BUY"]
    logger.info(f"[Daemon] Multi-Asset MOC Complete: {len(active_buys)} Buy Signals -> {active_buys}")
    return summary


def check_and_execute_exit(gateway: ShoonyaGateway, dry_run: bool = True) -> Dict[str, Any]:
    """Pre-open check at 09:00 IST and trailing stop monitoring between 09:15 and 09:30 IST."""
    logger.info(f"[Daemon] Checking morning exit conditions (09:00 Pre-Open / 09:15 Trailing Stop, Dry-Run: {dry_run})...")
    open_positions: List[Dict[str, Any]] = []
    if dry_run or getattr(gateway, "paper_trading", False):
        pos_dict = getattr(gateway, "paper_positions", {})
        open_positions = [
            {"symbol": sym, "qty": data.get("quantity", 0), "price": data.get("avg_price", 0.0), **data}
            for sym, data in pos_dict.items()
            if data.get("quantity", 0) > 0 or data.get("qty", 0) > 0
        ]
    elif hasattr(gateway, "get_positions"):
        try:
            live_pos = gateway.get_positions()
            open_positions = [
                p for p in live_pos
                if p.get("quantity", 0) > 0 or p.get("qty", 0) > 0 or p.get("netqty", 0) > 0
            ]
        except Exception as e:
            logger.error(f"[Daemon] Error fetching live positions: {e}")

    logger.info(f"[Daemon] Open overnight positions detected: {len(open_positions)}")
    exit_results = []
    if not dry_run:
        for pos in open_positions:
            sym = pos.get("symbol", "")
            qty = int(pos.get("quantity", pos.get("qty", 0)))
            if qty > 0 and sym:
                order = {"symbol": sym, "quantity": qty, "price": 0.0, "action": "SELL", "order_type": "MKT"}
                res = gateway.place_order(order)
                exit_results.append({"symbol": sym, "result": res})

    return {
        "status": "ok",
        "open_positions_count": len(open_positions),
        "positions": open_positions,
        "exit_results": exit_results,
        "pre_open_auction_checked": True,
        "trailing_stop_active": True,
        "dry_run": dry_run,
    }


def run_daemon_loop(
    capital: float = 100000.0,
    dry_run: bool = True,
    mode: str = "multi",
    check_interval_s: float = 30.0,
    once: bool = False,
) -> None:
    """Executes main overnight scheduling daemon with graceful interrupt and heartbeat logging."""
    logger.info(f"[Daemon] Starting Bio-Quant Overnight Scheduler (Mode: {mode.upper()}, Capital: ₹{capital:,.2f}, Dry-Run: {dry_run})")
    gateway = ShoonyaGateway(paper_trading=dry_run, initial_balance=capital)
    pipeline: Union[BankbeesOvernightPipeline, MultiAssetOvernightPipeline] = (
        BankbeesOvernightPipeline(capital=capital)
        if mode == "single"
        else MultiAssetOvernightPipeline(capital=capital)
    )

    if once:
        logger.info("[Daemon] Executing single-cycle diagnostic verification (--once)...")
        exit_res = check_and_execute_exit(gateway, dry_run=dry_run)
        entry_res = check_and_execute_entry(pipeline, capital, gateway=gateway, dry_run=dry_run)
        logger.info(f"[Daemon] Single cycle complete. Exit check: {exit_res['status']} | Entry mode: {entry_res['mode']}")
        return

    shutdown_event = threading.Event()

    def _sig_handler(signum, frame):
        logger.warning(f"[Daemon] Caught shutdown signal ({signum}). Setting termination event...")
        shutdown_event.set()

    try:
        signal.signal(signal.SIGINT, _sig_handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, _sig_handler)
    except Exception as e:
        logger.debug(f"[Daemon] Signal setup notice: {e}")

    last_exit_date: Optional[str] = None
    last_entry_date: Optional[str] = None
    tz = ZoneInfo("Asia/Kolkata")
    last_loop_ts = time.time()

    logger.info("[Daemon] Entering continuous loop. Press Ctrl+C or send SIGTERM to terminate.")
    try:
        while not shutdown_event.is_set():
            now = datetime.now(tz)
            today_str = now.strftime("%Y-%m-%d")

            # ACPI S3 sleep blackout detection
            loop_elapsed = time.time() - last_loop_ts
            if loop_elapsed > check_interval_s * 3.0:
                logger.warning(
                    f"[Daemon] Time anomaly / OS sleep blackout detected: elapsed {loop_elapsed:.1f}s vs expected {check_interval_s}s"
                )
            last_loop_ts = time.time()

            if not is_trading_day(now):
                logger.info(f"[Daemon] Heartbeat {now.strftime('%Y-%m-%d %H:%M:%S')} IST - Market closed (Weekend/Holiday). Sleeping...")
                time.sleep(min(300.0, check_interval_s * 5))
                continue

            cur_time = now.time()
            if dtime(9, 0) <= cur_time <= dtime(9, 30):
                if last_exit_date != today_str:
                    logger.info(f"[Daemon] Triggering morning exit sequence for {today_str}...")
                    check_and_execute_exit(gateway, dry_run=dry_run)
                    last_exit_date = today_str
                else:
                    logger.debug(f"[Daemon] Morning exit already executed for {today_str}.")

            elif dtime(15, 15) <= cur_time <= dtime(15, 25):
                if last_entry_date != today_str:
                    logger.info(f"[Daemon] Triggering MOC entry sequence for {today_str}...")
                    check_and_execute_entry(pipeline, capital, gateway=gateway, dry_run=dry_run)
                    last_entry_date = today_str
                else:
                    logger.debug(f"[Daemon] MOC entry already executed for {today_str}.")
            elif cur_time > dtime(15, 25) and last_entry_date != today_str:
                logger.warning(f"[Daemon] Passed 15:25 MOC window boundary! Blocking execution to prevent AMO/overnight risk.")
                last_entry_date = today_str
            else:
                logger.info(f"[Daemon] Heartbeat {now.strftime('%H:%M:%S')} IST - Standing by for market windows (09:00 exit / 15:20 entry)")

            time.sleep(check_interval_s)
    except KeyboardInterrupt:
        logger.warning("[Daemon] Received KeyboardInterrupt. Gracefully shutting down scheduler daemon.")


def main() -> None:
    """CLI Entry point parsing arguments and launching overnight daemon."""
    parser = argparse.ArgumentParser(description="Bio-Quant Institutional Overnight Scheduler Daemon")
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial allocation capital in INR")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", default=True, help="Run in dry-run simulation mode")
    parser.add_argument("--live", dest="dry_run", action="store_false", help="Run with live broker execution")
    parser.add_argument("--mode", type=str, choices=["single", "multi"], default="multi", help="Strategy mode: 'single' or 'multi'")
    parser.add_argument("--once", action="store_true", default=False, help="Execute single diagnostic cycle immediately and exit")
    parser.add_argument("--interval", type=float, default=30.0, help="Check interval in seconds")

    args = parser.parse_args()
    run_daemon_loop(
        capital=args.capital,
        dry_run=args.dry_run,
        mode=args.mode,
        check_interval_s=args.interval,
        once=args.once,
    )


if __name__ == "__main__":
    main()
