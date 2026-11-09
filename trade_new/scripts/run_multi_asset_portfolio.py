"""Institutional CLI Runner for Multi-Asset Overnight Quantitative Portfolio Engine.

Executes end-to-end evaluation, active asset selection, Ledoit-Wolf HRP risk-parity allocation,
single-asset exposure limit enforcement, multi-asset 3D portfolio backtesting, and paper order routing.
"""
import argparse
import os
import sys
from typing import Dict, List

# Ensure trade_new is first on sys.path and remove parent collisions
script_dir = os.path.dirname(os.path.abspath(__file__))
trade_new_dir = os.path.abspath(os.path.join(script_dir, ".."))
if "" in sys.path:
    sys.path.remove("")
cwd = os.getcwd()
if cwd in sys.path and os.path.abspath(cwd) != trade_new_dir:
    sys.path.remove(cwd)
if trade_new_dir not in sys.path:
    sys.path.insert(0, trade_new_dir)

from src.execution.shoonya_gateway import ShoonyaGateway
from src.pipeline.multi_asset_pipeline import (
    AssetEvaluation,
    CORE_MULTI_ASSET_BASKET,
    MultiAssetOvernightPipeline,
    MultiAssetPipelineResult,
    PortfolioBacktestSummary,
)


def _parse_lookback_years(lookback_str: str) -> float:
    """Parses lookback strings like '1y', '3y', '5y' or raw floats into numerical years."""
    s = lookback_str.strip().lower()
    if s.endswith("y"):
        s = s[:-1]
    try:
        val = float(s)
        return max(0.2, val)
    except ValueError:
        return 5.0


def _print_asset_audit_table(result: MultiAssetPipelineResult) -> None:
    """Renders formatted ASCII audit table showing each asset's allocation telemetry."""
    print("\n" + "=" * 104)
    print("                        MULTI-ASSET PORTFOLIO CAPITAL ALLOCATION AUDIT TABLE")
    print("=" * 104)
    header = (
        f"{'Asset':<11} | {'Instrument':<10} | {'Friction':<9} | {'Conviction':<10} | "
        f"{'Target Wt':<9} | {'Allocated INR':<15} | {'Lots':<6} | {'Action':<10}"
    )
    print(header)
    print("-" * 104)

    for ticker in result.evaluated_assets:
        ev = result.evaluations[ticker]
        inst_name = ev.instrument_type.value
        fric_str = f"{ev.statutory_friction_bps:.1f} bps"
        conv_str = f"{ev.conviction_score * 100:.1f}%"
        wt_str = f"{ev.target_weight * 100:.2f}%"
        alloc_str = f"₹{ev.allocated_capital:,.2f}"
        row = (
            f"{ticker:<11} | {inst_name:<10} | {fric_str:<9} | {conv_str:<10} | "
            f"{wt_str:<9} | {alloc_str:<15} | {ev.quantity:<6} | {ev.action:<10}"
        )
        print(row)

    print("-" * 104)
    tot_alloc_pct = (result.total_allocated_capital / result.total_capital) * 100.0 if result.total_capital > 0 else 0.0
    print(
        f"TOTAL ALLOCATED CAPITAL : ₹{result.total_allocated_capital:,.2f} / "
        f"₹{result.total_capital:,.2f} ({tot_alloc_pct:.1f}% exposure | Cash Reserve: ₹{result.cash_reserve:,.2f})"
    )
    print(
        f"TOTAL STATUTORY FRICTION: ₹{result.total_statutory_friction:,.4f} "
        f"(1.2 bps ETF delivery STT, 3.0 bps Futures turnover/STT)"
    )
    print("=" * 104)


def _print_backtest_summary(summary: PortfolioBacktestSummary, capital: float) -> None:
    """Renders historical performance metrics of the combined multi-asset portfolio."""
    print("\n" + "=" * 104)
    print(f"       COMBINED MULTI-ASSET PORTFOLIO HISTORICAL BACKTEST ({summary.date_range[0]} -> {summary.date_range[1]})")
    print("=" * 104)
    exp_pct = (summary.total_trades / max(1, summary.total_bars)) * 100.0 if summary.total_bars > 0 else 0.0
    print(f"  Historical Overlapping Bars : {summary.total_bars} market sessions")
    print(f"  Active Overnight Trades     : {summary.total_trades} sessions ({exp_pct:.1f}% active market exposure)")
    print(f"  Portfolio CAGR              : {summary.cagr_pct:+.2f}%")
    print(f"  Sortino Semi-Deviation      : {summary.sortino:.2f} (20 bps institutional volatility floor)")
    print(f"  Max Historical Drawdown     : {summary.max_drawdown_pct:.2f}%")
    print(f"  Trade Win Rate              : {summary.win_rate_pct:.2f}% ({summary.win_bars} winning sessions)")
    print(f"  Initial Portfolio Base      : ₹{capital:,.2f}")
    print(f"  Final Simulated Capital     : ₹{summary.final_equity:,.2f}")
    print("=" * 104)


def _print_action_list(evaluations: Dict[str, AssetEvaluation], dry_run: bool) -> None:
    """Renders today's execution orders, pre-open limits, and paper broker acknowledgments."""
    print("\n" + "=" * 104)
    print("                        TODAY'S MULTI-ASSET MARKET CLOSE ACTIONS")
    print("=" * 104)
    buy_orders = [ev for ev in evaluations.values() if ev.action == "BUY"]

    if not buy_orders:
        print("  [ALL CASH] Conviction below threshold across basket. 100% Cash / Liquid BeES preservation mode.")
    else:
        for ev in buy_orders:
            print(f"  [{ev.action}] {ev.ticker} ({ev.instrument_type.value}):")
            print(f"     Allocated Capital : ₹{ev.allocated_capital:,.2f} ({ev.quantity} units @ ₹{ev.latest_close:.2f})")
            print(f"     Statutory Tax     : ₹{ev.statutory_friction_inr:.4f} ({ev.statutory_friction_bps:.1f} bps)")
            if ev.pre_open_order:
                po = ev.pre_open_order
                print(f"     Pre-Open Auction  : Limit ₹{po['limit_price']:.2f} (Indicative: ₹{po['indicative_price']:.2f})")
            if ev.client_order_id:
                print(f"     Client Order ID   : {ev.client_order_id}")
            if ev.paper_order_result:
                stat = ev.paper_order_result.get("stat", "Ok")
                ord_no = ev.paper_order_result.get("norenordno", "N/A")
                print(f"     Broker Dispatch   : Status [{stat}], OrderNo [{ord_no}]")
            elif dry_run:
                print("     Broker Dispatch   : Simulated [DRY RUN]")
            print()

    print("=" * 104 + "\n")


def main() -> int:
    """Main CLI execution routine for Multi-Asset Overnight Quantitative Engine."""
    parser = argparse.ArgumentParser(
        description="Bio-Quant Multi-Asset Overnight Quantitative Portfolio Engine."
    )
    parser.add_argument("--capital", type=float, default=100000.0, help="Total trading capital in INR (default: 100,000.0)")
    parser.add_argument("--lookback", type=str, default="5y", help="Historical evaluation lookback horizon (default: 5y)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate execution without writing paper broker journal")
    parser.add_argument("--conviction-threshold", type=float, default=0.40, help="Bullish conviction filter threshold (default: 0.40)")
    parser.add_argument("--max-weight", type=float, default=0.30, help="Single-asset maximum weight cap (default: 0.30)")
    parser.add_argument("--limit", type=int, default=50, help="Number of elite formulas to ingest per asset (default: 50)")
    parser.add_argument("--assets", nargs="+", default=CORE_MULTI_ASSET_BASKET, help="Custom list of asset tickers to evaluate")

    args = parser.parse_args()
    lookback_years = _parse_lookback_years(args.lookback)

    print("=" * 104)
    print("       BIO-QUANT MULTI-ASSET SOVEREIGN PORTFOLIO ENGINE (trade_new)")
    print(f"       Basket: {args.assets}")
    print(f"       Capital: ₹{args.capital:,.2f} | Horizon: {args.lookback} ({lookback_years:.1f}y)")
    print(f"       Conviction Threshold: {args.conviction_threshold:.2f} | Single-Asset Cap: {args.max_weight * 100:.0f}%")
    print(f"       Mode: {'DRY RUN (Simulation Only)' if args.dry_run else 'PAPER TRADING BROKER GATEWAY'}")
    print("=" * 104)

    gateway = ShoonyaGateway(paper_trading=True, initial_balance=args.capital)
    pipeline = MultiAssetOvernightPipeline(
        capital=args.capital,
        conviction_threshold=args.conviction_threshold,
        max_asset_weight=args.max_weight,
    )

    result = pipeline.run(
        assets=args.assets,
        lookback_years=lookback_years,
        gateway=gateway,
        dry_run=args.dry_run,
        limit_candidates=args.limit,
    )

    _print_asset_audit_table(result)
    _print_backtest_summary(result.portfolio_backtest, args.capital)
    _print_action_list(result.evaluations, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    sys.exit(main())
