"""Institutional CLI Runner for BANKBEES Overnight Paper Execution Pipeline.

Executes end-to-end evaluation, return-space clustering, ensemble compounding backtest,
and paper trade routing with full ASCII telemetry tables.
"""
import argparse
import os
import sys
from typing import List

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

from src.execution.shoonya_gateway import ShoonyaGateway
from src.pipeline.bankbees_pipeline import (
    AlphaChampion,
    BacktestSummary,
    BankbeesOvernightPipeline,
    EnsembleDecision,
    PipelineResult,
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


def _print_champions_table(champions: List[AlphaChampion], verbose: bool) -> None:
    """Renders formatted ASCII table of selected orthogonal alpha champions."""
    print("\n" + "=" * 90)
    print("                    SELECTED ORTHOGONAL ALPHA CHAMPIONS")
    print("=" * 90)
    header = f"{'Idx':<4} | {'Clust':<5} | {'Sortino':<8} | {'CAGR %':<8} | {'MaxDD %':<8} | {'Win %':<6} | {'Sig':<6} | {'Formula'}"
    print(header)
    print("-" * 90)
    for c in champions:
        sig_str = f"{c.current_signal:+.3f}"
        f_display = c.formula_str if verbose else (c.formula_str[:38] + "..." if len(c.formula_str) > 38 else c.formula_str)
        row = f"{c.index:<4} | {c.cluster_id:<5} | {c.sortino:<8.2f} | {c.cagr_pct:<8.2f} | {c.max_drawdown:<8.2f} | {c.win_rate:<6.1f} | {sig_str:<6} | {f_display}"
        print(row)
    print("-" * 90)


def _print_backtest_summary(
    summary: BacktestSummary, ticker: str, date_range: tuple[str, str], capital: float
) -> None:
    """Renders historical backtest performance metrics of the ensemble."""
    print("\n" + "=" * 90)
    print(f"       HISTORICAL ENSEMBLE BACKTEST SUMMARY: {ticker} ({date_range[0]} -> {date_range[1]})")
    print("=" * 90)
    print(f"  Historical Horizon Bars : {summary.total_bars} trading sessions")
    print(f"  Active Overnight Trades : {summary.total_trades} ({summary.total_trades / max(1, summary.total_bars) * 100:.1f}% market exposure)")
    print(f"  Ensemble CAGR           : {summary.cagr_pct:+.2f}%")
    print(f"  Sortino Semi-Deviation  : {summary.sortino:.2f} (20 bps institutional volatility floor)")
    print(f"  Max Historical Drawdown : {summary.max_drawdown_pct:.2f}%")
    print(f"  Trade Win Rate          : {summary.win_rate_pct:.2f}% ({summary.win_bars} winning sessions)")
    print(f"  Initial Capital Base    : ₹{capital:,.2f}")
    print(f"  Final Simulated Capital : ₹{summary.final_equity:,.2f}")
    print("=" * 90)


def _print_trading_decision(decision: EnsembleDecision, ticker: str) -> None:
    """Renders today's live close execution decision and pre-open auction routing."""
    print("\n" + "=" * 90)
    print(f"       TODAY'S MARKET CLOSE SIGNAL & PAPER EXECUTION: {ticker} ({decision.latest_date})")
    print("=" * 90)
    print(f"  Underlying Benchmark Price : ₹{decision.latest_close:.2f}")
    print(f"  Alpha Conviction Score     : {decision.conviction_score * 100:.1f}% ({decision.active_signals}/{decision.total_champions} champions bullish)")
    print(f"  Consolidated Action Bias   : [{decision.action}]")

    if decision.action == "BUY":
        print(f"  Statutory Allocated Capital: ₹{decision.trade_value:,.2f}")
        print(f"  Allocated Integer Lots     : {decision.quantity} units (zero-fraction invariant verified)")
        print(f"  Statutory ETF Friction     : ₹{decision.statutory_friction:.4f} (1.2 bps delivery STT & exchange levy)")
        if decision.pre_open_order:
            po = decision.pre_open_order
            print(f"  Pre-Open Auction Limit     : ₹{po['limit_price']:.2f} (Indicative: ₹{po['indicative_price']:.2f}, +15 bps ceiling)")
            print(f"  Time-In-Force Routing      : {po['time_in_force']}")
        if decision.client_order_id:
            print(f"  Client Order ID            : {decision.client_order_id}")
        if decision.paper_order_result:
            stat = decision.paper_order_result.get("stat", "Ok")
            print(f"  Shoonya Virtual Broker     : Order Accepted [{stat}]")
    else:
        print("  Capital State              : 100% Cash / Liquid BeES Capital Preservation Mode")
    print("=" * 90 + "\n")


def main() -> int:
    """Main CLI execution routine for BANKBEES overnight strategy."""
    parser = argparse.ArgumentParser(
        description="Bio-Quant BANKBEES Overnight Quantitative Paper Execution Pipeline."
    )
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial trading capital in INR (default: 10000.0)")
    parser.add_argument("--lookback", type=str, default="5y", help="Historical evaluation lookback horizon (default: 5y)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate execution without writing paper broker journal")
    parser.add_argument("--verbose", action="store_true", help="Display full formula strings and detailed diagnostics")
    parser.add_argument("--limit", type=int, default=50, help="Number of elite formulas to ingest (default: 50)")

    args = parser.parse_args()
    lookback_years = _parse_lookback_years(args.lookback)

    print("=" * 90)
    print("       BIO-QUANT SOVEREIGN OVERNIGHT EXECUTION PIPELINE (trade_new)")
    print(f"       Target: BANKBEES | Horizon: {args.lookback} ({lookback_years:.1f}y) | Capital: ₹{args.capital:,.2f}")
    print(f"       Mode: {'DRY RUN' if args.dry_run else 'PAPER TRADING BROKER GATEWAY'}")
    print("=" * 90)

    gateway = ShoonyaGateway(paper_trading=True, initial_balance=args.capital)
    pipeline = BankbeesOvernightPipeline(capital=args.capital)

    res: PipelineResult = pipeline.run(
        lookback_years=lookback_years,
        gateway=gateway,
        dry_run=args.dry_run,
        limit_candidates=args.limit,
    )

    print(f"\n[INFO] Evaluated {res.evaluated_count} candidate formulas across {res.total_bars} market sessions.")
    print(f"[INFO] Partitioned into {res.cluster_count} orthogonal return-space clusters.")
    print(f"[INFO] Selected {len(res.champions)} champion alphas (Occam's parsimony penalty applied).")

    _print_champions_table(res.champions, verbose=args.verbose)
    _print_backtest_summary(res.ensemble_backtest, res.ticker, res.date_range, args.capital)
    _print_trading_decision(res.decision, res.ticker)

    return 0


if __name__ == "__main__":
    sys.exit(main())
