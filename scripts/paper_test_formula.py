"""
Formula Paper Trading Tester & Performance Evaluator.
Validates elite formulas from the vault or custom user expressions in zero-risk paper trading mode.
Compares Overnight vs Intraday execution horizons with exact Indian statutory friction.
"""
import os
import sys
import json
import argparse
from typing import Optional, Any
import numpy as np
import pandas as pd
from loguru import logger

# UTF-8 Console Safety
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.manifold.lake_dataset_loader import LakeDatasetLoader
from src.domain.formula_parser import FormulaParser
from src.domain.formula_ast import ASTEvaluator
from src.domain.models import InstrumentType
from src.execution.shoonya_gateway import ShoonyaGateway
from src.execution.risk_controller import MicroCapitalController
from src.execution.execution_dispatcher import ExecutionDispatcher
from src.execution.overnight_engine import TaxCalculator


def load_vault_elites(snapshot_path: Optional[str] = None) -> list[dict[str, Any]]:
    """Loads all elite strategies from the latest strategy snapshot vault."""
    vault_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "strategies", "vault"))
    if not snapshot_path:
        files = [os.path.join(vault_dir, f) for f in os.listdir(vault_dir) if f.endswith(".json")]
        if not files:
            logger.warning("No snapshot files found in strategies/vault.")
            return []
        snapshot_path = sorted(files)[-1]

    with open(snapshot_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_elites: list[dict[str, Any]] = []
    for grid in data.get("strategies", []):
        for e in grid.get("elites", []):
            all_elites.append(e)
    return all_elites


def select_formula(
    mode: str,
    formula_arg: str = "",
    vault_elites: Optional[list[dict[str, Any]]] = None
) -> tuple[str, dict[str, Any]]:
    """Selects target formula based on requested mode."""
    if mode == "custom":
        if not formula_arg:
            formula_arg = "((Vol_Price_Flow / Downside_Vol_Ratio) - (ADR_Ret + Garman_Klass_Vol))"
        return formula_arg, {"name": "Custom User Formula", "source": "CLI Input"}

    if not vault_elites:
        raise ValueError("Cannot select from vault: vault contains 0 elites.")

    if mode == "top_earner":
        # Sort by net return percentage
        sorted_elites = sorted(vault_elites, key=lambda x: x.get("net_return_pct", 0), reverse=True)
        chosen = sorted_elites[0]
        return chosen["formula_str"], chosen

    if mode == "top_fitness":
        # Sort by institutional fitness (Calmar * Sortino * Confidence)
        sorted_elites = sorted(vault_elites, key=lambda x: x.get("fitness", 0), reverse=True)
        chosen = sorted_elites[0]
        return chosen["formula_str"], chosen

    if mode == "top_realistic":
        # Top earner within realistic CAGR bounds (20% to 300%)
        realistic = [e for e in vault_elites if 20.0 <= e.get("net_return_pct", 0) <= 300.0]
        if not realistic:
            realistic = vault_elites
        sorted_elites = sorted(realistic, key=lambda x: x.get("fitness", 0), reverse=True)
        chosen = sorted_elites[0]
        return chosen["formula_str"], chosen

    # Default fallback
    return vault_elites[0]["formula_str"], vault_elites[0]


def simulate_horizon(
    df: pd.DataFrame,
    signals: np.ndarray,
    horizon: str = "overnight",
    is_etf: bool = True
) -> dict[str, Any]:
    """
    Simulates historical returns under exact statutory friction.
    Overnight: Enter at Close_t -> Exit at Open_t+1
    Intraday: Enter at Open_t -> Exit at Close_t
    """
    inst = InstrumentType.ETF_CASH if is_etf else InstrumentType.EQUITY_CASH
    friction = 0.00012 if is_etf else 0.00250

    open_p = df["Open"].to_numpy(dtype=float)
    close_p = (df["Close_Adj"] if "Close_Adj" in df.columns else df["Close_Raw"]).to_numpy(dtype=float)

    binary_positions = np.where(signals > 0.0, 1.0, 0.0)

    if horizon == "overnight":
        gross_returns = (open_p[1:] - close_p[:-1]) / (close_p[:-1] + 1e-8)
        trade_positions = binary_positions[:-1]
    else:  # intraday
        gross_returns = (close_p - open_p) / (open_p + 1e-8)
        trade_positions = binary_positions

    pos_flips = np.abs(np.diff(trade_positions, prepend=0))
    net_returns = (trade_positions * gross_returns) - (pos_flips * friction)

    cum_equity = np.cumprod(1.0 + net_returns)
    total_net_ret = float(cum_equity[-1] - 1.0) if len(cum_equity) > 0 else 0.0
    n_bars = len(net_returns)
    years = max(0.08, n_bars / 252.0)
    cagr = float(((1.0 + total_net_ret) ** (1.0 / years)) - 1.0) * 100.0 if total_net_ret > -0.999 else -100.0

    peak = np.maximum.accumulate(cum_equity)
    drawdowns = (peak - cum_equity) / (peak + 1e-8)
    max_drawdown = float(np.max(drawdowns) * 100.0) if len(drawdowns) > 0 else 0.0

    active_trades = net_returns[trade_positions > 0]
    win_rate = float(np.mean(active_trades > 0) * 100.0) if len(active_trades) > 0 else 0.0

    mean_ret = float(np.mean(active_trades)) if len(active_trades) > 0 else 0.0
    downside_trades = active_trades[active_trades < 0]
    downside_dev = float(np.std(downside_trades)) if len(downside_trades) > 0 else 0.001
    sortino = float((mean_ret / (downside_dev + 1e-6)) * np.sqrt(252))

    calmar = max(0.0, cagr) / max(4.0, max_drawdown)
    total_friction_bps = float(np.sum(pos_flips * friction) * 10000.0)

    return {
        "horizon": horizon,
        "total_net_ret_pct": round(total_net_ret * 100.0, 2),
        "cagr_pct": round(cagr, 2),
        "win_rate_pct": round(win_rate, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "sortino": round(sortino, 2),
        "calmar": round(calmar, 2),
        "trades_count": int(np.sum(pos_flips > 0)),
        "total_friction_bps": round(total_friction_bps, 1)
    }


def execute_paper_order(
    ticker: str,
    df: pd.DataFrame,
    formula_str: str,
    initial_capital: float = 10000.0,
    is_etf: bool = True
) -> dict[str, Any]:
    """Generates today's signal and dispatches a virtual order through ShoonyaGateway."""
    gw = ShoonyaGateway(paper_trading=True, initial_balance=initial_capital)
    risk = MicroCapitalController(total_capital=initial_capital)
    dispatcher = ExecutionDispatcher(gateway=gw, risk_controller=risk, signal_threshold=0.01)

    decision = dispatcher.evaluate_and_dispatch(
        ticker=ticker,
        df=df,
        formula=formula_str,
        regime_id=0,
        is_etf=is_etf
    )

    return {
        "action": decision.action,
        "signal_value": decision.signal_value,
        "raw_price": decision.raw_price,
        "quantity": decision.quantity,
        "executed": decision.executed,
        "order_id": decision.order_id,
        "rejection_reason": decision.rejection_reason,
        "remaining_virtual_cash": round(gw.balance, 2),
        "paper_positions": gw.paper_positions
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test formulas in Paper Trading mode.")
    parser.add_argument("--mode", choices=["top_earner", "top_fitness", "top_realistic", "custom"], default="top_realistic",
                        help="Mode for formula selection (default: top_realistic)")
    parser.add_argument("--formula", type=str, default="", help="Custom mathematical formula string")
    parser.add_argument("--ticker", type=str, default="NIFTYBEES", help="Target asset ticker (default: NIFTYBEES)")
    parser.add_argument("--lookback", type=float, default=5.0, help="Historical lookback in years (default: 5.0)")
    parser.add_argument("--capital", type=float, default=10000.0, help="Virtual paper capital in INR (default: 10000.0)")
    args = parser.parse_args()

    print("=" * 80)
    print(f"  QUANT ENGINE: FORMULA PAPER TRADING & ASYMMETRY BENCHMARK")
    print("=" * 80)

    # 1. Load Data Lake
    print(f"\n[1/4] Loading Data Lake for {args.ticker} (Horizon: {args.lookback} Years)...")
    loader = LakeDatasetLoader()
    df = loader.load_dataset(args.ticker, lookback_years=args.lookback)
    print(f"      Loaded {len(df)} bars from {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}.")

    # 2. Select Formula
    vault_elites = load_vault_elites()
    formula_str, meta = select_formula(args.mode, args.formula, vault_elites)
    print(f"\n[2/4] Selected Formula (Mode: {args.mode.upper()}):")
    print(f"      Expression: {formula_str}")
    if "fitness" in meta:
        print(f"      Vault Fitness: {meta.get('fitness'):.2f} | Vault Return: {meta.get('net_return_pct'):.2f}% | Win Rate: {meta.get('win_rate'):.1f}%")

    # 3. Evaluate Signals & Compare Horizons
    print(f"\n[3/4] Evaluating AST Signals & Simulating Execution Horizons...")
    ast_root = FormulaParser.parse(formula_str)
    signals = ASTEvaluator.eval_node(ast_root, df)

    is_etf = "BEES" in args.ticker or "ETF" in args.ticker
    res_overnight = simulate_horizon(df, signals, horizon="overnight", is_etf=is_etf)
    res_intraday = simulate_horizon(df, signals, horizon="intraday", is_etf=is_etf)

    print("\n" + "-" * 75)
    print(f"{'Performance Metric':<28} | {'Overnight Drift':<20} | {'Intraday Session':<20}")
    print("-" * 75)
    print(f"{'Cumulative Net Return':<28} | {res_overnight['total_net_ret_pct']:>18}% | {res_intraday['total_net_ret_pct']:>18}%")
    print(f"{'Annualized CAGR':<28} | {res_overnight['cagr_pct']:>18}% | {res_intraday['cagr_pct']:>18}%")
    print(f"{'Win Rate':<28} | {res_overnight['win_rate_pct']:>18}% | {res_intraday['win_rate_pct']:>18}%")
    print(f"{'Maximum Drawdown':<28} | {res_overnight['max_drawdown_pct']:>18}% | {res_intraday['max_drawdown_pct']:>18}%")
    print(f"{'Downside Sortino':<28} | {res_overnight['sortino']:>18.2f} | {res_intraday['sortino']:>18.2f}")
    print(f"{'Institutional Calmar':<28} | {res_overnight['calmar']:>18.2f} | {res_intraday['calmar']:>18.2f}")
    print(f"{'Total Trades Executed':<28} | {res_overnight['trades_count']:>18} | {res_intraday['trades_count']:>18}")
    print(f"{'Friction Incurred (bps)':<28} | {res_overnight['total_friction_bps']:>18.1f} | {res_intraday['total_friction_bps']:>18.1f}")
    print("-" * 75)

    # 4. Dispatch Paper Order to Shoonya Gateway
    print(f"\n[4/4] Executing Live Simulated Paper Order via ShoonyaGateway...")
    dispatch_res = execute_paper_order(args.ticker, df, formula_str, initial_capital=args.capital, is_etf=is_etf)

    print(f"      Decision Action       : {dispatch_res['action']}")
    print(f"      Raw Signal Value      : {dispatch_res['signal_value']:.4f}")
    print(f"      Execution Price       : ₹{dispatch_res['raw_price']:.2f}")
    print(f"      Allocated Units       : {dispatch_res['quantity']} units")
    print(f"      Order ID              : {dispatch_res['order_id']}")
    print(f"      Execution Status      : {'COMPLETE (Simulated Fill)' if dispatch_res['executed'] else 'HALTED'}")
    if dispatch_res['rejection_reason']:
        print(f"      Halt / Gate Reason    : {dispatch_res['rejection_reason']}")
    print(f"      Post-Trade Balance    : ₹{dispatch_res['remaining_virtual_cash']:.2f}")
    print(f"      Active Paper Positions: {dispatch_res['paper_positions']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
