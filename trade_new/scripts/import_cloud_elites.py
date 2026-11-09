"""
Institutional Cloud Elite Import & Audit Pipeline.
Parses formulas discovered on Kaggle or Google Colab, validates AST integrity,
executes acute crash regime stress testing in ThymicChamber (March 2020 & June 2024),
and commits surviving champions to the sovereign SQLite alpha archive store.
"""
import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure trade_new is first on sys.path and avoid namespace collisions
script_dir = os.path.dirname(os.path.abspath(__file__))
trade_new_dir = os.path.abspath(os.path.join(script_dir, ".."))
if "" in sys.path:
    sys.path.remove("")
cwd = os.getcwd()
if cwd in sys.path and os.path.abspath(cwd) != trade_new_dir:
    sys.path.remove(cwd)
if trade_new_dir not in sys.path:
    sys.path.insert(0, trade_new_dir)

from src.domain.formula_ast import ASTNode
from src.domain.formula_parser import FormulaParser
from src.evolution.archive_store import ArchiveStore
from src.evolution.thymic_chamber import ThymicChamber


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds and returns the command line argument parser for cloud elite imports."""
    parser = argparse.ArgumentParser(
        description="Audit and import cloud-discovered alpha formulas into sovereign alpha_elites.db"
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        required=True,
        help="Path to exported cloud JSON file (e.g. discovered_elites_BANKBEES.json)"
    )
    parser.add_argument(
        "--ticker", "-t",
        type=str,
        default="BANKBEES",
        help="Asset ticker symbol (default: BANKBEES)"
    )
    parser.add_argument(
        "--min-sortino", "-s",
        type=float,
        default=2.0,
        help="Minimum required Sortino ratio floor (default: 2.0)"
    )
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        default=False,
        help="Run full Thymic crash audit without persisting to database"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Target SQLite archive store path (default: trade_new/audit/alpha_elites.db)"
    )
    parser.add_argument(
        "--lookback",
        type=str,
        default="5y",
        help="Historical lookback period label (default: 5y)"
    )
    parser.add_argument(
        "--max-drawdown",
        type=float,
        default=0.06,
        help="Max tolerable drawdown in crash regimes before quarantine (default: 0.06 = 6.0%%)"
    )
    return parser


def load_cloud_export(file_path: str) -> Tuple[Optional[str], Optional[str], List[Dict[str, Any]]]:
    """Loads and unpacks JSON export payload from Kaggle or Google Colab."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Export file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return None, None, [normalize_candidate(item) for item in data]

    if isinstance(data, dict):
        ticker = data.get("ticker")
        lookback = data.get("lookback")
        raw_list = data.get("elites", data.get("records", data.get("candidates", [])))
        if not isinstance(raw_list, list):
            raw_list = [raw_list]
        candidates = [normalize_candidate(item) for item in raw_list if item]
        return ticker, lookback, candidates

    return None, None, []


def normalize_candidate(raw: Any) -> Dict[str, Any]:
    """Normalizes candidate items into standardized dictionary structure."""
    if isinstance(raw, str):
        return {"formula_str": raw.strip(), "fitness": 0.0, "sortino": 0.0}

    if not isinstance(raw, dict):
        return {"formula_str": str(raw), "fitness": 0.0, "sortino": 0.0}

    f_str = raw.get("formula_str", raw.get("formula", "")).strip()
    return {
        "formula_str": f_str,
        "fitness": float(raw.get("fitness", raw.get("sortino", 0.0))),
        "sortino": float(raw.get("sortino", raw.get("fitness", 0.0))),
        "net_return_pct": float(raw.get("net_return_pct", 0.0)),
        "cagr_pct": float(raw.get("cagr_pct", raw.get("net_return_pct", 0.0))),
        "max_drawdown": float(raw.get("max_drawdown", 0.0)),
        "win_rate": float(raw.get("win_rate", 0.0)),
        "turnover": float(raw.get("turnover", 0.0)),
        "volatility": float(raw.get("volatility", 0.0)),
        "profit_factor": float(raw.get("profit_factor", 0.0)),
        "trades_count": int(raw.get("trades_count", 0)),
        "row": int(raw.get("row", 0)),
        "col": int(raw.get("col", 0)),
        "active_regimes": raw.get("active_regimes", list(range(10)))
    }


def is_valid_ast(node: Optional[ASTNode]) -> bool:
    """Recursively validates that the AST contains no corrupted or stray syntactic tokens."""
    if node is None:
        return False
    if node.op == "FEATURE":
        return not any(c in node.val for c in ("(", ")", ","))
    return all(is_valid_ast(c) for c in node.children)


def audit_candidate(
    candidate: Dict[str, Any],
    chamber: ThymicChamber,
    min_sortino: float,
    max_dd_tolerance: float
) -> Dict[str, Any]:
    """Executes gate checks: Sortino floor, AST parsing, and Thymic Chamber crash stress."""
    f_str = candidate.get("formula_str", "")
    sortino = candidate.get("sortino", 0.0)

    if sortino < min_sortino:
        return {**candidate, "status": "BELOW_SORTINO", "stress_dd": 0.0, "passed": False}

    try:
        ast_node = FormulaParser.parse(f_str)
        if not is_valid_ast(ast_node):
            return {**candidate, "status": "PARSE_ERROR", "stress_dd": 1.0, "passed": False}
    except Exception:
        return {**candidate, "status": "PARSE_ERROR", "stress_dd": 1.0, "passed": False}

    stress_dd = chamber.measure_stress_drawdown(ast_node, active_regime=0)
    passed = (stress_dd <= max_dd_tolerance)
    status = "SURVIVOR" if passed else "QUARANTINED"

    return {**candidate, "status": status, "stress_dd": stress_dd, "passed": passed}


def format_audit_row(result: Dict[str, Any], idx: int) -> str:
    """Formats single candidate audit result into an aligned ASCII row."""
    f_str = result.get("formula_str", "")
    f_short = (f_str[:32] + "..") if len(f_str) > 34 else f_str
    sortino_str = f"{result.get('sortino', 0.0):.2f}"
    ret_str = f"+{result.get('net_return_pct', 0.0):.1f}%"
    dd_val = result.get("stress_dd", 0.0) * 100.0
    dd_str = f"{dd_val:.1f}%"
    status = result.get("status", "UNKNOWN")
    action = "COMMITTED" if status == "SURVIVOR" else "DISCARDED"
    return f"{idx:<3} | {f_short:<34} | {sortino_str:<7} | {ret_str:<9} | {dd_str:<10} | {status:<11} | {action}"


def render_audit_report(
    results: List[Dict[str, Any]],
    ticker: str,
    db_path: str,
    resident_count: int,
    dry_run: bool
) -> None:
    """Renders formatted ASCII audit leaderboard and summary metrics."""
    print("\n" + "=" * 94)
    print(f"       THYMIC CHAMBER CLOUD ALPHA AUDIT: {ticker} (March 2020 & June 2024 Regimes)")
    print("=" * 94)
    header = f"{'#':<3} | {'Candidate Formula':<34} | {'Sortino':<7} | {'Net Ret %':<9} | {'Stress DD%':<10} | {'Thymic':<11} | Action"
    print(header)
    print("-" * 94)

    for idx, r in enumerate(results, 1):
        print(format_audit_row(r, idx))

    print("-" * 94)
    total = len(results)
    survivors = sum(1 for r in results if r.get("status") == "SURVIVOR")
    quarantined = sum(1 for r in results if r.get("status") == "QUARANTINED")
    below_sortino = sum(1 for r in results if r.get("status") == "BELOW_SORTINO")
    parse_errors = sum(1 for r in results if r.get("status") == "PARSE_ERROR")

    print(f"Audit Summary for {ticker}:")
    print(f"  Total Candidates Ingested  : {total}")
    print(f"  Below Sortino Floor (<2.0) : {below_sortino}")
    print(f"  Parse Failures             : {parse_errors}")
    print(f"  Quarantined (Stress DD >6%): {quarantined}")
    print(f"  Surviving Champions        : {survivors}")
    mode_str = "[DRY RUN - DATABASE UNCHANGED]" if dry_run else f"Committed to {db_path}"
    print(f"  Persistence Status         : {mode_str}")
    print(f"  Total Resident Champions   : {resident_count}")
    print("=" * 94 + "\n")


def persist_survivors(
    store: ArchiveStore,
    ticker: str,
    lookback: str,
    survivors: List[Dict[str, Any]],
    dry_run: bool
) -> int:
    """Inserts surviving champions into ArchiveStore if not in dry-run mode."""
    if not dry_run and survivors:
        store.batch_upsert_elites(ticker=ticker, lookback=lookback, elites=survivors)
    return store.count_champions(ticker=ticker)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI Entry point for cloud alpha audit and import pipeline."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        json_ticker, json_lookback, candidates = load_cloud_export(args.file)
    except Exception as exc:
        print(f"[ERROR] Failed to load JSON export file '{args.file}': {exc}", file=sys.stderr)
        return 1

    ticker = (args.ticker if args.ticker != "BANKBEES" else (json_ticker or args.ticker)).upper()
    lookback = json_lookback or args.lookback
    db_path = args.db_path or os.path.join(trade_new_dir, "audit", "alpha_elites.db")

    chamber = ThymicChamber()
    audit_results = [
        audit_candidate(c, chamber, min_sortino=args.min_sortino, max_dd_tolerance=args.max_drawdown)
        for c in candidates
    ]

    store = ArchiveStore(db_path=db_path)
    survivors = [r for r in audit_results if r.get("status") == "SURVIVOR"]
    resident_count = persist_survivors(store, ticker, lookback, survivors, args.dry_run)

    render_audit_report(audit_results, ticker, db_path, resident_count, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
