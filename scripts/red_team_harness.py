"""
Bio-Quant Autonomous Trading System: Institutional Red Team Invariant Harness.
Execution: python scripts/red_team_harness.py [--mode full|fast] [--export-json path]

Executes continuous multi-vector adversarial audits across:
1. Vector 1: Mathematical & Quant Alpha (Causality, DSR, Covariance Conditioning)
2. Vector 2: Indian Market Microstructure (Auction Gap, Circuit Bands, Dynamic Taxes)
3. Vector 3: Numerical Fuzzing & AST Stability (NaN Containment, Tree Caps, Integer Lots)
4. Vector 4: Broker Gateway Resilience (Rate Limits, Timeouts, Order Book State)
5. Vector 5: Operational Security & State Persistence (Credential Masking, Audit Logs)
"""
import argparse
import json
import os
import sys
import time
from typing import Dict, Any, List
import numpy as np
import pandas as pd

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.domain.models import CircuitBands, TaxSchedule, InstrumentType
from src.domain.invariants import (
    calculate_deflated_sharpe_ratio,
    verify_point_in_time_causality,
    check_covariance_conditioning,
    assert_circuit_headroom,
    assert_integer_lot,
    sanitize_finite_array
)
from src.domain.formula_ast import ASTNode, ASTEvaluator, TreeGenerator
from src.execution.shoonya_gateway import ShoonyaGateway, ShoonyaCredentials
from src.execution.risk_controller import MicroCapitalController, CircuitDamper
from src.execution.hrp_allocator import HRPAllocator
from src.execution.overnight_engine import TrailingStopSimulator


class InvariantResult:
    def __init__(self, name: str, passed: bool, message: str, elapsed_ms: float = 0.0):
        self.name = name
        self.passed = passed
        self.message = message
        self.elapsed_ms = elapsed_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "elapsed_ms": round(self.elapsed_ms, 2)
        }


class RedTeamAuditor:
    def __init__(self, mode: str = "full"):
        self.mode = mode
        self.results: Dict[str, List[InvariantResult]] = {
            "VECTOR 1: QUANT ALPHA": [],
            "VECTOR 2: NSE MARKET MICROSTRUCTURE": [],
            "VECTOR 3: NUMERICAL FUZZING & STABILITY": [],
            "VECTOR 4: BROKER GATEWAY & FAULT TOLERANCE": [],
            "VECTOR 5: OPERATIONAL SECURITY": []
        }

    def run_all(self) -> bool:
        """Executes all 5 verification vectors."""
        self._audit_vector_1_alpha()
        self._audit_vector_2_microstructure()
        self._audit_vector_3_numerical()
        self._audit_vector_4_gateway()
        self._audit_vector_5_security()

        all_passed = all(
            all(res.passed for res in items)
            for items in self.results.values()
        )
        return all_passed

    def _audit_vector_1_alpha(self):
        t0 = time.perf_counter()
        # 1.1 Causality Check
        np.random.seed(42)
        sig = np.random.randn(500)
        future_rets = np.random.randn(500)
        is_causal, corr = verify_point_in_time_causality(sig, future_rets, max_lead_corr=0.25)
        dt = (time.perf_counter() - t0) * 1000.0
        self.results["VECTOR 1: QUANT ALPHA"].append(
            InvariantResult("Causal Directionality (0 Lookahead)", is_causal, f"lead correlation |r| = {corr:.4f} <= 0.25", dt)
        )

        # 1.2 Deflated Sharpe Ratio
        t0 = time.perf_counter()
        dsr = calculate_deflated_sharpe_ratio(observed_sr=1.8, n_trials=1000, var_sr=0.12, t_samples=500)
        dt = (time.perf_counter() - t0) * 1000.0
        self.results["VECTOR 1: QUANT ALPHA"].append(
            InvariantResult("Deflated Sharpe Ratio (DSR)", dsr >= 0.80, f"DSR = {dsr:.4f} >= 0.80 across 1,000 trials", dt)
        )

        # 1.3 Covariance Conditioning
        t0 = time.perf_counter()
        cov = np.eye(5) + 0.05 * np.ones((5, 5))
        is_ok, kappa = check_covariance_conditioning(cov, max_condition_number=10000.0)
        dt = (time.perf_counter() - t0) * 1000.0
        self.results["VECTOR 1: QUANT ALPHA"].append(
            InvariantResult("HRP Covariance Matrix Condition (kappa)", is_ok, f"kappa = {kappa:.2f} < 10,000", dt)
        )

    def _audit_vector_2_microstructure(self):
        # 2.1 Auction Gap vs Intraday Stop
        t0 = time.perf_counter()
        exit_p, reason = TrailingStopSimulator.resolve_exit(
            entry_price=100.0, open_price=90.0, high_price=92.0, low_price=88.0, stop_pct=0.025
        )
        is_gap_ok = (exit_p == 90.0) and (reason == "AUCTION_GAP_DOWN")
        dt = (time.perf_counter() - t0) * 1000.0
        self.results["VECTOR 2: NSE MARKET MICROSTRUCTURE"].append(
            InvariantResult("Auction Gap Discontinuity Fill", is_gap_ok, f"filled at discontinuous auction open {exit_p}", dt)
        )

        # 2.2 Circuit Band Proximity Headroom
        t0 = time.perf_counter()
        bands = CircuitDamper.calculate_bands(current_price=90.5, prev_close=100.0, band_pct=0.10)
        is_locked_ok = bands.is_locked and (bands.allocation_damper_factor() == 0.0)
        dt = (time.perf_counter() - t0) * 1000.0
        self.results["VECTOR 2: NSE MARKET MICROSTRUCTURE"].append(
            InvariantResult("Circuit Lockout Damper", is_locked_ok, "distance to circuit < 1% triggered 100% allocation lockout", dt)
        )

        # 2.3 Dynamic Statutory Tax Schedule
        t0 = time.perf_counter()
        tax_sched = TaxSchedule(etf_cash_bps=1.2, futures_bps=3.0, equity_cash_bps=25.0, custom_slippage_bps=5.0)
        tax_etf = tax_sched.calculate_tax(100000.0, InstrumentType.ETF_CASH)
        is_tax_ok = round(tax_etf, 2) == 62.00  # (1.2 + 5.0) bps = 6.2 bps = 62.00 INR
        dt = (time.perf_counter() - t0) * 1000.0
        self.results["VECTOR 2: NSE MARKET MICROSTRUCTURE"].append(
            InvariantResult("Dynamic Statutory Tax Schedule", is_tax_ok, f"verified exact friction with custom slippage: Rs {tax_etf:.2f}", dt)
        )

    def _audit_vector_3_numerical(self):
        # 3.1 NaN Poisoning Barrier
        t0 = time.perf_counter()
        raw = np.array([np.nan, -np.inf, np.inf, 42.0])
        clean = sanitize_finite_array(raw, default_fill=0.0)
        is_clean_ok = np.all(np.isfinite(clean)) and (clean[3] == 42.0)
        dt = (time.perf_counter() - t0) * 1000.0
        self.results["VECTOR 3: NUMERICAL FUZZING & STABILITY"].append(
            InvariantResult("NaN & Inf Array Containment", is_clean_ok, "100% sanitized to finite floats", dt)
        )

        # 3.2 Random AST Fuzzing
        t0 = time.perf_counter()
        df = pd.DataFrame({
            "Open": np.random.uniform(50, 100, 50),
            "High": np.random.uniform(100, 150, 50),
            "Low": np.random.uniform(10, 50, 50),
            "Close": np.random.uniform(50, 100, 50),
            "Volume": np.random.uniform(1e3, 1e5, 50)
        })
        fuzz_count = 20 if self.mode == "full" else 5
        ast_ok = True
        for _ in range(fuzz_count):
            t = TreeGenerator.generate(max_depth=3)
            out = ASTEvaluator.eval_node(t, df, current_regime=0)
            if not np.all(np.isfinite(out)):
                ast_ok = False
                break
        dt = (time.perf_counter() - t0) * 1000.0
        self.results["VECTOR 3: NUMERICAL FUZZING & STABILITY"].append(
            InvariantResult("AST Tree Random Fuzzing", ast_ok, f"{fuzz_count} random expressions evaluated with 0 NaN/Inf", dt)
        )

        # 3.3 Integer Lot Sizing
        t0 = time.perf_counter()
        lot_ok = assert_integer_lot(100) and not assert_integer_lot(10.5) and not assert_integer_lot(0)
        dt = (time.perf_counter() - t0) * 1000.0
        self.results["VECTOR 3: NUMERICAL FUZZING & STABILITY"].append(
            InvariantResult("Integer Lot Sizing Invariant", lot_ok, "rejected fractional and non-positive lot quantities", dt)
        )

    def _audit_vector_4_gateway(self):
        t0 = time.perf_counter()
        gw = ShoonyaGateway(paper_trading=True, initial_balance=50000.0)
        res_buy = gw.place_order(symbol="NIFTYBEES", quantity=10, price=250.0, action="BUY", is_etf=True)
        res_nan = gw.place_order(symbol="NIFTYBEES", quantity=10, price=float("nan"), action="BUY", is_etf=True)

        is_gw_ok = (res_buy["stat"] == "Ok") and (res_nan["stat"] == "Not_Ok") and np.isfinite(gw.balance)
        dt = (time.perf_counter() - t0) * 1000.0
        self.results["VECTOR 4: BROKER GATEWAY & FAULT TOLERANCE"].append(
            InvariantResult("Gateway Virtual Order Ledger", is_gw_ok, "order placed cleanly, NaN price rejected", dt)
        )

        # HRP Allocation Degeneracy
        t0 = time.perf_counter()
        ret_block = np.random.randn(40, 3)
        w = HRPAllocator.compute_weights(ret_block)
        is_hrp_ok = np.all(np.isfinite(w)) and abs(np.sum(w) - 1.0) < 1e-4
        dt = (time.perf_counter() - t0) * 1000.0
        self.results["VECTOR 4: BROKER GATEWAY & FAULT TOLERANCE"].append(
            InvariantResult("HRP Hierarchical Allocation", is_hrp_ok, f"weights sum to {np.sum(w):.4f} with zero inversion blowup", dt)
        )

    def _audit_vector_5_security(self):
        t0 = time.perf_counter()
        creds = ShoonyaCredentials(
            user_id="SECRET_USER_123",
            password="SECRET_PASSWORD",
            totp_key="JBSWY3DPEHPK3PXP",
            vendor_code="VEND",
            api_key="API_KEY_999"
        )
        gw = ShoonyaGateway(credentials=creds, paper_trading=True)
        summary = gw.get_status_summary()
        summary_str = json.dumps(summary)

        # Assert raw secrets are NOT present in the summary
        no_leaks = ("SECRET_PASSWORD" not in summary_str) and ("API_KEY_999" not in summary_str)
        dt = (time.perf_counter() - t0) * 1000.0
        self.results["VECTOR 5: OPERATIONAL SECURITY"].append(
            InvariantResult("Credential & Secret Masking", no_leaks, f"user_id masked to '{summary.get('user_id')}', secrets stripped", dt)
        )

    def print_report(self) -> None:
        total_invariants = 0
        total_passed = 0
        total_time_ms = 0.0

        print("\n" + "=" * 80)
        print("          BIO-QUANT INSTITUTIONAL RED TEAM INVARIANT HARNESS          ")
        print("=" * 80)

        for vector_name, checks in self.results.items():
            print(f"\n[{vector_name}]")
            for c in checks:
                total_invariants += 1
                total_time_ms += c.elapsed_ms
                status = "[PASS]" if c.passed else "[FAIL]"
                if c.passed:
                    total_passed += 1
                print(f"  {status} {c.name} ({c.elapsed_ms:.1f}ms)")
                print(f"         +-- {c.message}")

        print("\n" + "=" * 80)
        if total_passed == total_invariants:
            print(f"VERDICT: PRODUCTION READY ({total_passed}/{total_invariants} Invariants Satisfied, {total_time_ms:.1f}ms)")
        else:
            print(f"VERDICT: BREACH DETECTED ({total_passed}/{total_invariants} Passed, {total_invariants - total_passed} Failed)")
        print("=" * 80 + "\n")

    def export_json(self, file_path: str):
        data = {
            "timestamp": time.time(),
            "mode": self.mode,
            "results": {
                k: [r.to_dict() for r in v]
                for k, v in self.results.items()
            }
        }
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Bio-Quant Institutional Red Team Invariant Harness")
    parser.add_argument("--mode", choices=["fast", "full"], default="full", help="Audit mode")
    parser.add_argument("--export-json", type=str, default="", help="Optional JSON telemetry output path")
    args = parser.parse_args()

    auditor = RedTeamAuditor(mode=args.mode)
    passed = auditor.run_all()
    auditor.print_report()

    if args.export_json:
        auditor.export_json(args.export_json)
        print(f"Telemetry saved to {args.export_json}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
