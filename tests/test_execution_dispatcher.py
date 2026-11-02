"""
Empirical End-to-End Validation Test Suite for ExecutionDispatcher.
Proves the continuous logic chain across phases:
Phase A (Data Lake & Features) -> Phase B (Formula AST) -> Phase C (Risk & Gateway Routing).
"""
import os
import sys
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.manifold.lake_dataset_loader import LakeDatasetLoader
from src.domain.formula_parser import FormulaParser
from src.domain.formula_ast import ASTEvaluator
from src.execution.shoonya_gateway import ShoonyaGateway, ShoonyaCredentials
from src.execution.risk_controller import MicroCapitalController
from src.execution.execution_dispatcher import ExecutionDispatcher


def _create_mock_dataset(n_rows: int = 60) -> pd.DataFrame:
    """Generates a consistent test DataFrame with required Phase A features."""
    dates = pd.date_range("2026-01-01", periods=n_rows, freq="B")
    base_price = 250.0 + np.cumsum(np.random.normal(0.2, 1.0, n_rows))
    df = pd.DataFrame(index=dates)
    df["Close_Raw"] = base_price
    df["Close_Adj"] = base_price
    df["Open"] = base_price - 0.5
    df["High"] = base_price + 1.5
    df["Low"] = base_price - 1.5
    df["Volume"] = 100000.0
    df["True_Range"] = 3.0 / base_price
    df["TR_Expansion"] = 1.1
    df["Vol_ZScore"] = 1.2
    df["Vol_Price_Flow"] = 0.8
    df["EMA_Slope"] = 0.05
    df["Dist_EMA20"] = 0.02
    df["Dist_VWAP"] = 0.01
    df["RSI_14"] = 0.55
    df["Downside_Vol_Ratio"] = 0.4
    df["Parkinson_Vol"] = 0.015
    df["Garman_Klass_Vol"] = 0.012
    df["SP500_Ret"] = 0.008
    df["ADR_Ret"] = 0.010
    df["Global_Confluence"] = 1.0
    return df


def test_dispatcher_successful_buy():
    """Verifies that a valid Phase B buy formula executes an order through Phase C gateway."""
    df = _create_mock_dataset()
    gw = ShoonyaGateway(paper_trading=True, initial_balance=10000.0)
    risk = MicroCapitalController(total_capital=10000.0)
    dispatcher = ExecutionDispatcher(gateway=gw, risk_controller=risk, signal_threshold=0.01)

    formula_str = "EMA_10(Dist_VWAP) + SP500_Ret"
    decision = dispatcher.evaluate_and_dispatch(
        ticker="NIFTYBEES.NS",
        df=df,
        formula=formula_str,
        regime_id=0,
        is_etf=True
    )

    assert decision.action == "BUY"
    assert decision.executed is True
    assert decision.quantity > 0
    assert decision.order_id is not None
    assert gw.get_limits()["cash"] < 10000.0  # Friction & cost deducted
    print(f"[PASS] test_dispatcher_successful_buy: {decision.quantity} units executed.")


def test_dispatcher_regime_noise_halt():
    """Verifies that Phase A noise regime (-1) immediately halts trading via Phase C risk gate."""
    df = _create_mock_dataset()
    gw = ShoonyaGateway(paper_trading=True, initial_balance=10000.0)
    risk = MicroCapitalController(total_capital=10000.0)
    dispatcher = ExecutionDispatcher(gateway=gw, risk_controller=risk, signal_threshold=0.01)

    formula_str = "EMA_10(Dist_VWAP) + SP500_Ret"
    decision = dispatcher.evaluate_and_dispatch(
        ticker="NIFTYBEES.NS",
        df=df,
        formula=formula_str,
        regime_id=-1,  # Phase A noise state
        is_etf=True
    )

    assert decision.action == "HALT"
    assert decision.executed is False
    assert decision.quantity == 0
    assert "CRASH_PROTECT_NOISE" in decision.rejection_reason
    assert gw.get_limits()["cash"] == 10000.0  # 100% Cash preserved
    print(f"[PASS] test_dispatcher_regime_noise_halt: Cash preserved ({decision.rejection_reason}).")


def test_dispatcher_hold_on_flat_signal():
    """Verifies that weak signals below threshold result in HOLD with zero orders."""
    df = _create_mock_dataset()
    gw = ShoonyaGateway(paper_trading=True, initial_balance=10000.0)
    risk = MicroCapitalController(total_capital=10000.0)
    dispatcher = ExecutionDispatcher(gateway=gw, risk_controller=risk, signal_threshold=5.0)  # High threshold

    formula_str = "EMA_10(Dist_VWAP) + SP500_Ret"
    decision = dispatcher.evaluate_and_dispatch(
        ticker="NIFTYBEES.NS",
        df=df,
        formula=formula_str,
        regime_id=0,
        is_etf=True
    )

    assert decision.action == "HOLD"
    assert decision.executed is False
    assert decision.quantity == 0
    print("[PASS] test_dispatcher_hold_on_flat_signal: Successfully held in cash.")
