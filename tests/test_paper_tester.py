"""Unit test suite for scripts/paper_test_formula.py."""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.paper_test_formula import (
    load_vault_elites,
    select_formula,
    simulate_horizon,
    execute_paper_order
)


def _mock_df(n: int = 50) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq="B")
    p = 250.0 + np.cumsum(np.random.normal(0.1, 0.5, n))
    df = pd.DataFrame(index=idx)
    df["Open"] = p - 0.2
    df["Close_Adj"] = p
    df["Close_Raw"] = p
    df["High"] = p + 0.8
    df["Low"] = p - 0.8
    df["Volume"] = 10000.0
    df["Dist_VWAP"] = 0.01
    df["SP500_Ret"] = 0.005
    return df


def test_select_formula_modes():
    """Validates custom and vault selection modes."""
    custom_str, meta = select_formula("custom", "EMA_10(Close) - Close")
    assert custom_str == "EMA_10(Close) - Close"
    assert meta["source"] == "CLI Input"

    vault = [{"formula_str": "F1", "fitness": 10.0, "net_return_pct": 50.0},
             {"formula_str": "F2", "fitness": 50.0, "net_return_pct": 25.0}]
    f_earner, m_earner = select_formula("top_earner", vault_elites=vault)
    assert f_earner == "F1"
    f_fit, m_fit = select_formula("top_fitness", vault_elites=vault)
    assert f_fit == "F2"
    print("[PASS] test_select_formula_modes")


def test_simulate_horizon_metrics():
    """Validates that overnight and intraday simulations produce expected metrics structure."""
    df = _mock_df(60)
    signals = np.ones(60)
    res_on = simulate_horizon(df, signals, horizon="overnight", is_etf=True)
    res_id = simulate_horizon(df, signals, horizon="intraday", is_etf=True)

    for res in (res_on, res_id):
        assert "cagr_pct" in res
        assert "win_rate_pct" in res
        assert "sortino" in res
        assert "calmar" in res
        assert "total_friction_bps" in res
    print("[PASS] test_simulate_horizon_metrics")


def test_execute_paper_order_routing():
    """Validates virtual order generation through risk controller and gateway."""
    df = _mock_df(60)
    res = execute_paper_order("NIFTYBEES", df, "EMA_10(Dist_VWAP) + SP500_Ret", initial_capital=10000.0, is_etf=True)
    assert res["action"] in ("BUY", "HOLD", "SELL", "HALT")
    assert "remaining_virtual_cash" in res
    print("[PASS] test_execute_paper_order_routing")
