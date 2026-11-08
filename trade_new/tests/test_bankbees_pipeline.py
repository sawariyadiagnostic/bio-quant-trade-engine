"""Unit test suite for BANKBEES Overnight Paper Execution Pipeline.

Validates:
1. Pipeline initialization and VectorDataLake loading from real BANKBEES.parquet.
2. Vectorized batch AST formula evaluation, return-space clustering, and Occam's pruning.
3. Paper trading order generation via ShoonyaGateway with integer lot sizing and exact 1.2 bps ETF friction.
4. Full end-to-end pipeline execution with zero crashes and strict invariant adherence.
"""
import inspect
import os
import sys
import unittest
import numpy as np

# Ensure trade_new is first on sys.path
test_dir = os.path.dirname(os.path.abspath(__file__))
trade_new_dir = os.path.abspath(os.path.join(test_dir, ".."))
if "" in sys.path:
    sys.path.remove("")
cwd = os.getcwd()
if cwd in sys.path and os.path.abspath(cwd) != trade_new_dir:
    sys.path.remove(cwd)
if trade_new_dir not in sys.path:
    sys.path.insert(0, trade_new_dir)

from src.domain.invariants import assert_integer_lot
from src.domain.models import InstrumentType
from src.execution.overnight_engine import TaxCalculator
from src.execution.shoonya_gateway import ShoonyaGateway
from src.pipeline.bankbees_pipeline import (
    AlphaChampion,
    BankbeesOvernightPipeline,
    PipelineResult,
)


def test_bankbees_pipeline_initialization():
    """Verify pipeline initializes cleanly and loads real BANKBEES parquet into VectorDataLake."""
    pipeline = BankbeesOvernightPipeline(capital=10000.0)
    lake, df = pipeline.load_data(lookback_years=5.0)

    assert lake.ticker == "BANKBEES"
    assert len(df) > 1000, f"Expected >1000 bars in BANKBEES.parquet, got {len(df)}"
    assert "gross_overnight_rets" in lake.tensors
    assert "gross_intraday_rets" in lake.tensors
    assert "gross_close_to_close_rets" in lake.tensors
    assert "Dist_VWAP" in lake.tensors
    assert "Vol_ZScore" in lake.tensors

    # Test PyTorch feature tensor construction
    X, feat_names = pipeline.prepare_feature_tensor(lake, df)
    assert X.ndim == 2
    assert X.shape[0] == len(df)
    assert X.shape[1] == len(feat_names) == 20
    assert not np.isnan(X.numpy()).any(), "Feature tensor contains NaNs"
    assert np.all(np.isfinite(X.numpy())), "Feature tensor contains non-finite values"
    print("[PASS] test_bankbees_pipeline_initialization")


def test_batch_formula_evaluation_and_clustering():
    """Verify batch AST evaluation, return-space clustering, and Occam's champion pruning."""
    pipeline = BankbeesOvernightPipeline(capital=10000.0)
    lake, df = pipeline.load_data(lookback_years=2.0)
    X, _ = pipeline.prepare_feature_tensor(lake, df)

    candidates = [
        {"formula_str": "DELTA_5(TS_RANK_20(TS_MIN_10(TS_MIN_10((Vol_Price_Flow - TS_MAX_10(Vol_ZScore))))))", "fitness": 25.0},
        {"formula_str": "DELTA_5(TS_RANK_20(TS_MIN_10(TS_MIN_10((Vol_Price_Flow - TS_MAX_10(Vol_ZScore))))))", "fitness": 24.0},  # Duplicate clone
        {"formula_str": "DELTA_5((Global_Confluence - LOG_ABS(EMA_Slope)))", "fitness": 19.5},
        {"formula_str": "ABS(DELTA_5(DELTA_5(RELU((Global_Confluence - Vol_Price_Flow)))))", "fitness": 23.3},
        {"formula_str": "SIGMOID((Vol_Price_Flow - Downside_Vol_Ratio))", "fitness": 15.0},
    ]

    signals, valid_items, ast_nodes = pipeline.evaluate_formulas_batch(candidates, X)
    assert signals.shape == (len(candidates), len(df))
    assert np.all(np.isfinite(signals))

    labels, champions = pipeline.cluster_and_prune(
        signals, lake, valid_items, ast_nodes, lookback_years=2.0
    )

    assert len(labels) == len(candidates)
    assert len(champions) == len(np.unique(labels))
    assert len(champions) <= len(candidates)

    for champ in champions:
        assert isinstance(champ, AlphaChampion)
        assert champ.sortino >= 0.0
        assert champ.win_rate >= 0.0 and champ.win_rate <= 100.0
        assert champ.max_drawdown >= 0.0

    # Test ensemble backtesting
    champ_indices = [c.index for c in champions]
    summary = pipeline.backtest_ensemble(
        signals[champ_indices], lake.tensors["gross_overnight_rets"], lookback_years=2.0
    )
    assert summary.total_bars == len(df)
    assert summary.win_rate_pct >= 0.0 and summary.win_rate_pct <= 100.0
    assert summary.max_drawdown_pct >= 0.0
    assert summary.final_equity > 0.0
    print("[PASS] test_batch_formula_evaluation_and_clustering")


def test_paper_order_generation_and_tax_accounting():
    """Verify paper trading order generation with integer lot sizing and exact 1.2 bps friction."""
    temp_audit = os.path.join(trade_new_dir, "audit", "test_shoonya_audit.json")
    if os.path.exists(temp_audit):
        os.remove(temp_audit)

    capital = 15000.0
    pipeline = BankbeesOvernightPipeline(capital=capital)
    lake, df = pipeline.load_data(lookback_years=1.0)
    gateway = ShoonyaGateway(paper_trading=True, initial_balance=capital, audit_file=temp_audit)

    # Force a bullish consensus champion to trigger BUY action
    bullish_champion = AlphaChampion(
        index=0, cluster_id=1, formula_str="TEST_BULLISH",
        sortino=20.0, cagr_pct=50.0, max_drawdown=5.0, win_rate=75.0,
        current_signal=1.0, depth=3
    )

    decision = pipeline.generate_decision(
        champions=[bullish_champion], lake=lake, df=df, gateway=gateway, dry_run=False
    )

    assert decision.action == "BUY"
    assert decision.conviction_score == 1.0
    assert decision.active_signals == 1
    assert decision.total_champions == 1

    # Verify integer lot sizing invariant
    assert assert_integer_lot(decision.quantity) is True
    assert decision.quantity > 0

    latest_close = float(df.get("Close_Adj", df["Close"]).iloc[-1])
    assert decision.trade_value == round(decision.quantity * latest_close, 2)

    # Verify exact statutory friction: 1.2 bps ETF delivery STT
    expected_friction = TaxCalculator.calculate(decision.trade_value, InstrumentType.ETF_CASH)
    assert np.isclose(decision.statutory_friction, expected_friction, atol=1e-3)
    assert decision.statutory_friction == round(decision.trade_value * 0.00012, 4)

    # Verify Pre-Open Auction routing with 15 bps slippage ceiling
    assert decision.pre_open_order is not None
    assert decision.pre_open_order["side"] == "BUY"
    assert decision.pre_open_order["time_in_force"] == "PRE_OPEN_AUCTION"
    expected_limit = round(latest_close * (1.0 + 15.0 / 10000.0), 2)
    assert np.isclose(decision.pre_open_order["limit_price"], expected_limit, atol=1e-2)

    # Verify ShoonyaGateway execution in paper trading mode
    assert decision.paper_order_result is not None
    assert decision.paper_order_result["stat"] == "Ok"
    assert decision.client_order_id in gateway.clordid_ledger

    # Verify virtual position and balance updates
    pos = gateway.paper_positions.get("BANKBEES-EQ")
    assert pos is not None
    assert pos["quantity"] == decision.quantity
    expected_balance = capital - (decision.trade_value + decision.statutory_friction)
    assert np.isclose(gateway.balance, expected_balance, atol=1e-2)

    if os.path.exists(temp_audit):
        os.remove(temp_audit)
    print("[PASS] test_paper_order_generation_and_tax_accounting")


def test_end_to_end_pipeline_run():
    """Verify end-to-end pipeline execution returns consistent, validated PipelineResult."""
    pipeline = BankbeesOvernightPipeline(capital=10000.0)
    result = pipeline.run(lookback_years=1.0, limit_candidates=10, dry_run=True)

    assert isinstance(result, PipelineResult)
    assert result.ticker == "BANKBEES"
    assert result.total_bars > 200
    assert result.evaluated_count > 0
    assert result.cluster_count > 0
    assert len(result.champions) > 0
    assert result.ensemble_backtest.cagr_pct is not None
    assert result.decision.action in ("BUY", "HOLD_CASH")
    if result.decision.action == "BUY":
        assert assert_integer_lot(result.decision.quantity) is True
    print("[PASS] test_end_to_end_pipeline_run")


def load_tests(loader, standard_tests, pattern):
    """Enables unittest discover to find all test functions in this module."""
    suite = unittest.TestSuite()
    current_module = sys.modules[__name__]
    for name, func in inspect.getmembers(current_module, inspect.isfunction):
        if name.startswith("test_"):
            suite.addTest(unittest.FunctionTestCase(func))
    return suite


if __name__ == "__main__":
    unittest.main()
