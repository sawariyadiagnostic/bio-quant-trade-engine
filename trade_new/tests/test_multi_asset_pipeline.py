"""Unit test suite for Multi-Asset Overnight Quantitative Portfolio Engine.

Validates:
1. Instrument classification (ETF_CASH at 1.2 bps vs FUTURES at 3.0 bps) and tax friction calculations.
2. HRP risk-parity weight allocation with Ledoit-Wolf shrinkage and single-asset exposure limits (max 30%, sum <= 1.0).
3. Simultaneous multi-asset paper order generation and execution via ShoonyaGateway.
4. End-to-end multi-asset pipeline execution with strict invariant assertions.
"""
import inspect
import os
import sys
import tempfile
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
from src.pipeline.multi_asset_pipeline import (
    MultiAssetOvernightPipeline,
    MultiAssetPipelineResult,
    apply_exposure_limits,
    classify_instrument,
)


def test_instrument_classification_and_tax_friction():
    """Verify dynamic classification of ETFs vs Stock Futures and statutory frictions."""
    # 1. ETFs (*BEES) classification
    for etf_symbol in ["BANKBEES", "NIFTYBEES", "GOLDBEES", "JUNIORBEES", "NIFTYBEES.NS"]:
        itype, fric_bps = classify_instrument(etf_symbol)
        assert itype == InstrumentType.ETF_CASH, f"Expected ETF_CASH for {etf_symbol}, got {itype}"
        assert fric_bps == 1.2, f"Expected 1.2 bps for {etf_symbol}, got {fric_bps}"

    # 2. Large-Cap Stock Futures classification
    for fut_symbol in ["RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", "LT", "SBIN", "RELIANCE.NS"]:
        itype, fric_bps = classify_instrument(fut_symbol)
        assert itype == InstrumentType.FUTURES, f"Expected FUTURES for {fut_symbol}, got {itype}"
        assert fric_bps == 3.0, f"Expected 3.0 bps for {fut_symbol}, got {fric_bps}"

    # 3. Exact Indian statutory tax friction calculation
    trade_val = 100000.0  # ₹1 Lakh trade
    etf_tax = TaxCalculator.calculate(trade_val, InstrumentType.ETF_CASH)
    fut_tax = TaxCalculator.calculate(trade_val, InstrumentType.FUTURES)
    eq_tax = TaxCalculator.calculate(trade_val, InstrumentType.EQUITY_CASH)

    assert round(etf_tax, 2) == 12.00, f"Expected ₹12.00 (1.2 bps), got ₹{etf_tax}"
    assert round(fut_tax, 2) == 30.00, f"Expected ₹30.00 (3.0 bps), got ₹{fut_tax}"
    assert round(eq_tax, 2) == 250.00, f"Expected ₹250.00 (25.0 bps), got ₹{eq_tax}"
    print("[PASS] test_instrument_classification_and_tax_friction")


def test_hrp_weight_allocation_and_exposure_cap():
    """Verify HRP weight allocation respects max 30% cap and sums to <= 1.0."""
    pipeline = MultiAssetOvernightPipeline(max_asset_weight=0.30)

    # 1. Five active assets with synthetic returns
    np.random.seed(42)
    rets = np.random.normal(0.0005, 0.01, (100, 5))
    tickers = ["ASSET_A", "ASSET_B", "ASSET_C", "ASSET_D", "ASSET_E"]

    weights = pipeline.compute_risk_parity_weights(rets, tickers)
    assert len(weights) == 5
    for t, w in weights.items():
        assert w <= 0.30 + 1e-5, f"Asset {t} weight {w} exceeded 30% exposure limit"
        assert w >= 0.0, f"Asset {t} weight {w} is negative"
    assert abs(sum(weights.values()) - 1.0) < 1e-4, f"Weights sum {sum(weights.values())} != 1.0"

    # 2. Skewed raw weights redistribution test
    skewed_raw = np.array([0.60, 0.15, 0.10, 0.10, 0.05])
    capped = apply_exposure_limits(skewed_raw, max_weight=0.30)
    assert np.all(capped <= 0.30 + 1e-5), f"Capping failed: {capped}"
    assert abs(np.sum(capped) - 1.0) < 1e-4, f"Sum {np.sum(capped)} != 1.0"

    # 3. Two assets test (each capped at 30%, remaining 40% preserved in cash)
    two_raw = np.array([0.70, 0.30])
    capped_two = apply_exposure_limits(two_raw, max_weight=0.30)
    assert np.all(capped_two <= 0.30 + 1e-5)
    assert np.sum(capped_two) <= 1.0
    assert abs(np.sum(capped_two) - 0.60) < 1e-4

    # 4. Zero and single active asset edge cases
    assert pipeline.compute_risk_parity_weights(np.empty((0, 0)), []) == {}
    single_w = pipeline.compute_risk_parity_weights(rets[:, :1], ["SOLO"])
    assert single_w["SOLO"] <= 0.30 + 1e-5
    print("[PASS] test_hrp_weight_allocation_and_exposure_cap")


def test_multi_asset_paper_order_generation_and_shoonya():
    """Verify simultaneous multi-asset paper order routing through ShoonyaGateway."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        audit_file = os.path.join(tmp_dir, "test_multi_orders.json")
        gw = ShoonyaGateway(paper_trading=True, initial_balance=100000.0, audit_file=audit_file)

        # Place ETF order: 100 units of GOLDBEES @ ₹116.50
        etf_res = gw.place_order(
            symbol="GOLDBEES",
            quantity=100,
            price=116.50,
            action="BUY",
            is_etf=True,
            custom_bps=1.2,
            client_order_id="CLORD_GOLDBEES_1",
        )
        assert etf_res["stat"] == "Ok"
        assert etf_res["norenordno"].startswith("SIM_")
        # Trade val = 11,650.0, friction = 11,650 * 0.00012 = 1.398 -> rounded 1.40
        assert round(etf_res["friction"], 2) == 1.40

        # Place Stock Futures order: 30 units of RELIANCE @ ₹1340.00
        fut_res = gw.place_order(
            symbol="RELIANCE",
            quantity=30,
            price=1340.00,
            action="BUY",
            is_etf=False,
            custom_bps=3.0,
            client_order_id="CLORD_RELIANCE_1",
        )
        assert fut_res["stat"] == "Ok"
        assert fut_res["norenordno"].startswith("SIM_")
        # Trade val = 40,200.0, friction = 40,200 * 0.00030 = 12.06
        assert round(fut_res["friction"], 2) == 12.06

        # Check virtual positions
        assert gw.paper_positions["GOLDBEES-EQ"]["quantity"] == 100
        assert gw.paper_positions["RELIANCE-EQ"]["quantity"] == 30
        assert assert_integer_lot(gw.paper_positions["GOLDBEES-EQ"]["quantity"]) is True
        assert assert_integer_lot(gw.paper_positions["RELIANCE-EQ"]["quantity"]) is True

        # Test idempotency: replaying same client_order_id returns cached response
        bal_before = gw.balance
        replay_res = gw.place_order(
            symbol="GOLDBEES", quantity=100, price=116.50, client_order_id="CLORD_GOLDBEES_1"
        )
        assert replay_res["norenordno"] == etf_res["norenordno"]
        assert gw.balance == bal_before
        print("[PASS] test_multi_asset_paper_order_generation_and_shoonya")


def test_multi_asset_pipeline_end_to_end():
    """Verify end-to-end multi-asset pipeline execution returns consistent, validated results."""
    pipeline = MultiAssetOvernightPipeline(
        capital=50000.0,
        conviction_threshold=0.30,
        max_asset_weight=0.30,
    )

    res = pipeline.run(
        assets=["BANKBEES", "NIFTYBEES"],
        lookback_years=0.5,
        limit_candidates=5,
        dry_run=True,
    )

    assert isinstance(res, MultiAssetPipelineResult)
    assert res.evaluated_assets == ["BANKBEES", "NIFTYBEES"]
    assert len(res.evaluations) == 2
    assert res.total_capital == 50000.0
    assert res.cash_reserve >= 0.0

    for ticker, ev in res.evaluations.items():
        assert ev.ticker == ticker
        assert ev.instrument_type == InstrumentType.ETF_CASH
        assert ev.statutory_friction_bps == 1.2
        assert 0.0 <= ev.conviction_score <= 1.0
        assert ev.target_weight <= 0.30 + 1e-5
        if ev.action == "BUY":
            assert assert_integer_lot(ev.quantity) is True
            assert ev.client_order_id is not None
            assert ev.pre_open_order is not None

    print("[PASS] test_multi_asset_pipeline_end_to_end")


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
