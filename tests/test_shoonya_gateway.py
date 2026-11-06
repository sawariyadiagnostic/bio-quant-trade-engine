"""
Unit tests for Shoonya (Finvasia) Gateway and ₹10,000 Micro-Capital Controller.
Empirically verifies:
1. Paper Trading order lifecycle, balance updates, and friction deduction.
2. Symbol formatting and TOTP generation.
3. Micro-Capital Controller: 0% leverage, ETF vs Single Stock caps, and crash regime gating.
"""
import os
import sys
import tempfile
import pyotp
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.domain.models import CircuitBands
from src.execution.shoonya_gateway import ShoonyaGateway, ShoonyaCredentials
from src.execution.risk_controller import MicroCapitalController

def test_totp_generation():
    """Verifies that a valid base32 key generates a 6-digit numeric TOTP code."""
    secret = pyotp.random_base32()
    creds = ShoonyaCredentials(
        user_id="FA1234",
        password="dummy_password",
        totp_key=secret,
        vendor_code="FA1234_U",
        api_key="dummy_api_key"
    )
    gw = ShoonyaGateway(credentials=creds, paper_trading=True)
    totp = gw.generate_totp()
    assert len(totp) == 6
    assert totp.isdigit()

def test_symbol_formatting():
    """Verifies standard NSE suffix normalization."""
    assert ShoonyaGateway.format_trading_symbol("NIFTYBEES") == "NIFTYBEES-EQ"
    assert ShoonyaGateway.format_trading_symbol("NIFTYBEES.NS") == "NIFTYBEES-EQ"
    assert ShoonyaGateway.format_trading_symbol("INFY-EQ") == "INFY-EQ"
    assert ShoonyaGateway.format_trading_symbol("HAL") == "HAL-EQ"

def test_friction_calculation():
    """Verifies statutory friction: 1.2 bps for ETF vs 25 bps for single equity."""
    trade_value = 10000.0
    etf_friction = ShoonyaGateway.calculate_friction(trade_value, is_etf=True)
    equity_friction = ShoonyaGateway.calculate_friction(trade_value, is_etf=False)

    assert round(etf_friction, 2) == 1.20 # ₹1.20 per ₹10,000
    assert round(equity_friction, 2) == 25.00 # ₹25.00 per ₹10,000

def test_paper_trading_order_flow():
    """Verifies order execution, portfolio tracking, and balance updates."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        audit_file = os.path.join(tmp_dir, "test_orders.json")
        gw = ShoonyaGateway(
            paper_trading=True,
            initial_balance=10000.0,
            audit_file=audit_file
        )

        # 1. Buy 36 units of NIFTYBEES @ ₹270 = ₹9,720 + ~₹1.17 friction
        res_buy = gw.place_order(
            symbol="NIFTYBEES",
            quantity=36,
            price=270.0,
            action="BUY",
            is_etf=True
        )
        assert res_buy["stat"] == "Ok"
        assert res_buy["norenordno"].startswith("SIM_")
        assert gw.paper_positions["NIFTYBEES-EQ"]["quantity"] == 36
        assert gw.paper_positions["NIFTYBEES-EQ"]["avg_price"] == 270.0
        assert gw.balance < (10000.0 - 9720.0) # deducted cost + friction

        # 2. Try to buy more when balance is insufficient
        res_fail = gw.place_order(
            symbol="NIFTYBEES",
            quantity=10,
            price=270.0,
            action="BUY",
            is_etf=True
        )
        assert res_fail["stat"] == "Not_Ok"
        assert "Insufficient virtual cash" in res_fail["emsg"]

        # 3. Sell 36 units @ ₹275 = ₹9,900 - friction
        res_sell = gw.place_order(
            symbol="NIFTYBEES",
            quantity=36,
            price=275.0,
            action="SELL",
            is_etf=True
        )
        assert res_sell["stat"] == "Ok"
        assert "NIFTYBEES-EQ" not in gw.paper_positions # closed out
        assert gw.balance > 10000.0 # net profit realized!

def test_micro_capital_controller_etf_vs_equity():
    """Verifies zero-leverage bounds and ETF vs Single Equity allocation caps."""
    ctrl = MicroCapitalController(total_capital=10000.0)

    # NIFTYBEES @ ₹270: ETF allows 98% of ₹10k = ₹9,800 // 270 = 36 units
    units_etf, reason_etf = ctrl.allocate(current_price=270.0, is_etf=True)
    assert reason_etf == "ALLOCATED_OK"
    assert units_etf == 36
    assert (units_etf * 270.0) <= 10000.0

    # Single Equity (INFY) @ ₹1,850: Capped at 30% of ₹10k = ₹2,940 // 1850 = 1 share
    units_eq, reason_eq = ctrl.allocate(current_price=1850.0, is_etf=False)
    assert reason_eq == "ALLOCATED_OK"
    assert units_eq == 1
    assert (units_eq * 1850.0) <= 3000.0

def test_micro_capital_crash_regime_protection():
    """Verifies that noise regime (-1) triggers immediate capital preservation."""
    ctrl = MicroCapitalController(total_capital=10000.0)

    # Regime 0 (Normal): Allowed
    units, reason = ctrl.allocate(current_price=270.0, is_etf=True, regime_id=0)
    assert units > 0
    assert reason == "ALLOCATED_OK"

    # Regime -1 (Market Crash / Noise State): Halted
    units_halted, reason_halted = ctrl.allocate(current_price=270.0, is_etf=True, regime_id=-1)
    assert units_halted == 0
    assert reason_halted == "REGIME_CRASH_PROTECT_NOISE"

def test_micro_capital_drawdown_circuit_breaker():
    """Verifies that exceeding 3.5% drawdown halts new allocations."""
    ctrl = MicroCapitalController(total_capital=10000.0, max_drawdown_limit_pct=3.5)

    # Balance drops from ₹10,000 to ₹9,600 (4.0% drawdown > 3.5%)
    ctrl.update_balance(9600.0)
    assert ctrl.current_drawdown_pct == 4.0

    units, reason = ctrl.allocate(current_price=270.0, is_etf=True, regime_id=0)
    assert units == 0
    assert "MAX_DRAWDOWN_BREACHED" in reason

def test_micro_capital_circuit_band_lockout():
    """Verifies statutory lockout if within 1% of daily circuit band."""
    ctrl = MicroCapitalController(total_capital=10000.0)
    bands_locked = CircuitBands(
        lower_circuit=90.0,
        upper_circuit=110.0,
        distance_to_lower_pct=0.005, # 0.5% away (locked)
        distance_to_upper_pct=0.15
    )
    units, reason = ctrl.allocate(current_price=100.0, is_etf=True, circuit_bands=bands_locked)
    assert units == 0
    assert reason == "CIRCUIT_LOCKED"


def test_gateway_idempotency_and_sell_routing():
    """Verify client_order_id idempotency and dispatcher active sell liquidation."""
    from src.execution.execution_dispatcher import ExecutionDispatcher
    from src.execution.risk_controller import MicroCapitalController

    gw = ShoonyaGateway(paper_trading=True, initial_balance=50000.0)

    # 1. Idempotency test: duplicate client_order_id must return cached confirmation
    clordid = "CLORD_TEST_12345"
    res1 = gw.place_order("NIFTYBEES", quantity=10, price=250.0, action="BUY", client_order_id=clordid)
    assert res1["stat"] == "Ok"
    bal_after_1 = gw.balance

    # Duplicate call with same clordid
    res2 = gw.place_order("NIFTYBEES", quantity=10, price=250.0, action="BUY", client_order_id=clordid)
    assert res2["stat"] == "Ok"
    assert res2["norenordno"] == res1["norenordno"]
    assert gw.balance == bal_after_1  # Balance must NOT be deducted twice!

    # 2. Dispatcher sell routing: liquidation executes through gateway
    rc = MicroCapitalController(total_capital=50000.0)
    dispatcher = ExecutionDispatcher(gateway=gw, risk_controller=rc)

    df = pd.DataFrame({"Close": [250.0, 252.0]})
    dec = dispatcher.evaluate_and_dispatch("NIFTYBEES", df, formula="-0.50", is_etf=True)
    assert dec.action == "SELL"
    assert dec.quantity == 10
    assert dec.executed is True


if __name__ == "__main__":
    print("Running Shoonya Gateway and Micro-Capital Controller Test Suite...")
    test_totp_generation()
    print("  [PASS] test_totp_generation")
    test_symbol_formatting()
    print("  [PASS] test_symbol_formatting")
    test_friction_calculation()
    print("  [PASS] test_friction_calculation")
    test_paper_trading_order_flow()
    print("  [PASS] test_paper_trading_order_flow")
    test_micro_capital_controller_etf_vs_equity()
    print("  [PASS] test_micro_capital_controller_etf_vs_equity")
    test_micro_capital_crash_regime_protection()
    print("  [PASS] test_micro_capital_crash_regime_protection")
    test_micro_capital_drawdown_circuit_breaker()
    print("  [PASS] test_micro_capital_drawdown_circuit_breaker")
    test_micro_capital_circuit_band_lockout()
    print("  [PASS] test_micro_capital_circuit_band_lockout")
    test_gateway_idempotency_and_sell_routing()
    print("  [PASS] test_gateway_idempotency_and_sell_routing")
    print("ALL 9 EMPIRICAL TESTS PASSED SUCCESSFULLY!")

