"""Unit tests for Overnight Execution Engine & GIFT Nifty Ingestion (Wave 6 / P2)."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.gift_nifty_loader import GiftNiftyLoader


def test_gift_nifty_premarket_implied_gap():
    """Verify GiftNiftyLoader calculates implied opening gap with circuit limits and bias direction."""
    loader = GiftNiftyLoader()

    # Case 1: Positive overnight jump (+150 pts on 24,000 Nifty = +0.625%)
    res_bull = loader.calculate_implied_gap(gift_nifty_price=24150.0, nse_prev_close=24000.0)
    assert np.isclose(res_bull["implied_gap_pct"], 150.0 / 24000.0)
    assert res_bull["signal"] == "BULLISH_GAP"
    assert res_bull["circuit_clipped"] is False

    # Case 2: Deep gap crash down (-3000 pts = -12.5%, clipped to statutory circuit -10%)
    res_crash = loader.calculate_implied_gap(gift_nifty_price=21000.0, nse_prev_close=24000.0)
    assert np.isclose(res_crash["implied_gap_pct"], -0.10)
    assert res_crash["signal"] == "BEARISH_GAP"
    assert res_crash["circuit_clipped"] is True

    # Case 3: Flat overnight session
    res_flat = loader.calculate_implied_gap(gift_nifty_price=24010.0, nse_prev_close=24000.0)
    assert res_flat["signal"] == "NEUTRAL"
    print("[PASS] test_gift_nifty_premarket_implied_gap")


def test_preopen_auction_limit_order_routing():
    """Verify PreOpenAuctionRouter generates limit orders with slippage bounds."""
    from src.execution.overnight_engine import PreOpenAuctionRouter

    router = PreOpenAuctionRouter()
    order_buy = router.route_auction_order(
        ticker="NIFTYBEES.NS",
        side="BUY",
        indicative_price=250.0,
        quantity=100,
        slippage_ceiling_bps=15.0
    )
    assert order_buy["ticker"] == "NIFTYBEES.NS"
    assert order_buy["side"] == "BUY"
    assert order_buy["quantity"] == 100
    assert order_buy["limit_price"] == 250.38
    assert order_buy["time_in_force"] == "DAY"
    assert order_buy.get("session") == "PRE_OPEN"

    order_sell = router.route_auction_order(
        ticker="NIFTYBEES.NS",
        side="SELL",
        indicative_price=250.0,
        quantity=100,
        slippage_ceiling_bps=15.0
    )
    assert order_sell["limit_price"] == 249.62
    print("[PASS] test_preopen_auction_limit_order_routing")


def test_trailing_stop_decay():
    """Verify trailing stop tightens from 25 bps down to 10 bps as minutes elapse."""
    from src.execution.overnight_engine import TrailingStopSimulator

    entry_price = 250.0
    open_price = 251.0
    high_price = 251.5
    low_price = 251.2

    # At minute 0 (09:15), wide buffer does not trigger
    exit_p_0, reason_0 = TrailingStopSimulator.resolve_decaying_exit(
        entry_price, open_price, high_price, low_price, minutes_elapsed=0
    )
    assert reason_0 == "OPEN_AUCTION_EXIT"

    # At minute 30 (09:45), tight buffer triggers
    exit_p_30, reason_30 = TrailingStopSimulator.resolve_decaying_exit(
        entry_price, open_price, high_price, low_price, minutes_elapsed=30
    )
    assert reason_30 == "TRAILING_STOP_HIT"
    print("[PASS] test_trailing_stop_decay")


def load_tests(loader, standard_tests, pattern):
    import inspect
    import unittest
    suite = unittest.TestSuite()
    current_module = sys.modules[__name__]
    for name, func in inspect.getmembers(current_module, inspect.isfunction):
        if name.startswith("test_"):
            suite.addTest(unittest.FunctionTestCase(func))
    return suite


if __name__ == "__main__":
    import unittest
    unittest.main()

