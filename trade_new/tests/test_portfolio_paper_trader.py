"""Unit tests for Portfolio Paper Trader (Wave 4: Task 4.1)."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_portfolio_paper_trader_multi_asset_rebalance():
    """Verify PortfolioPaperTrader rebalances capital across multiple assets with statutory friction."""
    from src.execution.portfolio_paper_trader import PortfolioPaperTrader

    trader = PortfolioPaperTrader(initial_capital=50000.0, friction_bps=1.2)
    assert trader.cash == 50000.0
    assert len(trader.positions) == 0

    prices = {"NIFTYBEES": 260.0, "GOLDBEES": 60.0}
    weights = {"NIFTYBEES": 0.50, "GOLDBEES": 0.50}

    # Execute Rebalance
    trades = trader.rebalance(weights, prices)

    assert len(trades) == 2
    assert "NIFTYBEES" in trader.positions
    assert "GOLDBEES" in trader.positions

    # Total equity must account for statutory friction deducted on execution
    total_eq = trader.get_total_equity(prices)
    assert total_eq < 50000.0
    assert total_eq > 49990.0  # ~1.2 bps friction on 50,000 is ~6.0 INR

    # Cash must remain positive
    assert trader.cash >= 0.0
    print("[PASS] test_portfolio_paper_trader_multi_asset_rebalance")


def test_portfolio_paper_trader_exit_to_cash():
    """Verify PortfolioPaperTrader cleanly liquidates positions back to cash."""
    from src.execution.portfolio_paper_trader import PortfolioPaperTrader

    trader = PortfolioPaperTrader(initial_capital=20000.0)
    prices = {"NIFTYBEES": 250.0}

    trader.rebalance({"NIFTYBEES": 1.0}, prices)
    assert trader.positions["NIFTYBEES"]["quantity"] > 0

    # Liquidate: Rebalance to 0% weight (100% Cash preservation)
    trades = trader.rebalance({"NIFTYBEES": 0.0}, prices)
    assert trader.positions["NIFTYBEES"]["quantity"] == 0
    assert trader.cash > 19990.0
    print("[PASS] test_portfolio_paper_trader_exit_to_cash")


def test_two_pass_rebalancing_and_liquidbees_sweep():
    """Verify sells execute before buys and idle cash is swept into Liquid BeES earning yield."""
    from src.execution.portfolio_paper_trader import PortfolioPaperTrader

    trader = PortfolioPaperTrader(initial_capital=50000.0)
    prices = {"NIFTYBEES": 250.0, "BANKBEES": 500.0, "LIQUIDBEES": 1000.0}

    # Start with 100% NIFTYBEES
    trader.rebalance({"NIFTYBEES": 1.0}, prices)
    assert trader.positions["NIFTYBEES"]["quantity"] > 0
    assert trader.cash < 250.0

    # Rebalance 100% from NIFTYBEES to BANKBEES: Sells must precede buys
    trades = trader.rebalance({"NIFTYBEES": 0.0, "BANKBEES": 1.0}, prices)
    assert len(trades) >= 2
    assert trades[0]["action"] == "SELL"
    assert trades[0]["ticker"] == "NIFTYBEES"
    assert trades[1]["action"] == "BUY"
    assert trades[1]["ticker"] == "BANKBEES"
    assert trader.positions["BANKBEES"]["quantity"] > 0

    # Auto-sweep idle cash into LIQUIDBEES
    trader.rebalance({"BANKBEES": 0.0}, prices)
    sweep_res = trader.auto_sweep_liquid_bees(liquid_bees_price=1000.0, annual_yield=0.068)
    assert "swept_units" in sweep_res
    assert sweep_res["swept_units"] > 40
    assert "LIQUIDBEES" in trader.positions
    print("[PASS] test_two_pass_rebalancing_and_liquidbees_sweep")


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

