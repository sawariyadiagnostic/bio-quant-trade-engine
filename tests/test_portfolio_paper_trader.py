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


if __name__ == "__main__":
    test_portfolio_paper_trader_multi_asset_rebalance()
    test_portfolio_paper_trader_exit_to_cash()
