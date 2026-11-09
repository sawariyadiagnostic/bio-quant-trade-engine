"""
Abstract Broker Gateway Interface.
Standardizes order routing, lifecycle tracking, and balance inspection across
domestic (NSE / Shoonya), international equities (Alpaca / IBKR), and continuous
crypto perpetuals (CCXT / Binance / Bybit).
"""
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any, Optional
from src.domain.models import Currency


class AbstractBrokerGateway(ABC):
    """
    Abstract Broker Gateway enforcing strict interface conformance for
    all execution backends.
    """

    @abstractmethod
    def place_order(self, order: dict[str, Any]) -> dict[str, Any]:
        """
        Dispatches an order to the execution engine or broker API.

        Expected order payload schema:
            symbol (str): Instrument symbol / ticker.
            quantity (int | float): Order size.
            price (float): Limit price or zero for market orders.
            action (str): 'BUY' or 'SELL'.
            order_type (str): 'LMT', 'MKT', 'SL', etc.
            client_order_id (str, optional): Idempotent order tracking token.
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancels an active working order.

        Args:
            order_id (str): Broker order ID or client order ID.

        Returns:
            bool: True if cancellation succeeded or was accepted, False otherwise.
        """
        pass

    @abstractmethod
    def get_positions(self) -> list[dict[str, Any]]:
        """
        Retrieves current active open positions.

        Returns:
            list[dict]: Standardized positions list with symbol, quantity,
                        average entry price, and currency.
        """
        pass

    @abstractmethod
    def get_cash_balance(self, currency: Currency) -> Decimal:
        """
        Retrieves available settled cash balance for the given currency.

        Args:
            currency (Currency): Statutory currency ISO token.

        Returns:
            Decimal: Available cash balance.
        """
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> dict[str, Any]:
        """
        Queries status and fill metadata for a given order ID.

        Args:
            order_id (str): Broker order ID or client tracking identifier.

        Returns:
            dict: Standardized status dictionary containing status, filled_qty, etc.
        """
        pass
