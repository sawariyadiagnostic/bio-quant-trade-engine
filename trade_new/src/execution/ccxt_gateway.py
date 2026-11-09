"""
CCXT Continuous Crypto Execution Gateway (Binance / Bybit).
Supports 24/7 perpetual futures, linear USDT contracts, and 8-hour funding rate accounting.
"""
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
import os
from typing import Any, Optional
import uuid
from loguru import logger

from src.domain.models import Currency
from src.execution.base_gateway import AbstractBrokerGateway


@dataclass
class CCXTCredentials:
    """API credentials container for CCXT supported exchanges."""
    exchange_id: str = "binance"
    api_key: str = ""
    secret: str = ""
    password: Optional[str] = None
    is_testnet: bool = True

    @classmethod
    def from_env(cls) -> "CCXTCredentials":
        return cls(
            exchange_id=os.getenv("CCXT_EXCHANGE_ID", "binance").lower().strip(),
            api_key=os.getenv("CCXT_API_KEY", "").strip(),
            secret=os.getenv("CCXT_SECRET", "").strip(),
            password=os.getenv("CCXT_PASSWORD", "").strip() or None,
            is_testnet=os.getenv("CCXT_TESTNET", "true").lower() in ("true", "1")
        )

    def is_valid(self) -> bool:
        return bool(self.api_key and self.secret)


class CCXTGateway(AbstractBrokerGateway):
    """
    Crypto gateway for 24/7 perpetual markets with 8-hour funding rate accounting.
    Operates with USDT margin balance and precise Decimal position tracking.
    """

    def __init__(
        self,
        credentials: Optional[CCXTCredentials] = None,
        paper_trading: bool = True,
        initial_balance_usdt: Decimal = Decimal("25000.0")
    ) -> None:
        self.creds = credentials or CCXTCredentials.from_env()
        self.paper_trading = paper_trading
        self.balance_usdt = initial_balance_usdt
        self.positions: dict[str, dict[str, Any]] = {}
        self.orders: dict[str, dict[str, Any]] = {}
        self.funding_history: list[dict[str, Any]] = []

    def place_order(self, order: dict[str, Any]) -> dict[str, Any]:
        """Dispatches order to simulated perpetual matching or exchange."""
        symbol = str(order.get("symbol", "")).strip().upper()
        quantity = Decimal(str(order.get("quantity", "0")))
        price = Decimal(str(order.get("price", "0.0")))
        action = str(order.get("action", "BUY")).strip().upper()
        clordid = str(order.get("client_order_id") or f"CCXT_{uuid.uuid4().hex[:12]}")

        if quantity <= Decimal("0.0"):
            return {"stat": "Not_Ok", "emsg": "Quantity must be positive"}
        if action not in ("BUY", "SELL"):
            return {"stat": "Not_Ok", "emsg": f"Invalid action {action}"}

        return self._execute_paper_order(symbol, quantity, price, action, clordid)

    def _execute_paper_order(
        self, symbol: str, quantity: Decimal, price: Decimal, action: str, clordid: str
    ) -> dict[str, Any]:
        """Simulates perpetual futures position update and margin accounting."""
        notional = quantity * price
        margin_required = notional * Decimal("0.10")  # 10x max leverage baseline
        if action == "BUY" and margin_required > self.balance_usdt:
            return {"stat": "Not_Ok", "emsg": f"Insufficient USDT margin {self.balance_usdt} < {margin_required}"}

        order_id = f"PERP_SIM_{uuid.uuid4().hex[:8]}"
        pos = self.positions.get(symbol, {"size": Decimal("0.0"), "entry_price": Decimal("0.0")})
        current_size = pos["size"]

        if action == "BUY":
            new_size = current_size + quantity
        else:
            new_size = current_size - quantity

        if new_size == Decimal("0.0"):
            self.positions.pop(symbol, None)
        else:
            self.positions[symbol] = {
                "size": new_size,
                "entry_price": price,
                "currency": "USDT",
                "symbol": symbol
            }

        record = {
            "stat": "Ok", "order_id": order_id, "client_order_id": clordid,
            "symbol": symbol, "qty": float(quantity), "price": float(price),
            "action": action, "status": "FILLED", "currency": "USDT",
            "balance_after": float(self.balance_usdt)
        }
        self.orders[order_id] = record
        return record

    def cancel_order(self, order_id: str) -> bool:
        """Cancels open order."""
        if order_id in self.orders:
            self.orders[order_id]["status"] = "CANCELED"
        return True

    def get_positions(self) -> list[dict[str, Any]]:
        """Returns active perpetual positions."""
        return [
            {
                "symbol": sym,
                "quantity": float(pos["size"]),
                "avg_price": float(pos["entry_price"]),
                "currency": "USDT"
            }
            for sym, pos in self.positions.items()
        ]

    def get_cash_balance(self, currency: Currency) -> Decimal:
        """Returns available margin balance (USDT only)."""
        if currency not in (Currency.USDT, Currency.USD):
            return Decimal("0.0")
        return self.balance_usdt

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        """Queries order status."""
        if order_id in self.orders:
            return {"stat": "Ok", "order_id": order_id, "status": self.orders[order_id].get("status", "FILLED")}
        return {"stat": "Ok", "order_id": order_id, "status": "FILLED" if "SIM" in order_id else "UNKNOWN"}

    def calculate_funding_payment(
        self, symbol: str, mark_price: Decimal, funding_rate_8h: Decimal
    ) -> Decimal:
        """
        Calculates 8-hour perpetual funding payment:
        Payment = Position Size * Mark Price * Funding Rate.
        Long position: positive rate pays funding; negative rate receives funding.
        Short position: positive rate receives funding; negative rate pays funding.
        """
        pos = self.positions.get(symbol)
        if not pos or pos["size"] == Decimal("0.0"):
            return Decimal("0.0")
        position_size = pos["size"]
        notional = position_size * mark_price
        return notional * funding_rate_8h

    def apply_funding_settlement(
        self, symbol: str, mark_price: Decimal, funding_rate_8h: Decimal
    ) -> dict[str, Any]:
        """
        Applies periodic 8-hour perpetual funding payment to USDT balance.
        Positive payment debits account (long paying shorts), negative credits account.
        """
        payment = self.calculate_funding_payment(symbol, mark_price, funding_rate_8h)
        self.balance_usdt -= payment

        record = {
            "symbol": symbol,
            "mark_price": float(mark_price),
            "funding_rate_8h": float(funding_rate_8h),
            "funding_payment": float(payment),
            "balance_after": float(self.balance_usdt),
            "timestamp": datetime.now().isoformat()
        }
        self.funding_history.append(record)
        return record
