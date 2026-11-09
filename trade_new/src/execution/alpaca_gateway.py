"""
Alpaca Markets Execution Gateway for US Equities.
Provides paper trading simulation and direct REST API connectivity to Alpaca v2.
"""
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
import os
from typing import Any, Optional
import uuid
import requests
from loguru import logger

from src.domain.models import Currency
from src.execution.base_gateway import AbstractBrokerGateway


@dataclass
class AlpacaCredentials:
    """Authentication credentials for Alpaca Trading API."""
    api_key: str = ""
    secret_key: str = ""
    paper: bool = True

    @classmethod
    def from_env(cls) -> "AlpacaCredentials":
        return cls(
            api_key=os.getenv("ALPACA_API_KEY", "").strip(),
            secret_key=os.getenv("ALPACA_SECRET_KEY", "").strip(),
            paper=os.getenv("ALPACA_PAPER", "true").lower() in ("true", "1")
        )

    def is_valid(self) -> bool:
        return bool(self.api_key and self.secret_key)


class AlpacaGateway(AbstractBrokerGateway):
    """
    Broker gateway interface for US Equities through Alpaca Markets.
    Supports both simulated paper execution and authenticated REST routing.
    """

    def __init__(
        self,
        credentials: Optional[AlpacaCredentials] = None,
        paper_trading: bool = True,
        initial_balance_usd: Decimal = Decimal("100000.0")
    ) -> None:
        self.creds = credentials or AlpacaCredentials.from_env()
        self.paper_trading = paper_trading
        self.balance_usd = initial_balance_usd
        self.positions: dict[str, dict[str, Any]] = {}
        self.orders: dict[str, dict[str, Any]] = {}
        self.base_url = (
            "https://paper-api.alpaca.markets/v2"
            if (self.paper_trading or self.creds.paper)
            else "https://api.alpaca.markets/v2"
        )

    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.creds.api_key,
            "APCA-API-SECRET-KEY": self.creds.secret_key,
            "Content-Type": "application/json"
        }

    def place_order(self, order: dict[str, Any]) -> dict[str, Any]:
        """Dispatches an order to Alpaca or paper portfolio."""
        symbol = str(order.get("symbol", "")).strip().upper()
        quantity = Decimal(str(order.get("quantity", "0")))
        price = Decimal(str(order.get("price", "0.0")))
        action = str(order.get("action", "BUY")).strip().upper()
        order_type = str(order.get("order_type", "market")).lower()
        clordid = str(order.get("client_order_id") or f"ALP_{uuid.uuid4().hex[:12]}")

        if quantity <= Decimal("0.0"):
            return {"stat": "Not_Ok", "emsg": "Quantity must be positive"}
        if action not in ("BUY", "SELL"):
            return {"stat": "Not_Ok", "emsg": f"Invalid action {action}"}

        if self.paper_trading or not self.creds.is_valid():
            return self._execute_paper_order(symbol, quantity, price, action, order_type, clordid)
        return self._send_live_order(symbol, quantity, price, action, order_type, clordid)

    def _execute_paper_order(
        self, symbol: str, quantity: Decimal, price: Decimal, action: str, order_type: str, clordid: str
    ) -> dict[str, Any]:
        """Executes simulated paper trade with cash balance tracking."""
        trade_value = quantity * price
        if action == "BUY" and trade_value > self.balance_usd:
            return {"stat": "Not_Ok", "emsg": f"Insufficient USD cash {self.balance_usd} < {trade_value}"}

        order_id = f"APCA_SIM_{uuid.uuid4().hex[:8]}"
        if action == "BUY":
            self.balance_usd -= trade_value
            pos = self.positions.get(symbol, {"quantity": Decimal("0.0"), "avg_price": Decimal("0.0")})
            new_qty = pos["quantity"] + quantity
            new_avg = ((pos["quantity"] * pos["avg_price"]) + trade_value) / new_qty
            self.positions[symbol] = {"quantity": new_qty, "avg_price": round(new_avg, 4), "currency": "USD"}
        else:
            pos = self.positions.get(symbol, {"quantity": Decimal("0.0"), "avg_price": Decimal("0.0")})
            if pos["quantity"] < quantity:
                return {"stat": "Not_Ok", "emsg": f"Cannot sell {quantity}, holding {pos['quantity']}"}
            self.balance_usd += trade_value
            new_qty = pos["quantity"] - quantity
            if new_qty <= Decimal("0.0"):
                self.positions.pop(symbol, None)
            else:
                self.positions[symbol]["quantity"] = new_qty

        record = {
            "stat": "Ok", "order_id": order_id, "client_order_id": clordid,
            "symbol": symbol, "qty": float(quantity), "price": float(price),
            "action": action, "status": "FILLED", "currency": "USD",
            "balance_after": float(self.balance_usd)
        }
        self.orders[order_id] = record
        return record

    def _send_live_order(
        self, symbol: str, quantity: Decimal, price: Decimal, action: str, order_type: str, clordid: str
    ) -> dict[str, Any]:
        """Dispatches live order to Alpaca REST endpoint."""
        payload = {
            "symbol": symbol,
            "qty": str(quantity),
            "side": action.lower(),
            "type": order_type,
            "time_in_force": "day",
            "client_order_id": clordid
        }
        if order_type in ("limit", "lmt") and price > Decimal("0.0"):
            payload["limit_price"] = str(price)

        try:
            resp = requests.post(f"{self.base_url}/orders", json=payload, headers=self._headers(), timeout=10)
            if resp.status_code in (200, 201):
                data = resp.json()
                order_id = data.get("id", "")
                res = {"stat": "Ok", "order_id": order_id, "details": data}
                self.orders[order_id] = res
                return res
            return {"stat": "Not_Ok", "emsg": resp.text}
        except Exception as e:
            logger.error(f"Alpaca order dispatch failed: {e}")
            return {"stat": "Not_Ok", "emsg": str(e)}

    def cancel_order(self, order_id: str) -> bool:
        """Cancels open order by Alpaca order ID."""
        if self.paper_trading or not self.creds.is_valid():
            if order_id in self.orders:
                self.orders[order_id]["status"] = "CANCELED"
                return True
            return True
        try:
            resp = requests.delete(f"{self.base_url}/orders/{order_id}", headers=self._headers(), timeout=10)
            return resp.status_code in (200, 204)
        except Exception as e:
            logger.error(f"Alpaca order cancellation failed: {e}")
            return False

    def get_positions(self) -> list[dict[str, Any]]:
        """Returns list of open positions."""
        if self.paper_trading or not self.creds.is_valid():
            return [
                {
                    "symbol": sym,
                    "quantity": float(pos["quantity"]),
                    "avg_price": float(pos["avg_price"]),
                    "currency": "USD"
                }
                for sym, pos in self.positions.items()
            ]
        try:
            resp = requests.get(f"{self.base_url}/positions", headers=self._headers(), timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return [
                    {
                        "symbol": item.get("symbol"),
                        "quantity": float(item.get("qty", 0)),
                        "avg_price": float(item.get("avg_entry_price", 0.0)),
                        "currency": "USD"
                    }
                    for item in data
                ]
        except Exception as e:
            logger.error(f"Failed to query Alpaca positions: {e}")
        return []

    def get_cash_balance(self, currency: Currency) -> Decimal:
        """Returns settled available cash balance (USD only)."""
        if currency != Currency.USD:
            return Decimal("0.0")
        if self.paper_trading or not self.creds.is_valid():
            return self.balance_usd
        try:
            resp = requests.get(f"{self.base_url}/account", headers=self._headers(), timeout=10)
            if resp.status_code == 200:
                cash = resp.json().get("cash", "0.0")
                return Decimal(str(cash))
        except Exception as e:
            logger.error(f"Failed to query Alpaca account cash: {e}")
        return Decimal("0.0")

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        """Fetches order lifecycle state."""
        if self.paper_trading or not self.creds.is_valid():
            order = self.orders.get(order_id)
            if order:
                return {"stat": "Ok", "order_id": order_id, "status": order.get("status", "FILLED")}
            return {"stat": "Ok", "order_id": order_id, "status": "FILLED" if "SIM" in order_id else "UNKNOWN"}
        try:
            resp = requests.get(f"{self.base_url}/orders/{order_id}", headers=self._headers(), timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return {"stat": "Ok", "order_id": order_id, "status": data.get("status", ""), "details": data}
        except Exception as e:
            logger.error(f"Alpaca get_order_status error: {e}")
        return {"stat": "Not_Ok", "order_id": order_id, "emsg": "Order not found"}
