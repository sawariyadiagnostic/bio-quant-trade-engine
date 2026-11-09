"""
Interactive Brokers (IBKR) Execution Gateway.
Supports multi-currency global execution (USD, EUR, GBP, INR) across equities,
futures, and FX instruments via IBKR Client Portal Web API / TWS.
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
class IBKRCredentials:
    """Credentials and host configuration for IBKR Client Portal Gateway."""
    account_id: str = ""
    gateway_host: str = "127.0.0.1"
    gateway_port: int = 5000
    ssl_verify: bool = False

    @classmethod
    def from_env(cls) -> "IBKRCredentials":
        return cls(
            account_id=os.getenv("IBKR_ACCOUNT_ID", "").strip(),
            gateway_host=os.getenv("IBKR_HOST", "127.0.0.1").strip(),
            gateway_port=int(os.getenv("IBKR_PORT", "5000")),
            ssl_verify=os.getenv("IBKR_SSL_VERIFY", "false").lower() in ("true", "1")
        )

    @property
    def base_url(self) -> str:
        return f"https://{self.gateway_host}:{self.gateway_port}/v1/api"


class InteractiveBrokersGateway(AbstractBrokerGateway):
    """
    Multi-currency gateway for Interactive Brokers accounts.
    Manages isolated currency balances across global venues.
    """

    def __init__(
        self,
        credentials: Optional[IBKRCredentials] = None,
        paper_trading: bool = True,
        initial_balances: Optional[dict[Currency, Decimal]] = None
    ) -> None:
        self.creds = credentials or IBKRCredentials.from_env()
        self.paper_trading = paper_trading
        self.balances: dict[Currency, Decimal] = initial_balances or {
            Currency.USD: Decimal("50000.0"),
            Currency.EUR: Decimal("25000.0"),
            Currency.GBP: Decimal("20000.0"),
            Currency.INR: Decimal("500000.0")
        }
        self.positions: dict[str, dict[str, Any]] = {}
        self.orders: dict[str, dict[str, Any]] = {}

    def place_order(self, order: dict[str, Any]) -> dict[str, Any]:
        """Submits an order to IBKR Client Portal or local simulator."""
        symbol = str(order.get("symbol", "")).strip().upper()
        quantity = Decimal(str(order.get("quantity", "0")))
        price = Decimal(str(order.get("price", "0.0")))
        action = str(order.get("action", "BUY")).strip().upper()
        currency_str = str(order.get("currency", "USD")).upper()
        currency = Currency(currency_str) if currency_str in Currency._value2member_map_ else Currency.USD
        order_type = str(order.get("order_type", "LMT")).upper()
        clordid = str(order.get("client_order_id") or f"IBKR_{uuid.uuid4().hex[:12]}")

        if quantity <= Decimal("0.0"):
            return {"stat": "Not_Ok", "emsg": "Quantity must be positive"}
        if action not in ("BUY", "SELL"):
            return {"stat": "Not_Ok", "emsg": f"Invalid action {action}"}

        if self.paper_trading or not self.creds.account_id:
            return self._execute_paper_order(symbol, quantity, price, action, currency, clordid)
        return self._send_live_order(symbol, quantity, price, action, currency, order_type, clordid)

    def _execute_paper_order(
        self, symbol: str, quantity: Decimal, price: Decimal, action: str, currency: Currency, clordid: str
    ) -> dict[str, Any]:
        """Simulates multi-currency portfolio fill with FX isolation."""
        if price <= Decimal("0.0"):
            return {"stat": "Not_Ok", "emsg": "Execution price must be positive for paper fill"}
        trade_value = quantity * price
        cur_bal = self.balances.get(currency, Decimal("0.0"))
        if action == "BUY" and trade_value > cur_bal:
            return {"stat": "Not_Ok", "emsg": f"Insufficient {currency.value} balance: {cur_bal} < {trade_value}"}

        order_id = f"IB_SIM_{uuid.uuid4().hex[:8]}"
        if action == "BUY":
            self.balances[currency] = cur_bal - trade_value
            pos = self.positions.get(symbol, {"quantity": Decimal("0.0"), "avg_price": Decimal("0.0")})
            new_qty = pos["quantity"] + quantity
            new_avg = ((pos["quantity"] * pos["avg_price"]) + trade_value) / new_qty
            self.positions[symbol] = {"quantity": new_qty, "avg_price": round(new_avg, 4), "currency": currency.value}
        else:
            pos = self.positions.get(symbol, {"quantity": Decimal("0.0"), "avg_price": Decimal("0.0")})
            if pos["quantity"] < quantity:
                return {"stat": "Not_Ok", "emsg": f"Cannot sell {quantity}, holding {pos['quantity']}"}
            self.balances[currency] = cur_bal + trade_value
            new_qty = pos["quantity"] - quantity
            if new_qty <= Decimal("0.0"):
                self.positions.pop(symbol, None)
            else:
                self.positions[symbol]["quantity"] = new_qty

        record = {
            "stat": "Ok", "order_id": order_id, "client_order_id": clordid,
            "symbol": symbol, "qty": float(quantity), "price": float(price),
            "action": action, "status": "FILLED", "currency": currency.value,
            "balance_after": float(self.balances[currency])
        }
        self.orders[order_id] = record
        return record

    def _resolve_conid(self, symbol: str) -> int:
        """Resolves exchange contract ID (conid) for ticker symbol."""
        if symbol.isdigit():
            return int(symbol)
        if not hasattr(self, "_conid_cache"):
            self._conid_cache: dict[str, int] = {}
        if symbol in self._conid_cache:
            return self._conid_cache[symbol]
        try:
            url = f"{self.creds.base_url}/iserver/secdef/search"
            resp = requests.post(
                url, json={"symbol": symbol, "name": False, "secType": "STK"},
                verify=self.creds.ssl_verify, timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and data and "conid" in data[0]:
                    cid = int(data[0]["conid"])
                    self._conid_cache[symbol] = cid
                    return cid
        except Exception as e:
            logger.warning(f"Could not resolve conid for {symbol}: {e}")
        return 0

    def _send_live_order(
        self, symbol: str, quantity: Decimal, price: Decimal, action: str,
        currency: Currency, order_type: str, clordid: str
    ) -> dict[str, Any]:
        """Routes real order to IBKR Client Portal API with conid resolution and confirmation."""
        conid = self._resolve_conid(symbol)
        if conid == 0 and not symbol.isdigit():
            return {"stat": "Not_Ok", "emsg": f"Unable to resolve IBKR conid for symbol {symbol}"}

        payload = {
            "orders": [{
                "acctId": self.creds.account_id,
                "conid": conid,
                "orderType": order_type,
                "price": float(price) if order_type != "MKT" else 0.0,
                "side": action,
                "quantity": float(quantity),
                "tif": "DAY",
                "cOID": clordid
            }]
        }
        try:
            url = f"{self.creds.base_url}/iserver/account/{self.creds.account_id}/orders"
            resp = requests.post(url, json=payload, verify=self.creds.ssl_verify, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                # Check for interactive confirmation prompt
                if isinstance(data, list) and data and "id" in data[0] and "message" in data[0]:
                    reply_id = data[0]["id"]
                    logger.info(f"Auto-replying confirmation to IBKR prompt {reply_id}: {data[0]['message']}")
                    reply_url = f"{self.creds.base_url}/iserver/reply/{reply_id}"
                    reply_resp = requests.post(
                        reply_url, json={"confirmed": True}, verify=self.creds.ssl_verify, timeout=10
                    )
                    if reply_resp.status_code == 200:
                        data = reply_resp.json()
                order_id = str(data[0].get("order_id", "")) if isinstance(data, list) and data else clordid
                res = {"stat": "Ok", "order_id": order_id, "details": data}
                self.orders[order_id] = res
                return res
            return {"stat": "Not_Ok", "emsg": resp.text}
        except Exception as e:
            logger.error(f"IBKR order dispatch failed: {e}")
            return {"stat": "Not_Ok", "emsg": str(e)}

    def cancel_order(self, order_id: str) -> bool:
        """Cancels an order via IBKR API."""
        if self.paper_trading or not self.creds.account_id:
            if order_id in self.orders:
                self.orders[order_id]["status"] = "CANCELED"
            return True
        try:
            url = f"{self.creds.base_url}/iserver/account/{self.creds.account_id}/order/{order_id}"
            resp = requests.delete(url, verify=self.creds.ssl_verify, timeout=10)
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"IBKR cancel order failed: {e}")
            return False

    def get_positions(self) -> list[dict[str, Any]]:
        """Returns list of open positions across currencies."""
        if self.paper_trading or not self.creds.account_id:
            return [
                {
                    "symbol": sym,
                    "quantity": float(pos["quantity"]),
                    "avg_price": float(pos["avg_price"]),
                    "currency": pos.get("currency", "USD")
                }
                for sym, pos in self.positions.items()
            ]
        try:
            url = f"{self.creds.base_url}/portfolio/{self.creds.account_id}/positions/0"
            resp = requests.get(url, verify=self.creds.ssl_verify, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return [
                    {
                        "symbol": item.get("contractDesc", str(item.get("conid"))),
                        "quantity": float(item.get("position", 0)),
                        "avg_price": float(item.get("mktPrice", 0.0)),
                        "currency": item.get("currency", "USD")
                    }
                    for item in data
                ]
        except Exception as e:
            logger.error(f"IBKR positions query failed: {e}")
        return []

    def get_cash_balance(self, currency: Currency) -> Decimal:
        """Returns settled available cash balance for specific currency."""
        if self.paper_trading or not self.creds.account_id:
            return self.balances.get(currency, Decimal("0.0"))
        try:
            url = f"{self.creds.base_url}/portfolio/{self.creds.account_id}/ledger"
            resp = requests.get(url, verify=self.creds.ssl_verify, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                curr_data = data.get(currency.value, {})
                cash = curr_data.get("cashbalance", "0.0")
                return Decimal(str(cash))
        except Exception as e:
            logger.error(f"IBKR get_cash_balance error: {e}")
        return self.balances.get(currency, Decimal("0.0"))

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        """Queries status of an order."""
        if self.paper_trading or not self.creds.account_id:
            order = self.orders.get(order_id)
            if order:
                return {"stat": "Ok", "order_id": order_id, "status": order.get("status", "FILLED")}
            return {"stat": "Ok", "order_id": order_id, "status": "FILLED" if "SIM" in order_id else "UNKNOWN"}
        try:
            url = f"{self.creds.base_url}/iserver/account/order/status/{order_id}"
            resp = requests.get(url, verify=self.creds.ssl_verify, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return {"stat": "Ok", "order_id": order_id, "status": data.get("order_status", ""), "details": data}
        except Exception as e:
            logger.error(f"IBKR get_order_status error: {e}")
        return {"stat": "Not_Ok", "order_id": order_id, "emsg": "Order not found"}
