"""
Shoonya (Finvasia) High-Performance Trading Gateway.
Implements official Noren REST API protocol with dynamic TOTP 2FA.
Supports both zero-risk Paper Trading simulation and live Exchange execution.
"""
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import os
from typing import Any, Optional
from dotenv import load_dotenv
from loguru import logger
import math
import numpy as np
import pyotp
import requests
import sqlite3
import time

# Load environment configuration
load_dotenv()

@dataclass
class ShoonyaCredentials:
    """Secure credentials container for Shoonya Finvasia (Supports both OAuth 2.0 & REST WSTP)."""
    user_id: str = ""
    password: str = ""
    totp_key: str = ""
    vendor_code: str = ""
    api_key: str = ""
    client_id: str = ""
    secret_code: str = ""
    auth_code: str = ""
    access_token: str = ""

    @classmethod
    def from_env(cls) -> "ShoonyaCredentials":
        return cls(
            user_id=os.getenv("SHOONYA_USER_ID", "").strip(),
            password=os.getenv("SHOONYA_PASSWORD", "").strip(),
            totp_key=os.getenv("SHOONYA_TOTP_KEY", "").strip(),
            vendor_code=os.getenv("SHOONYA_VENDOR_CODE", "").strip(),
            api_key=os.getenv("SHOONYA_API_KEY", "").strip(),
            client_id=os.getenv("SHOONYA_CLIENT_ID", os.getenv("SHOONYA_API_KEY", "")).strip(),
            secret_code=os.getenv("SHOONYA_SECRET_CODE", "").strip(),
            auth_code=os.getenv("SHOONYA_AUTH_CODE", "").strip(),
            access_token=os.getenv("SHOONYA_ACCESS_TOKEN", "").strip()
        )

    def is_valid(self) -> bool:
        if self.access_token:
            return True
        if self.client_id and self.secret_code:
            return True
        return bool(self.user_id and self.password and self.totp_key and self.vendor_code and self.api_key)

class ShoonyaGateway:
    """
    Shoonya execution gateway with dual-engine capability:
    - Default: Paper Trading simulation (0 risk, exact statutory friction, local audit journal)
    - Live Mode: Direct Noren REST API routing with automatic session management
    """
    BASE_URL = "https://api.shoonya.com/NorenWClientAPI"

    def __init__(
        self,
        credentials: Optional[ShoonyaCredentials] = None,
        paper_trading: bool = True,
        initial_balance: float = 10000.0,
        audit_file: Optional[str] = None
    ):
        self.creds = credentials or ShoonyaCredentials.from_env()
        self.paper_trading = paper_trading
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.susertoken: Optional[str] = None
        self.actid: Optional[str] = self.creds.user_id
        self.paper_positions: dict[str, dict[str, Any]] = {}
        self.clordid_ledger: dict[str, dict[str, Any]] = {}
        self.paper_order_counter = 1000
        self.api: Optional[Any] = None

        if not self.paper_trading:
            try:
                from NorenRestApiPy.NorenApi import NorenApi
                self.api = NorenApi(
                    host="https://api.shoonya.com/NorenWClientAPI/",
                    websocket="wss://api.shoonya.com/NorenWSAPI/"
                )
            except ImportError:
                self.api = None

        # Audit journal configuration
        if audit_file is None:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "audit"))
            os.makedirs(base_dir, exist_ok=True)
            self.audit_file = os.path.join(base_dir, "shoonya_paper_orders.json")
        else:
            self.audit_file = audit_file

    def get_status_summary(self) -> dict[str, Any]:
        """Returns safe operational diagnostics without exposing secrets."""
        masked_uid = f"{self.creds.user_id[:2]}***" if len(self.creds.user_id) >= 2 else "NOT_SET"
        return {
            "mode": "PAPER_TRADING" if self.paper_trading else "LIVE_EXCHANGE",
            "user_id": masked_uid,
            "credentials_configured": self.creds.is_valid(),
            "authenticated": self.susertoken is not None if not self.paper_trading else True,
            "current_balance": round(self.balance, 2),
            "active_positions_count": len(self.paper_positions) if self.paper_trading else "LIVE"
        }

    def generate_totp(self) -> str:
        """Generates real-time 6-digit TOTP token using pyotp."""
        if not self.creds.totp_key:
            raise ValueError("Shoonya TOTP key is not configured.")
        clean_key = self.creds.totp_key.replace(" ", "").upper()
        return pyotp.TOTP(clean_key).now()

    def get_oauth_url(self) -> str:
        """Returns the official Finvasia OAuth Authorize URL for this client."""
        client = self.creds.client_id or self.creds.vendor_code or self.creds.user_id
        return f"https://api.shoonya.com/OAuthlogin/authorize/oauth?client_id={client}"

    def exchange_auth_code(self, auth_code: str) -> dict[str, Any]:
        """Exchanges OAuth authorization code for session access token."""
        try:
            from NorenRestApiPy.NorenApi import NorenApi
            if not self.api:
                self.api = NorenApi(
                    host="https://api.shoonya.com/NorenWClientAPI/",
                    websocket="wss://api.shoonya.com/NorenWSAPI/"
                )
            res = self.api.getAccessToken(
                auth_code,
                self.creds.secret_code,
                self.creds.client_id,
                self.creds.user_id
            )
            if res and len(res) >= 4:
                asc_tok, usrid, ref_tok, actid = res
                self.susertoken = asc_tok
                self.actid = actid or self.creds.user_id
                self.api.injectOAuthHeader(asc_tok, usrid, self.actid)
                from dotenv import set_key
                env_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
                if os.path.exists(env_file):
                    set_key(env_file, "SHOONYA_ACCESS_TOKEN", asc_tok)
                    set_key(env_file, "SHOONYA_AUTH_CODE", auth_code)
                logger.success(f"ShoonyaGateway: OAuth exchange succeeded for user {usrid}")
                return {"stat": "Ok", "access_token": asc_tok, "actid": self.actid}

            # Direct fallback inspection if getAccessToken returned None
            raw_hash = (self.creds.client_id + self.creds.secret_code + auth_code).encode("utf-8")
            ck = hashlib.sha256(raw_hash).hexdigest()
            vals = {"code": auth_code, "checksum": ck, "uid": self.creds.user_id}
            resp = requests.post(f"{self.BASE_URL}/GenAcsTok", data="jData=" + json.dumps(vals), timeout=10)
            data = resp.json() if resp.status_code == 200 else {}
            err_msg = data.get("emsg", "Failed to exchange auth code for access token")
            return {"stat": "Not_Ok", "emsg": err_msg}
        except Exception as e:
            logger.error(f"ShoonyaGateway: OAuth exchange exception: {e}")
            return {"stat": "Not_Ok", "emsg": str(e)}

    def authenticate(self) -> bool:
        """Authenticates with Finvasia Noren server and stores session token."""
        if self.paper_trading:
            logger.info("ShoonyaGateway: Running in Paper Trading Mode. Skipping remote authentication.")
            return True

        if not self.creds.is_valid():
            logger.error("ShoonyaGateway: Incomplete credentials in .env. Cannot authenticate.")
            return False

        # 1. Active access token from environment
        if self.creds.access_token:
            self.susertoken = self.creds.access_token
            if self.api:
                self.api.injectOAuthHeader(self.susertoken, self.creds.user_id, self.actid or self.creds.user_id)
            logger.success("ShoonyaGateway: Using active OAuth access token from environment.")
            return True

        # 2. OAuth 2.0 Token Exchange if client_id + secret_code + auth_code are present
        if self.creds.client_id and self.creds.secret_code and self.creds.auth_code:
            ret = self.exchange_auth_code(self.creds.auth_code)
            return ret.get("stat") == "Ok"

        logger.warning("ShoonyaGateway: No active access token. Run scripts/connect_shoonya.py to authorize.")
        return False

    @staticmethod
    def format_trading_symbol(symbol: str) -> str:
        """Converts raw symbol to NSE trading symbol (e.g. NIFTYBEES -> NIFTYBEES-EQ). Preserves futures and series."""
        clean = symbol.replace(".NS", "").strip().upper()
        if clean.endswith("FUT") or clean.endswith("-EQ") or clean.endswith("-BE") or clean.endswith("-BZ"):
            return clean
        return f"{clean}-EQ"

    @staticmethod
    def calculate_friction(trade_value: float, is_etf: bool, custom_bps: Optional[float] = None) -> float:
        """
        Computes statutory friction:
        - Default: 1.2 bps for ETFs, 25 bps for equities
        - Or dynamic custom_bps if specified
        """
        if custom_bps is not None:
            rate = custom_bps / 10000.0
        else:
            rate = 0.00012 if is_etf else 0.00250
        return float(trade_value * rate)

    def place_order(
        self,
        symbol: str,
        quantity: int,
        price: float = 0.0,
        action: str = "BUY",
        product: str = "C",
        order_type: str = "MKT",
        is_etf: bool = True,
        client_order_id: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Places order in either Paper Trading or Live Exchange mode.
        Guarded against NaN, Inf, non-integer quantity, and negative prices.
        Supports client_order_id idempotency to prevent duplicate broker executions.
        """
        if client_order_id and client_order_id in self.clordid_ledger:
            logger.info(f"ShoonyaGateway: Idempotent replay for client_order_id={client_order_id}")
            return self.clordid_ledger[client_order_id]

        if not isinstance(quantity, (int, np.integer)) or quantity <= 0:
            return {"stat": "Not_Ok", "emsg": "Quantity must be a positive integer"}

        if order_type.upper() == "MKT":
            if not np.isfinite(price) or price < 0.0:
                return {"stat": "Not_Ok", "emsg": f"Price must be a non-negative finite number for MKT orders, got {price}"}
        else:
            if not np.isfinite(price) or price <= 0.0:
                return {"stat": "Not_Ok", "emsg": f"Price must be a positive finite number, got {price}"}

        clean_action = action.strip().upper()
        if clean_action not in ("BUY", "SELL"):
            return {"stat": "Not_Ok", "emsg": f"Invalid action '{action}'. Must be 'BUY' or 'SELL'"}

        trading_symbol = self.format_trading_symbol(symbol)
        if self.paper_trading:
            res = self._execute_paper_order(trading_symbol, int(quantity), float(price), clean_action, is_etf)
        else:
            res = self._send_live_order(trading_symbol, int(quantity), float(price), clean_action, product, order_type)

        if client_order_id and res.get("stat") == "Ok":
            self.clordid_ledger[client_order_id] = res
        return res

    def _execute_paper_order(
        self,
        tsym: str,
        quantity: int,
        price: float,
        action: str,
        is_etf: bool
    ) -> dict[str, Any]:
        """Internal paper order simulation with statutory friction tracking."""
        trade_value = price * quantity
        friction = self.calculate_friction(trade_value, is_etf=is_etf)
        total_cost = trade_value + friction if action == "BUY" else friction

        if action == "BUY" and total_cost > self.balance:
            return {
                "stat": "Not_Ok",
                "emsg": f"Insufficient virtual cash ₹{self.balance:.2f} for required ₹{total_cost:.2f}"
            }

        self.paper_order_counter += 1
        order_no = f"SIM_{datetime.now().strftime('%Y%m%d')}_{self.paper_order_counter}"

        # Update simulated portfolio
        if action == "BUY":
            self.balance -= total_cost
            pos = self.paper_positions.get(tsym, {"quantity": 0, "avg_price": 0.0})
            new_qty = pos["quantity"] + quantity
            new_avg = ((pos["quantity"] * pos["avg_price"]) + trade_value) / new_qty
            self.paper_positions[tsym] = {"quantity": new_qty, "avg_price": round(new_avg, 2)}
        else: # SELL
            pos = self.paper_positions.get(tsym, {"quantity": 0, "avg_price": 0.0})
            if pos["quantity"] < quantity:
                return {"stat": "Not_Ok", "emsg": f"Cannot sell {quantity} units, held: {pos['quantity']}"}
            proceeds = trade_value - friction
            self.balance += proceeds
            new_qty = pos["quantity"] - quantity
            if new_qty == 0:
                self.paper_positions.pop(tsym, None)
            else:
                self.paper_positions[tsym]["quantity"] = new_qty

        order_record = {
            "stat": "Ok",
            "norenordno": order_no,
            "tsym": tsym,
            "action": action,
            "qty": quantity,
            "price": price,
            "friction": round(friction, 2),
            "balance_after": round(self.balance, 2),
            "timestamp": datetime.now().isoformat()
        }
        self._record_audit(order_record)
        return order_record

    def _send_live_order(
        self,
        tsym: str,
        quantity: int,
        price: float,
        action: str,
        product: str,
        order_type: str
    ) -> dict[str, Any]:
        """Routes real order to Shoonya Noren API server."""
        if not self.susertoken and not self.authenticate():
            return {"stat": "Not_Ok", "emsg": "Session expired or authentication failed"}

        try:
            if self.api:
                ret = self.api.place_order(
                    buy_or_sell="B" if action.upper() == "BUY" else "S",
                    product_type=product,
                    exchange="NSE",
                    tradingsymbol=tsym,
                    quantity=quantity,
                    discloseqty=0,
                    price_type=order_type,
                    price=price if order_type != "MKT" else 0.0,
                    trigger_price=None,
                    retention="DAY",
                    remarks="auto_trade"
                )
                if ret is None or (isinstance(ret, dict) and ret.get("stat") != "Ok"):
                    emsg = ret.get("emsg", "Order rejected by broker (None response)") if isinstance(ret, dict) else "Order rejected by broker (None response)"
                    return {"stat": "Not_Ok", "emsg": emsg}
                return ret if isinstance(ret, dict) else {"stat": "Ok", "result": ret}

            # Direct REST fallback
            j_data = {
                "uid": self.creds.user_id,
                "actid": self.actid or self.creds.user_id,
                "exch": "NSE",
                "tsym": tsym,
                "qty": str(quantity),
                "prc": str(price) if order_type != "MKT" else "0",
                "prd": product,
                "trantype": "B" if action.upper() == "BUY" else "S",
                "prctyp": order_type,
                "ret": "DAY"
            }
            headers = {"Authorization": f"Bearer {self.susertoken}", "Content-Type": "application/json"}
            resp = requests.post(
                f"{self.BASE_URL}/PlaceOrder",
                data="jData=" + json.dumps(j_data),
                headers=headers,
                timeout=10
            )
            return resp.json()
        except Exception as e:
            logger.error(f"Shoonya live order failed: {e}")
            return {"stat": "Not_Ok", "emsg": str(e)}

    def get_limits(self) -> dict[str, Any]:
        """Retrieves account balance and available margin."""
        if self.paper_trading:
            return {"stat": "Ok", "cash": self.balance, "marginused": 0.0}

        if not self.susertoken and not self.authenticate():
            return {"stat": "Not_Ok", "emsg": "Authentication required"}

        try:
            if self.api:
                res = self.api.get_limits()
                if isinstance(res, dict):
                    return res
            # Direct REST fallback
            j_data = {"uid": self.creds.user_id, "actid": self.actid or self.creds.user_id}
            headers = {"Authorization": f"Bearer {self.susertoken}", "Content-Type": "application/json"}
            resp = requests.post(f"{self.BASE_URL}/Limits", data="jData=" + json.dumps(j_data), headers=headers, timeout=10)
            return resp.json()
        except Exception as e:
            return {"stat": "Not_Ok", "emsg": str(e)}

    def cancel_all_orders(self) -> dict[str, Any]:
        """Emergency order cancellation for all pending orders."""
        if self.paper_trading:
            return {"stat": "Ok", "cancelled_orders": 0, "message": "All pending paper orders cancelled."}
        if self.api and self.susertoken:
            try:
                orders = self.api.get_order_book()
                cancelled = 0
                if isinstance(orders, list):
                    for o in orders:
                        if o.get("status") in ["OPEN", "TRIGGER_PENDING"]:
                            orderno = o.get("norenordno") or o.get("orderno")
                            self.api.cancel_order(orderno=orderno)
                            cancelled += 1
                return {"stat": "Ok", "cancelled_orders": cancelled}
            except Exception as e:
                return {"stat": "Not_Ok", "emsg": str(e)}
        return {"stat": "Ok", "cancelled_orders": 0, "message": "Gateway offline or paper mode."}

    def liquidate_positions(self) -> dict[str, Any]:
        """Emergency panic liquidation of all active positions to 100% cash."""
        if self.paper_trading:
            liquidated = []
            symbols = list(self.paper_positions.keys())
            for sym in symbols:
                pos = self.paper_positions.get(sym)
                if pos:
                    qty = int(pos.get("quantity", 0))
                    price = float(pos.get("avg_price", 100.0))
                    if qty > 0:
                        res = self.place_order(symbol=sym, quantity=qty, price=price, action="SELL", is_etf="BEES" in sym)
                        liquidated.append({"symbol": sym, "quantity": qty, "result": res})
                    elif qty < 0:
                        res = self.place_order(symbol=sym, quantity=abs(qty), price=price, action="BUY", is_etf="BEES" in sym)
                        liquidated.append({"symbol": sym, "quantity": abs(qty), "result": res})
            # Clear paper positions to prevent residual exposure
            self.paper_positions.clear()
            return {"stat": "Ok", "liquidated_count": len(liquidated), "liquidated": liquidated, "balance": self.balance}
        if self.api and self.susertoken:
            try:
                positions = self.api.get_positions()
                liquidated = []
                if isinstance(positions, list):
                    for p in positions:
                        netqty = int(p.get("netqty", 0))
                        if netqty != 0:
                            tsym = p.get("tsym", "")
                            prd = p.get("prd", "C")
                            action = "SELL" if netqty > 0 else "BUY"
                            is_etf = "BEES" in tsym
                            res = self.place_order(
                                symbol=tsym,
                                quantity=abs(netqty),
                                price=0.0,
                                action=action,
                                product=prd,
                                order_type="MKT",
                                is_etf=is_etf
                            )
                            liquidated.append({"symbol": tsym, "quantity": abs(netqty), "result": res})
                return {"stat": "Ok", "liquidated_count": len(liquidated), "liquidated": liquidated}
            except Exception as e:
                return {"stat": "Not_Ok", "emsg": str(e)}
        return {"stat": "Ok", "liquidated_count": 0, "message": "Gateway offline or paper mode."}

    def _record_audit(self, record: dict[str, Any]) -> None:
        """Appends order execution to persistent audit log (SQLite WAL or append-only JSONL)."""
        try:
            if self.audit_file.endswith(".db"):
                os.makedirs(os.path.dirname(self.audit_file) or ".", exist_ok=True)
                with sqlite3.connect(self.audit_file, timeout=5.0) as conn:
                    conn.execute("PRAGMA journal_mode = WAL;")
                    conn.execute("PRAGMA synchronous = NORMAL;")
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS order_audit_journal (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp REAL,
                            symbol TEXT,
                            action TEXT,
                            quantity INTEGER,
                            price REAL,
                            order_type TEXT,
                            product TEXT,
                            status TEXT,
                            raw_payload TEXT
                        );
                    """)
                    conn.execute("""
                        INSERT INTO order_audit_journal (
                            timestamp, symbol, action, quantity, price, order_type, product, status, raw_payload
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """, (
                        record.get("timestamp", time.time()),
                        record.get("symbol", ""),
                        record.get("action", ""),
                        int(record.get("quantity", 0)),
                        float(record.get("price", 0.0)),
                        record.get("order_type", ""),
                        record.get("product", ""),
                        record.get("status", record.get("stat", "Ok")),
                        json.dumps(record)
                    ))
            else:
                os.makedirs(os.path.dirname(self.audit_file) or ".", exist_ok=True)
                line = json.dumps(record) + "\n"
                with open(self.audit_file, "a", encoding="utf-8") as f:
                    f.write(line)
        except Exception as err:
            logger.warning(f"Failed to record audit journal: {err}")
