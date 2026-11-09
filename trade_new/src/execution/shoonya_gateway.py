"""
Shoonya (Finvasia) High-Performance Trading Gateway.
Implements official Noren REST API protocol with dynamic TOTP 2FA.
Supports both zero-risk Paper Trading simulation and live Exchange execution.
"""
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from decimal import Decimal
import hashlib
import json
import os
from typing import Any, Optional
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from loguru import logger
import math
import numpy as np
import pyotp
import re
import requests
import sqlite3
import time

from src.domain.models import Currency
from src.execution.base_gateway import AbstractBrokerGateway
from .rate_limiter import TokenBucketRateLimiter
from .clordid_ledger import ClientOrderIdLedger

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

class ShoonyaGateway(AbstractBrokerGateway):
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

        base_audit_dir = os.path.dirname(os.path.abspath(self.audit_file))
        ledger_path = os.path.join(base_audit_dir, "clordid_ledger.db")
        self.clordid_ledger = ClientOrderIdLedger(ledger_path)
        self.rate_limiter = TokenBucketRateLimiter(rate=20.0, capacity=20.0)
        self.last_auth_time = 0.0
        self.last_auth_date: Optional[str] = None

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

    def quick_auth(self) -> dict[str, Any]:
        """Executes zero-touch headless authentication using Noren REST POST /QuickAuth."""
        if not (self.creds.user_id and self.creds.password and self.creds.totp_key and self.creds.vendor_code and self.creds.api_key):
            return {"stat": "Not_Ok", "emsg": "Incomplete QuickAuth credentials"}
        self.rate_limiter.acquire()
        pwd_hash = hashlib.sha256(self.creds.password.encode("utf-8")).hexdigest()
        totp_factor2 = self.generate_totp()
        app_key_raw = f"{self.creds.user_id}|{self.creds.api_key}".encode("utf-8")
        app_key_hash = hashlib.sha256(app_key_raw).hexdigest()
        imei = os.getenv("SHOONYA_IMEI", "PC_AUTO_TRADE_01").strip()

        payload = {
            "uid": self.creds.user_id,
            "pwd": pwd_hash,
            "factor2": totp_factor2,
            "vc": self.creds.vendor_code,
            "app_key": app_key_hash,
            "imei": imei,
        }
        try:
            resp = requests.post(
                f"{self.BASE_URL}/QuickAuth",
                data="jData=" + json.dumps(payload),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=10,
            )
            res = resp.json() if resp.status_code == 200 else {"stat": "Not_Ok", "emsg": f"HTTP {resp.status_code}"}
            if isinstance(res, dict) and res.get("stat") == "Ok":
                self.susertoken = res.get("susertoken")
                self.actid = res.get("actid", self.creds.user_id)
                self.last_auth_time = time.time()
                self.last_auth_date = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d")
                if self.api and hasattr(self.api, "set_session"):
                    self.api.set_session(userid=self.creds.user_id, password=pwd_hash, usertoken=self.susertoken)
                logger.success(f"ShoonyaGateway: QuickAuth successful for user {self.creds.user_id}")
            return res
        except Exception as exc:
            logger.error(f"ShoonyaGateway: QuickAuth error: {exc}")
            return {"stat": "Not_Ok", "emsg": str(exc)}

    def should_refresh_daily_auth(self) -> bool:
        """Determines whether 04:00 AM IST daily re-authentication boundary has been crossed."""
        if self.paper_trading:
            return False
        if not self.susertoken:
            return True
        now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
        today_str = now_ist.strftime("%Y-%m-%d")
        if self.last_auth_date and self.last_auth_date != today_str and now_ist.time() >= dtime(4, 0):
            return True
        return False

    def check_and_refresh_auth(self, force: bool = False) -> bool:
        """Re-authenticates at 04:00 AM IST or upon expiry."""
        if force or self.should_refresh_daily_auth():
            logger.info("ShoonyaGateway: Refreshing session credentials...")
            self.susertoken = None
            return self.authenticate()
        return True

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
            if self.api and hasattr(self.api, "injectOAuthHeader"):
                self.api.injectOAuthHeader(self.susertoken, self.creds.user_id, self.actid or self.creds.user_id)
            self.last_auth_time = time.time()
            self.last_auth_date = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d")
            logger.success("ShoonyaGateway: Using active OAuth access token from environment.")
            return True

        # 2. Headless Noren REST POST /QuickAuth
        if bool(self.creds.user_id and self.creds.password and self.creds.totp_key and self.creds.vendor_code and self.creds.api_key):
            res = self.quick_auth()
            return res.get("stat") == "Ok"

        # 3. OAuth 2.0 Token Exchange if client_id + secret_code + auth_code are present
        if self.creds.client_id and self.creds.secret_code and self.creds.auth_code:
            ret = self.exchange_auth_code(self.creds.auth_code)
            return ret.get("stat") == "Ok"

        logger.warning("ShoonyaGateway: No active access token or QuickAuth credentials.")
        return False

    def _is_session_expired(self, res: Any) -> bool:
        """Detects whether Shoonya response indicates an expired or invalid session token."""
        if not isinstance(res, dict):
            return False
        msg = str(res.get("emsg", "")).lower()
        stat = str(res.get("stat", "")).lower()
        if stat == "not_ok" and "session expired" in msg:
            return True
        return any(k in msg for k in ("session expired", "invalid token", "token expired", "session not found", "session error"))

    def handle_session_expiry(self) -> bool:
        """Invalidates dead session token and executes zero-touch re-authentication."""
        logger.warning("ShoonyaGateway: Broker session token expired. Clearing token and refreshing auth...")
        self.susertoken = None
        if self.creds.access_token:
            # Token from env expired, cannot magically refresh without fresh auth_code or credentials
            pass
        return self.authenticate()

    @staticmethod
    def format_trading_symbol(symbol: str) -> str:
        """Converts raw symbol to NSE trading symbol (e.g. NIFTYBEES -> NIFTYBEES-EQ). Preserves futures, options, and series."""
        clean = symbol.replace(".NS", "").strip().upper()
        is_option = bool(re.search(r"\d+(CE|PE)$", clean))
        if (
            clean.endswith("FUT")
            or is_option
            or clean.endswith(("-EQ", "-BE", "-BZ", "-SM"))
        ):
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

    def _validate_order_params(
        self, quantity: int, price: float, order_type: str, action: str
    ) -> Optional[dict[str, Any]]:
        """Validates numerical and categorical order constraints."""
        if not isinstance(quantity, (int, np.integer)) or quantity <= 0:
            return {"stat": "Not_Ok", "emsg": "Quantity must be a positive integer"}
        if order_type.upper() == "MKT":
            if not np.isfinite(price) or price < 0.0:
                return {"stat": "Not_Ok", "emsg": f"Price must be non-negative finite for MKT, got {price}"}
        else:
            if not np.isfinite(price) or price <= 0.0:
                return {"stat": "Not_Ok", "emsg": f"Price must be positive finite, got {price}"}
        clean_action = action.strip().upper()
        if clean_action not in ("BUY", "SELL"):
            return {"stat": "Not_Ok", "emsg": f"Invalid action '{action}'. Must be 'BUY' or 'SELL'"}
        return None

    def place_order(
        self,
        order: Optional[dict[str, Any] | str] = None,
        symbol: str = "",
        quantity: int = 0,
        price: float = 0.0,
        action: str = "BUY",
        product: str = "C",
        order_type: str = "MKT",
        is_etf: bool = True,
        client_order_id: Optional[str] = None,
        custom_bps: Optional[float] = None,
        **kwargs: Any
    ) -> dict[str, Any]:
        """Places order in either Paper Trading or Live Exchange mode."""
        if isinstance(order, dict):
            symbol = str(order.get("symbol", symbol))
            quantity = int(order.get("quantity", quantity))
            price = float(order.get("price", price))
            action = str(order.get("action", action))
            product = str(order.get("product", product))
            order_type = str(order.get("order_type", order_type))
            is_etf = bool(order.get("is_etf", is_etf))
            client_order_id = order.get("client_order_id", client_order_id)
            custom_bps = order.get("custom_bps", custom_bps)
        elif isinstance(order, str):
            symbol = order

        if client_order_id and client_order_id in self.clordid_ledger:
            logger.info(f"ShoonyaGateway: Idempotent replay for client_order_id={client_order_id}")
            return self.clordid_ledger[client_order_id]

        err = self._validate_order_params(quantity, price, order_type, action)
        if err:
            return err

        clean_action = action.strip().upper()
        tsym = self.format_trading_symbol(symbol)
        if self.paper_trading:
            res = self._execute_paper_order(tsym, int(quantity), float(price), clean_action, is_etf, custom_bps=custom_bps)
        else:
            res = self._send_live_order(tsym, int(quantity), float(price), clean_action, product, order_type)

        if client_order_id and res.get("stat") == "Ok":
            self.clordid_ledger[client_order_id] = res
        return res

    def _update_paper_position(
        self, tsym: str, quantity: int, price: float, action: str, trade_value: float, friction: float
    ) -> None:
        """Mutates paper portfolio positions and cash ledger."""
        if action == "BUY":
            self.balance -= (trade_value + friction)
            pos = self.paper_positions.get(tsym, {"quantity": 0, "avg_price": 0.0})
            new_qty = pos["quantity"] + quantity
            new_avg = ((pos["quantity"] * pos["avg_price"]) + trade_value) / new_qty
            self.paper_positions[tsym] = {"quantity": new_qty, "avg_price": round(new_avg, 2)}
            return

        pos = self.paper_positions.get(tsym, {"quantity": 0, "avg_price": 0.0})
        self.balance += (trade_value - friction)
        new_qty = pos["quantity"] - quantity
        if new_qty <= 0:
            self.paper_positions.pop(tsym, None)
        else:
            self.paper_positions[tsym]["quantity"] = new_qty

    def _execute_paper_order(
        self,
        tsym: str,
        quantity: int,
        price: float,
        action: str,
        is_etf: bool,
        custom_bps: Optional[float] = None,
    ) -> dict[str, Any]:
        """Internal paper order simulation with statutory friction tracking."""
        trade_value = price * quantity
        friction = self.calculate_friction(trade_value, is_etf=is_etf, custom_bps=custom_bps)
        total_cost = trade_value + friction if action == "BUY" else friction

        if action == "BUY" and total_cost > self.balance:
            return {
                "stat": "Not_Ok",
                "emsg": f"Insufficient virtual cash ₹{self.balance:.2f} for required ₹{total_cost:.2f}"
            }

        pos = self.paper_positions.get(tsym, {"quantity": 0, "avg_price": 0.0})
        if action == "SELL" and pos["quantity"] < quantity:
            return {"stat": "Not_Ok", "emsg": f"Cannot sell {quantity} units, held: {pos['quantity']}"}

        self.paper_order_counter += 1
        order_no = f"SIM_{datetime.now().strftime('%Y%m%d')}_{self.paper_order_counter}"
        self._update_paper_position(tsym, quantity, price, action, trade_value, friction)

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

    def _send_live_order_api(
        self, tsym: str, quantity: int, price: float, action: str, product: str, order_type: str
    ) -> dict[str, Any]:
        """Dispatches live order through Noren API client wrapper."""
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
            if isinstance(ret, dict) and self._is_session_expired(ret):
                if self.handle_session_expiry():
                    return self._send_live_order_api(tsym, quantity, price, action, product, order_type)
            emsg = ret.get("emsg", "Order rejected by broker") if isinstance(ret, dict) else "Order rejected"
            return {"stat": "Not_Ok", "emsg": emsg}
        return ret if isinstance(ret, dict) else {"stat": "Ok", "result": ret}

    def _send_live_order_rest(
        self, tsym: str, quantity: int, price: float, action: str, product: str, order_type: str
    ) -> dict[str, Any]:
        """Dispatches live order through direct REST HTTP endpoints."""
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
        try:
            resp = requests.post(f"{self.BASE_URL}/PlaceOrder", data="jData=" + json.dumps(j_data), headers=headers, timeout=10)
            if resp.status_code != 200:
                return {"stat": "Not_Ok", "emsg": f"HTTP {resp.status_code} proxy error"}
            res_dict = resp.json()
        except Exception as e:
            return {"stat": "Not_Ok", "emsg": f"Reverse proxy or network error: {e}"}

        if self._is_session_expired(res_dict):
            if self.handle_session_expiry():
                headers["Authorization"] = f"Bearer {self.susertoken}"
                try:
                    resp = requests.post(f"{self.BASE_URL}/PlaceOrder", data="jData=" + json.dumps(j_data), headers=headers, timeout=10)
                    if resp.status_code == 200:
                        return resp.json()
                except Exception:
                    pass
        return res_dict

    def poll_order_confirmation(
        self, orderno: str, max_wait_s: float = 10.0, poll_interval_s: float = 0.5
    ) -> dict[str, Any]:
        """Polls order history until order reaches confirmed COMPLETE or REJECTED state."""
        start = time.perf_counter()
        last_info: dict[str, Any] = {}
        while time.perf_counter() - start < max_wait_s:
            self.rate_limiter.acquire()
            hist = self._get_single_order_history(orderno)
            if hist:
                records = hist if isinstance(hist, list) else [hist]
                latest = records[-1] if records else {}
                last_info = latest
                status = str(latest.get("status", "")).upper()
                fill_qty = int(latest.get("fillshares", latest.get("qty", 0)) or 0)
                if status == "COMPLETE":
                    fill_price = float(latest.get("avgprc", latest.get("prc", 0.0)) or 0.0)
                    logger.success(f"ShoonyaGateway: Order {orderno} confirmed COMPLETE ({fill_qty} @ ₹{fill_price:.2f})")
                    return {
                        "stat": "Ok", "status": "COMPLETE", "norenordno": orderno,
                        "fill_price": fill_price, "fill_qty": fill_qty, "details": latest
                    }
                if status in ("PARTIALLY_FILLED", "OPEN") and fill_qty > 0:
                    fill_price = float(latest.get("avgprc", latest.get("prc", 0.0)) or 0.0)
                    logger.info(f"ShoonyaGateway: Order {orderno} PARTIALLY_FILLED ({fill_qty} shares @ ₹{fill_price:.2f})")
                    last_info["fill_qty"] = fill_qty
                    last_info["fill_price"] = fill_price
                if status in ("REJECTED", "CANCELLED"):
                    rej_msg = latest.get("rejreason", latest.get("emsg", f"Order {status}"))
                    logger.warning(f"ShoonyaGateway: Order {orderno} {status}: {rej_msg}")
                    return {
                        "stat": "Not_Ok", "status": status, "norenordno": orderno,
                        "emsg": rej_msg, "details": latest
                    }
            time.sleep(poll_interval_s)
        logger.warning(f"ShoonyaGateway: Order {orderno} poll timeout. Latest: {last_info.get('status', 'OPEN')}")
        return {
            "stat": "Ok", "status": last_info.get("status", "OPEN"),
            "norenordno": orderno, "pending": True,
            "fill_qty": last_info.get("fill_qty", 0),
            "fill_price": last_info.get("fill_price", 0.0),
            "details": last_info
        }

    def _get_single_order_history(self, orderno: str) -> Any:
        """Retrieves single order history via API or REST."""
        try:
            if self.api and hasattr(self.api, "single_order_history"):
                return self.api.single_order_history(orderno=orderno)
            j_data = {"uid": self.creds.user_id, "norenordno": orderno}
            headers = {"Authorization": f"Bearer {self.susertoken}", "Content-Type": "application/json"}
            resp = requests.post(f"{self.BASE_URL}/SingleOrderHist", data="jData=" + json.dumps(j_data), headers=headers, timeout=5)
            return resp.json() if resp.status_code == 200 else None
        except Exception as e:
            logger.debug(f"ShoonyaGateway: order history fetch error: {e}")
            return None

    def _send_live_order(
        self, tsym: str, quantity: int, price: float, action: str, product: str, order_type: str
    ) -> dict[str, Any]:
        """Routes real order to Shoonya Noren API server and polls confirmation."""
        if not self.susertoken and not self.authenticate():
            return {"stat": "Not_Ok", "emsg": "Session expired or authentication failed"}

        try:
            self.rate_limiter.acquire()
            if self.api:
                res = self._send_live_order_api(tsym, quantity, price, action, product, order_type)
            else:
                res = self._send_live_order_rest(tsym, quantity, price, action, product, order_type)

            if isinstance(res, dict) and res.get("stat") == "Ok":
                orderno = res.get("norenordno") or res.get("norenord")
                if orderno:
                    return self.poll_order_confirmation(orderno)
            return res
        except Exception as e:
            logger.error(f"Shoonya live order failed: {e}")
            return {"stat": "Not_Ok", "emsg": str(e)}

    def cancel_order(self, order_id: str) -> bool:
        """Cancels an active working order and verifies it did not execute concurrently."""
        if self.paper_trading:
            return True
        if not self.susertoken and not self.authenticate():
            return False
        try:
            cancelled = False
            if self.api:
                ret = self.api.cancel_order(orderno=order_id)
                cancelled = isinstance(ret, dict) and ret.get("stat") == "Ok"
            else:
                j_data = {"uid": self.creds.user_id, "norenordno": order_id}
                headers = {"Authorization": f"Bearer {self.susertoken}", "Content-Type": "application/json"}
                resp = requests.post(f"{self.BASE_URL}/CancelOrder", data="jData=" + json.dumps(j_data), headers=headers, timeout=10)
                cancelled = bool(resp.status_code == 200 and resp.json().get("stat") == "Ok")

            # Verify no matching engine fill race condition
            hist = self._get_single_order_history(order_id)
            if hist:
                latest = hist[-1] if isinstance(hist, list) else hist
                status = str(latest.get("status", "")).upper()
                if status == "COMPLETE":
                    logger.warning(f"ShoonyaGateway: Cancel on {order_id} collided with match fill! Order is COMPLETE.")
                    return False
            return cancelled
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def get_positions(self) -> list[dict[str, Any]]:
        """Returns standardized position list."""
        if self.paper_trading:
            return [
                {
                    "symbol": sym,
                    "quantity": pos.get("quantity", 0),
                    "avg_price": pos.get("avg_price", 0.0),
                    "currency": "INR"
                }
                for sym, pos in self.paper_positions.items()
            ]
        if self.api:
            ret = self.api.get_positions()
            if isinstance(ret, list):
                return ret
        return []

    def get_cash_balance(self, currency: Currency) -> Decimal:
        """Returns settled available cash balance in Currency (INR only for Shoonya)."""
        if currency != Currency.INR:
            return Decimal("0.0")
        if self.paper_trading:
            return Decimal(str(round(self.balance, 4)))
        lim = self.get_limits()
        cash_val = lim.get("cash", lim.get("margin", self.balance))
        return Decimal(str(round(float(cash_val), 4)))

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        """Queries status and fill metadata for a given order ID."""
        if self.paper_trading:
            if order_id in self.clordid_ledger:
                return {"stat": "Ok", "order_id": order_id, "status": "COMPLETE", "details": self.clordid_ledger[order_id]}
            return {"stat": "Ok", "order_id": order_id, "status": "COMPLETE" if order_id.startswith("SIM_") else "UNKNOWN"}
        try:
            if self.api:
                ret = self.api.single_order_history(orderno=order_id)
                return ret if isinstance(ret, dict) else {"stat": "Ok", "details": ret}
        except Exception as e:
            logger.error(f"Failed to fetch order status {order_id}: {e}")
        return {"stat": "Not_Ok", "order_id": order_id, "emsg": "Order not found"}

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
                        record.get("symbol", record.get("tsym", "")),
                        record.get("action", ""),
                        int(record.get("quantity", record.get("qty", 0))),
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
