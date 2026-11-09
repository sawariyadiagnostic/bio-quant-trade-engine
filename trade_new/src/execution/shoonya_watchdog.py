"""Headless QuickAuth Session Watchdog for Shoonya Broker Gateway.

Provides automated session keepalives, periodic health probes, and automated
TOTP generation via pyotp for zero-touch headless authentication during
pre-market readiness (08:45 IST) and active market sessions.
"""
from __future__ import annotations
from datetime import datetime, time as dtime
import time
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from loguru import logger
import pyotp

IST = ZoneInfo("Asia/Kolkata")


class ShoonyaSessionWatchdog:
    """Manages broker session token validity and automated TOTP re-authentication."""

    def __init__(
        self,
        gateway: Any,
        totp_key: str = "",
        totp_secret: str = "",
        interval_seconds: float = 60.0,
    ) -> None:
        self.gateway = gateway
        # Support both creds.totp_key and legacy totp_secret
        resolved_key = totp_key or totp_secret
        if not resolved_key and hasattr(gateway, "creds"):
            resolved_key = getattr(gateway.creds, "totp_key", "")
        self.totp_key = resolved_key.replace(" ", "").upper() if resolved_key else ""
        self.interval_seconds = interval_seconds
        self.last_ping_time = time.time()
        self.last_auth_time = (
            time.time() if getattr(gateway, "susertoken", None) else 0.0
        )
        self.last_daily_reauth_date: Optional[str] = None

    def generate_totp(self) -> str:
        """Generates the current 6-digit RFC 6238 TOTP token."""
        key = self.totp_key
        if not key and hasattr(self.gateway, "creds"):
            key = getattr(self.gateway.creds, "totp_key", "").replace(" ", "").upper()
        if not key:
            raise ValueError("TOTP key is empty or not configured.")
        totp = pyotp.TOTP(key)
        return str(totp.now())

    def check_session_alive(self) -> bool:
        """Probes the broker gateway session with a lightweight limits query."""
        if not getattr(self.gateway, "susertoken", None):
            return False
        try:
            res = self.gateway.get_limits()
            if isinstance(res, dict) and res.get("stat") == "Ok":
                self.last_ping_time = time.time()
                return True
            logger.warning(f"[Watchdog] Broker probe returned non-Ok: {res}")
            return False
        except Exception as e:
            logger.error(f"[Watchdog] Session probe exception: {e}")
            return False

    def refresh_session(self) -> bool:
        """Attempts an automated re-authentication using TOTP credentials."""
        logger.info("[Watchdog] Initiating headless session re-authentication...")
        try:
            if self.totp_key and hasattr(self.gateway, "creds"):
                self.gateway.creds.totp_key = self.totp_key
            success = bool(self.gateway.authenticate())
            if success:
                self.last_auth_time = time.time()
                self.last_ping_time = time.time()
                self.last_daily_reauth_date = datetime.now(IST).strftime("%Y-%m-%d")
                logger.info("[Watchdog] Session re-authentication successful.")
                return True
            logger.error("[Watchdog] Session re-authentication failed.")
            return False
        except Exception as e:
            logger.error(f"[Watchdog] Session refresh error: {e}")
            return False

    def probe_and_heal(self) -> Dict[str, Any]:
        """Performs a health check and automatically heals the session if expired."""
        alive = self.check_session_alive()
        if alive:
            return {
                "stat": "Ok",
                "healthy": True,
                "reauthenticated": False,
                "token_age_s": round(time.time() - self.last_auth_time, 1),
            }
        reauthed = self.refresh_session()
        return {
            "stat": "Ok" if reauthed else "Not_Ok",
            "healthy": reauthed,
            "reauthenticated": True,
            "token_age_s": 0.0 if reauthed else round(time.time() - self.last_auth_time, 1),
        }

    def check_daily_boundary(self) -> bool:
        """Executes daily re-authentication at 04:00 AM IST."""
        now = datetime.now(IST)
        today_str = now.strftime("%Y-%m-%d")
        if now.time() >= dtime(4, 0) and self.last_daily_reauth_date != today_str:
            logger.info(f"[Watchdog] 04:00 AM IST boundary reached for {today_str}. Refreshing session...")
            ok = self.refresh_session()
            if ok:
                self.last_daily_reauth_date = today_str
            return ok
        return False
