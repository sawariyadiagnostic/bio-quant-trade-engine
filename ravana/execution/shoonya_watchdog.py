"""Headless QuickAuth Session Watchdog for Shoonya Broker Gateway.

Provides automated session keepalives, periodic health probes, and automated
TOTP generation via pyotp for zero-touch headless authentication during
pre-market readiness (08:45 IST) and active market sessions.
"""
from __future__ import annotations
import threading
import time
from typing import Any, Optional
import pyotp
from loguru import logger


class ShoonyaSessionWatchdog:
    """Manages broker session token validity and automated TOTP re-authentication."""

    def __init__(
        self,
        gateway: Any,
        totp_secret: str = "",
        interval_seconds: float = 60.0
    ):
        self.gateway = gateway
        self.totp_secret = totp_secret
        self.interval_seconds = max(1.0, float(interval_seconds))
        self.last_ping_time = time.time()
        self.last_auth_time = time.time() if getattr(gateway, "susertoken", None) else 0.0
        self._stop_event = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None

    def wait_for_totp_epoch_buffer(self, min_buffer_seconds: float = 4.0) -> None:
        """Wait if current 30s TOTP cycle has less than min_buffer_seconds remaining."""
        remaining = 30.0 - (time.time() % 30.0)
        if remaining < min_buffer_seconds:
            time.sleep(remaining + 0.1)

    def start_heartbeat(self) -> None:
        """Start active 60-second background keepalive heartbeat thread."""
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return
        self._stop_event.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="ShoonyaSessionHeartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()
        logger.info("[Watchdog] Active background heartbeat thread started.")

    def stop_heartbeat(self) -> None:
        """Signal and join the background heartbeat thread."""
        self._stop_event.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2.0)
            self._heartbeat_thread = None
        logger.info("[Watchdog] Background heartbeat stopped.")

    def _heartbeat_loop(self) -> None:
        """Periodic background probe and auto-heal loop."""
        while not self._stop_event.is_set():
            try:
                self.probe_and_heal()
            except Exception as e:
                logger.error(f"[Watchdog] Heartbeat loop error: {e}")
            self._stop_event.wait(timeout=self.interval_seconds)

    def generate_totp(self) -> str:
        """Generates the current 6-digit RFC 6238 TOTP token with epoch buffer protection."""
        if not self.totp_secret:
            raise ValueError("TOTP secret key is empty or not configured.")
        self.wait_for_totp_epoch_buffer()
        totp = pyotp.TOTP(self.totp_secret)
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
            if self.totp_secret and hasattr(self.gateway, "creds"):
                self.gateway.creds.totp_secret = self.totp_secret

            success = bool(self.gateway.authenticate())
            if success:
                self.last_auth_time = time.time()
                self.last_ping_time = time.time()
                logger.info("[Watchdog] Session re-authentication successful.")
                return True

            logger.error("[Watchdog] Session re-authentication failed.")
            return False
        except Exception as e:
            logger.error(f"[Watchdog] Session refresh error: {e}")
            return False

    def probe_and_heal(self) -> dict[str, Any]:
        """Performs a health check and automatically heals the session if expired."""
        alive = self.check_session_alive()
        if alive:
            return {
                "stat": "Ok",
                "healthy": True,
                "reauthenticated": False,
                "token_age_s": round(time.time() - self.last_auth_time, 1)
            }

        reauthed = self.refresh_session()
        return {
            "stat": "Ok" if reauthed else "Not_Ok",
            "healthy": reauthed,
            "reauthenticated": True,
            "token_age_s": 0.0 if reauthed else round(time.time() - self.last_auth_time, 1)
        }
