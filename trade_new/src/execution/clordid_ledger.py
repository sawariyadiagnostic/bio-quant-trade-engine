"""Persistent SQLite-backed Client Order ID Ledger ensuring idempotency across restarts."""
from __future__ import annotations
import json
import os
import sqlite3
import threading
import time
from decimal import Decimal
from typing import Any, Dict, Iterator, Optional


def _clordid_json_default(obj: Any) -> Any:
    """Serializes Decimal and date objects safely to JSON."""
    if isinstance(obj, Decimal):
        return float(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


class ClientOrderIdLedger:
    """Thread-safe persistent ledger tracking client order IDs across process restarts."""

    def __init__(self, db_path: str) -> None:
        if db_path == ":memory:":
            self.db_path = ":memory:"
        else:
            self.db_path = os.path.abspath(db_path)
            os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Initializes SQLite schema in WAL mode."""
        with self._lock, sqlite3.connect(self.db_path, timeout=10.0) as conn:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS clordid_ledger (
                    client_order_id TEXT PRIMARY KEY,
                    timestamp REAL,
                    symbol TEXT,
                    action TEXT,
                    quantity INTEGER,
                    price REAL,
                    status TEXT,
                    norenordno TEXT,
                    response_json TEXT
                );
            """)

    def get(self, client_order_id: str, default: Any = None) -> Any:
        """Retrieves order response dictionary for a client order ID."""
        with self._lock, sqlite3.connect(self.db_path, timeout=5.0) as conn:
            cur = conn.execute(
                "SELECT response_json FROM clordid_ledger WHERE client_order_id = ?",
                (client_order_id,),
            )
            row = cur.fetchone()
            if row and row[0]:
                try:
                    return json.loads(row[0])
                except json.JSONDecodeError:
                    return default
            return default

    def put(
        self,
        client_order_id: str,
        response: Dict[str, Any],
        symbol: str = "",
        action: str = "",
        quantity: int = 0,
        price: float = 0.0,
    ) -> None:
        """Inserts or updates order response for client order ID."""
        with self._lock, sqlite3.connect(self.db_path, timeout=5.0) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO clordid_ledger (
                    client_order_id, timestamp, symbol, action, quantity, price, status, norenordno, response_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                client_order_id,
                time.time(),
                symbol or str(response.get("tsym", "")),
                action or str(response.get("action", "")),
                int(quantity or response.get("qty", 0)),
                float(price or response.get("price", 0.0)),
                str(response.get("status", response.get("stat", "Ok"))),
                str(response.get("norenordno", "")),
                json.dumps(response, default=_clordid_json_default),
            ))

    def __contains__(self, client_order_id: str) -> bool:
        """Checks if client order ID exists in ledger."""
        return self.get(client_order_id) is not None

    def __getitem__(self, client_order_id: str) -> Dict[str, Any]:
        """Index access for ledger items."""
        val = self.get(client_order_id)
        if val is None:
            raise KeyError(client_order_id)
        return val

    def __setitem__(self, client_order_id: str, response: Dict[str, Any]) -> None:
        """Sets response dictionary for client order ID."""
        self.put(client_order_id, response)

    def __len__(self) -> int:
        """Returns total distinct client orders recorded."""
        with self._lock, sqlite3.connect(self.db_path, timeout=5.0) as conn:
            cur = conn.execute("SELECT COUNT(*) FROM clordid_ledger")
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def keys(self) -> list[str]:
        """Returns all recorded client order IDs."""
        with self._lock, sqlite3.connect(self.db_path, timeout=5.0) as conn:
            cur = conn.execute("SELECT client_order_id FROM clordid_ledger")
            return [str(row[0]) for row in cur.fetchall()]
