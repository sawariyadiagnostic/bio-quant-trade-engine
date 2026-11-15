"""ACID SQLite State Ledger & 11:32 AM Reboot Recovery Protocol.

Anchors runtime strategy state into an ACID SQLite database with Write-Ahead Logging (WAL):
1. Active Position Tracking with Dynamic Profit Targets & Trailing Ratchets.
2. Tranche 1 50% Scale-Out and Position Lifecycle.
3. Immutable Order Audit Trail with Exchange ID Reconciliation.
4. The 11:32 AM Reboot Recovery Protocol (broker vs. SQLite reconciliation & ATR-based adoption).
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PositionRecord:
    """In-memory representation of an active ledger position."""

    ticker: str
    originating_trigger: str
    product_type: str  # 'MIS' | 'CNC'
    shares_total: int
    shares_remaining: int
    tranche_1_filled: bool
    entry_price: float
    entry_timestamp: str
    dynamic_profit_target: float
    trailing_stop_loss: float
    peak_price_observed: float
    holding_bars: int = 0
    is_locked: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert position record to serializable dictionary."""
        return asdict(self)


@dataclass
class OrderAuditRecord:
    """Immutable audit trail row for an order submitted to broker/exchange."""

    client_order_id: str
    exchange_order_id: Optional[str]
    ticker: str
    side: str
    shares: int
    limit_price: Optional[float]
    status: str
    trigger_source: str
    submitted_at: str
    reconciled_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert order audit record to serializable dictionary."""
        return asdict(self)


@dataclass
class DecisionAuditRecord:
    """Immutable audit record for a Gemini Decision API evaluation."""

    decision_id: str
    timestamp: str
    timeframe: str  # '1m' | '5m' | '1d'
    cluster_id: str  # 'veto' | 'elastic_string' | 'alpha_crown' | etc.
    question_key: str
    selected_choice: Optional[str]
    probability_score: Optional[float]
    latency_ms: float
    status: str  # 'APPLIED' | 'VETOED' | 'BYPASSED'
    symbol: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision audit record to serializable dictionary."""
        return asdict(self)


class SQLiteStateLedger:
    """ACID SQLite State Machine maintaining active positions and order audit trail."""

    def __init__(self, db_path: str = "audit/ravana_state_ledger.db") -> None:
        self.db_path = db_path
        self._lock = threading.RLock()
        self._ensure_storage_dir()
        self._init_database()

    def _ensure_storage_dir(self) -> None:
        """Create parent directory for SQLite database if not present."""
        dirname = os.path.dirname(self.db_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

    @contextmanager
    def _get_connection(self):
        """Create a configured SQLite connection with WAL mode, closing on context exit."""
        conn = sqlite3.connect(self.db_path, timeout=10.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_database(self) -> None:
        """Initialize required relational schemas and indexes."""
        with self._lock, self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS active_positions (
                    ticker TEXT PRIMARY KEY,
                    originating_trigger TEXT NOT NULL,
                    product_type TEXT NOT NULL,
                    shares_total INTEGER NOT NULL,
                    shares_remaining INTEGER NOT NULL,
                    tranche_1_filled INTEGER NOT NULL DEFAULT 0,
                    entry_price REAL NOT NULL,
                    entry_timestamp TEXT NOT NULL,
                    dynamic_profit_target REAL NOT NULL,
                    trailing_stop_loss REAL NOT NULL,
                    peak_price_observed REAL NOT NULL,
                    holding_bars INTEGER NOT NULL DEFAULT 0,
                    is_locked INTEGER NOT NULL DEFAULT 0
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS order_audit_trail (
                    client_order_id TEXT PRIMARY KEY,
                    exchange_order_id TEXT,
                    ticker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    shares INTEGER NOT NULL,
                    limit_price REAL,
                    status TEXT NOT NULL,
                    trigger_source TEXT NOT NULL,
                    submitted_at TEXT NOT NULL,
                    reconciled_at TEXT
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_order_ticker ON order_audit_trail(ticker);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_order_status ON order_audit_trail(status);")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS gemini_decisions_audit (
                    decision_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    cluster_id TEXT NOT NULL,
                    symbol TEXT,
                    question_key TEXT NOT NULL,
                    selected_choice TEXT,
                    probability_score REAL,
                    latency_ms REAL NOT NULL,
                    status TEXT NOT NULL
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_decision_ts ON gemini_decisions_audit(timestamp);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_decision_sym ON gemini_decisions_audit(symbol);")

    def record_entry(
        self,
        ticker: str,
        originating_trigger: str,
        product_type: str,
        shares_total: int,
        entry_price: float,
        dynamic_profit_target: float,
        trailing_stop_loss: float,
        peak_price_observed: Optional[float] = None,
        entry_timestamp: Optional[str] = None,
        is_locked: bool = False,
    ) -> None:
        """Record new open position and initial stop/target thresholds."""
        ts = entry_timestamp or datetime.now(timezone.utc).isoformat()
        peak = peak_price_observed if peak_price_observed is not None else entry_price
        product = str(product_type).upper().strip()

        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO active_positions (
                    ticker, originating_trigger, product_type,
                    shares_total, shares_remaining, tranche_1_filled,
                    entry_price, entry_timestamp, dynamic_profit_target,
                    trailing_stop_loss, peak_price_observed, holding_bars, is_locked
                ) VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, 0, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    originating_trigger=excluded.originating_trigger,
                    product_type=excluded.product_type,
                    shares_total=excluded.shares_total,
                    shares_remaining=excluded.shares_remaining,
                    tranche_1_filled=0,
                    entry_price=excluded.entry_price,
                    entry_timestamp=excluded.entry_timestamp,
                    dynamic_profit_target=excluded.dynamic_profit_target,
                    trailing_stop_loss=excluded.trailing_stop_loss,
                    peak_price_observed=excluded.peak_price_observed,
                    holding_bars=0,
                    is_locked=excluded.is_locked;
                """,
                (
                    ticker,
                    originating_trigger,
                    product,
                    int(shares_total),
                    int(shares_total),
                    float(entry_price),
                    ts,
                    float(dynamic_profit_target),
                    float(trailing_stop_loss),
                    float(peak),
                    1 if is_locked else 0,
                ),
            )

    def update_ratchet(
        self,
        ticker: str,
        trailing_stop_loss: float,
        peak_price_observed: float,
        holding_bars: Optional[int] = None,
    ) -> bool:
        """Update trailing stop loss, peak observed price, and elapsed holding bars."""
        with self._lock, self._get_connection() as conn:
            cur = conn.cursor()
            if holding_bars is None:
                cur.execute(
                    """
                    UPDATE active_positions
                    SET trailing_stop_loss = ?,
                        peak_price_observed = ?,
                        holding_bars = holding_bars + 1
                    WHERE ticker = ?;
                    """,
                    (float(trailing_stop_loss), float(peak_price_observed), ticker),
                )
            else:
                cur.execute(
                    """
                    UPDATE active_positions
                    SET trailing_stop_loss = ?,
                        peak_price_observed = ?,
                        holding_bars = ?
                    WHERE ticker = ?;
                    """,
                    (float(trailing_stop_loss), float(peak_price_observed), int(holding_bars), ticker),
                )
            return cur.rowcount > 0

    def record_tranche_1_exit(
        self,
        ticker: str,
        exit_shares: Optional[int] = None,
        exit_price: Optional[float] = None,
    ) -> Tuple[bool, int]:
        """Record 50% scale-out fill (Tranche 1), updating remaining inventory."""
        with self._lock, self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT shares_total, shares_remaining, tranche_1_filled FROM active_positions WHERE ticker = ?;", (ticker,))
            row = cur.fetchone()
            if not row:
                return False, 0

            shares_total = int(row["shares_total"])
            shares_remaining = int(row["shares_remaining"])
            qty_to_exit = exit_shares if exit_shares is not None else max(1, shares_total // 2)
            qty_to_exit = min(qty_to_exit, shares_remaining)
            new_remaining = max(0, shares_remaining - qty_to_exit)

            cur.execute(
                """
                UPDATE active_positions
                SET shares_remaining = ?,
                    tranche_1_filled = 1
                WHERE ticker = ?;
                """,
                (new_remaining, ticker),
            )
            return True, qty_to_exit

    def close_position(self, ticker: str) -> bool:
        """Remove active position from runtime state upon full exit or liquidation."""
        with self._lock, self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM active_positions WHERE ticker = ?;", (ticker,))
            return cur.rowcount > 0

    def get_active_positions(self) -> Dict[str, Dict[str, Any]]:
        """Retrieve all active open positions across the system upon reboot."""
        with self._lock, self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM active_positions ORDER BY ticker ASC;")
            rows = cur.fetchall()

            positions: Dict[str, Dict[str, Any]] = {}
            for r in rows:
                rec = PositionRecord(
                    ticker=str(r["ticker"]),
                    originating_trigger=str(r["originating_trigger"]),
                    product_type=str(r["product_type"]),
                    shares_total=int(r["shares_total"]),
                    shares_remaining=int(r["shares_remaining"]),
                    tranche_1_filled=bool(r["tranche_1_filled"]),
                    entry_price=float(r["entry_price"]),
                    entry_timestamp=str(r["entry_timestamp"]),
                    dynamic_profit_target=float(r["dynamic_profit_target"]),
                    trailing_stop_loss=float(r["trailing_stop_loss"]),
                    peak_price_observed=float(r["peak_price_observed"]),
                    holding_bars=int(r["holding_bars"]),
                    is_locked=bool(r["is_locked"]),
                )
                positions[rec.ticker] = rec.to_dict()
            return positions

    def record_order(
        self,
        client_order_id: str,
        ticker: str,
        side: str,
        shares: int,
        trigger_source: str,
        exchange_order_id: Optional[str] = None,
        limit_price: Optional[float] = None,
        status: str = "SUBMITTED",
        submitted_at: Optional[str] = None,
        reconciled_at: Optional[str] = None,
    ) -> None:
        """Insert or update order audit trail row."""
        ts = submitted_at or datetime.now(timezone.utc).isoformat()
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO order_audit_trail (
                    client_order_id, exchange_order_id, ticker,
                    side, shares, limit_price, status,
                    trigger_source, submitted_at, reconciled_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(client_order_id) DO UPDATE SET
                    exchange_order_id=COALESCE(excluded.exchange_order_id, order_audit_trail.exchange_order_id),
                    status=excluded.status,
                    reconciled_at=COALESCE(excluded.reconciled_at, order_audit_trail.reconciled_at);
                """,
                (
                    client_order_id,
                    exchange_order_id,
                    ticker,
                    str(side).upper().strip(),
                    int(shares),
                    float(limit_price) if limit_price is not None else None,
                    str(status).upper().strip(),
                    trigger_source,
                    ts,
                    reconciled_at,
                ),
            )

    def reconcile_order(
        self,
        client_order_id: str,
        exchange_order_id: str,
        status: str,
        reconciled_at: Optional[str] = None,
    ) -> bool:
        """Update order audit record with exchange order ID and execution status."""
        rec_ts = reconciled_at or datetime.now(timezone.utc).isoformat()
        with self._lock, self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE order_audit_trail
                SET exchange_order_id = ?,
                    status = ?,
                    reconciled_at = ?
                WHERE client_order_id = ?;
                """,
                (exchange_order_id, str(status).upper().strip(), rec_ts, client_order_id),
            )
            return cur.rowcount > 0

    def get_order_audit_trail(self, ticker: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve recent audit records optionally filtered by ticker."""
        with self._lock, self._get_connection() as conn:
            cur = conn.cursor()
            if ticker:
                cur.execute("SELECT * FROM order_audit_trail WHERE ticker = ? ORDER BY submitted_at ASC;", (ticker,))
            else:
                cur.execute("SELECT * FROM order_audit_trail ORDER BY submitted_at ASC;")
            rows = cur.fetchall()
            return [dict(r) for r in rows]

    def reconcile_with_broker(
        self,
        broker_positions: Dict[str, int],
        current_prices: Optional[Dict[str, float]] = None,
        current_atrs: Optional[Dict[str, float]] = None,
        default_atr_pct: float = 0.015,
    ) -> Dict[str, Any]:
        """The 11:32 AM Reboot Recovery Protocol.

        Reconciles SQLite state against live broker positions.
        If broker holds shares unrecorded in SQLite, adopts with emergency trailing stop at P - 2.0*ATR.
        """
        prices = current_prices or {}
        atrs = current_atrs or {}
        active = self.get_active_positions()

        # Clean broker position dictionary (filter out zero or negative shares)
        live_holdings = {k: int(v) for k, v in broker_positions.items() if int(v) > 0}

        in_sync: List[Dict[str, Any]] = []
        adopted: List[Dict[str, Any]] = []
        closed: List[Dict[str, Any]] = []
        adjusted: List[Dict[str, Any]] = []

        all_tickers = sorted(set(active.keys()) | set(live_holdings.keys()))

        for sym in all_tickers:
            db_rec = active.get(sym)
            live_qty = live_holdings.get(sym, 0)

            if db_rec is not None and live_qty > 0:
                self._reconcile_existing_position(
                    sym, db_rec, live_qty, in_sync, adjusted
                )
            elif db_rec is not None and live_qty == 0:
                self.close_position(sym)
                closed.append({
                    "ticker": sym,
                    "previous_shares": db_rec["shares_remaining"],
                    "action": "CLOSED_OFFLINE",
                })
            elif db_rec is None and live_qty > 0:
                adopt_record = self._adopt_unrecorded_position(
                    sym, live_qty, prices, atrs, default_atr_pct
                )
                adopted.append(adopt_record)

        return {
            "reconciled_at": datetime.now(timezone.utc).isoformat(),
            "discrepancies_found": bool(adopted or closed or adjusted),
            "in_sync": in_sync,
            "adopted": adopted,
            "closed": closed,
            "adjusted": adjusted,
        }

    # -------------------------------------------------------------------------
    # Reconciliation Helpers (< 50 lines each)
    # -------------------------------------------------------------------------

    def _reconcile_existing_position(
        self,
        ticker: str,
        db_rec: Dict[str, Any],
        live_qty: int,
        in_sync: List[Dict[str, Any]],
        adjusted: List[Dict[str, Any]],
    ) -> None:
        """Handle reconciliation when ticker is present in both SQLite and Broker."""
        db_qty = int(db_rec["shares_remaining"])
        if db_qty == live_qty:
            in_sync.append({"ticker": ticker, "shares": live_qty})
            return

        new_total = max(int(db_rec["shares_total"]), live_qty)
        tranche_1 = 1 if (live_qty <= new_total // 2) else (1 if db_rec["tranche_1_filled"] else 0)

        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                UPDATE active_positions
                SET shares_remaining = ?,
                    shares_total = ?,
                    tranche_1_filled = ?
                WHERE ticker = ?;
                """,
                (live_qty, new_total, tranche_1, ticker),
            )

        adjusted.append({
            "ticker": ticker,
            "previous_shares": db_qty,
            "broker_shares": live_qty,
            "action": "SHARES_RECONCILED",
        })

    def _adopt_unrecorded_position(
        self,
        ticker: str,
        live_qty: int,
        prices: Dict[str, float],
        atrs: Dict[str, float],
        default_atr_pct: float,
    ) -> Dict[str, Any]:
        """Adopt unrecorded broker position with emergency ATR trailing stop."""
        px = float(prices.get(ticker, 100.0))
        atr = float(atrs.get(ticker, px * default_atr_pct))
        emergency_stop = round(px - (2.0 * atr), 2)
        profit_target = round(px + (2.0 * atr), 2)

        self.record_entry(
            ticker=ticker,
            originating_trigger="EMERGENCY_REBOOT_RECOVERY",
            product_type="CNC",
            shares_total=live_qty,
            entry_price=px,
            dynamic_profit_target=profit_target,
            trailing_stop_loss=emergency_stop,
            peak_price_observed=px,
            is_locked=False,
        )

        return {
            "ticker": ticker,
            "shares": live_qty,
            "entry_price": px,
            "trailing_stop_loss": emergency_stop,
            "dynamic_profit_target": profit_target,
            "action": "ADOPTED_EMERGENCY_REBOOT",
        }

    # -------------------------------------------------------------------------
    # 5. Gemini Decisions Audit Trail
    # -------------------------------------------------------------------------
    def record_decision(self, record: DecisionAuditRecord) -> None:
        """Persist a single Gemini Decision audit entry."""
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO gemini_decisions_audit (
                    decision_id, timestamp, timeframe, cluster_id, symbol,
                    question_key, selected_choice, probability_score, latency_ms, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(decision_id) DO UPDATE SET
                    selected_choice = excluded.selected_choice,
                    probability_score = excluded.probability_score,
                    latency_ms = excluded.latency_ms,
                    status = excluded.status;
                """,
                (
                    record.decision_id,
                    record.timestamp,
                    record.timeframe,
                    record.cluster_id,
                    record.symbol,
                    record.question_key,
                    record.selected_choice,
                    record.probability_score,
                    float(record.latency_ms),
                    record.status,
                ),
            )

    def record_decision_batch(self, records: List[DecisionAuditRecord]) -> None:
        """Persist a batch of Gemini Decision audit entries in a single transaction."""
        if not records:
            return
        with self._lock, self._get_connection() as conn:
            conn.executemany(
                """
                INSERT INTO gemini_decisions_audit (
                    decision_id, timestamp, timeframe, cluster_id, symbol,
                    question_key, selected_choice, probability_score, latency_ms, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(decision_id) DO UPDATE SET
                    selected_choice = excluded.selected_choice,
                    probability_score = excluded.probability_score,
                    latency_ms = excluded.latency_ms,
                    status = excluded.status;
                """,
                [
                    (
                        r.decision_id,
                        r.timestamp,
                        r.timeframe,
                        r.cluster_id,
                        r.symbol,
                        r.question_key,
                        r.selected_choice,
                        r.probability_score,
                        float(r.latency_ms),
                        r.status,
                    )
                    for r in records
                ],
            )

    def get_recent_decisions(
        self,
        symbol: Optional[str] = None,
        cluster_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Retrieve recent decision audit entries, optionally filtered by symbol or cluster."""
        with self._lock, self._get_connection() as conn:
            query = "SELECT * FROM gemini_decisions_audit"
            params: List[Any] = []
            clauses: List[str] = []
            if symbol:
                clauses.append("symbol = ?")
                params.append(symbol)
            if cluster_id:
                clauses.append("cluster_id = ?")
                params.append(cluster_id)
            if clauses:
                query += " WHERE " + " AND ".join(clauses)
            query += " ORDER BY ROWID DESC LIMIT ?"
            params.append(limit)

            cur = conn.execute(query, tuple(params))
            return [dict(row) for row in cur.fetchall()]
