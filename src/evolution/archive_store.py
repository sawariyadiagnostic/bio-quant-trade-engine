"""
src/evolution/archive_store.py
Institutional SQLite Write-Ahead Logging (WAL) Archive Store.
Supports concurrent readers from FastAPI without WinError 32 file lock contention.
Protects against power-loss and abort corruption with ACID single-transaction batch commits.
"""
import json
import math
import sqlite3
import time
from typing import Any, Dict, List, Optional
from loguru import logger
from src.evolution.archive_migrator import init_archive_database


class ArchiveStore:
    """Manages high-throughput concurrent persistence of resident alpha champions."""

    def __init__(self, db_path: str = "audit/alpha_elites.db"):
        self.db_path = db_path
        init_archive_database(self.db_path)

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        conn.execute("PRAGMA temp_store = MEMORY;")
        return conn

    def batch_upsert_elites(self, ticker: str, lookback: str, elites: List[Dict[str, Any]]) -> None:
        """Atomic batch commit of discovered elites in a single ACID transaction (<1ms)."""
        if not elites:
            return
        now = time.time()
        sql = """
            INSERT INTO resident_elites (
                ticker, lookback, niche_row, niche_col, fitness, net_return_pct, cagr_pct,
                max_drawdown, win_rate, turnover, volatility, sortino, profit_factor,
                trades_count, formula_str, active_regimes, behavior_signature, updated_at
            ) VALUES (
                :ticker, :lookback, :row, :col, :fitness, :net_return_pct, :cagr_pct,
                :max_drawdown, :win_rate, :turnover, :volatility, :sortino, :profit_factor,
                :trades_count, :formula_str, :active_regimes, :behavior_signature, :updated_at
            ) ON CONFLICT(ticker, lookback, niche_row, niche_col) DO UPDATE SET
                fitness = excluded.fitness,
                net_return_pct = excluded.net_return_pct,
                cagr_pct = excluded.cagr_pct,
                max_drawdown = excluded.max_drawdown,
                win_rate = excluded.win_rate,
                turnover = excluded.turnover,
                volatility = excluded.volatility,
                sortino = excluded.sortino,
                profit_factor = excluded.profit_factor,
                trades_count = excluded.trades_count,
                formula_str = excluded.formula_str,
                active_regimes = excluded.active_regimes,
                behavior_signature = excluded.behavior_signature,
                updated_at = excluded.updated_at;
        """
        rows = []
        def safe_float(v: Any, default: float = 0.0) -> float:
            try:
                f = float(v)
                if math.isinf(f):
                    return 999999.0 if f > 0 else -999999.0
                if math.isnan(f):
                    return default
                return f
            except Exception:
                return default

        for e in elites:
            rows.append({
                "ticker": ticker,
                "lookback": lookback,
                "row": int(e.get("row", 0)),
                "col": int(e.get("col", 0)),
                "fitness": safe_float(e.get("fitness", 0.0)),
                "net_return_pct": safe_float(e.get("net_return_pct", 0.0)),
                "cagr_pct": safe_float(e.get("cagr_pct", e.get("net_return_pct", 0.0))),
                "max_drawdown": safe_float(e.get("max_drawdown", 0.0)),
                "win_rate": safe_float(e.get("win_rate", 0.0)),
                "turnover": safe_float(e.get("turnover", 0.0)),
                "volatility": safe_float(e.get("volatility", 0.0)),
                "sortino": safe_float(e.get("sortino", 0.0)),
                "profit_factor": safe_float(e.get("profit_factor", 0.0)),
                "trades_count": int(e.get("trades_count", 0)),
                "formula_str": str(e.get("formula_str", "")),
                "active_regimes": json.dumps(e.get("active_regimes", [])),
                "behavior_signature": bytes(str(e.get("behavior_signature", "")).encode("utf-8")),
                "updated_at": now
            })

        with self._get_connection() as conn:
            conn.executemany(sql, rows)
            conn.commit()

    @staticmethod
    def _sanitize_row(row: Any) -> Dict[str, Any]:
        """Ensures all fields are JSON compliant (decodes BLOB/bytes, sanitizes inf/nan)."""
        res = dict(row)
        for k, v in res.items():
            if isinstance(v, (bytes, bytearray, memoryview)):
                try:
                    res[k] = bytes(v).decode("utf-8")
                except Exception:
                    res[k] = bytes(v).hex()
            elif isinstance(v, float):
                if math.isinf(v):
                    res[k] = 999999.0 if v > 0 else -999999.0
                elif math.isnan(v):
                    res[k] = 0.0
        return res

    def query_leaderboard(
        self,
        ticker: str,
        lookback: str,
        sort_by: str = "fitness",
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Instant point query without file lock contention."""
        allowed_sort = {"fitness", "net_return_pct", "cagr_pct", "max_drawdown", "win_rate", "trades_count"}
        safe_sort = sort_by if sort_by in allowed_sort else "fitness"
        order_dir = "ASC" if safe_sort == "max_drawdown" else "DESC"

        sql = f"""
            SELECT * FROM resident_elites
            WHERE ticker = ? AND lookback = ?
            ORDER BY {safe_sort} {order_dir}
            LIMIT ?;
        """
        with self._get_connection() as conn:
            cursor = conn.execute(sql, (ticker, lookback, limit))
            return [self._sanitize_row(row) for row in cursor.fetchall()]

    def get_top_elites(
        self,
        limit: int = 50,
        metric: str = "fitness",
        ticker: Optional[str] = None,
        lookback: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Returns top resident elites, optionally filtering by ticker and lookback."""
        allowed_metrics = {"fitness", "net_return_pct", "cagr_pct", "max_drawdown", "win_rate", "trades_count"}
        # Map 'sharpe' to 'fitness' if requested
        safe_metric = "fitness" if metric in {"sharpe", "fitness"} else metric
        if safe_metric not in allowed_metrics:
            safe_metric = "fitness"
        order_dir = "ASC" if safe_metric == "max_drawdown" else "DESC"

        where_clauses = []
        params = []
        if ticker:
            base_t = ticker[:-3] if ticker.endswith(".NS") else ticker
            ns_t = f"{base_t}.NS"
            where_clauses.append("(ticker = ? OR ticker = ?)")
            params.extend([base_t, ns_t])
        if lookback and lookback != "all":
            where_clauses.append("lookback = ?")
            params.append(lookback)

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        sql = f"""
            SELECT * FROM resident_elites
            {where_sql}
            ORDER BY {safe_metric} {order_dir}
            LIMIT ?;
        """
        params.append(limit)
        with self._get_connection() as conn:
            cursor = conn.execute(sql, tuple(params))
            return [self._sanitize_row(row) for row in cursor.fetchall()]

    def get_recent_elites(
        self,
        limit: int = 50,
        ticker: Optional[str] = None,
        lookback: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Returns most recently updated resident elites, ensuring JSON serializability."""
        where_clauses = []
        params = []
        if ticker:
            base_t = ticker[:-3] if ticker.endswith(".NS") else ticker
            ns_t = f"{base_t}.NS"
            where_clauses.append("(ticker = ? OR ticker = ?)")
            params.extend([base_t, ns_t])
        if lookback and lookback != "all":
            where_clauses.append("lookback = ?")
            params.append(lookback)

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        sql = f"""
            SELECT * FROM resident_elites
            {where_sql}
            ORDER BY updated_at DESC
            LIMIT ?;
        """
        params.append(limit)
        with self._get_connection() as conn:
            cursor = conn.execute(sql, tuple(params))
            return [self._sanitize_row(row) for row in cursor.fetchall()]

    def count_champions(self, ticker: Optional[str] = None, lookback: Optional[str] = None) -> int:
        """Returns total champion count."""
        with self._get_connection() as conn:
            if ticker and lookback:
                base_t = ticker[:-3] if ticker.endswith(".NS") else ticker
                ns_t = f"{base_t}.NS"
                cur = conn.execute("SELECT count(*) FROM resident_elites WHERE (ticker = ? OR ticker = ?) AND lookback = ?", (base_t, ns_t, lookback))
            elif ticker:
                base_t = ticker[:-3] if ticker.endswith(".NS") else ticker
                ns_t = f"{base_t}.NS"
                cur = conn.execute("SELECT count(*) FROM resident_elites WHERE (ticker = ? OR ticker = ?)", (base_t, ns_t))
            else:
                cur = conn.execute("SELECT count(*) FROM resident_elites")
            return cur.fetchone()[0]

    def get_resident_tickers(self) -> List[str]:
        """Returns distinct tickers currently resident in the archive."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT DISTINCT ticker FROM resident_elites ORDER BY ticker").fetchall()
            return [str(r[0]) for r in rows]

    def wipe_archive(self, ticker: str, lookback: str) -> int:
        """Deletes resident champions for specific universe slice."""
        with self._get_connection() as conn:
            cur = conn.execute("DELETE FROM resident_elites WHERE ticker = ? AND lookback = ?", (ticker, lookback))
            conn.commit()
            return cur.rowcount

    def get_grid_matrix(
        self,
        ticker: str,
        lookback: Optional[str] = None,
        metric: str = "fitness"
    ) -> Dict[str, Any]:
        """Builds dense 20x20 matrix of resident elites for WebGL heatmap visualization."""
        allowed_metrics = {"fitness", "net_return_pct", "cagr_pct", "max_drawdown", "win_rate", "trades_count"}
        safe_metric = "fitness" if metric in {"sharpe", "fitness"} else metric
        if safe_metric not in allowed_metrics:
            safe_metric = "fitness"

        base_t = ticker[:-3] if ticker.endswith(".NS") else ticker
        ns_t = f"{base_t}.NS"
        params: List[Any] = [base_t, ns_t]
        where_lookback = ""
        if lookback and lookback != "all":
            where_lookback = "AND lookback = ?"
            params.append(lookback)

        sql = f"""
            SELECT niche_row, niche_col, fitness, net_return_pct, cagr_pct,
                   max_drawdown, win_rate, trades_count, formula_str
            FROM resident_elites
            WHERE (ticker = ? OR ticker = ?) {where_lookback}
        """
        with self._get_connection() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()

            # If lookback was specified but returned 0 rows, fallback to any lookback for ticker
            if not rows and lookback and lookback != "all":
                rows = conn.execute(
                    "SELECT niche_row, niche_col, fitness, net_return_pct, cagr_pct, max_drawdown, win_rate, trades_count, formula_str FROM resident_elites WHERE (ticker = ? OR ticker = ?)",
                    (base_t, ns_t)
                ).fetchall()

        z = [[None for _ in range(20)] for _ in range(20)]
        hover_text = [["Empty Niche (Fog of War)" for _ in range(20)] for _ in range(20)]
        formulas = [["" for _ in range(20)] for _ in range(20)]
        seen_fit = [[-999999.0 for _ in range(20)] for _ in range(20)]
        occupied_count = 0

        for r_data in rows:
            r = int(r_data["niche_row"])
            c = int(r_data["niche_col"])
            if 0 <= r < 20 and 0 <= c < 20:
                fit_val = float(r_data["fitness"])
                if z[r][c] is not None and fit_val <= seen_fit[r][c]:
                    continue
                seen_fit[r][c] = fit_val

                val = float(r_data[safe_metric])
                if math.isinf(val):
                    val = 999999.0 if val > 0 else -999999.0
                elif math.isnan(val):
                    val = 0.0
                if z[r][c] is None:
                    occupied_count += 1
                z[r][c] = round(val, 2)
                f_short = str(r_data["formula_str"])
                if len(f_short) > 60:
                    f_short = f_short[:57] + "..."
                hover_text[r][c] = (
                    f"Niche [{r}, {c}]<br>"
                    f"Formula: {f_short}<br>"
                    f"Sharpe: {float(r_data['fitness']):.2f}<br>"
                    f"CAGR: {float(r_data['cagr_pct']):.1f}%<br>"
                    f"MaxDD: {float(r_data['max_drawdown']):.1f}%<br>"
                    f"Trades: {int(r_data['trades_count'])}"
                )
                formulas[r][c] = str(r_data["formula_str"])

        occupancy_pct = (occupied_count / 400.0) * 100.0
        return {
            "status": "ok",
            "ticker": ticker,
            "lookback": lookback or "all",
            "metric": safe_metric,
            "occupied_count": occupied_count,
            "occupancy_pct": round(occupancy_pct, 1),
            "z": z,
            "hover_text": hover_text,
            "formulas": formulas,
            "x_bins": [f"Tau {c}" for c in range(20)],
            "y_bins": [f"Sigma {r}" for r in range(20)]
        }
