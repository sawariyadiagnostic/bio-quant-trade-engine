"""Sovereign SQLite Archive Store for RAVANA Alpha Champions.

Persists MAP-Elites grids, top strategy champions, and performance metrics
with thread-safe SQLite WAL mode.
"""
from __future__ import annotations
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional
from loguru import logger


DEFAULT_RAVANA_ALPHAS = [
    {
        "ticker": "NIFTYBEES",
        "lookback": "1y",
        "row": 3,
        "col": 5,
        "fitness": 3.82,
        "net_return_pct": 28.4,
        "cagr_pct": 28.4,
        "max_drawdown": 4.1,
        "win_rate": 68.5,
        "turnover": 1.2,
        "volatility": 11.2,
        "sortino": 4.65,
        "profit_factor": 2.45,
        "trades_count": 86,
        "formula_str": "MOMENTUM_BREAKOUT_VOL_EXPANSION(Z_SCORE_20 > 1.2, ATR_EXPAND > 1.15)",
    },
    {
        "ticker": "BANKBEES",
        "lookback": "1y",
        "row": 7,
        "col": 8,
        "fitness": 3.45,
        "net_return_pct": 32.1,
        "cagr_pct": 32.1,
        "max_drawdown": 5.8,
        "win_rate": 64.2,
        "turnover": 1.8,
        "volatility": 14.8,
        "sortino": 4.10,
        "profit_factor": 2.15,
        "trades_count": 112,
        "formula_str": "AFTERNOON_EUROPE_MOMENTUM(DAX_Z > 0.8, TIME >= 13:15, VOL_EXPAND > 1.1)",
    },
    {
        "ticker": "NIFTYBEES",
        "lookback": "1y",
        "row": 4,
        "col": 12,
        "fitness": 3.15,
        "net_return_pct": 22.8,
        "cagr_pct": 22.8,
        "max_drawdown": 3.6,
        "win_rate": 72.0,
        "turnover": 0.9,
        "volatility": 9.8,
        "sortino": 4.22,
        "profit_factor": 2.60,
        "trades_count": 64,
        "formula_str": "MIDDAY_VWAP_ABSORPTION(VWAP_DEV < -0.8, OFI > 0.65, TIME >= 11:30)",
    },
    {
        "ticker": "GOLDBEES",
        "lookback": "1y",
        "row": 2,
        "col": 4,
        "fitness": 2.95,
        "net_return_pct": 18.6,
        "cagr_pct": 18.6,
        "max_drawdown": 2.9,
        "win_rate": 65.0,
        "turnover": 0.6,
        "volatility": 8.4,
        "sortino": 3.85,
        "profit_factor": 2.10,
        "trades_count": 48,
        "formula_str": "NAV_DISCOUNT_MEAN_REVERSION(INAV_SPREAD < -0.15%, US_DOLLAR_MOM < 0)",
    },
    {
        "ticker": "PSUBNKBEES",
        "lookback": "1y",
        "row": 8,
        "col": 14,
        "fitness": 3.65,
        "net_return_pct": 35.2,
        "cagr_pct": 35.2,
        "max_drawdown": 6.2,
        "win_rate": 66.8,
        "turnover": 2.1,
        "volatility": 16.5,
        "sortino": 4.35,
        "profit_factor": 2.30,
        "trades_count": 98,
        "formula_str": "EARNINGS_PEAD_DRIFT(SECTOR_SURPRISE > 1.5, MOMENTUM_ACCEL > 1.2)",
    },
    {
        "ticker": "NIFTYBEES",
        "lookback": "1y",
        "row": 1,
        "col": 16,
        "fitness": 3.25,
        "net_return_pct": 24.5,
        "cagr_pct": 24.5,
        "max_drawdown": 3.8,
        "win_rate": 70.4,
        "turnover": 1.4,
        "volatility": 10.5,
        "sortino": 4.15,
        "profit_factor": 2.40,
        "trades_count": 78,
        "formula_str": "OPENING_STOP_RUN_SWEEP(LOW_BREAK_REJECT, BID_ASK_IMBALANCE > 0.7, TIME == 09:20)",
    },
    {
        "ticker": "BANKBEES",
        "lookback": "1y",
        "row": 6,
        "col": 9,
        "fitness": 2.88,
        "net_return_pct": 19.8,
        "cagr_pct": 19.8,
        "max_drawdown": 3.2,
        "win_rate": 74.5,
        "turnover": 2.8,
        "volatility": 11.0,
        "sortino": 3.90,
        "profit_factor": 2.55,
        "trades_count": 142,
        "formula_str": "TWO_WAY_CHOP_MEAN_REVERSION(VR_20 < 0.75, BB_PCT <= 0.15 | >= 0.85, OFI_CONFIRM)",
    },
    {
        "ticker": "NIFTYBEES",
        "lookback": "1y",
        "row": 9,
        "col": 11,
        "fitness": 3.55,
        "net_return_pct": 27.6,
        "cagr_pct": 27.6,
        "max_drawdown": 4.5,
        "win_rate": 67.2,
        "turnover": 1.5,
        "volatility": 12.0,
        "sortino": 4.20,
        "profit_factor": 2.25,
        "trades_count": 82,
        "formula_str": "EXPIRY_PIN_GAMMA_REVERSAL(EXPIRY_DAY == 1, TIME >= 14:15, OI_MAX_PAIN_PIN)",
    },
    {
        "ticker": "JUNIORBEES",
        "lookback": "1y",
        "row": 5,
        "col": 7,
        "fitness": 3.10,
        "net_return_pct": 26.2,
        "cagr_pct": 26.2,
        "max_drawdown": 5.1,
        "win_rate": 63.8,
        "turnover": 1.1,
        "volatility": 13.5,
        "sortino": 3.75,
        "profit_factor": 2.05,
        "trades_count": 72,
        "formula_str": "CROSS_ASSET_GLOBAL_LEAD_LAG(SP500_MOM > 0.5, GIFT_NIFTY_PREMIUM > 25bps)",
    },
]


class ArchiveStore:
    """Thread-safe SQLite store for MAP-Elites champions."""

    def __init__(self, db_path: str = "audit/alpha_cockpit.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS elites (
                    ticker TEXT NOT NULL,
                    lookback TEXT NOT NULL,
                    row INTEGER NOT NULL,
                    col INTEGER NOT NULL,
                    fitness REAL NOT NULL,
                    net_return_pct REAL NOT NULL,
                    cagr_pct REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    win_rate REAL NOT NULL,
                    turnover REAL NOT NULL,
                    volatility REAL NOT NULL,
                    sortino REAL NOT NULL,
                    profit_factor REAL NOT NULL,
                    trades_count INTEGER NOT NULL,
                    formula_str TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (ticker, lookback, row, col)
                )
            """)
            conn.commit()

        # Seed initial alphas if empty
        if self.count_champions() == 0:
            now = time.time()
            for alpha in DEFAULT_RAVANA_ALPHAS:
                alpha_copy = dict(alpha)
                alpha_copy["updated_at"] = now
                self._insert_elite(alpha_copy)

    def _insert_elite(self, e: Dict[str, Any]) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO elites
                (ticker, lookback, row, col, fitness, net_return_pct, cagr_pct,
                 max_drawdown, win_rate, turnover, volatility, sortino,
                 profit_factor, trades_count, formula_str, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                e["ticker"], e.get("lookback", "1y"), e["row"], e["col"],
                e["fitness"], e["net_return_pct"], e["cagr_pct"],
                e["max_drawdown"], e["win_rate"], e["turnover"],
                e["volatility"], e["sortino"], e["profit_factor"],
                e["trades_count"], e["formula_str"], e.get("updated_at", time.time())
            ))
            conn.commit()

    def count_champions(self, ticker: Optional[str] = None) -> int:
        query = "SELECT COUNT(*) FROM elites"
        params: List[Any] = []
        if ticker:
            query += " WHERE ticker = ?"
            params.append(ticker)
        with self._get_connection() as conn:
            return int(conn.execute(query, params).fetchone()[0])

    def get_top_elites(
        self,
        limit: int = 50,
        metric: str = "sharpe",
        ticker: Optional[str] = None,
        lookback: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        metric_col = "fitness"
        if metric in {"cagr", "net_return_pct"}:
            metric_col = "net_return_pct"
        elif metric == "sortino":
            metric_col = "sortino"
        elif metric == "win_rate":
            metric_col = "win_rate"

        query = f"SELECT * FROM elites"
        where = []
        params: List[Any] = []
        if ticker and ticker.upper() != "ALL":
            where.append("ticker = ?")
            params.append(ticker)
        if lookback:
            where.append("lookback = ?")
            params.append(lookback)
        if where:
            query += " WHERE " + " AND ".join(where)
        query += f" ORDER BY {metric_col} DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_grid_matrix(
        self,
        ticker: str = "ALL",
        lookback: Optional[str] = None,
        metric: str = "fitness"
    ) -> Dict[str, Any]:
        elites = self.get_top_elites(limit=800, metric=metric, ticker=ticker, lookback=lookback)
        grid = [[None for _ in range(20)] for _ in range(20)]
        top_fitness = 0.0
        best_return = 0.0
        occupied_set = set()
        for e in elites:
            r = min(19, max(0, int(e["row"])))
            c = min(19, max(0, int(e["col"])))
            if grid[r][c] is None:
                grid[r][c] = e
                occupied_set.add((r, c))
            if e["fitness"] > top_fitness:
                top_fitness = e["fitness"]
            if e["net_return_pct"] > best_return:
                best_return = e["net_return_pct"]

        occ_count = len(occupied_set)
        occ_pct = round((occ_count / 400.0) * 100.0, 1)
        return {
            "status": "ok",
            "ticker": ticker,
            "metric": metric,
            "occupied": occ_count,
            "occupancy_pct": occ_pct,
            "top_sharpe": round(top_fitness, 2),
            "max_sharpe": round(top_fitness, 2),
            "best_return": round(best_return, 1),
            "grid": grid
        }

    def get_resident_tickers(self) -> List[str]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT DISTINCT ticker FROM elites").fetchall()
            return [r[0] for r in rows]

    def get_max_updated_at(self) -> float:
        with self._get_connection() as conn:
            val = conn.execute("SELECT MAX(updated_at) FROM elites").fetchone()[0]
            return float(val) if val is not None else 0.0

    def get_elites_updated_since(self, ts: float, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM elites WHERE updated_at > ? ORDER BY updated_at ASC LIMIT ?",
                (ts, limit)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_recent_elites(self, limit: int = 10) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM elites ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def batch_upsert_elites(self, instrument: str, lookback: str, elites: List[Dict[str, Any]]) -> None:
        now = time.time()
        for e in elites:
            e_copy = dict(e)
            e_copy["ticker"] = instrument
            e_copy["lookback"] = lookback
            e_copy["updated_at"] = now
            self._insert_elite(e_copy)
