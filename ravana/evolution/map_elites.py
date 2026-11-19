"""MAP-Elites Behavioral Archive Engine for Symbolic Alpha Formula Evolution (Muscle 9).

Maintains an institutional 20x20 diversity matrix:
- Axis Y (Row): Realized Annualized Volatility [0.0, 0.40] (20 bins)
- Axis X (Col): Portfolio Churn / Turnover [0.0, 1.0] (20 bins)

Evaluates candidates using the Composite Institutional Alpha Score (CIAS)
and persists champions directly to SQLite audit/alpha_cockpit.db.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ravana.domain.formula_ast import ASTNode


@dataclass
class EliteIndividual:
    """Represents a resident champion within a specific behavioral niche."""

    ticker: str
    lookback: str
    row: int
    col: int
    fitness: float
    net_return_pct: float
    cagr_pct: float
    max_drawdown: float
    win_rate: float
    turnover: float
    volatility: float
    sortino: float
    profit_factor: float
    trades_count: int
    formula_str: str
    cias: float
    tree: Optional[ASTNode] = None
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert elite record to serializable dictionary."""
        return {
            "ticker": self.ticker,
            "lookback": self.lookback,
            "row": self.row,
            "col": self.col,
            "fitness": round(self.fitness, 4),
            "net_return_pct": round(self.net_return_pct, 2),
            "cagr_pct": round(self.cagr_pct, 2),
            "max_drawdown": round(self.max_drawdown, 4),
            "win_rate": round(self.win_rate, 4),
            "turnover": round(self.turnover, 4),
            "volatility": round(self.volatility, 4),
            "sortino": round(self.sortino, 4),
            "profit_factor": round(self.profit_factor, 4),
            "trades_count": self.trades_count,
            "formula_str": self.formula_str,
            "cias": round(self.cias, 4),
            "updated_at": self.updated_at,
        }


class MapElitesArchive:
    """Maintains a 20x20 behavioral grid of optimal trading alpha formulas."""

    GRID_ROWS: int = 20
    GRID_COLS: int = 20
    MAX_VOLATILITY: float = 0.40
    MAX_TURNOVER: float = 1.00

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.archive: Dict[Tuple[int, int], EliteIndividual] = {}
        self.db_path = db_path or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "audit", "alpha_cockpit.db")
        )

    def compute_cell_coordinates(self, volatility: float, turnover: float) -> Tuple[int, int]:
        """Map continuous behavioral features into discrete 20x20 grid coordinates."""
        v = float(np.clip(volatility, 0.0, self.MAX_VOLATILITY - 1e-6))
        t = float(np.clip(turnover, 0.0, self.MAX_TURNOVER - 1e-6))
        row = int(np.floor((v / self.MAX_VOLATILITY) * self.GRID_ROWS))
        col = int(np.floor((t / self.MAX_TURNOVER) * self.GRID_COLS))
        return min(row, self.GRID_ROWS - 1), min(col, self.GRID_COLS - 1)

    def compute_cias(
        self,
        ic: float,
        rank_ic: float,
        dsr: float,
        sharpe: float,
        max_drawdown: float,
    ) -> float:
        """Calculate Composite Institutional Alpha Score (CIAS)."""
        norm_sharpe = float(np.clip(sharpe / 3.0, -1.0, 1.0))
        score = (
            0.35 * ic
            + 0.25 * rank_ic
            + 0.20 * dsr
            + 0.20 * norm_sharpe
            - 0.05 * max_drawdown
        )
        return float(score)

    def add(
        self,
        formula_str: str,
        volatility: float,
        turnover: float,
        fitness: float,
        ic: float = 0.0,
        rank_ic: float = 0.0,
        dsr: float = 0.5,
        sharpe: float = 1.0,
        max_drawdown: float = 0.05,
        net_return_pct: float = 10.0,
        cagr_pct: float = 12.0,
        win_rate: float = 0.55,
        sortino: float = 1.5,
        profit_factor: float = 1.4,
        trades_count: int = 50,
        ticker: str = "NIFTYBEES",
        lookback: str = "5m",
        tree: Optional[ASTNode] = None,
    ) -> bool:
        """Add candidate to archive; replaces resident if CIAS exceeds resident's score."""
        r, c = self.compute_cell_coordinates(volatility, turnover)
        cias = self.compute_cias(ic, rank_ic, dsr, sharpe, max_drawdown)

        candidate = EliteIndividual(
            ticker=ticker,
            lookback=lookback,
            row=r,
            col=c,
            fitness=fitness,
            net_return_pct=net_return_pct,
            cagr_pct=cagr_pct,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            turnover=turnover,
            volatility=volatility,
            sortino=sortino,
            profit_factor=profit_factor,
            trades_count=trades_count,
            formula_str=formula_str,
            cias=cias,
            tree=tree,
            updated_at=time.time(),
        )

        resident = self.archive.get((r, c))
        if resident is None or candidate.cias > resident.cias:
            self.archive[(r, c)] = candidate
            return True
        return False

    def get_elite(self, row: int, col: int) -> Optional[EliteIndividual]:
        """Retrieve elite at specific niche cell."""
        return self.archive.get((row, col))

    def get_all_elites(self) -> List[EliteIndividual]:
        """Return all occupied elites in the archive."""
        return list(self.archive.values())

    def occupancy_rate(self) -> float:
        """Return proportion of grid niches populated."""
        return len(self.archive) / float(self.GRID_ROWS * self.GRID_COLS)

    def sync_to_db(self, db_path: Optional[str] = None) -> int:
        """Persist all in-memory champions to SQLite alpha cockpit database."""
        target_db = db_path or self.db_path
        os.makedirs(os.path.dirname(target_db), exist_ok=True)
        conn = sqlite3.connect(target_db)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            self._ensure_table_exists(conn)
            records = [
                (
                    e.ticker, e.lookback, e.row, e.col, e.fitness, e.net_return_pct,
                    e.cagr_pct, e.max_drawdown, e.win_rate, e.turnover, e.volatility,
                    e.sortino, e.profit_factor, e.trades_count, e.formula_str, e.updated_at
                )
                for e in self.archive.values()
            ]
            conn.executemany("""
                INSERT OR REPLACE INTO elites
                (ticker, lookback, row, col, fitness, net_return_pct, cagr_pct,
                 max_drawdown, win_rate, turnover, volatility, sortino,
                 profit_factor, trades_count, formula_str, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
            conn.commit()
            return len(records)
        finally:
            conn.close()

    def load_from_db(
        self,
        db_path: Optional[str] = None,
        ticker: Optional[str] = None,
        lookback: Optional[str] = None,
    ) -> int:
        """Load archive from SQLite database."""
        target_db = db_path or self.db_path
        if not os.path.isfile(target_db):
            return 0

        conn = sqlite3.connect(target_db)
        try:
            query = "SELECT * FROM elites WHERE 1=1"
            params: List[Any] = []
            if ticker and ticker.upper() != "ALL":
                query += " AND ticker = ?"
                params.append(ticker)
            if lookback:
                query += " AND lookback = ?"
                params.append(lookback)

            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            loaded = 0
            for row in rows:
                r = int(row["row"])
                c = int(row["col"])
                e = EliteIndividual(
                    ticker=row["ticker"],
                    lookback=row["lookback"],
                    row=r,
                    col=c,
                    fitness=float(row["fitness"]),
                    net_return_pct=float(row["net_return_pct"]),
                    cagr_pct=float(row["cagr_pct"]),
                    max_drawdown=float(row["max_drawdown"]),
                    win_rate=float(row["win_rate"]),
                    turnover=float(row["turnover"]),
                    volatility=float(row["volatility"]),
                    sortino=float(row["sortino"]),
                    profit_factor=float(row["profit_factor"]),
                    trades_count=int(row["trades_count"]),
                    formula_str=row["formula_str"],
                    cias=float(row["fitness"]),
                    updated_at=float(row["updated_at"]),
                )
                self.archive[(r, c)] = e
                loaded += 1
            return loaded
        finally:
            conn.close()

    def get_best_champion(self, ticker: Optional[str] = None) -> Optional[EliteIndividual]:
        """Retrieve highest fitness elite, optionally filtering by ticker, ensuring AST tree is parsed."""
        candidates = list(self.archive.values())
        if ticker and ticker.upper() != "ALL":
            filtered = [e for e in candidates if e.ticker.upper() == ticker.upper()]
            if filtered:
                candidates = filtered

        if not candidates:
            return None

        best = max(candidates, key=lambda e: e.fitness)
        if best.tree is None and best.formula_str:
            try:
                from ravana.domain.formula_parser import FormulaParser
                best.tree = FormulaParser.parse(best.formula_str)
            except Exception:
                pass
        return best

    def _ensure_table_exists(self, conn: sqlite3.Connection) -> None:
        """Create elites table if not already present."""
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
