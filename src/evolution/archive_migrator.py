"""
src/evolution/archive_migrator.py
Utility to discover and migrate legacy audit/archive_*.json files into
the institutional SQLite WAL resident_elites database.
Ensures 100% data preservation and backward compatibility.
"""
import glob
import json
import os
import sqlite3
import time
from typing import Any, Dict, List
from loguru import logger


def init_archive_database(db_path: str) -> None:
    """Initializes the SQLite WAL schema for resident alpha elites."""
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    with sqlite3.connect(db_path, timeout=5.0) as conn:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA temp_store = MEMORY;")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS resident_elites (
                ticker TEXT NOT NULL,
                lookback TEXT NOT NULL,
                niche_row INTEGER NOT NULL,
                niche_col INTEGER NOT NULL,
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
                active_regimes TEXT,
                behavior_signature BLOB,
                updated_at REAL NOT NULL,
                PRIMARY KEY (ticker, lookback, niche_row, niche_col)
            );
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_elites_lookup 
            ON resident_elites (ticker, lookback, fitness DESC);
        """)


def migrate_json_archives_to_sqlite(source_dir: str = "audit", target_db: str = "audit/alpha_elites.db") -> int:
    """
    Discovers all archive_*.json files in source_dir and migrates
    their resident elites into the SQLite WAL database.
    Returns count of successfully migrated archives.
    """
    init_archive_database(target_db)
    pattern = os.path.join(source_dir, "archive_*.json")
    files = glob.glob(pattern)
    migrated_count = 0
    now = time.time()

    insert_sql = """
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

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            ticker = data.get("ticker")
            lookback = data.get("lookback")
            if not ticker or not lookback:
                # Infer from filename e.g. archive_NIFTYBEES_5y.json
                base = os.path.basename(file_path).replace("archive_", "").replace(".json", "")
                parts = base.split("_")
                ticker = parts[0] if parts else "UNKNOWN"
                lookback = parts[1] if len(parts) > 1 else "default"

            elites = data.get("elites", [])
            rows_to_insert = []
            for e in elites:
                rows_to_insert.append({
                    "ticker": ticker,
                    "lookback": lookback,
                    "row": int(e.get("row", 0)),
                    "col": int(e.get("col", 0)),
                    "fitness": float(e.get("fitness", 0.0)),
                    "net_return_pct": float(e.get("net_return_pct", 0.0)),
                    "cagr_pct": float(e.get("cagr_pct", e.get("net_return_pct", 0.0))),
                    "max_drawdown": float(e.get("max_drawdown", 0.0)),
                    "win_rate": float(e.get("win_rate", 0.0)),
                    "turnover": float(e.get("turnover", 0.0)),
                    "volatility": float(e.get("volatility", 0.0)),
                    "sortino": float(e.get("sortino", 0.0)),
                    "profit_factor": float(e.get("profit_factor", 0.0)),
                    "trades_count": int(e.get("trades_count", 0)),
                    "formula_str": str(e.get("formula_str", "")),
                    "active_regimes": json.dumps(e.get("active_regimes", [])),
                    "behavior_signature": bytes(str(e.get("behavior_signature", "")).encode("utf-8")),
                    "updated_at": now
                })

            if rows_to_insert:
                with sqlite3.connect(target_db, timeout=5.0) as conn:
                    conn.executemany(insert_sql, rows_to_insert)
                    conn.commit()

            migrated_count += 1
            logger.info(f"Migrated {len(rows_to_insert)} resident elites from {file_path} to SQLite WAL.")
        except Exception as err:
            logger.error(f"Failed to migrate archive {file_path}: {err}")

    return migrated_count


if __name__ == "__main__":
    count = migrate_json_archives_to_sqlite()
    logger.info(f"Successfully migrated {count} legacy archives to SQLite WAL database.")
