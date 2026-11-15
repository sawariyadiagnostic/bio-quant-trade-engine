"""
Multi-Sector Lakehouse Ingestion Engine for RAVANA v4.1.
Ingests 60 days of 5-minute intraday bars across all 10 Sector BEES ETFs
and paired underlying benchmark indices, with session gatekeeping,
physical invariant validation, and atomic incremental upsert.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import yfinance as yf

# UTF-8 Console Safety
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ravana.data.ingestion.config import CORE_ETF_UNIVERSE, ETF_BENCHMARK_MAP
from ravana.data.ingestion.loader import CandlestickSanityValidator
from ravana.data.ingestion.session_gatekeeper import NSESessionGatekeeper
from ravana.data.storage.upsert import IncrementalUpsertEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest_all_sectors")

OUTPUT_DIR = os.path.join(BASE_DIR, "ravana", "data", "lake", "m3_ingested", "intraday_5m")
MANIFEST_PATH = os.path.join(BASE_DIR, "ravana", "data", "lake", "m3_ingested", "m3_manifest.json")

# 10 Core Sector ETF Targets and Benchmarks
SECTOR_TARGETS: Dict[str, Dict[str, str]] = {
    "BANKBEES": {"ticker": "BANKBEES.NS", "benchmark": "^NSEBANK", "sector": "Banking_Financials"},
    "NIFTYBEES": {"ticker": "NIFTYBEES.NS", "benchmark": "^NSEI", "sector": "Broad_Market"},
    "ITBEES": {"ticker": "ITBEES.NS", "benchmark": "^CNXIT", "sector": "Information_Technology"},
    "AUTOBEES": {"ticker": "AUTOBEES.NS", "benchmark": "^CNXAUTO", "sector": "Automotive"},
    "PHARMABEES": {"ticker": "PHARMABEES.NS", "benchmark": "^CNXPHARMA", "sector": "Pharma_Healthcare"},
    "PSUBNKBEES": {"ticker": "PSUBNKBEES.NS", "benchmark": "^CNXPSUBANK", "sector": "PSU_Banking"},
    "JUNIORBEES": {"ticker": "JUNIORBEES.NS", "benchmark": "^NSEI", "sector": "Next50_Midcap"},
    "CPSEETF": {"ticker": "CPSEETF.NS", "benchmark": "^NSEI", "sector": "Energy_CPSE"},
    "GOLDBEES": {"ticker": "GOLDBEES.NS", "benchmark": "GC=F", "sector": "Commodities_Gold"},
    "SILVERBEES": {"ticker": "SILVERBEES.NS", "benchmark": "SI=F", "sector": "Commodities_Silver"},
}


def download_and_clean_ticker(
    ticker: str,
    period: str = "60d",
    interval: str = "5m",
) -> pd.DataFrame:
    """Download market bars from Yahoo Finance, sanitize invariants, and enforce NSE hours."""
    logger.info("Downloading %s (period=%s, interval=%s)...", ticker, period, interval)
    raw = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if raw.empty:
        logger.warning("Empty response for %s", ticker)
        return pd.DataFrame()

    # Flatten multi-level columns if present (e.g. from yfinance tuple headers)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]

    df = raw.copy()
    col_map = {
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Adj Close": "Close_Adj",
        "Volume": "Volume",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Validate and clean candlestick geometric bounds
    df = CandlestickSanityValidator.sanitize(df)

    # Filter to official NSE trading hours (09:15 to 15:30 IST) for Indian assets
    if ticker.endswith(".NS") or ticker.startswith("^CNX") or ticker.startswith("^NSE"):
        gatekeeper = NSESessionGatekeeper()
        df = gatekeeper.filter_trading_hours(
            df,
            is_intraday=True,
            filter_weekends_flag=True,
            target_tz="Asia/Kolkata",
            market_open="09:15:00",
            market_close="15:30:00",
        )

    # Ensure DatetimeIndex is sorted and deduplicated
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def ingest_sector_universe() -> Dict[str, Any]:
    """Ingest, clean, and upsert all 10 Sector BEES and their benchmark indices."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    upsert_engine = IncrementalUpsertEngine(enable_snapshots=True, max_snapshots=2)

    stats: Dict[str, Any] = {}
    total_tickers = set()

    for sym, meta in SECTOR_TARGETS.items():
        total_tickers.add((sym, meta["ticker"]))
        # Also include underlying benchmark
        bench_sym = meta["benchmark"].replace("^", "").replace("=F", "")
        total_tickers.add((bench_sym, meta["benchmark"]))

    logger.info("Starting ingestion across %d unique instruments...", len(total_tickers))

    for sym, ticker_str in sorted(total_tickers, key=lambda x: x[0]):
        try:
            target_file = os.path.join(OUTPUT_DIR, f"{sym}.parquet")
            incoming_df = download_and_clean_ticker(ticker_str, period="60d", interval="5m")

            if incoming_df.empty:
                logger.warning("No data retrieved for %s (%s). Skipping.", sym, ticker_str)
                continue

            # Atomic Upsert into local Lakehouse partition
            res = upsert_engine.upsert(target_file, incoming_df, allow_schema_evolution=True)
            stats[sym] = {
                "ticker": ticker_str,
                "initial_bars": res["initial_bars"],
                "incoming_bars": res["incoming_bars"],
                "final_bars": res["final_bars"],
                "start": str(incoming_df.index[0]),
                "end": str(incoming_df.index[-1]),
            }
            logger.info(
                "Ingested %s: %d incoming -> %d final bars (%s to %s)",
                sym,
                res["incoming_bars"],
                res["final_bars"],
                incoming_df.index[0].strftime("%Y-%m-%d"),
                incoming_df.index[-1].strftime("%Y-%m-%d"),
            )
        except Exception as e:
            logger.error("Failed to ingest %s (%s): %s", sym, ticker_str, e, exc_info=True)

    # Update manifest
    manifest_data = {
        "updated_at": pd.Timestamp.now().isoformat(),
        "universe_size": len(stats),
        "sectors": SECTOR_TARGETS,
        "instruments": stats,
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    logger.info("Ingestion complete. Updated manifest written to %s", MANIFEST_PATH)
    return stats


def main() -> None:
    t0 = time.perf_counter()
    print("=" * 78)
    print("  RAVANA MULTI-SECTOR LAKEHOUSE INGESTION ENGINE (60 Days x 5m)")
    print("=" * 78)
    stats = ingest_sector_universe()
    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 78)
    print("  INGESTION SUMMARY")
    print("=" * 78)
    print(f"{'Symbol':<15} | {'Ticker':<16} | {'Final Bars':<12} | {'Date Range'}")
    print("-" * 78)
    for sym, info in sorted(stats.items()):
        dt_range = f"{info['start'][:10]} to {info['end'][:10]}"
        print(f"{sym:<15} | {info['ticker']:<16} | {info['final_bars']:<12} | {dt_range}")
    print("=" * 78)
    print(f"Ingested {len(stats)} instruments in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
