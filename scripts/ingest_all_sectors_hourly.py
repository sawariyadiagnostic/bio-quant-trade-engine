"""
Ingest 6 months of 1-Hour (1H) intraday bars across all 10 Sector BEES ETFs and benchmarks.
Stores partitions into ravana/data/lake/m3_ingested/intraday_1h/ with incremental upsert.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Dict
import pandas as pd
import yfinance as yf

# UTF-8 Console Safety
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ravana.data.ingestion.loader import CandlestickSanityValidator
from ravana.data.storage.upsert import IncrementalUpsertEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest_hourly")

HOURLY_TARGETS: Dict[str, str] = {
    "BANKBEES": "BANKBEES.NS",
    "NIFTYBEES": "NIFTYBEES.NS",
    "ITBEES": "ITBEES.NS",
    "AUTOBEES": "AUTOBEES.NS",
    "PHARMABEES": "PHARMABEES.NS",
    "PSUBNKBEES": "PSUBNKBEES.NS",
    "JUNIORBEES": "JUNIORBEES.NS",
    "CPSEETF": "CPSEETF.NS",
    "GOLDBEES": "GOLDBEES.NS",
    "SILVERBEES": "SILVERBEES.NS",
    "NSEBANK": "^NSEBANK",
    "NSEI": "^NSEI",
    "CNXIT": "^CNXIT",
    "CNXAUTO": "^CNXAUTO",
    "CNXPHARMA": "^CNXPHARMA",
    "CNXPSUBANK": "^CNXPSUBANK",
}

OUTPUT_DIR = os.path.join(BASE_DIR, "ravana", "data", "lake", "m3_ingested", "intraday_1h")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    engine = IncrementalUpsertEngine()
    print("=" * 80)
    print("  INGESTING 6 MONTHS OF 1-HOUR (1H) INTRADAY BARS ACROSS ALL SECTORS")
    print("=" * 80)

    for sym, ticker in HOURLY_TARGETS.items():
        target_path = os.path.join(OUTPUT_DIR, f"{sym}.parquet")
        try:
            print(f"Fetching {sym} ({ticker}) 1h data...")
            raw = yf.download(ticker, period="6mo", interval="1h", progress=False, auto_adjust=False)
            if raw.empty:
                print(f"  [WARN] No data returned for {ticker}")
                continue

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
            df = CandlestickSanityValidator.sanitize(df)

            if df.index.tz is None:
                df.index = df.index.tz_localize("Asia/Kolkata")
            else:
                df.index = df.index.tz_convert("Asia/Kolkata")

            res = engine.upsert(target_path, df, allow_schema_evolution=True)
            print(f"  [OK] {sym}: {res['final_bars']} bars (+{res['incoming_bars'] - res['collisions']} new)")
        except Exception as e:
            print(f"  [ERROR] Failed to ingest {sym}: {e}")

    print("=" * 80)
    print("  HOURLY (1H) LAKEHOUSE INGESTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
