"""Automated Batch Pipeline Runner for RAVANA Data Lake.
Ingests full market universes (Muscle 3) -> Persists to lake/m3_ingested/
Cleans, regularizes, and aligns (Muscle 4) -> Persists to lake/m4_cleaned/
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from ravana.data.cleaner import CleanerConfig, CleanedDataset, DataCleaningEngine
from ravana.data.resampler import (
    ResamplerConfig,
    ResampledDataset,
    ResamplingEngine,
    SessionAwareResampler,
)
from ravana.data.storage import LakeManager, StorageConfig
from ravana.data.ingestion import (
    ETF_BENCHMARK_MAP,
    ExchangeType,
    IngestedDataset,
    IngestionConfig,
    MarketDataLoader,
)

logger = logging.getLogger(__name__)


def sanitize_filename(ticker: str) -> str:
    """Normalize ticker string into a filesystem-safe file basename."""
    clean = ticker.replace("^", "").replace(".NS", "").replace("=X", "").replace("=F", "")
    if ticker.upper() == "INR=X":
        return "USDINR"
    return clean


class BatchPipelineRunner:
    """Orchestrates end-to-end ingestion, cleaning, and persistent lake storage."""

    def __init__(self, lake_root: Optional[Path] = None) -> None:
        self.lake_root = lake_root or Path("ravana/data/lake")
        self.m3_dir = self.lake_root / "m3_ingested"
        self.m4_dir = self.lake_root / "m4_cleaned"
        self.m5_dir = self.lake_root / "m5_resampled"
        self.storage_config = StorageConfig(self.lake_root)
        self.lake_manager = LakeManager(self.storage_config)
        self.loader = MarketDataLoader()
        self.cleaner = DataCleaningEngine(CleanerConfig.standard())
        self.resampler = ResamplingEngine(ResamplerConfig())

        # Ensure directory structures exist
        for d in [
            self.m3_dir / "intraday_5m",
            self.m3_dir / "daily_1d",
            self.m4_dir / "intraday_5m",
            self.m4_dir / "daily_1d",
            self.m5_dir / "intraday_15m",
            self.m5_dir / "intraday_30m",
            self.m5_dir / "intraday_1h",
            self.m5_dir / "macro_1wk",
            self.m5_dir / "macro_1mo",
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def ingest_and_persist_m3(self) -> Dict[str, Any]:
        """Ingest complete multi-asset universes across intraday and daily horizons."""
        manifest: Dict[str, Any] = {"intraday_5m": {}, "daily_1d": {}}

        # 1. Intraday 5m universe (60 days)
        intraday_tickers = ["BANKBEES.NS", "NIFTYBEES.NS", "^NSEBANK", "^NSEI"]
        for ticker in intraday_tickers:
            df = self._ingest_ticker(ticker, interval="5m", period="60d")
            if not df.empty:
                fname = f"{sanitize_filename(ticker)}.parquet"
                fpath = self.m3_dir / "intraday_5m" / fname
                self.lake_manager.upsert_engine.upsert(fpath, df)
                manifest["intraday_5m"][ticker] = self._make_meta(ticker, df, fpath)

        # 2. Daily 1d universe (Full max history: 2008–present)
        daily_tickers = [
            "BANKBEES.NS", "NIFTYBEES.NS", "^NSEBANK", "^NSEI",
            "^GSPC", "^VIX", "INR=X", "^INDIAVIX"
        ]
        for ticker in daily_tickers:
            df = self._ingest_ticker(ticker, interval="1d", period="max")
            if not df.empty:
                fname = f"{sanitize_filename(ticker)}.parquet"
                fpath = self.m3_dir / "daily_1d" / fname
                self.lake_manager.upsert_engine.upsert(fpath, df)
                manifest["daily_1d"][ticker] = self._make_meta(ticker, df, fpath)

        # Write manifest
        manifest_path = self.m3_dir / "m3_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest

    def clean_and_persist_m4(self) -> Dict[str, Any]:
        """Clean and regularize all ingested datasets and persist to Silver Tier."""
        reports: Dict[str, Any] = {"intraday_5m": {}, "daily_1d": {}}

        # 1. Clean Intraday 5m Datasets
        intraday_pairs = [("BANKBEES.NS", "^NSEBANK"), ("NIFTYBEES.NS", "^NSEI")]
        for etf_t, bench_t in intraday_pairs:
            dataset = self._load_m3_dataset(etf_t, bench_t, interval="5m")
            if dataset is not None:
                rep = self._process_and_save_m4(dataset, subfolder="intraday_5m")
                reports["intraday_5m"][etf_t] = rep

        # 2. Clean Daily 1d Datasets
        daily_pairs = [("BANKBEES.NS", "^NSEBANK"), ("NIFTYBEES.NS", "^NSEI")]
        for etf_t, bench_t in daily_pairs:
            dataset = self._load_m3_dataset(etf_t, bench_t, interval="1d")
            if dataset is not None:
                rep = self._process_and_save_m4(dataset, subfolder="daily_1d")
                reports["daily_1d"][etf_t] = rep

        # Write quality reports
        report_path = self.m4_dir / "m4_quality_reports.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2)

        return reports

    def resample_and_persist_m5(self) -> Dict[str, Any]:
        """Resample cleaned datasets to multi-timeframes and persist to Gold Tier."""
        manifest: Dict[str, Any] = {"intraday": {}, "macro": {}}

        # 1. Resample Intraday Pairs (15m, 30m, 1h)
        intraday_pairs = [("BANKBEES.NS", "^NSEBANK"), ("NIFTYBEES.NS", "^NSEI")]
        for etf_t, bench_t in intraday_pairs:
            cleaned = self._load_m4_dataset(etf_t, bench_t, interval="5m")
            if cleaned is not None:
                manifest["intraday"][etf_t] = self._resample_and_save_etf(cleaned)

        # 2. Resample Daily Pairs (1wk, 1mo)
        daily_pairs = [("BANKBEES.NS", "^NSEBANK"), ("NIFTYBEES.NS", "^NSEI")]
        for etf_t, bench_t in daily_pairs:
            cleaned = self._load_m4_dataset(etf_t, bench_t, interval="1d")
            if cleaned is not None:
                manifest["macro"][etf_t] = self._resample_and_save_etf(cleaned)

        # 3. Resample Macro Drivers (1wk, 1mo)
        for m_tick in ["^GSPC", "^VIX", "INR=X", "^INDIAVIX"]:
            manifest["macro"][m_tick] = self._resample_macro_ticker(m_tick)

        manifest_path = self.m5_dir / "m5_resampled_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest

    def _resample_and_save_etf(self, cleaned: CleanedDataset) -> Dict[str, Any]:
        """Resample a single CleanedDataset and persist all derived timeframes."""
        res_ds = self.resampler.resample_dataset(cleaned)
        base = sanitize_filename(cleaned.ticker)
        meta: Dict[str, Any] = {"ticker": cleaned.ticker, "timeframes": {}}

        for tf, df in res_ds.timeframes.items():
            if tf == cleaned.interval:
                continue
            is_intra = IngestionConfig.is_interval_intraday(tf)
            subfolder = f"intraday_{tf}" if is_intra else f"macro_{tf}"
            target_dir = self.m5_dir / subfolder
            target_dir.mkdir(parents=True, exist_ok=True)

            fpath = target_dir / f"{base}_{tf}.parquet"
            self.lake_manager.upsert_engine.upsert(fpath, df)
            meta["timeframes"][tf] = self._make_meta(cleaned.ticker, df, fpath)

            if cleaned.benchmark_ticker and tf in res_ds.benchmark_timeframes:
                b_df = res_ds.benchmark_timeframes[tf]
                b_base = sanitize_filename(cleaned.benchmark_ticker)
                b_path = target_dir / f"{b_base}_{tf}.parquet"
                self.lake_manager.upsert_engine.upsert(b_path, b_df)

            if tf in res_ds.spread_timeframes:
                s_df = res_ds.spread_timeframes[tf]
                s_path = target_dir / f"{base}_spread_{tf}.parquet"
                self.lake_manager.upsert_engine.upsert(s_path, s_df)

        return meta

    def _resample_macro_ticker(self, ticker: str) -> Dict[str, Any]:
        """Resample standalone daily macro series to weekly and monthly."""
        base = sanitize_filename(ticker)
        p_in = self.m3_dir / "daily_1d" / f"{base}.parquet"
        if not p_in.exists():
            return {}
        df = pd.read_parquet(p_in)
        meta: Dict[str, Any] = {"ticker": ticker, "timeframes": {}}

        for tf in self.resampler.config.daily_intervals:
            freq = self.resampler.config.parse_daily_to_freq(tf)
            res_df = SessionAwareResampler.resample_daily_to_higher(
                df, freq=freq, require_closed_period=self.resampler.config.require_closed_period
            )
            fpath = self.m5_dir / f"macro_{tf}" / f"{base}_{tf}.parquet"
            self.lake_manager.upsert_engine.upsert(fpath, res_df)
            meta["timeframes"][tf] = self._make_meta(ticker, res_df, fpath)

        return meta

    def _load_m4_dataset(
        self, etf_ticker: str, bench_ticker: str, interval: str
    ) -> Optional[CleanedDataset]:
        """Reconstruct CleanedDataset from locally persisted M4 Silver Tier."""
        sub = "intraday_5m" if interval == "5m" else "daily_1d"
        base = sanitize_filename(etf_ticker)
        clean_path = self.m4_dir / sub / f"{base}_clean.parquet"
        if not clean_path.exists():
            return None
        df_clean = pd.read_parquet(clean_path)

        spread_path = self.m4_dir / sub / f"{base}_spread.parquet"
        df_spread = pd.read_parquet(spread_path) if spread_path.exists() else None

        bench_base = sanitize_filename(bench_ticker)
        bench_m4_path = self.m4_dir / sub / f"{bench_base}_clean.parquet"
        bench_m3_path = self.m3_dir / sub / f"{bench_base}.parquet"
        if bench_m4_path.exists():
            df_bench = pd.read_parquet(bench_m4_path)
        elif bench_m3_path.exists():
            df_bench = pd.read_parquet(bench_m3_path)
        else:
            df_bench = None

        cfg = IngestionConfig.create(ticker=etf_ticker, interval=interval)
        return CleanedDataset(
            ticker=etf_ticker,
            interval=interval,
            exchange=cfg.exchange,
            data=df_clean,
            benchmark_ticker=bench_ticker,
            benchmark_data=df_bench,
            spread_data=df_spread,
            quality_report={"source": "lake_m4"},
        )

    def _ingest_ticker(self, ticker: str, interval: str, period: str) -> pd.DataFrame:
        """Fetch ticker with automatic configuration."""
        cfg = IngestionConfig.create(ticker=ticker, interval=interval, period=period)
        return self.loader.ingest(cfg)

    def _load_m3_dataset(
        self, etf_ticker: str, bench_ticker: str, interval: str
    ) -> Optional[IngestedDataset]:
        """Construct IngestedDataset from locally stored M3 Parquet files."""
        subfolder = "intraday_5m" if interval == "5m" else "daily_1d"
        etf_path = self.m3_dir / subfolder / f"{sanitize_filename(etf_ticker)}.parquet"
        bench_path = self.m3_dir / subfolder / f"{sanitize_filename(bench_ticker)}.parquet"

        if not etf_path.exists():
            logger.warning("ETF file not found in M3: %s", etf_path)
            return None

        df_etf = pd.read_parquet(etf_path)
        df_bench = pd.read_parquet(bench_path) if bench_path.exists() else None

        # Load macro drivers for daily datasets
        macro_dict: Dict[str, pd.DataFrame] = {}
        if interval == "1d":
            for m_tick in ["^GSPC", "^VIX", "INR=X"]:
                m_path = self.m3_dir / "daily_1d" / f"{sanitize_filename(m_tick)}.parquet"
                if m_path.exists():
                    macro_dict[m_tick] = pd.read_parquet(m_path)

        cfg = IngestionConfig.create(ticker=etf_ticker, interval=interval)
        return IngestedDataset(
            ticker=etf_ticker,
            interval=interval,
            exchange=cfg.exchange,
            data=df_etf,
            benchmark_ticker=bench_ticker,
            benchmark_data=df_bench,
            macro_data=macro_dict,
            metadata={"source": "lake_m3"},
        )

    def _process_and_save_m4(
        self, dataset: IngestedDataset, subfolder: str
    ) -> Dict[str, Any]:
        """Clean an IngestedDataset and save clean data and spread data."""
        cleaned = self.cleaner.clean(dataset)
        base = sanitize_filename(dataset.ticker)

        # Save Clean Primary
        clean_path = self.m4_dir / subfolder / f"{base}_clean.parquet"
        self.lake_manager.upsert_engine.upsert(clean_path, cleaned.data)

        # Save Tracking Spread Analytics
        if cleaned.spread_data is not None and not cleaned.spread_data.empty:
            spread_path = self.m4_dir / subfolder / f"{base}_spread.parquet"
            self.lake_manager.upsert_engine.upsert(spread_path, cleaned.spread_data)

        # Save Clean Benchmark if present
        if cleaned.benchmark_data is not None and not cleaned.benchmark_data.empty and dataset.benchmark_ticker:
            bench_base = sanitize_filename(dataset.benchmark_ticker)
            bench_clean_path = self.m4_dir / subfolder / f"{bench_base}_clean.parquet"
            self.lake_manager.upsert_engine.upsert(bench_clean_path, cleaned.benchmark_data)

        return {
            "ticker": dataset.ticker,
            "interval": dataset.interval,
            "raw_bars": len(dataset.data),
            "cleaned_bars": cleaned.bar_count,
            "synthetic_bars": cleaned.synthetic_bar_count,
            "spikes_flagged": cleaned.spike_count,
            "zero_nans": not cleaned.data.isna().any().any(),
            "quality_metrics": cleaned.quality_report,
        }

    def _make_meta(self, ticker: str, df: pd.DataFrame, fpath: Path) -> Dict[str, Any]:
        """Construct audit metadata dictionary for a stored parquet file."""
        return {
            "ticker": ticker,
            "bars": len(df),
            "start": str(df.index[0]) if not df.empty else None,
            "end": str(df.index[-1]) if not df.empty else None,
            "file_size_bytes": fpath.stat().st_size,
        }

    def verify_lake(self) -> Dict[str, Any]:
        """Audit all persisted lake tiers and verify complete mathematical invariants."""
        audit: Dict[str, Any] = {"m3_files": {}, "m4_files": {}, "m5_files": {}, "all_passed": True}

        for p in self.m3_dir.rglob("*.parquet"):
            df = pd.read_parquet(p)
            audit["m3_files"][p.name] = {"bars": len(df), "columns": list(df.columns)}

        for p in self.m4_dir.rglob("*.parquet"):
            df = pd.read_parquet(p)
            has_nan = bool(df.isna().any().any())
            has_inf = bool(df.isin([float("inf"), float("-inf")]).any().any())
            passed = (not has_nan) and (not has_inf)
            if not passed:
                audit["all_passed"] = False
            audit["m4_files"][p.name] = {
                "bars": len(df),
                "has_nans": has_nan,
                "has_infs": has_inf,
                "passed": passed,
            }

        for p in self.m5_dir.rglob("*.parquet"):
            df = pd.read_parquet(p)
            has_nan = bool(df.isna().any().any())
            has_inf = bool(df.isin([float("inf"), float("-inf")]).any().any())
            passed = (not has_nan) and (not has_inf)
            if not passed:
                audit["all_passed"] = False
            audit["m5_files"][f"{p.parent.name}/{p.name}"] = {
                "bars": len(df),
                "has_nans": has_nan,
                "has_infs": has_inf,
                "passed": passed,
            }

        return audit

    def execute_all(self) -> Dict[str, Any]:
        """Execute full end-to-end ingestion, cleaning, and persistence pipeline."""
        logger.info("Executing Muscle 3 Ingestion...")
        m3_res = self.ingest_and_persist_m3()
        logger.info("Executing Muscle 4 Cleaning...")
        m4_res = self.clean_and_persist_m4()
        logger.info("Executing Muscle 5 Resampling...")
        m5_res = self.resample_and_persist_m5()
        logger.info("Auditing Lake Integrity...")
        audit = self.verify_lake()
        return {
            "m3_manifest": m3_res,
            "m4_reports": m4_res,
            "m5_manifest": m5_res,
            "audit": audit,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    runner = BatchPipelineRunner()
    results = runner.execute_all()
    print("\n" + "=" * 80)
    print("PIPELINE RUN COMPLETE — LAKE AUDIT SUMMARY:")
    print("=" * 80)
    print(json.dumps(results["audit"], indent=2))
