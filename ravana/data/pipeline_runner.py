"""Automated Batch Pipeline Runner for RAVANA Data Lake.

Orchestrates end-to-end ingestion (M3), cleaning (M4), resampling (M5),
feature matrix extraction (M7), overnight target building (M8), and genetic evolution (M9).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch

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
from ravana.features import FeatureConfig, FeatureMatrixEngine
from ravana.targets import TargetConfig, OvernightTargetBuilder
from ravana.data.pipeline_helpers import (
    enrich_intraday_for_m7,
    make_meta,
    sanitize_filename,
)

logger = logging.getLogger(__name__)


class BatchPipelineRunner:
    """Orchestrates end-to-end ingestion, cleaning, feature engineering, and genetic evolution."""

    def __init__(self, lake_root: Optional[Path] = None) -> None:
        self.lake_root = lake_root or Path("ravana/data/lake")
        self.m3_dir = self.lake_root / "m3_ingested"
        self.m4_dir = self.lake_root / "m4_cleaned"
        self.m5_dir = self.lake_root / "m5_resampled"
        self.m7_dir = self.lake_root / "m7_features"
        self.m8_dir = self.lake_root / "m8_targets"
        self.m9_dir = self.lake_root / "m9_alphas"
        self.storage_config = StorageConfig(self.lake_root)
        self.lake_manager = LakeManager(self.storage_config)
        self.loader = MarketDataLoader()
        self.cleaner = DataCleaningEngine(CleanerConfig.standard())
        self.resampler = ResamplingEngine(ResamplerConfig())

        # Ensure directory structures exist
        for d in [self.m3_dir, self.m4_dir, self.m5_dir, self.m7_dir, self.m8_dir, self.m9_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def ingest_and_persist_m3(self) -> Dict[str, Any]:
        """Fetch primary ETF pairs and global macro drivers and persist to Bronze Tier."""
        manifest: Dict[str, Any] = {"intraday": {}, "daily": {}, "macro": {}}

        # 1. Primary Intraday 5m ETFs + Benchmarks
        intraday_pairs = [("BANKBEES.NS", "60d"), ("NIFTYBEES.NS", "60d")]
        for ticker, period in intraday_pairs:
            ds = self.loader.ingest_etf(ticker=ticker, interval="5m", period=period, include_macro=False)
            base = sanitize_filename(ticker)
            fpath = self.m3_dir / "intraday_5m" / f"{base}.parquet"
            self.lake_manager.upsert_engine.upsert(fpath, ds.data)
            manifest["intraday"][ticker] = make_meta(ticker, ds.data, fpath)

            if ds.benchmark_data is not None and not ds.benchmark_data.empty and ds.benchmark_ticker:
                bench_base = sanitize_filename(ds.benchmark_ticker)
                bench_path = self.m3_dir / "intraday_5m" / f"{bench_base}.parquet"
                self.lake_manager.upsert_engine.upsert(bench_path, ds.benchmark_data)
                manifest["intraday"][ds.benchmark_ticker] = make_meta(
                    ds.benchmark_ticker, ds.benchmark_data, bench_path
                )

        # 2. Daily Historical Horizons (2y)
        daily_tickers = ["BANKBEES.NS", "NIFTYBEES.NS", "^NSEBANK", "^NSEI"]
        for ticker in daily_tickers:
            df = self._ingest_ticker(ticker, interval="1d", period="2y")
            base = sanitize_filename(ticker)
            fpath = self.m3_dir / "daily_1d" / f"{base}.parquet"
            self.lake_manager.upsert_engine.upsert(fpath, df)
            manifest["daily"][ticker] = make_meta(ticker, df, fpath)

        # 3. Global Macro Drivers (Daily 2y)
        macro_tickers = ["^GSPC", "^VIX", "INR=X", "^INDIAVIX"]
        for ticker in macro_tickers:
            df = self._ingest_ticker(ticker, interval="1d", period="2y")
            base = sanitize_filename(ticker)
            fpath = self.m3_dir / "daily_1d" / f"{base}.parquet"
            self.lake_manager.upsert_engine.upsert(fpath, df)
            manifest["macro"][ticker] = make_meta(ticker, df, fpath)

        manifest_path = self.m3_dir / "m3_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest

    def clean_and_persist_m4(self) -> Dict[str, Any]:
        """Clean and regularize all ingested datasets and persist to Silver Tier."""
        reports: Dict[str, Any] = {"intraday_5m": {}, "daily_1d": {}}

        # 1. Clean Intraday 5m Datasets
        for etf_t, bench_t in [("BANKBEES.NS", "^NSEBANK"), ("NIFTYBEES.NS", "^NSEI")]:
            dataset = self._load_m3_dataset(etf_t, bench_t, interval="5m")
            if dataset is not None:
                rep = self._process_and_save_m4(dataset, subfolder="intraday_5m")
                reports["intraday_5m"][etf_t] = rep

        # 2. Clean Daily 1d Datasets
        for etf_t, bench_t in [("BANKBEES.NS", "^NSEBANK"), ("NIFTYBEES.NS", "^NSEI")]:
            dataset = self._load_m3_dataset(etf_t, bench_t, interval="1d")
            if dataset is not None:
                rep = self._process_and_save_m4(dataset, subfolder="daily_1d")
                reports["daily_1d"][etf_t] = rep

        # 3. Clean Standalone Daily Macro Series
        for m_tick in ["^GSPC", "^VIX", "INR=X", "^INDIAVIX"]:
            m_base = sanitize_filename(m_tick)
            m_p = self.m3_dir / "daily_1d" / f"{m_base}.parquet"
            if m_p.exists():
                m_df = pd.read_parquet(m_p)
                m_cfg = IngestionConfig.create(ticker=m_tick, interval="1d")
                m_ds = IngestedDataset(ticker=m_tick, interval="1d", exchange=m_cfg.exchange, data=m_df)
                cleaned = self.cleaner.clean(m_ds)
                clean_p = self.m4_dir / "daily_1d" / f"{m_base}_clean.parquet"
                self.lake_manager.upsert_engine.upsert(clean_p, cleaned.data)
                reports["daily_1d"][m_tick] = {"raw_bars": len(m_df), "cleaned_bars": cleaned.bar_count}

        report_path = self.m4_dir / "m4_quality_reports.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2)

        return reports

    def resample_and_persist_m5(self) -> Dict[str, Any]:
        """Resample cleaned datasets to multi-timeframes and persist to Gold Tier."""
        manifest: Dict[str, Any] = {"intraday": {}, "macro": {}}

        # 1. Resample Intraday Pairs (15m, 30m, 1h)
        for etf_t, bench_t in [("BANKBEES.NS", "^NSEBANK"), ("NIFTYBEES.NS", "^NSEI")]:
            cleaned = self._load_m4_dataset(etf_t, bench_t, interval="5m")
            if cleaned is not None:
                manifest["intraday"][etf_t] = self._resample_and_save_etf(cleaned)

        # 2. Resample Daily Pairs (1wk, 1mo)
        for etf_t, bench_t in [("BANKBEES.NS", "^NSEBANK"), ("NIFTYBEES.NS", "^NSEI")]:
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
            meta["timeframes"][tf] = make_meta(cleaned.ticker, df, fpath)

        for tf in self.resampler.config.intraday_intervals:
            if tf in res_ds.spread_timeframes:
                s_df = res_ds.spread_timeframes[tf]
                s_path = target_dir / f"{base}_spread_{tf}.parquet"
                self.lake_manager.upsert_engine.upsert(s_path, s_df)

        return meta

    def _resample_macro_ticker(self, ticker: str) -> Dict[str, Any]:
        """Resample standalone daily macro series to weekly and monthly using M4 cleaned inputs."""
        base = sanitize_filename(ticker)
        p_clean = self.m4_dir / "daily_1d" / f"{base}_clean.parquet"
        p_raw = self.m3_dir / "daily_1d" / f"{base}.parquet"
        p_in = p_clean if p_clean.exists() else p_raw
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
            meta["timeframes"][tf] = make_meta(ticker, res_df, fpath)

        return meta

    def _load_m4_dataset(
        self, etf_ticker: str, bench_ticker: str, interval: str
    ) -> Optional[CleanedDataset]:
        """Reconstruct CleanedDataset from locally persisted M4 Silver Tier."""
        sub = "intraday_5m" if interval == "5m" else "daily_1d"
        base = sanitize_filename(etf_ticker)
        m4_path = self.m4_dir / sub / f"{base}_clean.parquet"
        if not m4_path.exists():
            return None
        df_clean = pd.read_parquet(m4_path)
        spread_path = self.m4_dir / sub / f"{base}_spread.parquet"
        df_spread = pd.read_parquet(spread_path) if spread_path.exists() else None

        bench_base = sanitize_filename(bench_ticker)
        bench_m4_path = self.m4_dir / sub / f"{bench_base}_clean.parquet"
        bench_m3_path = self.m3_dir / sub / f"{bench_base}.parquet"
        df_bench = pd.read_parquet(bench_m4_path) if bench_m4_path.exists() else (
            pd.read_parquet(bench_m3_path) if bench_m3_path.exists() else None
        )

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
            return None

        df_etf = pd.read_parquet(etf_path)
        df_bench = pd.read_parquet(bench_path) if bench_path.exists() else None

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
        """Clean an IngestedDataset and save clean data, spread data, and clean macros."""
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

        # Save Clean Macro drivers if present
        if cleaned.macro_data:
            for m_tick, m_df in cleaned.macro_data.items():
                m_base = sanitize_filename(m_tick)
                m_clean_path = self.m4_dir / "daily_1d" / f"{m_base}_clean.parquet"
                self.lake_manager.upsert_engine.upsert(m_clean_path, m_df)

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

    def verify_lake(self) -> Dict[str, Any]:
        """Audit all persisted lake tiers and verify complete mathematical invariants."""
        audit: Dict[str, Any] = {"all_passed": True}
        tiers = [
            ("m3_files", self.m3_dir),
            ("m4_files", self.m4_dir),
            ("m5_files", self.m5_dir),
            ("m7_files", self.m7_dir),
            ("m8_files", self.m8_dir),
        ]
        for tier_name, tier_dir in tiers:
            audit[tier_name] = {}
            for p in tier_dir.rglob("*.parquet"):
                df = pd.read_parquet(p)
                passed = bool((not df.isna().any().any()) and (not df.isin([float("inf"), float("-inf")]).any().any()))
                if not passed:
                    audit["all_passed"] = False
                key = f"{p.parent.name}/{p.name}" if tier_name == "m5_files" else p.name
                audit[tier_name][key] = {"bars": len(df), "columns": len(df.columns), "passed": passed}
        return audit

    def compute_and_persist_m7(self) -> Dict[str, Any]:
        """Compute 21 stationary features for ETFs and multi-asset 3D tensor."""
        manifest: Dict[str, Any] = {"features": {}}
        engine = FeatureMatrixEngine(FeatureConfig())
        etf_dfs: Dict[str, pd.DataFrame] = {}

        for ticker in ["BANKBEES.NS", "NIFTYBEES.NS"]:
            base = sanitize_filename(ticker)
            clean_path = self.m4_dir / "intraday_5m" / f"{base}_clean.parquet"
            if not clean_path.exists():
                continue
            df_clean = pd.read_parquet(clean_path)
            df_enriched = enrich_intraday_for_m7(base, df_clean, self.m4_dir)
            feat_df = engine.compute_feature_matrix(df_enriched, symbol=base)
            out_path = self.m7_dir / "intraday_5m" / f"{base}_features.parquet"
            self.lake_manager.upsert_engine.upsert(out_path, feat_df)
            manifest["features"][base] = make_meta(ticker, feat_df, out_path)
            etf_dfs[base] = df_enriched

        if len(etf_dfs) >= 2:
            tensor, symbols, feature_names, time_index = engine.compute_multi_asset_tensor(etf_dfs)
            tensor_path = self.m7_dir / "multi_asset_features_3d.pt"
            torch.save(
                {"tensor": tensor, "symbols": symbols, "features": feature_names, "time_index": time_index},
                tensor_path,
            )
            manifest["tensor_3d"] = {"shape": list(tensor.shape), "path": str(tensor_path)}

        return manifest

    def compute_and_persist_m8(self) -> Dict[str, Any]:
        """Compute forward overnight economic targets and statutory delivery frictions."""
        manifest: Dict[str, Any] = {"targets": {}}
        builder = OvernightTargetBuilder(TargetConfig(capital_inr=200000.0))

        for ticker in ["BANKBEES.NS", "NIFTYBEES.NS"]:
            base = sanitize_filename(ticker)
            clean_path = self.m4_dir / "intraday_5m" / f"{base}_clean.parquet"
            if not clean_path.exists():
                continue
            df_clean = pd.read_parquet(clean_path)
            target_df = builder.build_targets(df_clean)
            out_path = self.m8_dir / "intraday_5m" / f"{base}_targets.parquet"
            self.lake_manager.upsert_engine.upsert(out_path, target_df)
            manifest["targets"][base] = make_meta(ticker, target_df, out_path)

        return manifest

    def compute_and_persist_m9(self) -> Dict[str, Any]:
        """Evolve predictive alpha formulas (Muscle 9) using M7 features and M8 targets."""
        manifest: Dict[str, Any] = {"champions": {}}
        from ravana.evolution import EvolutionConfig, GeneticAlphaEngine

        for ticker in ["BANKBEES.NS", "NIFTYBEES.NS"]:
            base = sanitize_filename(ticker)
            f_path = self.m7_dir / "intraday_5m" / f"{base}_features.parquet"
            t_path = self.m8_dir / "intraday_5m" / f"{base}_targets.parquet"
            if not f_path.exists() or not t_path.exists():
                continue
            f_df = pd.read_parquet(f_path)
            t_df = pd.read_parquet(t_path)

            cfg = EvolutionConfig(population_size=32, max_generations=3, random_seed=42)
            engine = GeneticAlphaEngine(cfg)
            res = engine.evolve(f_df, t_df)

            champ = res.champion or res.best_individual
            manifest["champions"][base] = {
                "formula": champ.formula_str() if champ else "NONE",
                "ic": champ.ic if champ else 0.0,
                "fitness": champ.fitness if champ else 0.0,
                "graveyard_size": res.graveyard_count,
            }

            champ_path = self.m9_dir / f"{base}_champion.json"
            grave_path = self.m9_dir / f"{base}_graveyard.json"
            with open(champ_path, "w", encoding="utf-8") as f:
                json.dump(manifest["champions"][base], f, indent=2)
            with open(grave_path, "w", encoding="utf-8") as f:
                json.dump(engine.graveyard.get_records(), f, indent=2)

        return manifest

    def execute_all(self) -> Dict[str, Any]:
        """Execute full end-to-end ingestion, cleaning, feature engineering, and genetic evolution."""
        logger.info("Executing Muscle 3 Ingestion...")
        m3_res = self.ingest_and_persist_m3()
        logger.info("Executing Muscle 4 Cleaning...")
        m4_res = self.clean_and_persist_m4()
        logger.info("Executing Muscle 5 Resampling...")
        m5_res = self.resample_and_persist_m5()
        logger.info("Executing Muscle 7 Feature Matrix Extraction...")
        m7_res = self.compute_and_persist_m7()
        logger.info("Executing Muscle 8 Overnight Target Building...")
        m8_res = self.compute_and_persist_m8()
        logger.info("Executing Muscle 9 Genetic Alpha Evolution...")
        m9_res = self.compute_and_persist_m9()
        logger.info("Auditing Lake Integrity...")
        audit = self.verify_lake()
        return {
            "m3_manifest": m3_res,
            "m4_reports": m4_res,
            "m5_manifest": m5_res,
            "m7_manifest": m7_res,
            "m8_manifest": m8_res,
            "m9_manifest": m9_res,
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
