"""Multi-Sector Automated Batch Pipeline Runner for RAVANA Data Lake.

Processes raw parquet data from data/lake/<Sector>/ into:
- M3 IngestedDataset
- M4 CleanedDataset (causal MAD outlier filtering)
- M5 ResampledDataset (session-aware right-labeled bars)
- M6 Lakehouse Bronze/Silver/Gold storage in ravana/data/lake/
- M7 Feature Matrix (26 stationary features)
- M8 Multi-Horizon Targets
- Muscle 19 & 11 Shoonya execution simulation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ravana.data.ingestion.config import ExchangeType
from ravana.data.ingestion.dataset import IngestedDataset
from ravana.data.ingestion.loader import MarketDataLoader, CandlestickSanityValidator
from ravana.data.cleaner import DataCleaningEngine, CleanerConfig
from ravana.data.resampler import ResamplingEngine, ResamplerConfig
from ravana.data.storage import LakeManager, StorageConfig
from ravana.features import FeatureMatrixEngine, FeatureConfig
from ravana.targets import OvernightTargetBuilder, TargetConfig
from ravana.execution.shoonya_adapter import ShoonyaBrokerGateway, ShoonyaGatewayConfig
from ravana.backtest.orders import Order, OrderSide


def process_single_asset(
    file_path: Path,
    sector_name: str,
    output_root: Path,
    loader: MarketDataLoader,
    cleaner: DataCleaningEngine,
    resampler: ResamplingEngine,
    feature_engine: FeatureMatrixEngine,
    target_builder: OvernightTargetBuilder,
    gateway: ShoonyaBrokerGateway,
    lake_manager: LakeManager,
) -> Dict[str, Any]:
    """Processes a single parquet asset through M3 -> M19."""
    ticker = file_path.stem
    t0 = time.time()
    result: Dict[str, Any] = {"ticker": ticker, "sector": sector_name, "status": "FAILED"}

    try:
        # 1. Load & Normalize
        raw_df = pd.read_parquet(file_path)
        std_df = loader.standardize_columns(raw_df)
        std_df = CandlestickSanityValidator.sanitize(std_df)
        std_df = loader.clean_index_and_chronology(std_df, "Asia/Kolkata")

        interval = "1d" if len(std_df) < 2000 else "5m"
        ingested = IngestedDataset(
            ticker=ticker,
            interval=interval,
            exchange=ExchangeType.NSE,
            data=std_df,
        )

        # 2. Clean (M4)
        cleaned = cleaner.clean(ingested)

        # 3. Resample (M5)
        resampled = resampler.resample_dataset(cleaned)
        tf_key = list(resampled.timeframes.keys())[0]
        resampled_df = resampled.timeframes[tf_key]

        # 4. Features (M7 - 26 stationary features)
        features_df = feature_engine.compute_feature_matrix(resampled_df)

        # 5. Targets (M8)
        targets_df = target_builder.build_targets(resampled_df)

        # 6. Storage (M6 Lakehouse)
        dest_features_dir = output_root / "m7_features" / sector_name
        dest_features_dir.mkdir(parents=True, exist_ok=True)
        feat_path = dest_features_dir / f"{ticker}_features.parquet"
        features_df.to_parquet(feat_path)

        dest_targets_dir = output_root / "m8_targets" / sector_name
        dest_targets_dir.mkdir(parents=True, exist_ok=True)
        tgt_path = dest_targets_dir / f"{ticker}_targets.parquet"
        targets_df.to_parquet(tgt_path)

        # 7. Execution Simulation (Muscle 19 & 11)
        ref_price = float(resampled_df["Close"].iloc[-1])
        test_order = Order(
            order_id=f"BATCH_{sector_name}_{ticker}",
            ticker=ticker,
            timestamp=resampled_df.index[-1],
            side=OrderSide.BUY,
            shares=10,
        )
        trade = gateway.place_order(test_order, reference_price=ref_price, bar_volume=1e6)

        elapsed = round(time.time() - t0, 3)
        result.update({
            "status": "PASSED",
            "raw_bars": len(raw_df),
            "cleaned_bars": cleaned.bar_count,
            "resampled_bars": len(resampled_df),
            "feature_columns": features_df.shape[1],
            "target_rows": len(targets_df),
            "execution_price": trade.price if trade else 0.0,
            "statutory_friction_inr": trade.friction.total_friction_inr if trade else 0.0,
            "brokerage_inr": trade.friction.brokerage_inr if trade else 0.0,
            "elapsed_seconds": elapsed,
        })
        print(f"[{sector_name}] {ticker}: OK ({cleaned.bar_count} bars -> {features_df.shape[1]} feats -> Trade executed in {elapsed}s)")

    except Exception as e:
        result["error"] = str(e)
        print(f"[{sector_name}] {ticker}: ERROR - {e}")

    return result


def run_cohort(sector_folders: List[str], cohort_name: str, lake_in_dir: Path, output_root: Path):
    """Processes all parquet files in specified sector folders."""
    print(f"\n==========================================")
    print(f"STARTING COHORT: {cohort_name}")
    print(f"Sectors: {', '.join(sector_folders)}")
    print(f"==========================================")

    loader = MarketDataLoader()
    cleaner = DataCleaningEngine(CleanerConfig.standard())
    resampler = ResamplingEngine(ResamplerConfig())
    feature_engine = FeatureMatrixEngine(FeatureConfig())
    target_builder = OvernightTargetBuilder(TargetConfig())
    gateway = ShoonyaBrokerGateway(ShoonyaGatewayConfig(paper_trading=True))
    lake_manager = LakeManager(StorageConfig(output_root))

    results = []
    total_assets = 0
    passed_assets = 0

    for sector in sector_folders:
        sector_dir = lake_in_dir / sector
        if not sector_dir.exists():
            print(f"Warning: Sector folder {sector_dir} does not exist. Skipping.")
            continue

        parquet_files = sorted(sector_dir.glob("*.parquet"))
        print(f"\nProcessing sector '{sector}' ({len(parquet_files)} assets)...")

        for fpath in parquet_files:
            total_assets += 1
            res = process_single_asset(
                file_path=fpath,
                sector_name=sector,
                output_root=output_root,
                loader=loader,
                cleaner=cleaner,
                resampler=resampler,
                feature_engine=feature_engine,
                target_builder=target_builder,
                gateway=gateway,
                lake_manager=lake_manager,
            )
            results.append(res)
            if res["status"] == "PASSED":
                passed_assets += 1

    summary_file = output_root / f"cohort_{cohort_name}_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({"cohort": cohort_name, "total": total_assets, "passed": passed_assets, "assets": results}, f, indent=2)

    print(f"\n==========================================")
    print(f"COHORT {cohort_name} COMPLETE: {passed_assets}/{total_assets} PASSED")
    print(f"Summary written to: {summary_file}")
    print(f"==========================================\n")
    return passed_assets, total_assets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Sector Cohorts in RAVANA Lakehouse")
    parser.add_argument("--cohort", required=True, help="Cohort identifier name")
    parser.add_argument("--sectors", required=True, nargs="+", help="List of sector folder names")
    parser.add_argument("--lake-in", default="data/lake", help="Input lake directory")
    parser.add_argument("--lake-out", default="ravana/data/lake", help="Output lake directory")

    args = parser.parse_args()
    run_cohort(
        sector_folders=args.sectors,
        cohort_name=args.cohort,
        lake_in_dir=Path(args.lake_in),
        output_root=Path(args.lake_out),
    )
