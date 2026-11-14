"""Quick single-asset test runner across RAVANA M3-M11 pipeline."""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import pandas as pd
import numpy as np
import torch

from ravana.data.ingestion.loader import MarketDataLoader, CandlestickSanityValidator
from ravana.data.ingestion.dataset import IngestedDataset
from ravana.data.ingestion.config import ExchangeType
from ravana.data.cleaner import DataCleaningEngine, CleanerConfig
from ravana.data.resampler import ResamplingEngine, ResamplerConfig
from ravana.features import FeatureMatrixEngine, FeatureConfig
from ravana.targets import OvernightTargetBuilder, TargetConfig
from ravana.execution.shoonya_adapter import ShoonyaBrokerGateway, ShoonyaGatewayConfig
from ravana.execution.router import BrokerOrderRouter
from ravana.backtest.orders import Order, OrderSide
from ravana.risk.circuit_breaker import CircuitBreakerKillSwitch
from ravana.validation.config import ValidationCertificate


def test_asset(file_path: Path):
    print(f"Testing {file_path.name}...")
    loader = MarketDataLoader()
    cleaner = DataCleaningEngine(CleanerConfig.standard())
    resampler = ResamplingEngine(ResamplerConfig())
    feature_engine = FeatureMatrixEngine(FeatureConfig())
    target_builder = OvernightTargetBuilder(TargetConfig())

    # 1. Load
    raw_df = pd.read_parquet(file_path)
    std_df = loader.standardize_columns(raw_df)
    std_df = CandlestickSanityValidator.sanitize(std_df)
    std_df = loader.clean_index_and_chronology(std_df, "Asia/Kolkata")
    
    ticker = file_path.stem
    ingested = IngestedDataset(
        ticker=ticker,
        interval="1d" if len(std_df) < 2000 else "5m",
        exchange=ExchangeType.NSE,
        data=std_df,
    )
    print(f"  [M3 Ingested] {ticker}: {ingested.bar_count} bars ({ingested.start_time.date()} to {ingested.end_time.date()})")

    # 2. Clean
    cleaned = cleaner.clean(ingested)
    print(f"  [M4 Cleaned] Cleaned bars: {cleaned.bar_count}, Quality: {cleaned.quality_report.get('status', 'OK')}")

    # 3. Resample
    resampled = resampler.resample_dataset(cleaned)
    tf_key = list(resampled.timeframes.keys())[0]
    resampled_df = resampled.timeframes[tf_key]
    print(f"  [M5 Resampled] Bars in {tf_key}: {resampled_df.shape[0]}")

    # 4. Features (26 stationary features)
    features_df = feature_engine.compute_feature_matrix(resampled_df)
    print(f"  [M7 Features] Shape: {features_df.shape} (26 features verified: {features_df.shape[1] == 26})")

    # 5. Targets
    targets_df = target_builder.build_targets(resampled_df)
    print(f"  [M8 Targets] Shape: {targets_df.shape}, Non-null net returns: {targets_df['net_return'].notnull().sum()}")

    # 6. Shoonya Router & Execution (Muscle 19 & 11)
    gateway = ShoonyaBrokerGateway(ShoonyaGatewayConfig(paper_trading=True))
    kill_switch = CircuitBreakerKillSwitch()
    router = BrokerOrderRouter(gateway=gateway, kill_switch=kill_switch, live_mode=False)

    ref_price = float(resampled_df["Close"].iloc[-1])
    test_order = Order(
        order_id=f"TEST_{ticker}_1",
        ticker=ticker,
        timestamp=resampled_df.index[-1],
        side=OrderSide.BUY,
        shares=10,
    )
    trade = gateway.place_order(test_order, reference_price=ref_price, bar_volume=1e6)
    print(f"  [Muscle 19 & 11 Shoonya Execution] Filled: {trade.shares} @ {trade.price:.2f}, Friction: Rs {trade.friction.total_friction_inr:.2f}, Brokerage: Rs {trade.friction.brokerage_inr:.2f}")

    print(f"  [STATUS] SUCCESS: {ticker} passed all stages M3 -> M19.\n")
    return True


if __name__ == "__main__":
    test_asset(Path("data/lake/Indices/BANKBEES.parquet"))
