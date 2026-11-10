"""Muscle 3: Market Data Ingestion & Session Gatekeeper.
Universal multi-asset feed connectors with session gatekeeping, physical invariant validation,
and strongly-typed IngestedDataset contract.
"""
from ravana.data.ingestion.config import (
    IngestionConfig,
    ExchangeType,
    CORE_ETF_UNIVERSE,
    GLOBAL_MACRO_UNIVERSE,
    DOMESTIC_MACRO_UNIVERSE,
    ETF_BENCHMARK_MAP,
)
from ravana.data.ingestion.session_gatekeeper import NSESessionGatekeeper
from ravana.data.ingestion.loader import MarketDataLoader, CandlestickSanityValidator
from ravana.data.ingestion.dataset import IngestedDataset

__all__ = [
    "IngestionConfig",
    "ExchangeType",
    "CORE_ETF_UNIVERSE",
    "GLOBAL_MACRO_UNIVERSE",
    "DOMESTIC_MACRO_UNIVERSE",
    "ETF_BENCHMARK_MAP",
    "NSESessionGatekeeper",
    "MarketDataLoader",
    "CandlestickSanityValidator",
    "IngestedDataset",
]
