"""Storage configuration and partition routing for Muscle 6 Parquet Lakehouse."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class StorageConfig:
    """Configuration for Sovereign Columnar Parquet Data Lake."""

    lake_root: Path = field(default_factory=lambda: Path("ravana/data/lake").resolve())
    compression: str = "snappy"
    tier_map: Dict[str, str] = field(
        default_factory=lambda: {
            "bronze": "m3_ingested",
            "silver": "m4_cleaned",
            "gold": "m5_resampled",
            "m3": "m3_ingested",
            "m4": "m4_cleaned",
            "m5": "m5_resampled",
            "m3_ingested": "m3_ingested",
            "m4_cleaned": "m4_cleaned",
            "m5_resampled": "m5_resampled",
            "m7": "m7_features",
            "features": "m7_features",
            "m7_features": "m7_features",
            "m8": "m8_targets",
            "targets": "m8_targets",
            "m8_targets": "m8_targets",
        }
    )
    symbol_aliases: Dict[str, str] = field(
        default_factory=lambda: {
            "INR=X": "USDINR",
            "USDINR=X": "USDINR",
            "^NSEBANK": "NSEBANK",
            "^NSEI": "NIFTY50",
            "^INDIAVIX": "INDIAVIX",
            "^VIX": "VIX",
            "^GSPC": "GSPC",
            "BZ=F": "BRENT",
            "^TNX": "US10Y",
        }
    )

    def get_tier_dir(self, tier: str) -> Path:
        """Get directory path for given medallion tier."""
        tier_normalized = tier.strip().lower()
        subfolder = self.tier_map.get(tier_normalized, tier_normalized)
        return self.lake_root / subfolder

    def get_timeframe_subfolder(self, timeframe: str) -> str:
        """Map canonical interval code to directory subfolder."""
        tf = timeframe.strip().lower()
        if tf in {"5m", "15m", "30m", "1h"}:
            return f"intraday_{tf}"
        if tf in {"1d", "daily"}:
            return "daily_1d"
        if tf in {"1wk", "weekly"}:
            return "macro_1wk"
        if tf in {"1mo", "monthly"}:
            return "macro_1mo"
        return tf

    def normalize_ticker(self, ticker: str) -> str:
        """Standardize raw ticker symbol to canonical lake symbol."""
        cleaned = self.symbol_aliases.get(ticker, ticker)
        return cleaned.replace(".NS", "").replace("^", "").replace("=X", "").replace("=F", "")

    def resolve_path(self, tier: str, timeframe: str, ticker: str) -> Path:
        """Resolve partition file path across medallion tiers with convention awareness."""
        tier_dir = self.get_tier_dir(tier)
        tf_dir = tier_dir / self.get_timeframe_subfolder(timeframe)

        clean_ticker = self.normalize_ticker(ticker)
        clean_tf = timeframe.replace("intraday_", "").replace("macro_", "").replace("daily_", "")

        candidates = [
            tf_dir / f"{clean_ticker}_{clean_tf}.parquet",
            tf_dir / f"{clean_ticker}_clean.parquet",
            tf_dir / f"{clean_ticker}_spread.parquet",
            tf_dir / f"{clean_ticker}_spread_{clean_tf}.parquet",
            tf_dir / f"{clean_ticker}.parquet",
            tf_dir / f"{ticker}_{clean_tf}.parquet",
            tf_dir / f"{ticker}_clean.parquet",
            tf_dir / f"{ticker}.parquet",
        ]

        for cand in candidates:
            if cand.exists():
                return cand

        tier_norm = self.tier_map.get(tier.strip().lower(), tier.strip().lower())
        if tier_norm == "m4_cleaned":
            return tf_dir / f"{clean_ticker}_clean.parquet"
        if tier_norm == "m7_features":
            return tf_dir / f"{clean_ticker}_features.parquet"
        if tier_norm == "m8_targets":
            return tf_dir / f"{clean_ticker}_targets.parquet"
        return tf_dir / f"{clean_ticker}_{clean_tf}.parquet"
