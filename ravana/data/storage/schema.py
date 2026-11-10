"""Institutional schema and physical invariant validator for Parquet Lakehouse."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
import numpy as np
import pandas as pd


class SchemaValidationError(Exception):
    """Raised when DataFrame violates physical storage invariants."""
    pass


@dataclass(frozen=True)
class SchemaValidationReport:
    """Immutable audit report for partition validation."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class LakeSchemaValidator:
    """Validates structural and physical invariants on financial time series."""

    @classmethod
    def validate(cls, df: pd.DataFrame, tier: str = "silver") -> SchemaValidationReport:
        """Validate DataFrame schema, dtypes, nulls, and candlestick geometry."""
        errors: List[str] = []
        warnings: List[str] = []

        if df is None or df.empty:
            errors.append("DataFrame is empty or None")
            return SchemaValidationReport(is_valid=False, errors=errors, warnings=warnings)

        cls._check_index(df, errors)
        cls._check_nulls_and_infs(df, errors)
        cls._check_candlestick_geometry(df, errors)
        cls._check_volume(df, errors)
        cls._check_quality_flags(df, tier, warnings)

        metadata = {
            "tier": tier,
            "total_bars": len(df),
            "columns": list(df.columns),
            "start": str(df.index[0]) if len(df) > 0 and isinstance(df.index, pd.DatetimeIndex) else None,
            "end": str(df.index[-1]) if len(df) > 0 and isinstance(df.index, pd.DatetimeIndex) else None,
        }
        return SchemaValidationReport(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
        )

    @staticmethod
    def _check_index(df: pd.DataFrame, errors: List[str]) -> None:
        """Validate temporal index monotonicity and type."""
        if not isinstance(df.index, pd.DatetimeIndex):
            errors.append("Index must be pd.DatetimeIndex")
        elif not df.index.is_monotonic_increasing:
            errors.append("DatetimeIndex is not monotonically increasing")

    @staticmethod
    def _check_nulls_and_infs(df: pd.DataFrame, errors: List[str]) -> None:
        """Validate zero NaNs and zero infinite values."""
        nan_count = int(df.isna().sum().sum())
        if nan_count > 0:
            errors.append(f"Dataset contains {nan_count} NaN values")

        num_cols = df.select_dtypes(include=np.number)
        inf_count = int(np.isinf(num_cols).sum().sum())
        if inf_count > 0:
            errors.append(f"Dataset contains {inf_count} Inf values")

    @staticmethod
    def _check_candlestick_geometry(df: pd.DataFrame, errors: List[str]) -> None:
        """Enforce strict candlestick physical bounds: Low <= min(O, C) <= max(O, C) <= High."""
        if not {"Open", "High", "Low", "Close"}.issubset(df.columns):
            return

        hl_inv = (df["High"] < df["Low"]).sum()
        if hl_inv > 0:
            errors.append(f"Candlestick inversion: High < Low on {hl_inv} bars")

        ho_inv = (df["High"] < df["Open"]).sum()
        hc_inv = (df["High"] < df["Close"]).sum()
        if ho_inv > 0 or hc_inv > 0:
            errors.append(f"Candlestick inversion: High < Open ({ho_inv}) or High < Close ({hc_inv})")

        lo_inv = (df["Low"] > df["Open"]).sum()
        lc_inv = (df["Low"] > df["Close"]).sum()
        if lo_inv > 0 or lc_inv > 0:
            errors.append(f"Candlestick inversion: Low > Open ({lo_inv}) or Low > Close ({lc_inv})")

    @staticmethod
    def _check_volume(df: pd.DataFrame, errors: List[str]) -> None:
        """Verify non-negative traded volume."""
        if "Volume" in df.columns:
            neg_vol = (df["Volume"] < 0).sum()
            if neg_vol > 0:
                errors.append(f"Volume contains negative values on {neg_vol} bars")

    @staticmethod
    def _check_quality_flags(df: pd.DataFrame, tier: str, warnings: List[str]) -> None:
        """Check for recommended quality flags on silver tier."""
        tier_norm = tier.strip().lower()
        if tier_norm in ("silver", "m4_cleaned"):
            for flag in ["Is_Synthetic", "Is_Spike"]:
                if flag not in df.columns:
                    warnings.append(f"Quality flag '{flag}' missing in {tier} partition")
