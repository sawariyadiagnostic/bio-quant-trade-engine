"""Institutional schema, series contract, and physical invariant validator for Parquet Lakehouse."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class SeriesType(str, Enum):
    """Canonical classification for time-series datasets stored in lakehouse."""

    OHLCV = "ohlcv"
    ETF_SPREAD = "etf_spread"
    MACRO_VIX = "macro_vix"
    MACRO_FX_YIELD = "macro_fx_yield"
    FEATURES = "features"
    TARGETS = "targets"
    GENERIC = "generic"


class SchemaValidationError(Exception):
    """Raised when DataFrame violates physical storage invariants."""

    pass


@dataclass(frozen=True)
class SchemaValidationReport:
    """Immutable audit report for partition validation."""

    is_valid: bool
    series_type: SeriesType
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class LakeSchemaValidator:
    """Validates structural, contract, and physical invariants on financial time series."""

    @classmethod
    def detect_series_type(cls, df: pd.DataFrame, ticker: str = "") -> SeriesType:
        """Infer series contract based on columns and symbol naming."""
        cols = set(df.columns)
        ticker_upper = ticker.upper()

        if {"Spread_INR", "Theoretical_Close"}.intersection(cols) or "SPREAD" in ticker_upper:
            return SeriesType.ETF_SPREAD
        if {"gross_return", "net_return", "binary_label"}.issubset(cols):
            return SeriesType.TARGETS
        if {"LOG_RET_1", "ATR_14_NORM", "VOLATILITY_20"}.issubset(cols):
            return SeriesType.FEATURES
        if "VIX" in ticker_upper:
            return SeriesType.MACRO_VIX
        if any(k in ticker_upper for k in ["USDINR", "YIELD", "10Y", "CRUDE", "BRENT"]):
            return SeriesType.MACRO_FX_YIELD
        if {"Open", "High", "Low", "Close"}.issubset(cols):
            return SeriesType.OHLCV
        return SeriesType.GENERIC

    @classmethod
    def validate(
        cls,
        df: pd.DataFrame,
        tier: str = "silver",
        ticker: str = "",
        expected_tz: Optional[str] = None,
    ) -> SchemaValidationReport:
        """Validate DataFrame schema, dtypes, nulls, and contract-specific invariants."""
        errors: List[str] = []
        warnings: List[str] = []

        if df is None or df.empty:
            errors.append("DataFrame is empty or None")
            return SchemaValidationReport(is_valid=False, series_type=SeriesType.GENERIC, errors=errors)

        series_type = cls.detect_series_type(df, ticker)

        # 1. Temporal Index & Timezone
        cls._check_index(df, errors, expected_tz)

        # 2. General Nulls and Infs
        cls._check_nulls_and_infs(df, errors)

        # 3. Contract-Specific Physical Invariants
        if series_type == SeriesType.OHLCV:
            cls._check_ohlcv(df, errors)
            cls._check_quality_flags(df, tier, warnings)
        elif series_type == SeriesType.ETF_SPREAD:
            cls._check_spread_analytics(df, errors)
        elif series_type == SeriesType.MACRO_VIX:
            cls._check_macro_vix(df, errors)
        elif series_type == SeriesType.MACRO_FX_YIELD:
            cls._check_macro_fx_yield(df, errors)
        elif series_type == SeriesType.FEATURES:
            cls._check_features(df, errors)
        elif series_type == SeriesType.TARGETS:
            cls._check_targets(df, errors)

        metadata = {
            "tier": tier,
            "series_type": series_type.value,
            "total_bars": len(df),
            "columns": list(df.columns),
            "start": str(df.index[0]) if len(df) > 0 and isinstance(df.index, pd.DatetimeIndex) else None,
            "end": str(df.index[-1]) if len(df) > 0 and isinstance(df.index, pd.DatetimeIndex) else None,
            "tz": str(df.index.tz) if isinstance(df.index, pd.DatetimeIndex) else None,
        }
        return SchemaValidationReport(
            is_valid=len(errors) == 0,
            series_type=series_type,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
        )

    @staticmethod
    def _check_index(df: pd.DataFrame, errors: List[str], expected_tz: Optional[str]) -> None:
        """Enforce DatetimeIndex, monotonicity, uniqueness, and timezone consistency."""
        if not isinstance(df.index, pd.DatetimeIndex):
            errors.append("Index must be pd.DatetimeIndex")
            return

        if not df.index.is_monotonic_increasing:
            errors.append("DatetimeIndex is not monotonically increasing")

        if not df.index.is_unique:
            dup_count = int(df.index.duplicated().sum())
            errors.append(f"DatetimeIndex contains {dup_count} duplicate timestamps")

        if expected_tz is not None:
            if df.index.tz is None:
                errors.append(f"DatetimeIndex is timezone-naive; expected {expected_tz}")
            elif str(df.index.tz) != expected_tz:
                errors.append(f"Timezone mismatch: got {df.index.tz}, expected {expected_tz}")

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
    def _check_ohlcv(df: pd.DataFrame, errors: List[str]) -> None:
        """Enforce positive prices and strict candlestick geometry."""
        for col in ["Open", "High", "Low", "Close"]:
            if (df[col] <= 0).any():
                errors.append(f"Physical violation: {col} contains non-positive values")

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

        if "Volume" in df.columns and (df["Volume"] < 0).any():
            errors.append("Volume contains negative values")

    @staticmethod
    def _check_spread_analytics(df: pd.DataFrame, errors: List[str]) -> None:
        """Validate ETF tracking spread mathematical invariants."""
        required = {"ETF_Close", "Index_Close", "Multiplier", "Theoretical_Close", "Spread_INR", "Basis_Bps"}
        missing = required - set(df.columns)
        if missing:
            errors.append(f"ETF Spread partition missing required columns: {sorted(missing)}")
            return

        if (df["ETF_Close"] <= 0).any() or (df["Index_Close"] <= 0).any() or (df["Multiplier"] <= 0).any():
            errors.append("Non-positive values found in ETF_Close, Index_Close, or Multiplier")

        theo_calc = df["Index_Close"] / np.maximum(1e-4, df["Multiplier"])
        if not np.allclose(df["Theoretical_Close"], theo_calc, rtol=1e-3, atol=1e-2):
            errors.append("Theoretical_Close does not equal Index_Close / Multiplier")

        spread_calc = df["ETF_Close"] - df["Theoretical_Close"]
        if not np.allclose(df["Spread_INR"], spread_calc, rtol=1e-3, atol=1e-2):
            errors.append("Spread_INR does not equal ETF_Close - Theoretical_Close")

        if "Rolling_Correlation" in df.columns:
            corr = df["Rolling_Correlation"].dropna()
            if ((corr < -1.0001) | (corr > 1.0001)).any():
                errors.append("Rolling_Correlation violates [-1, 1] bounds")

    @staticmethod
    def _check_macro_vix(df: pd.DataFrame, errors: List[str]) -> None:
        """Enforce volatility bounds and positivity."""
        col = "Close" if "Close" in df.columns else df.columns[0]
        if (df[col] <= 0).any():
            errors.append(f"VIX values must be strictly positive in {col}")
        if (df[col] < 3.0).any() or (df[col] > 250.0).any():
            errors.append(f"VIX value outside empirical bounds [3.0, 250.0] in {col}")

    @staticmethod
    def _check_macro_fx_yield(df: pd.DataFrame, errors: List[str]) -> None:
        """Enforce non-negative currency and yield bounds."""
        col = "Close" if "Close" in df.columns else df.columns[0]
        if (df[col] < 0).any():
            errors.append(f"Negative values detected in {col}")

    @staticmethod
    def _check_quality_flags(df: pd.DataFrame, tier: str, warnings: List[str]) -> None:
        """Check for recommended quality flags on silver tier."""
        tier_norm = tier.strip().lower()
        if tier_norm in ("silver", "m4_cleaned"):
            for flag in ["Is_Synthetic", "Is_Spike"]:
                if flag not in df.columns:
                    warnings.append(f"Quality flag '{flag}' missing in {tier} partition")

    @staticmethod
    def _check_features(df: pd.DataFrame, errors: List[str]) -> None:
        """Verify feature matrix invariants: strictly non-empty, finite."""
        if len(df.columns) == 0:
            errors.append("Feature matrix has 0 columns")
        non_finite = ~np.isfinite(df.values)
        if non_finite.any():
            errors.append(f"Feature matrix contains {non_finite.sum()} non-finite values")

    @staticmethod
    def _check_targets(df: pd.DataFrame, errors: List[str]) -> None:
        """Verify target labels: binary bounds, non-negative sample weights."""
        if "binary_label" in df.columns:
            invalid_binary = ~df["binary_label"].isin([0, 1])
            if invalid_binary.any():
                errors.append(f"binary_label contains non-binary values ({invalid_binary.sum()} instances)")
        if "sample_weight" in df.columns:
            if (df["sample_weight"] < 0.0).any():
                errors.append("sample_weight contains negative values")
