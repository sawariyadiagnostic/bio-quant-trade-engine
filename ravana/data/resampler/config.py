"""Configuration dataclass for Muscle 5 Causal Resampler."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ResamplerConfig:
    """Configuration governing session-anchored causal resampling."""

    intraday_intervals: Tuple[str, ...] = ("15m", "30m", "1h")
    daily_intervals: Tuple[str, ...] = ("1wk", "1mo")
    base_bar_minutes: int = 5
    require_closed_period: bool = True

    @staticmethod
    def parse_interval_to_minutes(interval: str) -> int:
        """Convert standard interval string to integer minutes.
        
        Supports '5m', '15m', '30m', '60m', '1h', '75m', '125m'.
        """
        clean = interval.strip().lower()
        if clean.endswith("h"):
            hours = int(clean[:-1])
            return hours * 60
        if clean.endswith("m"):
            return int(clean[:-1])
        if clean.isdigit():
            return int(clean)
        raise ValueError(f"Unsupported intraday interval format: {interval}")

    @staticmethod
    def parse_daily_to_freq(interval: str) -> str:
        """Map daily interval string to canonical pandas offset alias."""
        clean = interval.strip().lower()
        if clean in ("1w", "1wk", "w", "wk", "weekly"):
            return "W-FRI"
        if clean in ("1m", "1mo", "m", "mo", "monthly"):
            return "ME"
        return interval
