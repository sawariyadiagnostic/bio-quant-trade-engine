"""Configuration dataclasses for Muscle 4: Data Quality & Cleaning Engine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CleanerConfig:
    """Institutional configuration for time grid reindexing, spike filtering, and spread alignment."""

    fill_gaps: bool = True
    spike_filter_enabled: bool = True
    spike_window: int = 20
    spike_threshold_std: float = 4.0
    clamp_spikes: bool = True
    use_volume_gate: bool = True
    volume_surge_multiplier: float = 2.0
    volume_surge_expansion: float = 2.0
    align_spread: bool = True
    spread_rolling_window: int = 60
    contract_multiplier: Optional[float] = None
    strict_causality: bool = True

    @classmethod
    def standard(cls) -> "CleanerConfig":
        """Factory for default institutional cleaning configuration."""
        return cls()

    @classmethod
    def permissive(cls) -> "CleanerConfig":
        """Factory for looser thresholds during exploratory backtesting."""
        return cls(
            spike_threshold_std=6.0,
            clamp_spikes=False,
            use_volume_gate=True,
        )
