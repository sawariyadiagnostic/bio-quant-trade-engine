"""Real-time Elastic String (Rubber-Band Basis Tension) Statistical Arbitrage Engine.

RAVANA Structural Strategy Module for ETF-Anchor Basis Tension:
1. Computes log spread: spread = log(etf_prices) - beta * log(anchor_prices)
2. Causal rolling basis tension: mu = rolling_mean(spread), sigma = rolling_std(spread)
3. Rolling z-score: z = (spread - mu) / sigma
4. Discrete TriggerSignal generation:
   - Z <= -entry_z: TriggerAction.BUY ("PULL_LONG", ETF is undervalued vs anchor)
   - Z >= +entry_z: TriggerAction.SELL ("PULL_SHORT", ETF is overvalued vs anchor)
   - Conviction scaled by |Z| / 3.0 capped at 1.0
   - Metadata stores: spread_z, elastic_tension, entry_z, exit_z, stop_z
5. Dynamic snap release & risk exits:
   - |current_z| <= exit_z: (True, "SNAP_EQUILIBRIUM")
   - (is_long and current_z <= -stop_z) or (not is_long and current_z >= stop_z): (True, "BROKEN_STRING_STOP")
6. Strict Point-in-Time Causality (Zero Lookahead / No Future Row Indexing).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

try:
    from ravana.strategies.base import (
        StructuralTrigger,
        TriggerAction,
        TriggerSignal,
    )
except ImportError:
    from strategies.base import (
        StructuralTrigger,
        TriggerAction,
        TriggerSignal,
    )


@dataclass
class ElasticPairConfig:
    """Configuration for an ETF <-> Anchor statistical arbitrage pair."""

    etf_symbol: str
    anchor_symbol: str
    beta: float = 1.0
    lookback_bars: int = 20
    entry_z_threshold: float = 1.8
    exit_z_threshold: float = 0.2
    stop_z_threshold: float = 3.2


# Pre-configured 5 Core Sovereign Arbitrage Pairs
CORE_ELASTIC_PAIRS: List[ElasticPairConfig] = [
    ElasticPairConfig(
        etf_symbol="BANKBEES",
        anchor_symbol="^NSEBANK",
        beta=1.0,
        lookback_bars=20,
        entry_z_threshold=1.8,
        exit_z_threshold=0.2,
        stop_z_threshold=3.2,
    ),
    ElasticPairConfig(
        etf_symbol="PSUBNKBEES",
        anchor_symbol="BANKBEES",
        beta=1.0,
        lookback_bars=20,
        entry_z_threshold=1.8,
        exit_z_threshold=0.2,
        stop_z_threshold=3.2,
    ),
    ElasticPairConfig(
        etf_symbol="ITBEES",
        anchor_symbol="^CNXIT",
        beta=1.0,
        lookback_bars=20,
        entry_z_threshold=1.8,
        exit_z_threshold=0.2,
        stop_z_threshold=3.2,
    ),
    ElasticPairConfig(
        etf_symbol="SILVERBEES",
        anchor_symbol="GOLDBEES",
        beta=1.0,
        lookback_bars=20,
        entry_z_threshold=1.8,
        exit_z_threshold=0.2,
        stop_z_threshold=3.2,
    ),
    ElasticPairConfig(
        etf_symbol="CPSEETF",
        anchor_symbol="BZ=F",
        beta=1.0,
        lookback_bars=20,
        entry_z_threshold=1.8,
        exit_z_threshold=0.2,
        stop_z_threshold=3.2,
    ),
]

CORE_ELASTIC_PAIRS_MAP: Dict[str, ElasticPairConfig] = {
    cfg.etf_symbol: cfg for cfg in CORE_ELASTIC_PAIRS
}


def compute_basis_tension(
    etf_prices: pd.Series,
    anchor_prices: pd.Series,
    lookback: int = 20,
    beta: float = 1.0,
) -> Tuple[pd.Series, pd.Series]:
    """Compute causal log spread and rolling basis tension z-score.

    Args:
        etf_prices: Time series of ETF prices.
        anchor_prices: Time series of Anchor/Benchmark prices.
        lookback: Rolling lookback bars for baseline mean and variance.
        beta: Hedge ratio / cointegration elasticity exponent.

    Returns:
        Tuple of (spread Series, z-score Series).
    """
    common_idx = etf_prices.index.intersection(anchor_prices.index)
    if common_idx.empty:
        empty_s = pd.Series(dtype=float)
        return empty_s, empty_s

    etf_aligned = etf_prices.loc[common_idx].astype(float)
    anchor_aligned = anchor_prices.loc[common_idx].astype(float)

    # Sanitize positive prices for logarithm
    clean_etf = np.where(etf_aligned > 0.0, etf_aligned, np.nan)
    clean_anchor = np.where(anchor_aligned > 0.0, anchor_aligned, np.nan)

    # Log Spread: spread = log(etf) - beta * log(anchor)
    spread = pd.Series(
        np.log(clean_etf) - float(beta) * np.log(clean_anchor),
        index=common_idx,
        name="spread",
    )

    # Strict point-in-time rolling metrics (zero future lookahead)
    mu = spread.rolling(window=lookback, min_periods=lookback).mean()
    sigma = spread.rolling(window=lookback, min_periods=lookback).std(ddof=1)

    # Rolling Z-score: z = (spread - mu) / sigma
    sigma_safe = sigma.copy()
    sigma_safe[sigma_safe.isna() | (sigma_safe < 1e-7)] = np.nan
    z_score = ((spread - mu) / sigma_safe).fillna(0.0)
    z_score.name = "spread_z"

    return spread, z_score


def check_snap_release(
    current_z: float,
    entry_z: float,
    is_long: bool,
    exit_z: float = 0.2,
    stop_z: float = 3.2,
) -> Tuple[bool, str]:
    """Check snap-release mean reversion equilibrium or broken-string stop.

    Args:
        current_z: Current basis tension z-score.
        entry_z: Z-score at trade inception.
        is_long: True for PULL_LONG position, False for PULL_SHORT position.
        exit_z: Symmetric equilibrium exit threshold (default: 0.2).
        stop_z: Tail risk broken-string invalidation threshold (default: 3.2).

    Returns:
        Tuple of (should_exit: bool, exit_reason: str).
    """
    z_val = float(current_z)
    exit_thresh = float(exit_z)
    stop_thresh = float(stop_z)

    # 1. Equilibrium Snap Release (Mean reversion completed)
    if abs(z_val) <= exit_thresh:
        return True, "SNAP_EQUILIBRIUM"

    # 2. Broken String Stop (Adverse dislocation breakout)
    if is_long and z_val <= -stop_thresh:
        return True, "BROKEN_STRING_STOP"
    if (not is_long) and z_val >= stop_thresh:
        return True, "BROKEN_STRING_STOP"

    return False, ""


class ElasticStringEngine:
    """Real-time Elastic String (Rubber-Band Basis Tension) Arbitrage Engine."""

    name: str = "ElasticStringEngine"

    def __init__(self, config: Optional[ElasticPairConfig] = None) -> None:
        self.config = config

    @staticmethod
    def compute_basis_tension(
        etf_prices: pd.Series,
        anchor_prices: pd.Series,
        lookback: int = 20,
        beta: float = 1.0,
    ) -> Tuple[pd.Series, pd.Series]:
        """Compute log spread and rolling z-score tension."""
        return compute_basis_tension(etf_prices, anchor_prices, lookback, beta)

    @staticmethod
    def check_snap_release(
        current_z: float,
        entry_z: float,
        is_long: bool,
        exit_z: float = 0.2,
        stop_z: float = 3.2,
    ) -> Tuple[bool, str]:
        """Evaluate exit state machine."""
        return check_snap_release(current_z, entry_z, is_long, exit_z, stop_z)

    def generate_signals(
        self,
        etf_df: Union[pd.DataFrame, pd.Series],
        anchor_df: Union[pd.DataFrame, pd.Series],
        pair_config: Optional[ElasticPairConfig] = None,
    ) -> List[TriggerSignal]:
        """Generate discrete execution signals from ETF and Anchor price data."""
        cfg = pair_config or self.config
        if cfg is None:
            raise ValueError("ElasticPairConfig must be provided to generate_signals.")

        etf_s = self._extract_series(etf_df)
        anchor_s = self._extract_series(anchor_df)

        if etf_s.empty or anchor_s.empty or len(etf_s) < cfg.lookback_bars:
            return []

        spread, z_score = compute_basis_tension(
            etf_s, anchor_s, lookback=cfg.lookback_bars, beta=cfg.beta
        )
        mu = spread.rolling(window=cfg.lookback_bars, min_periods=cfg.lookback_bars).mean()

        signals: List[TriggerSignal] = []
        for i in range(len(spread)):
            z_val = float(z_score.iloc[i])
            if np.isnan(z_val) or np.isnan(spread.iloc[i]) or np.isnan(mu.iloc[i]):
                continue

            ts = spread.index[i]
            elastic_tension = float(spread.iloc[i] - mu.iloc[i])
            conviction = min(1.0, max(0.0, float(abs(z_val) / 3.0)))

            metadata: Dict[str, Any] = {
                "spread_z": round(z_val, 4),
                "elastic_tension": round(elastic_tension, 6),
                "entry_z": float(cfg.entry_z_threshold),
                "exit_z": float(cfg.exit_z_threshold),
                "stop_z": float(cfg.stop_z_threshold),
                "anchor_symbol": cfg.anchor_symbol,
                "beta": float(cfg.beta),
            }

            if z_val <= -cfg.entry_z_threshold:
                metadata["mode"] = "PULL_LONG"
                signals.append(
                    TriggerSignal(
                        timestamp=pd.Timestamp(ts),
                        symbol=cfg.etf_symbol,
                        action=TriggerAction.BUY,
                        conviction=conviction,
                        trigger_name=f"{self.name}_PULL_LONG",
                        metadata=metadata,
                    )
                )
            elif z_val >= cfg.entry_z_threshold:
                metadata["mode"] = "PULL_SHORT"
                signals.append(
                    TriggerSignal(
                        timestamp=pd.Timestamp(ts),
                        symbol=cfg.etf_symbol,
                        action=TriggerAction.SELL,
                        conviction=conviction,
                        trigger_name=f"{self.name}_PULL_SHORT",
                        metadata=metadata,
                    )
                )

        return signals

    @staticmethod
    def _extract_series(data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """Extract 1D price series from DataFrame or Series."""
        if isinstance(data, pd.Series):
            return data
        if data.empty:
            return pd.Series(dtype=float)
        if "Close" in data.columns:
            return data["Close"]
        if "close" in data.columns:
            return data["close"]
        return data.iloc[:, 0]


# Alias for trigger integration
ElasticStringTrigger = ElasticStringEngine


def generate_signals(
    etf_df: Union[pd.DataFrame, pd.Series],
    anchor_df: Union[pd.DataFrame, pd.Series],
    pair_config: Optional[ElasticPairConfig] = None,
) -> List[TriggerSignal]:
    """Module-level generator for discrete execution signals using ElasticStringEngine."""
    engine = ElasticStringEngine(config=pair_config)
    return engine.generate_signals(etf_df, anchor_df, pair_config)

