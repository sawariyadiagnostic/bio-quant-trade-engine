"""Configuration and Data Contracts for Muscle 10: Multi-Objective Alpha Fitness Engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class ObjectiveWeights:
    """Weights for composite scalar fitness calculation."""

    w_ic: float = 0.35
    w_rank_ic: float = 0.25
    w_dsr: float = 0.20
    w_turnover: float = 0.10
    w_mdd: float = 0.10
    parsimony_penalty_per_node: float = 0.001


@dataclass
class FitnessConfig:
    """Hyperparameters and significance controls for alpha evaluation."""

    weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)
    fdr_alpha: float = 0.05  # Benjamini-Hochberg False Discovery Rate threshold
    risk_free_rate: float = 0.065  # Indian 91-day T-bill risk-free rate annualized (6.5%)
    timeframe: str = "5m"  # Default intraday bar timeframe ("5m", "1m", "1d")
    annualization_factor: float = 18900.0  # Annualization periods (18,900 for 5m, 252 for daily)
    variance_of_trials: float = 0.5  # Prior variance of Sharpe ratios across tested trials
    num_tested_trials: int = 100  # Number of tested hypotheses for DSR multiple-testing adjustment
    min_track_record_bars: int = 30  # Minimum valid observations required for statistics
    adjust_autocorrelation: bool = True  # Andrew Lo (2002) serial correlation adjustment
    autocorr_lags: int = 5  # Autocorrelation lags for Lo correction

    def __post_init__(self) -> None:
        """Resolve default annualization factor based on timeframe if not explicitly overridden."""
        if self.timeframe in ("1d", "daily") and self.annualization_factor == 18900.0:
            self.annualization_factor = 252.0
        elif self.timeframe == "1m" and self.annualization_factor == 18900.0:
            self.annualization_factor = 94500.0


@dataclass
class AlphaMetrics:
    """Comprehensive econometric, statistical, and capacity evaluation metrics."""

    formula_str: str
    node_count: int
    ic: float
    rank_ic: float
    annualized_sharpe: float
    dsr: float
    p_value: float
    passes_fdr: bool
    turnover: float
    max_drawdown: float
    composite_fitness: float
    is_pareto_optimal: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to serializable dictionary."""
        return {
            "formula_str": self.formula_str,
            "node_count": self.node_count,
            "ic": round(self.ic, 5),
            "rank_ic": round(self.rank_ic, 5),
            "annualized_sharpe": round(self.annualized_sharpe, 4),
            "dsr": round(self.dsr, 4),
            "p_value": round(self.p_value, 6),
            "passes_fdr": self.passes_fdr,
            "turnover": round(self.turnover, 5),
            "max_drawdown": round(self.max_drawdown, 5),
            "composite_fitness": round(self.composite_fitness, 5),
            "is_pareto_optimal": self.is_pareto_optimal,
        }
