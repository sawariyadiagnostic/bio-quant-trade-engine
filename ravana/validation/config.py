"""Configuration and Immutable Report Contracts for Phase 5 (Muscles 13-16)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass(frozen=True)
class WalkForwardConfig:
    """Configuration for Muscle 13 Walk-Forward Analysis Engine."""

    n_splits: int = 4
    anchored: bool = False  # False = Rolling window, True = Expanding anchored window
    purge_bars: int = 75    # Minimum 1 full NSE 5m session purged between Train and OOS Test
    min_oos_sharpe: float = 1.0
    min_oos_ic: float = 0.02
    min_wfe: float = 0.50   # Walk-Forward Efficiency (OOS Sharpe / IS Sharpe)


@dataclass(frozen=True)
class WalkForwardFold:
    """Telemetry for a single In-Sample (IS) -> Out-of-Sample (OOS) fold."""

    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    champion_formula: str
    is_ic: float
    oos_ic: float
    is_sharpe: float
    oos_sharpe: float
    oos_return: float


@dataclass(frozen=True)
class WalkForwardResult:
    """Aggregate Out-of-Sample Walk-Forward verification report."""

    folds: List[WalkForwardFold]
    stitched_oos_returns: np.ndarray
    stitched_oos_equity: np.ndarray
    aggregate_oos_ic: float
    aggregate_oos_rank_ic: float
    aggregate_oos_sharpe: float
    aggregate_oos_mdd: float
    walk_forward_efficiency: float
    passed_validation: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_folds": len(self.folds),
            "folds": [f.__dict__ for f in self.folds],
            "aggregate_oos_ic": self.aggregate_oos_ic,
            "aggregate_oos_rank_ic": self.aggregate_oos_rank_ic,
            "aggregate_oos_sharpe": self.aggregate_oos_sharpe,
            "aggregate_oos_mdd": self.aggregate_oos_mdd,
            "walk_forward_efficiency": self.walk_forward_efficiency,
            "passed_validation": self.passed_validation,
        }


@dataclass(frozen=True)
class CPCVConfig:
    """Configuration for Muscle 14 Combinatorial Purged Cross-Validation."""

    n_groups: int = 6
    k_test_groups: int = 2
    purge_window: int = 1
    embargo_pct: float = 0.02
    max_pbo_threshold: float = 0.20


@dataclass(frozen=True)
class CPCVResult:
    """Output contract for Combinatorial Purged Cross-Validation and PBO."""

    num_combinations: int
    num_backtest_paths: int
    pbo: float
    oos_sharpe_distribution: List[float]
    mean_oos_sharpe: float
    median_oos_sharpe: float
    logit_distribution: List[float]
    passed_pbo_gate: bool

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class DSRGuardConfig:
    """Configuration for Muscle 15 Deflated Sharpe Ratio & MinTRL Overfitting Guard."""

    confidence_level: float = 0.95
    min_dsr_threshold: float = 0.95
    risk_free_rate_annual: float = 0.065
    periods_per_year: float = 252.0


@dataclass(frozen=True)
class DSRGuardReport:
    """Diagnostic audit report for Muscle 15 Overfitting Guard."""

    observed_sharpe_annual: float
    observed_sharpe_per_period: float
    expected_max_sharpe_per_period: float
    expected_max_sharpe_annual: float
    empirical_trial_std_per_period: float
    num_independent_trials: int
    sample_length_bars: int
    min_track_record_length: float
    deflated_sharpe_ratio: float
    skewness: float
    kurtosis: float
    has_sufficient_track_record: bool
    passed_guard: bool
    rejection_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        if not np.isfinite(d["min_track_record_length"]):
            d["min_track_record_length"] = 999999.0
        return d


@dataclass(frozen=True)
class MonteCarloConfig:
    """Configuration for Muscle 16 Monte Carlo Path Reshuffler & Stress Tester."""

    num_simulations: int = 1000
    block_size: Optional[int] = None
    caution_drawdown_threshold: float = 0.04  # Matches Muscle 12 CAUTION_HALVED (4% DD)
    ruin_drawdown_threshold: float = 0.08     # Matches Muscle 12 HALTED (8% DD)
    max_acceptable_ruin_prob: float = 0.05
    periods_per_year: float = 252.0
    random_seed: int = 42


@dataclass(frozen=True)
class MonteCarloReport:
    """Stress test output from Muscle 16 stationary block bootstrap."""

    num_simulations: int
    block_size: int
    median_cagr: float
    var_95: float
    var_99: float
    cvar_95: float
    median_max_drawdown: float
    dd_95_percentile: float
    dd_99_percentile: float
    probability_of_caution: float
    probability_of_ruin: float
    passed_stress_test: bool

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class ValidationCertificate:
    """Unified Phase 5 certification contract chaining Muscles 13, 14, 15, and 16."""

    ticker: str
    champion_formula: str
    walk_forward: WalkForwardResult
    cpcv: CPCVResult
    dsr_guard: DSRGuardReport
    monte_carlo: MonteCarloReport
    is_certified_for_live: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "champion_formula": self.champion_formula,
            "is_certified_for_live": self.is_certified_for_live,
            "walk_forward": self.walk_forward.to_dict(),
            "cpcv": self.cpcv.to_dict(),
            "dsr_guard": self.dsr_guard.to_dict(),
            "monte_carlo": self.monte_carlo.to_dict(),
        }
