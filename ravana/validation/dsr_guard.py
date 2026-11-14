"""Muscle 15: Deflated Sharpe Ratio (DSR) & Minimum Track Record Length (MinTRL) Guard.

Implements Bailey & López de Prado (2014) exact multiple-testing deflation using
empirical cross-trial Sharpe variance and computes the Minimum Track Record Length
required to prove statistical significance at 95% confidence.
"""
from __future__ import annotations

import math
from typing import List, Optional
import numpy as np
from scipy import stats

from ravana.fitness.deflated_sharpe import (
    EULER_MASCHERONI,
    compute_moments,
    compute_psr,
)
from ravana.validation.config import DSRGuardConfig, DSRGuardReport


class OverfittingGuardEngine:
    """Enforces Deflated Sharpe Ratio and Minimum Track Record Length (MinTRL) gates."""

    def __init__(self, config: Optional[DSRGuardConfig] = None) -> None:
        self.config = config or DSRGuardConfig()

    def compute_min_track_record_length(
        self,
        sharpe_per_period: float,
        benchmark_sharpe_per_period: float,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
    ) -> float:
        """Compute Minimum Track Record Length (MinTRL) in observations (bars/days).

        Formula (Bailey & López de Prado, 2012):
        MinTRL = 1 + [1 - skew * SR + ((kurt - 1) / 4) * SR^2] * (Z_alpha / (SR - SR*))^2
        """
        if sharpe_per_period <= benchmark_sharpe_per_period + 1e-6:
            return float("inf")

        z_alpha = float(stats.norm.ppf(self.config.confidence_level))
        non_normal_adj = (
            1.0
            - skewness * sharpe_per_period
            + ((kurtosis - 1.0) / 4.0) * (sharpe_per_period ** 2)
        )
        non_normal_adj = max(0.1, non_normal_adj)
        min_trl = 1.0 + non_normal_adj * ((z_alpha / (sharpe_per_period - benchmark_sharpe_per_period)) ** 2)
        return float(min_trl)

    @staticmethod
    def expected_max_sharpe_from_std(num_trials: int, std_sharpe_trials: float) -> float:
        """Compute E[max(SR)] under the False Strategy Theorem on the exact trial scale."""
        if num_trials <= 1 or std_sharpe_trials <= 0.0:
            return 0.0
        z1 = float(stats.norm.ppf(1.0 - 1.0 / num_trials))
        z2 = float(stats.norm.ppf(1.0 - 1.0 / (num_trials * math.e)))
        return max(0.0, std_sharpe_trials * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2))

    def audit_strategy(
        self,
        champion_returns: np.ndarray,
        num_trials: int = 100,
        trial_sharpes_per_period: Optional[np.ndarray] = None,
    ) -> DSRGuardReport:
        """Audit a candidate strategy's returns against selection bias and sample length."""
        valid = champion_returns[np.isfinite(champion_returns)]
        n_obs = len(valid)
        rf_per_period = self.config.risk_free_rate_annual / self.config.periods_per_year
        sqrt_ann = math.sqrt(self.config.periods_per_year)

        mean, std, skew, kurt = compute_moments(valid)
        sr_per_period = (mean - rf_per_period) / std if std > 1e-8 else 0.0
        sr_annual = sr_per_period * sqrt_ann

        # Empirical cross-sectional standard deviation of per-period Sharpe ratios across trials
        if trial_sharpes_per_period is not None and len(trial_sharpes_per_period) > 2:
            empirical_std_sr = float(np.std(trial_sharpes_per_period, ddof=1))
        else:
            # Theoretical null variance of per-period Sharpe across N observations is ~ 1 / sqrt(n_obs)
            empirical_std_sr = 1.0 / math.sqrt(max(2, n_obs))

        sr_star_per_period = self.expected_max_sharpe_from_std(num_trials, empirical_std_sr)
        sr_star_annual = sr_star_per_period * sqrt_ann

        dsr = compute_psr(
            sharpe_hat=sr_per_period,
            benchmark_sharpe=sr_star_per_period,
            num_samples=n_obs,
            skewness=skew,
            kurtosis=kurt,
        )

        min_trl = self.compute_min_track_record_length(
            sharpe_per_period=sr_per_period,
            benchmark_sharpe_per_period=sr_star_per_period,
            skewness=skew,
            kurtosis=kurt,
        )

        has_track_record = float(n_obs) >= min_trl
        reasons: List[str] = []

        if not has_track_record:
            trl_str = f"{min_trl:.0f}" if math.isfinite(min_trl) else "INF (SR <= E[max SR])"
            reasons.append(f"Insufficient sample length: {n_obs} bars < MinTRL ({trl_str} bars required).")
        if dsr < self.config.min_dsr_threshold:
            reasons.append(
                f"DSR ({dsr:.3f}) below {self.config.min_dsr_threshold:.2f} confidence threshold after {num_trials} trials."
            )

        return DSRGuardReport(
            observed_sharpe_annual=round(sr_annual, 4),
            observed_sharpe_per_period=round(sr_per_period, 5),
            expected_max_sharpe_per_period=round(sr_star_per_period, 5),
            expected_max_sharpe_annual=round(sr_star_annual, 4),
            empirical_trial_std_per_period=round(empirical_std_sr, 5),
            num_independent_trials=num_trials,
            sample_length_bars=n_obs,
            min_track_record_length=round(min_trl, 2) if math.isfinite(min_trl) else float("inf"),
            deflated_sharpe_ratio=round(dsr, 4),
            skewness=round(skew, 4),
            kurtosis=round(kurt, 4),
            has_sufficient_track_record=has_track_record,
            passed_guard=(has_track_record and dsr >= self.config.min_dsr_threshold),
            rejection_reasons=reasons,
        )
