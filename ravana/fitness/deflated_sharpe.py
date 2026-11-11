"""Deflated Sharpe Ratio (DSR) and Probabilistic Sharpe Ratio (PSR) Engine.

Implements the Marcos López de Prado (2014) framework correcting for:
1. Selection bias from multiple testing (N trials / candidate formulas evaluated).
2. Non-normality of return distributions (skewness and kurtosis).
3. Sample length track record variance.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple
import numpy as np
from scipy import stats

EULER_MASCHERONI: float = 0.57721566490153286060


def compute_moments(returns: np.ndarray) -> Tuple[float, float, float, float]:
    """Compute mean, std, skewness, and Pearson kurtosis for return series."""
    valid = returns[np.isfinite(returns)]
    if len(valid) < 5:
        return 0.0, 1.0, 0.0, 3.0

    mean = float(np.mean(valid))
    diff = valid - mean
    var = float(np.mean(diff ** 2))
    std = math.sqrt(max(1e-12, var))

    skew = float(np.mean(diff ** 3) / (std ** 3))
    kurt = float(np.mean(diff ** 4) / (std ** 4))  # Pearson kurtosis (Normal = 3)
    return mean, std, skew, kurt


def expected_maximum_sharpe(
    num_trials: int, variance_of_trials: float = 0.5
) -> float:
    """Calculate the expected maximum Sharpe ratio under the null hypothesis of false discovery."""
    if num_trials <= 1:
        return 0.0
    sigma_sr = math.sqrt(max(1e-8, variance_of_trials))
    z1 = float(stats.norm.ppf(1.0 - 1.0 / num_trials))
    z2 = float(stats.norm.ppf(1.0 - 1.0 / (num_trials * math.e)))
    sr_star = sigma_sr * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)
    return max(0.0, float(sr_star))


def compute_psr(
    sharpe_hat: float,
    benchmark_sharpe: float,
    num_samples: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Compute the Probabilistic Sharpe Ratio (PSR) for a single strategy vs benchmark."""
    if num_samples <= 2:
        return 0.5

    # Standard error of estimated Sharpe ratio under non-normality
    var_sharpe = (
        1.0
        - skewness * sharpe_hat
        + ((kurtosis - 1.0) / 4.0) * (sharpe_hat ** 2)
    ) / float(num_samples - 1)

    std_sharpe = math.sqrt(max(1e-12, var_sharpe))
    z_stat = (sharpe_hat - benchmark_sharpe) / std_sharpe
    return float(np.clip(stats.norm.cdf(z_stat), 0.0, 1.0))


def compute_dsr(
    returns: np.ndarray,
    num_trials: int = 100,
    variance_of_trials: float = 0.5,
    rf_per_period: float = 0.0,
) -> float:
    """Compute the Deflated Sharpe Ratio (DSR) deflating for multiple testing and non-normality."""
    valid = returns[np.isfinite(returns)]
    if len(valid) < 5:
        return 0.0

    mean, std, skew, kurt = compute_moments(valid)
    if std < 1e-8:
        return 0.0

    sample_sharpe = (mean - rf_per_period) / std
    expected_max_sr = expected_maximum_sharpe(num_trials, variance_of_trials)

    return compute_psr(
        sharpe_hat=sample_sharpe,
        benchmark_sharpe=expected_max_sr,
        num_samples=len(valid),
        skewness=skew,
        kurtosis=kurt,
    )
