"""
Institutional System Invariants & Quantitative Safeguards Core.
Enforces strict mathematical, operational, and microstructural contracts:
1. Deflated Sharpe Ratio (Bailey & LÃ³pez de Prado, 2014).
2. Point-in-Time Causality & Lead-Lag Leakage Tripwires.
3. Covariance Matrix Condition Number (\u03ba) Safeguards.
4. Statutory Circuit Band Headroom Enforcement.
5. Integer Lot Sizing & NaN/Inf Sanitization.
"""
import math
from typing import Tuple, Union
import numpy as np
from scipy.stats import norm


def calculate_expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """Computes expected maximum Sharpe ratio under the null hypothesis of noise."""
    if n_trials <= 1:
        return 0.0
    euler_mascheroni = 0.5772156649
    sqrt_log = math.sqrt(2.0 * math.log(n_trials))
    sr_star = (sqrt_log + (euler_mascheroni / max(1e-8, sqrt_log))) * math.sqrt(max(1e-8, var_sr))
    return float(sr_star)


def calculate_deflated_sharpe_ratio(
    observed_sr: float,
    n_trials: int = 1,
    var_sr: float = 0.1,
    skew: float = 0.0,
    kurt: float = 3.0,
    t_samples: int = 252,
    benchmark_sr: float = 0.0
) -> float:
    """
    Computes Deflated Sharpe Ratio (DSR) correcting for selection bias and non-normality.
    Returns probability in [0.0, 1.0] that observed Sharpe is statistically significant.
    """
    if not np.isfinite(observed_sr) or t_samples <= 1:
        return 0.0

    sr_star = max(benchmark_sr, calculate_expected_max_sharpe(n_trials, var_sr))
    denom_sq = 1.0 - (skew * observed_sr) + (((kurt - 1.0) / 4.0) * (observed_sr ** 2))
    denom = math.sqrt(max(1e-6, denom_sq))

    z_stat = ((observed_sr - sr_star) * math.sqrt(t_samples - 1)) / denom
    dsr = float(norm.cdf(z_stat))
    return float(np.clip(dsr, 0.0, 1.0))


def verify_point_in_time_causality(
    signal_vector: np.ndarray,
    future_returns: np.ndarray,
    max_lead_corr: float = 0.25
) -> Tuple[bool, float]:
    """
    Checks if signal vector has anomalous forward correlation with future returns.
    Catches lookahead bias and label leakage before deployment.
    """
    if len(signal_vector) < 10 or len(future_returns) < 10:
        return True, 0.0

    sig = np.nan_to_num(signal_vector, nan=0.0)
    rets = np.nan_to_num(future_returns, nan=0.0)

    if np.std(sig) < 1e-8 or np.std(rets) < 1e-8:
        return True, 0.0

    # Calculate Pearson correlation with forward step
    corr = float(np.corrcoef(sig, rets)[0, 1])
    is_causal = abs(corr) <= max_lead_corr
    return is_causal, abs(corr)


def check_covariance_conditioning(
    cov_matrix: np.ndarray,
    max_condition_number: float = 10000.0
) -> Tuple[bool, float]:
    """
    Computes condition number \u03ba(\u03a3) = \u03bb_max / \u03bb_min.
    Warns before matrix inversion or hierarchical clustering degeneration.
    """
    if cov_matrix.ndim != 2 or cov_matrix.shape[0] != cov_matrix.shape[1]:
        return False, float("inf")

    if not np.all(np.isfinite(cov_matrix)):
        return False, float("inf")

    eigvals = np.linalg.eigvalsh(cov_matrix)
    min_eig = float(np.min(eigvals))
    max_eig = float(np.max(eigvals))

    if min_eig <= 1e-12:
        return False, float("inf")

    kappa = float(max_eig / min_eig)
    return (kappa <= max_condition_number), kappa


def assert_circuit_headroom(
    current_price: float,
    lower_circuit: float,
    upper_circuit: float,
    min_dist_pct: float = 0.01
) -> Tuple[bool, str]:
    """Validates that price is not locked against NSE daily circuit bands."""
    if not np.isfinite(current_price) or current_price <= 0:
        return False, "INVALID_PRICE"
    if not np.isfinite(lower_circuit) or not np.isfinite(upper_circuit):
        return False, "INVALID_CIRCUITS"

    dist_l = (current_price - lower_circuit) / max(1e-4, lower_circuit)
    dist_u = (upper_circuit - current_price) / max(1e-4, current_price)

    if dist_l < min_dist_pct:
        return False, f"CIRCUIT_BREACH_LOWER_{dist_l * 100.0:.2f}PCT"
    if dist_u < min_dist_pct:
        return False, f"CIRCUIT_BREACH_UPPER_{dist_u * 100.0:.2f}PCT"

    return True, "CIRCUIT_HEADROOM_OK"


def assert_integer_lot(quantity: Union[int, float]) -> bool:
    """Verifies that an order lot is a strictly positive finite integer."""
    if isinstance(quantity, bool):
        return False
    if not isinstance(quantity, (int, np.integer, float)):
        return False
    if not np.isfinite(quantity) or quantity <= 0:
        return False
    return float(quantity).is_integer()


def sanitize_finite_array(arr: np.ndarray, default_fill: float = 0.0) -> np.ndarray:
    """Replaces NaNs, \u221e, and -\u221e in-place or returns sanitized array."""
    sanitized = np.nan_to_num(arr, nan=default_fill, posinf=default_fill, neginf=default_fill)
    return sanitized
