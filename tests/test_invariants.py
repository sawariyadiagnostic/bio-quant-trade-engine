"""Unit tests for System Invariants Engine (TDD Phase 1)."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.domain.invariants import (
    calculate_deflated_sharpe_ratio,
    estimate_return_moments,
    verify_point_in_time_causality,
    check_covariance_conditioning,
    assert_circuit_headroom,
    assert_integer_lot,
    sanitize_finite_array
)


def test_deflated_sharpe_ratio():
    """Verify DSR penalizes large trial counts (P-hacking defense)."""
    # High Sharpe with only 1 trial -> High confidence
    dsr_single = calculate_deflated_sharpe_ratio(
        observed_sr=2.0,
        n_trials=1,
        var_sr=0.1,
        skew=0.0,
        kurt=3.0,
        t_samples=252
    )
    assert dsr_single > 0.95

    # Moderate Sharpe (1.2) evaluated across 10,000 formula trials -> Severe penalty
    dsr_mult = calculate_deflated_sharpe_ratio(
        observed_sr=1.2,
        n_trials=10000,
        var_sr=0.15,
        skew=-0.2,
        kurt=4.0,
        t_samples=252
    )
    assert dsr_mult < 0.50


def test_estimate_return_moments():
    """Verify empirical moment estimation handles normal and fat-tailed series."""
    np.random.seed(42)
    # Normal returns
    normal_rets = np.random.normal(0.0005, 0.015, 2000)
    mu, var, skew, kurt = estimate_return_moments(normal_rets)
    assert abs(mu - 0.0005) < 0.002
    assert abs(skew) < 0.25
    assert abs(kurt - 3.0) < 0.50

    # Fat-tailed series with negative shock (crashes)
    fat_rets = normal_rets.copy()
    fat_rets[10:15] = -0.08  # 8% gap crash down
    _, _, fat_skew, fat_kurt = estimate_return_moments(fat_rets)
    assert fat_skew < -0.50  # Left skewed
    assert fat_kurt > 5.0    # Leptokurtic / fat-tailed

    # Edge cases
    mu_0, var_0, skew_0, kurt_0 = estimate_return_moments(np.array([]))
    assert (mu_0, var_0, skew_0, kurt_0) == (0.0, 0.0, 0.0, 3.0)


def test_point_in_time_causality():
    """Verify lookahead detector distinguishes valid predictive alpha from leakage."""
    T = 200
    np.random.seed(42)
    clean_signal = np.random.randn(T)
    future_rets = np.random.randn(T)

    # Clean signal has near zero correlation with future returns
    is_valid_clean, lead_corr = verify_point_in_time_causality(clean_signal, future_rets, max_lead_corr=0.30)
    assert is_valid_clean is True

    # Leaked signal directly correlates with future return T+1 identically
    leaked_signal = future_rets.copy()
    is_valid_leak, leak_corr = verify_point_in_time_causality(leaked_signal, future_rets, max_lead_corr=0.30)
    assert is_valid_leak is False
    assert leak_corr > 0.90

    # Predictive signal with realistic positive forward IC (e.g. r=0.15) should pass
    predictive_signal = 0.15 * future_rets + np.random.randn(T) * 0.98
    is_valid_pred, pred_corr = verify_point_in_time_causality(predictive_signal, future_rets, max_lead_corr=0.50)
    assert is_valid_pred is True
    assert pred_corr < 0.50


def test_covariance_conditioning():
    """Verify covariance condition number kappa is bounded."""
    # Well-conditioned 3x3 identity
    cov_good = np.eye(3, dtype=np.float64)
    is_ok, kappa = check_covariance_conditioning(cov_good, max_condition_number=10000.0)
    assert is_ok is True
    assert round(kappa, 1) == 1.0

    # Singular collinear matrix
    cov_bad = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0]
    ], dtype=np.float64)
    is_ok_bad, kappa_bad = check_covariance_conditioning(cov_bad, max_condition_number=10000.0)
    assert is_ok_bad is False
    assert kappa_bad > 10000.0


def test_circuit_headroom():
    """Verify headroom check warns when price is too close to circuit bands."""
    # 5% away from upper and lower circuit -> Safe
    safe_ok, _ = assert_circuit_headroom(current_price=100.0, lower_circuit=90.0, upper_circuit=110.0, min_dist_pct=0.01)
    assert safe_ok is True

    # 0.5% away from lower circuit -> Locked
    locked_ok, reason = assert_circuit_headroom(current_price=90.45, lower_circuit=90.0, upper_circuit=110.0, min_dist_pct=0.01)
    assert locked_ok is False
    assert "CIRCUIT_BREACH" in reason


def test_integer_lot():
    """Verify integer lot validator rejects fractions, negatives, and NaNs."""
    assert assert_integer_lot(10) is True
    assert assert_integer_lot(1) is True

    # Fractions / Floats
    assert assert_integer_lot(10.5) is False
    # Zero or negative
    assert assert_integer_lot(0) is False
    assert assert_integer_lot(-5) is False
    # Non-finite
    assert assert_integer_lot(float("nan")) is False


def test_sanitize_finite_array():
    """Verify NaN and Inf array sanitization."""
    arr = np.array([1.0, np.nan, -np.inf, np.inf, 2.5], dtype=np.float64)
    sanitized = sanitize_finite_array(arr, default_fill=0.0)
    assert np.all(np.isfinite(sanitized))
    assert sanitized[1] == 0.0
    assert sanitized[2] == 0.0
    assert sanitized[3] == 0.0
    assert sanitized[0] == 1.0
    assert sanitized[4] == 2.5


if __name__ == "__main__":
    test_deflated_sharpe_ratio()
    test_point_in_time_causality()
    test_covariance_conditioning()
    test_circuit_headroom()
    test_integer_lot()
    test_sanitize_finite_array()
    print("ALL INVARIANT TESTS PASSED!")
