"""Core quantitative metric estimators: IC, Rank IC, Turnover, Sharpe, and Max Drawdown."""
from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd


def compute_ic_and_rank_ic(signal: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    """Compute Pearson Information Coefficient (ddof=1) and Spearman Rank IC with tie handling."""
    valid = np.isfinite(signal) & np.isfinite(target)
    if np.sum(valid) < 5:
        return 0.0, 0.0

    s = signal[valid]
    y = target[valid]
    std_s = float(np.std(s, ddof=1))
    std_y = float(np.std(y, ddof=1))

    if std_s < 1e-8 or std_y < 1e-8:
        return 0.0, 0.0

    # Pearson correlation with sample degrees of freedom
    cov = float(np.cov(s, y)[0, 1])
    ic = float(np.clip(cov / (std_s * std_y), -1.0, 1.0))

    # Spearman rank correlation with fractional tie ranking
    rank_s = pd.Series(s).rank().to_numpy(dtype=np.float64)
    rank_y = pd.Series(y).rank().to_numpy(dtype=np.float64)
    cov_r = float(np.cov(rank_s, rank_y)[0, 1])
    std_rs = float(np.std(rank_s, ddof=1))
    std_ry = float(np.std(rank_y, ddof=1))

    rank_ic = float(np.clip(cov_r / (std_rs * std_ry), -1.0, 1.0)) if std_rs > 0 and std_ry > 0 else 0.0
    return ic, rank_ic


def compute_turnover(signal: np.ndarray) -> float:
    """Compute mean absolute period-over-period position change (churn rate)."""
    valid = signal[np.isfinite(signal)]
    if len(valid) < 2:
        return 0.0
    # Standardize signal to [-1.0, 1.0] scale before computing turnover
    max_abs = float(np.max(np.abs(valid)))
    norm_s = valid / max(max_abs, 1e-6)
    delta = np.abs(np.diff(norm_s))
    return float(np.mean(delta))


def compute_strategy_returns(
    signal: np.ndarray, target: np.ndarray, long_only: bool = True
) -> np.ndarray:
    """Compute realized strategy returns given signal weights and forward targets."""
    valid = np.isfinite(signal) & np.isfinite(target)
    s = np.zeros_like(signal)
    s[valid] = signal[valid]

    # Standardize signal
    std = float(np.std(s[valid], ddof=1)) if np.sum(valid) > 2 else 1.0
    mean = float(np.mean(s[valid])) if np.sum(valid) > 2 else 0.0
    z = (s - mean) / max(std, 1e-6)

    # Transform into position weight
    pos = np.clip(z, 0.0, 1.0) if long_only else np.clip(z, -1.0, 1.0)
    strat_ret = pos * target
    return np.nan_to_num(strat_ret, nan=0.0)


def compute_oos_strategy_returns(
    oos_signal: np.ndarray,
    oos_target: np.ndarray,
    train_mean: float,
    train_std: float,
    long_only: bool = True,
) -> np.ndarray:
    """Compute strictly Out-of-Sample strategy returns using frozen In-Sample (mean, std)."""
    valid = np.isfinite(oos_signal) & np.isfinite(oos_target)
    s = np.zeros_like(oos_signal, dtype=np.float64)
    s[valid] = oos_signal[valid]

    safe_std = max(float(train_std), 1e-6) if np.isfinite(train_std) else 1.0
    safe_mean = float(train_mean) if np.isfinite(train_mean) else 0.0
    z = (s - safe_mean) / safe_std

    pos = np.clip(z, 0.0, 1.0) if long_only else np.clip(z, -1.0, 1.0)
    strat_ret = pos * oos_target
    return np.nan_to_num(strat_ret, nan=0.0)


def compute_annualized_sharpe(
    returns: np.ndarray, rf_annual: float = 0.065, periods_per_year: float = 252.0
) -> float:
    """Compute annualized Sharpe ratio using sample standard deviation."""
    valid_rets = returns[np.isfinite(returns)]
    if len(valid_rets) < 5:
        return 0.0

    mean_ret = float(np.mean(valid_rets))
    std_ret = float(np.std(valid_rets, ddof=1))
    if std_ret < 1e-8:
        return 0.0

    rf_per_period = rf_annual / periods_per_year
    excess_mean = mean_ret - rf_per_period
    return float((excess_mean / std_ret) * np.sqrt(periods_per_year))


def compute_max_drawdown(returns: np.ndarray) -> float:
    """Compute maximum peak-to-trough equity drawdown."""
    valid = returns[np.isfinite(returns)]
    if len(valid) < 2:
        return 0.0
    # Cumulative wealth path
    wealth = np.cumprod(1.0 + np.clip(valid, -0.99, 10.0))
    peaks = np.maximum.accumulate(wealth)
    drawdowns = (wealth - peaks) / np.maximum(peaks, 1e-8)
    return float(np.abs(np.min(drawdowns)))
