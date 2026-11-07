"""Institutional Factor Analytics & Information Coefficient Scoring (Wave 1: Task 1.3 - Qlib Standard)."""
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats


def calculate_factor_ic(signals: np.ndarray, fwd_returns: np.ndarray) -> Dict[str, float]:
    """
    Computes Pearson IC, Spearman Rank IC, and statistical significance against forward returns.

    Args:
        signals: 1D array of alpha factor scores.
        fwd_returns: 1D array of subsequent forward returns.

    Returns:
        Dictionary containing 'ic', 'rank_ic', and 'p_value'.
    """
    if len(signals) != len(fwd_returns):
        raise ValueError(f"signals length ({len(signals)}) must match fwd_returns ({len(fwd_returns)})")

    mask = np.isfinite(signals) & np.isfinite(fwd_returns)
    s_clean = signals[mask]
    r_clean = fwd_returns[mask]

    if len(s_clean) < 5 or np.std(s_clean) < 1e-8 or np.std(r_clean) < 1e-8:
        return {"ic": 0.0, "rank_ic": 0.0, "p_value": 1.0}

    # Pearson Correlation (IC)
    pearson_res = stats.pearsonr(s_clean, r_clean)
    ic = float(pearson_res.statistic)
    p_val = float(pearson_res.pvalue)

    # Spearman Rank Correlation (Rank IC)
    spearman_res = stats.spearmanr(s_clean, r_clean)
    rank_ic = float(spearman_res.statistic)

    return {
        "ic": round(ic, 4),
        "rank_ic": round(rank_ic, 4),
        "p_value": float(p_val)
    }


def calculate_factor_decay(signal: np.ndarray, max_lags: int = 5) -> List[float]:
    """
    Computes factor autocorrelation across multiple lags to determine predictive persistence.

    Args:
        signal: 1D time-series array of factor values.
        max_lags: Maximum number of trailing bar lags to evaluate.

    Returns:
        List of autocorrelation coefficients for lags 1 through max_lags.
    """
    n = len(signal)
    if n <= max_lags:
        return [0.0] * max_lags

    decay: List[float] = []
    for lag in range(1, max_lags + 1):
        s_base = signal[:-lag]
        s_lagged = signal[lag:]
        mask = np.isfinite(s_base) & np.isfinite(s_lagged)
        if np.sum(mask) < 5 or np.std(s_base[mask]) < 1e-8:
            decay.append(0.0)
            continue
        corr = float(np.corrcoef(s_base[mask], s_lagged[mask])[0, 1])
        decay.append(round(corr, 4) if np.isfinite(corr) else 0.0)

    return decay
