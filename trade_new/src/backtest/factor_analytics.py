"""Institutional Factor Analytics & Information Coefficient Scoring (Wave 1: Task 1.3 - Qlib Standard)."""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats


def _compute_slice_ic(s: np.ndarray, r: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """Helper to compute single-slice Pearson IC and Spearman Rank IC."""
    mask = np.isfinite(s) & np.isfinite(r)
    s_c, r_c = s[mask], r[mask]
    if len(s_c) < 3 or float(np.std(s_c)) < 1e-8 or float(np.std(r_c)) < 1e-8:
        return None, None
    p_ic = float(stats.pearsonr(s_c, r_c).statistic)
    s_ic = float(stats.spearmanr(s_c, r_c).statistic)
    return p_ic, s_ic


def calculate_factor_ic(
    signals: np.ndarray,
    fwd_returns: np.ndarray,
    cross_sectional: Optional[bool] = None,
    axis: int = 0,
) -> Dict[str, float]:
    """Computes longitudinal or cross-sectional Pearson IC and Spearman Rank IC."""
    sig_arr = np.asarray(signals, dtype=np.float64)
    ret_arr = np.asarray(fwd_returns, dtype=np.float64)
    if sig_arr.shape != ret_arr.shape:
        raise ValueError(f"signals shape {sig_arr.shape} must match fwd_returns {ret_arr.shape}")

    is_cs = (sig_arr.ndim >= 2) if cross_sectional is None else cross_sectional
    if not is_cs:
        p_ic, s_ic = _compute_slice_ic(sig_arr.ravel(), ret_arr.ravel())
        if p_ic is None or s_ic is None:
            return {"ic": 0.0, "rank_ic": 0.0, "p_value": 1.0}
        mask = np.isfinite(sig_arr.ravel()) & np.isfinite(ret_arr.ravel())
        p_val = float(stats.pearsonr(sig_arr.ravel()[mask], ret_arr.ravel()[mask]).pvalue)
        return {"ic": round(p_ic, 4), "rank_ic": round(s_ic, 4), "p_value": float(p_val)}

    # True cross-sectional IC across slices along time axis
    time_axis = 1 if axis == 0 else 0
    t_steps = sig_arr.shape[time_axis]
    ics, rank_ics = [], []
    for t in range(t_steps):
        s_slice = sig_arr[:, t] if axis == 0 else sig_arr[t, :]
        r_slice = ret_arr[:, t] if axis == 0 else ret_arr[t, :]
        p_ic, s_ic = _compute_slice_ic(s_slice, r_slice)
        if p_ic is not None and s_ic is not None:
            ics.append(p_ic)
            rank_ics.append(s_ic)

    if len(ics) < 2:
        return {"ic": 0.0, "rank_ic": 0.0, "p_value": 1.0, "icir": 0.0}

    mean_ic = float(np.mean(ics))
    mean_rank_ic = float(np.mean(rank_ics))
    std_ic = float(np.std(ics, ddof=1))
    icir = (mean_ic / std_ic) if std_ic > 1e-8 else 0.0
    t_stat = stats.ttest_1samp(ics, 0.0)
    p_val = float(t_stat.pvalue) if np.isfinite(t_stat.pvalue) else 1.0

    return {
        "ic": round(mean_ic, 4),
        "rank_ic": round(mean_rank_ic, 4),
        "p_value": float(p_val),
        "icir": round(icir, 4),
    }


def calculate_factor_decay(
    signal: np.ndarray,
    fwd_returns: Optional[np.ndarray] = None,
    max_lags: int = 5,
) -> List[float]:
    """Computes predictive forward return decay or signal autocorrelation."""
    s_arr = np.asarray(signal, dtype=np.float64)
    if fwd_returns is None:
        # Autocorrelation fallback for backward compatibility
        n = s_arr.shape[-1]
        if n <= max_lags:
            return [0.0] * max_lags
        decay: List[float] = []
        for lag in range(1, max_lags + 1):
            s_base = s_arr[..., :-lag].ravel()
            s_lag = s_arr[..., lag:].ravel()
            mask = np.isfinite(s_base) & np.isfinite(s_lag)
            if np.sum(mask) < 5 or float(np.std(s_base[mask])) < 1e-8:
                decay.append(0.0)
                continue
            corr = float(np.corrcoef(s_base[mask], s_lag[mask])[0, 1])
            decay.append(round(corr, 4) if np.isfinite(corr) else 0.0)
        return decay

    # Predictive forward return decay IC(s_t, r_{t -> t+tau})
    r_arr = np.asarray(fwd_returns, dtype=np.float64)
    n = s_arr.shape[-1]
    if n <= max_lags or s_arr.shape != r_arr.shape:
        return [0.0] * max_lags

    decay_ics: List[float] = []
    is_cs = s_arr.ndim >= 2
    for lag in range(1, max_lags + 1):
        s_base = s_arr[..., :-lag]
        r_fwd = r_arr[..., lag:]
        ic_res = calculate_factor_ic(s_base, r_fwd, cross_sectional=is_cs)
        decay_ics.append(ic_res["rank_ic"])
    return decay_ics


def calculate_vif(
    factors: np.ndarray,
    factor_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Computes Variance Inflation Factor (VIF) for each factor column."""
    x = np.asarray(factors, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"factors must be 2D matrix (samples, factors), got shape {x.shape}")
    n_samples, n_factors = x.shape
    names = factor_names if factor_names else [f"Factor_{i}" for i in range(n_factors)]
    if n_factors <= 1:
        return {names[0]: 1.0}

    vif_dict: Dict[str, float] = {}
    for j in range(n_factors):
        y = x[:, j]
        x_others = np.delete(x, j, axis=1)
        valid = np.isfinite(y) & np.all(np.isfinite(x_others), axis=1)
        y_c, x_c = y[valid], x_others[valid]
        if len(y_c) < n_factors + 2 or float(np.std(y_c)) < 1e-8:
            vif_dict[names[j]] = 1.0
            continue
        design = np.column_stack([np.ones_like(y_c), x_c])
        coeffs, _, _, _ = np.linalg.lstsq(design, y_c, rcond=None)
        residuals = y_c - (design @ coeffs)
        ss_tot = float(np.sum((y_c - np.mean(y_c)) ** 2))
        ss_res = float(np.sum(residuals ** 2))
        r2 = max(0.0, 1.0 - (ss_res / max(ss_tot, 1e-12)))
        vif = 1.0 / max(1.0 - r2, 1e-4)
        vif_dict[names[j]] = round(min(float(vif), 1e4), 2)
    return vif_dict


def check_multicollinearity(
    factors: np.ndarray,
    factor_names: Optional[List[str]] = None,
    threshold: float = 10.0,
) -> Dict[str, Union[Dict[str, float], List[str], bool]]:
    """Evaluates multi-factor matrix for multicollinearity against VIF threshold."""
    vif_res = calculate_vif(factors, factor_names)
    collinear = [f for f, v in vif_res.items() if v >= threshold]
    return {
        "vif": vif_res,
        "collinear_factors": collinear,
        "has_multicollinearity": len(collinear) > 0,
    }
