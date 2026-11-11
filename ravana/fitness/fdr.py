"""Benjamini-Hochberg False Discovery Rate (FDR) Multiple Testing Controller."""
from __future__ import annotations

import math
from typing import List, Tuple
import numpy as np
from scipy import stats


def ic_to_p_value(ic: float, n_samples: int) -> float:
    """Compute two-tailed p-value for an observed Information Coefficient."""
    if n_samples <= 2 or abs(ic) >= 1.0:
        return 0.0 if abs(ic) >= 1.0 else 1.0

    df = n_samples - 2
    clamped_ic = float(np.clip(ic, -0.999999, 0.999999))
    t_stat = clamped_ic * math.sqrt(df / (1.0 - clamped_ic ** 2))
    p_val = float(2.0 * stats.t.sf(abs(t_stat), df=df))
    return float(np.clip(p_val, 0.0, 1.0))


def benjamini_hochberg_fdr(
    p_values: List[float], q_alpha: float = 0.05
) -> List[bool]:
    """Applies Benjamini-Hochberg FDR procedure to control family-wise false discovery rate."""
    m = len(p_values)
    if m == 0:
        return []

    # Sort indices by p-value
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    passes = [False] * m

    # Find largest k where p_(k) <= (k / m) * q_alpha
    max_k = -1
    for rank_1b, (orig_idx, p_val) in enumerate(indexed_p, start=1):
        threshold = (rank_1b / float(m)) * q_alpha
        if p_val <= threshold:
            max_k = rank_1b

    if max_k != -1:
        # All hypotheses up to max_k are significant
        for rank_1b in range(1, max_k + 1):
            orig_idx = indexed_p[rank_1b - 1][0]
            passes[orig_idx] = True

    return passes
