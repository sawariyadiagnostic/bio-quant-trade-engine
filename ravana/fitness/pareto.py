"""Multi-Objective Non-Dominated Sorting and Pareto Frontier Identification."""
from __future__ import annotations

from typing import List, Tuple
import numpy as np


def dominates(
    obj_a: Tuple[float, ...], obj_b: Tuple[float, ...], maximize: Tuple[bool, ...]
) -> bool:
    """Check if candidate A Pareto-dominates candidate B."""
    not_worse = True
    strictly_better = False

    for a, b, is_max in zip(obj_a, obj_b, maximize):
        if is_max:
            if a < b:
                not_worse = False
                break
            if a > b:
                strictly_better = True
        else:
            if a > b:
                not_worse = False
                break
            if a < b:
                strictly_better = True

    return not_worse and strictly_better


def find_pareto_frontier(
    objectives_matrix: List[Tuple[float, ...]],
    maximize: Tuple[bool, ...] = (True, True, True, False, False, False),
) -> List[int]:
    """Find indices of non-dominated individuals on the Pareto frontier.
    
    Default objectives: (IC [max], RankIC [max], DSR [max], Turnover [min], MDD [min], Complexity [min])
    """
    n = len(objectives_matrix)
    if n == 0:
        return []

    pareto_indices: List[int] = []
    for i in range(n):
        is_dominated = False
        cand_i = objectives_matrix[i]
        for j in range(n):
            if i != j and dominates(objectives_matrix[j], cand_i, maximize):
                is_dominated = True
                break
        if not is_dominated:
            pareto_indices.append(i)

    return pareto_indices
