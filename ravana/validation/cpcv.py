"""Muscle 14: Combinatorial Purged Cross-Validation (CPCV) & PBO Engine.

Implements Marcos López de Prado's CPCV algorithm with target-overlap purging,
post-test embargoing, In-Sample frozen standardization, and Probability of
Backtest Overfitting (PBO) estimation across Muscle 9 populations.
"""
from __future__ import annotations

import itertools
import math
from typing import List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from ravana.compute.dispatcher import ComputeDispatcher
from ravana.domain.formula_ast import ASTNode
from ravana.evolution.config import Individual
from ravana.fitness.metrics import (
    compute_annualized_sharpe,
    compute_oos_strategy_returns,
    compute_strategy_returns,
)
from ravana.validation.config import CPCVConfig, CPCVResult


class CombinatorialPurgedCV:
    """Generates purged/embargoed combinatorial splits and computes PBO."""

    def __init__(self, config: Optional[CPCVConfig] = None) -> None:
        self.config = config or CPCVConfig()

    def generate_combinatorial_splits(
        self,
        n_samples: int,
        hold_durations: Optional[np.ndarray] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]]:
        """Generate all C(N, k) (train_indices, test_indices, test_group_ids) splits."""
        n_groups = self.config.n_groups
        k_test = self.config.k_test_groups
        if n_samples < n_groups * 3:
            raise ValueError(f"Insufficient samples ({n_samples}) for {n_groups} CPCV groups.")

        indices = np.arange(n_samples)
        groups = np.array_split(indices, n_groups)
        embargo_len = max(1, int(math.ceil(n_samples * self.config.embargo_pct)))
        base_purge = max(1, self.config.purge_window)

        splits: List[Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]] = []

        for test_group_ids in itertools.combinations(range(n_groups), k_test):
            test_idx = np.concatenate([groups[g] for g in test_group_ids])

            forbidden = np.zeros(n_samples, dtype=bool)
            forbidden[test_idx] = True

            for g in test_group_ids:
                g_start = int(groups[g][0])
                g_end = int(groups[g][-1])

                dynamic_purge = base_purge
                if hold_durations is not None and g_start > 0:
                    max_hold = int(np.max(hold_durations[max(0, g_start - 5) : g_start]))
                    dynamic_purge = max(base_purge, max_hold)

                p_start = max(0, g_start - dynamic_purge)
                forbidden[p_start:g_start] = True

                e_end = min(n_samples, g_end + 1 + embargo_len)
                forbidden[g_end + 1 : e_end] = True

            train_idx = indices[~forbidden]
            splits.append((train_idx, test_idx, test_group_ids))

        return splits

    def evaluate_population_pbo(
        self,
        candidates: Sequence[Individual | ASTNode],
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
    ) -> CPCVResult:
        """Chain Muscle 9 population and Muscle 7/8 dataframes directly into CPCV PBO."""
        trees = [c.tree if isinstance(c, Individual) else c for c in candidates]
        raw_signals = ComputeDispatcher.evaluate_batch(trees, features_df)
        if hasattr(raw_signals, "cpu"):
            raw_signals = raw_signals.cpu().numpy()
        if isinstance(raw_signals, np.ndarray) and raw_signals.ndim == 1:
            raw_signals = np.expand_dims(raw_signals, axis=0)

        entry_mask = (
            targets_df["is_entry_bar"].to_numpy(dtype=bool)
            if "is_entry_bar" in targets_df.columns
            else np.ones(len(targets_df), dtype=bool)
        )
        sig_entry = raw_signals[:, entry_mask]
        y_entry = targets_df["net_return"].to_numpy(dtype=np.float64)[entry_mask]
        holds = (
            targets_df["calendar_hold_days"].to_numpy(dtype=np.int32)[entry_mask]
            if "calendar_hold_days" in targets_df.columns
            else None
        )
        return self.evaluate_pbo(sig_entry, y_entry, hold_durations=holds)

    def evaluate_pbo(
        self,
        signals_matrix: np.ndarray,
        target: np.ndarray,
        hold_durations: Optional[np.ndarray] = None,
    ) -> CPCVResult:
        """Compute Probability of Backtest Overfitting (PBO) across M candidate strategies."""
        if signals_matrix.ndim == 1:
            signals_matrix = np.expand_dims(signals_matrix, axis=0)

        n_strategies, n_samples = signals_matrix.shape
        splits = self.generate_combinatorial_splits(n_samples, hold_durations=hold_durations)

        logits: List[float] = []
        oos_sharpes_of_best_is: List[float] = []

        for train_idx, test_idx, _ in splits:
            is_scores = np.zeros(n_strategies, dtype=np.float64)
            oos_scores = np.zeros(n_strategies, dtype=np.float64)

            y_train = target[train_idx]
            y_test = target[test_idx]

            for m in range(n_strategies):
                s_train = signals_matrix[m, train_idx]
                s_test = signals_matrix[m, test_idx]

                valid_tr = np.isfinite(s_train)
                tr_mean = float(np.mean(s_train[valid_tr])) if np.sum(valid_tr) > 2 else 0.0
                tr_std = float(np.std(s_train[valid_tr], ddof=1)) if np.sum(valid_tr) > 2 else 1.0

                rets_is = compute_strategy_returns(s_train, y_train, long_only=True)
                rets_oos = compute_oos_strategy_returns(
                    s_test, y_test, train_mean=tr_mean, train_std=tr_std, long_only=True
                )
                is_scores[m] = compute_annualized_sharpe(rets_is, periods_per_year=252.0)
                oos_scores[m] = compute_annualized_sharpe(rets_oos, periods_per_year=252.0)

            best_is_idx = int(np.argmax(is_scores))
            chosen_oos_sr = float(oos_scores[best_is_idx])
            oos_sharpes_of_best_is.append(chosen_oos_sr)

            if n_strategies > 1:
                less_count = float(np.sum(oos_scores < chosen_oos_sr))
                eq_count = float(np.sum(oos_scores == chosen_oos_sr))
                omega = (less_count + 0.5 * eq_count) / float(n_strategies)
                omega_clamped = float(np.clip(omega, 1.0 / (2.0 * n_strategies), 1.0 - 1.0 / (2.0 * n_strategies)))
                logit = math.log(omega_clamped / (1.0 - omega_clamped))
            else:
                logit = 1.0 if chosen_oos_sr > 0.0 else -1.0

            logits.append(logit)

        logits_arr = np.array(logits, dtype=np.float64)
        pbo = float(np.mean(logits_arr <= 0.0)) if len(logits_arr) > 0 else 1.0

        n_groups = self.config.n_groups
        k_test = self.config.k_test_groups
        num_paths = math.comb(n_groups - 1, k_test - 1)

        mean_oos = float(np.mean(oos_sharpes_of_best_is)) if oos_sharpes_of_best_is else 0.0
        med_oos = float(np.median(oos_sharpes_of_best_is)) if oos_sharpes_of_best_is else 0.0

        return CPCVResult(
            num_combinations=len(splits),
            num_backtest_paths=num_paths,
            pbo=round(pbo, 4),
            oos_sharpe_distribution=[round(x, 4) for x in oos_sharpes_of_best_is],
            mean_oos_sharpe=round(mean_oos, 4),
            median_oos_sharpe=round(med_oos, 4),
            logit_distribution=[round(x, 4) for x in logits],
            passed_pbo_gate=(pbo <= self.config.max_pbo_threshold and med_oos > 0.0),
        )
