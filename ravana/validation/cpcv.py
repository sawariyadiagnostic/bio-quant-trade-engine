"""Muscle 14: Combinatorial Purged Cross-Validation (CPCV) & PBO Engine.

Implements Marcos López de Prado's CPCV algorithm with target-overlap purging,
post-test embargoing, In-Sample frozen standardization, and Probability of
Backtest Overfitting (PBO) estimation across Muscle 9 populations.
"""
from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from ravana.compute.dispatcher import ComputeDispatcher
from ravana.domain.formula_ast import ASTNode
from ravana.evolution.config import Individual
from ravana.fitness.metrics import (
    compute_annualized_sharpe,
    compute_max_drawdown,
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
        max_lookback = getattr(self.config, "max_hold_lookback_bars", 150)

        splits: List[Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]] = []

        for test_group_ids in itertools.combinations(range(n_groups), k_test):
            test_idx = np.concatenate([groups[g] for g in test_group_ids])
            forbidden = np.zeros(n_samples, dtype=bool)
            forbidden[test_idx] = True

            for g in test_group_ids:
                g_start = int(groups[g][0])
                g_end = int(groups[g][-1])

                lookback = min(g_start, max_lookback)
                if hold_durations is not None and lookback > 0:
                    for idx in range(g_start - lookback, g_start):
                        hold = int(hold_durations[idx])
                        if idx + hold >= g_start:
                            forbidden[idx] = True
                else:
                    p_start = max(0, g_start - base_purge)
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

    def evaluate_multi_asset_pbo(
        self,
        candidates: Sequence[Individual | ASTNode],
        features_dict: Dict[str, pd.DataFrame],
        targets_dict: Dict[str, pd.DataFrame],
        asset_weights: Optional[Dict[str, float]] = None,
    ) -> CPCVResult:
        """Evaluate CPCV PBO across a multi-asset 4-sector ETF portfolio."""
        tickers = list(features_dict.keys())
        if not tickers:
            raise ValueError("features_dict cannot be empty for multi-asset CPCV.")

        n_assets = len(tickers)
        weights = asset_weights or {sym: 1.0 / float(n_assets) for sym in tickers}
        primary = tickers[0]
        n_samples = len(targets_dict[primary])
        n_strat = len(candidates)

        trees = [c.tree if isinstance(c, Individual) else c for c in candidates]
        portfolio_sigs = np.zeros((n_strat, n_samples), dtype=np.float64)
        portfolio_target = np.zeros(n_samples, dtype=np.float64)

        for sym in tickers:
            w = weights.get(sym, 1.0 / float(n_assets))
            raw = ComputeDispatcher.evaluate_batch(trees, features_dict[sym])
            if hasattr(raw, "cpu"):
                raw = raw.cpu().numpy()
            if isinstance(raw, np.ndarray) and raw.ndim == 1:
                raw = np.expand_dims(raw, axis=0)
            portfolio_sigs += w * raw
            portfolio_target += w * targets_dict[sym]["net_return"].to_numpy(dtype=np.float64)

        return self.evaluate_pbo(portfolio_sigs, portfolio_target)

    @staticmethod
    def _evaluate_split(
        signals_matrix: np.ndarray,
        target: np.ndarray,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        periods_per_yr: float,
    ) -> Tuple[float, float, float]:
        """Evaluate a single CPCV split returning (chosen_oos_sr, logit, oos_mdd)."""
        n_strategies = signals_matrix.shape[0]
        is_scores = np.zeros(n_strategies, dtype=np.float64)
        oos_scores = np.zeros(n_strategies, dtype=np.float64)
        y_train, y_test = target[train_idx], target[test_idx]

        for m in range(n_strategies):
            s_tr, s_te = signals_matrix[m, train_idx], signals_matrix[m, test_idx]
            v_tr = np.isfinite(s_tr)
            tr_m = float(np.mean(s_tr[v_tr])) if np.sum(v_tr) > 2 else 0.0
            tr_s = float(np.std(s_tr[v_tr], ddof=1)) if np.sum(v_tr) > 2 else 1.0
            r_is = compute_strategy_returns(s_tr, y_train, long_only=True)
            r_oos = compute_oos_strategy_returns(s_te, y_test, train_mean=tr_m, train_std=tr_s, long_only=True)
            is_scores[m] = compute_annualized_sharpe(r_is, periods_per_year=periods_per_yr)
            oos_scores[m] = compute_annualized_sharpe(r_oos, periods_per_year=periods_per_yr)

        best_idx = int(np.argmax(is_scores))
        chosen_sr = float(oos_scores[best_idx])
        if n_strategies > 1:
            less_c = float(np.sum(oos_scores < chosen_sr))
            eq_c = float(np.sum(oos_scores == chosen_sr))
            rank = 1.0 + less_c + 0.5 * max(0.0, eq_c - 1.0)
            omega = rank / float(n_strategies + 1)
            logit = math.log(omega / (1.0 - omega))
        else:
            logit = 1.0 if chosen_sr > 0.0 else -1.0

        b_te = signals_matrix[best_idx, test_idx]
        v_b = np.isfinite(signals_matrix[best_idx, train_idx])
        bm = float(np.mean(signals_matrix[best_idx, train_idx][v_b])) if np.sum(v_b) > 2 else 0.0
        bs = float(np.std(signals_matrix[best_idx, train_idx][v_b], ddof=1)) if np.sum(v_b) > 2 else 1.0
        b_rets = compute_oos_strategy_returns(b_te, y_test, train_mean=bm, train_std=bs, long_only=True)
        return chosen_sr, logit, compute_max_drawdown(b_rets)

    def evaluate_pbo(
        self,
        signals_matrix: np.ndarray,
        target: np.ndarray,
        hold_durations: Optional[np.ndarray] = None,
    ) -> CPCVResult:
        """Compute Probability of Backtest Overfitting (PBO) across M candidate strategies."""
        if signals_matrix.ndim == 1:
            signals_matrix = np.expand_dims(signals_matrix, axis=0)

        splits = self.generate_combinatorial_splits(signals_matrix.shape[1], hold_durations=hold_durations)
        logits: List[float] = []
        oos_sharpes: List[float] = []
        path_mdds: List[float] = []
        periods_per_yr = getattr(self.config, "periods_per_year", 252.0)

        for train_idx, test_idx, _ in splits:
            chosen_sr, logit, mdd = self._evaluate_split(
                signals_matrix, target, train_idx, test_idx, periods_per_yr
            )
            oos_sharpes.append(chosen_sr)
            logits.append(logit)
            path_mdds.append(mdd)

        logits_arr = np.array(logits, dtype=np.float64)
        pbo = float(np.mean(logits_arr <= 0.0)) if len(logits_arr) > 0 else 1.0
        num_paths = math.comb(self.config.n_groups - 1, self.config.k_test_groups - 1)
        mean_oos = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0
        med_oos = float(np.median(oos_sharpes)) if oos_sharpes else 0.0
        worst_mdd = float(np.max(path_mdds)) if path_mdds else 0.0

        return CPCVResult(
            num_combinations=len(splits),
            num_backtest_paths=num_paths,
            pbo=round(pbo, 4),
            oos_sharpe_distribution=[round(x, 4) for x in oos_sharpes],
            mean_oos_sharpe=round(mean_oos, 4),
            median_oos_sharpe=round(med_oos, 4),
            logit_distribution=[round(x, 4) for x in logits],
            passed_pbo_gate=(pbo <= self.config.max_pbo_threshold and med_oos > 0.0 and worst_mdd <= 0.25),
            synthetic_path_sharpes=[round(x, 4) for x in oos_sharpes],
            worst_path_mdd=round(worst_mdd, 5),
        )
