"""Muscle 13: Walk-Forward Analysis Engine.

Chains Muscle 7 (features_df), Muscle 8 (targets_df with calendar_hold_days),
and Muscle 9 (GeneticAlphaEngine). Evaluates rolling operators with causal warmup
and standardizes Out-of-Sample signals strictly using In-Sample (mean, std).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from ravana.compute.dispatcher import ComputeDispatcher
from ravana.domain.formula_ast import ASTNode
from ravana.evolution.config import EvolutionConfig
from ravana.evolution.engine import GeneticAlphaEngine
from ravana.fitness.metrics import (
    compute_annualized_sharpe,
    compute_ic_and_rank_ic,
    compute_max_drawdown,
    compute_oos_strategy_returns,
    compute_strategy_returns,
)
from ravana.validation.config import (
    WalkForwardConfig,
    WalkForwardFold,
    WalkForwardResult,
)


class WalkForwardEngine:
    """Chronological Walk-Forward validator preventing in-sample data snooping."""

    def __init__(self, config: Optional[WalkForwardConfig] = None) -> None:
        self.config = config or WalkForwardConfig()

    def _generate_session_splits(self, targets_df: pd.DataFrame, session_dates: pd.DatetimeIndex, k: int) -> List[Tuple[slice, slice]]:
        """Generate session-aware chronological splits respecting multi-day hold embargos."""
        n_sessions = len(session_dates)
        step_sess = n_sessions // (k + 1)
        splits: List[Tuple[slice, slice]] = []
        for i in range(k):
            tr_s_start = 0 if self.config.anchored else i * step_sess
            tr_s_end = (i + 1) * step_sess
            tr_dates = session_dates[tr_s_start:tr_s_end]
            embargo_sessions = 0
            if "calendar_hold_days" in targets_df.columns:
                tr_mask = targets_df.index.normalize().isin(tr_dates)
                sub_t = targets_df[tr_mask]
                if not sub_t.empty:
                    entry_col = sub_t["is_entry_bar"] if "is_entry_bar" in sub_t.columns else pd.Series(1, index=sub_t.index)
                    last_entries = sub_t[entry_col == 1]
                    if not last_entries.empty:
                        embargo_sessions = max(0, int(last_entries["calendar_hold_days"].iloc[-1]) - 1)

            te_s_start = tr_s_end + embargo_sessions
            if te_s_start >= n_sessions:
                continue
            te_s_end = n_sessions if i == k - 1 else min(n_sessions, (i + 2) * step_sess)
            if te_s_end <= te_s_start:
                te_s_end = min(n_sessions, te_s_start + max(1, step_sess))
            te_dates = session_dates[te_s_start:te_s_end]

            train_indices = np.where(targets_df.index.normalize().isin(tr_dates))[0]
            test_indices = np.where(targets_df.index.normalize().isin(te_dates))[0]
            if len(train_indices) == 0 or len(test_indices) == 0:
                continue
            t_start, t_end = int(train_indices[0]), int(train_indices[-1]) + 1
            te_start, te_end = int(test_indices[0]), int(test_indices[-1]) + 1
            if te_end > te_start and t_end > t_start:
                splits.append((slice(t_start, t_end), slice(te_start, te_end)))
        return splits

    def _generate_bar_splits(self, targets_df: pd.DataFrame, n_bars: int, k: int, base_purge: int) -> List[Tuple[slice, slice]]:
        """Fallback bar-based partitioning for synthetic or non-datetime indices."""
        step = n_bars // (k + 1)
        splits: List[Tuple[slice, slice]] = []
        for i in range(k):
            train_start = 0 if self.config.anchored else i * step
            train_end = (i + 1) * step
            extra_purge = 0
            if "calendar_hold_days" in targets_df.columns and train_end > 0:
                scan_start = max(0, train_end - 375)
                sub_targets = targets_df.iloc[scan_start:train_end]
                entry_col = sub_targets["is_entry_bar"] if "is_entry_bar" in sub_targets.columns else pd.Series(True, index=sub_targets.index)
                entry_indices = np.where(entry_col == 1)[0] + scan_start
                for e_idx in entry_indices:
                    hold_days = int(targets_df["calendar_hold_days"].iloc[e_idx])
                    overflow = (e_idx + max(1, hold_days * 75)) - train_end
                    if overflow > extra_purge:
                        extra_purge = overflow

            purge = max(base_purge, extra_purge)
            test_start = min(n_bars, train_end + purge)
            min_test_len = max(5, step // 4)
            test_end = n_bars if i == k - 1 else min(n_bars, max(test_start + min_test_len, (i + 2) * step))
            if test_end > test_start and train_end > train_start:
                splits.append((slice(train_start, train_end), slice(test_start, test_end)))
        return splits

    def generate_splits(self, targets_df: pd.DataFrame) -> List[Tuple[slice, slice]]:
        """Generate chronological (train_slice, test_slice) pairs respecting hold horizons."""
        n_bars = len(targets_df)
        k = max(1, self.config.n_splits)
        base_purge = max(0, self.config.purge_bars)

        if isinstance(targets_df.index, pd.DatetimeIndex):
            session_dates = targets_df.index.normalize().unique()
            if len(session_dates) >= k + 1:
                splits = self._generate_session_splits(targets_df, session_dates, k)
                if splits:
                    return splits

        if n_bars - base_purge * k < k * 10:
            raise ValueError(f"Insufficient bars ({n_bars}) for {k} splits with purge={base_purge}.")
        return self._generate_bar_splits(targets_df, n_bars, k, base_purge)

    def evaluate_formula_walk_forward(
        self,
        node: ASTNode,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
    ) -> WalkForwardResult:
        """Evaluate a fixed AST formula across walk-forward splits with zero OOS lookahead."""
        return self.run_walk_forward(
            features_df=features_df,
            targets_df=targets_df,
            discover_fn=lambda f_train, t_train: node,
        )

    def run_evolution_walk_forward(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        evo_config: Optional[EvolutionConfig] = None,
    ) -> WalkForwardResult:
        """Chain Muscle 9 GeneticAlphaEngine directly inside each Walk-Forward training fold."""
        cfg = evo_config or EvolutionConfig(population_size=32, max_generations=3)

        def _discover(f_tr: pd.DataFrame, t_tr: pd.DataFrame) -> ASTNode:
            engine = GeneticAlphaEngine(cfg)
            res = engine.evolve(f_tr, t_tr)
            champ = res.champion or res.best_individual
            return champ.tree if champ is not None else ASTNode("CONST", "0.0")

        return self.run_walk_forward(features_df, targets_df, discover_fn=_discover)

    def _evaluate_fold_metrics(
        self,
        fold_idx: int,
        train_sl: slice,
        test_sl: slice,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        discover_fn: Callable[[pd.DataFrame, pd.DataFrame], ASTNode],
    ) -> Tuple[WalkForwardFold, float, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate a single walk-forward fold returning fold telemetry and return arrays."""
        f_tr, t_tr = features_df.iloc[train_sl], targets_df.iloc[train_sl]
        f_te, t_te = features_df.iloc[test_sl], targets_df.iloc[test_sl]
        champ = discover_fn(f_tr, t_tr)

        is_sig, is_y, oos_sig, oos_y = self._evaluate_fold_causally(champ, features_df, targets_df, train_sl, test_sl)
        v_is = np.isfinite(is_sig)
        tr_m = float(np.mean(is_sig[v_is])) if np.sum(v_is) > 2 else 0.0
        tr_s = max(float(np.std(is_sig[v_is], ddof=1)) if np.sum(v_is) > 2 else 1.0, 1e-6)

        is_ic, _ = compute_ic_and_rank_ic(is_sig, is_y)
        oos_ic, _ = compute_ic_and_rank_ic(oos_sig, oos_y)
        is_rets = compute_strategy_returns(is_sig, is_y, long_only=True)
        oos_rets = compute_oos_strategy_returns(oos_sig, oos_y, train_mean=tr_m, train_std=tr_s, long_only=True)

        is_sr = compute_annualized_sharpe(is_rets, periods_per_year=self.config.periods_per_year, adjust_autocorrelation=self.config.adjust_autocorrelation)
        oos_sr = compute_annualized_sharpe(oos_rets, periods_per_year=self.config.periods_per_year, adjust_autocorrelation=self.config.adjust_autocorrelation)
        is_cum = float(np.prod(1.0 + is_rets) - 1.0) if len(is_rets) > 0 else 0.0
        oos_cum = float(np.prod(1.0 + oos_rets) - 1.0) if len(oos_rets) > 0 else 0.0
        wfe_fold = round(oos_sr / max(1e-4, is_sr), 4) if is_sr > 0 else 0.0

        fold = WalkForwardFold(
            fold_id=fold_idx, train_start=str(f_tr.index[0]), train_end=str(f_tr.index[-1]),
            test_start=str(f_te.index[0]), test_end=str(f_te.index[-1]), champion_formula=champ.to_infix(),
            is_ic=round(is_ic, 5), oos_ic=round(oos_ic, 5), is_sharpe=round(is_sr, 4),
            oos_sharpe=round(oos_sr, 4), oos_return=round(oos_cum, 5), wfe=wfe_fold, is_return=round(is_cum, 5),
        )
        return fold, is_sr, (oos_sig - tr_m) / tr_s, oos_y, oos_rets

    def run_walk_forward(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        discover_fn: Callable[[pd.DataFrame, pd.DataFrame], ASTNode],
    ) -> WalkForwardResult:
        """Run full walk-forward optimization: discover on Train, validate on OOS Test."""
        splits = self.generate_splits(targets_df)
        folds: List[WalkForwardFold] = []
        oos_z_sigs, oos_targs, oos_strat_rets, is_sharpes = [], [], [], []

        for fold_idx, (tr_sl, te_sl) in enumerate(splits):
            fold, is_sr, z_sig, y_oos, r_oos = self._evaluate_fold_metrics(
                fold_idx, tr_sl, te_sl, features_df, targets_df, discover_fn
            )
            folds.append(fold)
            is_sharpes.append(is_sr)
            oos_z_sigs.append(z_sig)
            oos_targs.append(y_oos)
            oos_strat_rets.append(r_oos)

        stitched_sig = np.concatenate(oos_z_sigs) if oos_z_sigs else np.zeros(0)
        stitched_y = np.concatenate(oos_targs) if oos_targs else np.zeros(0)
        stitched_rets = np.concatenate(oos_strat_rets) if oos_strat_rets else np.zeros(0)
        stitched_equity = np.cumprod(1.0 + stitched_rets) if len(stitched_rets) > 0 else np.ones(1)

        agg_ic, agg_rank_ic = compute_ic_and_rank_ic(stitched_sig, stitched_y)
        agg_sharpe = compute_annualized_sharpe(stitched_rets, periods_per_year=self.config.periods_per_year, adjust_autocorrelation=self.config.adjust_autocorrelation)
        agg_mdd = compute_max_drawdown(stitched_rets)

        mean_is = float(np.mean(is_sharpes)) if is_sharpes else 1e-6
        wfe = float(agg_sharpe / max(abs(mean_is), 1e-4)) if mean_is > 0 else 0.0
        fold_wfes = [f.wfe for f in folds]
        min_fold_wfe = round(min(fold_wfes), 4) if fold_wfes else 0.0
        consistency = round(sum(1 for f in folds if f.oos_sharpe > 0) / float(len(folds)), 4) if folds else 0.0

        passed = (agg_sharpe >= self.config.min_oos_sharpe and agg_ic >= self.config.min_oos_ic and wfe >= self.config.min_wfe)

        return WalkForwardResult(
            folds=folds, stitched_oos_returns=stitched_rets, stitched_oos_equity=stitched_equity,
            aggregate_oos_ic=round(agg_ic, 5), aggregate_oos_rank_ic=round(agg_rank_ic, 5),
            aggregate_oos_sharpe=round(agg_sharpe, 4), aggregate_oos_mdd=round(agg_mdd, 5),
            walk_forward_efficiency=round(wfe, 4), passed_validation=passed,
            min_fold_wfe=min_fold_wfe, consistency_score=consistency,
        )

    def compare_anchored_vs_rolling(
        self,
        node: ASTNode,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Compare anchored (expanding) vs rolling walk-forward efficiency."""
        cfg_roll = self.config
        cfg_anch = WalkForwardConfig(
            n_splits=self.config.n_splits,
            anchored=True,
            purge_bars=self.config.purge_bars,
            min_oos_sharpe=self.config.min_oos_sharpe,
            min_oos_ic=self.config.min_oos_ic,
            min_wfe=self.config.min_wfe,
            periods_per_year=self.config.periods_per_year,
            adjust_autocorrelation=self.config.adjust_autocorrelation,
        )
        res_roll = self.run_walk_forward(features_df, targets_df, lambda f, t: node)
        res_anch = WalkForwardEngine(cfg_anch).run_walk_forward(features_df, targets_df, lambda f, t: node)
        return {
            "rolling_wfe": res_roll.walk_forward_efficiency,
            "anchored_wfe": res_anch.walk_forward_efficiency,
            "rolling_sharpe": res_roll.aggregate_oos_sharpe,
            "anchored_sharpe": res_anch.aggregate_oos_sharpe,
            "regime_memory_decay": bool(res_roll.walk_forward_efficiency > res_anch.walk_forward_efficiency * 1.25),
        }

    @staticmethod
    def _evaluate_fold_causally(
        node: ASTNode,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        train_sl: slice,
        test_sl: slice,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate AST node with causal warmup so rolling operators don't cold-start at test_sl."""
        warmup = 75  # 1 full session of 5-minute bars
        warm_is_start = max(0, train_sl.start - warmup)
        offset_is = train_sl.start - warm_is_start
        f_warm_train = features_df.iloc[warm_is_start : train_sl.stop]
        raw_is = ComputeDispatcher.evaluate(node, f_warm_train, backend="numpy")
        if hasattr(raw_is, "cpu"):
            raw_is = raw_is.cpu().numpy()
        is_arr = np.asarray(raw_is, dtype=np.float64)[offset_is:]

        warm_start = max(0, test_sl.start - warmup)
        offset = test_sl.start - warm_start
        f_warm_test = features_df.iloc[warm_start : test_sl.stop]
        raw_oos_warm = ComputeDispatcher.evaluate(node, f_warm_test, backend="numpy")
        if hasattr(raw_oos_warm, "cpu"):
            raw_oos_warm = raw_oos_warm.cpu().numpy()
        oos_arr = np.asarray(raw_oos_warm, dtype=np.float64)[offset:]

        t_train = targets_df.iloc[train_sl]
        t_test = targets_df.iloc[test_sl]
        mask_is = (
            t_train["is_entry_bar"].to_numpy(dtype=bool)
            if "is_entry_bar" in t_train.columns
            else np.ones(len(t_train), dtype=bool)
        )
        mask_oos = (
            t_test["is_entry_bar"].to_numpy(dtype=bool)
            if "is_entry_bar" in t_test.columns
            else np.ones(len(t_test), dtype=bool)
        )
        return (
            is_arr[mask_is],
            t_train["net_return"].to_numpy(dtype=np.float64)[mask_is],
            oos_arr[mask_oos],
            t_test["net_return"].to_numpy(dtype=np.float64)[mask_oos],
        )
