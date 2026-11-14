"""Unified Multi-Objective Alpha Fitness Engine (Muscle 10).

Integrates Information Coefficient, Rank IC, Deflated Sharpe Ratio (DSR),
Benjamini-Hochberg FDR control, Turnover penalty, Max Drawdown, and Pareto frontier selection.
"""
from __future__ import annotations

from typing import List, Optional
import numpy as np

from ravana.domain.formula_ast import ASTNode
from ravana.fitness.config import AlphaMetrics, FitnessConfig
from ravana.fitness.deflated_sharpe import compute_dsr
from ravana.fitness.fdr import benjamini_hochberg_fdr, ic_to_p_value
from ravana.fitness.metrics import (
    compute_annualized_sharpe,
    compute_ic_and_rank_ic,
    compute_max_drawdown,
    compute_strategy_returns,
    compute_turnover,
)
from ravana.fitness.pareto import find_pareto_frontier


class MultiObjectiveFitnessEngine:
    """Evaluates alpha formulas across multi-objective econometric and statistical dimensions."""

    def __init__(self, config: Optional[FitnessConfig] = None) -> None:
        self.config = config or FitnessConfig()

    def evaluate_single(
        self,
        signal: np.ndarray,
        target: np.ndarray,
        node: Optional[ASTNode] = None,
        formula_str: Optional[str] = None,
        empirical_var_annual: Optional[float] = None,
    ) -> AlphaMetrics:
        """Evaluate a single alpha formula signal vector against forward targets."""
        f_str = formula_str or (node.to_infix() if node else "CUSTOM_SIGNAL")
        node_count = node.size() if node else 1

        ic, rank_ic = compute_ic_and_rank_ic(signal, target)
        strat_rets = compute_strategy_returns(signal, target, long_only=True)
        ann_sharpe = compute_annualized_sharpe(
            strat_rets,
            rf_annual=self.config.risk_free_rate,
            periods_per_year=self.config.annualization_factor,
        )
        var_trials = (
            empirical_var_annual
            if (empirical_var_annual is not None and empirical_var_annual > 1e-6)
            else self.config.variance_of_trials
        )
        dsr_val = compute_dsr(
            strat_rets,
            num_trials=self.config.num_tested_trials,
            variance_of_trials=var_trials,
            rf_per_period=self.config.risk_free_rate / self.config.annualization_factor,
            periods_per_year=self.config.annualization_factor,
            is_variance_annualized=True,
        )
        p_val = ic_to_p_value(ic, len(signal))
        to_val = compute_turnover(signal)
        mdd_val = compute_max_drawdown(strat_rets)

        composite_fit = self._calc_composite_fitness(
            ic=ic,
            rank_ic=rank_ic,
            dsr=dsr_val,
            turnover=to_val,
            mdd=mdd_val,
            node_count=node_count,
        )

        return AlphaMetrics(
            formula_str=f_str,
            node_count=node_count,
            ic=ic,
            rank_ic=rank_ic,
            annualized_sharpe=ann_sharpe,
            dsr=dsr_val,
            p_value=p_val,
            passes_fdr=p_val <= self.config.fdr_alpha,
            turnover=to_val,
            max_drawdown=mdd_val,
            composite_fitness=composite_fit,
        )

    def evaluate_population(
        self,
        signals_matrix: np.ndarray,
        target: np.ndarray,
        nodes: Optional[List[ASTNode]] = None,
        formulas: Optional[List[str]] = None,
        sort_by_fitness: bool = True,
    ) -> List[AlphaMetrics]:
        """Evaluate a population of alphas with empirical trial variance DSR, FDR, and Pareto sorting."""
        n_alphas = signals_matrix.shape[0] if signals_matrix.ndim == 2 else 0
        if n_alphas == 0:
            return []

        # Pre-compute empirical variance of annualized Sharpe ratios across the population
        emp_var: Optional[float] = None
        if n_alphas >= 4:
            pop_sharpes = [
                compute_annualized_sharpe(
                    compute_strategy_returns(signals_matrix[i], target, long_only=True),
                    rf_annual=self.config.risk_free_rate,
                    periods_per_year=self.config.annualization_factor,
                )
                for i in range(n_alphas)
            ]
            emp_var = max(0.05, float(np.var(pop_sharpes, ddof=1)))

        metrics_list: List[AlphaMetrics] = []
        for i in range(n_alphas):
            node = nodes[i] if nodes and i < len(nodes) else None
            f_str = formulas[i] if formulas and i < len(formulas) else None
            m = self.evaluate_single(
                signals_matrix[i], target, node=node, formula_str=f_str, empirical_var_annual=emp_var
            )
            metrics_list.append(m)

        p_vals = [m.p_value for m in metrics_list]
        fdr_results = benjamini_hochberg_fdr(p_vals, q_alpha=self.config.fdr_alpha)
        for i, passed in enumerate(fdr_results):
            metrics_list[i].passes_fdr = passed

        objs = [
            (m.ic, m.rank_ic, m.dsr, m.turnover, m.max_drawdown, float(m.node_count))
            for m in metrics_list
        ]
        pareto_indices = set(find_pareto_frontier(objs))
        for i in pareto_indices:
            metrics_list[i].is_pareto_optimal = True

        if sort_by_fitness:
            metrics_list.sort(key=lambda x: x.composite_fitness, reverse=True)
        return metrics_list

    def _calc_composite_fitness(
        self,
        ic: float,
        rank_ic: float,
        dsr: float,
        turnover: float,
        mdd: float,
        node_count: int,
    ) -> float:
        """Calculate weighted composite scalar fitness."""
        w = self.config.weights
        parsimony = node_count * w.parsimony_penalty_per_node
        fit = (
            w.w_ic * ic
            + w.w_rank_ic * rank_ic
            + w.w_dsr * dsr
            - w.w_turnover * turnover
            - w.w_mdd * mdd
            - parsimony
        )
        return float(fit)
