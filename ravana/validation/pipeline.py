"""Unified Phase 5 Validation Chain Orchestrator.

Connects Muscle 7 (features_df), Muscle 8 (targets_df), Muscle 9 (EvolutionResult),
Muscle 13 (Walk-Forward), Muscle 14 (CPCV PBO), Muscle 15 (DSR & MinTRL Guard),
and Muscle 16 (Monte Carlo Stress Tester) into a single end-to-end certification chain.
"""
from __future__ import annotations

from typing import Optional
import pandas as pd

from ravana.domain.formula_ast import ASTNode
from ravana.evolution.config import EvolutionResult
from ravana.validation.config import (
    CPCVConfig,
    DSRGuardConfig,
    MonteCarloConfig,
    ValidationCertificate,
    WalkForwardConfig,
)
from ravana.validation.cpcv import CombinatorialPurgedCV
from ravana.validation.dsr_guard import OverfittingGuardEngine
from ravana.validation.monte_carlo import MonteCarloStressTester
from ravana.validation.walk_forward import WalkForwardEngine


class ValidationPipeline:
    """Executes the complete Phase 5 (Muscles 13-16) overfitting defense chain."""

    def __init__(
        self,
        wf_config: Optional[WalkForwardConfig] = None,
        cpcv_config: Optional[CPCVConfig] = None,
        dsr_config: Optional[DSRGuardConfig] = None,
        mc_config: Optional[MonteCarloConfig] = None,
    ) -> None:
        self.wf_engine = WalkForwardEngine(wf_config)
        self.cpcv_engine = CombinatorialPurgedCV(cpcv_config)
        self.dsr_engine = OverfittingGuardEngine(dsr_config)
        self.mc_engine = MonteCarloStressTester(mc_config)

    def certify_evolution_result(
        self,
        ticker: str,
        evolution_result: EvolutionResult,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
    ) -> ValidationCertificate:
        """Run Muscles 13, 14, 15, and 16 sequentially on an EvolutionResult."""
        champ_ind = evolution_result.champion or evolution_result.best_individual
        champ_node = champ_ind.tree if champ_ind is not None else ASTNode("CONST", "0.0")

        # 1. Muscle 13: Out-of-Sample Walk-Forward Analysis
        wf_res = self.wf_engine.evaluate_formula_walk_forward(
            champ_node, features_df, targets_df
        )

        # 2. Muscle 14: Combinatorial Purged Cross-Validation (PBO) across final population
        candidates = evolution_result.final_population[:16] if evolution_result.final_population else [champ_node]
        cpcv_res = self.cpcv_engine.evaluate_population_pbo(
            candidates, features_df, targets_df
        )

        # 3. Muscle 15: Deflated Sharpe Ratio & MinTRL Guard on stitched OOS returns
        num_trials = max(1, evolution_result.total_trials_evaluated)
        dsr_res = self.dsr_engine.audit_strategy(
            wf_res.stitched_oos_returns,
            num_trials=num_trials,
        )

        # 4. Muscle 16: Monte Carlo Block-Bootstrap Stress Test on stitched OOS returns
        mc_res = self.mc_engine.run_from_result(wf_res)

        is_certified = (
            wf_res.passed_validation
            and cpcv_res.passed_pbo_gate
            and dsr_res.passed_guard
            and mc_res.passed_stress_test
        )

        return ValidationCertificate(
            ticker=ticker,
            champion_formula=champ_node.to_infix(),
            walk_forward=wf_res,
            cpcv=cpcv_res,
            dsr_guard=dsr_res,
            monte_carlo=mc_res,
            is_certified_for_live=is_certified,
        )
