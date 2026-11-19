"""Unified Phase 5 Validation Chain Orchestrator.

Connects Muscle 7 (features_df), Muscle 8 (targets_df), Muscle 9 (EvolutionResult),
Muscle 13 (Walk-Forward), Muscle 14 (CPCV PBO), Muscle 15 (DSR & MinTRL Guard),
and Muscle 16 (Monte Carlo Stress Tester) into a single end-to-end certification chain.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from typing import Optional
import numpy as np
import pandas as pd

from ravana.domain.formula_ast import ASTNode
from ravana.evolution.config import EvolutionResult
from ravana.validation.config import (
    CPCVConfig,
    CPCVResult,
    DSRGuardConfig,
    DSRGuardReport,
    MonteCarloConfig,
    MonteCarloReport,
    ValidationCertificate,
    WalkForwardConfig,
    WalkForwardResult,
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

    def _run_m13(self, node: ASTNode, feat_df: pd.DataFrame, targ_df: pd.DataFrame) -> WalkForwardResult:
        """Run Muscle 13 Walk-Forward validation with safe fallback."""
        try:
            return self.wf_engine.evaluate_formula_walk_forward(node, feat_df, targ_df)
        except Exception:
            return WalkForwardResult(
                folds=[], stitched_oos_returns=np.zeros(0), stitched_oos_equity=np.ones(1),
                aggregate_oos_ic=0.0, aggregate_oos_rank_ic=0.0, aggregate_oos_sharpe=0.0,
                aggregate_oos_mdd=1.0, walk_forward_efficiency=0.0, passed_validation=False,
            )

    def _run_m14(self, candidates: list, feat_df: pd.DataFrame, targ_df: pd.DataFrame) -> CPCVResult:
        """Run Muscle 14 CPCV PBO validation with safe fallback."""
        try:
            return self.cpcv_engine.evaluate_population_pbo(candidates, feat_df, targ_df)
        except Exception:
            return CPCVResult(
                num_combinations=0, num_backtest_paths=0, pbo=1.0, oos_sharpe_distribution=[],
                mean_oos_sharpe=0.0, median_oos_sharpe=0.0, logit_distribution=[], passed_pbo_gate=False,
            )

    def _run_m15(self, oos_rets: np.ndarray, total_trials: int) -> DSRGuardReport:
        """Run Muscle 15 Deflated Sharpe Ratio audit with safe fallback."""
        try:
            return self.dsr_engine.audit_strategy(oos_rets, num_trials=max(1, total_trials))
        except Exception as e:
            return DSRGuardReport(
                observed_sharpe_annual=0.0, observed_sharpe_per_period=0.0,
                expected_max_sharpe_per_period=0.0, expected_max_sharpe_annual=0.0,
                empirical_trial_std_per_period=0.0, num_independent_trials=max(1, total_trials),
                sample_length_bars=len(oos_rets), min_track_record_length=float("inf"),
                deflated_sharpe_ratio=0.0, skewness=0.0, kurtosis=3.0,
                has_sufficient_track_record=False, passed_guard=False,
                rejection_reasons=[f"Exception in DSR audit: {str(e)}"],
            )

    def _run_m16(self, wf_res: WalkForwardResult) -> MonteCarloReport:
        """Run Muscle 16 Monte Carlo block-bootstrap stress test with safe fallback."""
        try:
            return self.mc_engine.run_from_result(wf_res)
        except Exception:
            return MonteCarloReport(
                num_simulations=0, block_size=1, median_cagr=0.0, var_95=0.0, var_99=0.0,
                cvar_95=0.0, median_max_drawdown=1.0, dd_95_percentile=1.0, dd_99_percentile=1.0,
                probability_of_caution=1.0, probability_of_ruin=1.0, passed_stress_test=False,
            )

    def certify_evolution_result(
        self,
        ticker: str,
        evolution_result: EvolutionResult,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        persist_audit: bool = True,
    ) -> ValidationCertificate:
        """Run Muscles 13, 14, 15, and 16 sequentially on an EvolutionResult."""
        champ_ind = evolution_result.champion or evolution_result.best_individual
        champ_node = champ_ind.tree if champ_ind is not None else ASTNode("CONST", "0.0")
        candidates = evolution_result.final_population[:16] if evolution_result.final_population else [champ_node]

        wf_res = self._run_m13(champ_node, features_df, targets_df)
        cpcv_res = self._run_m14(candidates, features_df, targets_df)
        dsr_res = self._run_m15(wf_res.stitched_oos_returns, evolution_result.total_trials_evaluated)
        mc_res = self._run_m16(wf_res)

        is_certified = (
            wf_res.passed_validation and cpcv_res.passed_pbo_gate
            and dsr_res.passed_guard and mc_res.passed_stress_test
        )
        cert = ValidationCertificate(
            ticker=ticker, champion_formula=champ_node.to_infix(), walk_forward=wf_res,
            cpcv=cpcv_res, dsr_guard=dsr_res, monte_carlo=mc_res, is_certified_for_live=is_certified,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )
        cert_final = ValidationCertificate(
            ticker=cert.ticker, champion_formula=cert.champion_formula, walk_forward=cert.walk_forward,
            cpcv=cert.cpcv, dsr_guard=cert.dsr_guard, monte_carlo=cert.monte_carlo,
            is_certified_for_live=cert.is_certified_for_live, certificate_hash=cert.compute_sha256(),
            timestamp_utc=cert.timestamp_utc,
        )
        if persist_audit:
            self.save_certificate(cert_final)
        return cert_final

    @staticmethod
    def save_certificate(cert: ValidationCertificate, path: str = "audit/validation_cert.json") -> str:
        """Persist validation certificate as audit trail."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cert.to_dict(), f, indent=2)
        return path
