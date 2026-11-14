"""Unit tests for Phase 5 (Muscles 13-16): Quantitative Rigor & Overfitting Defense."""
import unittest
import numpy as np
import pandas as pd

from ravana.domain.formula_ast import ASTNode
from ravana.evolution.config import EvolutionResult, Individual
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
from ravana.validation.pipeline import ValidationPipeline
from ravana.validation.walk_forward import WalkForwardEngine


class TestValidationEngines(unittest.TestCase):
    """Test suite for Walk-Forward (M13), CPCV (M14), DSR/MinTRL Guard (M15), and Monte Carlo (M16)."""

    def setUp(self) -> None:
        np.random.seed(42)
        self.n_bars = 400
        dates = pd.date_range("2025-01-01 09:15", periods=self.n_bars, freq="5min", tz="Asia/Kolkata")
        signal = np.sin(np.linspace(0, 20, self.n_bars)) + np.random.normal(0, 0.2, self.n_bars)
        noise = np.random.normal(0, 1.0, self.n_bars)

        self.features_df = pd.DataFrame(
            {"LOG_RET_1": signal, "VWAP_DRIFT": noise},
            index=dates,
        )
        # True predictive relationship with LOG_RET_1
        net_return = 0.005 * signal + np.random.normal(0, 0.002, self.n_bars)
        is_entry = np.ones(self.n_bars, dtype=np.int32)

        self.targets_df = pd.DataFrame(
            {
                "net_return": net_return,
                "is_entry_bar": is_entry,
                "calendar_hold_days": np.ones(self.n_bars, dtype=np.int32),
            },
            index=dates,
        )

    def test_muscle13_walk_forward_true_signal_vs_noise(self) -> None:
        """Verify Muscle 13 passes a genuine out-of-sample signal and rejects pure noise."""
        cfg = WalkForwardConfig(n_splits=4, purge_bars=10, min_oos_sharpe=0.5, min_oos_ic=0.05)
        wf_engine = WalkForwardEngine(cfg)

        true_node = ASTNode("FEATURE", val="LOG_RET_1")
        res_true = wf_engine.evaluate_formula_walk_forward(true_node, self.features_df, self.targets_df)

        self.assertEqual(len(res_true.folds), 4)
        self.assertGreater(res_true.aggregate_oos_ic, 0.30)
        self.assertGreater(res_true.aggregate_oos_sharpe, 1.0)
        self.assertTrue(res_true.passed_validation)

        noise_node = ASTNode("FEATURE", val="VWAP_DRIFT")
        res_noise = wf_engine.evaluate_formula_walk_forward(noise_node, self.features_df, self.targets_df)
        self.assertFalse(res_noise.passed_validation)

    def test_muscle14_cpcv_purging_and_pbo(self) -> None:
        """Verify Muscle 14 generates C(6,2)=15 purged splits and computes PBO accurately."""
        cfg = CPCVConfig(n_groups=6, k_test_groups=2, purge_window=2, embargo_pct=0.02)
        cpcv = CombinatorialPurgedCV(cfg)

        splits = cpcv.generate_combinatorial_splits(120)
        self.assertEqual(len(splits), 15)  # C(6, 2) = 15

        for train_idx, test_idx, _ in splits:
            overlap = np.intersect1d(train_idx, test_idx)
            self.assertEqual(len(overlap), 0)

        # Evaluate PBO when 1 strong genuine strategy is mixed with 5 weak strategies
        y = self.targets_df["net_return"].to_numpy()
        strong_sig = self.features_df["LOG_RET_1"].to_numpy()
        noise_sigs = np.random.normal(0, 1.0, size=(5, len(y)))
        mat = np.vstack([strong_sig, noise_sigs])

        res = cpcv.evaluate_pbo(mat, y)
        self.assertEqual(res.num_combinations, 15)
        self.assertEqual(res.num_backtest_paths, 5)  # C(5, 1) = 5
        self.assertLessEqual(res.pbo, 0.20)
        self.assertTrue(res.passed_pbo_gate)

    def test_muscle14_cpcv_evaluate_population_pbo(self) -> None:
        """Verify Muscle 14 evaluates candidates directly from AST nodes and dataframes."""
        cfg = CPCVConfig(n_groups=6, k_test_groups=2, purge_window=2, embargo_pct=0.02)
        cpcv = CombinatorialPurgedCV(cfg)

        candidates = [
            ASTNode("FEATURE", val="LOG_RET_1"),
            ASTNode("FEATURE", val="VWAP_DRIFT"),
            ASTNode("CONST", val="0.0"),
        ]
        res = cpcv.evaluate_population_pbo(candidates, self.features_df, self.targets_df)
        self.assertEqual(res.num_combinations, 15)
        self.assertIsInstance(res.pbo, float)
        self.assertTrue(0.0 <= res.pbo <= 1.0)

    def test_muscle15_dsr_and_mintrl_catches_61_day_overfit(self) -> None:
        """Verify Muscle 15 rejects a 61-day sample searched across 7,500 trials."""
        guard = OverfittingGuardEngine(DSRGuardConfig(confidence_level=0.95, min_dsr_threshold=0.95))

        np.random.seed(7)
        rets_61d = np.random.normal(0.0012, 0.012, 61)
        report = guard.audit_strategy(rets_61d, num_trials=7500)

        self.assertFalse(report.passed_guard)
        self.assertFalse(report.has_sufficient_track_record)
        self.assertGreater(report.min_track_record_length, 61.0)
        self.assertGreater(len(report.rejection_reasons), 0)

    def test_muscle15_dsr_passes_long_track_record(self) -> None:
        """Verify Muscle 15 passes a strong strategy with 750+ days of track record."""
        guard = OverfittingGuardEngine(DSRGuardConfig(confidence_level=0.95, min_dsr_threshold=0.95))
        np.random.seed(42)
        rets_750d = np.random.normal(0.0025, 0.008, 750)
        report = guard.audit_strategy(rets_750d, num_trials=50)

        self.assertTrue(report.has_sufficient_track_record)
        self.assertGreaterEqual(report.deflated_sharpe_ratio, 0.95)
        self.assertTrue(report.passed_guard)

    def test_muscle16_monte_carlo_block_bootstrap(self) -> None:
        """Verify Muscle 16 computes VaR, CVaR, drawdown percentiles, and probability of ruin."""
        mc = MonteCarloStressTester(
            MonteCarloConfig(num_simulations=500, ruin_drawdown_threshold=0.08, random_seed=42)
        )

        safe_rets = np.random.normal(0.0015, 0.003, 250)
        rep_safe = mc.run_stress_test(safe_rets)
        self.assertEqual(rep_safe.num_simulations, 500)
        self.assertLessEqual(rep_safe.cvar_95, rep_safe.var_95)
        self.assertLess(rep_safe.probability_of_ruin, 0.05)
        self.assertTrue(rep_safe.passed_stress_test)

        risky_rets = np.random.normal(-0.002, 0.025, 250)
        rep_risky = mc.run_stress_test(risky_rets)
        self.assertGreater(rep_risky.probability_of_ruin, 0.50)
        self.assertFalse(rep_risky.passed_stress_test)

    def test_validation_pipeline_end_to_end(self) -> None:
        """Verify ValidationPipeline chains M13, M14, M15, M16 into a ValidationCertificate."""
        pipe = ValidationPipeline(
            wf_config=WalkForwardConfig(n_splits=3, purge_bars=5, min_oos_sharpe=0.5, min_oos_ic=0.05),
            cpcv_config=CPCVConfig(n_groups=5, k_test_groups=2, purge_window=1),
            dsr_config=DSRGuardConfig(min_dsr_threshold=0.50),
            mc_config=MonteCarloConfig(num_simulations=200, random_seed=42),
        )

        champ_tree = ASTNode("FEATURE", val="LOG_RET_1")
        mock_ind = Individual(tree=champ_tree, fitness=0.85, ic=0.45)
        mock_evo = EvolutionResult(
            champion=mock_ind,
            best_individual=mock_ind,
            final_population=[mock_ind, Individual(tree=ASTNode("CONST", val="0.0"))],
            generation_history=[],
            graveyard_count=0,
            total_trials_evaluated=20,
        )

        cert = pipe.certify_evolution_result(
            ticker="TEST_TICKER",
            evolution_result=mock_evo,
            features_df=self.features_df,
            targets_df=self.targets_df,
        )

        self.assertIsInstance(cert, ValidationCertificate)
        self.assertEqual(cert.ticker, "TEST_TICKER")
        self.assertEqual(cert.champion_formula, "LOG_RET_1")
        self.assertIsInstance(cert.to_dict(), dict)
        self.assertTrue(cert.walk_forward.passed_validation)


if __name__ == "__main__":
    unittest.main()
