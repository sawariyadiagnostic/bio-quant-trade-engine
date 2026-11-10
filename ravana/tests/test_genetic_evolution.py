"""Tests for Muscle 9: Genetic Alpha Evolution & Search Engine.

Verifies:
1. Ramped population initialization with strict depth and syntax bounds.
2. Vectorized batch fitness evaluation (IC and parsimony penalty) using Muscle 2.
3. Idea Graveyard tracking and deduplication of failed hypotheses.
4. Challenger-Champion v1.1 protocol with atomic mutation and promotion hurdle.
5. Generational evolution step with elitism, tournament selection, and convergence.
"""

import unittest
import numpy as np
import pandas as pd
import torch

from ravana.domain.formula_ast import ASTNode
from ravana.domain.formula_parser import FormulaParser
from ravana.evolution.config import EvolutionConfig
from ravana.evolution.graveyard import IdeaGraveyard
from ravana.evolution.engine import GeneticAlphaEngine


class TestGeneticAlphaEvolution(unittest.TestCase):
    """Test suite for Genetic Alpha Evolution engine components."""

    def setUp(self) -> None:
        np.random.seed(42)
        torch.manual_seed(42)

        # Create synthetic feature matrix with 100 bars and 3 features
        dates = pd.date_range("2026-01-01 09:15", periods=100, freq="5min", tz="Asia/Kolkata")
        f1 = np.sin(np.linspace(0, 10, 100))
        f2 = np.cos(np.linspace(0, 10, 100))
        f3 = np.random.normal(0, 1, 100)

        self.features_df = pd.DataFrame(
            {"LOG_RET_1": f1, "VWAP_DRIFT": f2, "RSI_14": f3},
            index=dates,
        )

        # Target: closely aligned with f1 (positive correlation with LOG_RET_1)
        targets = f1 * 0.01 + np.random.normal(0, 0.001, 100)
        is_entry = np.zeros(100, dtype=np.int32)
        is_entry[10::5] = 1  # 18 entry bars

        self.targets_df = pd.DataFrame(
            {
                "net_return": targets,
                "is_entry_bar": is_entry,
                "sample_weight": is_entry.astype(np.float64),
            },
            index=dates,
        )

        self.config = EvolutionConfig(
            population_size=16,
            max_generations=3,
            tournament_size=3,
            mutation_prob=0.3,
            crossover_prob=0.6,
            elitism_count=2,
            max_depth=4,
            parsimony_penalty=0.001,
            champion_hurdle=0.02,
            feature_names=["LOG_RET_1", "VWAP_DRIFT", "RSI_14"],
        )

    def test_idea_graveyard_deduplication(self) -> None:
        """Verify IdeaGraveyard records visited formulas and rejects duplicates."""
        graveyard = IdeaGraveyard()
        f1 = "ADD(LOG_RET_1, VWAP_DRIFT)"
        f2 = "SUB(RSI_14, LOG_RET_1)"

        self.assertFalse(graveyard.contains(f1))
        graveyard.record_rejection(f1, reason="Negative IC", ic=-0.12)
        self.assertTrue(graveyard.contains(f1))
        self.assertFalse(graveyard.contains(f2))

        # Check graveyard records
        records = graveyard.get_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["formula"], f1)
        self.assertEqual(records[0]["reason"], "Negative IC")

    def test_population_initialization(self) -> None:
        """Verify population initializes to exact size with valid depth-bounded ASTs."""
        engine = GeneticAlphaEngine(self.config)
        pop = engine.initialize_population()

        self.assertEqual(len(pop), self.config.population_size)
        for ind in pop:
            self.assertIsInstance(ind.tree, ASTNode)
            self.assertLessEqual(ind.tree.depth(), self.config.max_depth)
            self.assertTrue(len(ind.tree.to_infix()) > 0)

    def test_fitness_evaluation_and_ranking(self) -> None:
        """Verify individual fitness evaluates IC against forward targets."""
        engine = GeneticAlphaEngine(self.config)
        pop = engine.initialize_population()

        evaluated = engine.evaluate_population(pop, self.features_df, self.targets_df)
        self.assertEqual(len(evaluated), len(pop))

        # Check metrics are computed
        for ind in evaluated:
            self.assertIsNotNone(ind.fitness)
            self.assertIsNotNone(ind.ic)
            self.assertTrue(np.isfinite(ind.fitness))
            self.assertTrue(np.isfinite(ind.ic))

        # Verify sorted descending by fitness
        for i in range(len(evaluated) - 1):
            self.assertGreaterEqual(evaluated[i].fitness, evaluated[i + 1].fitness)

    def test_challenger_champion_promotion(self) -> None:
        """Verify Challenger beats Champion only when exceeding the hurdle."""
        engine = GeneticAlphaEngine(self.config)

        # Baseline champion
        champ_tree = FormulaParser.parse("LOG_RET_1")
        engine.set_champion(champ_tree, fitness=0.50)

        # Weak challenger (+0.01 improvement < 0.02 hurdle) -> Reject
        challenger_weak = FormulaParser.parse("ABS(LOG_RET_1)")
        promoted = engine.consider_challenger(challenger_weak, fitness=0.51)
        self.assertFalse(promoted)
        self.assertEqual(engine.champion.tree.to_infix(), "LOG_RET_1")
        self.assertTrue(engine.graveyard.contains(challenger_weak.to_infix()))

        # Strong challenger (+0.05 improvement >= 0.02 hurdle) -> Promote
        challenger_strong = FormulaParser.parse("MUL(LOG_RET_1, 2.0)")
        promoted = engine.consider_challenger(challenger_strong, fitness=0.55)
        self.assertTrue(promoted)
        self.assertEqual(engine.champion.tree.to_infix(), challenger_strong.to_infix())
        self.assertEqual(engine.champion.fitness, 0.55)

    def test_generational_step_convergence(self) -> None:
        """Verify evolutionary step preserves elites and runs across generations."""
        engine = GeneticAlphaEngine(self.config)
        result = engine.evolve(self.features_df, self.targets_df, generations=2)

        self.assertEqual(len(result.final_population), self.config.population_size)
        self.assertIsNotNone(result.best_individual)
        self.assertGreater(len(result.generation_history), 0)

        # Elites must be preserved across generations
        best_gen0 = result.generation_history[0]["best_fitness"]
        best_gen1 = result.generation_history[1]["best_fitness"]
        self.assertGreaterEqual(best_gen1, best_gen0 - 1e-6)


if __name__ == "__main__":
    unittest.main()
