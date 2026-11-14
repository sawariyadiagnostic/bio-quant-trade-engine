"""Genetic Alpha Evolution Engine for Muscle 9.

Chains Muscle 1 (TreeGenerator/GeneticMutator with active vocabulary),
Muscle 2 (ComputeDispatcher), Muscle 7 (features_df), Muscle 8 (targets_df),
and Muscle 10 (MultiObjectiveFitnessEngine) with non-destructive bloat pruning
and phenotypic correlation niching.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from ravana.compute.dispatcher import ComputeDispatcher
from ravana.domain.formula_ast import (
    ASTNode,
    FeatureDimension,
    GeneticCrossover,
    GeneticMutator,
    TreeGenerator,
)
from ravana.evolution.config import EvolutionConfig, EvolutionResult, Individual
from ravana.evolution.graveyard import IdeaGraveyard
from ravana.features.config import FeatureRegistry
from ravana.fitness import FitnessConfig, MultiObjectiveFitnessEngine, ObjectiveWeights
from ravana.fitness.metrics import compute_ic_and_rank_ic


class GeneticAlphaEngine:
    """Evolutionary algorithm discovering predictive symbolic trading formulas."""

    def __init__(self, config: Optional[EvolutionConfig] = None) -> None:
        self.config = config or EvolutionConfig()
        if not self.config.feature_names:
            self.config.feature_names = FeatureRegistry.get_feature_names()
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)

        self.graveyard = IdeaGraveyard()
        self.champion: Optional[Individual] = None
        self.total_trials_evaluated: int = 0

    def _active_vocab(self, features_df: Optional[pd.DataFrame] = None) -> List[str]:
        """Resolve active feature vocabulary present in the input dataframe."""
        if features_df is not None and len(features_df.columns) > 0:
            valid = [c for c in self.config.feature_names if c in features_df.columns]
            return valid if valid else list(features_df.columns)
        return self.config.feature_names

    def initialize_population(self, features_df: Optional[pd.DataFrame] = None) -> List[Individual]:
        """Generate diverse initial population bounded by maximum tree depth and node cap."""
        vocab = self._active_vocab(features_df)
        pop: List[Individual] = []
        attempts = 0
        max_attempts = self.config.population_size * 15

        while len(pop) < self.config.population_size and attempts < max_attempts:
            attempts += 1
            depth = random.randint(2, max(2, self.config.max_depth))
            method = "grow" if random.random() < 0.5 else "full"
            raw_tree = TreeGenerator.generate(
                max_depth=depth,
                target_dimension=FeatureDimension.DIMENSIONLESS,
                method=method,
                allowed_features=vocab,
            )
            tree = TreeGenerator.prune_to_limits(
                raw_tree,
                max_depth=self.config.max_depth,
                max_nodes=self.config.max_nodes,
                allowed_features=vocab,
            )
            infix = tree.to_infix()

            if not self.graveyard.contains(infix) and not any(ind.formula_str() == infix for ind in pop):
                pop.append(Individual(tree=tree, generation_born=0))

        while len(pop) < self.config.population_size:
            f = random.choice(vocab)
            pop.append(Individual(tree=ASTNode("FEATURE", f), generation_born=0))

        return pop

    def evaluate_population(
        self,
        population: List[Individual],
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        num_trials: Optional[int] = None,
    ) -> List[Individual]:
        """Evaluate formulas in batch, penalizing bloat, multiple trials, and phenotypic clones."""
        if not population:
            return []

        self.total_trials_evaluated += len(population)
        trees = [ind.tree for ind in population]
        signals = ComputeDispatcher.evaluate_batch(trees, features_df)
        if hasattr(signals, "cpu"):
            signals = signals.cpu().numpy()
        if isinstance(signals, np.ndarray) and signals.ndim == 1:
            signals = np.expand_dims(signals, axis=0)

        entry_mask = (
            targets_df["is_entry_bar"].to_numpy(dtype=bool)
            if "is_entry_bar" in targets_df.columns
            else np.ones(len(targets_df), dtype=bool)
        )
        net_ret = targets_df["net_return"].to_numpy(dtype=np.float64)
        y_entry = net_ret[entry_mask]

        effective_trials = num_trials if num_trials is not None else max(
            self.config.population_size, self.total_trials_evaluated
        )

        fit_cfg = FitnessConfig(
            weights=ObjectiveWeights(parsimony_penalty_per_node=self.config.parsimony_penalty),
            num_tested_trials=effective_trials,
        )
        fit_engine = MultiObjectiveFitnessEngine(fit_cfg)
        sig_entry = (
            signals[:, entry_mask]
            if signals.ndim == 2
            else np.expand_dims(signals[entry_mask], 0)
        )
        sig_entry = np.array(sig_entry, copy=True, dtype=np.float64)
        pop_nodes = [ind.tree for ind in population]
        metrics = fit_engine.evaluate_population(sig_entry, y_entry, nodes=pop_nodes, sort_by_fitness=False)

        for idx, ind in enumerate(population):
            m = metrics[idx]
            if m.ic < 0.0 and ind.tree.size() < self.config.max_nodes:
                ind.tree = ASTNode("NEG", children=[ind.tree.clone()]).simplify()
                sig_entry[idx] = -sig_entry[idx]
                m = fit_engine.evaluate_single(sig_entry[idx], y_entry, node=ind.tree)
            ind.ic = m.ic
            ind.rank_ic = m.rank_ic
            ind.fitness = m.composite_fitness
            ind.metrics = m.to_dict()

        self._apply_phenotypic_niching(population, sig_entry)
        population.sort(key=lambda x: x.fitness if x.fitness is not None else -999.0, reverse=True)
        return population

    def _apply_phenotypic_niching(self, population: List[Individual], sig_entry: np.ndarray) -> None:
        """Demote phenotypic clones whose signal correlation exceeds max_clone_correlation."""
        order = np.argsort([-(ind.fitness if ind.fitness is not None else -999.0) for ind in population])
        accepted_sigs: List[np.ndarray] = []

        for idx in order:
            ind = population[int(idx)]
            s = sig_entry[int(idx)]
            std_s = float(np.std(s))
            if std_s < 1e-8 or not np.all(np.isfinite(s)):
                ind.fitness = -10.0
                if ind.metrics:
                    ind.metrics["is_pareto_optimal"] = False
                continue

            s_norm = (s - float(np.mean(s))) / std_s
            is_clone = any(abs(float(np.mean(s_norm * acc))) > self.config.max_clone_correlation for acc in accepted_sigs)

            if is_clone:
                ind.fitness = (ind.fitness or 0.0) - 0.50
                if ind.metrics:
                    ind.metrics["is_pareto_optimal"] = False
                    ind.metrics["composite_fitness"] = round(ind.fitness, 5)
            else:
                accepted_sigs.append(s_norm)

    def _compute_ic(self, sig: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
        """Calculate Pearson Information Coefficient and Spearman Rank IC."""
        return compute_ic_and_rank_ic(sig, target)

    def set_champion(self, tree: ASTNode, fitness: float, ic: float = 0.0) -> None:
        """Establish the reigning production Champion formula."""
        self.champion = Individual(tree=tree, fitness=fitness, ic=ic)

    def consider_challenger(
        self, challenger_tree: ASTNode, fitness: float, ic: float = 0.0
    ) -> bool:
        """Evaluate Challenger against Champion under the strict hurdle rule."""
        infix = challenger_tree.to_infix()
        if self.champion is None:
            self.set_champion(challenger_tree, fitness, ic)
            return True

        hurdle_req = (self.champion.fitness or 0.0) + self.config.champion_hurdle
        if fitness >= hurdle_req:
            self.set_champion(challenger_tree, fitness, ic)
            return True

        self.graveyard.record_rejection(
            formula=infix,
            reason="Failed Challenger Hurdle",
            ic=ic,
            fitness=fitness,
        )
        return False

    def _tournament_select(self, evaluated_pop: List[Individual]) -> Individual:
        """Select winner from random tournament subset."""
        k = min(self.config.tournament_size, len(evaluated_pop))
        contestants = random.sample(evaluated_pop, k)
        return max(contestants, key=lambda ind: ind.fitness or -999.0)

    def step_generation(
        self,
        evaluated_pop: List[Individual],
        gen: int,
        features_df: Optional[pd.DataFrame] = None,
    ) -> List[Individual]:
        """Advance population by one generational step via selection, crossover, and mutation."""
        vocab = self._active_vocab(features_df)
        new_pop: List[Individual] = []

        n_elites = min(self.config.elitism_count, len(evaluated_pop))
        for i in range(n_elites):
            elite = evaluated_pop[i]
            new_pop.append(Individual(tree=elite.tree.clone(), generation_born=gen))

        attempts = 0
        max_attempts = self.config.population_size * 12
        while len(new_pop) < self.config.population_size and attempts < max_attempts:
            attempts += 1
            p1 = self._tournament_select(evaluated_pop)
            p2 = self._tournament_select(evaluated_pop)

            if random.random() < self.config.crossover_prob:
                c1, _ = GeneticCrossover.crossover(p1.tree, p2.tree)
                offspring = c1
            else:
                offspring = p1.tree.clone()

            if random.random() < self.config.mutation_prob:
                offspring = (
                    GeneticMutator.mutate_point(offspring, allowed_features=vocab)
                    if random.random() < 0.5
                    else GeneticMutator.mutate_subtree(offspring, max_depth=2, allowed_features=vocab)
                )

            offspring = TreeGenerator.prune_to_limits(
                offspring,
                max_depth=self.config.max_depth,
                max_nodes=self.config.max_nodes,
                allowed_features=vocab,
            )
            infix = offspring.to_infix()
            if not self.graveyard.contains(infix) and not any(ind.formula_str() == infix for ind in new_pop):
                new_pop.append(Individual(tree=offspring, generation_born=gen))

        while len(new_pop) < self.config.population_size:
            raw = TreeGenerator.generate(2, FeatureDimension.DIMENSIONLESS, "grow", allowed_features=vocab)
            pruned = TreeGenerator.prune_to_limits(raw, self.config.max_depth, self.config.max_nodes, vocab)
            new_pop.append(Individual(tree=pruned, generation_born=gen))

        return new_pop

    def evolve(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        generations: Optional[int] = None,
    ) -> EvolutionResult:
        """Execute full multi-generation symbolic search across feature space."""
        self.config.feature_names = self._active_vocab(features_df)
        n_gens = generations or self.config.max_generations
        trial_budget = max(
            self.config.population_size * n_gens,
            self.total_trials_evaluated + self.config.population_size * n_gens,
        )
        population = self.initialize_population(features_df)
        history: List[Dict[str, Any]] = []

        for g in range(n_gens):
            evaluated = self.evaluate_population(
                population, features_df, targets_df, num_trials=trial_budget
            )
            best = evaluated[0]
            mean_fit = float(np.mean([ind.fitness for ind in evaluated if ind.fitness is not None]))

            history.append({
                "generation": g,
                "best_fitness": best.fitness,
                "best_ic": best.ic,
                "mean_fitness": mean_fit,
                "best_formula": best.formula_str(),
                "best_nodes": best.node_count(),
            })

            if best.fitness is not None:
                self.consider_challenger(best.tree, best.fitness, best.ic or 0.0)

            if g < n_gens - 1:
                population = self.step_generation(evaluated, gen=g + 1, features_df=features_df)

        final_eval = self.evaluate_population(
            population, features_df, targets_df, num_trials=trial_budget
        )
        return EvolutionResult(
            champion=self.champion,
            best_individual=final_eval[0] if final_eval else None,
            final_population=final_eval,
            generation_history=history,
            graveyard_count=self.graveyard.size(),
            total_trials_evaluated=self.total_trials_evaluated,
        )
