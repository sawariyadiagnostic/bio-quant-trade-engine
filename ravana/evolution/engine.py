"""Genetic Alpha Evolution Engine for Muscle 9.

Implements population lifecycle, tournament selection, dimensional crossover,
point mutation, Challenger-Champion v1.1 protocol, and Idea Graveyard recording.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from ravana.compute.dispatcher import ComputeDispatcher
from ravana.domain.formula_ast import (
    ASTNode,
    OPERATORS_BINARY_ARITHMETIC,
    OPERATORS_BINARY_ROLLING,
    OPERATORS_UNARY,
    OPERATORS_ROLLING_UNARY,
    GeneticCrossover,
    GeneticMutator,
)
from ravana.evolution.config import EvolutionConfig, EvolutionResult, Individual
from ravana.evolution.graveyard import IdeaGraveyard
from ravana.features.config import FeatureRegistry


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

    def initialize_population(self) -> List[Individual]:
        """Generate diverse initial population bounded by maximum tree depth."""
        pop: List[Individual] = []
        attempts = 0
        max_attempts = self.config.population_size * 10

        while len(pop) < self.config.population_size and attempts < max_attempts:
            attempts += 1
            depth = random.randint(2, max(2, self.config.max_depth))
            method = "grow" if random.random() < 0.5 else "full"
            tree = self._generate_random_tree(depth, method=method)
            infix = tree.to_infix()

            if not self.graveyard.contains(infix) and not any(ind.formula_str() == infix for ind in pop):
                pop.append(Individual(tree=tree, generation_born=0))

        # Fallback if universe is constrained
        while len(pop) < self.config.population_size:
            f = random.choice(self.config.feature_names)
            tree = ASTNode("FEATURE", f)
            pop.append(Individual(tree=tree, generation_born=0))

        return pop

    def _generate_random_tree(self, depth: int, method: str = "grow") -> ASTNode:
        """Construct random AST tree conforming to registered feature vocabulary."""
        if depth <= 1 or (method == "grow" and random.random() < 0.3):
            if random.random() < 0.2:
                c_val = str(round(random.uniform(-2.0, 2.0), 2))
                return ASTNode("CONST", c_val)
            feat = random.choice(self.config.feature_names)
            return ASTNode("FEATURE", feat)

        op_pool = (
            OPERATORS_BINARY_ARITHMETIC + OPERATORS_UNARY +
            OPERATORS_ROLLING_UNARY + OPERATORS_BINARY_ROLLING
        )
        op = random.choice(op_pool)

        if op in OPERATORS_BINARY_ARITHMETIC or op in OPERATORS_BINARY_ROLLING:
            left = self._generate_random_tree(depth - 1, method)
            right = self._generate_random_tree(depth - 1, method)
            return ASTNode(op, children=[left, right]).simplify()

        child = self._generate_random_tree(depth - 1, method)
        return ASTNode(op, children=[child]).simplify()

    def evaluate_population(
        self,
        population: List[Individual],
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
    ) -> List[Individual]:
        """Evaluate formulas in batch, compute Information Coefficient and parsimony."""
        if not population:
            return []

        trees = [ind.tree for ind in population]
        signals = ComputeDispatcher.evaluate_batch(trees, features_df)

        if isinstance(signals, np.ndarray) and signals.ndim == 1:
            signals = np.expand_dims(signals, axis=0)

        entry_mask = (
            targets_df["is_entry_bar"].to_numpy(dtype=bool)
            if "is_entry_bar" in targets_df.columns
            else np.ones(len(targets_df), dtype=bool)
        )
        net_ret = targets_df["net_return"].to_numpy(dtype=np.float64)
        y_entry = net_ret[entry_mask]

        for idx, ind in enumerate(population):
            sig = signals[idx]
            sig_entry = sig[entry_mask]
            ic, rank_ic = self._compute_ic(sig_entry, y_entry)
            penalty = self.config.parsimony_penalty * ind.node_count()
            ind.ic = ic
            ind.rank_ic = rank_ic
            ind.fitness = float(ic - penalty)

        population.sort(key=lambda x: x.fitness if x.fitness is not None else -999.0, reverse=True)
        return population

    def _compute_ic(self, sig: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
        """Calculate Pearson Information Coefficient and Spearman Rank IC."""
        valid = np.isfinite(sig) & np.isfinite(target)
        if np.sum(valid) < 5:
            return 0.0, 0.0

        s = sig[valid]
        y = target[valid]
        std_s = float(np.std(s))
        std_y = float(np.std(y))

        if std_s < 1e-8 or std_y < 1e-8:
            return 0.0, 0.0

        # Pearson correlation
        cov = float(np.cov(s, y)[0, 1])
        ic = float(np.clip(cov / (std_s * std_y), -1.0, 1.0))

        # Spearman rank correlation
        rank_s = np.argsort(np.argsort(s))
        rank_y = np.argsort(np.argsort(y))
        cov_r = float(np.cov(rank_s, rank_y)[0, 1])
        std_rs = float(np.std(rank_s))
        std_ry = float(np.std(rank_y))
        rank_ic = float(np.clip(cov_r / (std_rs * std_ry), -1.0, 1.0)) if std_rs > 0 and std_ry > 0 else 0.0

        return ic, rank_ic

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

    def step_generation(self, evaluated_pop: List[Individual], gen: int) -> List[Individual]:
        """Advance population by one generational step via selection, crossover, and mutation."""
        new_pop: List[Individual] = []

        # 1. Elitism: clone top performers directly
        n_elites = min(self.config.elitism_count, len(evaluated_pop))
        for i in range(n_elites):
            elite = evaluated_pop[i]
            new_pop.append(Individual(tree=elite.tree.clone(), generation_born=gen))

        # 2. Breed next generation
        while len(new_pop) < self.config.population_size:
            p1 = self._tournament_select(evaluated_pop)
            p2 = self._tournament_select(evaluated_pop)

            if random.random() < self.config.crossover_prob:
                c1, _ = GeneticCrossover.crossover(p1.tree, p2.tree)
                offspring = c1
            else:
                offspring = p1.tree.clone()

            if random.random() < self.config.mutation_prob:
                offspring = (
                    GeneticMutator.mutate_point(offspring)
                    if random.random() < 0.5
                    else GeneticMutator.mutate_subtree(offspring, max_depth=3)
                )

            # Enforce max depth
            while offspring.depth() > self.config.max_depth:
                offspring = random.choice(offspring.children) if offspring.children else offspring

            infix = offspring.to_infix()
            if not self.graveyard.contains(infix):
                new_pop.append(Individual(tree=offspring.simplify(), generation_born=gen))

        return new_pop

    def evolve(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        generations: Optional[int] = None,
    ) -> EvolutionResult:
        """Execute full multi-generation symbolic search across feature space."""
        n_gens = generations or self.config.max_generations
        population = self.initialize_population()
        history: List[Dict[str, Any]] = []

        for g in range(n_gens):
            evaluated = self.evaluate_population(population, features_df, targets_df)
            best = evaluated[0]
            mean_fit = float(np.mean([ind.fitness for ind in evaluated if ind.fitness is not None]))

            history.append({
                "generation": g,
                "best_fitness": best.fitness,
                "best_ic": best.ic,
                "mean_fitness": mean_fit,
                "best_formula": best.formula_str(),
            })

            # Check for champion upgrade
            if best.fitness is not None:
                self.consider_challenger(best.tree, best.fitness, best.ic or 0.0)

            if g < n_gens - 1:
                population = self.step_generation(evaluated, gen=g + 1)

        final_eval = self.evaluate_population(population, features_df, targets_df)
        return EvolutionResult(
            champion=self.champion,
            best_individual=final_eval[0] if final_eval else None,
            final_population=final_eval,
            generation_history=history,
            graveyard_count=self.graveyard.size(),
        )
