"""
Multi-Generational Epigenetic Evolution Engine (Phase B).
Coordinates:
- Population initialization and random AST generation.
- Epigenetic subtree crossover and methylation mutations.
- Adversarial Thymic Stress Chamber quarantine.
- 20x20 MAP-Elites behavioral archive illumination.
- Hardware-safe thermal pacing (50ms pauses, gc.collect()).
"""
import os
import time
import gc
import random
from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger

from ..domain.formula_ast import (
    ASTNode,
    TreeGenerator,
    GeneticMutator,
    GeneticCrossover,
    STANDARD_FEATURES
)
from ..domain.models import InstrumentType
from .map_elites import MapElitesArchive, EliteIndividual
from .thymic_chamber import ThymicStressChamber

class GeneticEvolutionEngine:
    def __init__(
        self,
        population_size: int = 50,
        archive_grid: tuple[int, int] = (20, 20),
        p_crossover: float = 0.60,
        p_mutation: float = 0.30,
        p_methylation: float = 0.20,
        max_depth: int = 4,
        instrument: InstrumentType = InstrumentType.ETF_CASH,
        archive_path: Optional[str] = None
    ):
        self.pop_size = population_size
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.p_methylation = p_methylation
        self.max_depth = max_depth
        self.instrument = instrument
        self.archive_path = archive_path

        self.archive = MapElitesArchive(grid_size=archive_grid)
        self.thymic_chamber = ThymicStressChamber()
        self.population: list[ASTNode] = []
        self.generation: int = 0

        # Auto-load existing persistent archive if available
        if self.archive_path and os.path.exists(self.archive_path):
            self.archive.load_from_disk(self.archive_path)

    def initialize_population(
        self, 
        feature_names: list[str] = STANDARD_FEATURES,
        seed_nodes: list[ASTNode] = None
    ):
        """
        Initializes population with:
        1. Resident elites from the persistent archive (continual learning).
        2. Injected custom seeds and classical literature alphas.
        3. Remaining slots filled with diverse random ASTs.
        """
        self.population.clear()
        
        # 1. First inject resident elites from archive
        resident_elites = self.archive.get_resident_elites()
        for e in resident_elites:
            if len(self.population) < self.pop_size:
                self.population.append(e.formula.clone())

        # 2. Inject explicit custom seed nodes
        if seed_nodes:
            for node in seed_nodes:
                if len(self.population) < self.pop_size:
                    self.population.append(node.clone())

        # 3. Fill remaining space with random trees
        remaining = self.pop_size - len(self.population)
        for i in range(remaining):
            depth = random.randint(2, self.max_depth)
            method = "grow" if i % 2 == 0 else "full"
            tree = TreeGenerator.generate(max_depth=depth, features=feature_names, method=method)
            self.population.append(tree)

    def evolve_generation(
        self,
        df: pd.DataFrame,
        current_regime: int = 0,
        target_mode: str = "overnight",
        chunk_size: int = 250
    ) -> dict:
        """
        Runs a streaming micro-batch evolutionary generation.
        Supports arbitrarily large population sizes (from 10 to 1,000,000+) with constant O(1) RAM.
        """
        self.generation += 1
        elites = self.archive.get_resident_elites()

        # Build mating pool (active population + top resident archive champions)
        pool = list(self.population)
        if elites:
            pool.extend([e.formula for e in elites[:15]])

        if not pool:
            self.initialize_population()
            pool = list(self.population)

        passed_thymic_total = 0
        inserted_elites_total = 0
        total_produced = 0

        # Determine number of streaming chunks to evaluate
        effective_chunk = min(chunk_size, self.pop_size)
        num_chunks = max(1, int(np.ceil(self.pop_size / effective_chunk)))

        next_population_candidates: list[ASTNode] = []

        for chunk_idx in range(num_chunks):
            chunk_offspring: list[ASTNode] = []
            chunk_target = min(effective_chunk, self.pop_size - total_produced)
            if chunk_target <= 0:
                break

            while len(chunk_offspring) < chunk_target:
                p1 = random.choice(pool)
                p2 = random.choice(pool)

                if random.random() < self.p_crossover:
                    c1, c2 = GeneticCrossover.crossover(p1, p2)
                else:
                    c1, c2 = p1.clone(), p2.clone()

                if random.random() < self.p_mutation:
                    c1 = GeneticMutator.mutate_subtree(c1, max_depth=self.max_depth)
                if random.random() < self.p_mutation:
                    c2 = GeneticMutator.mutate_subtree(c2, max_depth=self.max_depth)

                if random.random() < self.p_methylation:
                    c1 = GeneticMutator.mutate_methylation(c1)
                if random.random() < self.p_methylation:
                    c2 = GeneticMutator.mutate_methylation(c2)

                chunk_offspring.extend([c1, c2])

            chunk_offspring = chunk_offspring[:chunk_target]
            total_produced += len(chunk_offspring)

            # Thymic Quarantine & MAP-Elites Insertion for this chunk
            for ind in chunk_offspring:
                if not self.thymic_chamber.test_formula_safety(ind, active_regime=current_regime):
                    continue
                passed_thymic_total += 1

                if self.archive.evaluate_and_insert(
                    ind, df, current_regime=current_regime,
                    instrument=self.instrument, target_mode=target_mode
                ):
                    inserted_elites_total += 1
                    # Highly fit candidate retained for breeding pool
                    if len(next_population_candidates) < 100:
                        next_population_candidates.append(ind.clone())

            # Seed breeding pool with top chunk survivors if needed
            if len(next_population_candidates) < 50:
                for ind in chunk_offspring[:10]:
                    if len(next_population_candidates) < 50:
                        next_population_candidates.append(ind.clone())

            del chunk_offspring
            if chunk_idx % 4 == 0:
                gc.collect()

        # Refresh compact active population for next generation's breeding pool
        if next_population_candidates:
            self.population = next_population_candidates[:min(self.pop_size, 100)]
        elif not self.population:
            self.initialize_population()

        time.sleep(0.020)
        gc.collect()

        return {
            "generation": self.generation,
            "offspring_count": total_produced,
            "thymic_survivors": passed_thymic_total,
            "archive_inserted": inserted_elites_total,
            "total_elites": len(self.archive.get_resident_elites()),
            "archive_occupancy_pct": round(self.archive.occupancy_rate() * 100.0, 2)
        }

    def run_evolution(
        self,
        df: pd.DataFrame,
        generations: int = 10,
        current_regime: int = 0,
        target_mode: str = "overnight",
        seed_nodes: list[ASTNode] = None,
        callback = None
    ) -> list[dict]:
        """Executes multi-generation evolution loop with live callback support."""
        if not self.population:
            self.initialize_population(seed_nodes=seed_nodes)

        history = []
        for g in range(generations):
            stats = self.evolve_generation(df, current_regime=current_regime, target_mode=target_mode)
            stats["top_elites"] = self.archive.get_top_elites(5)
            history.append(stats)
            if callback:
                try:
                    callback(stats)
                except Exception as e:
                    logger.warning(f"Evolution callback exception: {e}")
            logger.info(
                f"[GEN {stats['generation']:02d}] Thymic Survivors: {stats['thymic_survivors']}/{stats['offspring_count']} "
                f"| New Elites: {stats['archive_inserted']} | Total Elites: {stats['total_elites']} "
                f"| Occupancy: {stats['archive_occupancy_pct']}%"
            )

        if self.archive_path:
            self.archive.save_to_disk(self.archive_path)

        return history
