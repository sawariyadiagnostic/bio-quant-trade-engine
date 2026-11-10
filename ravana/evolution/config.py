"""Configuration and State Contracts for Muscle 9: Genetic Alpha Evolution Engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from ravana.domain.formula_ast import ASTNode


@dataclass
class EvolutionConfig:
    """Hyperparameters and constraints for symbolic formula genetic evolution."""

    population_size: int = 64
    max_generations: int = 10
    tournament_size: int = 4
    mutation_prob: float = 0.3
    crossover_prob: float = 0.6
    elitism_count: int = 2
    max_depth: int = 5
    parsimony_penalty: float = 0.001
    champion_hurdle: float = 0.02
    feature_names: List[str] = field(default_factory=list)
    random_seed: Optional[int] = 42


@dataclass
class Individual:
    """Represents a candidate alpha formula in the evolutionary population."""

    tree: ASTNode
    fitness: Optional[float] = None
    ic: Optional[float] = None
    rank_ic: Optional[float] = None
    generation_born: int = 0

    def formula_str(self) -> str:
        """Return standardized infix representation of candidate formula."""
        return self.tree.to_infix()

    def node_count(self) -> int:
        """Calculate total number of AST nodes (tree parsimony size)."""
        count = 1
        for child in self.tree.children:
            count += self._count_nodes(child)
        return count

    def _count_nodes(self, node: ASTNode) -> int:
        """Recursive helper for node counting."""
        c = 1
        for child in node.children:
            c += self._count_nodes(child)
        return c


@dataclass
class EvolutionResult:
    """Immutable outcome of an evolutionary alpha optimization run."""

    champion: Optional[Individual]
    best_individual: Optional[Individual]
    final_population: List[Individual]
    generation_history: List[Dict[str, Any]]
    graveyard_count: int
