"""Evolutionary Nursery & NEAT Speciation Engine (Wave 2: Task 2.2)."""
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass
class NurseryRecord:
    """Tracks individual lineage, novelty, and immunity lifecycle."""
    individual_id: str
    novelty_score: float
    birth_gen: int
    is_mutant: bool
    immunity_expiry: int
    species_id: int = 0


class EvolutionaryNursery:
    """
    Shelters nascent radical mutations from premature death by mature incumbents.
    Implements NEAT-style explicit speciation and grace-period immunity shields.
    """
    def __init__(
        self,
        immunity_generations: int = 15,
        novelty_threshold: float = 0.50,
        compatibility_threshold: float = 0.30
    ):
        self.immunity_generations = immunity_generations
        self.novelty_threshold = novelty_threshold
        self.compatibility_threshold = compatibility_threshold

        self.records: Dict[str, NurseryRecord] = {}
        self.species_representatives: Dict[int, np.ndarray] = {}
        self._next_species_id = 0

    def register_individual(
        self,
        individual_id: str,
        novelty_score: float,
        birth_gen: int
    ) -> NurseryRecord:
        """Enrolls an individual in the nursery and assigns immunity if novelty passes threshold."""
        is_mutant = novelty_score >= self.novelty_threshold
        expiry = birth_gen + (self.immunity_generations if is_mutant else 0)

        record = NurseryRecord(
            individual_id=individual_id,
            novelty_score=novelty_score,
            birth_gen=birth_gen,
            is_mutant=is_mutant,
            immunity_expiry=expiry
        )
        self.records[individual_id] = record
        return record

    def is_immune(self, individual_id: str, current_gen: int) -> bool:
        """Returns True if the individual is currently protected by an active immunity shield."""
        record = self.records.get(individual_id)
        if not record or not record.is_mutant:
            return False
        return current_gen < record.immunity_expiry

    def assign_species(self, individual_id: str, signature: np.ndarray) -> int:
        """
        Assigns an individual to a behavioral species clade based on normalized distance.
        If distance exceeds compatibility threshold from all clades, a new species is born.
        """
        sig = signature.astype(np.float64)
        norm = np.linalg.norm(sig)
        if norm > 1e-8:
            sig = sig / norm

        assigned_species: Optional[int] = None
        for sp_id, rep in self.species_representatives.items():
            dist = float(np.mean(np.abs(sig - rep)))
            if dist < self.compatibility_threshold:
                assigned_species = sp_id
                break

        if assigned_species is None:
            assigned_species = self._next_species_id
            self._next_species_id += 1
            self.species_representatives[assigned_species] = sig

        if individual_id in self.records:
            self.records[individual_id].species_id = assigned_species

        return assigned_species

    def species_count(self) -> int:
        """Returns the total number of distinct behavioral species discovered."""
        return len(self.species_representatives)

    def can_overwrite(
        self,
        incumbent_id: str,
        challenger_fitness: float,
        incumbent_fitness: float,
        current_gen: int
    ) -> bool:
        """
        Evaluates whether a challenger is permitted to replace the incumbent.
        Guarantees that shielded nascent mutants cannot be killed by higher-fitness incumbents.
        """
        if self.is_immune(incumbent_id, current_gen):
            return False  # Protected by nursery immunity shield
        return challenger_fitness > incumbent_fitness
