"""Unit tests for Evolutionary Nursery & NEAT Speciation (Wave 2: Task 2.2)."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_nursery_immunity_protection():
    """Verify EvolutionaryNursery shields nascent mutants for 15 generations."""
    from src.evolution.nursery import EvolutionaryNursery

    nursery = EvolutionaryNursery(immunity_generations=15, novelty_threshold=0.50)

    # Register mutant at generation 10 with high novelty score (0.85)
    ind = nursery.register_individual(
        individual_id="mutant_alpha_1",
        novelty_score=0.85,
        birth_gen=10
    )
    assert ind.is_mutant is True
    assert ind.immunity_expiry == 25

    # Test protection at generation 15 (age 5 < 15): Must be IMMUNE
    assert nursery.is_immune("mutant_alpha_1", current_gen=15) is True

    # Test protection at generation 24 (age 14 < 15): Still IMMUNE
    assert nursery.is_immune("mutant_alpha_1", current_gen=24) is True

    # Test protection at generation 25 (age 15 >= 15): Shield expired
    assert nursery.is_immune("mutant_alpha_1", current_gen=25) is False
    print("[PASS] test_nursery_immunity_protection")


def test_nursery_speciation_grouping():
    """Verify individuals are grouped into species clades based on behavioral distance."""
    from src.evolution.nursery import EvolutionaryNursery

    nursery = EvolutionaryNursery(compatibility_threshold=0.30)

    sig_a1 = np.array([1, 1, 1, 0, 0], dtype=np.float32)
    sig_a2 = np.array([1, 1, 0, 0, 0], dtype=np.float32)  # Distance from a1 is low (0.2)
    sig_b1 = np.array([0, 0, 0, 1, 1], dtype=np.float32)  # Distance from a1 is high (1.0)

    sp_a1 = nursery.assign_species("ind_1", sig_a1)
    sp_a2 = nursery.assign_species("ind_2", sig_a2)
    sp_b1 = nursery.assign_species("ind_3", sig_b1)

    assert sp_a1 == sp_a2  # Clustered into same species
    assert sp_a1 != sp_b1  # Separated into novel species
    assert nursery.species_count() == 2
    print("[PASS] test_nursery_speciation_grouping")


if __name__ == "__main__":
    test_nursery_immunity_protection()
    test_nursery_speciation_grouping()
