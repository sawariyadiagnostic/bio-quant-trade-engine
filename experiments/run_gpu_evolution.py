"""
High-Throughput GPU / Multi-Core Epigenetic Evolution Runner (Phase B).
Optimized for local multi-core CPUs and Google Colab / Kaggle GPUs.
Runs deep multi-generational evolution over historical Indian equities + global lead feeds.
"""
import os
import sys
import time
import json
import numpy as np
import pandas as pd
from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.domain.models import InstrumentType
from src.evolution.genetic_engine import GeneticEvolutionEngine
from experiments.test_phase_b_evolution import prepare_dataset

def main():
    print("=" * 75)
    print("BIO-QUANT PHASE B: HIGH-THROUGHPUT EPIGENETIC EVOLUTION RUNNER")
    print("=" * 75)

    # Check hardware
    try:
        import torch
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Vectorized CPU"
        device = "CUDA GPU" if torch.cuda.is_available() else "CPU"
    except ImportError:
        device_name = "Vectorized Multi-Core CPU"
        device = "CPU"

    print(f"Hardware Compute Platform: {device} ({device_name})")

    df = prepare_dataset()
    print(f"Ingested Market Lake: {len(df)} sessions with 20 continuous & lead features.")

    # Instantiate engine with larger population for deep search
    engine = GeneticEvolutionEngine(
        population_size=60,
        archive_grid=(20, 20),
        p_crossover=0.65,
        p_mutation=0.35,
        p_methylation=0.25,
        max_depth=4,
        instrument=InstrumentType.ETF_CASH
    )

    start_time = time.time()
    generations = 15
    print(f"\nLaunching Evolutionary Search: {generations} Generations, Population 60...")

    history = engine.run_evolution(df, generations=generations, current_regime=0, target_mode="overnight")
    total_time = time.time() - start_time
    total_evals = sum(h["offspring_count"] for h in history)
    eval_rate = total_evals / max(1.0, total_time)

    elites = engine.archive.get_resident_elites()
    top_elites = engine.archive.get_top_elites(10)

    print("\n" + "=" * 75)
    print(f"EVOLUTION SUMMARY:")
    print(f"  Total Wall Clock Time: {total_time:.2f} seconds ({eval_rate:.1f} formulas/sec)")
    print(f"  MAP-Elites Archive Occupancy: {engine.archive.occupancy_rate() * 100.0:.2f}% ({len(elites)}/400 niches)")
    print("=" * 75)

    top_list = []
    print("\nTOP 5 SURVIVING ALPHA GENOMES (NET CALMAR RANKED):")
    for r, e in enumerate(top_elites[:5], 1):
        f_str = e.formula.to_formula_str()
        print(f"\n[Rank {r:02d}] Net Calmar: {e.fitness:,.2f} | Win Rate: {e.win_rate:.1f}% | Net Return: +{e.net_return_pct:,.1f}% | Turnover: {e.turnover:.1f}x")
        print(f"  Formula: {f_str}")
        print(f"  Active Regimes: {sorted(list(e.formula.active_regimes))}")
        top_list.append({
            "rank": r,
            "net_calmar": e.fitness,
            "win_rate": e.win_rate,
            "net_return_pct": e.net_return_pct,
            "turnover": e.turnover,
            "volatility": e.volatility,
            "formula": f_str,
            "active_regimes": sorted(list(e.formula.active_regimes))
        })

    # Save complete audit
    audit = {
        "device": device_name,
        "total_wall_time_sec": round(total_time, 2),
        "evaluation_rate_per_sec": round(eval_rate, 2),
        "generations": generations,
        "population_size": 60,
        "resident_elites_count": len(elites),
        "occupancy_rate_pct": round(engine.archive.occupancy_rate() * 100.0, 2),
        "top_elites": top_list,
        "generation_history": history
    }

    audit_path = "audit/gpu_evolution_audit.json"
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"\n[OK] Complete audit log saved to {audit_path}")

if __name__ == "__main__":
    main()
