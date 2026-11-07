"""
src/pipeline/evolution_worker.py
Standalone Quant Compute Worker running in an isolated OS process via multiprocessing.
Guaranteed GIL-free operation: does not import FastAPI, Uvicorn, or Gradio.
Pushes zero-copy telemetry into SharedMemoryBus and commits champions to SQLite WAL.
"""
import multiprocessing
import os
import sys
import time
from typing import Any, Dict, Optional
from loguru import logger
import numpy as np

from src.pipeline.shared_telemetry import SharedMemoryBus
from src.pipeline.process_governor import ProcessGovernor
from src.evolution.archive_migrator import init_archive_database
from src.evolution.archive_store import ArchiveStore


def run_evolution_process(
    shm_name: str,
    ticker: str = "NIFTYBEES",
    lookback: str = "5y",
    generations: int = 100,
    population: int = 100,
    custom_formula: Optional[str] = None,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Child process entrypoint for genetic evolution.
    Executes in a separate OS process with BELOW_NORMAL priority on Cores 2-3.
    """
    my_pid = os.getpid()
    ProcessGovernor.configure_worker(my_pid)
    logger.info(f"[Worker PID {my_pid}] Evolution Worker initialized for {ticker} ({lookback}), {generations} gens (Seed: {seed}).")

    bus = SharedMemoryBus(name=shm_name, create=False)
    archive_store = ArchiveStore(db_path="audit/alpha_elites.db")

    try:
        if seed is not None:
            import random
            import numpy as np
            random.seed(seed)
            np.random.seed(seed)
            logger.info(f"[Worker PID {my_pid}] Seed applied: {seed}")

        bus.update_header(
            current_gen=0,
            total_gens=generations,
            resident_count=0,
            worker_status=1 # RUNNING
        )

        # Import quant dependencies exclusively inside spawned process
        from src.manifold.lake_dataset_loader import LakeDatasetLoader
        from src.evolution.genetic_engine import GeneticEvolutionEngine
        from src.domain.models import InstrumentType
        from src.domain.formula_parser import FormulaParser

        loader = LakeDatasetLoader()
        years_map = {"1y": 1.0, "2y": 2.0, "3y": 3.0, "5y": 5.0, "10y": 10.0, "max": 20.0}
        years_val = years_map.get(lookback.lower(), 5.0)
        df = loader.load_dataset(ticker, lookback_years=years_val)
        if df.empty:
            logger.error(f"[Worker] Dataset slice for {ticker} ({lookback}) is empty!")
            bus.update_header(worker_status=4) # ERROR
            return {"status": "ERROR", "emsg": "Empty dataset slice"}

        instrument = InstrumentType.ETF_CASH if "BEES" in ticker else InstrumentType.FUTURES
        friction_bps = 1.2 if "BEES" in ticker else 25.0
        engine = GeneticEvolutionEngine(
            population_size=int(population),
            archive_grid=(20, 20),
            p_crossover=0.60,
            p_mutation=0.30,
            p_methylation=0.20,
            max_depth=4,
            instrument=instrument
        )

        # Attempt to warm-start from SQLite WAL
        existing_elites = archive_store.query_leaderboard(ticker=ticker, lookback=lookback, limit=400)
        for e in existing_elites:
            if bus.is_abort_requested():
                bus.update_header(worker_status=2)
                return {"status": "ABORTED"}
            try:
                node = FormulaParser.parse(e["formula_str"])
                engine.archive.insert(
                    row=e["niche_row"],
                    col=e["niche_col"],
                    fitness=e["fitness"],
                    formula=node,
                    metrics=e
                )
            except Exception:
                pass
            bus.write_elite(
                row=e["niche_row"],
                col=e["niche_col"],
                fitness=e["fitness"],
                net_return_pct=e.get("net_return_pct", 0.0),
                cagr_pct=e.get("cagr_pct", 0.0),
                max_drawdown=e.get("max_drawdown", 0.0),
                win_rate=e.get("win_rate", 0.0),
                turnover=e.get("turnover", 0.0),
                trades_count=e.get("trades_count", 0),
                formula_str=e["formula_str"]
            )

        # Broadcast initial warm-start telemetry to shared memory bus immediately
        warm_residents = engine.archive.get_resident_elites()
        top_elite = max(warm_residents, key=lambda el: el.fitness, default=None)
        bus.update_header(
            current_gen=0,
            total_gens=generations,
            resident_count=len(warm_residents),
            occupancy_pct=float(len(warm_residents)) / float(engine.archive.rows * engine.archive.cols) * 100.0,
            top_sharpe=top_elite.fitness if top_elite else 0.0,
            best_net_return=top_elite.net_return_pct if top_elite else 0.0,
            worker_status=1 # RUNNING
        )

        for gen in range(1, generations + 1):
            bus.update_heartbeat()

            # Check atomic abort flag from Emergency Panic Kill-Switch (<10ns)
            if bus.is_abort_requested():
                logger.warning(f"[Worker] Atomic abort signal detected at Generation {gen}! Stopping.")
                residents = engine.archive.get_resident_elites()
                bus.update_header(
                    current_gen=gen,
                    total_gens=generations,
                    resident_count=len(residents),
                    worker_status=2 # ABORTED
                )
                return {"status": "ABORTED", "generation": gen}

            summary = engine.evolve_generation(df, current_regime=0)
            bus.update_heartbeat()

            # Push elites to shared memory and SQLite WAL
            new_records = []
            for r in range(engine.archive.rows):
                for c in range(engine.archive.cols):
                    elite = engine.archive.grid[r, c]
                    if elite is not None:
                        bus.write_elite(
                            row=r,
                            col=c,
                            fitness=elite.fitness,
                            net_return_pct=elite.net_return_pct,
                            cagr_pct=getattr(elite, "cagr_pct", elite.net_return_pct),
                            max_drawdown=getattr(elite, "max_drawdown", 0.0),
                            win_rate=elite.win_rate,
                            turnover=elite.turnover,
                            volatility=elite.volatility,
                            sortino=getattr(elite, "sortino", 0.0),
                            profit_factor=getattr(elite, "profit_factor", 0.0),
                            trades_count=getattr(elite, "trades_count", 0),
                            formula_str=elite.formula.to_formula_str() if hasattr(elite.formula, "to_formula_str") else str(elite.formula)
                        )
                        new_records.append({
                            "row": r,
                            "col": c,
                            "fitness": elite.fitness,
                            "net_return_pct": elite.net_return_pct,
                            "cagr_pct": getattr(elite, "cagr_pct", elite.net_return_pct),
                            "max_drawdown": getattr(elite, "max_drawdown", 0.0),
                            "win_rate": elite.win_rate,
                            "turnover": elite.turnover,
                            "volatility": elite.volatility,
                            "sortino": getattr(elite, "sortino", 0.0),
                            "profit_factor": getattr(elite, "profit_factor", 0.0),
                            "trades_count": getattr(elite, "trades_count", 0),
                            "formula_str": elite.formula.to_formula_str() if hasattr(elite.formula, "to_formula_str") else str(elite.formula)
                        })

            if new_records:
                archive_store.batch_upsert_elites(ticker=ticker, lookback=lookback, elites=new_records)

            all_residents = engine.archive.get_resident_elites()
            top_elite = max(all_residents, key=lambda e: e.fitness, default=None)
            bus.update_header(
                current_gen=gen,
                total_gens=generations,
                resident_count=len(all_residents),
                survivor_count=summary.get("thymic_survivors", 0),
                occupancy_pct=summary.get("archive_occupancy_pct", 0.0),
                top_sharpe=top_elite.fitness if top_elite else 0.0,
                best_net_return=top_elite.net_return_pct if top_elite else 0.0,
                worker_status=1 # RUNNING
            )

        all_residents = engine.archive.get_resident_elites()
        final_count = len(all_residents)
        top_elite = max(all_residents, key=lambda e: e.fitness, default=None)
        occ = float(final_count) / float(engine.archive.rows * engine.archive.cols) * 100.0
        bus.update_header(
            current_gen=generations,
            total_gens=generations,
            resident_count=final_count,
            occupancy_pct=occ,
            top_sharpe=top_elite.fitness if top_elite else 0.0,
            best_net_return=top_elite.net_return_pct if top_elite else 0.0,
            worker_status=3 # COMPLETED
        )
        logger.success(f"[Worker] Evolution complete: {final_count} resident champions (Occupancy: {occ:.1f}%).")
        return {"status": "SUCCESS", "resident_elites": final_count}

    except Exception as err:
        logger.error(f"[Worker PID {my_pid}] Unhandled exception: {err}")
        bus.update_header(worker_status=4) # ERROR
        return {"status": "ERROR", "emsg": str(err)}
    finally:
        bus.close()


def spawn_evolution_worker(
    instrument: str = "NIFTYBEES",
    lookback_years: float = 1.0,
    generations: int = 100,
    population_size: int = 100,
    archive_db_path: str = "audit/alpha_elites.db",
    shm_name: str = "",
    seed: Optional[int] = None
) -> multiprocessing.Process:
    """Spawns an isolated evolution worker process via multiprocessing 'spawn' context."""
    lookback_str = f"{int(lookback_years)}y" if lookback_years >= 1.0 else "1y"
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(
        target=run_evolution_process,
        kwargs={
            "shm_name": shm_name,
            "ticker": instrument,
            "lookback": lookback_str,
            "generations": generations,
            "population": population_size,
            "seed": seed,
        },
        daemon=True
    )
    proc.start()
    return proc


if __name__ == "__main__":
    multiprocessing.freeze_support()
