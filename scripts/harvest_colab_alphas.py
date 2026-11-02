"""
Autonomous Alpha Harvester for Bio-Quant Cloud GPU Worker.
Mines mathematical alpha formula champions across liquid NSE equities and ETFs
using the remote Google Colab Tesla T4 GPU.
"""
import os
import json
import time
from typing import Dict, Any, List
from loguru import logger
from src.remote.colab_bridge_client import ColabBridgeClient

COLAB_URL = "https://proxy-defence-cliff-strikes.trycloudflare.com"

MINING_ROSTER = [
    {
        "ticker": "NIFTYBEES.NS",
        "clean_name": "NIFTYBEES",
        "lookback": "5y",
        "generations": 25,
        "population_size": 100,
        "description": "Benchmark Nifty 50 ETF (5Y Horizon - Low Friction 1.2 bps)"
    },
    {
        "ticker": "NIFTYBEES.NS",
        "clean_name": "NIFTYBEES_10Y",
        "lookback": "10y",
        "generations": 20,
        "population_size": 100,
        "description": "Benchmark Nifty 50 ETF (10Y Full Cycle Macro Horizon)"
    },
    {
        "ticker": "BANKBEES.NS",
        "clean_name": "BANKBEES",
        "lookback": "5y",
        "generations": 25,
        "population_size": 100,
        "description": "High-Beta Banking Sector ETF (5Y Horizon)"
    },
    {
        "ticker": "HAL.NS",
        "clean_name": "HAL",
        "lookback": "5y",
        "generations": 25,
        "population_size": 100,
        "description": "Hindustan Aeronautics (Sovereign Capex Momentum Leader)"
    },
    {
        "ticker": "BEL.NS",
        "clean_name": "BEL",
        "lookback": "5y",
        "generations": 20,
        "population_size": 100,
        "description": "Bharat Electronics (Defense Systems Leader)"
    },
    {
        "ticker": "INFY.NS",
        "clean_name": "INFY",
        "lookback": "5y",
        "generations": 20,
        "population_size": 100,
        "description": "Infosys (IT Clade High-Beta Overnight Leader)"
    },
    {
        "ticker": "RELIANCE.NS",
        "clean_name": "RELIANCE",
        "lookback": "5y",
        "generations": 20,
        "population_size": 100,
        "description": "Reliance Industries (Heavyweight Mega-Cap)"
    },
    {
        "ticker": "^NSEI",
        "clean_name": "NIFTY_50",
        "lookback": "5y",
        "generations": 20,
        "population_size": 100,
        "description": "Nifty 50 Pure Continuous Spot Index"
    }
]

def run_harvest():
    os.makedirs("audit", exist_ok=True)
    logger.info(f"=== Starting Autonomous Alpha Harvester on Colab GPU: {COLAB_URL} ===")

    # 1. Health check
    health = ColabBridgeClient.check_health(COLAB_URL)
    if not health.get("online"):
        logger.error(f"Cannot reach Colab GPU worker: {health.get('error')}")
        return

    logger.info(f"Connected to Cloud GPU: {health.get('device_name')} ({health.get('vram_gb')} GB VRAM) - Latency: {health.get('latency_ms')} ms")

    total_formulas_evaluated = 0
    total_elites_harvested = 0
    harvest_catalog = {}

    start_time = time.time()

    for job_idx, job in enumerate(MINING_ROSTER, 1):
        ticker = job["ticker"]
        clean_name = job["clean_name"]
        lookback = job["lookback"]
        gens = job["generations"]
        pop = job["population_size"]
        desc = job["description"]

        logger.info(f"\n[{job_idx}/{len(MINING_ROSTER)}] Initiating mining on {ticker} ({desc}) | {pop} formulas x {gens} gens...")

        job_params = {
            "ticker": ticker,
            "lookback": lookback,
            "generations": gens,
            "population_size": pop,
            "custom_formulas": [],
            "warm_start": True,
            "target_mode": "overnight",
            "chunk_size": 50
        }

        job_start = time.time()
        latest_elites_count = 0
        latest_leaderboard = []

        try:
            for event in ColabBridgeClient.stream_evolution(COLAB_URL, job_params):
                ev = event.get("event")
                if ev == "generation":
                    gen = event.get("gen")
                    tot = event.get("total_gens")
                    ins = event.get("inserted", 0)
                    total_elites = event.get("total_elites", 0)
                    latest_elites_count = total_elites
                    occ = event.get("occupancy_pct", 0.0)
                    latest_leaderboard = event.get("leaderboard", [])

                    top_row = latest_leaderboard[0] if latest_leaderboard else {}
                    top_ret = top_row.get("Net Return %", "N/A")
                    top_calmar = top_row.get("Net Calmar", "N/A")

                    logger.info(f"  [Gen {gen:02d}/{tot:02d}] Discovered: +{ins} | Resident Elites: {total_elites} ({occ}%) | Top Net Return: {top_ret} | Calmar: {top_calmar}")
                elif ev == "complete":
                    latest_elites_count = event.get("total_elites", latest_elites_count)
                    logger.info(f"  --> Job Completed! Total Resident Champions: {latest_elites_count}")

            formulas_in_job = pop + (pop * gens)
            total_formulas_evaluated += formulas_in_job

            # Fetch full archive from remote Colab
            time.sleep(1.0)
            archive_data = ColabBridgeClient.fetch_archive(COLAB_URL, ticker)
            elites = archive_data.get("elites", [])

            if elites:
                local_file = os.path.join("audit", f"archive_{clean_name}.json")
                with open(local_file, "w", encoding="utf-8") as f:
                    json.dump(archive_data, f, indent=2)
                logger.info(f"  Saved full archive to {local_file} ({len(elites)} elites)")

                # Extract standout champions
                top_earner = max(elites, key=lambda x: x.get("net_return_pct", 0.0))
                top_calmar = max(elites, key=lambda x: x.get("fitness", 0.0))
                top_winrate = max(elites, key=lambda x: x.get("win_rate", 0.0))
                low_dd = min(elites, key=lambda x: x.get("max_drawdown", 999.0))

                total_elites_harvested += len(elites)

                harvest_catalog[clean_name] = {
                    "ticker": ticker,
                    "description": desc,
                    "lookback": lookback,
                    "resident_count": len(elites),
                    "archive_file": local_file,
                    "duration_sec": round(time.time() - job_start, 1),
                    "top_earner": {
                        "net_return_pct": top_earner.get("net_return_pct"),
                        "max_drawdown": top_earner.get("max_drawdown"),
                        "win_rate": top_earner.get("win_rate"),
                        "calmar_cias": top_earner.get("fitness"),
                        "formula": top_earner.get("formula_str")
                    },
                    "top_institutional_calmar": {
                        "net_return_pct": top_calmar.get("net_return_pct"),
                        "max_drawdown": top_calmar.get("max_drawdown"),
                        "win_rate": top_calmar.get("win_rate"),
                        "calmar_cias": top_calmar.get("fitness"),
                        "formula": top_calmar.get("formula_str")
                    },
                    "top_win_rate": {
                        "net_return_pct": top_winrate.get("net_return_pct"),
                        "max_drawdown": top_winrate.get("max_drawdown"),
                        "win_rate": top_winrate.get("win_rate"),
                        "calmar_cias": top_winrate.get("fitness"),
                        "formula": top_winrate.get("formula_str")
                    },
                    "lowest_drawdown": {
                        "net_return_pct": low_dd.get("net_return_pct"),
                        "max_drawdown": low_dd.get("max_drawdown"),
                        "win_rate": low_dd.get("win_rate"),
                        "calmar_cias": low_dd.get("fitness"),
                        "formula": low_dd.get("formula_str")
                    }
                }
            else:
                logger.warning(f"  No elites retrieved for {clean_name}")

        except Exception as e:
            logger.error(f"Error during harvest of {ticker}: {e}")

        time.sleep(2.0)

    total_time = round(time.time() - start_time, 1)

    # Save master catalog
    catalog_path = os.path.join("audit", "HARVESTED_ALPHAS_CATALOG.json")
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "colab_url": COLAB_URL,
            "device": health.get("device_name"),
            "total_time_seconds": total_time,
            "total_formulas_evaluated": total_formulas_evaluated,
            "total_elites_harvested": total_elites_harvested,
            "assets": harvest_catalog
        }, f, indent=2)

    logger.info(f"\n=== Alpha Mining Completed Successfully! ===")
    logger.info(f"Total Time: {total_time}s | Formulas Evaluated: {total_formulas_evaluated:,} | Elites Harvested: {total_elites_harvested}")
    logger.info(f"Master Catalog saved to: {catalog_path}")

if __name__ == "__main__":
    run_harvest()
