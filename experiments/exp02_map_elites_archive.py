"""
EXPERIMENT 2: Quality-Diversity Formula Illumination Run.
Evolves mathematical alpha genomes across a 20x20 MAP-Elites grid (400 niches),
filtered by the Thymic Stress Chamber and statutory Indian tax hurdles.
"""
import os
import sys
import numpy as np
import pandas as pd
from loguru import logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.manifold.data_loader import NSEDataLoader
from src.domain.formula_ast import ASTNode
from src.evolution.thymic_chamber import ThymicStressChamber
from src.evolution.map_elites import MapElitesArchive

def build_candidate_library() -> list[ASTNode]:
    """Generates a diverse initial population of mathematical expression trees."""
    f_close = ASTNode(op="FEATURE", val="Close_Adj")
    f_high = ASTNode(op="FEATURE", val="High")
    f_low = ASTNode(op="FEATURE", val="Low")
    f_vol = ASTNode(op="FEATURE", val="Volume")
    f_spread = ASTNode(op="FEATURE", val="Spread")

    c1 = ASTNode(op="FEATURE", val="1.0")
    c2 = ASTNode(op="FEATURE", val="2.0")

    candidates = [
        # 1. Momentum & Trend Following
        ASTNode(op="DELTA_5", children=[f_close], active_regimes={0, 1}),
        ASTNode(op="SUB", children=[f_close, ASTNode(op="EMA_10", children=[f_close])], active_regimes={0, 1}),
        
        # 2. Mean Reversion & Volatility Pullback
        ASTNode(op="DIV_SAFE", children=[f_spread, ASTNode(op="EMA_10", children=[f_close])], active_regimes={2, 3}),
        ASTNode(op="SUB", children=[f_high, f_close], active_regimes={1, 2}),
        
        # 3. Volume Surge
        ASTNode(op="DIV_SAFE", children=[f_vol, ASTNode(op="EMA_10", children=[f_vol])], active_regimes={0, 3}),
        
        # 4. Syntactic Duplicate (Bloat test)
        ASTNode(op="ADD", children=[ASTNode(op="DELTA_5", children=[f_close]), ASTNode(op="FEATURE", val="0.0")]),
        
        # 5. Composite Alphas
        ASTNode(op="MUL", children=[
            ASTNode(op="DELTA_5", children=[f_close]),
            ASTNode(op="DIV_SAFE", children=[f_vol, ASTNode(op="EMA_10", children=[f_vol])])
        ], active_regimes={0, 1, 2}),
        
        ASTNode(op="SUB", children=[
            ASTNode(op="EMA_10", children=[f_close]),
            ASTNode(op="MUL", children=[f_spread, f_close])
        ], active_regimes={1, 3})
    ]
    return candidates

def run_map_elites_illumination(ticker: str = "SUNPHARMA.NS", sector: str = "Pharma_Healthcare"):
    logger.info(f"=== Starting Experiment 2: MAP-Elites Illumination on {ticker} ===")
    loader = NSEDataLoader()
    df = loader.fetch_ticker(ticker, sector=sector, period="2y")
    if df.empty:
        logger.error("Could not fetch training calibration data.")
        return

    archive = MapElitesArchive(grid_size=(20, 20))
    chamber = ThymicStressChamber()
    candidates = build_candidate_library()

    inserted_count = 0
    quarantined_count = 0

    for idx, formula in enumerate(candidates):
        # 1. Thymic Counterfactual Negative Selection (Red Team 3)
        is_safe = chamber.test_formula_safety(formula, active_regime=0)
        if not is_safe:
            quarantined_count += 1
            continue

        # 2. Evaluate & Insert into MAP-Elites Grid (with Red Team 5 tax hurdle)
        success = archive.evaluate_and_insert(formula, df, current_regime=0)
        if success:
            inserted_count += 1

    occupancy = archive.occupancy_rate()
    elites = archive.get_resident_elites()

    print("\n" + "=" * 65)
    print("   MAP-ELITES ARCHIVE ILLUMINATION SUMMARY")
    print("=" * 65)
    print(f"Total Evaluated Candidates:  {len(candidates)}")
    print(f"Quarantined by Thymic Gate:  {quarantined_count}")
    print(f"Successfully Inserted Elites:{inserted_count}")
    print(f"Grid Behavioral Occupancy:   {occupancy:.2%}")
    print("-" * 65)
    print("Top Resident Behavioral Elites:")
    for i, elite in enumerate(elites[:5]):
        print(f"  [{i+1}] Volatility: {elite.volatility:.1%} | Turnover: {elite.turnover:4.1f}x | Calmar Ratio: {elite.fitness:.2f} | Mean Ret: {elite.mean_return_pct:.2%}")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    run_map_elites_illumination()
