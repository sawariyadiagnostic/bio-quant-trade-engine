"""
Unit & Integration Test Suite for Phase B Evolution Studio & Seeding Engine.
Tests:
1. FormulaParser string-to-AST compilation & serialization.
2. Classical AlphaSeedLibrary loading and AST validation.
3. Universal LakeDatasetLoader asset ingestion & feature calculation.
4. GeneticEvolutionEngine warm-starting with custom user formulas.
"""
import os
import sys
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.domain.models import InstrumentType
from src.domain.formula_ast import ASTNode, ASTEvaluator
from src.domain.formula_parser import FormulaParser, AlphaSeedLibrary
from src.manifold.lake_dataset_loader import LakeDatasetLoader
from src.evolution.genetic_engine import GeneticEvolutionEngine

def test_formula_parser_binary():
    f_str = "EMA_10(Dist_VWAP) + SP500_Ret"
    node = FormulaParser.parse(f_str)
    assert node.op == "ADD"
    assert len(node.children) == 2
    assert node.children[0].op == "EMA_10"
    assert node.children[1].op == "FEATURE"
    assert node.children[1].val == "SP500_Ret"

def test_formula_parser_nested():
    f_str = "(TS_RANK_20(Close_Adj) - RSI_14) * DELTA_5(Volume)"
    node = FormulaParser.parse(f_str)
    assert node.op == "MUL"
    assert node.children[0].op == "SUB"
    assert node.children[1].op == "DELTA_5"

def test_alpha_seed_library():
    seeds = AlphaSeedLibrary.load_seed_nodes(custom_formulas=["SIGN(ADR_Ret) * True_Range"])
    assert len(seeds) >= 15
    for s in seeds:
        assert isinstance(s, ASTNode)
        assert len(s.to_formula_str()) > 0

def test_lake_dataset_loader():
    loader = LakeDatasetLoader()
    tickers = loader.list_available_tickers()
    assert len(tickers) >= 50
    assert "NIFTYBEES" in tickers or "INFY" in tickers

    df = loader.load_dataset("NIFTYBEES" if "NIFTYBEES" in tickers else tickers[0])
    assert len(df) > 50
    expected_cols = ["Close_Adj", "Open", "High", "Low", "Volume", "Dist_VWAP", "RSI_14", "SP500_Ret"]
    for col in expected_cols:
        assert col in df.columns, f"Missing column {col} in loaded dataset"

def test_engine_warm_start():
    custom_formula = "RELU(EMA_Slope) * TS_RANK_20(Close_Adj)"
    custom_node = FormulaParser.parse(custom_formula)

    engine = GeneticEvolutionEngine(
        population_size=20,
        archive_grid=(10, 10),
        instrument=InstrumentType.ETF_CASH
    )
    engine.initialize_population(seed_nodes=[custom_node])
    assert len(engine.population) == 20
    # Check that custom node was injected
    found = any(ind.to_formula_str() == custom_node.to_formula_str() for ind in engine.population)
    assert found, "Custom seed formula was not found in initialized population!"

def test_large_population_streaming_chunking():
    # Test that pop_size=500 runs without memory explosion via chunking
    loader = LakeDatasetLoader()
    df = loader.load_dataset("NIFTYBEES")
    engine = GeneticEvolutionEngine(
        population_size=500,
        archive_grid=(10, 10),
        instrument=InstrumentType.ETF_CASH
    )
    engine.initialize_population()
    stats = engine.evolve_generation(df, chunk_size=100)
    assert stats["offspring_count"] == 500
    assert stats["generation"] == 1
    assert len(engine.population) <= 100 # Active breeding pool kept compact

if __name__ == "__main__":
    test_formula_parser_binary()
    test_formula_parser_nested()
    test_alpha_seed_library()
    test_lake_dataset_loader()
    test_engine_warm_start()
    test_large_population_streaming_chunking()
    print("ALL TESTS PASSED SUCCESSFULLY!")
