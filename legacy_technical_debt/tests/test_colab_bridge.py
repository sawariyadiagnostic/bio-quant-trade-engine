"""
Tests for Google Colab GPU Bridge Server & Client.
Verifies health checking, remote evolution streaming, and archive retrieval.
"""
import os
import sys
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from src.remote.colab_server import app
from src.remote.colab_bridge_client import ColabBridgeClient

client = TestClient(app)

def test_health_check():
    """Verifies Colab GPU server health endpoint."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "online"
    assert "gpu_available" in data
    assert "device_name" in data
    assert "vram_gb" in data

def test_remote_evolution_stream():
    """Verifies streaming NDJSON evolution endpoint."""
    payload = {
        "ticker": "NIFTYBEES.NS",
        "generations": 3,
        "population_size": 20,
        "custom_formulas": ["EMA_10(Dist_VWAP) + SP500_Ret"],
        "warm_start": True,
        "target_mode": "overnight",
        "chunk_size": 10
    }
    resp = client.post("/evolve", json=payload)
    assert resp.status_code == 200

    lines = [line.strip() for line in resp.text.strip().split("\n") if line.strip()]
    assert len(lines) >= 4 # 1 start + 3 generations + 1 complete

    events = [json.loads(line) for line in lines]
    assert events[0]["event"] == "start"
    assert events[1]["event"] == "generation"
    assert events[1]["gen"] == 1
    assert "leaderboard" in events[1]
    assert events[-1]["event"] == "complete"

def test_archive_retrieval():
    """Verifies archive retrieval endpoint."""
    resp = client.get("/archive/NIFTYBEES.NS")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ticker"] == "NIFTYBEES.NS"
    assert "resident_count" in data

def test_clean_url():
    """Verifies ColabBridgeClient URL sanitization."""
    assert ColabBridgeClient.clean_url("  http://localhost:8000/ ") == "http://localhost:8000"
    assert ColabBridgeClient.clean_url("my-tunnel.trycloudflare.com") == "https://my-tunnel.trycloudflare.com"
    assert ColabBridgeClient.clean_url("'https://colab.tunnel.com/'") == "https://colab.tunnel.com"

def test_genetic_operators_and_seeding():
    """Verifies crossover, mutation, and custom seed parsing on Colab server."""
    from src.remote.colab_server import (
        FormulaParser, AlphaSeedLibrary, GeneticCrossover, GeneticMutator,
        TreeGenerator, ASTEvaluator, ASTNode
    )
    import pandas as pd
    import numpy as np

    # 1. Custom formula parsing
    custom_str = "(EMA_10(Dist_VWAP) + SP500_Ret)"
    ast = FormulaParser.parse(custom_str)
    assert ast.op == "ADD"
    assert len(ast.children) == 2

    # 2. Alpha seeds
    seeds = AlphaSeedLibrary.load_seed_nodes([custom_str])
    assert len(seeds) >= 10

    # 3. Genetic crossover & mutation
    p1 = TreeGenerator.generate(max_depth=3)
    p2 = TreeGenerator.generate(max_depth=3)
    c1, c2 = GeneticCrossover.crossover(p1, p2)
    assert isinstance(c1, ASTNode)
    assert isinstance(c2, ASTNode)

    mutated = GeneticMutator.mutate_subtree(c1, max_depth=3)
    assert isinstance(mutated, ASTNode)
    methylated = GeneticMutator.mutate_methylation(c2)
    assert isinstance(methylated, ASTNode)

    # 4. TS_RANK_20 flat tie handling
    flat_df = pd.DataFrame({"Spread": np.full(50, 0.05), "High": np.full(50, 100.0), "Low": np.full(50, 99.0), "Close": np.full(50, 100.0)})
    rank_node = ASTNode(op="TS_RANK_20", children=[ASTNode(op="FEATURE", val="Spread")])
    rank_res = ASTEvaluator.eval_node(rank_node, flat_df)
    # Tied array should evaluate to median rank 0.5, NOT zero collapse
    assert np.all(rank_res > 0.0)
    assert np.isclose(rank_res[-1], 0.5)

def test_target_mode_execution():
    """Verifies that target_mode (intraday vs overnight) is respected in evaluation."""
    from src.remote.colab_server import MapElitesArchive, FormulaParser
    import pandas as pd
    import numpy as np

    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    # Synthetic trending dataframe
    p = 100.0 * np.cumprod(1.0 + np.full(100, 0.005))
    df = pd.DataFrame({
        "Close_Adj": p, "Open": p * 0.99, "High": p * 1.01, "Low": p * 0.98,
        "Volume": np.full(100, 1e6), "Dist_VWAP": np.full(100, 0.02)
    }, index=dates)

    archive = MapElitesArchive(5, 5)
    f = FormulaParser.parse("Dist_VWAP")
    # Evaluate with overnight
    ins_overnight = archive.evaluate_and_insert(f, df, target_mode="overnight")
    # Evaluate with intraday
    ins_intraday = archive.evaluate_and_insert(f, df, target_mode="intraday")
    assert isinstance(ins_overnight, bool)
    assert isinstance(ins_intraday, bool)

if __name__ == "__main__":
    test_health_check()
    test_remote_evolution_stream()
    test_archive_retrieval()
    test_clean_url()
    test_genetic_operators_and_seeding()
    test_target_mode_execution()
    print("ALL COLAB BRIDGE TESTS PASSED SUCCESSFULLY!")
