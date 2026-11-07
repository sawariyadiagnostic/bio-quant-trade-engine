"""Unit tests for Alpha Engine Pipeline Orchestrator (Wave 5: Task 5.1)."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_alpha_engine_pipeline_cycle():
    """Verify AlphaEnginePipeline executes end-to-end cycle: Ingestion -> Evolution -> Manifold Pruning -> Paper Execution."""
    from src.pipeline.alpha_engine import AlphaEnginePipeline

    pipeline = AlphaEnginePipeline(dry_run=True)
    report = pipeline.run_cycle(
        tickers=["NIFTYBEES.NS", "GOLDBEES.NS"],
        generations=3,
        pop_size=20,
        n_bars=80
    )

    assert "success" in report
    assert report["success"] is True
    assert "discovered_candidates" in report
    assert "pruned_champions" in report
    assert "allocated_weights" in report
    assert "portfolio_equity" in report

    assert report["discovered_candidates"] > 0
    assert len(report["pruned_champions"]) > 0
    assert len(report["allocated_weights"]) > 0
    assert report["portfolio_equity"] > 0.0
    print("[PASS] test_alpha_engine_pipeline_cycle")


if __name__ == "__main__":
    test_alpha_engine_pipeline_cycle()
