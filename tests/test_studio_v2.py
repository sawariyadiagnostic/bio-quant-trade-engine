"""Unit tests for Studio v2.0 Dashboard (Wave 5: Task 5.2)."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_studio_v2_dashboard_creation():
    """Verify create_dashboard_app builds a valid Gradio application."""
    from src.webui.studio_v2 import create_dashboard_app, render_pipeline_summary
    import gradio as gr

    app = create_dashboard_app()
    assert isinstance(app, gr.Blocks)

    mock_report = {
        "success": True,
        "discovered_candidates": 20,
        "pruned_champions": ["(Close_Adj > Open)"],
        "orthogonal_count": 3,
        "compression_ratio": 0.85,
        "archive_coverage": 0.42,
        "allocated_weights": {"NIFTYBEES.NS": 0.5, "GOLDBEES.NS": 0.5},
        "portfolio_equity": 50124.50
    }
    summary_md = render_pipeline_summary(mock_report)
    assert "₹50,124.50" in summary_md or "50124.5" in summary_md
    assert "85.0%" in summary_md or "0.85" in summary_md
    print("[PASS] test_studio_v2_dashboard_creation")


if __name__ == "__main__":
    test_studio_v2_dashboard_creation()
