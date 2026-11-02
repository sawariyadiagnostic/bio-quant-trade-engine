"""
Empirical Validation Test Suite for Interactive Visual Spaces & Leaderboard Engine.
Verifies:
1. Plotly MAP-Elites Heatmap generation with empty and populated archive.
2. Plotly Risk-Reward Pareto Optimal Frontier calculation and marker rendering.
3. Spotlight markdown card formatting across Top Earner, Institutional Alpha, and Accuracy.
4. Multi-metric dynamic sorting across Return, CIAS, and Drawdown.
"""
import os
import sys
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ui.visualizations import (
    build_map_elites_heatmap,
    build_pareto_scatter,
    build_spotlight_markdown,
    format_leaderboard_table,
    EMPTY_LEADERBOARD
)
from src.ui.evolution_dashboard import re_sort_leaderboard, load_ticker_archive

def generate_mock_records():
    return [
        {
            "row": 2, "col": 3, "fitness": 18.5, "net_return_pct": 142.5,
            "max_drawdown": 8.2, "win_rate": 64.0, "sortino": 3.8,
            "profit_factor": 2.4, "turnover": 4.5, "volatility": 0.12,
            "trades_count": 85, "formula_str": "EMA_10(Dist_VWAP) + SP500_Ret"
        },
        {
            "row": 0, "col": 1, "fitness": 24.2, "net_return_pct": 52.0,
            "max_drawdown": 1.4, "win_rate": 78.5, "sortino": 5.2,
            "profit_factor": 3.6, "turnover": 1.2, "volatility": 0.06,
            "trades_count": 42, "formula_str": "TS_RANK_20(Close_Adj) - RSI_14"
        },
        {
            "row": 8, "col": 12, "fitness": 9.4, "net_return_pct": 210.0,
            "max_drawdown": 24.5, "win_rate": 56.0, "sortino": 2.1,
            "profit_factor": 1.8, "turnover": 15.0, "volatility": 0.28,
            "trades_count": 160, "formula_str": "Vol_Price_Flow / Downside_Vol_Ratio"
        }
    ]

def test_empty_visuals():
    heatmap = build_map_elites_heatmap([])
    assert heatmap is not None
    assert len(heatmap.data) == 1

    scatter = build_pareto_scatter([])
    assert scatter is not None

    spotlight = build_spotlight_markdown([])
    assert "Awaiting initial generation" in spotlight

    tbl = format_leaderboard_table([])
    assert isinstance(tbl, pd.DataFrame)
    assert len(tbl) == 1

def test_populated_visuals():
    records = generate_mock_records()

    # Heatmap test
    heatmap = build_map_elites_heatmap(records)
    assert heatmap is not None
    assert "MAP-Elites 400-Niche Behavioral Space" in heatmap.layout.title.text

    # Pareto Scatter test
    scatter = build_pareto_scatter(records)
    assert scatter is not None
    assert len(scatter.data) >= 1
    # Check that Pareto Frontier was calculated
    trace_names = [t.name for t in scatter.data]
    assert "🏆 Pareto Frontier" in trace_names

    # Spotlight Markdown test
    spotlight = build_spotlight_markdown(records)
    assert "+210.0%" in spotlight  # Top earner
    assert "24.20" in spotlight    # Top institutional CIAS
    assert "78.5%" in spotlight    # Top accuracy win rate

def test_sorting_leaderboard():
    records = generate_mock_records()

    # Sort by Net Return (Top Earners First)
    df_ret = format_leaderboard_table(records, sort_by="net_return_pct")
    assert df_ret.iloc[0]["Net Return %"] == "+210.0%"
    assert df_ret.iloc[-1]["Net Return %"] == "+52.0%"

    # Sort by Institutional CIAS Score
    df_fit = format_leaderboard_table(records, sort_by="fitness")
    assert df_fit.iloc[0]["CIAS Score"] == 24.2
    assert df_fit.iloc[0]["Net Return %"] == "+52.0%"  # Explains why less return has higher CIAS

    # Sort by Lowest Drawdown (Safest First)
    df_dd = format_leaderboard_table(records, sort_by="max_drawdown")
    assert df_dd.iloc[0]["Max Drawdown %"] == "1.4%"

if __name__ == "__main__":
    test_empty_visuals()
    test_populated_visuals()
    test_sorting_leaderboard()
    print("ALL VISUAL SPACE & SORTING TESTS PASSED EMPIRICALLY!")
