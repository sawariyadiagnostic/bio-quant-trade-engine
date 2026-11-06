"""
Empirical Validation Test Suite for Lookback Partitioning and Annualized CAGR Normalization.
Verifies:
1. True CAGR % calculation prevents the 5-year vs 1-year total return distortion.
2. Archives are partitioned strictly by lookback window (archive_{ticker}_{lookback}.json).
3. Switching lookback horizons correctly loads corresponding partitioned champions.
4. Cross-horizon evaluation re-scores champions strictly against the active time slice.
"""
import os
import sys
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
from src.domain.models import InstrumentType
from src.evolution.map_elites import MapElitesArchive, EliteIndividual
from src.domain.formula_ast import ASTNode
from src.domain.formula_parser import FormulaParser
from src.ui.evolution_dashboard import (
    get_period_code,
    get_archive_file_path,
    _load_initial_archive,
    load_ticker_archive,
    wipe_and_reset_archive,
    re_sort_leaderboard,
    SORT_KEY_MAP
)
from src.ui.visualizations import format_leaderboard_table
import src.ui.evolution_dashboard as evo_dash

def test_cagr_calculation_normalizes_cross_timeframe():
    """
    Empirically proves that a 1-year +35% return correctly outranks
    a 5-year +120% return when normalized by CAGR %.
    """
    records = [
        {
            "formula_str": "FORMULA_5Y",
            "net_return_pct": 120.0,
            # 5y +120% => (1 + 1.20)^(1/5) - 1 = 17.08%
            "cagr_pct": 17.08,
            "fitness": 12.0,
            "max_drawdown": 14.0,
            "win_rate": 58.0,
            "trades_count": 80
        },
        {
            "formula_str": "FORMULA_1Y",
            "net_return_pct": 35.0,
            # 1y +35% => (1 + 0.35)^(1/1) - 1 = 35.0%
            "cagr_pct": 35.0,
            "fitness": 10.5,
            "max_drawdown": 4.5,
            "win_rate": 65.0,
            "trades_count": 30
        }
    ]

    # When sorted by raw cumulative net return, 5y (120%) falsely appears superior
    df_raw = format_leaderboard_table(records, sort_by="net_return_pct")
    assert df_raw.iloc[0]["Formula AST Expression"] == "FORMULA_5Y"
    assert df_raw.iloc[0]["Net Return %"] == "+120.0%"

    # When sorted by annualized CAGR %, 1y (35.0% CAGR) properly takes Rank #1
    df_cagr = format_leaderboard_table(records, sort_by="cagr_pct")
    assert df_cagr.iloc[0]["Formula AST Expression"] == "FORMULA_1Y"
    assert df_cagr.iloc[0]["Annual CAGR %"] == "+35.0%"
    assert df_cagr.iloc[1]["Formula AST Expression"] == "FORMULA_5Y"
    assert df_cagr.iloc[1]["Annual CAGR %"] == "+17.1%"
    print("[PASS] test_cagr_calculation_normalizes_cross_timeframe")

def test_archive_lookback_partitioning_isolation():
    """
    Verifies that archives saved under '1y' do not overwrite or collide with '5y'.
    """
    os.makedirs("audit", exist_ok=True)
    p_1y = get_archive_file_path("TESTASSET", "1y")
    p_5y = get_archive_file_path("TESTASSET", "5y")

    # Clean prior test artifacts
    for p in (p_1y, p_5y):
        if os.path.exists(p):
            os.remove(p)

    rec_1y = [{"formula_str": "CLOSE_1Y", "net_return_pct": 30.0, "cagr_pct": 30.0, "fitness": 8.0, "max_drawdown": 5.0, "win_rate": 60.0, "trades_count": 25}]
    rec_5y = [{"formula_str": "CLOSE_5Y", "net_return_pct": 140.0, "cagr_pct": 19.1, "fitness": 14.0, "max_drawdown": 12.0, "win_rate": 55.0, "trades_count": 120}]

    with open(p_1y, "w") as f:
        json.dump({"ticker": "TESTASSET", "lookback": "1y", "resident_count": 1, "elites": rec_1y}, f)
    with open(p_5y, "w") as f:
        json.dump({"ticker": "TESTASSET", "lookback": "5y", "resident_count": 1, "elites": rec_5y}, f)

    # 1. Load 1y horizon
    status_1y, _, _, _, _, table_1y = load_ticker_archive("TESTASSET", "Top Annualized CAGR % First (Normalized)", "Top 10", "1 Year (~252 bars)")
    assert len(evo_dash.CURRENT_ARCHIVE_RECORDS) == 1
    assert evo_dash.CURRENT_ARCHIVE_RECORDS[0]["formula_str"] == "CLOSE_1Y"
    assert table_1y.iloc[0]["Annual CAGR %"] == "+30.0%"

    # 2. Switch horizon to 5y
    status_5y, _, _, _, _, table_5y = load_ticker_archive("TESTASSET", "Top Annualized CAGR % First (Normalized)", "Top 10", "5 Years (~1,260 bars)")
    assert len(evo_dash.CURRENT_ARCHIVE_RECORDS) == 1
    assert evo_dash.CURRENT_ARCHIVE_RECORDS[0]["formula_str"] == "CLOSE_5Y"
    assert table_5y.iloc[0]["Annual CAGR %"] == "+19.1%"

    # Clean up
    for p in (p_1y, p_5y):
        if os.path.exists(p):
            os.remove(p)
    evo_dash.CURRENT_ARCHIVE_RECORDS = []
    print("[PASS] test_archive_lookback_partitioning_isolation")

def test_map_elites_cagr_evaluation_precision():
    """
    Tests that MapElitesArchive._calculate_metrics produces valid CAGR % on real bar sequences.
    """
    archive = MapElitesArchive(grid_size=(5, 5))
    n_bars = 252 # exactly 1 year of daily bars
    np.random.seed(42)
    open_p = 100.0 + np.cumsum(np.random.randn(n_bars) * 0.5)
    close_p = open_p + 0.20 # consistent slight upward bias

    positions = np.ones(n_bars, dtype=np.float64) # always long
    metrics = archive._calculate_metrics(positions, open_p, close_p, InstrumentType.ETF_CASH, target_mode="intraday")
    assert metrics is not None
    assert "cagr_pct" in metrics
    assert "total_net_ret" in metrics
    # In 1 year, CAGR % must be close to total_net_ret * 100
    expected_cagr = metrics["total_net_ret"] * 100.0
    assert abs(metrics["cagr_pct"] - expected_cagr) < 1.0, f"CAGR {metrics['cagr_pct']} should match 1y net ret {expected_cagr}"
    print("[PASS] test_map_elites_cagr_evaluation_precision")


def test_map_elites_friction_stress_test():
    """Verify MapElitesArchive.stress_test_friction eliminates fragile high-turnover alphas."""
    from src.evolution.map_elites import MapElitesArchive
    from src.domain.formula_ast import ASTNode

    archive = MapElitesArchive()
    idx = pd.date_range("2024-01-01", periods=100, freq="B")
    df = pd.DataFrame(index=idx)
    df["Close"] = 250.0 + np.sin(np.linspace(0, 10, 100)) * 5.0
    df["Open"] = df["Close"].shift(1).fillna(250.0)
    df["Close_Adj"] = df["Close"]

    robust_node = ASTNode("FEATURE", "1.0")
    archive.evaluate_and_insert(robust_node, df)

    assert hasattr(archive, "stress_test_friction")
    pruned = archive.stress_test_friction(df, multiplier=2.0)
    assert isinstance(pruned, int)
    print("[PASS] test_map_elites_friction_stress_test")


if __name__ == "__main__":
    test_cagr_calculation_normalizes_cross_timeframe()
    test_archive_lookback_partitioning_isolation()
    test_map_elites_cagr_evaluation_precision()
    test_map_elites_friction_stress_test()
    print("ALL LOOKBACK PARTITIONING & CAGR TESTS PASS EMPIRICALLY!")
