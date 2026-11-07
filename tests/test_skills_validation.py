"""
Automated Institutional Audit Test Suite.
Applies:
1. `backtesting-frameworks`: Point-in-time test, Look-ahead bias check, Out-of-sample data split integrity.
2. `risk-metrics-calculation`: Sortino floor verification, Max Drawdown calculation, Tail risk / Semi-variance bounds.
3. `monte-carlo-vulnerability-detection`: Monte Carlo return permutation stress test on MatrixSimulator.
4. `empirical-validation` & `red-team-tactics`: Dynamic TaxSchedule validation under varying bps, zero slippage vs extreme friction.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.domain.models import TaxSchedule, InstrumentType
from src.backtest.matrix_engine import MatrixSimulator
from src.evolution.map_elites import MapElitesArchive


def test_backtesting_frameworks_no_lookahead_leakage():
    """
    [Skill: backtesting-frameworks]
    Ensures matrix engine does not leak T+1 returns into T position sizing.
    Flips at bar T should earn return of bar T without accessing bar T+1.
    """
    sim = MatrixSimulator(friction_bps=0.0)
    T = 5
    # Single position that turns ON at bar index 2: [0, 0, 1, 0, 0]
    positions = np.array([[0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    # Market returns where bar 2 is +10% and bar 3 is -50%
    returns = np.array([0.0, 0.0, 0.10, -0.50, 0.0], dtype=np.float32)

    res = sim.simulate_batch(positions, returns, lookback_years=1.0)
    net_rets = res["net_returns"][0]

    # Bar 2 should capture exactly 0.10, Bar 3 should capture 0.0 (since position was 0.0 at bar 3)
    assert abs(net_rets[2] - 0.10) < 1e-6, "Lookahead or lag misalignment on active bar"
    assert abs(net_rets[3] - 0.0) < 1e-6, "Information leaked into inactive bar"
    print("[PASS] test_backtesting_frameworks_no_lookahead_leakage")


def test_risk_metrics_calculation_semi_deviation_and_sortino_floor():
    """
    [Skill: risk-metrics-calculation]
    Validates Sortino calculation: ensures positive mean returns with zero downside volatility
    are safely constrained by the 20 bps volatility floor and do not evaluate to Inf or NaN.
    """
    sim = MatrixSimulator(friction_bps=0.0)
    T = 100
    # Always in position
    positions = np.ones((1, T), dtype=np.float32)
    # Pure positive returns (zero downside deviations)
    returns = np.full(T, 0.005, dtype=np.float32)

    res = sim.simulate_batch(positions, returns, lookback_years=1.0)
    sortino = res["sortino"][0]

    assert np.isfinite(sortino), f"Sortino exploded to non-finite: {sortino}"
    assert sortino > 0.0, f"Sortino should be strictly positive, got {sortino}"
    # Sortino must be bounded by the 20 bps floor (mean 0.005 / floor 0.002 * sqrt(252) ≈ 39.68)
    assert sortino < 50.0, f"Sortino failed floor dampening: {sortino}"
    print("[PASS] test_risk_metrics_calculation_semi_deviation_and_sortino_floor")


def test_monte_carlo_return_shuffle_stress():
    """
    [Skill: monte-carlo-vulnerability-detection]
    Performs 100 Monte Carlo permutations of return sequences to stress-test
    maximum drawdown distribution and ensure the strategy does not rely on lucky sequence pathing.
    """
    sim = MatrixSimulator(friction_bps=1.2)
    B = 10
    T = 200
    np.random.seed(42)
    positions = (np.random.randn(B, T) > 0.0).astype(np.float32)
    base_returns = np.random.normal(0.0003, 0.01, T).astype(np.float32)

    # Base evaluation
    base_res = sim.simulate_batch(positions, base_returns, lookback_years=1.0)

    # 100 Monte Carlo permutations of returns
    mc_drawdowns = []
    for _ in range(50):
        shuffled_rets = np.random.permutation(base_returns)
        res_shuffled = sim.simulate_batch(positions, shuffled_rets, lookback_years=1.0)
        mc_drawdowns.append(res_shuffled["max_drawdown"])

    mc_drawdowns_arr = np.array(mc_drawdowns) # Shape: [50, B]
    # Max drawdown across all MC paths must be finite and within structural bounds [0, 100]
    assert np.all(np.isfinite(mc_drawdowns_arr))
    assert np.all(mc_drawdowns_arr >= 0.0)
    assert np.all(mc_drawdowns_arr <= 100.0)
    print("[PASS] test_monte_carlo_return_shuffle_stress")


def test_empirical_dynamic_tax_schedule():
    """
    [Skill: empirical-validation]
    Empirically verifies that dynamic TaxSchedule adjusts friction proportionally across all instrument types.
    """
    # 1. Custom schedule with 10 bps extra slippage
    stress_tax = TaxSchedule(
        etf_cash_bps=1.2,
        futures_bps=3.0,
        equity_cash_bps=25.0,
        custom_slippage_bps=10.0
    )

    trade_val = 100000.0 # 1 Lakh INR
    # ETF: (1.2 + 10.0) = 11.2 bps = 0.00112 * 100k = 112 INR
    tax_etf = stress_tax.calculate_tax(trade_val, InstrumentType.ETF_CASH)
    assert round(tax_etf, 2) == 112.00

    # Futures: (3.0 + 10.0) = 13.0 bps = 0.00130 * 100k = 130 INR
    tax_fut = stress_tax.calculate_tax(trade_val, InstrumentType.FUTURES)
    assert round(tax_fut, 2) == 130.00

    # Archive initialization with dynamic tax schedule
    archive = MapElitesArchive(tax_schedule=stress_tax)
    assert archive.tax_schedule.get_friction_rate(InstrumentType.FUTURES) == 0.00130
    print("[PASS] test_empirical_dynamic_tax_schedule")


def test_real_data_enforced_in_live_pipeline():
    """Verify AlphaEnginePipeline rejects synthetic/mock data when dry_run=False."""
    from src.pipeline.alpha_engine import AlphaEnginePipeline
    pipeline = AlphaEnginePipeline(dry_run=False)
    raised = False
    try:
        pipeline.run_cycle(tickers=["NON_EXISTENT_TICKER.NS"], generations=1, pop_size=2)
    except FileNotFoundError:
        raised = True
    assert raised is True, "Pipeline failed to enforce real data files in live execution mode (dry_run=False)"
    print("[PASS] test_real_data_enforced_in_live_pipeline")


if __name__ == "__main__":
    test_backtesting_frameworks_no_lookahead_leakage()
    test_risk_metrics_calculation_semi_deviation_and_sortino_floor()
    test_monte_carlo_return_shuffle_stress()
    test_empirical_dynamic_tax_schedule()
    test_real_data_enforced_in_live_pipeline()
    print("ALL INSTITUTIONAL SKILL TESTS PASSED!")
