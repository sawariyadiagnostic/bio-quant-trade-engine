"""Unit tests for MatrixSimulator (Wave 1: Task 1.2)."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_matrix_simulator_batch_speed_and_sortino():
    """Verify MatrixSimulator evaluates batch of formulas with institutional metrics and Sortino floor."""
    from src.backtest.matrix_engine import MatrixSimulator

    sim = MatrixSimulator(friction_bps=1.2)
    B, T = 100, 500  # 100 formulas, 500 bars
    np.random.seed(42)
    mock_positions = (np.random.randn(B, T) > 0.0).astype(np.float32)
    mock_returns = np.random.normal(0.0005, 0.01, T).astype(np.float32)

    res = sim.simulate_batch(mock_positions, mock_returns, lookback_years=2.0)

    assert "cagr" in res
    assert "sortino" in res
    assert "max_drawdown" in res
    assert "win_rate" in res
    assert "net_returns" in res

    assert res["cagr"].shape == (B,)
    assert res["sortino"].shape == (B,)
    assert res["max_drawdown"].shape == (B,)
    assert res["win_rate"].shape == (B,)

    # Verify Sortino values are bounded and realistic (20 bps floor prevents explosion)
    assert np.all(res["sortino"] < 100.0), f"Sortino exploded: {res['sortino'].max()}"
    assert np.all(res["max_drawdown"] >= 0.0)
    assert np.all(res["win_rate"] >= 0.0) and np.all(res["win_rate"] <= 100.0)

    print("[PASS] test_matrix_simulator_batch_speed_and_sortino")


def test_matrix_simulator_friction_deduction():
    """Verify statutory friction is strictly deducted on trade flips."""
    from src.backtest.matrix_engine import MatrixSimulator

    sim = MatrixSimulator(friction_bps=1.2)  # 0.00012 friction
    # Formula that flips every single day: 0 -> 1 -> 0 -> 1
    T = 10
    positions = np.array([[0, 1, 0, 1, 0, 1, 0, 1, 0, 1]], dtype=np.float32)
    flat_returns = np.zeros(T, dtype=np.float32)

    res = sim.simulate_batch(positions, flat_returns, lookback_years=1.0)
    # Total flips should be 9, friction = 9 * 0.00012 = 0.00108
    assert res["cagr"][0] < 0.0  # Must be negative due to friction
    print("[PASS] test_matrix_simulator_friction_deduction")


def test_matrix_simulator_3d_multi_asset():
    """Verify 3D tensor simulation (B, N, T) across multi-asset ETF portfolio."""
    from src.backtest.matrix_engine import MatrixSimulator

    sim = MatrixSimulator(friction_bps=1.2)
    B, N, T = 10, 3, 100  # 10 strategies, 3 assets (NIFTYBEES, BANKBEES, GOLDBEES), 100 bars
    positions = np.zeros((B, N, T), dtype=np.float32)
    positions[:, 0, :] = 0.33
    positions[:, 1, :] = 0.33
    positions[:, 2, :] = 0.34

    np.random.seed(42)
    returns = np.random.normal(0.001, 0.01, (N, T)).astype(np.float32)

    res = sim.simulate_batch(positions, returns, lookback_years=1.0)
    assert res["cagr"].shape == (B,)
    assert res["sortino"].shape == (B,)
    assert res["net_returns"].shape == (B, T)
    assert np.all(np.isfinite(res["cagr"]))
    print("[PASS] test_matrix_simulator_3d_multi_asset")


if __name__ == "__main__":
    test_matrix_simulator_batch_speed_and_sortino()
    test_matrix_simulator_friction_deduction()
    test_matrix_simulator_3d_multi_asset()
