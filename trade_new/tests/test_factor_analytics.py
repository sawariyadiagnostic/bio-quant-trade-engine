"""Unit tests for Factor Analytics (Wave 1: Task 1.3 - Qlib Standards)."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_factor_ic_calculation():
    """Verify Information Coefficient (IC) and Rank IC calculations."""
    from src.backtest.factor_analytics import calculate_factor_ic

    np.random.seed(42)
    # Generate signals that are moderately predictive of forward returns
    noise = np.random.normal(0, 0.5, 100)
    signals = np.linspace(-2.0, 2.0, 100)
    fwd_returns = 0.01 * signals + 0.005 * noise

    ic_metrics = calculate_factor_ic(signals, fwd_returns)

    assert "ic" in ic_metrics
    assert "rank_ic" in ic_metrics
    assert "p_value" in ic_metrics

    assert 0.70 <= ic_metrics["ic"] <= 1.0
    assert 0.70 <= ic_metrics["rank_ic"] <= 1.0
    assert ic_metrics["p_value"] < 0.001
    print("[PASS] test_factor_ic_calculation")


def test_factor_decay_and_autocorrelation():
    """Verify factor autocorrelation and multi-lag decay profiling."""
    from src.backtest.factor_analytics import calculate_factor_decay

    np.random.seed(42)
    T = 200
    # Create persistent signal (e.g. slow EMA)
    signal = np.cumsum(np.random.normal(0, 1, T))
    decay = calculate_factor_decay(signal, max_lags=5)

    assert len(decay) == 5
    assert decay[0] > 0.80  # 1-day lag should be high for persistent signal
    assert decay[4] <= decay[0]  # Monotonically decaying autocorrelation
    print("[PASS] test_factor_decay_and_autocorrelation")


def load_tests(loader, standard_tests, pattern):
    import inspect
    import unittest
    suite = unittest.TestSuite()
    current_module = sys.modules[__name__]
    for name, func in inspect.getmembers(current_module, inspect.isfunction):
        if name.startswith("test_"):
            suite.addTest(unittest.FunctionTestCase(func))
    return suite


if __name__ == "__main__":
    import unittest
    unittest.main()

