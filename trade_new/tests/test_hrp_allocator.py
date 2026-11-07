"""Unit tests for Hierarchical Risk Parity (HRP) Allocator (Wave 4: Task 4.2)."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_hrp_allocation_weights():
    """Verify HRPAllocator computes valid risk parity weights summing to 1.0 without matrix inversion."""
    from src.execution.hrp_allocator import HRPAllocator

    np.random.seed(42)
    T, N = 500, 5
    # 5 strategy return streams: 2 highly correlated momentum strategies, 3 diverse strategies
    base = np.random.normal(0.001, 0.02, T)
    ret_0 = base + np.random.normal(0, 0.005, T)
    ret_1 = base + np.random.normal(0, 0.005, T)  # Highly correlated with 0
    ret_2 = np.random.normal(0.0005, 0.015, T)
    ret_3 = np.random.normal(0.0008, 0.025, T)
    ret_4 = np.random.normal(0.0002, 0.010, T)

    rets_matrix = np.column_stack([ret_0, ret_1, ret_2, ret_3, ret_4])

    weights = HRPAllocator.compute_weights(rets_matrix)

    assert isinstance(weights, np.ndarray)
    assert len(weights) == 5
    assert np.isclose(np.sum(weights), 1.0)
    assert np.all(weights >= 0.0)

    # Correlated strategies 0 & 1 should have their joint weight curtailed relative to uncorrelated strategies
    joint_corr_weight = weights[0] + weights[1]
    assert joint_corr_weight < 0.60
    print("[PASS] test_hrp_allocation_weights")


def test_hrp_allocation_edge_cases():
    """Verify HRPAllocator handles single asset and identical assets gracefully."""
    from src.execution.hrp_allocator import HRPAllocator

    # Single asset
    single_ret = np.random.normal(0.001, 0.01, (100, 1))
    w_single = HRPAllocator.compute_weights(single_ret)
    assert np.isclose(w_single[0], 1.0)

    # Identical assets
    two_identical = np.column_stack([single_ret, single_ret])
    w_two = HRPAllocator.compute_weights(two_identical)
    assert np.isclose(np.sum(w_two), 1.0)
    assert np.isclose(w_two[0], 0.5)
    assert np.isclose(w_two[1], 0.5)
    print("[PASS] test_hrp_allocation_edge_cases")


def test_hrp_ward_linkage_and_ledoit_wolf():
    """Verify HRPAllocator supports Ward/Average linkage and Ledoit-Wolf shrinkage."""
    from src.execution.hrp_allocator import HRPAllocator

    np.random.seed(42)
    T, N = 100, 4
    rets = np.random.normal(0.0005, 0.015, (T, N))

    # Test linkage_method and use_shrinkage
    w_ward = HRPAllocator.compute_weights(rets, linkage_method="ward", use_shrinkage=True)
    assert isinstance(w_ward, np.ndarray)
    assert len(w_ward) == N
    assert np.isclose(np.sum(w_ward), 1.0)
    assert np.all(w_ward >= 0.0)

    w_avg = HRPAllocator.compute_weights(rets, linkage_method="average", use_shrinkage=True)
    assert np.isclose(np.sum(w_avg), 1.0)
    assert np.all(w_avg >= 0.0)
    print("[PASS] test_hrp_ward_linkage_and_ledoit_wolf")


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

