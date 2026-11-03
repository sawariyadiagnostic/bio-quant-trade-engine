"""Unit tests for VectorDataLake (Wave 1: Task 1.1)."""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_vector_lake_returns_and_statutory_clipping():
    """Verify VectorDataLake computes overnight and intraday returns with NSE circuit limits."""
    from src.data.vector_lake import VectorDataLake

    lake = VectorDataLake.load_mock(periods=100, ticker="NIFTYBEES.NS")
    assert lake.ticker == "NIFTYBEES.NS"
    assert "gross_overnight_rets" in lake.tensors
    assert "gross_intraday_rets" in lake.tensors
    assert "gross_close_to_close_rets" in lake.tensors

    overnight = lake.tensors["gross_overnight_rets"]
    intraday = lake.tensors["gross_intraday_rets"]

    assert isinstance(overnight, np.ndarray)
    assert isinstance(intraday, np.ndarray)
    assert len(overnight) == 100
    assert len(intraday) == 100

    # Statutory circuit limits (-10% to +10%)
    assert float(np.max(overnight)) <= 0.100001
    assert float(np.min(overnight)) >= -0.100001
    assert float(np.max(intraday)) <= 0.100001
    assert float(np.min(intraday)) >= -0.100001
    print("[PASS] test_vector_lake_returns_and_statutory_clipping")


def test_vector_lake_standard_features():
    """Verify standard features are pre-computed and normalized."""
    from src.data.vector_lake import VectorDataLake

    lake = VectorDataLake.load_mock(periods=120)
    expected_feats = [
        "Close_Adj", "Open", "High", "Low", "Volume",
        "Dist_VWAP", "True_Range", "Vol_ZScore",
        "Parkinson_Vol", "Garman_Klass_Vol", "Downside_Vol_Ratio"
    ]
    for feat in expected_feats:
        assert feat in lake.tensors, f"Missing feature: {feat}"
        assert not np.isnan(lake.tensors[feat]).any(), f"NaN found in feature {feat}"

    print("[PASS] test_vector_lake_standard_features")


if __name__ == "__main__":
    test_vector_lake_returns_and_statutory_clipping()
    test_vector_lake_standard_features()
