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


def test_daily_resetting_vwap_and_auction_jump():
    """Verify overnight auction jump decomposition and session-resetting VWAP."""
    from src.data.vector_lake import VectorDataLake

    lake = VectorDataLake.load_mock(periods=50)
    assert "overnight_auction_jump" in lake.tensors
    assert "intraday_core_return" in lake.tensors

    o_jump = lake.tensors["overnight_auction_jump"]
    i_core = lake.tensors["intraday_core_return"]
    assert len(o_jump) == 50
    assert len(i_core) == 50
    assert o_jump[0] == 0.0

    # Multi-day intraday series test: VWAP must reset daily
    idx = pd.date_range("2024-01-01 09:15", periods=10, freq="15min").union(
        pd.date_range("2024-01-02 09:15", periods=10, freq="15min")
    )
    df_intra = pd.DataFrame(index=idx)
    df_intra["Open"] = 100.0
    df_intra["High"] = 105.0
    df_intra["Low"] = 95.0
    df_intra["Close"] = 100.0
    df_intra["Volume"] = 1000.0
    # Day 2 first bar has price 200.0, volume 1000.0
    df_intra.loc[df_intra.index >= "2024-01-02", "Close"] = 200.0
    df_intra.loc[df_intra.index >= "2024-01-02", "High"] = 205.0
    df_intra.loc[df_intra.index >= "2024-01-02", "Low"] = 195.0
    df_intra.loc[df_intra.index >= "2024-01-02", "Open"] = 200.0

    lake_intra = VectorDataLake.from_dataframe(df_intra, ticker="NIFTYBEES.NS")
    # At start of Day 2, Dist_VWAP should be near 0 because typical price is 200 and daily VWAP reset to 200
    day2_first_bar_dist = lake_intra.tensors["Dist_VWAP"][10]
    assert abs(day2_first_bar_dist) < 0.01, f"VWAP did not reset on Day 2: dist={day2_first_bar_dist}"

    print("[PASS] test_daily_resetting_vwap_and_auction_jump")


if __name__ == "__main__":
    test_vector_lake_returns_and_statutory_clipping()
    test_vector_lake_standard_features()
    test_daily_resetting_vwap_and_auction_jump()
