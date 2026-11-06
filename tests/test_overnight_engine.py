"""Unit tests for Overnight Execution Engine & GIFT Nifty Ingestion (Wave 6 / P2)."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.gift_nifty_loader import GiftNiftyLoader


def test_gift_nifty_premarket_implied_gap():
    """Verify GiftNiftyLoader calculates implied opening gap with circuit limits and bias direction."""
    loader = GiftNiftyLoader()

    # Case 1: Positive overnight jump (+150 pts on 24,000 Nifty = +0.625%)
    res_bull = loader.calculate_implied_gap(gift_nifty_price=24150.0, nse_prev_close=24000.0)
    assert np.isclose(res_bull["implied_gap_pct"], 150.0 / 24000.0)
    assert res_bull["signal"] == "BULLISH_GAP"
    assert res_bull["circuit_clipped"] is False

    # Case 2: Deep gap crash down (-3000 pts = -12.5%, clipped to statutory circuit -10%)
    res_crash = loader.calculate_implied_gap(gift_nifty_price=21000.0, nse_prev_close=24000.0)
    assert np.isclose(res_crash["implied_gap_pct"], -0.10)
    assert res_crash["signal"] == "BEARISH_GAP"
    assert res_crash["circuit_clipped"] is True

    # Case 3: Flat overnight session
    res_flat = loader.calculate_implied_gap(gift_nifty_price=24010.0, nse_prev_close=24000.0)
    assert res_flat["signal"] == "NEUTRAL"
    print("[PASS] test_gift_nifty_premarket_implied_gap")


if __name__ == "__main__":
    test_gift_nifty_premarket_implied_gap()
    print("OVERNIGHT ENGINE TESTS PASSED!")
