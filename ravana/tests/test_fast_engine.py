"""Unit tests and benchmarks for NumbaFastBacktester."""
from __future__ import annotations

import time
import unittest
import numpy as np
import pandas as pd

from ravana.backtest.config import BacktestConfig, TradingRegime
from ravana.backtest.engine import BacktestResult
from ravana.backtest.fast_engine import FastBacktester, NumbaFastBacktester, compute_fast_atr
from ravana.backtest.orders import OrderSide


class TestFastEngine(unittest.TestCase):
    """Test suite verifying NumbaFastBacktester mechanics and performance."""

    def setUp(self) -> None:
        self.config = BacktestConfig(initial_capital=200000.0)
        self.engine = FastBacktester(self.config)

    def _build_test_df(self, n: int = 10, base_price: float = 100.0) -> pd.DataFrame:
        """Helper generating synthetic 5-min bars with timestamps."""
        times = pd.date_range("2026-01-05 09:15", periods=n, freq="5min")
        return pd.DataFrame(
            {
                "Open": np.full(n, base_price),
                "High": np.full(n, base_price + 0.5),
                "Low": np.full(n, base_price - 0.5),
                "Close": np.full(n, base_price),
                "Volume": np.full(n, 50000.0),
                "ATR": np.full(n, 2.0),
            },
            index=times,
        )

    def test_basic_contract_and_types(self) -> None:
        """Verify FastBacktester returns valid BacktestResult with required fields."""
        df = self._build_test_df(20, 100.0)
        signals = pd.Series(0.0, index=df.index)
        signals.iloc[0] = 1.0  # Enter at bar 0

        res = self.engine.run(df, signals, signal_threshold=0.5)
        self.assertIsInstance(res, BacktestResult)
        self.assertEqual(len(res.equity_curve), 20)
        self.assertEqual(len(res.cash_curve), 20)
        self.assertEqual(len(res.drawdown_curve), 20)
        self.assertIn("sharpe_ratio", res.metrics)
        self.assertIn("total_trades", res.metrics)
        self.assertGreater(len(res.trades), 0)

    def test_stop_loss_trigger(self) -> None:
        """Verify low <= trailing_stop triggers stop-loss exit at stop price."""
        # Entry at 100.0, ATR=2.0, initial_stop = 100.0 - (1.0 * 2.0) = 98.0
        df = self._build_test_df(5, 100.0)
        # Bar 1 plunges below 98.0
        df.loc[df.index[1], "Low"] = 97.0
        df.loc[df.index[1], "Close"] = 97.5

        signals = pd.Series([1.0, 0.0, 0.0, 0.0, 0.0], index=df.index)
        res = self.engine.run(df, signals, signal_threshold=0.5, initial_stop_atr=1.0)

        self.assertGreaterEqual(len(res.trades), 2)
        exit_trade = [t for t in res.trades if t.side == OrderSide.SELL][0]
        self.assertEqual(exit_trade.price, 98.0)

    def test_ratchet_layer_2_breakeven_plus_friction(self) -> None:
        """Verify high >= p_entry + 0.45 * atr raises stop to p_entry * (1 + friction)."""
        # Entry at 100.0, ATR=2.0, 0.45 * 2.0 = 0.90 -> threshold 100.90
        # Friction = 0.0005 -> new stop = 100.0 * 1.0005 = 100.05
        df = self._build_test_df(6, 100.0)
        # Bar 1 rallies to 101.0 (>= 100.90), triggering Ratchet 2
        df.loc[df.index[1], "High"] = 101.0
        df.loc[df.index[1], "Close"] = 100.80
        # Bar 2 opens at 100.80 (above stop) and drops to 99.50 (hits the new ratcheted stop of 100.05)
        df.loc[df.index[2], "Open"] = 100.80
        df.loc[df.index[2], "High"] = 100.80
        df.loc[df.index[2], "Low"] = 99.50
        df.loc[df.index[2], "Close"] = 99.80

        signals = pd.Series([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], index=df.index)
        res = self.engine.run(
            df, signals, signal_threshold=0.5, initial_stop_atr=1.0, friction_rate=0.0005
        )

        exit_trade = [t for t in res.trades if t.side == OrderSide.SELL][0]
        expected_stop = round(100.0 * 1.0005, 4)
        self.assertAlmostEqual(exit_trade.price, expected_stop, places=3)

    def test_ratchet_layer_3_mis_two_tranche(self) -> None:
        """Verify MIS two-tranche: high >= p_entry + 0.75 * atr sells 50% and ratchets stop."""
        # Entry at 100.0, ATR=2.0, shares = 1960. 0.75 * 2.0 = 1.50 -> threshold 101.50
        df = self._build_test_df(6, 100.0)
        # Bar 1 reaches 101.60 -> triggers tranche 1 exit (50% shares)
        df.loc[df.index[1], "High"] = 101.60
        df.loc[df.index[1], "Close"] = 101.20
        # Bar 2 opens at 101.20 and drops to 100.0 -> hits the new stop (100 + 0.25 * 2 = 100.50)
        df.loc[df.index[2], "Open"] = 101.20
        df.loc[df.index[2], "High"] = 101.20
        df.loc[df.index[2], "Low"] = 100.0

        signals = pd.Series([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], index=df.index)
        res = self.engine.run(df, signals, signal_threshold=0.5, is_mis=True)

        sells = [t for t in res.trades if t.side == OrderSide.SELL]
        self.assertEqual(len(sells), 2)  # Tranche 1 + Stop-loss on remainder
        buy_trade = res.trades[0]
        self.assertEqual(sells[0].shares, buy_trade.shares // 2)
        self.assertEqual(sells[0].price, 101.50)  # p_entry + 0.75 * atr
        self.assertEqual(sells[1].price, 100.50)  # p_entry + 0.25 * atr

    def test_ratchet_layer_4_chandelier_stop(self) -> None:
        """Verify Chandelier stop: high >= p_entry + 1.0 * atr moves stop to peak - 0.35 * atr."""
        # Entry at 100.0, ATR=2.0, 1.0 * 2.0 = 2.0 -> threshold 102.0
        # Target multiplier = 3.0 (target = 106.0) so 104.0 does not hit target
        df = self._build_test_df(6, 100.0)
        # Bar 1 surges to peak 104.0 -> stop moves to 104.0 - (0.35 * 2.0) = 103.30
        df.loc[df.index[1], "High"] = 104.0
        df.loc[df.index[1], "Close"] = 103.80
        # Bar 2 opens at 103.80 and drops to 102.0 (hits Chandelier stop 103.30)
        df.loc[df.index[2], "Open"] = 103.80
        df.loc[df.index[2], "High"] = 103.80
        df.loc[df.index[2], "Low"] = 102.0

        signals = pd.Series([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], index=df.index)
        res = self.engine.run(df, signals, signal_threshold=0.5, target_atr_mult=3.0, is_mis=False)

        exit_trade = [t for t in res.trades if t.side == OrderSide.SELL][0]
        self.assertEqual(exit_trade.price, 103.30)

    def test_target_fill(self) -> None:
        """Verify high >= target exits remaining shares at target price."""
        # Entry at 100.0, ATR=2.0, target_atr_mult=2.0 -> target = 104.0
        df = self._build_test_df(5, 100.0)
        # Bar 1 rallies to 104.50 >= 104.0
        df.loc[df.index[1], "High"] = 104.50

        signals = pd.Series([1.0, 0.0, 0.0, 0.0, 0.0], index=df.index)
        res = self.engine.run(df, signals, signal_threshold=0.5, target_atr_mult=2.0)

        exit_trade = [t for t in res.trades if t.side == OrderSide.SELL][0]
        self.assertEqual(exit_trade.price, 104.0)

    def test_dead_money_time_stop(self) -> None:
        """Verify bars_held >= 4 and peak_price < p_entry + 0.20 * atr exits flat at close."""
        # Entry at 100.0, ATR=2.0. 0.20 * 2.0 = 0.40 -> threshold 100.40
        # Price stays between 99.85 and 100.15 for 5 bars
        df = self._build_test_df(8, 100.0)
        for idx in df.index:
            df.loc[idx, "High"] = 100.15
            df.loc[idx, "Low"] = 99.85
            df.loc[idx, "Close"] = 100.05

        signals = pd.Series(0.0, index=df.index)
        signals.iloc[0] = 1.0

        res = self.engine.run(df, signals, signal_threshold=0.5)
        # Should exit on bar 3 (after 4 bars of holding: bars 0, 1, 2, 3) at bar close (100.05)
        exit_trade = [t for t in res.trades if t.side == OrderSide.SELL][0]
        self.assertEqual(exit_trade.price, 100.05)
        self.assertEqual(exit_trade.timestamp, df.index[3])

    def test_ieee_754_nan_safety(self) -> None:
        """Verify IEEE-754 np.isnan safety prevents invalid trades on NaN signals."""
        df = self._build_test_df(10, 100.0)
        signals = pd.Series([np.nan, np.nan, 1.0, np.nan, 0.0, np.nan, 0.0, 0.0, 0.0, 0.0], index=df.index)

        res = self.engine.run(df, signals, signal_threshold=0.5)
        self.assertGreater(len(res.trades), 0)
        self.assertEqual(res.trades[0].timestamp, df.index[2])

    def test_benchmark_100k_bars_sub_50ms(self) -> None:
        """Benchmark: Verify 100,000 bars executes in strictly under 50ms."""
        n = 100000
        dates = pd.date_range("2020-01-01 09:15", periods=n, freq="5min")
        rng = np.random.default_rng(42)
        base = 500.0 + np.cumsum(rng.normal(0, 0.1, n))

        df = pd.DataFrame(
            {
                "Open": base,
                "High": base + rng.uniform(0.1, 1.5, n),
                "Low": base - rng.uniform(0.1, 1.5, n),
                "Close": base + rng.normal(0, 0.2, n),
                "Volume": rng.uniform(10000, 50000, n),
                "ATR": np.full(n, 1.25),
            },
            index=dates,
        )
        signals = rng.uniform(-1.0, 1.0, n)

        # Warmup JIT cache on small slice
        self.engine.run_fast(df.iloc[:20], signals[:20], signal_threshold=0.5)

        # Benchmark run_fast execution over 100,000 bars
        start_time = time.perf_counter()
        eq, cash, records, count = self.engine.run_fast(
            df=df, signals=signals, signal_threshold=0.6, is_mis=True
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        print(f"\n[BENCHMARK] Executed {n:,} bars in {elapsed_ms:.2f} ms with {count} trade events.")
        self.assertLess(elapsed_ms, 50.0, f"Execution time {elapsed_ms:.2f} ms exceeded 50ms limit!")
        self.assertEqual(len(eq), n)
        self.assertGreater(count, 0)


if __name__ == "__main__":
    unittest.main()
