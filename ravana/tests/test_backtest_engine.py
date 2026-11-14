"""Unit tests for Muscle 11 Event-Driven Backtesting Engine."""
import unittest
import numpy as np
import pandas as pd

from ravana.backtest.config import (
    AssetType,
    BacktestConfig,
    TradingRegime,
)
from ravana.backtest.engine import EventDrivenBacktester
from ravana.backtest.friction import IndianFrictionModel
from ravana.backtest.orders import OrderSide, Position, Trade
from ravana.backtest.slippage import MicrostructureSlippageModel


class TestBacktestEngine(unittest.TestCase):
    """Test suite for Muscle 11 causal event-driven backtesting engine."""

    def test_friction_etf_delivery(self):
        """Verify ETF delivery has concessional 0.001% STT and ₹17 DP fee on sell only."""
        # Buy 1000 shares @ ₹500 = ₹5,00,000 turnover
        buy_f = IndianFrictionModel.calculate_friction(
            side="BUY", price=500.0, shares=1000,
            asset_type=AssetType.ETF, regime=TradingRegime.OVERNIGHT,
        )
        self.assertEqual(buy_f.stt_inr, 0.0)  # No STT on buy for ETF delivery
        self.assertEqual(buy_f.dp_charges_inr, 0.0)  # No DP fee on buy
        self.assertAlmostEqual(buy_f.stamp_duty_inr, 75.0, places=1)  # 0.015% stamp duty

        # Sell 1000 shares @ ₹505 = ₹5,05,000 turnover
        sell_f = IndianFrictionModel.calculate_friction(
            side="SELL", price=505.0, shares=1000,
            asset_type=AssetType.ETF, regime=TradingRegime.OVERNIGHT,
            fixed_dp_fee_inr=17.0,
        )
        self.assertAlmostEqual(sell_f.stt_inr, 5.05, places=2)  # 0.001% STT
        self.assertEqual(sell_f.dp_charges_inr, 17.0)  # ₹17 CDSL DP fee
        self.assertEqual(sell_f.stamp_duty_inr, 0.0)  # No stamp duty on sell

    def test_friction_stock_delivery(self):
        """Verify stock delivery incurs 0.1% STT on both buy and sell."""
        buy_f = IndianFrictionModel.calculate_friction(
            side="BUY", price=1000.0, shares=100,
            asset_type=AssetType.EQUITY_STOCK, regime=TradingRegime.OVERNIGHT,
        )
        self.assertAlmostEqual(buy_f.stt_inr, 100.0, places=1)  # 0.1% STT

        sell_f = IndianFrictionModel.calculate_friction(
            side="SELL", price=1050.0, shares=100,
            asset_type=AssetType.EQUITY_STOCK, regime=TradingRegime.OVERNIGHT,
            fixed_dp_fee_inr=17.0,
        )
        self.assertAlmostEqual(sell_f.stt_inr, 105.0, places=1)  # 0.1% STT
        self.assertEqual(sell_f.dp_charges_inr, 17.0)

    def test_friction_intraday_mis(self):
        """Verify intraday MIS has zero DP fee, statutory NIL STT on ETFs, and 0.025% on stocks."""
        # Stock intraday sell incurs 0.025% STT
        sell_stock = IndianFrictionModel.calculate_friction(
            side="SELL", price=500.0, shares=1000,
            asset_type=AssetType.EQUITY_STOCK, regime=TradingRegime.INTRADAY_MIS,
        )
        self.assertEqual(sell_stock.dp_charges_inr, 0.0)  # Zero DP fee for intraday
        self.assertAlmostEqual(sell_stock.stt_inr, 125.0, places=1)  # 0.025% STT

        # ETF intraday sell incurs 0.0% STT (NIL per Indian Finance Act)
        sell_etf = IndianFrictionModel.calculate_friction(
            side="SELL", price=500.0, shares=1000,
            asset_type=AssetType.ETF, regime=TradingRegime.INTRADAY_MIS,
        )
        self.assertEqual(sell_etf.dp_charges_inr, 0.0)
        self.assertEqual(sell_etf.stt_inr, 0.0)

    def test_slippage_and_circuit_locks(self):
        """Verify circuit lock rejects fills and volume participation caps orders."""
        model = MicrostructureSlippageModel(max_volume_participation=0.05)

        # Normal fill
        fill = model.simulate_fill(
            side="BUY", desired_shares=100, nominal_price=500.0,
            bar_volume=10000, bar_high=502.0, bar_low=498.0
        )
        self.assertEqual(fill.filled_shares, 100)
        self.assertFalse(fill.is_circuit_locked)
        self.assertFalse(fill.is_volume_capped)
        self.assertGreater(fill.executed_price, 500.0)

        # Circuit locked bar (High == Low and zero volume)
        locked_fill = model.simulate_fill(
            side="BUY", desired_shares=100, nominal_price=500.0,
            bar_volume=0, bar_high=500.0, bar_low=500.0
        )
        self.assertTrue(locked_fill.is_circuit_locked)
        self.assertEqual(locked_fill.filled_shares, 0)

        # Volume capped fill (desired 500 shares, volume 1000 -> max 5% = 50 shares)
        capped_fill = model.simulate_fill(
            side="BUY", desired_shares=500, nominal_price=500.0,
            bar_volume=1000, bar_high=502.0, bar_low=498.0
        )
        self.assertEqual(capped_fill.filled_shares, 50)
        self.assertTrue(capped_fill.is_volume_capped)

    def test_overnight_backtest_execution(self):
        """Verify causal overnight 15:20 entry -> 09:15 exit simulation."""
        # Generate 2 days of 5m bars: 15:20 on Day 1, 09:15 on Day 2
        dates = pd.date_range("2026-01-01 09:15", "2026-01-02 15:30", freq="5min")
        # Filter to NSE hours [09:15, 15:30]
        dates = [d for d in dates if "09:15" <= d.strftime("%H:%M") <= "15:25"]
        n_bars = len(dates)

        df = pd.DataFrame({
            "Open": np.full(n_bars, 500.0),
            "High": np.full(n_bars, 502.0),
            "Low": np.full(n_bars, 498.0),
            "Close": np.full(n_bars, 500.0),
            "Volume": np.full(n_bars, 100000.0),
        }, index=pd.DatetimeIndex(dates))

        # Set gap-up on Day 2 morning open to test profitable overnight exit
        day2_mask = [d.date() == pd.Timestamp("2026-01-02").date() for d in df.index]
        df.loc[day2_mask, "Open"] = 505.0  # 1% overnight gap up
        df.loc[day2_mask, "Close"] = 505.0

        # Positive signal at 15:20 on Day 1
        signals = pd.Series(0.0, index=df.index)
        for i, ts in enumerate(df.index):
            if ts.strftime("%H:%M") == "15:20" and ts.date() == pd.Timestamp("2026-01-01").date():
                signals.iloc[i] = 1.0

        cfg = BacktestConfig(
            initial_capital=200000.0,
            asset_type=AssetType.ETF,
            regime=TradingRegime.OVERNIGHT,
            fixed_dp_fee_inr=17.0,
        )
        engine = EventDrivenBacktester(cfg)
        result = engine.run(df, signals, ticker="BANKBEES.NS")

        self.assertGreaterEqual(len(result.trades), 2)  # Buy on Day 1 close/open, Sell on Day 2 open
        self.assertIn("sharpe_ratio", result.metrics)
        self.assertIn("total_fees_inr", result.metrics)
        self.assertGreater(result.metrics["final_equity"], 0.0)
        self.assertGreaterEqual(result.metrics["total_dp_fees_inr"], 17.0)

    def test_zero_volume_fill_rejection(self):
        """Verify zero or NaN volume rejects order fills rather than permitting infinite fills."""
        model = MicrostructureSlippageModel(max_volume_participation=0.05)
        # Zero volume
        fill_zero = model.simulate_fill(
            side="BUY", desired_shares=1000, nominal_price=500.0,
            bar_volume=0.0, bar_high=502.0, bar_low=498.0
        )
        self.assertEqual(fill_zero.filled_shares, 0)
        self.assertTrue(fill_zero.is_volume_capped)

        # NaN volume
        fill_nan = model.simulate_fill(
            side="BUY", desired_shares=1000, nominal_price=500.0,
            bar_volume=float("nan"), bar_high=502.0, bar_low=498.0
        )
        self.assertEqual(fill_nan.filled_shares, 0)

    def test_cash_ledger_synchronization(self):
        """Verify cash curve and equity curve synchronize on terminal force-close."""
        dates = pd.date_range("2026-01-01 09:15", "2026-01-01 15:25", freq="5min")
        dates = [d for d in dates if "09:15" <= d.strftime("%H:%M") <= "15:25"]
        n_bars = len(dates)
        df = pd.DataFrame({
            "Open": np.full(n_bars, 500.0),
            "High": np.full(n_bars, 501.0),
            "Low": np.full(n_bars, 499.0),
            "Close": np.full(n_bars, 500.0),
            "Volume": np.full(n_bars, 50000.0),
        }, index=pd.DatetimeIndex(dates))

        # Enter at 15:20 so position is open at termination
        signals = pd.Series(0.0, index=df.index)
        signals.iloc[-2] = 1.0  # Buy on second-to-last bar

        cfg = BacktestConfig(initial_capital=200000.0, regime=TradingRegime.OVERNIGHT)
        engine = EventDrivenBacktester(cfg)
        result = engine.run(df, signals, ticker="BANKBEES.NS")

        # Cash and equity must match exactly on final bar
        self.assertAlmostEqual(result.cash_curve.iloc[-1], result.equity_curve.iloc[-1], places=2)
        self.assertGreater(result.cash_curve.iloc[-1], 0.0)

    def test_backtest_config_phase_a_defaults(self):
        """Verify BacktestConfig has Phase A signal_exit_threshold and min_hold_bars."""
        cfg = BacktestConfig()
        self.assertEqual(cfg.signal_exit_threshold, 0.0)
        self.assertEqual(cfg.min_hold_bars, 2)

    def test_intraday_mis_signal_reversal_exit(self):
        """Verify INTRADAY_MIS respects min_hold_bars hysteresis before signal-reversal exit."""
        dates = pd.date_range("2026-01-01 09:15", "2026-01-01 15:25", freq="5min")
        n_bars = len(dates)
        df = pd.DataFrame(
            {
                "Open": np.full(n_bars, 500.0),
                "High": np.full(n_bars, 502.0),
                "Low": np.full(n_bars, 498.0),
                "Close": np.full(n_bars, 500.0),
                "Volume": np.full(n_bars, 50000.0),
            },
            index=dates,
        )

        signals = pd.Series(0.0, index=df.index)
        # Bar 1 (09:20): Signal = +1.0 triggers BUY
        signals.iloc[1] = 1.0
        # Bar 2 (09:25): BUY fills at Open. Signal reverses to -1.0.
        # But held = 1 < min_hold_bars (2), so exit should NOT trigger yet!
        signals.iloc[2] = -1.0
        # Bar 3 (09:30): Held = 2 >= min_hold_bars. Signal still -1.0 -> SELL order emitted!
        signals.iloc[3] = -1.0

        cfg = BacktestConfig(
            initial_capital=200000.0,
            regime=TradingRegime.INTRADAY_MIS,
            min_hold_bars=2,
            signal_exit_threshold=0.0,
        )
        engine = EventDrivenBacktester(cfg)
        result = engine.run(df, signals, ticker="BANKBEES.NS")

        # Expect at least 2 trades (BUY at bar 2, SELL at bar 4)
        self.assertGreaterEqual(len(result.trades), 2)
        buy_trade = result.trades[0]
        sell_trade = result.trades[1]
        self.assertEqual(buy_trade.side, OrderSide.BUY)
        self.assertEqual(sell_trade.side, OrderSide.SELL)
        # Fills should be at 09:25 (bar 2) and 09:35 (bar 4)
        self.assertEqual(buy_trade.timestamp, df.index[2])
        self.assertEqual(sell_trade.timestamp, df.index[4])

    def test_intraday_mis_entry_cutoff_and_square_off(self):
        """Verify INTRADAY_MIS prevents entries after 14:45 and auto squares off at 15:15."""
        dates = pd.date_range("2026-01-01 09:15", "2026-01-01 15:25", freq="5min")
        n_bars = len(dates)
        df = pd.DataFrame(
            {
                "Open": np.full(n_bars, 500.0),
                "High": np.full(n_bars, 502.0),
                "Low": np.full(n_bars, 498.0),
                "Close": np.full(n_bars, 500.0),
                "Volume": np.full(n_bars, 50000.0),
            },
            index=dates,
        )

        # Trigger BUY at 14:50 (past 14:45 cutoff) -> should be rejected/ignored
        signals_late = pd.Series(0.0, index=df.index)
        for i, ts in enumerate(df.index):
            if ts.strftime("%H:%M") == "14:50":
                signals_late.iloc[i] = 1.0

        cfg = BacktestConfig(regime=TradingRegime.INTRADAY_MIS)
        engine = EventDrivenBacktester(cfg)
        res_late = engine.run(df, signals_late, ticker="BANKBEES.NS")
        self.assertEqual(len(res_late.trades), 0)

        # Trigger BUY at 10:00 and hold past 15:15 -> auto square-off
        signals_sq = pd.Series(0.0, index=df.index)
        for i, ts in enumerate(df.index):
            if ts.strftime("%H:%M") == "10:00":
                signals_sq.iloc[i] = 1.0

        res_sq = engine.run(df, signals_sq, ticker="BANKBEES.NS")
        self.assertGreaterEqual(len(res_sq.trades), 2)
        # Second trade should be the 15:15 or later square-off
        sq_trade = res_sq.trades[-1]
        self.assertEqual(sq_trade.side, OrderSide.SELL)
        self.assertGreaterEqual(sq_trade.timestamp.strftime("%H:%M"), "15:15")

    def test_continuation_gate_saves_dp_fees(self):
        """Verify continuation gate skips 09:15 exits on strong signal, saving DP fees and lifting net PnL."""
        dates = pd.date_range("2026-01-05 09:15", "2026-01-07 15:25", freq="5min")
        dates = [d for d in dates if "09:15" <= d.strftime("%H:%M") <= "15:25"]
        n = len(dates)

        # 3-day rising price trend (500 to 520)
        df = pd.DataFrame(
            {
                "Open": np.linspace(500.0, 520.0, n),
                "High": np.linspace(501.0, 521.0, n),
                "Low": np.linspace(499.0, 519.0, n),
                "Close": np.linspace(500.0, 520.0, n),
                "Volume": np.full(n, 50000.0),
            },
            index=pd.DatetimeIndex(dates),
        )

        # Persistent positive alpha signal (0.8)
        signals = pd.Series(0.8, index=df.index)

        # 1. Without continuation gate: 09:15 exits happen, debiting ₹17 DP fee per exit
        cfg_no_gate = BacktestConfig(
            initial_capital=200000.0,
            asset_type=AssetType.ETF,
            regime=TradingRegime.OVERNIGHT,
            fixed_dp_fee_inr=17.0,
            hold_on_strong_continuation=False,
        )
        engine_no_gate = EventDrivenBacktester(cfg_no_gate)
        res_no_gate = engine_no_gate.run(df, signals, ticker="BANKBEES.NS")

        self.assertEqual(res_no_gate.metrics["continuation_holds_count"], 0)
        self.assertEqual(res_no_gate.metrics["total_dp_fees_inr"], 51.0)  # 3 days * ₹17 DP fee
        self.assertEqual(len(res_no_gate.trades), 6)  # 3 daily entry/exit round-trips

        # 2. With continuation gate: 09:15 exits skipped when signal >= continuation_threshold (0.50)
        cfg_gate = BacktestConfig(
            initial_capital=200000.0,
            asset_type=AssetType.ETF,
            regime=TradingRegime.OVERNIGHT,
            fixed_dp_fee_inr=17.0,
            hold_on_strong_continuation=True,
            continuation_threshold=0.50,
        )
        engine_gate = EventDrivenBacktester(cfg_gate)
        res_gate = engine_gate.run(df, signals, ticker="BANKBEES.NS")

        # Verify continuation holds occurred and saved DP fees
        self.assertGreater(res_gate.metrics["continuation_holds_count"], 0)
        self.assertEqual(res_gate.metrics["continuation_holds_count"], 2)
        self.assertLess(res_gate.metrics["total_dp_fees_inr"], res_no_gate.metrics["total_dp_fees_inr"])
        self.assertEqual(res_gate.metrics["total_dp_fees_inr"], 17.0)  # Only 1 terminal exit DP fee
        self.assertLess(len(res_gate.trades), len(res_no_gate.trades))
        self.assertEqual(len(res_gate.trades), 2)  # 1 initial entry + 1 terminal exit

        # Net PnL (final equity) is higher with continuation gate active
        self.assertGreater(res_gate.metrics["final_equity"], res_no_gate.metrics["final_equity"])

