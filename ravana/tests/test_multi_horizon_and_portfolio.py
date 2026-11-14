"""ADHD Red Team Exhaustive Unit & Stress Test Suite for Phase A & Phase B Upgrades.

Validates:
1. Multi-Horizon Target Builder (OVERNIGHT, INTRADAY_N_BARS, MULTI_DAY_SWING) in Muscle 8.
2. Causal forward return verification (no lookahead, same-session boundaries, 14:45 cutoff).
3. Indian Statutory Intraday Friction Model (ETF STT=0.0, DP=0.0, drag < 1.0 bps).
4. Intraday MIS Signal Reversal in Muscle 11 (causal exit on signal flip before 15:15).
5. Cross-Sectional operators CS_RANK and CS_ZSCORE in Muscle 1 & 2 (3D ranking/standardization, 2D/1D/B=1 fallbacks).
6. 10-ETF Indian Sovereign Universe in asset_catalog.yaml.
"""
from __future__ import annotations

import unittest
import numpy as np
import pandas as pd
import torch

from ravana.targets.config import TargetConfig, TargetHorizon
from ravana.targets.costs import DeliveryFrictionModel, IntradayFrictionModel
from ravana.targets.builder import OvernightTargetBuilder, MultiHorizonTargetBuilder
from ravana.backtest.config import BacktestConfig, TradingRegime
from ravana.backtest.engine import EventDrivenBacktester
from ravana.backtest.orders import OrderSide, Position
from ravana.compute.primitives import cs_rank, cs_zscore
from ravana.compute.tensor_evaluator import TensorASTEvaluator
from ravana.domain.formula_ast import ASTNode, ASTEvaluator, OPERATORS_CROSS_SECTIONAL
from ravana.domain.formula_parser import FormulaParser
from ravana.data.universe.models import AssetClass, ExchangeType
from ravana.data.universe.registry import AssetRegistry


class TestMultiHorizonTargetBuilder(unittest.TestCase):
    """Stress tests for Muscle 8 Multi-Horizon Target Builder contracts."""

    def setUp(self) -> None:
        np.random.seed(42)
        # Construct synthetic multi-session 5m intraday data across 5 trading days
        days = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"]
        bar_times = pd.date_range("09:15", "15:25", freq="5min").strftime("%H:%M")
        
        all_dts = []
        for d in days:
            for t in bar_times:
                all_dts.append(pd.Timestamp(f"{d} {t}"))
        
        n = len(all_dts)
        # Generate random walk with drift for prices
        ret = np.random.normal(0.0001, 0.002, n)
        price = 500.0 * np.cumprod(1.0 + ret)
        
        self.df = pd.DataFrame(
            {
                "Open": price * (1.0 - np.random.uniform(0.0, 0.0005, n)),
                "High": price * (1.0 + np.random.uniform(0.0005, 0.0015, n)),
                "Low": price * (1.0 - np.random.uniform(0.0005, 0.0015, n)),
                "Close": price,
                "Volume": np.random.uniform(10000.0, 100000.0, n),
            },
            index=pd.DatetimeIndex(all_dts),
        )

    def test_target_horizon_enum_and_defaults(self) -> None:
        """Verify TargetHorizon enum values and backward-compatible TargetConfig defaults."""
        self.assertEqual(TargetHorizon.OVERNIGHT, "OVERNIGHT")
        self.assertEqual(TargetHorizon.INTRADAY_N_BARS, "INTRADAY_N_BARS")
        self.assertEqual(TargetHorizon.MULTI_DAY_SWING, "MULTI_DAY_SWING")
        self.assertIs(MultiHorizonTargetBuilder, OvernightTargetBuilder)

        cfg = TargetConfig()
        self.assertEqual(cfg.horizon, TargetHorizon.OVERNIGHT)
        self.assertEqual(cfg.forward_bars, 6)
        self.assertEqual(cfg.intraday_cutoff_time, "14:45")
        self.assertEqual(cfg.holding_days, 3)
        self.assertEqual(cfg.capital_inr, 200000.0)
        self.assertEqual(cfg.min_capital_inr, 200000.0)

    def test_overnight_target_contract(self) -> None:
        """Verify OVERNIGHT target extracts session close to next session open returns."""
        cfg = TargetConfig(horizon=TargetHorizon.OVERNIGHT, hurdle_bps=5.0)
        builder = MultiHorizonTargetBuilder(cfg)
        targets = builder.build_targets(self.df)

        self.assertEqual(len(targets), len(self.df))
        entry_indices = np.where(targets["is_entry_bar"] == 1)[0]
        # In 5 trading days, there are exactly 4 overnight transitions
        self.assertEqual(len(entry_indices), 4)

        for idx in entry_indices:
            # Entry bar must be at end of day
            self.assertEqual(self.df.index[idx].strftime("%H:%M"), "15:25")
            self.assertGreaterEqual(targets["calendar_hold_days"].iloc[idx], 1)
            self.assertEqual(targets["sample_weight"].iloc[idx], 1.0)
            # Net return must be strictly less than gross return due to statutory delivery friction
            self.assertLess(targets["net_return"].iloc[idx], targets["gross_return"].iloc[idx])

    def test_intraday_n_bars_causal_forward_returns(self) -> None:
        """Prove that INTRADAY_N_BARS computes exact forward open-to-open returns without lookahead."""
        k = 6
        cfg = TargetConfig(
            horizon=TargetHorizon.INTRADAY_N_BARS,
            forward_bars=k,
            intraday_cutoff_time="14:45",
            hurdle_bps=5.0,
            capital_inr=200000.0,
        )
        builder = MultiHorizonTargetBuilder(cfg)
        targets = builder.build_targets(self.df)

        session_dates = self.df.index.normalize()
        friction_model = IntradayFrictionModel(
            variable_bps=cfg.intraday_variable_bps,
            slippage_bps=cfg.slippage_bps,
            default_capital_inr=cfg.capital_inr,
            min_capital_inr=cfg.min_capital_inr,
        )

        n = len(self.df)
        for i in range(n):
            t_str = self.df.index[i].strftime("%H:%M")
            is_entry = targets["is_entry_bar"].iloc[i]

            if t_str > "14:45":
                # Crucial audit invariant: bars after 14:45 must never permit entry
                self.assertEqual(is_entry, 0, f"Bar at {self.df.index[i]} > 14:45 was marked as entry bar!")
                self.assertEqual(targets["gross_return"].iloc[i], 0.0)
                self.assertEqual(targets["net_return"].iloc[i], 0.0)
                self.assertEqual(targets["sample_weight"].iloc[i], 0.0)
            elif is_entry == 1:
                # Must be on or before 14:45
                self.assertLessEqual(t_str, "14:45")
                # Both entry bar (i+1) and exit bar (i+1+k) must belong to the exact same trading date!
                self.assertEqual(
                    session_dates[i + 1],
                    session_dates[i + 1 + k],
                    f"Intraday trade at bar {i} crossed session boundary into {session_dates[i+1+k]}!",
                )

                # Exact causal mathematical verification:
                # Return must be (df.Open[i + 1 + k] - df.Open[i + 1]) / df.Open[i + 1]
                expected_entry_px = float(self.df["Open"].iloc[i + 1])
                expected_exit_px = float(self.df["Open"].iloc[i + 1 + k])
                expected_gross = (expected_exit_px - expected_entry_px) / expected_entry_px
                expected_net = friction_model.apply_friction(expected_gross, cfg.capital_inr)

                actual_gross = float(targets["gross_return"].iloc[i])
                actual_net = float(targets["net_return"].iloc[i])

                self.assertAlmostEqual(actual_gross, expected_gross, places=7)
                self.assertAlmostEqual(actual_net, expected_net, places=7)
                self.assertEqual(targets["calendar_hold_days"].iloc[i], 0)
                self.assertEqual(targets["sample_weight"].iloc[i], 1.0)
                expected_label = 1 if actual_net >= (cfg.hurdle_bps / 10000.0) else 0
                self.assertEqual(targets["binary_label"].iloc[i], expected_label)

    def test_multi_day_swing_target_contract(self) -> None:
        """Verify MULTI_DAY_SWING computes forward returns across session boundaries."""
        cfg = TargetConfig(horizon=TargetHorizon.MULTI_DAY_SWING, holding_days=3)
        builder = MultiHorizonTargetBuilder(cfg)
        targets = builder.build_targets(self.df)

        entry_indices = np.where(targets["is_entry_bar"] == 1)[0]
        # In 5 sessions with 3-day holding, there are exactly 2 entries (day 0 -> day 3, day 1 -> day 4)
        self.assertEqual(len(entry_indices), 2)
        for idx in entry_indices:
            self.assertGreaterEqual(targets["calendar_hold_days"].iloc[idx], 3)
            self.assertEqual(targets["sample_weight"].iloc[idx], 1.0)


class TestIntradayFrictionModel(unittest.TestCase):
    """Stress tests for statutory Indian intraday transaction cost attribution."""

    def test_etf_intraday_friction_under_1_bps(self) -> None:
        """Verify ETF intraday has STT=0.0, DP fee=0.0, and total friction is under 1.0 bps."""
        model = IntradayFrictionModel(variable_bps=0.9, slippage_bps=0.0, is_etf=True)

        # Statutory checks
        self.assertEqual(model.flat_dp_charge_inr, 0.0)
        self.assertTrue(model.is_etf)

        total_drag = model.compute_total_drag_bps(capital_inr=200000.0)
        # Must be exactly 0.9 bps, which is strictly less than 1.0 bps
        self.assertEqual(total_drag, 0.9)
        self.assertLess(total_drag, 1.0)

        # Apply friction to 1% gross return
        gross_ret = 0.0100  # 100 bps
        net_ret = model.apply_friction(gross_ret, capital_inr=200000.0)
        # Fractional drag = 0.9 / 10000 = 0.00009
        self.assertAlmostEqual(net_ret, 0.0100 - 0.00009, places=8)

    def test_equity_intraday_friction_with_sell_stt(self) -> None:
        """Verify non-ETF equity intraday includes 0.025% (2.5 bps) statutory sell STT."""
        model = IntradayFrictionModel(variable_bps=0.9, slippage_bps=0.0, is_etf=False)
        total_drag = model.compute_total_drag_bps(capital_inr=200000.0)
        # 0.9 bps variable + 2.5 bps STT = 3.4 bps
        self.assertEqual(total_drag, 3.4)

    def test_capital_viability_gate(self) -> None:
        """Verify institutional minimum capital gate at Rs. 2,00,000."""
        model = IntradayFrictionModel()
        self.assertTrue(model.is_capital_viable(200000.0))
        self.assertTrue(model.is_capital_viable(500000.0))
        self.assertFalse(model.is_capital_viable(199999.0))
        self.assertFalse(model.is_capital_viable(50000.0))


class TestIntradayMISSignalReversal(unittest.TestCase):
    """Stress tests for causal hysteresis signal reversal exits in EventDrivenBacktester."""

    def setUp(self) -> None:
        # Build 1 single day of 5m bars: 09:15 to 15:25 (75 bars)
        bar_times = pd.date_range("09:15", "15:25", freq="5min").strftime("%H:%M")
        dts = [pd.Timestamp(f"2026-01-05 {t}") for t in bar_times]
        n = len(dts)

        self.df = pd.DataFrame(
            {
                "Open": np.full(n, 500.0),
                "High": np.full(n, 502.0),
                "Low": np.full(n, 498.0),
                "Close": np.full(n, 500.0),
                "Volume": np.full(n, 50000.0),
            },
            index=pd.DatetimeIndex(dts),
        )

    def test_causal_signal_reversal_exit_before_squareoff(self) -> None:
        """Prove intraday MIS position exits on signal reversal (e.g. at 10:30) rather than holding till 15:15."""
        signals = pd.Series(0.0, index=self.df.index)

        # Trigger BUY at 09:20 bar close -> executes at 09:25 bar Open
        idx_0920 = self.df.index.get_loc("2026-01-05 09:20:00")
        signals.iloc[idx_0920] = 1.0

        # Maintain positive signal from 09:25 to 10:25
        idx_0925 = self.df.index.get_loc("2026-01-05 09:25:00")
        idx_1025 = self.df.index.get_loc("2026-01-05 10:25:00")
        signals.iloc[idx_0925:idx_1025 + 1] = 1.0

        # Signal reverses at 10:30 bar close -> becomes -1.0
        idx_1030 = self.df.index.get_loc("2026-01-05 10:30:00")
        signals.iloc[idx_1030:] = -1.0

        cfg = BacktestConfig(
            initial_capital=200000.0,
            regime=TradingRegime.INTRADAY_MIS,
            signal_exit_threshold=0.0,
            min_hold_bars=2,
        )
        engine = EventDrivenBacktester(cfg)
        result = engine.run(self.df, signals, ticker="BANKBEES.NS", signal_threshold=0.0)

        # Must have exactly 2 trades: 1 BUY and 1 causal reversal SELL
        self.assertEqual(len(result.trades), 2, f"Expected exactly 2 trades, got {len(result.trades)}")

        trade_buy = result.trades[0]
        trade_sell = result.trades[1]

        self.assertEqual(trade_buy.side, OrderSide.BUY)
        self.assertEqual(trade_buy.timestamp, pd.Timestamp("2026-01-05 09:25:00"))

        # The exit MUST be filled at 10:35 Open (causal reaction to 10:30 close reversal)
        self.assertEqual(trade_sell.side, OrderSide.SELL)
        self.assertEqual(
            trade_sell.timestamp,
            pd.Timestamp("2026-01-05 10:35:00"),
            f"Position did not exit at 10:35 on signal reversal! Exited at {trade_sell.timestamp}",
        )

        # Crucial check: verify trade exited hours before the 15:15 mandatory square-off
        self.assertLess(trade_sell.timestamp.strftime("%H:%M"), "15:15")

        # Verify shares are flat after 10:35
        self.assertEqual(len(result.closed_round_trips), 1)
        rt = result.closed_round_trips[0]
        self.assertEqual(rt.entry_time, pd.Timestamp("2026-01-05 09:25:00"))
        self.assertEqual(rt.exit_time, pd.Timestamp("2026-01-05 10:35:00"))

    def test_hysteresis_min_hold_bars_blocks_premature_exit(self) -> None:
        """Verify min_hold_bars prevents exiting on the immediately adjacent bar."""
        signals = pd.Series(0.0, index=self.df.index)

        # Signal positive at 09:15 -> BUY fills at 09:20 Open (bars_held = 1 at 09:20 close)
        signals.iloc[0] = 1.0
        # Signal flips negative immediately at 09:20 close
        signals.iloc[1] = -1.0
        # Remains negative at 09:25 close
        signals.iloc[2] = -1.0

        cfg = BacktestConfig(
            initial_capital=200000.0,
            regime=TradingRegime.INTRADAY_MIS,
            signal_exit_threshold=0.0,
            min_hold_bars=2,  # Must hold for at least 2 bars!
        )
        engine = EventDrivenBacktester(cfg)
        result = engine.run(self.df, signals, ticker="BANKBEES.NS", signal_threshold=0.0)

        self.assertEqual(len(result.trades), 2)
        trade_buy = result.trades[0]
        trade_sell = result.trades[1]

        self.assertEqual(trade_buy.timestamp, pd.Timestamp("2026-01-05 09:20:00"))
        # At 09:20 close, bars_held was 1 < 2, so it COULD NOT exit at 09:25.
        # At 09:25 close, bars_held was 2 >= 2, so it exits at 09:30 Open!
        self.assertEqual(trade_sell.timestamp, pd.Timestamp("2026-01-05 09:30:00"))

    def test_squareoff_fallback_when_signal_persists(self) -> None:
        """Verify mandatory square-off at 15:15 when signal never reverses."""
        signals = pd.Series(1.0, index=self.df.index)  # Always bullish

        cfg = BacktestConfig(
            initial_capital=200000.0,
            regime=TradingRegime.INTRADAY_MIS,
        )
        engine = EventDrivenBacktester(cfg)
        result = engine.run(self.df, signals, ticker="BANKBEES.NS", signal_threshold=0.0)

        self.assertEqual(len(result.trades), 2)
        self.assertEqual(result.trades[0].side, OrderSide.BUY)
        self.assertEqual(result.trades[1].side, OrderSide.SELL)
        # Must exit at 15:15 or 15:20
        self.assertGreaterEqual(result.trades[1].timestamp.strftime("%H:%M"), "15:15")


class TestCrossSectionalOperators(unittest.TestCase):
    """Stress tests for CS_RANK and CS_ZSCORE tensor and AST operators."""

    def test_cs_rank_3d_tensor(self) -> None:
        """On 3D tensor [B=3, T=20, K=1], verify cs_rank ranks [10, 20, 30] to [0.0, 0.5, 1.0]."""
        B, T, K = 3, 20, 1
        # Asset 0: 10.0, Asset 1: 20.0, Asset 2: 30.0 across all timestamps
        x = torch.zeros(B, T, K, dtype=torch.float32)
        x[0, :, :] = 10.0
        x[1, :, :] = 20.0
        x[2, :, :] = 30.0

        ranked = cs_rank(x)
        self.assertEqual(ranked.shape, (B, T, K))

        for t in range(T):
            self.assertAlmostEqual(float(ranked[0, t, 0]), 0.0, places=5)
            self.assertAlmostEqual(float(ranked[1, t, 0]), 0.5, places=5)
            self.assertAlmostEqual(float(ranked[2, t, 0]), 1.0, places=5)

    def test_cs_zscore_3d_tensor(self) -> None:
        """On 3D tensor [B=3, T=20, K=1], verify cs_zscore standardizes mean=0, std=1 across assets."""
        B, T, K = 3, 20, 1
        x = torch.zeros(B, T, K, dtype=torch.float32)
        x[0, :, :] = 10.0
        x[1, :, :] = 20.0
        x[2, :, :] = 30.0

        z = cs_zscore(x)
        self.assertEqual(z.shape, (B, T, K))

        for t in range(T):
            slice_vals = z[:, t, 0]
            # Mean across assets must be exactly 0.0
            self.assertAlmostEqual(float(torch.mean(slice_vals)), 0.0, places=5)
            # Sample std across assets must be exactly 1.0
            self.assertAlmostEqual(float(torch.std(slice_vals)), 1.0, places=5)
            # Individual z-scores for [10, 20, 30] with mean=20, std=10 are [-1.0, 0.0, 1.0]
            self.assertAlmostEqual(float(slice_vals[0]), -1.0, places=5)
            self.assertAlmostEqual(float(slice_vals[1]), 0.0, places=5)
            self.assertAlmostEqual(float(slice_vals[2]), 1.0, places=5)

    def test_single_asset_fallback_without_exception(self) -> None:
        """Verify single asset 1D, 2D, or 3D [B=1] falls back to ts_rank_20 and ts_zscore_20."""
        # 1D [T=50]
        x_1d = torch.randn(50)
        r_1d = cs_rank(x_1d)
        z_1d = cs_zscore(x_1d)
        self.assertTrue(torch.all(torch.isfinite(r_1d)))
        self.assertTrue(torch.all(r_1d >= 0.0) and torch.all(r_1d <= 1.0))
        self.assertTrue(torch.all(torch.isfinite(z_1d)))
        self.assertTrue(torch.all(z_1d >= -4.0) and torch.all(z_1d <= 4.0))

        # 2D [T=50, K=5]
        x_2d = torch.randn(50, 5)
        r_2d = cs_rank(x_2d)
        z_2d = cs_zscore(x_2d)
        self.assertTrue(torch.all(torch.isfinite(r_2d)))
        self.assertTrue(torch.all(r_2d >= 0.0) and torch.all(r_2d <= 1.0))
        self.assertTrue(torch.all(torch.isfinite(z_2d)))
        self.assertTrue(torch.all(z_2d >= -4.0) and torch.all(z_2d <= 4.0))

        # 3D [B=1, T=50, K=5]
        x_3d_single = torch.randn(1, 50, 5)
        r_3d = cs_rank(x_3d_single)
        z_3d = cs_zscore(x_3d_single)
        self.assertTrue(torch.all(torch.isfinite(r_3d)))
        self.assertTrue(torch.all(r_3d >= 0.0) and torch.all(r_3d <= 1.0))
        self.assertTrue(torch.all(torch.isfinite(z_3d)))
        self.assertTrue(torch.all(z_3d >= -4.0) and torch.all(z_3d <= 4.0))

    def test_nan_and_extreme_inputs_safety(self) -> None:
        """Verify robust clamping and NaN neutralization in cross-sectional primitives."""
        x_bad = torch.tensor([
            [[float("nan")], [float("inf")], [1e15]],
            [[0.0], [0.0], [0.0]],
            [[float("-inf")], [float("nan")], [-1e15]],
        ], dtype=torch.float32)

        ranked = cs_rank(x_bad)
        self.assertTrue(torch.all(torch.isfinite(ranked)))
        self.assertTrue(torch.all((ranked >= 0.0) & (ranked <= 1.0)))

        z = cs_zscore(x_bad)
        self.assertTrue(torch.all(torch.isfinite(z)))
        self.assertTrue(torch.all((z >= -3.0) & (z <= 3.0)))

    def test_ast_evaluator_and_parser_cross_sectional(self) -> None:
        """Verify AST parsing, simplification, and evaluation of CS_RANK and CS_ZSCORE."""
        self.assertIn("CS_RANK", OPERATORS_CROSS_SECTIONAL)
        self.assertIn("CS_ZSCORE", OPERATORS_CROSS_SECTIONAL)

        node_rank = FormulaParser.parse("CS_RANK(Close)")
        node_zscore = FormulaParser.parse("CS_ZSCORE(Volume)")

        self.assertEqual(node_rank.op, "CS_RANK")
        self.assertEqual(node_zscore.op, "CS_ZSCORE")

        # Single-asset DataFrame fallback
        df_single = pd.DataFrame({"Close": [10.0, 20.0, 30.0], "Volume": [100.0, 200.0, 300.0]})
        res_rank = ASTEvaluator.eval_node(node_rank, df_single)
        res_zscore = ASTEvaluator.eval_node(node_zscore, df_single)

        self.assertTrue(np.all(res_rank == 0.5))
        self.assertTrue(np.all(res_zscore == 0.0))

        # TensorASTEvaluator on 3D tensor
        B, T, F = 3, 10, 2
        X_3d = torch.zeros(B, T, F, dtype=torch.float32)
        X_3d[0, :, 0] = 100.0  # Close
        X_3d[1, :, 0] = 200.0
        X_3d[2, :, 0] = 300.0
        f_map = {"Close": 0, "Volume": 1}

        t_res_rank = TensorASTEvaluator.eval_node(node_rank, X_3d, f_map)
        self.assertEqual(t_res_rank.shape, (B, T))
        self.assertTrue(torch.allclose(t_res_rank[0, :], torch.tensor(0.0)))
        self.assertTrue(torch.allclose(t_res_rank[1, :], torch.tensor(0.5)))
        self.assertTrue(torch.allclose(t_res_rank[2, :], torch.tensor(1.0)))


class TestAssetCatalogUniverse(unittest.TestCase):
    """Stress tests for the 10-ETF Sovereign Universe in asset_catalog.yaml."""

    def setUp(self) -> None:
        self.registry = AssetRegistry()

    def test_10_etf_universe_completeness_and_attributes(self) -> None:
        """Verify all 10 liquid Indian ETFs exist with lot_size=1, currency=INR, asset_class=ETF."""
        required_etfs = [
            "BANKBEES.NS",
            "NIFTYBEES.NS",
            "ITBEES.NS",
            "GOLDBEES.NS",
            "SILVERBEES.NS",
            "PHARMABEES.NS",
            "AUTOBEES.NS",
            "JUNIORBEES.NS",
            "PSUBNKBEES.NS",
            "CPSEETF.NS",
        ]

        for sym in required_etfs:
            inst = self.registry.get(sym)
            self.assertIsNotNone(inst, f"ETF {sym} was missing from AssetRegistry!")
            self.assertEqual(inst.asset_class, AssetClass.ETF, f"{sym} asset_class != ETF")
            self.assertEqual(inst.exchange, ExchangeType.NSE, f"{sym} exchange != NSE")
            self.assertEqual(inst.currency, "INR", f"{sym} currency != INR")
            self.assertEqual(inst.lot_size, 1, f"{sym} lot_size != 1")
            self.assertEqual(inst.tick_size, 0.01, f"{sym} tick_size != 0.01")
            self.assertTrue(inst.is_active, f"{sym} is not active")
            self.assertIsNotNone(inst.security_id, f"{sym} has no ISIN")

    def test_all_etfs_basket_resolution(self) -> None:
        """Verify 'all_etfs' basket contains exactly the 10 liquid ETFs."""
        all_etfs = self.registry.get_basket("all_etfs")
        self.assertEqual(len(all_etfs), 10)
        symbols = [i.symbol for i in all_etfs]
        self.assertIn("BANKBEES.NS", symbols)
        self.assertIn("NIFTYBEES.NS", symbols)
        self.assertIn("CPSEETF.NS", symbols)
        self.assertIn("SILVERBEES.NS", symbols)


if __name__ == "__main__":
    unittest.main()
