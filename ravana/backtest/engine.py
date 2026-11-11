"""Institutional Event-Driven Backtesting Engine with Causal Order Queue."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from ravana.backtest.config import AssetType, BacktestConfig, TradingRegime
from ravana.backtest.friction import IndianFrictionModel
from ravana.backtest.orders import Order, OrderSide, OrderStatus, Position, Trade
from ravana.backtest.slippage import MicrostructureSlippageModel


@dataclass
class BacktestResult:
    """Comprehensive performance metrics, equity series, and trade ledger."""
    equity_curve: pd.Series
    cash_curve: pd.Series
    drawdown_curve: pd.Series
    trades: List[Trade]
    metrics: Dict[str, float]
    config: BacktestConfig


class EventDrivenBacktester:
    """Executes causal bar-by-bar matching with accurate statutory fees and slippage."""

    def __init__(self, config: Optional[BacktestConfig] = None) -> None:
        self.config = config or BacktestConfig()
        self.slippage_model = MicrostructureSlippageModel(
            max_volume_participation=self.config.max_volume_participation,
            impact_gamma=self.config.impact_gamma,
        )

    def run(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        ticker: str = "ASSET",
        signal_threshold: float = 0.0,
    ) -> BacktestResult:
        """Execute simulation across chronological bar series."""
        df = self._validate_and_prep_df(df)
        signals = signals.reindex(df.index).fillna(0.0)

        cash = self.config.initial_capital
        position = Position(ticker)
        trades: List[Trade] = []
        equity_records: List[float] = []
        cash_records: List[float] = []

        pending_orders: List[Order] = []

        for i in range(len(df)):
            current_bar = df.iloc[i]
            ts: pd.Timestamp = df.index[i]

            # 1. Process Pending Orders causally at current bar Open
            cash, filled_trades = self._process_pending_orders(
                pending_orders, current_bar, ts, ticker, cash, position
            )
            trades.extend(filled_trades)
            pending_orders.clear()

            # 2. Mark-to-Market Equity
            close_px = float(current_bar["Close"])
            current_equity = cash + (position.shares * close_px)
            equity_records.append(current_equity)
            cash_records.append(cash)

            # 3. Alpha Strategy Decision Logic
            sig_val = float(signals.iloc[i])
            new_order = self._evaluate_signals(
                ts, current_bar, sig_val, position, cash, ticker, signal_threshold
            )
            if new_order:
                pending_orders.append(new_order)

        # Force liquidating any remaining position at end of backtest
        if position.shares > 0:
            final_bar = df.iloc[-1]
            cash, final_trade = self._force_close_position(final_bar, df.index[-1], ticker, cash, position)
            if final_trade:
                trades.append(final_trade)
            equity_records[-1] = cash

        equity_s = pd.Series(equity_records, index=df.index, name="Equity")
        cash_s = pd.Series(cash_records, index=df.index, name="Cash")
        dd_s = self._compute_drawdown_series(equity_s)
        metrics = self._compute_performance_metrics(equity_s, trades)

        return BacktestResult(
            equity_curve=equity_s,
            cash_curve=cash_s,
            drawdown_curve=dd_s,
            trades=trades,
            metrics=metrics,
            config=self.config,
        )

    def _process_pending_orders(
        self,
        orders: List[Order],
        bar: pd.Series,
        ts: pd.Timestamp,
        ticker: str,
        cash: float,
        position: Position,
    ) -> Tuple[float, List[Trade]]:
        """Fill orders against bar Open price with volume cap and slippage."""
        filled: List[Trade] = []
        open_px = float(bar["Open"])
        vol = float(bar.get("Volume", 1e6))
        hi = float(bar.get("High", open_px))
        lo = float(bar.get("Low", open_px))

        for ord in orders:
            side_str = "BUY" if ord.side == OrderSide.BUY else "SELL"
            fill = self.slippage_model.simulate_fill(
                side=side_str,
                desired_shares=ord.shares,
                nominal_price=open_px,
                bar_volume=vol,
                bar_high=hi,
                bar_low=lo,
            )
            if fill.filled_shares <= 0 or fill.is_circuit_locked:
                continue

            friction = IndianFrictionModel.calculate_friction(
                side=side_str,
                price=fill.executed_price,
                shares=fill.filled_shares,
                asset_type=self.config.asset_type,
                regime=self.config.regime,
                fixed_dp_fee_inr=self.config.fixed_dp_fee_inr,
                brokerage_per_order_inr=self.config.brokerage_per_order_inr,
            )

            trade = Trade(
                trade_id=f"T_{len(filled)}_{int(ts.timestamp())}",
                timestamp=ts,
                ticker=ticker,
                side=ord.side,
                shares=fill.filled_shares,
                price=fill.executed_price,
                slippage_bps=fill.slippage_bps,
                friction=friction,
            )
            position.update_on_trade(trade)

            if ord.side == OrderSide.BUY:
                cash -= (fill.filled_shares * fill.executed_price + friction.total_friction_inr)
            else:
                cash += (fill.filled_shares * fill.executed_price - friction.total_friction_inr)

            filled.append(trade)

        return cash, filled

    def _evaluate_signals(
        self,
        ts: pd.Timestamp,
        bar: pd.Series,
        signal: float,
        position: Position,
        cash: float,
        ticker: str,
        threshold: float,
    ) -> Optional[Order]:
        """Apply regime rules (overnight vs intraday vs swing) to emit order."""
        t_str = ts.strftime("%H:%M")
        close_px = float(bar["Close"])

        if self.config.regime == TradingRegime.OVERNIGHT:
            # Entry condition: Near market close (15:20-15:25) and positive signal
            if t_str >= self.config.entry_cutoff_time and signal > threshold and position.shares == 0:
                shares = self._size_position(cash, close_px)
                if shares > 0:
                    return Order(f"ORD_BUY_{ts.timestamp()}", ts, ticker, OrderSide.BUY, shares)
            # Exit condition: Morning open (09:15-09:20) and currently holding
            elif t_str <= self.config.exit_cutoff_time and position.shares > 0:
                return Order(f"ORD_SELL_{ts.timestamp()}", ts, ticker, OrderSide.SELL, position.shares)

        elif self.config.regime == TradingRegime.INTRADAY_MIS:
            # Force square-off before session end (15:15 IST)
            if t_str >= "15:15" and position.shares != 0:
                side = OrderSide.SELL if position.shares > 0 else OrderSide.BUY
                return Order(f"ORD_SQ_{ts.timestamp()}", ts, ticker, side, abs(position.shares))
            elif signal > threshold and position.shares == 0:
                shares = self._size_position(cash, close_px)
                if shares > 0:
                    return Order(f"ORD_BUY_{ts.timestamp()}", ts, ticker, OrderSide.BUY, shares)

        elif self.config.regime == TradingRegime.SWING:
            if signal > threshold and position.shares == 0:
                shares = self._size_position(cash, close_px)
                if shares > 0:
                    return Order(f"ORD_BUY_{ts.timestamp()}", ts, ticker, OrderSide.BUY, shares)
            elif signal < -threshold and position.shares > 0:
                return Order(f"ORD_SELL_{ts.timestamp()}", ts, ticker, OrderSide.SELL, position.shares)

        return None

    def _size_position(self, cash: float, price: float) -> int:
        """Institutional capital allocation with friction reserve buffer."""
        usable_cash = max(0.0, cash * 0.98)  # Reserve 2% for statutory friction & buffer
        return int(usable_cash // price) if price > 0 else 0

    def _force_close_position(
        self, bar: pd.Series, ts: pd.Timestamp, ticker: str, cash: float, position: Position
    ) -> Tuple[float, Optional[Trade]]:
        """Liquidate remaining inventory on backtest completion."""
        close_px = float(bar["Close"])
        friction = IndianFrictionModel.calculate_friction(
            side="SELL", price=close_px, shares=position.shares,
            asset_type=self.config.asset_type, regime=self.config.regime,
            fixed_dp_fee_inr=self.config.fixed_dp_fee_inr,
        )
        trade = Trade(
            trade_id=f"T_FINAL_{int(ts.timestamp())}", timestamp=ts, ticker=ticker,
            side=OrderSide.SELL, shares=position.shares, price=close_px,
            slippage_bps=0.0, friction=friction
        )
        cash += (position.shares * close_px - friction.total_friction_inr)
        position.update_on_trade(trade)
        return cash, trade

    @staticmethod
    def _validate_and_prep_df(df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Input DataFrame must have DatetimeIndex.")
        if "Close" not in df.columns or "Open" not in df.columns:
            raise ValueError("DataFrame must contain Open and Close price columns.")
        return df.sort_index()

    @staticmethod
    def _compute_drawdown_series(equity: pd.Series) -> pd.Series:
        running_max = equity.cummax()
        return (equity - running_max) / running_max

    @staticmethod
    def _compute_performance_metrics(equity: pd.Series, trades: List[Trade]) -> Dict[str, float]:
        if len(equity) < 2:
            return {}

        ret_series = equity.pct_change().dropna()
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1.0
        n_periods = len(ret_series)

        # Annualized metrics (assuming 5m intraday grid, 75 bars/day, 252 days/yr)
        bars_per_yr = 252.0 * 75.0
        cagr = ((1.0 + total_return) ** (bars_per_yr / max(1.0, float(n_periods)))) - 1.0 if total_return > -1.0 else -1.0
        ann_vol = float(ret_series.std()) * math.sqrt(bars_per_yr) if ret_series.std() > 0 else 0.0
        sharpe = (float(ret_series.mean()) * bars_per_yr) / ann_vol if ann_vol > 0 else 0.0

        # Downside vol & Sortino
        downside = ret_series[ret_series < 0]
        downside_std = float(downside.std()) * math.sqrt(bars_per_yr) if len(downside) > 1 else 1e-6
        sortino = (float(ret_series.mean()) * bars_per_yr) / downside_std if downside_std > 0 else 0.0

        # Drawdowns
        running_max = equity.cummax()
        dd = (equity - running_max) / running_max
        max_dd = float(abs(dd.min()))

        # Friction totals
        total_fees = sum(t.friction.total_friction_inr for t in trades)
        total_dp = sum(t.friction.dp_charges_inr for t in trades)

        # Trade stats
        sells = [t for t in trades if t.side == OrderSide.SELL]
        win_count = sum(1 for i in range(1, len(trades), 2) if trades[i].price > trades[i-1].price)
        win_rate = (win_count / len(sells)) if sells else 0.0

        return {
            "initial_capital": round(float(equity.iloc[0]), 2),
            "final_equity": round(float(equity.iloc[-1]), 2),
            "total_return_pct": round(total_return * 100.0, 2),
            "cagr_pct": round(cagr * 100.0, 2),
            "annualized_vol_pct": round(ann_vol * 100.0, 2),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "max_drawdown_pct": round(max_dd * 100.0, 2),
            "total_trades": len(trades),
            "total_fees_inr": round(total_fees, 2),
            "total_dp_fees_inr": round(total_dp, 2),
            "win_rate_pct": round(win_rate * 100.0, 2),
        }
