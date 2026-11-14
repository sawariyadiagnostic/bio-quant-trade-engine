"""Institutional Event-Driven Backtesting Engine with Causal Order Queue."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from ravana.backtest.config import AssetType, BacktestConfig, TradingRegime
from ravana.backtest.friction import IndianFrictionModel
from ravana.backtest.orders import Order, OrderSide, OrderStatus, Position, RoundTripTrade, Trade
from ravana.backtest.slippage import MicrostructureSlippageModel


@dataclass(frozen=True)
class BacktestResult:
    """Rigorous audit-proof backtest run output."""
    equity_curve: pd.Series
    cash_curve: pd.Series
    drawdown_curve: pd.Series
    trades: List[Trade]
    metrics: Dict[str, float]
    config: BacktestConfig
    closed_round_trips: List[RoundTripTrade] = field(default_factory=list)


class EventDrivenBacktester:
    """Causal, bar-by-bar backtest engine with exact Indian statutory frictions."""

    def __init__(self, config: Optional[BacktestConfig] = None) -> None:
        self.config = config or BacktestConfig()
        self.slippage_model = MicrostructureSlippageModel(
            max_volume_participation=self.config.max_volume_participation,
            default_half_spread_bps=self.config.default_half_spread_bps,
        )

    def run(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        ticker: str = "BANKBEES.NS",
        signal_threshold: float = 0.0,
    ) -> BacktestResult:
        """Execute causal event-driven backtest across given market bar DataFrame."""
        df = self._validate_and_prep_df(df)
        cash = float(self.config.initial_capital_inr)
        position = Position(ticker, allow_shorting=self.config.allow_shorting)

        trades: List[Trade] = []
        equity_records: List[float] = []
        cash_records: List[float] = []
        debited_dates: Set[Tuple[str, str]] = set()
        pending_orders: List[Order] = []

        for i in range(len(df)):
            current_bar = df.iloc[i]
            ts: pd.Timestamp = df.index[i]

            # Auto-schedule 09:15 opening auction exit when carrying an overnight position into a new day
            if (
                i > 0
                and self.config.regime == TradingRegime.OVERNIGHT
                and ts.date() > df.index[i - 1].date()
                and position.shares > 0
                and not any(o.side == OrderSide.SELL for o in pending_orders)
            ):
                pending_orders.insert(
                    0, Order(f"ORD_OVN_EXIT_{ts.timestamp()}", ts, ticker, OrderSide.SELL, position.shares)
                )

            # 1. Process Pending Orders causally at current bar Open
            cash, filled_trades, pending_orders = self._process_pending_orders(
                pending_orders, current_bar, ts, ticker, cash, position, debited_dates
            )
            trades.extend(filled_trades)

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
                # If regime is OVERNIGHT and this is the final bar of the trading session,
                # execute as MOC (Market-on-Close) fill at bar Close so position is held overnight!
                is_last_bar_of_day = (i == len(df) - 1) or (df.index[i + 1].date() > ts.date())
                if self.config.regime == TradingRegime.OVERNIGHT and is_last_bar_of_day and new_order.side == OrderSide.BUY:
                    moc_cash, moc_trades, _ = self._process_pending_orders(
                        [new_order], current_bar, ts, ticker, cash, position, debited_dates, fill_price=close_px
                    )
                    cash = moc_cash
                    trades.extend(moc_trades)
                    equity_records[-1] = cash + (position.shares * close_px)
                    cash_records[-1] = cash
                else:
                    pending_orders.append(new_order)

        # Force liquidating remaining inventory on backtest completion
        if position.shares > 0:
            final_bar = df.iloc[-1]
            cash, final_trade = self._force_close_position(final_bar, df.index[-1], ticker, cash, position)
            if final_trade:
                trades.append(final_trade)
            equity_records[-1] = cash
            cash_records[-1] = cash  # Synchronize cash ledger

        equity_s = pd.Series(equity_records, index=df.index, name="Equity")
        cash_s = pd.Series(cash_records, index=df.index, name="Cash")
        dd_s = self._compute_drawdown_series(equity_s)
        metrics = self._compute_performance_metrics(
            equity_s, trades, position, risk_free_rate=self.config.risk_free_rate
        )

        return BacktestResult(
            equity_curve=equity_s,
            cash_curve=cash_s,
            drawdown_curve=dd_s,
            trades=trades,
            metrics=metrics,
            config=self.config,
            closed_round_trips=list(position.closed_trades),
        )

    def _process_pending_orders(
        self,
        orders: List[Order],
        bar: pd.Series,
        ts: pd.Timestamp,
        ticker: str,
        cash: float,
        position: Position,
        debited_dates: Set[Tuple[str, str]],
        fill_price: Optional[float] = None,
    ) -> Tuple[float, List[Trade], List[Order]]:
        """Fill orders against bar Open price (or explicit fill_price) with volume cap, slippage, and pre-trade margin guard."""
        filled: List[Trade] = []
        remaining_orders: List[Order] = []
        exec_price = float(fill_price) if fill_price is not None else float(bar["Open"])
        vol = float(bar.get("Volume", 0.0))
        hi = float(bar.get("High", exec_price))
        lo = float(bar.get("Low", exec_price))
        date_key = (ticker, ts.strftime("%Y-%m-%d"))

        for ord in orders:
            if ord.side == OrderSide.SELL and position.shares <= 0 and not self.config.allow_shorting:
                ord.status = OrderStatus.REJECTED
                continue

            side_str = "BUY" if ord.side == OrderSide.BUY else "SELL"
            fill = self.slippage_model.simulate_fill(
                side=side_str,
                desired_shares=ord.shares,
                nominal_price=exec_price,
                bar_volume=vol,
                bar_high=hi,
                bar_low=lo,
            )
            if fill.filled_shares <= 0 or fill.is_circuit_locked:
                ord.status = OrderStatus.PENDING
                remaining_orders.append(ord)
                continue

            # CDSL DP charges applied at most once per symbol per settlement date on delivery sell
            is_dp_eligible = False
            if ord.side == OrderSide.SELL and self.config.regime in (TradingRegime.OVERNIGHT, TradingRegime.SWING):
                if date_key not in debited_dates:
                    is_dp_eligible = True

            fill_shares = fill.filled_shares
            friction = IndianFrictionModel.calculate_friction(
                side=side_str,
                price=fill.executed_price,
                shares=fill_shares,
                asset_type=self.config.asset_type,
                regime=self.config.regime,
                fixed_dp_fee_inr=self.config.fixed_dp_fee_inr,
                brokerage_per_order_inr=self.config.brokerage_per_order_inr,
                is_dp_eligible=is_dp_eligible,
            )

            # Pre-trade cash check for BUY
            if ord.side == OrderSide.BUY:
                required_cost = (fill_shares * fill.executed_price) + friction.total_friction_inr
                if required_cost > cash:
                    max_affordable = int(max(0.0, cash - friction.total_friction_inr) // fill.executed_price)
                    if max_affordable <= 0:
                        ord.status = OrderStatus.REJECTED
                        continue
                    fill_shares = max_affordable
                    friction = IndianFrictionModel.calculate_friction(
                        side=side_str,
                        price=fill.executed_price,
                        shares=fill_shares,
                        asset_type=self.config.asset_type,
                        regime=self.config.regime,
                        fixed_dp_fee_inr=self.config.fixed_dp_fee_inr,
                        brokerage_per_order_inr=self.config.brokerage_per_order_inr,
                        is_dp_eligible=False,
                    )

            trade = Trade(
                trade_id=f"T_{len(filled)}_{int(ts.timestamp())}",
                timestamp=ts,
                ticker=ticker,
                side=ord.side,
                shares=fill_shares,
                price=fill.executed_price,
                slippage_bps=fill.slippage_bps,
                friction=friction,
            )
            position.update_on_trade(trade)

            if ord.side == OrderSide.BUY:
                cash -= (fill_shares * fill.executed_price + friction.total_friction_inr)
            else:
                cash += (fill_shares * fill.executed_price - friction.total_friction_inr)
                if is_dp_eligible:
                    debited_dates.add(date_key)

            filled.append(trade)

            if fill_shares < ord.shares:
                ord.shares -= fill_shares
                ord.status = OrderStatus.PARTIALLY_FILLED
                remaining_orders.append(ord)
            else:
                ord.status = OrderStatus.FILLED

        return cash, filled, remaining_orders

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
            if (t_str >= self.config.entry_cutoff_time or t_str == "00:00") and signal > threshold and position.shares == 0:
                shares = self._size_position(cash, close_px)
                if shares > 0:
                    return Order(f"ORD_BUY_{ts.timestamp()}", ts, ticker, OrderSide.BUY, shares)
            # Exit condition: Morning open (09:15-09:20) and currently holding
            elif (t_str <= self.config.exit_cutoff_time or t_str == "00:00") and position.shares > 0:
                return Order(f"ORD_SELL_{ts.timestamp()}", ts, ticker, OrderSide.SELL, position.shares)

        elif self.config.regime == TradingRegime.INTRADAY_MIS:
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
        usable_cash = max(0.0, cash * 0.98)
        return int(usable_cash // price) if price > 0 else 0

    def _force_close_position(
        self, bar: pd.Series, ts: pd.Timestamp, ticker: str, cash: float, position: Position
    ) -> Tuple[float, Optional[Trade]]:
        """Liquidate remaining inventory on backtest completion with slippage and frictions."""
        close_px = float(bar["Close"])
        vol = float(bar.get("Volume", 1e5))
        hi = float(bar.get("High", close_px))
        lo = float(bar.get("Low", close_px))

        fill = self.slippage_model.simulate_fill(
            side="SELL",
            desired_shares=position.shares,
            nominal_price=close_px,
            bar_volume=max(vol, float(position.shares) * 20.0),
            bar_high=hi,
            bar_low=lo,
        )
        exec_price = fill.executed_price if fill.executed_price > 0 else close_px
        friction = IndianFrictionModel.calculate_friction(
            side="SELL",
            price=exec_price,
            shares=position.shares,
            asset_type=self.config.asset_type,
            regime=self.config.regime,
            fixed_dp_fee_inr=self.config.fixed_dp_fee_inr,
            brokerage_per_order_inr=self.config.brokerage_per_order_inr,
            is_dp_eligible=True,
        )
        trade = Trade(
            trade_id=f"T_FINAL_{int(ts.timestamp())}",
            timestamp=ts,
            ticker=ticker,
            side=OrderSide.SELL,
            shares=position.shares,
            price=exec_price,
            slippage_bps=fill.slippage_bps,
            friction=friction,
        )
        cash += (position.shares * exec_price - friction.total_friction_inr)
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
    def _compute_performance_metrics(
        equity: pd.Series,
        trades: List[Trade],
        position: Optional[Position] = None,
        risk_free_rate: float = 0.065,
    ) -> Dict[str, float]:
        if len(equity) < 2:
            return {}

        ret_series = equity.pct_change().dropna()
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1.0
        n_periods = len(ret_series)

        delta_seconds = (equity.index[-1] - equity.index[0]).total_seconds()
        delta_days = max(1.0 / 24.0, delta_seconds / 86400.0)

        # Dynamically infer bars per trading day from index median spacing
        diffs = equity.index.to_series().diff().dropna()
        med_sec = float(diffs.median().total_seconds()) if not diffs.empty else 300.0
        if med_sec >= 43200.0:  # Daily or higher
            bars_per_yr = 252.0
        else:
            bars_per_day = max(1.0, round(22500.0 / max(60.0, med_sec)))  # 22,500s = 375 min NSE session
            bars_per_yr = 252.0 * bars_per_day
        if delta_days < 7.0:
            cagr = total_return
        else:
            ann_factor = 365.25 / delta_days
            cagr = ((1.0 + total_return) ** ann_factor) - 1.0 if total_return > -1.0 else -1.0

        ann_vol = float(ret_series.std()) * math.sqrt(bars_per_yr) if ret_series.std() > 0 else 0.0
        rf_rate = risk_free_rate
        excess_mean_ann = (float(ret_series.mean()) * bars_per_yr) - rf_rate
        sharpe = excess_mean_ann / ann_vol if ann_vol > 0 else 0.0

        neg_returns = np.minimum(0.0, ret_series.to_numpy())
        downside_semi_variance = float(np.mean(neg_returns ** 2)) if n_periods > 0 else 0.0
        downside_std = math.sqrt(downside_semi_variance) * math.sqrt(bars_per_yr) if downside_semi_variance > 0 else 0.0
        sortino = (
            excess_mean_ann / downside_std
            if downside_std > 1e-6
            else (sharpe if sharpe > 0 else 0.0)
        )

        running_max = equity.cummax()
        dd = (equity - running_max) / running_max
        max_dd = float(abs(dd.min()))

        total_fees = sum(t.friction.total_friction_inr for t in trades)
        total_dp = sum(t.friction.dp_charges_inr for t in trades)

        if position and position.closed_trades:
            rt_trades = position.closed_trades
            win_count = sum(1 for t in rt_trades if t.is_win)
            win_rate = (win_count / len(rt_trades)) * 100.0
            total_closed = len(rt_trades)
        else:
            sells = [t for t in trades if t.side == OrderSide.SELL]
            win_count = sum(1 for i in range(1, len(trades), 2) if trades[i].price > trades[i-1].price) if len(trades) >= 2 else 0
            win_rate = (win_count / len(sells)) * 100.0 if sells else 0.0
            total_closed = len(sells)

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
            "round_trip_trades": total_closed,
            "total_fees_inr": round(total_fees, 2),
            "total_dp_fees_inr": round(total_dp, 2),
            "win_rate_pct": round(win_rate, 2),
        }
