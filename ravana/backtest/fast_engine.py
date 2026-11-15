"""Numba-Accelerated High-Frequency and Intraday Backtesting Engine.

Provides NumbaFastBacktester and FastBacktester utilizing JIT-compiled
intrabar excursion simulation with multi-layer ratchet stops, Chandelier trailing,
MIS two-tranche scaling, and dead-money time stop gates.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
from numba import njit
import numpy as np
import pandas as pd

from ravana.backtest.config import AssetType, BacktestConfig, TradingRegime
from ravana.backtest.engine import BacktestResult
from ravana.backtest.friction import FrictionBreakdown
from ravana.backtest.orders import OrderSide, RoundTripTrade, Trade


@njit(cache=True, nogil=True)
def compute_fast_atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> np.ndarray:
    """Compute Average True Range using Wilder exponential smoothing."""
    n = len(high)
    atr = np.zeros(n, dtype=np.float64)
    if n == 0:
        return atr
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, max(hc, lc))
    atr[0] = tr[0]
    alpha = 1.0 / float(period)
    for i in range(1, n):
        atr[i] = (tr[i] * alpha) + (atr[i - 1] * (1.0 - alpha))
    return atr


@njit(cache=True, nogil=True)
def _update_ratchet_stops(
    high_px: float,
    p_entry: float,
    atr: float,
    peak_px: float,
    current_stop: float,
    fric_rate: float,
) -> float:
    """Apply Ratchet Layer 2 (friction breakeven) and Layer 4 (Chandelier)."""
    stop = current_stop
    if high_px >= p_entry + 0.45 * atr:
        stop = max(stop, p_entry * (1.0 + fric_rate))
    if high_px >= p_entry + 1.0 * atr:
        stop = max(stop, peak_px - 0.35 * atr)
    return stop


@njit(cache=True, nogil=True)
def _log_trade(
    records: np.ndarray,
    idx: int,
    bar: int,
    side: float,
    shares: int,
    px: float,
    fric: float,
    t_type: float,
    gross: float,
    net: float,
) -> int:
    """Append executed trade into records buffer."""
    if idx < len(records):
        records[idx, 0] = float(bar)
        records[idx, 1] = side
        records[idx, 2] = float(shares)
        records[idx, 3] = px
        records[idx, 4] = fric
        records[idx, 5] = t_type
        records[idx, 6] = gross
        records[idx, 7] = net
        return idx + 1
    return idx


@njit(cache=True, nogil=True)
def _run_numba_fast_loop(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    signals: np.ndarray,
    atrs: np.ndarray,
    initial_capital: float,
    friction_rate: float,
    signal_threshold: float,
    is_mis: bool,
    target_atr_mult: float,
    initial_stop_atr: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Inner JIT loop executing causal order fills and intrabar ratchets."""
    n = len(opens)
    equity = np.zeros(n, dtype=np.float64)
    cash_curve = np.zeros(n, dtype=np.float64)
    records = np.zeros((min(max(1000, n // 2), 20000), 8), dtype=np.float64)

    cash = initial_capital
    shares = 0
    p_entry = 0.0
    atr = 0.0
    trailing_stop = 0.0
    target = 0.0
    peak_price = 0.0
    bars_held = 0
    tranche_1_sold = False
    trade_idx = 0

    for i in range(n):
        if shares == 0:
            sig = signals[i]
            if (not np.isnan(sig)) and (sig > signal_threshold):
                p_entry = opens[i]
                atr = atrs[i] if atrs[i] > 0.0 else p_entry * 0.01
                trailing_stop = p_entry - (initial_stop_atr * atr)
                target = p_entry + (target_atr_mult * atr)
                peak_price = p_entry
                bars_held = 0
                tranche_1_sold = False
                new_shares = int((cash * 0.98) // p_entry) if p_entry > 0.0 else 0
                if new_shares > 0:
                    fric = p_entry * new_shares * friction_rate
                    if (new_shares * p_entry) + fric <= cash:
                        shares = new_shares
                        cash -= ((new_shares * p_entry) + fric)
                        trade_idx = _log_trade(records, trade_idx, i, 1.0, shares, p_entry, fric, 1.0, 0.0, 0.0)

        if shares > 0:
            if lows[i] <= trailing_stop:
                exit_px = trailing_stop if opens[i] >= trailing_stop else opens[i]
                fric = exit_px * shares * friction_rate
                cash += (shares * exit_px - fric)
                gross = (exit_px - p_entry) * shares
                net = gross - fric - (p_entry * shares * friction_rate)
                trade_idx = _log_trade(records, trade_idx, i, -1.0, shares, exit_px, fric, 2.0, gross, net)
                shares = 0
                equity[i] = cash
                cash_curve[i] = cash
                continue

            peak_price = max(peak_price, highs[i])
            trailing_stop = _update_ratchet_stops(highs[i], p_entry, atr, peak_price, trailing_stop, friction_rate)

            if is_mis and (not tranche_1_sold) and highs[i] >= p_entry + 0.75 * atr:
                t1_shares = shares // 2
                if t1_shares > 0:
                    t1_px = p_entry + 0.75 * atr
                    t1_fric = t1_px * t1_shares * friction_rate
                    cash += (t1_shares * t1_px - t1_fric)
                    t1_gross = (t1_px - p_entry) * t1_shares
                    t1_net = t1_gross - t1_fric - (p_entry * t1_shares * friction_rate)
                    trade_idx = _log_trade(records, trade_idx, i, -1.0, t1_shares, t1_px, t1_fric, 3.0, t1_gross, t1_net)
                    shares -= t1_shares
                tranche_1_sold = True
                trailing_stop = max(trailing_stop, p_entry + 0.25 * atr)

            if highs[i] >= target:
                fric = target * shares * friction_rate
                cash += (shares * target - fric)
                gross = (target - p_entry) * shares
                net = gross - fric - (p_entry * shares * friction_rate)
                trade_idx = _log_trade(records, trade_idx, i, -1.0, shares, target, fric, 4.0, gross, net)
                shares = 0
                equity[i] = cash
                cash_curve[i] = cash
                continue

            bars_held += 1
            if bars_held >= 4 and peak_price < p_entry + 0.20 * atr:
                fric = closes[i] * shares * friction_rate
                cash += (shares * closes[i] - fric)
                gross = (closes[i] - p_entry) * shares
                net = gross - fric - (p_entry * shares * friction_rate)
                trade_idx = _log_trade(records, trade_idx, i, -1.0, shares, closes[i], fric, 5.0, gross, net)
                shares = 0
                equity[i] = cash
                cash_curve[i] = cash
                continue

        equity[i] = cash + (shares * closes[i])
        cash_curve[i] = cash

    if shares > 0 and n > 0:
        exit_px = closes[n - 1]
        fric = exit_px * shares * friction_rate
        cash += (shares * exit_px - fric)
        gross = (exit_px - p_entry) * shares
        net = gross - fric - (p_entry * shares * friction_rate)
        trade_idx = _log_trade(records, trade_idx, n - 1, -1.0, shares, exit_px, fric, 6.0, gross, net)
        shares = 0
        equity[n - 1] = cash
        cash_curve[n - 1] = cash

    return equity, cash_curve, records, trade_idx


class FastBacktester:
    """High-throughput Numba-accelerated event-driven backtesting engine."""

    def __init__(self, config: Optional[BacktestConfig] = None) -> None:
        self.config = config or BacktestConfig()

    def run_fast(
        self,
        df: pd.DataFrame,
        signals: Union[pd.Series, np.ndarray],
        signal_threshold: float = 0.0,
        initial_stop_atr: float = 1.0,
        target_atr_mult: float = 2.0,
        friction_rate: float = 0.0005,
        is_mis: Optional[bool] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Execute JIT inner loop directly, returning raw numpy equity and trade records."""
        opens, highs, lows, closes, _, sigs, atrs = self._unpack_arrays(df, signals)
        mis = is_mis if is_mis is not None else (self.config.regime == TradingRegime.INTRADAY_MIS)
        cap = float(self.config.initial_capital_inr)

        return _run_numba_fast_loop(
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            signals=sigs,
            atrs=atrs,
            initial_capital=cap,
            friction_rate=float(friction_rate),
            signal_threshold=float(signal_threshold),
            is_mis=bool(mis),
            target_atr_mult=float(target_atr_mult),
            initial_stop_atr=float(initial_stop_atr),
        )

    def run(
        self,
        df: pd.DataFrame,
        signals: Union[pd.Series, np.ndarray],
        ticker: str = "BANKBEES.NS",
        signal_threshold: float = 0.0,
        initial_stop_atr: float = 1.0,
        target_atr_mult: float = 2.0,
        friction_rate: float = 0.0005,
        is_mis: Optional[bool] = None,
    ) -> BacktestResult:
        """Run JIT-accelerated backtest over market bar DataFrame and return BacktestResult."""
        eq_arr, cash_arr, records, count = self.run_fast(
            df=df,
            signals=signals,
            signal_threshold=signal_threshold,
            initial_stop_atr=initial_stop_atr,
            target_atr_mult=target_atr_mult,
            friction_rate=friction_rate,
            is_mis=is_mis,
        )

        equity_s = pd.Series(eq_arr, index=df.index, name="Equity")
        cash_s = pd.Series(cash_arr, index=df.index, name="Cash")
        dd_s = (equity_s - equity_s.cummax()) / equity_s.cummax()
        trades, round_trips = self._reconstruct_trades(records, count, df.index, ticker)
        metrics = self._compute_metrics(equity_s, trades, round_trips)

        return BacktestResult(
            equity_curve=equity_s,
            cash_curve=cash_s,
            drawdown_curve=dd_s,
            trades=trades,
            metrics=metrics,
            config=self.config,
            closed_round_trips=round_trips,
        )

    @staticmethod
    def _unpack_arrays(
        df: pd.DataFrame, signals: Union[pd.Series, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert DataFrame and signals into contiguous 1D float64 arrays."""
        opens = np.ascontiguousarray(df["Open"].to_numpy(dtype=np.float64))
        highs = np.ascontiguousarray(df["High"].to_numpy(dtype=np.float64))
        lows = np.ascontiguousarray(df["Low"].to_numpy(dtype=np.float64))
        closes = np.ascontiguousarray(df["Close"].to_numpy(dtype=np.float64))
        vol = (
            np.ascontiguousarray(df["Volume"].to_numpy(dtype=np.float64))
            if "Volume" in df.columns
            else np.full(len(df), 1e5, dtype=np.float64)
        )
        sigs = (
            np.ascontiguousarray(signals.to_numpy(dtype=np.float64))
            if isinstance(signals, pd.Series)
            else np.ascontiguousarray(signals, dtype=np.float64)
        )
        if "ATR" in df.columns:
            atrs = np.ascontiguousarray(df["ATR"].to_numpy(dtype=np.float64))
        else:
            atrs = compute_fast_atr(highs, lows, closes, period=14)
        return opens, highs, lows, closes, vol, sigs, atrs

    @staticmethod
    def _reconstruct_trades(
        records: np.ndarray, count: int, index: pd.DatetimeIndex, ticker: str
    ) -> Tuple[List[Trade], List[RoundTripTrade]]:
        """Parse raw Numba trade buffer into Trade and RoundTripTrade contracts."""
        trades: List[Trade] = []
        round_trips: List[RoundTripTrade] = []
        if count == 0:
            return trades, round_trips

        bar_indices = records[:count, 0].astype(np.int64)
        ts_array = index[bar_indices]
        buy_entry_time: Optional[pd.Timestamp] = None
        buy_entry_price: float = 0.0

        for k in range(count):
            ts = ts_array[k]
            side = OrderSide.BUY if records[k, 1] > 0 else OrderSide.SELL
            shares = int(records[k, 2])
            px = float(records[k, 3])
            fric_amt = float(records[k, 4])
            turnover = float(px * shares)
            bps = (fric_amt / turnover * 10000.0) if turnover > 0 else 0.0

            fric = FrictionBreakdown(
                turnover_inr=round(turnover, 2),
                stt_inr=0.0,
                exchange_charges_inr=0.0,
                sebi_charges_inr=0.0,
                stamp_duty_inr=0.0,
                gst_inr=0.0,
                brokerage_inr=0.0,
                dp_charges_inr=0.0,
                total_friction_inr=round(fric_amt, 2),
                effective_bps=round(bps, 2),
            )
            trade = Trade(
                trade_id=f"FAST_{k}_{int(ts.timestamp())}",
                timestamp=ts,
                ticker=ticker,
                side=side,
                shares=shares,
                price=round(px, 4),
                slippage_bps=0.0,
                friction=fric,
            )
            trades.append(trade)

            if side == OrderSide.BUY:
                buy_entry_time = ts
                buy_entry_price = px
            elif side == OrderSide.SELL and buy_entry_time is not None:
                gross = round(float(records[k, 6]), 2)
                net = round(float(records[k, 7]), 2)
                rt = RoundTripTrade(
                    entry_time=buy_entry_time,
                    exit_time=ts,
                    ticker=ticker,
                    shares=shares,
                    entry_price=round(buy_entry_price, 4),
                    exit_price=round(px, 4),
                    gross_pnl=gross,
                    net_pnl=net,
                    total_friction=round(gross - net, 2),
                )
                round_trips.append(rt)

        return trades, round_trips

    def _compute_metrics(
        self,
        equity: pd.Series,
        trades: List[Trade],
        round_trips: List[RoundTripTrade],
    ) -> Dict[str, float]:
        """Compute institutional performance metrics (Sharpe, Drawdown, CAGR, Win Rate)."""
        if len(equity) < 2:
            return {}
        ret_series = equity.pct_change().dropna()
        tot_ret = (equity.iloc[-1] / equity.iloc[0]) - 1.0
        delta_days = max(1.0 / 24.0, (equity.index[-1] - equity.index[0]).total_seconds() / 86400.0)
        ann_factor = 365.25 / delta_days if delta_days >= 7.0 else 1.0
        cagr = ((1.0 + tot_ret) ** ann_factor) - 1.0 if tot_ret > -1.0 else -1.0
        ann_vol = float(ret_series.std()) * math.sqrt(252.0 * 75.0) if ret_series.std() > 0 else 0.0
        sharpe = (float(ret_series.mean()) * (252.0 * 75.0) - self.config.risk_free_rate) / ann_vol if ann_vol > 0 else 0.0

        dd = (equity - equity.cummax()) / equity.cummax()
        max_dd = float(abs(dd.min()))
        wins = sum(1 for r in round_trips if r.is_win)
        win_rate = (wins / len(round_trips) * 100.0) if round_trips else 0.0
        tot_fric = sum(t.friction.total_friction_inr for t in trades)

        return {
            "initial_capital": round(float(equity.iloc[0]), 2),
            "final_equity": round(float(equity.iloc[-1]), 2),
            "total_return_pct": round(tot_ret * 100.0, 2),
            "cagr_pct": round(cagr * 100.0, 2),
            "annualized_vol_pct": round(ann_vol * 100.0, 2),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown_pct": round(max_dd * 100.0, 2),
            "total_trades": len(trades),
            "round_trip_trades": len(round_trips),
            "total_fees_inr": round(tot_fric, 2),
            "win_rate_pct": round(win_rate, 2),
        }


# Canonical alias for compatibility
NumbaFastBacktester = FastBacktester
