"""
Mixed Multi-Sector Overnight Portfolio Paper Trading Engine for RAVANA v4.1.
Executes an authentic single-portfolio (₹2,00,000 capital) overnight strategy
rotating across all 10 Sector BEES, dynamically selecting the #1 strongest
institutional setup at 15:25 IST daily, holding overnight CNC delivery,
and applying the 09:15 Continuation Gate to compound multi-day trends.
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# UTF-8 Console Safety
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ravana.backtest.config import AssetType, TradingRegime
from ravana.backtest.friction import IndianFrictionModel
from ravana.backtest.orders import Order, OrderSide, OrderStatus
from ravana.daemon.data_feeder import SECTOR_MAP, SectorDataFeeder
from ravana.execution.shoonya_adapter import (
    ShoonyaBrokerGateway,
    ShoonyaGatewayConfig,
)
from ravana.strategies.state_ledger import SQLiteStateLedger

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("mixed_overnight_portfolio")


class MixedOvernightPortfolioTrader:
    """Manages unified capital allocation across all 10 Sector BEES for overnight trading."""

    def __init__(
        self,
        initial_capital: float = 200000.0,
        days_to_trade: int = 22,
        db_path: Optional[str] = None,
        audit_file: Optional[str] = None,
    ) -> None:
        self.initial_capital = float(initial_capital)
        self.current_cash = float(initial_capital)
        self.days_to_trade = int(days_to_trade)

        base_audit = os.path.join(BASE_DIR, "audit")
        os.makedirs(base_audit, exist_ok=True)
        self.db_path = db_path or os.path.join(base_audit, "ravana_state_ledger.db")
        self.audit_file = audit_file or os.path.join(base_audit, "shoonya_paper_orders.json")

        self.ledger = SQLiteStateLedger(db_path=self.db_path)
        gw_cfg = ShoonyaGatewayConfig(
            paper_trading=True,
            default_product_type="C",  # CNC Delivery
            audit_file_path=self.audit_file,
        )
        self.gateway = ShoonyaBrokerGateway(config=gw_cfg)
        self.feeder = SectorDataFeeder()

        # Active holding state: (ticker, shares, entry_price, entry_time, last_bar_price)
        self.active_position: Optional[Dict[str, Any]] = None
        self.trade_history: List[Dict[str, Any]] = []
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []
        self.dp_fees_saved_count: int = 0

    def run_simulation(self) -> Dict[str, Any]:
        """Execute the 1-month mixed multi-sector overnight portfolio run."""
        datasets = self.feeder.load_all_sectors()
        if not datasets:
            raise RuntimeError("No sector datasets available in lakehouse.")

        # Determine the 22-day trading window
        ref_df = datasets["BANKBEES"]
        all_dates = sorted(set(ref_df.index.date))
        trading_dates = all_dates[-self.days_to_trade:]

        print("=" * 86)
        print("  RAVANA 1-MONTH MIXED MULTI-SECTOR OVERNIGHT PAPER TRADING SYSTEM")
        print(f"  Account Capital: ₹{self.initial_capital:,.2f} | Universe: 10 Sector BEES ETFs")
        print(f"  Horizon: {trading_dates[0]} to {trading_dates[-1]} ({len(trading_dates)} Trading Days)")
        print("=" * 86)

        for d_idx, current_date in enumerate(trading_dates):
            # -------------------------------------------------------------
            # 1. MORNING SESSION (09:15 - 09:20 IST): Handle Open Positions
            # -------------------------------------------------------------
            if self.active_position is not None:
                pos = self.active_position
                sym = pos["ticker"]
                df_sym = datasets[sym]
                morning_bars = df_sym[
                    (df_sym.index.date == current_date)
                    & (df_sym.index.strftime("%H:%M") >= "09:15")
                    & (df_sym.index.strftime("%H:%M") <= "09:20")
                ]

                if not morning_bars.empty:
                    m_bar = morning_bars.iloc[0]
                    open_ts = morning_bars.index[0]
                    curr_open_px = float(m_bar["Open"])
                    curr_close_px = float(m_bar["Close"])
                    m_vwap = float(m_bar.get("VWAP", curr_open_px))

                    # Evaluate Continuation Gate (Hold if strong momentum)
                    # Hold if current price >= entry price or above morning VWAP
                    should_hold_continuation = (curr_close_px >= pos["entry_price"]) and (
                        curr_close_px >= m_vwap * 0.999
                    )

                    is_last_day = (d_idx == len(trading_dates) - 1)

                    if should_hold_continuation and not is_last_day:
                        self.dp_fees_saved_count += 1
                        pos["holding_days"] += 1
                        pos["current_price"] = curr_close_px
                        # Log continuation hold
                        logger.info(
                            "Continuation Gate Active for %s: holding through day to save ₹15.93 DP fee.",
                            sym,
                        )
                    else:
                        # Exit position at morning open
                        self._close_active_position(
                            exit_price=curr_open_px,
                            exit_time=open_ts,
                            reason="CONTINUATION_MOMENTUM_EXIT" if not is_last_day else "END_OF_MONTH_AUDIT",
                        )

            # -------------------------------------------------------------
            # 2. AFTERNOON SESSION (15:20 - 15:25 IST): Pick #1 Champion
            # -------------------------------------------------------------
            # Only enter if not already holding a multi-day continuation position
            if self.active_position is None and self.current_cash >= 30000.0:
                best_candidate = self._rank_and_select_champion(datasets, current_date)
                if best_candidate is not None:
                    sym, close_px, bar_ts, strength = best_candidate
                    self._open_new_position(sym, close_px, bar_ts, strength)

            # Record daily closing equity
            portfolio_val = self.current_cash
            if self.active_position is not None:
                portfolio_val += self.active_position["shares"] * self.active_position["current_price"]
            self.equity_curve.append((pd.Timestamp(f"{current_date} 15:30:00+05:30"), portfolio_val))

        # Close any lingering position at end
        if self.active_position is not None:
            sym = self.active_position["ticker"]
            df_sym = datasets[sym]
            last_px = float(df_sym.iloc[-1]["Close"])
            last_ts = df_sym.index[-1]
            self._close_active_position(last_px, last_ts, "FINAL_SESSION_CLOSE")

        return self._generate_report(trading_dates)

    def _rank_and_select_champion(
        self, datasets: Dict[str, pd.DataFrame], current_date: Any
    ) -> Optional[Tuple[str, float, pd.Timestamp, float]]:
        """Evaluate all 10 sectors at 15:25 IST and pick the #1 strongest institutional setup."""
        candidates = []
        for sym, df in datasets.items():
            sub = df[
                (df.index.date == current_date)
                & (df.index.strftime("%H:%M") >= "15:20")
                & (df.index.strftime("%H:%M") <= "15:25")
            ]
            if sub.empty:
                continue
            bar = sub.iloc[-1]
            bar_ts = sub.index[-1]
            close = float(bar["Close"])
            vwap = float(bar.get("VWAP", close))
            spd_bps = abs(float(bar.get("SPREAD_BPS", 0.0)))
            ofi = float(bar.get("OFI_5", 0.0))

            # Qualification gates: Close >= VWAP * 0.999, Tracking spread <= 50 bps
            if close >= vwap * 0.999 and spd_bps <= 50.0:
                strength_bps = ((close - vwap) / vwap) * 10000.0
                score = strength_bps + (ofi * 50.0)
                candidates.append((sym, close, bar_ts, score))

        if not candidates:
            return None

        # Sort descending by score
        candidates.sort(key=lambda x: x[3], reverse=True)
        return candidates[0]

    def _open_new_position(
        self, sym: str, close_px: float, bar_ts: pd.Timestamp, score: float
    ) -> None:
        """Size and execute paper CNC BUY order for chosen champion."""
        # Cash cap 0.98x (max 98% cash allocated to position, 2% friction buffer)
        allocatable_cash = self.current_cash * 0.98
        shares = int(np.floor(allocatable_cash / close_px))
        if shares <= 0:
            return

        order_id = f"MIXED_ON_BUY_{sym}_{int(bar_ts.timestamp())}"
        order = Order(
            order_id=order_id,
            timestamp=bar_ts,
            ticker=sym,
            side=OrderSide.BUY,
            shares=shares,
        )
        trade = self.gateway.place_order(order, reference_price=close_px)
        if trade:
            cost = (trade.shares * trade.price) + trade.friction.total_friction_inr
            self.current_cash -= cost

            self.active_position = {
                "ticker": sym,
                "shares": trade.shares,
                "entry_price": trade.price,
                "current_price": trade.price,
                "entry_time": bar_ts,
                "score": score,
                "holding_days": 1,
            }
            # Record in SQLite WAL ledger
            self.ledger.record_entry(
                ticker=sym,
                originating_trigger="MixedOvernightSelector",
                product_type="CNC",
                shares_total=trade.shares,
                entry_price=trade.price,
                dynamic_profit_target=trade.price * 1.025,
                trailing_stop_loss=trade.price * 0.985,
                entry_timestamp=str(bar_ts),
            )
            logger.info("OPENED %s: %d shs @ ₹%.2f (Score: %.1f)", sym, trade.shares, trade.price, score)

    def _close_active_position(
        self, exit_price: float, exit_time: pd.Timestamp, reason: str
    ) -> None:
        """Execute paper CNC SELL order, account for statutory frictions and DP fees."""
        pos = self.active_position
        if pos is None:
            return
        sym = pos["ticker"]
        shares = pos["shares"]

        order_id = f"MIXED_ON_SELL_{sym}_{int(exit_time.timestamp())}"
        order = Order(
            order_id=order_id,
            timestamp=exit_time,
            ticker=sym,
            side=OrderSide.SELL,
            shares=shares,
        )
        trade = self.gateway.place_order(order, reference_price=exit_price)
        if trade:
            proceeds = (trade.shares * trade.price) - trade.friction.total_friction_inr
            self.current_cash += proceeds

            gross = (trade.price - pos["entry_price"]) * trade.shares
            fric = trade.friction.total_friction_inr
            net = gross - fric
            ret_pct = ((trade.price / pos["entry_price"]) - 1.0) * 100.0

            record = {
                "ticker": sym,
                "entry_time": str(pos["entry_time"]),
                "exit_time": str(exit_time),
                "shares": trade.shares,
                "entry_price": round(pos["entry_price"], 2),
                "exit_price": round(trade.price, 2),
                "holding_days": pos["holding_days"],
                "gross_pnl": round(gross, 2),
                "net_pnl": round(net, 2),
                "return_pct": round(ret_pct, 2),
                "friction_inr": round(fric, 2),
                "capital_after": round(self.current_cash, 2),
                "reason": reason,
            }
            self.trade_history.append(record)
            self.ledger.close_position(sym)
            self.active_position = None

            logger.info(
                "CLOSED %s: %d shs @ ₹%.2f | Net PnL: %+.2f | Capital: ₹%.2f",
                sym,
                trade.shares,
                trade.price,
                net,
                self.current_cash,
            )

    def _generate_report(self, trading_dates: List[Any]) -> Dict[str, Any]:
        """Compute institutional metrics across the mixed overnight portfolio."""
        tot_trades = len(self.trade_history)
        wins = sum(1 for t in self.trade_history if t["net_pnl"] > 0)
        losses = tot_trades - wins
        win_rate = (wins / tot_trades * 100.0) if tot_trades > 0 else 0.0

        gross_profit = sum(t["gross_pnl"] for t in self.trade_history if t["gross_pnl"] > 0)
        gross_loss = abs(sum(t["gross_pnl"] for t in self.trade_history if t["gross_pnl"] <= 0))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 1.0)

        tot_friction = sum(t["friction_inr"] for t in self.trade_history)
        net_pnl = self.current_cash - self.initial_capital
        total_ret_pct = (net_pnl / self.initial_capital) * 100.0

        # Compute drawdown curve
        vals = [v for _, v in self.equity_curve]
        peak = vals[0]
        max_dd = 0.0
        for v in vals:
            peak = max(peak, v)
            dd = (peak - v) / peak
            max_dd = max(max_dd, dd)

        # Sector distribution
        sector_counts: Dict[str, int] = {}
        sector_pnl: Dict[str, float] = {}
        for t in self.trade_history:
            sym = t["ticker"]
            sector_counts[sym] = sector_counts.get(sym, 0) + 1
            sector_pnl[sym] = sector_pnl.get(sym, 0.0) + t["net_pnl"]

        return {
            "initial_capital": self.initial_capital,
            "final_capital": round(self.current_cash, 2),
            "net_pnl": round(net_pnl, 2),
            "total_return_pct": round(total_ret_pct, 2),
            "total_trades": tot_trades,
            "wins": wins,
            "losses": losses,
            "win_rate_pct": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown_pct": round(max_dd * 100.0, 2),
            "total_friction_inr": round(tot_friction, 2),
            "dp_fees_saved": self.dp_fees_saved_count,
            "trading_days": len(trading_dates),
            "start_date": str(trading_dates[0]),
            "end_date": str(trading_dates[-1]),
            "sector_counts": sector_counts,
            "sector_pnl": sector_pnl,
            "trades": self.trade_history,
        }


def main() -> None:
    t0 = time.perf_counter()
    trader = MixedOvernightPortfolioTrader(initial_capital=200000.0, days_to_trade=22)
    report = trader.run_simulation()
    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 86)
    print("  PORTFOLIO PERFORMANCE AUDIT REPORT (1 MONTH MIXED OVERNIGHT)")
    print("=" * 86)
    print(f"  Starting Capital       : ₹{report['initial_capital']:,.2f}")
    print(f"  Final Capital          : ₹{report['final_capital']:,.2f}")
    print(f"  Net Realized PnL       : ₹{report['net_pnl']:+,.2f} ({report['total_return_pct']:+.2f}%)")
    print(f"  Total Trades Closed    : {report['total_trades']} (Wins: {report['wins']}, Losses: {report['losses']})")
    print(f"  Realized Win Rate      : {report['win_rate_pct']:.1f}%")
    print(f"  Profit Factor          : {report['profit_factor']:.2f}")
    print(f"  Max Drawdown           : {report['max_drawdown_pct']:.2f}%")
    print(f"  Total Friction Paid    : ₹{report['total_friction_inr']:,.2f} (Shoonya ₹0 Delivery Brokerage)")
    print(f"  Continuation Holds     : {report['dp_fees_saved']} sessions (Saved ₹{report['dp_fees_saved'] * 15.93:.2f} DP Fees)")
    print(f"  Execution Time         : {elapsed:.2f} seconds")

    print("\n--- Cross-Sector Capital Allocation Breakdown ---")
    for sym, count in sorted(report["sector_counts"].items(), key=lambda x: x[1], reverse=True):
        pnl = report["sector_pnl"].get(sym, 0.0)
        print(f"  * {sym:<12}: {count:>2} trades | Net PnL: ₹{pnl:>+8.2f}")

    print("\n--- COMPLETE 1-MONTH ITEM-BY-ITEM TRADE JOURNAL ---")
    print(f"{'Date In':<11} | {'Asset':<10} | {'Shares':<6} | {'Entry (₹)':<9} | {'Exit Date':<11} | {'Exit (₹)':<9} | {'Net PnL (₹)':<11} | {'Capital (₹)':<11}")
    print("-" * 86)
    for t in report["trades"]:
        d_in = t["entry_time"][:10]
        d_out = t["exit_time"][:10]
        print(
            f"{d_in:<11} | {t['ticker']:<10} | {t['shares']:<6} | "
            f"{t['entry_price']:>9.2f} | {d_out:<11} | {t['exit_price']:>9.2f} | "
            f"{t['net_pnl']:>+11.2f} | ₹{t['capital_after']:>10,.2f}"
        )

    print("=" * 86)
    print("AUDIT PERSISTENCE: Positions in audit/ravana_state_ledger.db | Orders in audit/shoonya_paper_orders.json")


if __name__ == "__main__":
    main()
