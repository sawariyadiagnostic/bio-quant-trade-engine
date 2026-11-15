"""
Mixed Multi-Sector Pure Intraday (MIS) Portfolio Paper Trading Engine for RAVANA v4.1.
Executes an authentic single-portfolio (₹2,00,000 capital) pure intraday rotation strategy
across all 10 Sector BEES, selecting the highest-conviction intraday setups,
strictly squaring off before 15:15 IST (zero overnight risk), and enforcing
the 5-layer stop-loss shield with two-tranche scale-out.
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

from ravana.backtest.orders import Order, OrderSide
from ravana.daemon.data_feeder import SECTOR_MAP, SectorDataFeeder
from ravana.execution.shoonya_adapter import (
    ShoonyaBrokerGateway,
    ShoonyaGatewayConfig,
)
from ravana.strategies.base import TriggerAction, TriggerSignal
from ravana.strategies.composite import TriggerCompositeStrategy
from ravana.strategies.risk_adapter import StrategyRiskBridge
from ravana.strategies.state_ledger import SQLiteStateLedger

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("mixed_intraday_portfolio")


class MixedIntradayPortfolioTrader:
    """Manages unified capital allocation across all 10 Sector BEES for pure intraday MIS trading."""

    def __init__(
        self,
        initial_capital: float = 200000.0,
        days_to_trade: int = 22,
        conviction_threshold: float = 0.70,
        max_concurrent_slots: int = 2,
    ) -> None:
        self.initial_capital = float(initial_capital)
        self.current_capital = float(initial_capital)
        self.days_to_trade = int(days_to_trade)
        self.conviction_threshold = float(conviction_threshold)
        self.max_concurrent_slots = int(max_concurrent_slots)

        base_audit = os.path.join(BASE_DIR, "audit")
        os.makedirs(base_audit, exist_ok=True)
        self.db_path = os.path.join(base_audit, "ravana_state_ledger.db")
        self.audit_file = os.path.join(base_audit, "shoonya_paper_orders.json")

        self.ledger = SQLiteStateLedger(db_path=self.db_path)
        gw_cfg = ShoonyaGatewayConfig(
            paper_trading=True,
            default_product_type="M",  # MIS Intraday
            audit_file_path=self.audit_file,
            statutory_dp_fee_inr=0.0,  # ₹0 DP fee on intraday MIS
        )
        self.gateway = ShoonyaBrokerGateway(config=gw_cfg)
        self.feeder = SectorDataFeeder()
        self.strategy = TriggerCompositeStrategy(vr_threshold=1.05)
        self.risk_bridge = StrategyRiskBridge(
            risk_budget_pct=0.0025,  # ₹500 risk cap on ₹2 Lakh
            max_mis_leverage=2.0,
            min_trade_notional=30000.0,
            max_correlated_allocation_pct=0.40,
        )

        self.open_trades: Dict[str, Dict[str, Any]] = {}
        self.trade_history: List[Dict[str, Any]] = []
        self.daily_equity: List[Tuple[Any, float]] = []

    def run_simulation(self) -> Dict[str, Any]:
        """Execute the 1-month pure intraday rotation simulation."""
        datasets = self.feeder.load_all_sectors()
        if not datasets:
            raise RuntimeError("No sector datasets available in lakehouse.")

        # Harmonize timelines across sectors
        ref_df = datasets["BANKBEES"]
        all_dates = sorted(set(ref_df.index.date))
        trading_dates = all_dates[-self.days_to_trade:]

        print("=" * 86)
        print("  RAVANA 1-MONTH PURE INTRADAY (MIS) MULTI-SECTOR ROTATION ENGINE")
        print(f"  Account Capital: ₹{self.initial_capital:,.2f} | Universe: 10 Sector BEES ETFs")
        print(f"  Horizon: {trading_dates[0]} to {trading_dates[-1]} ({len(trading_dates)} Trading Days)")
        print("  Rule: STRICT SQUARE-OFF BEFORE 15:15 IST (Zero Overnight Risk)")
        print("=" * 86)

        # Pre-evaluate signals once per sector
        print("Pre-evaluating intraday signals across all 10 sector ETFs...")
        signals_by_ts: Dict[pd.Timestamp, List[TriggerSignal]] = {}
        for sym, df in datasets.items():
            # Exclude OvernightContinuationTrigger for pure intraday run
            sigs = self.strategy.generate_signals(df, sym)
            for s in sigs:
                if s.trigger_name == "OvernightContinuationTrigger":
                    continue
                if s.action == TriggerAction.BUY and s.conviction >= self.conviction_threshold:
                    if s.timestamp not in signals_by_ts:
                        signals_by_ts[s.timestamp] = []
                    signals_by_ts[s.timestamp].append(s)

        # Filter timestamps to the 1-month window
        all_timestamps = sorted(set.union(*[set(df.index) for df in datasets.values()]))
        replay_timestamps = [ts for ts in all_timestamps if ts.date() in set(trading_dates)]

        print(f"Executing replay over {len(replay_timestamps)} 5-minute bars...")

        for ts in replay_timestamps:
            t_str = ts.strftime("%H:%M")
            current_bars: Dict[str, pd.Series] = {}
            for sym, df in datasets.items():
                if ts in df.index:
                    current_bars[sym] = df.loc[ts]

            # 1. Manage Active Positions with 5-Layer Stop Shield
            self._evaluate_active_positions(current_bars, ts, t_str)

            # 2. End-of-Day Mandatory MIS Square-Off (15:15 IST)
            if t_str >= "15:15":
                self._force_square_off_all(current_bars, ts)
                continue

            # 3. Candidate Entry Selection (only if slots open and time < 14:45)
            if t_str < "14:45" and len(self.open_trades) < self.max_concurrent_slots:
                candidates = signals_by_ts.get(ts, [])
                valid_candidates = []
                for c in candidates:
                    if c.symbol in self.open_trades:
                        continue
                    if c.symbol in current_bars:
                        valid_candidates.append(c)

                valid_candidates.sort(key=lambda s: s.conviction, reverse=True)
                available_slots = self.max_concurrent_slots - len(self.open_trades)

                for sig in valid_candidates[:available_slots]:
                    self._execute_buy(sig, current_bars[sig.symbol], ts)

        return self._generate_report(trading_dates)

    def _execute_buy(
        self, sig: TriggerSignal, bar: pd.Series, ts: pd.Timestamp
    ) -> None:
        """Size and execute paper BUY order for intraday setup."""
        curr_px = float(bar["Close"])
        raw_atr = float(bar.get("ATR_14", curr_px * 0.01))
        # Ensure minimum stop buffer of 35 bps to protect against micro-spread churn
        atr_val = max(raw_atr, curr_px * 0.0035)
        initial_stop = curr_px - (1.0 * atr_val)

        sizing = self.risk_bridge.calculate_risk_at_stop_size(
            ticker=sig.symbol,
            entry_price=curr_px,
            stop_loss_price=initial_stop,
            account_equity=self.current_capital,
            product_type="MIS",
        )
        shares = sizing.shares if sizing.is_approved else max(10, int(35000.0 / curr_px))
        if shares <= 0:
            return

        order_id = f"INTRA_BUY_{sig.symbol}_{int(ts.timestamp())}"
        order = Order(
            order_id=order_id,
            timestamp=ts,
            ticker=sig.symbol,
            side=OrderSide.BUY,
            shares=shares,
        )
        bar_vol = float(bar.get("Volume", 1e5))
        trade = self.gateway.place_order(order, reference_price=curr_px, bar_volume=bar_vol)

        if trade:
            target_px = curr_px + (2.0 * atr_val)
            self.ledger.record_entry(
                ticker=sig.symbol,
                originating_trigger=sig.trigger_name,
                product_type="MIS",
                shares_total=trade.shares,
                entry_price=trade.price,
                dynamic_profit_target=target_px,
                trailing_stop_loss=initial_stop,
                entry_timestamp=str(ts),
            )
            self.open_trades[sig.symbol] = {
                "ticker": sig.symbol,
                "shares": trade.shares,
                "entry_price": trade.price,
                "entry_time": ts,
                "current_stop": initial_stop,
                "atr": atr_val,
                "peak_price": trade.price,
                "trigger_name": sig.trigger_name,
                "tranche_1_filled": False,
                "bars_held": 0,
            }

    def _evaluate_active_positions(
        self,
        current_bars: Dict[str, pd.Series],
        ts: pd.Timestamp,
        t_str: str,
    ) -> None:
        """Evaluate 5-Layer Stop Shield on open intraday trades."""
        closed_syms = []

        for sym, trade in list(self.open_trades.items()):
            if sym not in current_bars:
                continue
            bar = current_bars[sym]
            high_px = float(bar["High"])
            low_px = float(bar["Low"])
            close_px = float(bar["Close"])
            bar_vol = float(bar.get("Volume", 1e5))

            trade["peak_price"] = max(trade["peak_price"], high_px)
            trade["bars_held"] += 1

            # Layer 1: Stop Loss
            if low_px <= trade["current_stop"]:
                self._close_trade(trade, trade["current_stop"], bar_vol, ts, "STOP_LOSS_HIT")
                closed_syms.append(sym)
                continue

            # Layer 2: Breakeven Ratchet (+0.45 ATR -> lock Entry + 2 bps)
            if high_px >= trade["entry_price"] + (0.45 * trade["atr"]):
                be_level = trade["entry_price"] * 1.0002
                if be_level > trade["current_stop"]:
                    trade["current_stop"] = be_level
                    self.ledger.update_ratchet(sym, trade["current_stop"], trade["peak_price"])

            # Layer 3: Two-Tranche Scale-Out at +0.75 ATR (Sell 50% shares)
            if not trade["tranche_1_filled"] and high_px >= trade["entry_price"] + (0.75 * trade["atr"]):
                half_shares = max(1, trade["shares"] // 2)
                ord_scale = Order(
                    order_id=f"INTRA_T1_{sym}_{int(ts.timestamp())}",
                    timestamp=ts,
                    ticker=sym,
                    side=OrderSide.SELL,
                    shares=half_shares,
                )
                fill = self.gateway.place_order(ord_scale, reference_price=high_px, bar_volume=bar_vol)
                if fill:
                    trade["shares"] -= fill.shares
                    trade["tranche_1_filled"] = True
                    trade["current_stop"] = max(trade["current_stop"], trade["entry_price"] + (0.25 * trade["atr"]))
                    self.ledger.record_tranche_1_exit(sym, fill.shares)
                    self.ledger.update_ratchet(sym, trade["current_stop"], trade["peak_price"])

            # Layer 4: Chandelier Trailing Stop at +1.0 ATR (Peak - 0.35 ATR)
            if high_px >= trade["entry_price"] + (1.0 * trade["atr"]):
                chand = trade["peak_price"] - (0.35 * trade["atr"])
                if chand > trade["current_stop"]:
                    trade["current_stop"] = chand
                    self.ledger.update_ratchet(sym, trade["current_stop"], trade["peak_price"])

            # Layer 5: Dead-Money Liquidation (4 bars with <0.20 ATR progress)
            if trade["bars_held"] >= 4 and trade["peak_price"] < trade["entry_price"] + (0.20 * trade["atr"]):
                self._close_trade(trade, close_px, bar_vol, ts, "DEAD_MONEY_EXIT")
                closed_syms.append(sym)
                continue

            # Special Afternoon Bond exit at 14:45 before retail MIS dump
            if trade["trigger_name"] == "AfternoonBondEuropeTWAPTrigger" and t_str >= "14:45":
                self._close_trade(trade, close_px, bar_vol, ts, "AFTERNOON_TWAP_1445_EXIT")
                closed_syms.append(sym)

        for s in closed_syms:
            self.open_trades.pop(s, None)

    def _force_square_off_all(
        self, current_bars: Dict[str, pd.Series], ts: pd.Timestamp
    ) -> None:
        """Square off all open MIS positions before 15:15 IST."""
        for sym, trade in list(self.open_trades.items()):
            bar = current_bars.get(sym)
            close_px = float(bar["Close"]) if bar is not None else trade["entry_price"]
            vol = float(bar.get("Volume", 1e5)) if bar is not None else 1e5
            self._close_trade(trade, close_px, vol, ts, "MIS_1515_AUTO_SQUARE_OFF")
        self.open_trades.clear()

    def _close_trade(
        self,
        trade: Dict[str, Any],
        exit_price: float,
        bar_vol: float,
        ts: pd.Timestamp,
        reason: str,
    ) -> None:
        """Liquidate remaining shares and record in audit logs."""
        sym = trade["ticker"]
        shares = trade["shares"]
        if shares <= 0:
            self.ledger.close_position(sym)
            return

        order = Order(
            order_id=f"INTRA_EXIT_{sym}_{int(ts.timestamp())}",
            timestamp=ts,
            ticker=sym,
            side=OrderSide.SELL,
            shares=shares,
        )
        fill = self.gateway.place_order(order, reference_price=exit_price, bar_volume=bar_vol)
        if fill:
            gross = (fill.price - trade["entry_price"]) * fill.shares
            fric = fill.friction.total_friction_inr
            net = gross - fric
            self.current_capital += net
            self.ledger.close_position(sym)

            # Update Bayesian Bandit Allocator
            self.strategy.bandit.update(trade["trigger_name"], net)

            dur_min = (ts - trade["entry_time"]).total_seconds() / 60.0
            self.trade_history.append({
                "ticker": sym,
                "entry_time": str(trade["entry_time"]),
                "exit_time": str(ts),
                "shares": fill.shares,
                "entry_price": round(trade["entry_price"], 2),
                "exit_price": round(fill.price, 2),
                "trigger": trade["trigger_name"],
                "reason": reason,
                "duration_min": round(dur_min, 0),
                "gross_pnl": round(gross, 2),
                "net_pnl": round(net, 2),
                "friction_inr": round(fric, 2),
                "capital_after": round(self.current_capital, 2),
            })

    def _generate_report(self, trading_dates: List[Any]) -> Dict[str, Any]:
        """Compile performance audit metrics for pure intraday rotation."""
        tot = len(self.trade_history)
        wins = sum(1 for t in self.trade_history if t["net_pnl"] > 0)
        losses = tot - wins
        wr = (wins / tot * 100.0) if tot > 0 else 0.0

        gp = sum(t["gross_pnl"] for t in self.trade_history if t["gross_pnl"] > 0)
        gl = abs(sum(t["gross_pnl"] for t in self.trade_history if t["gross_pnl"] <= 0))
        pf = (gp / gl) if gl > 0 else (99.0 if gp > 0 else 1.0)

        tot_fric = sum(t["friction_inr"] for t in self.trade_history)
        net_pnl = self.current_capital - self.initial_capital
        ret_pct = (net_pnl / self.initial_capital) * 100.0

        # Trigger breakdown
        trig_pnl: Dict[str, float] = {}
        trig_counts: Dict[str, int] = {}
        for t in self.trade_history:
            tr = t["trigger"]
            trig_counts[tr] = trig_counts.get(tr, 0) + 1
            trig_pnl[tr] = trig_pnl.get(tr, 0.0) + t["net_pnl"]

        # Sector breakdown
        sector_pnl: Dict[str, float] = {}
        for t in self.trade_history:
            s = t["ticker"]
            sector_pnl[s] = sector_pnl.get(s, 0.0) + t["net_pnl"]

        return {
            "initial_capital": self.initial_capital,
            "final_capital": round(self.current_capital, 2),
            "net_pnl": round(net_pnl, 2),
            "total_return_pct": round(ret_pct, 2),
            "total_trades": tot,
            "wins": wins,
            "losses": losses,
            "win_rate_pct": round(wr, 1),
            "profit_factor": round(pf, 2),
            "total_friction_inr": round(tot_fric, 2),
            "trading_days": len(trading_dates),
            "start_date": str(trading_dates[0]),
            "end_date": str(trading_dates[-1]),
            "trig_counts": trig_counts,
            "trig_pnl": trig_pnl,
            "sector_pnl": sector_pnl,
            "trades": self.trade_history,
        }


def main() -> None:
    t0 = time.perf_counter()
    trader = MixedIntradayPortfolioTrader(
        initial_capital=200000.0,
        days_to_trade=22,
        conviction_threshold=0.70,
        max_concurrent_slots=2,
    )
    report = trader.run_simulation()
    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 86)
    print("  PORTFOLIO PERFORMANCE AUDIT REPORT (1 MONTH PURE INTRADAY MIS)")
    print("=" * 86)
    print(f"  Starting Capital       : ₹{report['initial_capital']:,.2f}")
    print(f"  Final Capital          : ₹{report['final_capital']:,.2f}")
    print(f"  Net Realized PnL       : ₹{report['net_pnl']:+,.2f} ({report['total_return_pct']:+.2f}%)")
    print(f"  Total Trades Closed    : {report['total_trades']} (Wins: {report['wins']}, Losses: {report['losses']})")
    print(f"  Realized Win Rate      : {report['win_rate_pct']:.1f}%")
    print(f"  Profit Factor          : {report['profit_factor']:.2f}")
    print(f"  Total Friction Paid    : ₹{report['total_friction_inr']:,.2f} (Brokerage: ₹0.00, DP Fee: ₹0.00)")
    print(f"  Execution Time         : {elapsed:.2f} seconds")

    print("\n--- Structural Trigger Realized Performance ---")
    for trig, count in sorted(report["trig_counts"].items(), key=lambda x: x[1], reverse=True):
        pnl = report["trig_pnl"].get(trig, 0.0)
        print(f"  * {trig:<36}: {count:>3} trades | Net PnL: ₹{pnl:>+8.2f}")

    print("\n--- Cross-Sector Realized PnL Breakdown ---")
    for sym, pnl in sorted(report["sector_pnl"].items(), key=lambda x: x[1], reverse=True):
        print(f"  * {sym:<12}: Net PnL: ₹{pnl:>+8.2f}")

    print("\n--- SAMPLE ITEM-BY-ITEM INTRADAY TRADES ---")
    print(f"{'Date/Time In':<17} | {'Asset':<10} | {'Trigger':<24} | {'Dur':<6} | {'Net PnL (₹)':<11} | {'Reason'}")
    print("-" * 86)
    for t in report["trades"][:10]:
        print(
            f"{t['entry_time'][5:16]:<17} | {t['ticker']:<10} | {t['trigger'][:24]:<24} | "
            f"{t['duration_min']:>3.0f}m  | {t['net_pnl']:>+11.2f} | {t['reason']}"
        )

    print("=" * 86)


if __name__ == "__main__":
    main()
