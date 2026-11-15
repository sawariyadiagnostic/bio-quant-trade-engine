"""RAVANA v4.3: Strategy 3 Standalone Two-Way VWAP Absorption Paper Trading Engine.

Evaluates Strategy 3 (TwoWayVWAPAbsorptionTrigger) over the 60-day 5-minute dataset:
1. Mean Reversion in Range Regimes: Fades Bollinger Band extremes back to VWAP when VR < 0.90.
2. Long Absorption: Lower Band touch + positive OFI.
3. Short Absorption: Upper Band touch + negative OFI.
4. Account Capital: ₹200,000.00 | Risk-at-Stop Cap: 0.25% (₹500).
5. Strict Point-in-Time Causality (Zero Lookahead / No n+1 Leakage).
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

# UTF-8 Console Safety
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ravana.backtest.orders import Order, OrderSide
from ravana.daemon.data_feeder import SectorDataFeeder
from ravana.execution.shoonya_adapter import (
    ShoonyaBrokerGateway,
    ShoonyaGatewayConfig,
)
from ravana.strategies.base import TriggerAction, TriggerSignal
from ravana.strategies.risk_adapter import StrategyRiskBridge
from ravana.strategies.state_ledger import SQLiteStateLedger
from ravana.strategies.two_way_absorption import TwoWayVWAPAbsorptionTrigger

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


class StandaloneStrategy3Trader:
    """Manages standalone execution of Strategy 3 (TwoWayVWAPAbsorptionTrigger)."""

    def __init__(
        self,
        initial_capital: float = 200000.0,
        days_to_trade: int = 60,
        max_mis_leverage: float = 2.0,
    ) -> None:
        self.initial_capital = float(initial_capital)
        self.current_capital = float(initial_capital)
        self.days_to_trade = int(days_to_trade)
        self.max_mis_leverage = float(max_mis_leverage)

        base_audit = os.path.join(BASE_DIR, "audit")
        os.makedirs(base_audit, exist_ok=True)
        self.db_path = os.path.join(base_audit, "ravana_state_ledger.db")
        self.audit_file = os.path.join(base_audit, "shoonya_paper_orders.json")

        self.ledger = SQLiteStateLedger(db_path=self.db_path)
        gw_cfg = ShoonyaGatewayConfig(
            paper_trading=True,
            default_product_type="M",
            audit_file_path=self.audit_file,
            statutory_dp_fee_inr=0.0,
        )
        self.gateway = ShoonyaBrokerGateway(config=gw_cfg)
        self.feeder = SectorDataFeeder(
            data_dir=os.path.join(BASE_DIR, "ravana", "data", "lake", "m3_ingested", "intraday_5m")
        )
        self.trigger = TwoWayVWAPAbsorptionTrigger(
            bb_lower=0.18,
            bb_upper=0.82,
            ofi_threshold=0.12,
            vr_ceiling=0.90,
        )
        self.risk_bridge = StrategyRiskBridge(
            risk_budget_pct=0.0025,
            max_mis_leverage=self.max_mis_leverage,
            min_trade_notional=25000.0,
            max_correlated_allocation_pct=0.50,
        )

        self.open_trades: Dict[str, Dict[str, Any]] = {}
        self.trade_history: List[Dict[str, Any]] = []

    def run_simulation(self) -> Dict[str, Any]:
        """Execute standalone simulation for Strategy 3."""
        datasets = self.feeder.load_all_sectors()
        ref_df = datasets["BANKBEES"]
        all_dates = sorted(set(ref_df.index.date))
        trading_dates = all_dates[-self.days_to_trade:]

        print("=" * 86)
        print("  RAVANA v4.3 STRATEGY 3: TWO-WAY VWAP BOLLINGER ABSORPTION")
        print(f"  Account Capital: ₹{self.initial_capital:,.2f} | Universe: Mean-Reverting Sector & Index BEES")
        print(f"  Horizon: {trading_dates[0]} to {trading_dates[-1]} ({len(trading_dates)} Trading Days on 5M bars)")
        print("  Mechanics: Lower Band Absorption -> Long to VWAP | Upper Band Rejection -> Short to VWAP")
        print("  Risk Cap: 0.25% (₹500) | Mid-Day Window: 10:00–14:15 IST | Mandatory Exit: 15:00 IST")
        print("=" * 86)

        test_universe = ["AUTOBEES", "BANKBEES", "ITBEES", "NIFTYBEES", "PHARMABEES"]
        signals_by_ts: Dict[pd.Timestamp, List[TriggerSignal]] = {}
        for sym in test_universe:
            if sym in datasets:
                sigs = self.trigger.generate_signals(datasets[sym], sym)
                for s in sigs:
                    signals_by_ts.setdefault(s.timestamp, []).append(s)

        all_timestamps = sorted(set.union(*[set(df.index) for df in datasets.values()]))
        active_timestamps = [
            ts for ts in all_timestamps
            if (ts.date() in set(trading_dates)) and ("09:15" <= ts.strftime("%H:%M") <= "15:05")
        ]

        print(f"Executing session replay over {len(active_timestamps)} 5-minute bars...")

        for ts in active_timestamps:
            t_str = ts.strftime("%H:%M")
            current_bars = {sym: df.loc[ts] for sym, df in datasets.items() if ts in df.index}

            # 1. Active position evaluation
            self._evaluate_active_positions(current_bars, ts)

            # 2. Pre-15:15 MIS Square-Off at 15:00 IST
            if t_str >= "15:00":
                self._square_off_all(current_bars, ts, "PRE_1515_MIS_SQUARE_OFF")
                continue

            # 3. New Entries (10:00 to 14:15 IST)
            if ("10:00" <= t_str <= "14:15") and (len(self.open_trades) < 2):
                candidates = signals_by_ts.get(ts, [])
                valid_candidates = [
                    c for c in candidates
                    if c.symbol not in self.open_trades
                    and c.symbol in current_bars
                    and c.action in (TriggerAction.BUY, TriggerAction.SELL)
                ]
                valid_candidates.sort(key=lambda s: s.conviction, reverse=True)
                available = 2 - len(self.open_trades)

                for sig in valid_candidates[:available]:
                    self._execute_trade(sig, current_bars[sig.symbol], ts)

        return self._generate_report(trading_dates)

    def _execute_trade(self, sig: TriggerSignal, bar: pd.Series, ts: pd.Timestamp) -> None:
        """Execute Long or Short VWAP Absorption trade."""
        curr_px = float(bar["Close"])
        raw_atr = float(bar.get("ATR_14", curr_px * 0.01))
        atr_val = max(raw_atr, curr_px * 0.0035)

        is_long = (sig.action == TriggerAction.BUY)
        initial_stop = float(sig.metadata.get("stop_loss", curr_px - 1.0 * atr_val if is_long else curr_px + 1.0 * atr_val))
        target_px = float(sig.metadata.get("target_price", curr_px + 1.5 * atr_val if is_long else curr_px - 1.5 * atr_val))

        sizing = self.risk_bridge.calculate_risk_at_stop_size(
            ticker=sig.symbol,
            entry_price=curr_px,
            stop_loss_price=initial_stop,
            account_equity=self.current_capital,
            product_type="MIS",
        )
        shares = sizing.shares
        if shares <= 0:
            return

        side = OrderSide.BUY if is_long else OrderSide.SELL
        order = Order(
            order_id=f"ABSORB_{sig.symbol}_{int(ts.timestamp())}",
            timestamp=ts,
            ticker=sig.symbol,
            side=side,
            shares=shares,
        )
        trade = self.gateway.place_order(order, reference_price=curr_px, bar_volume=float(bar.get("Volume", 1e5)))
        if trade:
            self.open_trades[sig.symbol] = {
                "ticker": sig.symbol,
                "shares": trade.shares,
                "entry_price": trade.price,
                "entry_time": ts,
                "current_stop": initial_stop,
                "target_price": target_px,
                "atr": atr_val,
                "is_long": is_long,
                "trigger_name": sig.trigger_name,
            }

    def _evaluate_active_positions(self, current_bars: Dict[str, pd.Series], ts: pd.Timestamp) -> None:
        """Evaluate Stop and Target for active absorption positions."""
        for sym, trade in list(self.open_trades.items()):
            if sym not in current_bars:
                continue
            bar = current_bars[sym]
            high_px = float(bar["High"])
            low_px = float(bar["Low"])
            bar_vol = float(bar.get("Volume", 1e5))
            is_long = trade["is_long"]

            if is_long:
                if low_px <= trade["current_stop"]:
                    self._close_trade(trade, trade["current_stop"], bar_vol, ts, "STOP_LOSS_HIT")
                elif high_px >= trade["target_price"]:
                    self._close_trade(trade, trade["target_price"], bar_vol, ts, "TARGET_HIT")
                elif high_px >= trade["entry_price"] + (0.50 * trade["atr"]):
                    trade["current_stop"] = max(trade["current_stop"], trade["entry_price"] * 1.0002)
            else:
                if high_px >= trade["current_stop"]:
                    self._close_trade(trade, trade["current_stop"], bar_vol, ts, "STOP_LOSS_HIT")
                elif low_px <= trade["target_price"]:
                    self._close_trade(trade, trade["target_price"], bar_vol, ts, "TARGET_HIT")
                elif low_px <= trade["entry_price"] - (0.50 * trade["atr"]):
                    trade["current_stop"] = min(trade["current_stop"], trade["entry_price"] * 0.9998)

    def _close_trade(
        self, trade: Dict[str, Any], exit_px: float, bar_vol: float, ts: pd.Timestamp, reason: str
    ) -> None:
        """Close trade and record PnL."""
        sym = trade["ticker"]
        side = OrderSide.SELL if trade["is_long"] else OrderSide.BUY
        order = Order(
            order_id=f"ABSORB_EXIT_{sym}_{int(ts.timestamp())}",
            timestamp=ts,
            ticker=sym,
            side=side,
            shares=trade["shares"],
        )
        closed = self.gateway.place_order(order, reference_price=exit_px, bar_volume=bar_vol)
        if closed:
            realized_pnl = (
                (closed.price - trade["entry_price"]) * trade["shares"]
                if trade["is_long"]
                else (trade["entry_price"] - closed.price) * trade["shares"]
            )
            self.current_capital += realized_pnl
            self.trade_history.append(
                {
                    "ticker": sym,
                    "side": "LONG" if trade["is_long"] else "SHORT",
                    "entry_time": trade["entry_time"],
                    "exit_time": ts,
                    "entry_price": trade["entry_price"],
                    "exit_price": closed.price,
                    "shares": trade["shares"],
                    "pnl": realized_pnl,
                    "reason": reason,
                    "trigger_name": trade["trigger_name"],
                }
            )
            del self.open_trades[sym]

    def _square_off_all(self, current_bars: Dict[str, pd.Series], ts: pd.Timestamp, reason: str) -> None:
        """Square off all remaining open positions."""
        for sym, trade in list(self.open_trades.items()):
            if sym in current_bars:
                bar = current_bars[sym]
                px = float(bar["Close"])
                vol = float(bar.get("Volume", 1e5))
                self._close_trade(trade, px, vol, ts, reason)

    def _generate_report(self, trading_dates: List[Any]) -> Dict[str, Any]:
        """Generate audit report for Strategy 3."""
        print("\n" + "=" * 86)
        print("  RAVANA v4.3 STRATEGY 3: STANDALONE SIMULATION RESULTS")
        print("=" * 86)

        total_trades = len(self.trade_history)
        if total_trades == 0:
            print("  NO TRADES EXECUTED.")
            return {"total_trades": 0, "net_pnl": 0.0}

        df_trades = pd.DataFrame(self.trade_history)
        wins = df_trades[df_trades["pnl"] > 0]
        losses = df_trades[df_trades["pnl"] <= 0]
        long_trades = df_trades[df_trades["side"] == "LONG"]
        short_trades = df_trades[df_trades["side"] == "SHORT"]

        gross_profit = wins["pnl"].sum() if not wins.empty else 0.0
        gross_loss = abs(losses["pnl"].sum()) if not losses.empty else 0.0
        net_pnl = df_trades["pnl"].sum()
        win_rate = (len(wins) / total_trades) * 100.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        print(f"  Total Trades Executed : {total_trades}")
        print(f"  Winning Trades        : {len(wins)} ({win_rate:.1f}%)")
        print(f"  Losing Trades         : {len(losses)} ({100 - win_rate:.1f}%)")
        print(f"  Gross Profit          : ₹{gross_profit:,.2f}")
        print(f"  Gross Loss            : ₹{gross_loss:,.2f}")
        print(f"  Net Realized PnL      : ₹{net_pnl:,.2f} ({(net_pnl / self.initial_capital) * 100:+.2f}%)")
        print(f"  Profit Factor         : {profit_factor:.2f}")
        print(f"  Ending Account Equity : ₹{self.current_capital:,.2f}")
        print("-" * 86)
        print(f"  Long Trades PnL       : ₹{long_trades['pnl'].sum():,.2f} ({len(long_trades)} trades, WR: {(len(long_trades[long_trades['pnl'] > 0]) / max(len(long_trades), 1)) * 100:.1f}%)")
        print(f"  Short Trades PnL      : ₹{short_trades['pnl'].sum():,.2f} ({len(short_trades)} trades, WR: {(len(short_trades[short_trades['pnl'] > 0]) / max(len(short_trades), 1)) * 100:.1f}%)")
        print("-" * 86)
        print("  Asset Breakdown:")
        for sym, grp in df_trades.groupby("ticker"):
            sym_wins = len(grp[grp["pnl"] > 0])
            sym_wr = (sym_wins / len(grp)) * 100.0
            print(f"    - {sym:<10}: {len(grp):>2} trades | WR: {sym_wr:>5.1f}% | Net PnL: ₹{grp['pnl'].sum():>9.2f}")
        print("-" * 86)
        print("  Exit Reasons:")
        for r, grp in df_trades.groupby("reason"):
            print(f"    - {r:<25}: {len(grp):>2} trades | Net PnL: ₹{grp['pnl'].sum():>9.2f}")
        print("=" * 86)

        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "net_pnl": net_pnl,
            "ending_capital": self.current_capital,
        }


if __name__ == "__main__":
    trader = StandaloneStrategy3Trader()
    trader.run_simulation()
