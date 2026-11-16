"""RAVANA v4.3: Real-Time Autonomous Paper Trading Daemon.

Executes live multi-sector paper trading on 5-minute candle epochs:
1. Streams live NSE candles via Yahoo Finance / Shoonya Gateway.
2. Evaluates active positions (Breakeven ratchet at +0.50 ATR, Pre-Dump Shield at 14:45).
3. Evaluates real-time Google Decisions (gemini-decision-1) Veto Guard.
4. Executes paper orders through ShoonyaBrokerGateway.
5. Manages All-Weather Overnight Allocation at 15:15 IST.
6. Runs 100% autonomously without human intervention.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
import json
import logging
import math
import os
import signal
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ravana.backtest.orders import Order, OrderSide, OrderStatus, Trade
from ravana.daemon.data_feeder import SECTOR_MAP, SectorDataFeeder
from ravana.execution.shoonya_adapter import ShoonyaBrokerGateway, ShoonyaGatewayConfig
from ravana.strategies.base import TriggerAction, TriggerSignal
from ravana.strategies.bear_inverse import AllWeatherOvernightAllocator
from ravana.strategies.risk_adapter import StrategyRiskBridge
from ravana.strategies.state_ledger import SQLiteStateLedger
from ravana.strategies.two_way_afternoon_europe import TwoWayAfternoonEuropeTrigger
from ravana.strategies.two_way_inav_spread import TwoWayINAVSpreadTrigger
from ravana.strategies.two_way_orb import TwoWayMorningORBTrigger
from ravana.strategies.two_way_pead_drift import TwoWaySectorDriftTrigger
from ravana.strategies.two_way_absorption import TwoWayLiquidityStopSweepTrigger, TwoWayVWAPAbsorptionTrigger
from ravana.strategies.volatility_scaler import AdaptiveThresholdScaler
from ravana.strategies.two_way_chop_reversion import TwoWayChopMeanReversionTrigger
from ravana.telemetry.telegram_bot import TelegramTelemetryBridge
from ravana.decisions.client import GeminiDecisionClient
from ravana.decisions.universe_allocator import UniverseAlphaCrown
from ravana.decisions.veto_guard import DecisionVetoGuard
from ravana.risk.decision_shield import DecisionRiskShield
from scripts.fetch_today_live_data import fetch_today_all, fetch_daily_all

logger = logging.getLogger("ravana.live_daemon")


class FlushFileHandler(logging.FileHandler):
    """FileHandler that flushes immediately on every write."""
    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


class LiveAutonomousPaperDaemon:
    """Production Autonomous Daemon for Live Paper Trading."""

    def __init__(
        self,
        initial_capital: float = 200000.0,
        risk_budget_pct: float = 0.0020,
        enable_gemini_decisions: bool = True,
    ) -> None:
        self.initial_capital = float(initial_capital)
        self.current_cash = float(initial_capital)
        self.risk_budget_pct = float(risk_budget_pct)
        self.enable_gemini_decisions = bool(enable_gemini_decisions)
        self.running = True

        self._init_logging()
        self._init_subsystems()

    def _init_logging(self) -> None:
        """Configure dual console and file logging with immediate flush."""
        os.makedirs(os.path.join(BASE_DIR, "audit"), exist_ok=True)
        log_path = os.path.join(BASE_DIR, "audit", "ravana_live_daemon.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                FlushFileHandler(log_path, encoding="utf-8", mode="a"),
            ],
            force=True,
        )
        logger.setLevel(logging.INFO)

    def _init_subsystems(self) -> None:
        """Initialize all execution, risk, and decision components."""
        self.audit_file = os.path.join(BASE_DIR, "audit", "shoonya_paper_orders.json")
        self.db_path = os.path.join(BASE_DIR, "audit", "ravana_state_ledger.db")
        self.ledger = SQLiteStateLedger(db_path=self.db_path)

        gw_cfg = ShoonyaGatewayConfig(
            paper_trading=True,
            default_product_type="M",
            audit_file_path=self.audit_file,
            statutory_dp_fee_inr=15.93,
        )
        self.gateway = ShoonyaBrokerGateway(config=gw_cfg)
        self.daily_cash_yield = (1.0 + 0.065) ** (1.0 / 252.0) - 1.0

        # Triggers
        self.orb_trigger = TwoWayMorningORBTrigger(
            ofi_threshold=0.15, vr_threshold=0.95,
            excluded_symbols={"GOLDBEES", "SILVERBEES", "BANKBEES", "JUNIORBEES", "CPSEETF", "PSUBNKBEES"},
        )
        self.inav_trigger = TwoWayINAVSpreadTrigger(z_threshold=1.5, bps_threshold=15.0)
        self.drift_trigger = TwoWaySectorDriftTrigger(z_mom_threshold=1.20, vr_threshold=0.95)
        self.stop_sweep_trigger = TwoWayLiquidityStopSweepTrigger(
            sweep_window=10, ofi_threshold=0.08, vr_ceiling=0.85, min_wick_ratio=0.40,
            supported_symbols={"PHARMABEES", "GOLDBEES", "SILVERBEES"},
        )
        self.vwap_absorb_trigger = TwoWayVWAPAbsorptionTrigger(
            vr_ceiling=0.85, allowed_symbols={"AUTOBEES", "ITBEES", "NIFTYBEES"},
        )
        self.europe_trigger = TwoWayAfternoonEuropeTrigger(momentum_threshold=0.0015, ofi_threshold=0.10)
        self.europe_trigger.supported_symbols = {"ITBEES", "NIFTYBEES"}

        # Dynamic Hurdle Scaler & Chop Mean-Reversion Trigger
        self.scaler = AdaptiveThresholdScaler()
        self.chop_trigger = TwoWayChopMeanReversionTrigger(
            bb_lower=0.15,
            bb_upper=0.85,
            ofi_threshold=0.05,
            vr_ceiling=0.75,
            micro_rr=(0.40, 0.50),
        )

        # Encrypted Telegram Mobile C2 Bridge
        self.telegram = TelegramTelemetryBridge()

        # Google Decisions Integration
        self.decision_client = GeminiDecisionClient(backtest_mode=False, enable_cache=True)
        self.veto_guard = DecisionVetoGuard(
            client=self.decision_client, ledger=self.ledger,
            dead_money_threshold=0.75, false_breakout_threshold=0.72,
        )
        self.alpha_crown = UniverseAlphaCrown(client=self.decision_client, ledger=self.ledger)
        self.risk_shield = DecisionRiskShield(client=self.decision_client, ledger=self.ledger)
        self.overnight_allocator = AllWeatherOvernightAllocator()

        # State
        self.open_trades: Dict[str, Dict[str, Any]] = {}
        self.active_overnight_positions: Dict[str, Dict[str, Any]] = {}
        self.trade_history: List[Dict[str, Any]] = []
        self.total_intraday_pnl: float = 0.0
        self.total_overnight_pnl: float = 0.0
        self.liquid_yield_pnl: float = 0.0
        self.vetoed_trades_count: int = 0
        self.pre_dump_exits_count: int = 0

    def start(self) -> None:
        """Main autonomous execution loop running continuously."""
        logger.info("=" * 78)
        logger.info("RAVANA v4.3 AUTONOMOUS LIVE PAPER TRADING DAEMON INITIATED")
        logger.info("Capital: ₹%.2f | Risk-at-Stop: 0.20%% | Universe: 10 Sector ETFs", self.current_cash)
        logger.info("Google Decisions API: ACTIVE (gemini-decision-1 Veto Shield)")
        logger.info("=" * 78)

        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        session_summary_sent = False
        while self.running:
            try:
                now_ist = datetime.now()
                t_str = now_ist.strftime("%H:%M")

                if t_str >= "15:35":
                    if not session_summary_sent:
                        equity = self.current_cash + sum(
                            t.get("shares", 1) * t.get("entry_price", 0.0)
                            for t in self.active_overnight_positions.values()
                        )
                        self.telegram.send_pnl_summary(
                            self.total_intraday_pnl, equity, self.active_overnight_positions
                        )
                        session_summary_sent = True
                    logger.info("Market session concluded for today (%s). Daemon idling.", t_str)
                    self._sleep_until_next_boundary()
                    continue

                if "09:15" <= t_str <= "15:30":
                    session_summary_sent = False
                    self._process_bar_epoch(now_ist, t_str)

                self._sleep_until_next_boundary()

            except Exception as exc:
                logger.error("Unexpected error in daemon loop: %s", exc, exc_info=True)
                time.sleep(10)

    def _process_bar_epoch(self, now_ist: datetime, t_str: str) -> None:
        """Process 5-minute candle boundary execution."""
        logger.info("--- [EPOCH %s IST] Fetching Live Sector Candles ---", t_str)
        today_dfs = fetch_today_all()
        if not today_dfs or "NSEI" not in today_dfs:
            logger.warning("Failed to retrieve live data for epoch %s. Retrying next epoch.", t_str)
            return

        bench_df = today_dfs["NSEI"]
        etf_dfs = {k: v for k, v in today_dfs.items() if k != "NSEI"}
        latest_ts = max(df.index[-1] for df in etf_dfs.values() if not df.empty)
        current_bars = {sym: df.loc[latest_ts] for sym, df in etf_dfs.items() if latest_ts in df.index}

        # 1. Manage active open positions
        self._manage_intraday_positions(current_bars, latest_ts)

        # 2. Morning Session Lock (11:30 AM)
        if t_str == "11:30":
            self._square_off_all(current_bars, latest_ts, "MORNING_1130_LOCK")
            self.telegram.send_session_alert("MORNING_1130_LOCK", f"Intraday square-off executed. PnL: ₹{self.total_intraday_pnl:+.2f}")
            return

        # 3. Mandatory Pre-Dump Square-Off (14:45 PM)
        if t_str >= "14:45" and t_str < "15:10":
            self._square_off_all(current_bars, latest_ts, "PRE_DUMP_1445_LOCK")
            self.telegram.send_session_alert("PRE_DUMP_1445_LOCK", "All MIS squared off before 15:00 broker dump.")
            return

        # 4. Overnight CNC Allocation (15:15 PM)
        if "15:10" <= t_str <= "15:25":
            self._evaluate_overnight_close(now_ist.date())
            return

        # 5. Evaluate New Intraday MIS Entries
        in_morning = ("09:35" <= t_str <= "11:15")
        in_afternoon = ("13:00" <= t_str <= "14:30")
        if (in_morning or in_afternoon) and (len(self.open_trades) < 2):
            self._evaluate_new_entries(etf_dfs, current_bars, latest_ts, t_str)

    def _manage_intraday_positions(self, current_bars: Dict[str, pd.Series], ts: pd.Timestamp) -> None:
        """Active position monitoring: Breakeven Ratchet & Target/Stop Checks."""
        # Pre-Dump Exit Shield check at 14:35-14:45
        if self.open_trades and "14:35" <= ts.strftime("%H:%M") <= "14:45":
            active_list = [{"ticker": s, "shares": t["shares"], "entry_price": t["entry_price"]} for s, t in self.open_trades.items()]
            shield_exits = self.risk_shield.evaluate_pre_dump_exit(ts, active_list, current_bars)
            for exit_sym, reason in shield_exits:
                if exit_sym in self.open_trades and exit_sym in current_bars:
                    trade = self.open_trades[exit_sym]
                    close_px = float(current_bars[exit_sym]["Close"])
                    vol = float(current_bars[exit_sym].get("Volume", 1e5))
                    self._close_trade(trade, close_px, vol, ts, "PRE_DUMP_SHIELD_EXIT")
                    self.pre_dump_exits_count += 1

        for sym, trade in list(self.open_trades.items()):
            if sym not in current_bars:
                continue
            bar = current_bars[sym]
            high_px = float(bar["High"])
            low_px = float(bar["Low"])
            vol = float(bar.get("Volume", 1e5))
            is_long = trade["is_long"]

            if is_long:
                # Ratchet to Breakeven once in profit (+0.50 ATR)
                if high_px >= trade["entry_price"] + (0.50 * trade["atr"]):
                    be_level = trade["entry_price"] * 1.0002
                    if trade["current_stop"] < be_level:
                        trade["current_stop"] = be_level
                        logger.info("[STOP RATCHET] %s Stop moved to Breakeven: ₹%.2f", sym, be_level)
                        self.telegram.send_stop_ratchet(sym, be_level, trade["entry_price"])
                if low_px <= trade["current_stop"]:
                    self._close_trade(trade, trade["current_stop"], vol, ts, "STOP_LOSS_HIT")
                elif high_px >= trade["target_price"]:
                    self._close_trade(trade, trade["target_price"], vol, ts, "TARGET_HIT")
            else:
                if low_px <= trade["entry_price"] - (0.50 * trade["atr"]):
                    be_level = trade["entry_price"] * 0.9998
                    if trade["current_stop"] > be_level:
                        trade["current_stop"] = be_level
                        logger.info("[STOP RATCHET] %s Stop moved to Breakeven: ₹%.2f", sym, be_level)
                        self.telegram.send_stop_ratchet(sym, be_level, trade["entry_price"])
                if high_px >= trade["current_stop"]:
                    self._close_trade(trade, trade["current_stop"], vol, ts, "STOP_LOSS_HIT")
                elif low_px <= trade["target_price"]:
                    self._close_trade(trade, trade["target_price"], vol, ts, "TARGET_HIT")

    def _evaluate_new_entries(
        self,
        etf_dfs: Dict[str, pd.DataFrame],
        current_bars: Dict[str, pd.Series],
        ts: pd.Timestamp,
        t_str: str,
    ) -> None:
        """Scan candidate triggers and execute top Google Decisions approved setup."""
        candidates: List[TriggerSignal] = []

        if "09:35" <= t_str <= "11:15":
            for sym, df in etf_dfs.items():
                if "09:35" <= t_str <= "10:45":
                    candidates.extend(self.orb_trigger.generate_signals(df, sym))
                    candidates.extend(self.drift_trigger.generate_signals(df, sym))
                    candidates.extend(self.stop_sweep_trigger.generate_signals(df, sym))
                    candidates.extend(self.inav_trigger.generate_signals(df, sym))
                    candidates.extend(self.vwap_absorb_trigger.generate_signals(df, sym))
                vr_val = float(df.get("VR_20", pd.Series(1.0, index=df.index)).iloc[-1])
                if vr_val < 0.75:
                    candidates.extend(self.chop_trigger.generate_signals(df, sym))
        elif "13:00" <= t_str <= "14:30":
            for sym, df in etf_dfs.items():
                if "13:15" <= t_str <= "13:45" and sym in ("ITBEES", "NIFTYBEES"):
                    candidates.extend(self.europe_trigger.generate_signals(df, sym))
                vr_val = float(df.get("VR_20", pd.Series(1.0, index=df.index)).iloc[-1])
                if vr_val < 0.75:
                    candidates.extend(self.chop_trigger.generate_signals(df, sym))

        valid = [
            c for c in candidates
            if c.symbol not in self.open_trades
            and c.symbol in current_bars
            and c.action.value in ("BUY", "SELL")
            and c.timestamp == ts
        ]
        if not valid:
            return

        winning_sig, alloc_frac = self.alpha_crown.select_top_allocation(
            valid, current_bars, self.current_cash
        )
        if winning_sig is not None and alloc_frac > 0.0:
            self._open_trade(winning_sig, current_bars[winning_sig.symbol], ts)

    def _open_trade(self, sig: TriggerSignal, bar: pd.Series, ts: pd.Timestamp) -> None:
        """Evaluate Gemini Veto Guard and place risk-compounded paper order."""
        if self.enable_gemini_decisions and self.veto_guard is not None:
            approved, veto_reason, prob = self.veto_guard.evaluate_signal(sig, bar)
            if not approved:
                self.vetoed_trades_count += 1
                logger.info("[VETO GUARD] Rejected %s (%s): %s (Prob: %.2f)", sig.symbol, sig.trigger_name, veto_reason, prob)
                self.telegram.send_veto_alert(sig.symbol, sig.trigger_name, veto_reason, prob)
                return

        curr_px = float(bar["Close"])
        raw_atr = float(bar.get("ATR_14", curr_px * 0.01))
        atr_val = max(raw_atr, curr_px * 0.0050)
        is_long = (sig.action.value == "BUY")

        if "target_price" in sig.metadata and "stop_price" in sig.metadata:
            initial_stop = float(sig.metadata["stop_price"])
            target_px = float(sig.metadata["target_price"])
        else:
            stop_mult = self.scaler.get_stop_multiplier(0.75, 1.0)
            target_mult = self.scaler.get_target_multiplier(1.50, 1.0)
            initial_stop = curr_px - (stop_mult * atr_val) if is_long else curr_px + (stop_mult * atr_val)
            target_px = curr_px + (target_mult * atr_val) if is_long else curr_px - (target_mult * atr_val)

        equity = self.current_cash + sum(t["shares"] * t["entry_price"] for t in self.open_trades.values())
        risk_budget = max(400.0, equity * self.risk_budget_pct)
        stop_dist = abs(curr_px - initial_stop)
        if stop_dist <= 0:
            return
        shares = max(1, min(int(risk_budget / stop_dist), int((equity * 0.20) / curr_px)))

        order = Order(
            order_id=f"LIVE_{sig.symbol}_{int(ts.timestamp())}",
            timestamp=ts,
            ticker=sig.symbol,
            side=OrderSide.BUY if is_long else OrderSide.SELL,
            shares=shares,
        )
        trade = self.gateway.place_order(order, reference_price=curr_px, bar_volume=float(bar.get("Volume", 1e5)))
        if trade:
            self.open_trades[sig.symbol] = {
                "ticker": sig.symbol, "shares": trade.shares, "entry_price": trade.price,
                "entry_time": ts, "current_stop": initial_stop, "target_price": target_px,
                "atr": atr_val, "trigger_name": sig.trigger_name, "is_long": is_long,
            }
            logger.info(
                "[ORDER FILLED] %s %d %s @ ₹%.2f | Stop: ₹%.2f | Target: ₹%.2f | Trigger: %s",
                "BUY" if is_long else "SELL", trade.shares, sig.symbol, trade.price,
                initial_stop, target_px, sig.trigger_name
            )
            self.telegram.send_trade_fill({
                "side": "BUY" if is_long else "SELL",
                "shares": trade.shares,
                "symbol": sig.symbol,
                "price": trade.price,
                "stop_price": initial_stop,
                "target_price": target_px,
                "trigger_name": sig.trigger_name,
            })

    def _close_trade(self, trade: Dict[str, Any], exit_px: float, vol: float, ts: pd.Timestamp, reason: str) -> None:
        """Close active intraday position and record PnL."""
        sym = trade["ticker"]
        order = Order(
            order_id=f"EXIT_{sym}_{int(ts.timestamp())}",
            timestamp=ts,
            ticker=sym,
            side=OrderSide.SELL if trade["is_long"] else OrderSide.BUY,
            shares=trade["shares"],
        )
        fill = self.gateway.place_order(order, reference_price=exit_px, bar_volume=vol)
        if fill:
            pnl = (fill.price - trade["entry_price"]) * trade["shares"] if trade["is_long"] else (trade["entry_price"] - fill.price) * trade["shares"]
            self.current_cash += pnl
            self.total_intraday_pnl += pnl
            self.trade_history.append({
                "type": "INTRADAY", "ticker": sym, "pnl": round(pnl, 2), "reason": reason,
                "entry_time": str(trade["entry_time"]), "exit_time": str(ts)
            })
            del self.open_trades[sym]
            logger.info("[TRADE CLOSED] %s closed @ ₹%.2f | PnL: ₹%+.2f | Reason: %s", sym, fill.price, pnl, reason)

    def _square_off_all(self, current_bars: Dict[str, pd.Series], ts: pd.Timestamp, reason: str) -> None:
        """Square off all open positions immediately."""
        for sym, trade in list(self.open_trades.items()):
            if sym in current_bars:
                close_px = float(current_bars[sym]["Close"])
                vol = float(current_bars[sym].get("Volume", 1e5))
                self._close_trade(trade, close_px, vol, ts, reason)

    def _rank_sectors_daily(self, daily_dfs: Dict[str, pd.DataFrame], current_date: Any) -> List[Tuple[str, float, float]]:
        """Rank sectors by 20-day return persistence returning (sym, px, score)."""
        scores = []
        for sym, df in daily_dfs.items():
            if sym in ("GOLDBEES", "SILVERBEES", "NSEI", "NSEBANK"):
                continue
            hist = df[df.index.date <= current_date]
            if len(hist) >= 20:
                ret_20d = float(hist["Close"].iloc[-1] / hist["Close"].iloc[-20] - 1.0)
                c_px = float(hist["Close"].iloc[-1])
                scores.append((sym, c_px, ret_20d))
        scores.sort(key=lambda x: x[2], reverse=True)
        return scores

    def _evaluate_overnight_close(self, current_date: Any) -> None:
        """Execute 15:15 PM All-Weather Overnight CNC Allocation."""
        if self.active_overnight_positions:
            return
        daily_dfs = fetch_daily_all()
        bench_df = daily_dfs.get("NSEI")
        if bench_df is None or bench_df.empty:
            return
        regime = self.overnight_allocator.determine_regime(bench_df, current_date)
        ranked = self._rank_sectors_daily(daily_dfs, current_date)
        targets = self.overnight_allocator.compute_allocations(regime, ranked)

        logger.info("[OVERNIGHT REGIME] Market classified as: %s (is_bear=%s, is_bull=%s)",
                    regime.get("regime"), regime.get("is_bear"), regime.get("is_bull"))
        self.telegram.send_session_alert(
            "OVERNIGHT_ALLOCATION",
            f"Regime: {regime.get('regime')} (Bull: {regime.get('is_bull')}, Bear: {regime.get('is_bear')})"
        )
        for tgt in targets:
            if tgt.ticker == "LIQUIDBEES" or tgt.target_weight_pct <= 0.0:
                continue
            if tgt.ticker == "INVERSE_NIFTY":
                alloc = min(self.current_cash * tgt.target_weight_pct, 70000.0)
                self.active_overnight_positions["INVERSE_NIFTY"] = {
                    "shares": 1, "entry_price": float(bench_df.iloc[-1]["Close"]),
                    "notional": alloc, "entry_date": current_date, "side": "INVERSE_SHORT"
                }
                self.current_cash -= alloc
                logger.info("[OVERNIGHT HEDGE] Activated Synthetic Inverse NIFTY Short (₹%.2f)", alloc)
                continue
            sym = tgt.ticker
            if sym in daily_dfs and not daily_dfs[sym].empty:
                c_px = float(daily_dfs[sym].iloc[-1]["Close"])
                alloc = min(self.current_cash * tgt.target_weight_pct, 70000.0)
                shares = max(1, int(alloc / c_px))
                if shares > 0 and self.current_cash >= shares * c_px:
                    order = Order(
                        order_id=f"CNC_OVERNIGHT_{sym}_{int(time.time())}",
                        timestamp=pd.Timestamp.now(),
                        ticker=sym,
                        side=OrderSide.BUY,
                        shares=shares,
                    )
                    fill = self.gateway.place_order(order, reference_price=c_px, bar_volume=1e6)
                    if fill:
                        self.current_cash -= shares * fill.price
                        self.active_overnight_positions[sym] = {
                            "shares": shares, "entry_price": fill.price,
                            "entry_date": current_date, "side": "LONG"
                        }
                        logger.info("[OVERNIGHT ALLOC] Bought %d shares of %s @ ₹%.2f (Allocation: ₹%.2f)",
                                    shares, sym, fill.price, shares * fill.price)

    def _sleep_until_next_boundary(self) -> None:
        """Sleep until the exact 5-minute epoch boundary + 5 seconds buffer."""
        now = datetime.now()
        minutes_to_next = 5 - (now.minute % 5)
        next_boundary = (now + timedelta(minutes=minutes_to_next)).replace(second=5, microsecond=0)
        sleep_secs = max(2.0, (next_boundary - now).total_seconds())
        logger.info("[DAEMON SLEEP] Synchronizing with next 5m candle at %s (Sleeping %.1fs)...", next_boundary.strftime("%H:%M:%S"), sleep_secs)
        time.sleep(sleep_secs)

    def _handle_shutdown(self, signum: int, frame: Any) -> None:
        """Graceful shutdown handler."""
        logger.info("Shutdown signal (%d) received. Exiting daemon safely...", signum)
        self.running = False


if __name__ == "__main__":
    daemon = LiveAutonomousPaperDaemon(initial_capital=200000.0, risk_budget_pct=0.0020)
    daemon.start()
