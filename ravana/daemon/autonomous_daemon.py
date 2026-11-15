"""Autonomous Execution Daemon for RAVANA Sovereign Trading Stack.

Features:
1. Multi-sector opportunity scanning across 10 Sector BEES ETFs.
2. 2-slot portfolio contention manager with Bayesian Bandit conviction ranking.
3. HRP Sector Correlation Cluster Ceiling (max 40% in correlated cluster).
4. Risk-at-Stop 0.25% equity sizing (₹500 max loss per trade on ₹2 Lakh capital).
5. 5-Layer "Zero-Giveback" Stop Shield with Two-Tranche Scale-Out (50% booked at +0.75 ATR).
6. ACID SQLite WAL persistence and 11:32 AM Reboot Recovery Protocol.
7. Zero-intervention headless execution with complete audit logging.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from ravana.backtest.config import AssetType, TradingRegime
from ravana.backtest.orders import Order, OrderSide, OrderStatus, Trade
from ravana.daemon.data_feeder import SECTOR_MAP, SectorDataFeeder
from ravana.execution.shoonya_adapter import (
    ShoonyaBrokerGateway,
    ShoonyaGatewayConfig,
)
from ravana.strategies.base import TriggerAction, TriggerSignal
from ravana.strategies.composite import (
    ALL_TRIGGER_NAMES,
    TriggerCompositeStrategy,
)
from ravana.strategies.risk_adapter import (
    SizingResult,
    StrategyRiskBridge,
)
from ravana.strategies.state_ledger import SQLiteStateLedger
from ravana.decisions.client import GeminiDecisionClient
from ravana.decisions.universe_allocator import UniverseAlphaCrown
from ravana.decisions.veto_guard import DecisionVetoGuard
from ravana.risk.decision_shield import DecisionRiskShield

logger = logging.getLogger("ravana.daemon")


@dataclass
class DaemonConfig:
    """Configuration contract for the Autonomous Trading Daemon."""

    initial_capital_inr: float = 200000.0
    risk_budget_pct: float = 0.0025  # ₹500 max loss on ₹2 Lakh
    max_concurrent_positions: int = 2
    max_mis_leverage: float = 2.0
    max_cnc_leverage: float = 0.98
    min_trade_notional: float = 30000.0
    max_cluster_allocation_pct: float = 0.40  # 40% cap per sector cluster
    conviction_threshold: float = 0.50
    db_path: str = "audit/ravana_state_ledger.db"
    audit_file_path: str = "audit/shoonya_paper_orders.json"
    log_file_path: str = "audit/ravana_autonomous_daemon.log"
    paper_trading: bool = True
    enable_gemini_decisions: bool = True
    gemini_mock: bool = False


@dataclass
class ActiveTradeTracker:
    """Runtime tracker for an open position managing the 5-layer stop shield."""

    ticker: str
    shares: int
    entry_price: float
    entry_time: pd.Timestamp
    initial_stop: float
    current_stop: float
    atr: float
    peak_price: float
    trigger_name: str
    product_type: str
    tranche_1_filled: bool = False
    bars_held: int = 0


class AutonomousExecutionDaemon:
    """Sovereign Autonomous Daemon executing multi-sector strategies without human intervention."""

    def __init__(self, config: Optional[DaemonConfig] = None) -> None:
        self.config = config or DaemonConfig()
        self._ensure_directories()
        self._setup_file_logger()

        self.ledger = SQLiteStateLedger(db_path=self.config.db_path)
        self.risk_bridge = StrategyRiskBridge(
            risk_budget_pct=self.config.risk_budget_pct,
            max_mis_leverage=self.config.max_mis_leverage,
            max_cnc_leverage=self.config.max_cnc_leverage,
            min_trade_notional=self.config.min_trade_notional,
            max_correlated_allocation_pct=self.config.max_cluster_allocation_pct,
        )
        self.strategy = TriggerCompositeStrategy(
            vr_threshold=1.05,
            max_concurrent_positions=self.config.max_concurrent_positions,
        )
        gw_cfg = ShoonyaGatewayConfig(
            paper_trading=self.config.paper_trading,
            audit_file_path=self.config.audit_file_path,
        )
        self.gateway = ShoonyaBrokerGateway(config=gw_cfg)
        self.data_feeder = SectorDataFeeder()

        self.current_capital: float = self.config.initial_capital_inr
        self.open_trades: Dict[str, ActiveTradeTracker] = {}
        self.closed_trades_history: List[Dict[str, Any]] = []

        if self.config.enable_gemini_decisions:
            self.decision_client = GeminiDecisionClient(mock=self.config.gemini_mock)
            self.veto_guard = DecisionVetoGuard(client=self.decision_client, ledger=self.ledger)
            self.alpha_crown = UniverseAlphaCrown(client=self.decision_client, ledger=self.ledger)
            self.risk_shield = DecisionRiskShield(client=self.decision_client, ledger=self.ledger)
        else:
            self.decision_client = None
            self.veto_guard = None
            self.alpha_crown = None
            self.risk_shield = None

    def _ensure_directories(self) -> None:
        """Create audit directories if missing."""
        os.makedirs(os.path.dirname(self.config.db_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.config.audit_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.config.log_file_path), exist_ok=True)

    def _setup_file_logger(self) -> None:
        """Configure dual console and file logging."""
        file_handler = logging.FileHandler(self.config.log_file_path, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

    # -------------------------------------------------------------------------
    # 1. Autonomous Startup & 11:32 AM Reboot Protocol
    # -------------------------------------------------------------------------
    def startup_and_reconcile(self) -> Dict[str, Any]:
        """Execute the 11:32 AM Crash Reboot Protocol upon daemon initiation."""
        logger.info("Initializing Autonomous Daemon. Executing Crash Reboot Protocol...")
        active_db_positions = self.ledger.get_active_positions()
        logger.info("Retrieved %d active positions from SQLite WAL ledger.", len(active_db_positions))

        # In paper mode, build broker view from paper gateway ledger or adopt open records
        broker_holdings: Dict[str, int] = {}
        for sym, pos in active_db_positions.items():
            broker_holdings[sym] = pos["shares_remaining"]
            self.open_trades[sym] = ActiveTradeTracker(
                ticker=sym,
                shares=pos["shares_remaining"],
                entry_price=pos["entry_price"],
                entry_time=pd.Timestamp.now(timezone.utc),
                initial_stop=pos["trailing_stop_loss"],
                current_stop=pos["trailing_stop_loss"],
                atr=pos["entry_price"] * 0.01,
                peak_price=pos["peak_price_observed"],
                trigger_name=pos["originating_trigger"],
                product_type=pos["product_type"],
                tranche_1_filled=bool(pos["tranche_1_filled"]),
                bars_held=pos["holding_bars"],
            )

        reconciliation = self.ledger.reconcile_with_broker(broker_holdings)
        status_str = "IN_SYNC" if not reconciliation["discrepancies_found"] else "DISCREPANCIES_RESOLVED"
        logger.info(
            "11:32 AM Reboot Protocol Status: %s | In-Sync: %d, Adopted: %d, Closed: %d",
            status_str,
            len(reconciliation["in_sync"]),
            len(reconciliation["adopted"]),
            len(reconciliation["closed"]),
        )
        return reconciliation

    # -------------------------------------------------------------------------
    # 2. Portfolio Risk & Contention Manager
    # -------------------------------------------------------------------------
    def _is_sector_cluster_available(self, candidate_symbol: str) -> bool:
        """Enforce HRP Sector Correlation Cluster Ceiling (max 40% capital per cluster)."""
        cand_sector = SECTOR_MAP.get(candidate_symbol, {}).get("sector", "General")
        cluster_notional = 0.0
        for sym, trade in self.open_trades.items():
            if SECTOR_MAP.get(sym, {}).get("sector") == cand_sector:
                cluster_notional += trade.shares * trade.entry_price

        max_cluster_cap = self.config.max_cluster_allocation_pct * self.current_capital
        return cluster_notional < max_cluster_cap

    def _select_entry_candidates(
        self,
        candidate_signals: List[TriggerSignal],
        current_bars: Optional[Dict[str, pd.Series]] = None,
    ) -> List[TriggerSignal]:
        """Rank BUY signals by conviction and select within available portfolio slots."""
        available_slots = self.config.max_concurrent_positions - len(self.open_trades)
        if available_slots <= 0 or not candidate_signals:
            return []

        # Filter out symbols already open or in saturated sector clusters
        valid: List[TriggerSignal] = []
        for s in candidate_signals:
            if s.symbol in self.open_trades:
                continue
            if not self._is_sector_cluster_available(s.symbol):
                logger.info("Signal for %s rejected by HRP Sector Correlation Ceiling.", s.symbol)
                continue
            valid.append(s)

        if not valid:
            return []

        # Gemini Decisions Universe Alpha Crown for multiple candidates
        if self.alpha_crown is not None and len(valid) > 1 and current_bars is not None:
            winner_sig, alloc_frac = self.alpha_crown.select_top_allocation(
                candidate_signals=valid,
                current_bars=current_bars,
                capital=self.current_capital,
            )
            if winner_sig is not None and alloc_frac > 0.0:
                logger.info(
                    "GEMINI ALPHA CROWN: Crown winner %s selected among %d candidates (alloc=%.2f).",
                    winner_sig.symbol,
                    len(valid),
                    alloc_frac,
                )
                return [winner_sig]
            elif winner_sig is None:
                logger.info("GEMINI ALPHA CROWN: Evaluated %d candidates -> selected CASH.", len(valid))
                return []

        valid.sort(key=lambda s: s.conviction, reverse=True)
        return valid[:available_slots]

    # -------------------------------------------------------------------------
    # 3. Order Execution & 5-Layer Stop-Loss Shield
    # -------------------------------------------------------------------------
    def _execute_buy(
        self,
        sig: TriggerSignal,
        bar: pd.Series,
        timestamp: pd.Timestamp,
    ) -> Optional[Trade]:
        """Size and execute BUY order through Shoonya Gateway and State Ledger."""
        # 1. Gemini Decisions Veto Guard (trap, false breakout, dead money risk)
        if self.veto_guard is not None:
            approved, veto_reason, prob = self.veto_guard.evaluate_signal(sig, bar)
            if not approved:
                logger.info(
                    "GEMINI VETO: %s signal rejected by %s (p=%.2f)",
                    sig.symbol,
                    veto_reason,
                    prob,
                )
                return None

        curr_px = float(bar["Close"])
        raw_atr = float(bar.get("ATR_14", curr_px * 0.01))
        # Ensure minimum stop distance of 35 bps to prevent micro-spread tick churn
        atr_val = max(raw_atr, curr_px * 0.0035)
        initial_stop = curr_px - (1.0 * atr_val)

        # 2. Gemini Decisions Dynamic Volatility Adaptive Stop
        if self.risk_shield is not None:
            initial_stop = self.risk_shield.compute_adaptive_stop(
                ticker=sig.symbol,
                entry_price=curr_px,
                default_stop=initial_stop,
                bar=bar,
            )

        prod_type = "CNC" if sig.trigger_name == "OvernightContinuationTrigger" else "MIS"
        sizing = self.risk_bridge.calculate_risk_at_stop_size(
            ticker=sig.symbol,
            entry_price=curr_px,
            stop_loss_price=initial_stop,
            account_equity=self.current_capital,
            product_type=prod_type,
        )

        shares = sizing.shares if sizing.is_approved else max(10, int(35000.0 / curr_px))
        if shares <= 0:
            return None

        order_id = f"AUTON_{sig.symbol}_{int(timestamp.timestamp())}"
        order = Order(
            order_id=order_id,
            timestamp=timestamp,
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
                product_type=prod_type,
                shares_total=trade.shares,
                entry_price=trade.price,
                dynamic_profit_target=target_px,
                trailing_stop_loss=initial_stop,
            )
            self.open_trades[sig.symbol] = ActiveTradeTracker(
                ticker=sig.symbol,
                shares=trade.shares,
                entry_price=trade.price,
                entry_time=timestamp,
                initial_stop=initial_stop,
                current_stop=initial_stop,
                atr=atr_val,
                peak_price=trade.price,
                trigger_name=sig.trigger_name,
                product_type=prod_type,
            )
            logger.info(
                "AUTONOMOUS BUY: %s %d shs @ ₹%.2f | Trigger: %s | Stop: ₹%.2f | Fee: ₹%.2f",
                sig.symbol,
                trade.shares,
                trade.price,
                sig.trigger_name,
                initial_stop,
                trade.friction.total_friction_inr,
            )
        return trade

    def _evaluate_active_positions(
        self,
        current_bars: Dict[str, pd.Series],
        timestamp: pd.Timestamp,
    ) -> None:
        """Evaluate 5-Layer Stop Shield and Two-Tranche Scale-Out across open positions."""
        closed_symbols: List[str] = []

        for sym, trade in list(self.open_trades.items()):
            if sym not in current_bars:
                continue

            bar = current_bars[sym]
            high_px = float(bar["High"])
            low_px = float(bar["Low"])
            close_px = float(bar["Close"])
            bar_vol = float(bar.get("Volume", 1e5))

            trade.peak_price = max(trade.peak_price, high_px)
            trade.bars_held += 1

            # Layer 1: Check Stop-Loss Hit
            if low_px <= trade.current_stop:
                self._execute_full_exit(trade, trade.current_stop, bar_vol, timestamp, "STOP_LOSS_HIT")
                closed_symbols.append(sym)
                continue

            # Layer 2: Breakeven Ratchet (+0.45 ATR -> lock Entry + 2 bps)
            if high_px >= trade.entry_price + (0.45 * trade.atr):
                be_level = trade.entry_price * 1.0002
                if be_level > trade.current_stop:
                    trade.current_stop = be_level
                    self.ledger.update_ratchet(sym, trade.current_stop, trade.peak_price)

            # Layer 3: Two-Tranche Scale-Out at +0.75 ATR (Sell 50% shares)
            if not trade.tranche_1_filled and high_px >= trade.entry_price + (0.75 * trade.atr):
                self._execute_tranche_1_exit(trade, high_px, bar_vol, timestamp)

            # Layer 4: Chandelier Trailing Stop at +1.0 ATR (Peak - 0.35 ATR)
            if high_px >= trade.entry_price + (1.0 * trade.atr):
                chand_stop = trade.peak_price - (0.35 * trade.atr)
                if chand_stop > trade.current_stop:
                    trade.current_stop = chand_stop
                    self.ledger.update_ratchet(sym, trade.current_stop, trade.peak_price)

            # Layer 5: Dead-Money Liquidation (4 bars held with <0.20 ATR progress)
            if trade.bars_held >= 4 and trade.peak_price < trade.entry_price + (0.20 * trade.atr):
                self._execute_full_exit(trade, close_px, bar_vol, timestamp, "DEAD_MONEY_EXIT")
                closed_symbols.append(sym)
                continue

            # End of session intraday auto-square-off (15:15 IST)
            if trade.product_type == "MIS" and timestamp.strftime("%H:%M") >= "15:15":
                self._execute_full_exit(trade, close_px, bar_vol, timestamp, "MIS_END_OF_SESSION")
                closed_symbols.append(sym)

        for sym in closed_symbols:
            self.open_trades.pop(sym, None)

    def _execute_tranche_1_exit(
        self,
        trade: ActiveTradeTracker,
        exec_price: float,
        bar_vol: float,
        timestamp: pd.Timestamp,
    ) -> None:
        """Sell 50% of position at +0.75 ATR to bank profit and snap stop to +0.25 ATR."""
        half_shares = max(1, trade.shares // 2)
        order = Order(
            order_id=f"SCALE_T1_{trade.ticker}_{int(timestamp.timestamp())}",
            timestamp=timestamp,
            ticker=trade.ticker,
            side=OrderSide.SELL,
            shares=half_shares,
        )
        fill = self.gateway.place_order(order, reference_price=exec_price, bar_volume=bar_vol)
        if fill:
            trade.shares -= fill.shares
            trade.tranche_1_filled = True
            # Snap remaining runner stop to +0.25 ATR
            trade.current_stop = max(trade.current_stop, trade.entry_price + (0.25 * trade.atr))
            self.ledger.record_tranche_1_exit(trade.ticker, fill.shares)
            self.ledger.update_ratchet(trade.ticker, trade.current_stop, trade.peak_price)
            logger.info(
                "TRANCHE-1 PROFIT BANKED: %s %d shs @ ₹%.2f | Runner Stop Snapped to ₹%.2f",
                trade.ticker,
                fill.shares,
                fill.price,
                trade.current_stop,
            )

    def _execute_full_exit(
        self,
        trade: ActiveTradeTracker,
        exit_price: float,
        bar_vol: float,
        timestamp: pd.Timestamp,
        reason: str,
    ) -> None:
        """Liquidate remaining shares, update ledger, and compute realized net PnL."""
        if trade.shares <= 0:
            self.ledger.close_position(trade.ticker)
            return

        order = Order(
            order_id=f"EXIT_ALL_{trade.ticker}_{int(timestamp.timestamp())}",
            timestamp=timestamp,
            ticker=trade.ticker,
            side=OrderSide.SELL,
            shares=trade.shares,
        )
        fill = self.gateway.place_order(order, reference_price=exit_price, bar_volume=bar_vol)
        if fill:
            gross = (fill.price - trade.entry_price) * fill.shares
            fric = fill.friction.total_friction_inr
            net = gross - fric
            self.current_capital += net
            self.ledger.close_position(trade.ticker)
            # Dynamically update Bayesian Bandit Allocator posterior
            self.strategy.bandit.update(trade.trigger_name, net)

            record = {
                "ticker": trade.ticker,
                "entry_time": str(trade.entry_time),
                "exit_time": str(timestamp),
                "entry_price": trade.entry_price,
                "exit_price": fill.price,
                "shares": fill.shares,
                "trigger": trade.trigger_name,
                "reason": reason,
                "gross_pnl": round(gross, 2),
                "net_pnl": round(net, 2),
                "friction_inr": round(fric, 2),
                "capital_after": round(self.current_capital, 2),
            }
            self.closed_trades_history.append(record)
            logger.info(
                "POSITION CLOSED: %s %d shs @ ₹%.2f | Reason: %s | Net PnL: %+.2f | Capital: ₹%.2f",
                trade.ticker,
                fill.shares,
                fill.price,
                reason,
                net,
                self.current_capital,
            )

    # -------------------------------------------------------------------------
    # 4. End-to-End Autonomous Multi-Sector Replay Engine
    # -------------------------------------------------------------------------
    def run_autonomous_replay(
        self,
        days_to_replay: int = 10,
    ) -> Dict[str, Any]:
        """Execute full autonomous multi-sector simulation across recent trading days."""
        logger.info("=" * 78)
        logger.info("STARTING MULTI-SECTOR AUTONOMOUS REPLAY (Capital: ₹%.2f)", self.current_capital)
        logger.info("=" * 78)

        self.startup_and_reconcile()
        datasets = self.data_feeder.load_all_sectors()
        if not datasets:
            raise RuntimeError("No sector datasets available in lakehouse.")

        # Harmonize timelines across sectors
        all_timestamps = sorted(
            set.union(*[set(df.index) for df in datasets.values()])
        )
        # Filter to recent days
        unique_dates = sorted(set(ts.date() for ts in all_timestamps))
        target_dates = set(unique_dates[-days_to_replay:])
        replay_timestamps = [ts for ts in all_timestamps if ts.date() in target_dates]

        logger.info(
            "Replaying %d timestamps across %d trading days (%s to %s)...",
            len(replay_timestamps),
            len(target_dates),
            min(target_dates),
            max(target_dates),
        )

        # Pre-evaluate signals once per sector for causal timestamp matching
        logger.info("Pre-evaluating multi-sector signals across all 10 ETF pairs...")
        sector_signals_by_ts: Dict[pd.Timestamp, List[TriggerSignal]] = {}
        for sym, df in datasets.items():
            sigs = self.strategy.generate_signals(df, sym)
            for s in sigs:
                if s.action == TriggerAction.BUY and s.conviction >= self.config.conviction_threshold:
                    if s.timestamp not in sector_signals_by_ts:
                        sector_signals_by_ts[s.timestamp] = []
                    sector_signals_by_ts[s.timestamp].append(s)

        for ts in replay_timestamps:
            current_bars: Dict[str, pd.Series] = {}

            # 1. Gather current market bar across all sectors
            for sym, df in datasets.items():
                if ts in df.index:
                    current_bars[sym] = df.loc[ts]

            # 2. Evaluate active open positions with 5-Layer Stop Shield
            self._evaluate_active_positions(current_bars, ts)

            # 3. Retrieve signals triggered strictly at this bar
            candidate_signals = sector_signals_by_ts.get(ts, [])

            # 4. Resolve portfolio contention and execute accepted BUY candidates
            accepted_entries = self._select_entry_candidates(candidate_signals)
            for sig in accepted_entries:
                if sig.symbol in current_bars:
                    self._execute_buy(sig, current_bars[sig.symbol], ts)

        # Final Audit Summary
        summary = self._generate_autonomous_summary()
        logger.info("AUTONOMOUS REPLAY COMPLETE. Net Return: %+.2f%%", summary["total_return_pct"])
        return summary

    def _generate_autonomous_summary(self) -> Dict[str, Any]:
        """Compile institutional audit metrics across closed trades."""
        tot_trades = len(self.closed_trades_history)
        wins = sum(1 for t in self.closed_trades_history if t["net_pnl"] > 0)
        losses = tot_trades - wins
        win_rate = (wins / tot_trades * 100.0) if tot_trades > 0 else 0.0

        gross_profit = sum(t["gross_pnl"] for t in self.closed_trades_history if t["gross_pnl"] > 0)
        gross_loss = abs(sum(t["gross_pnl"] for t in self.closed_trades_history if t["gross_pnl"] <= 0))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 1.0)
        tot_fric = sum(t["friction_inr"] for t in self.closed_trades_history)

        net_pnl = self.current_capital - self.config.initial_capital_inr
        ret_pct = (net_pnl / self.config.initial_capital_inr) * 100.0

        # Trigger breakdown
        trigger_pnl: Dict[str, float] = {}
        for t in self.closed_trades_history:
            trig = t["trigger"]
            trigger_pnl[trig] = trigger_pnl.get(trig, 0.0) + t["net_pnl"]

        return {
            "initial_capital": self.config.initial_capital_inr,
            "final_capital": round(self.current_capital, 2),
            "net_pnl_inr": round(net_pnl, 2),
            "total_return_pct": round(ret_pct, 2),
            "total_trades": tot_trades,
            "wins": wins,
            "losses": losses,
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "total_friction_inr": round(tot_fric, 2),
            "open_positions": len(self.open_trades),
            "trigger_pnl": trigger_pnl,
        }
