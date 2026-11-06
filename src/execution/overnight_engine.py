"""
Institutional Overnight Execution Engine & Trailing Micro-Stop Controller.
Integrates Phase A empirical discoveries:
- Tier 1: Continuous Daily Unfiltered ETF compounding (NIFTYBEES, 1.2 bps statutory friction).
- Tier 2: Foreign ADR-gated equity compounding (INFY.NS, HDFCBANK.NS, ICICIBANK.NS).
- Dynamic Trailing Stopgap (0.15% trail, 0.25% stop) with pre-open auction slippage handling.
- Idle Cash Yield Engine (TREPS / Liquid BeES auto-sweep at 6.8% annualized).
"""
from typing import Optional, Any, Dict
from dataclasses import dataclass
from loguru import logger
from ..domain.models import (
    ExecutionTier,
    InstrumentType,
    ADRLeadSignal,
    OvernightExecutionRecord,
    TaxSchedule
)

class TaxCalculator:
    """Calculates exact Indian statutory friction across instrument tiers with optional dynamic tax schedule."""
    @staticmethod
    def calculate(trade_value: float, instrument: InstrumentType, schedule: Optional[TaxSchedule] = None) -> float:
        """
        Computes statutory exchange friction:
        - ETF_CASH: 0.001% delivery STT + fees = 1.2 bps (0.012%) default
        - FUTURES: 0.02% sell STT + fees = 3.0 bps (0.030%) default
        - EQUITY_CASH: 0.10% buy + 0.10% sell STT + fees = 25.0 bps (0.250%) default
        - Custom schedules can be injected dynamically.
        """
        if schedule is not None:
            return schedule.calculate_tax(trade_value, instrument)

        if instrument == InstrumentType.ETF_CASH:
            return trade_value * 0.00012
        if instrument == InstrumentType.FUTURES:
            return trade_value * 0.00030
        return trade_value * 0.00250

class PreOpenAuctionRouter:
    """Routes limit orders during the NSE 09:00–09:08 pre-open call auction."""
    @staticmethod
    def route_auction_order(
        ticker: str,
        side: str,
        indicative_price: float,
        quantity: int,
        slippage_ceiling_bps: float = 15.0
    ) -> Dict[str, Any]:
        """
        Pins limit price to the equilibrium auction discovery with slippage bounds.
        Buy orders add slippage ceiling; sell orders subtract slippage floor.
        """
        slippage_pct = slippage_ceiling_bps / 10000.0
        if side.upper() == "BUY":
            limit_p = round(indicative_price * (1.0 + slippage_pct), 2)
        else:
            limit_p = round(indicative_price * (1.0 - slippage_pct), 2)

        return {
            "ticker": ticker,
            "side": side.upper(),
            "quantity": int(quantity),
            "indicative_price": float(indicative_price),
            "limit_price": float(limit_p),
            "order_type": "LIMIT",
            "time_in_force": "PRE_OPEN_AUCTION"
        }


class TrailingStopSimulator:
    """Simulates intraday trailing stop-loss execution with auction gap physics."""
    @staticmethod
    def resolve_exit(
        entry_price: float,
        open_price: float,
        high_price: float,
        low_price: float,
        trail_pct: float = 0.0015,
        stop_pct: float = 0.0025
    ) -> tuple[float, str]:
        """
        Resolves exit price and reason.
        Handles discontinuous opening auction gaps and intraday trailing wicks.
        """
        hard_stop_price = entry_price * (1.0 - stop_pct)

        # 1. Opening auction gap-down: orders fill at discontinuous auction open
        if open_price <= hard_stop_price:
            return open_price, "AUCTION_GAP_DOWN"

        # 2. Track morning peak for trailing buffer
        morning_peak = max(open_price, high_price)
        trailing_floor = morning_peak * (1.0 - trail_pct)
        effective_stop = max(hard_stop_price, trailing_floor)

        # 3. Trailing or hard stop triggered during session
        if low_price <= effective_stop:
            reason = "TRAILING_STOP_HIT" if effective_stop > hard_stop_price else "HARD_STOP_HIT"
            return effective_stop, reason

        # 4. Standard morning profitable exit at Open auction
        return open_price, "OPEN_AUCTION_EXIT"

    @staticmethod
    def resolve_decaying_exit(
        entry_price: float,
        open_price: float,
        high_price: float,
        low_price: float,
        minutes_elapsed: int = 0,
        initial_trail_pct: float = 0.0015,
        min_trail_pct: float = 0.0010,
        initial_stop_pct: float = 0.0025,
        min_stop_pct: float = 0.0015,
        decay_horizon_mins: float = 30.0
    ) -> tuple[float, str]:
        """
        Decays trailing stop as minutes elapse into the regular trading session.
        Tightens protection to lock in morning opening gap gains.
        """
        factor = min(1.0, max(0.0, float(minutes_elapsed) / decay_horizon_mins))
        curr_trail = initial_trail_pct - factor * (initial_trail_pct - min_trail_pct)
        curr_stop = initial_stop_pct - factor * (initial_stop_pct - min_stop_pct)
        return TrailingStopSimulator.resolve_exit(
            entry_price, open_price, high_price, low_price,
            trail_pct=curr_trail, stop_pct=curr_stop
        )

class IdleCashManager:
    """Simulates automated overnight collateral sweep into TREPS / Liquid BeES."""
    @staticmethod
    def earn_interest(cash_balance: float, days: int = 1, annual_yield: float = 0.068) -> float:
        """Computes daily risk-free interest on idle uninvested margin."""
        if cash_balance <= 0:
            return 0.0
        daily_rate = annual_yield / 365.0
        return float(cash_balance * daily_rate * days)

class OvernightExecutionEngine:
    """Autonomous Phase A execution coordinator for overnight momentum fleets."""
    def __init__(self, initial_capital: float = 10000.0, leverage: float = 1.0):
        self.capital = initial_capital
        self.leverage = leverage
        self.history: list[OvernightExecutionRecord] = []

    def execute_session(
        self,
        date_str: str,
        ticker: str,
        tier: ExecutionTier,
        instrument: InstrumentType,
        entry_price: float,
        open_price: float,
        high_price: float,
        low_price: float,
        adr_signal: Optional[ADRLeadSignal] = None,
        trail_pct: float = 0.0015,
        stop_pct: float = 0.0025,
        idle_yield_enabled: bool = True
    ) -> OvernightExecutionRecord:
        """
        Executes a single session following Tier rules.
        Tier 1 (NIFTYBEES): Unfiltered daily execution.
        Tier 2 (Single Equity): Gated by bullish ADR signal.
        """
        # Guard clause: Tier 2 requires bullish ADR signal
        if tier == ExecutionTier.TIER2_ADR_LEAD_EQUITY:
            if adr_signal is None or not adr_signal.is_bullish:
                interest = IdleCashManager.earn_interest(self.capital) if idle_yield_enabled else 0.0
                start_cap = self.capital
                self.capital += interest
                rec = OvernightExecutionRecord(
                    session_date=date_str,
                    ticker=ticker,
                    tier=tier,
                    instrument=instrument,
                    leverage=self.leverage,
                    entry_price=entry_price,
                    open_price=open_price,
                    high_price=high_price,
                    low_price=low_price,
                    exit_price=entry_price,
                    exit_reason="CASH_PRESERVED_ADR_RED",
                    gross_return_pct=0.0,
                    net_return_pct=round((interest / start_cap) * 100.0, 4) if start_cap > 0 else 0.0,
                    statutory_friction=0.0,
                    starting_balance=round(start_cap, 2),
                    ending_balance=round(self.capital, 2)
                )
                self.history.append(rec)
                return rec

        # Execute Trade
        start_cap = self.capital
        position_size = start_cap * self.leverage
        exit_price, exit_reason = TrailingStopSimulator.resolve_exit(
            entry_price=entry_price,
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            trail_pct=trail_pct,
            stop_pct=stop_pct
        )

        raw_price_ret = (exit_price - entry_price) / entry_price
        gross_profit = position_size * raw_price_ret
        friction = TaxCalculator.calculate(position_size, instrument)
        net_profit = gross_profit - friction

        self.capital = max(0.0, self.capital + net_profit)
        net_ret_pct = ((self.capital - start_cap) / start_cap) * 100.0 if start_cap > 0 else 0.0

        record = OvernightExecutionRecord(
            session_date=date_str,
            ticker=ticker,
            tier=tier,
            instrument=instrument,
            leverage=self.leverage,
            entry_price=round(entry_price, 2),
            open_price=round(open_price, 2),
            high_price=round(high_price, 2),
            low_price=round(low_price, 2),
            exit_price=round(exit_price, 2),
            exit_reason=exit_reason,
            gross_return_pct=round(raw_price_ret * 100.0, 3),
            net_return_pct=round(net_ret_pct, 3),
            statutory_friction=round(friction, 2),
            starting_balance=round(start_cap, 2),
            ending_balance=round(self.capital, 2)
        )
        self.history.append(record)
        return record
