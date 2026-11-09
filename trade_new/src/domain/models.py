"""
Domain Core: Immutable Value Objects and Domain Entities.
Pure logic, zero third-party framework dependencies.
"""
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional
from enum import Enum
import uuid

@dataclass(frozen=True)
class Ticker:
    """Represents a validated NSE financial instrument."""
    symbol: str
    sector: str

    def clean_name(self) -> str:
        return self.symbol.replace(".NS", "")

@dataclass(frozen=True)
class RegimeState:
    """Represents the continuous structural manifold regime for a session."""
    cluster_id: int
    is_noise: bool
    density_score: float
    forward_expectancy: float = 0.0

    @property
    def allows_trading(self) -> bool:
        """Noise states (-1) force 100% Cash / Liquid BeES capital preservation."""
        return not self.is_noise and self.cluster_id >= 0

@dataclass(frozen=True)
class CircuitBands:
    """Statutory NSE Exchange Price Bands & Proximity Damper Metrics."""
    lower_circuit: float
    upper_circuit: float
    distance_to_lower_pct: float
    distance_to_upper_pct: float

    @property
    def is_locked(self) -> bool:
        """True if price is within 1.0% of either circuit freeze limit."""
        return self.distance_to_lower_pct < 0.01 or self.distance_to_upper_pct < 0.01

    def allocation_damper_factor(self) -> float:
        """
        Calculates an exponential dampening factor [0.0, 1.0].
        If distance < 1.0%: 0.0 (Absolute lockout).
        If distance < 2.5%: quadratic dampening.
        """
        min_dist = min(self.distance_to_lower_pct, self.distance_to_upper_pct)
        if min_dist < 0.01:
            return 0.0
        elif min_dist < 0.025:
            return float((min_dist / 0.025) ** 2)
        return 1.0


class CircuitDamper:
    """NSE Statutory Circuit Band Calculator."""
    @staticmethod
    def calculate_bands(current_price: float, prev_close: float, band_pct: float = 0.10) -> CircuitBands:
        import numpy as np
        if not np.isfinite(current_price) or not np.isfinite(prev_close) or prev_close <= 0 or current_price <= 0:
            return CircuitBands(0.0, 0.0, 0.0, 0.0)
        lower = round(prev_close * (1.0 - band_pct), 2)
        upper = round(prev_close * (1.0 + band_pct), 2)
        dist_l = max(0.0, (current_price - lower) / lower) if lower > 0 else 1.0
        dist_u = max(0.0, (upper - current_price) / current_price) if current_price > 0 else 1.0
        return CircuitBands(lower, upper, dist_l, dist_u)


@dataclass(frozen=True)
class OrderSignal:
    """Structured simulated paper order signal."""
    ticker: str
    action: str  # 'BUY', 'SELL', 'HOLD_CASH'
    raw_price: float
    stop_loss: float
    target: float
    quantity: int
    regime_id: int
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def risk_reward_ratio(self) -> float:
        risk = abs(self.raw_price - self.stop_loss)
        reward = abs(self.target - self.raw_price)
        return round(reward / risk, 2) if risk > 0 else 0.0

class ExecutionTier(str, Enum):
    """Execution tier determining lead-lag gating and compounding rules."""
    TIER1_DAILY_ETF = "TIER1_DAILY_ETF"             # Continuous daily compounding, no foreign gate
    TIER2_ADR_LEAD_EQUITY = "TIER2_ADR_LEAD_EQUITY" # Foreign ADR green-gated single equity

class InstrumentType(str, Enum):
    """Statutory instrument class determining tax and leverage constraints."""
    ETF_CASH = "ETF_CASH"         # 0.001% STT delivery (1.2 bps default baseline)
    EQUITY_CASH = "EQUITY_CASH"   # 0.10% buy + 0.10% sell STT (25 bps total friction)
    MTF_LEVERAGED = "MTF_LEVERAGED" # 2x-4x broker margin + ~12% annual interest
    FUTURES = "FUTURES"           # 0.02% sell STT only (3 bps friction baseline)
    INTRADAY_EQUITY = "INTRADAY_EQUITY" # 0.025% sell STT + turnover (~3.5 bps)


def quantize_nse_tick(price: float, tick_size: Optional[float] = None) -> float:
    """Enforces NSE tick quantization: 5-paise (0.05) for securities >= 250, 1-paisa (0.01) for < 250."""
    if not isinstance(price, (int, float)) or price <= 0:
        return price
    if tick_size is None:
        tick_size = 0.05 if price >= 250.0 else 0.01
    return round(round(price / tick_size) * tick_size, 2)


@dataclass(frozen=True)
class TaxSchedule:
    """
    Configurable dynamic statutory tax and friction schedule (in basis points).
    Enables custom and dynamic parameterization across Indian Equities, ETFs, and Futures.
    1 bps = 0.0001 (0.01%).
    """
    etf_cash_bps: float = 1.2          # 0.00012 (0.012%) for ETF delivery
    futures_bps: float = 3.0           # 0.00030 (0.030%) for Single-Stock Futures
    equity_cash_bps: float = 25.0      # 0.00250 (0.250%) for Equity Delivery
    intraday_bps: float = 3.5          # 0.00035 (0.035%) for Intraday cash
    custom_slippage_bps: float = 0.0   # Optional market impact / slippage buffer
    dp_charge: float = 15.93           # Statutory CDSL DP fee (₹13.50 + 18% GST) on delivery sells

    def get_friction_rate(self, instrument: InstrumentType | str) -> float:
        """Returns the decimal friction rate (e.g., 0.00012 for 1.2 bps)."""
        inst = instrument.value if isinstance(instrument, InstrumentType) else str(instrument).upper()
        if inst in ("ETF_CASH", "ETF"):
            base = self.etf_cash_bps
        elif inst in ("FUTURES", "FUT"):
            base = self.futures_bps
        elif inst in ("INTRADAY_EQUITY", "INTRADAY"):
            base = self.intraday_bps
        else:
            base = self.equity_cash_bps
        return float((base + self.custom_slippage_bps) / 10000.0)

    def calculate_tax(
        self,
        trade_value: float,
        instrument: InstrumentType | str,
        is_sell: bool = False,
        is_delivery: bool = True
    ) -> float:
        """Calculates total friction in currency units with bipartite DP fee on delivery sells."""
        if trade_value <= 0:
            return 0.0
        friction = float(trade_value * self.get_friction_rate(instrument))
        inst = instrument.value if isinstance(instrument, InstrumentType) else str(instrument).upper()
        if is_sell and is_delivery and inst in ("ETF_CASH", "EQUITY_CASH", "ETF", "EQUITY"):
            friction += self.dp_charge
        return friction

@dataclass(frozen=True)
class ADRLeadSignal:
    """Overnight predictive telemetry from foreign listing / ADR."""
    foreign_ticker: str
    domestic_ticker: str
    foreign_return_pct: float
    is_bullish: bool
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass(frozen=True)
class OvernightExecutionRecord:
    """Audit record for overnight trade execution and trailing stop fill."""
    session_date: str
    ticker: str
    tier: ExecutionTier
    instrument: InstrumentType
    leverage: float
    entry_price: float
    open_price: float
    high_price: float
    low_price: float
    exit_price: float
    exit_reason: str
    gross_return_pct: float
    net_return_pct: float
    statutory_friction: float
    starting_balance: float
    ending_balance: float


class Currency(str, Enum):
    """Statutory ISO-4217 Currency representation for International Portfolios."""
    INR = "INR"
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    USDT = "USDT"


class LedgerEntryType(str, Enum):
    """Categorization of balance mutations in the multi-currency ledger."""
    DEPOSIT = "DEPOSIT"
    WITHDRAWAL = "WITHDRAWAL"
    TRADE_BUY = "TRADE_BUY"
    TRADE_SELL = "TRADE_SELL"
    FX_CONVERSION = "FX_CONVERSION"
    FEE = "FEE"
    FUNDING_RATE = "FUNDING_RATE"
    SETTLEMENT = "SETTLEMENT"


@dataclass(frozen=True)
class FXRate:
    """Spot or settlement exchange rate between two currency pairs."""
    from_currency: Currency
    to_currency: Currency
    rate: Decimal
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class LedgerEntry:
    """Immutable audit record for double-entry / multi-currency movements."""
    entry_id: str
    timestamp: datetime
    currency: Currency
    amount: Decimal
    entry_type: LedgerEntryType
    account_id: str = "DEFAULT"
    description: str = ""


@dataclass
class MultiCurrencyAccount:
    """Sub-account with isolated currency balances preventing contagion across asset classes."""
    account_id: str
    base_currency: Currency
    is_isolated: bool = True
    balances: dict[Currency, Decimal] = field(default_factory=dict)

    def get_balance(self, currency: Currency) -> Decimal:
        return self.balances.get(currency, Decimal("0.0"))

    def set_balance(self, currency: Currency, amount: Decimal) -> None:
        self.balances[currency] = amount


@dataclass
class MultiCurrencyLedger:
    """
    Multi-Currency Ledger supporting FX risk isolation, perpetual funding fee
    settlement, and cross-border currency valuation.
    """
    accounts: dict[str, MultiCurrencyAccount] = field(default_factory=dict)
    entries: list[LedgerEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        if "DEFAULT" not in self.accounts:
            self.accounts["DEFAULT"] = MultiCurrencyAccount(
                account_id="DEFAULT",
                base_currency=Currency.INR,
                is_isolated=False
            )

    def get_or_create_account(
        self, account_id: str, base_currency: Currency = Currency.INR, is_isolated: bool = True
    ) -> MultiCurrencyAccount:
        """Retrieves or registers a dedicated isolated sub-account."""
        if account_id not in self.accounts:
            self.accounts[account_id] = MultiCurrencyAccount(
                account_id=account_id,
                base_currency=base_currency,
                is_isolated=is_isolated
            )
        return self.accounts[account_id]

    def get_balance(self, currency: Currency, account_id: str = "DEFAULT") -> Decimal:
        """Fetch cash balance for an account and currency."""
        acct = self.accounts.get(account_id)
        if acct is None:
            return Decimal("0.0")
        return acct.get_balance(currency)

    def deposit(
        self, currency: Currency, amount: Decimal, account_id: str = "DEFAULT", description: str = ""
    ) -> LedgerEntry:
        """Credit cash balance to an isolated account."""
        if amount <= Decimal("0.0"):
            raise ValueError(f"Deposit amount must be strictly positive, got {amount}")
        acct = self.get_or_create_account(account_id)
        current = acct.get_balance(currency)
        acct.set_balance(currency, current + amount)

        entry = LedgerEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            currency=currency,
            amount=amount,
            entry_type=LedgerEntryType.DEPOSIT,
            account_id=account_id,
            description=description or f"Deposit {amount} {currency.value}"
        )
        self.entries.append(entry)
        return entry

    def withdraw(
        self, currency: Currency, amount: Decimal, account_id: str = "DEFAULT", description: str = ""
    ) -> LedgerEntry:
        """Debit cash balance from an isolated account with solvency guard."""
        if amount <= Decimal("0.0"):
            raise ValueError(f"Withdrawal amount must be strictly positive, got {amount}")
        acct = self.get_or_create_account(account_id)
        current = acct.get_balance(currency)
        if current < amount:
            raise ValueError(f"Insufficient funds: {current} {currency.value} < {amount}")

        acct.set_balance(currency, current - amount)
        entry = LedgerEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            currency=currency,
            amount=-amount,
            entry_type=LedgerEntryType.WITHDRAWAL,
            account_id=account_id,
            description=description or f"Withdrawal {amount} {currency.value}"
        )
        self.entries.append(entry)
        return entry

    def convert_fx(
        self,
        from_currency: Currency,
        to_currency: Currency,
        amount: Decimal,
        fx_rate: Decimal,
        fee: Decimal = Decimal("0.0"),
        account_id: str = "DEFAULT"
    ) -> tuple[LedgerEntry, LedgerEntry]:
        """Atomic FX conversion with exchange rate and transaction fee."""
        if from_currency == to_currency:
            raise ValueError("Cannot convert currency to itself")
        if fx_rate <= Decimal("0.0"):
            raise ValueError(f"FX rate must be positive, got {fx_rate}")

        acct = self.get_or_create_account(account_id)
        from_bal = acct.get_balance(from_currency)
        total_debit = amount + fee
        if from_bal < total_debit:
            raise ValueError(f"Insufficient funds for FX: {from_bal} < {total_debit}")

        received_amount = amount * fx_rate
        acct.set_balance(from_currency, from_bal - total_debit)
        acct.set_balance(to_currency, acct.get_balance(to_currency) + received_amount)

        now = datetime.now()
        debit_entry = LedgerEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=now,
            currency=from_currency,
            amount=-total_debit,
            entry_type=LedgerEntryType.FX_CONVERSION,
            account_id=account_id,
            description=f"FX convert {amount} {from_currency.value} to {to_currency.value} (fee: {fee})"
        )
        credit_entry = LedgerEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=now,
            currency=to_currency,
            amount=received_amount,
            entry_type=LedgerEntryType.FX_CONVERSION,
            account_id=account_id,
            description=f"FX received {received_amount} {to_currency.value} @ {fx_rate}"
        )
        self.entries.extend([debit_entry, credit_entry])
        return debit_entry, credit_entry

    def apply_perpetual_funding(
        self,
        account_id: str,
        currency: Currency,
        position_notional: Decimal,
        funding_rate_8h: Decimal,
        description: str = ""
    ) -> LedgerEntry:
        """
        Settles 8-hour perpetual funding payments for continuous crypto markets.
        Funding Payment = Position Notional * 8h Funding Rate.
        Positive funding rate: Long pays short (negative cash flow for long).
        """
        acct = self.get_or_create_account(account_id)
        funding_payment = position_notional * funding_rate_8h
        current_bal = acct.get_balance(currency)
        acct.set_balance(currency, current_bal - funding_payment)

        entry = LedgerEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            currency=currency,
            amount=-funding_payment,
            entry_type=LedgerEntryType.FUNDING_RATE,
            account_id=account_id,
            description=description or f"Perp 8h funding payment (rate={funding_rate_8h})"
        )
        self.entries.append(entry)
        return entry

    def get_total_valuation(
        self,
        base_currency: Currency,
        fx_rates: dict[tuple[Currency, Currency], Decimal],
        account_id: Optional[str] = None
    ) -> Decimal:
        """Calculates portfolio cash valuation converted into target base currency."""
        target_accounts = [self.accounts[account_id]] if account_id else list(self.accounts.values())
        total = Decimal("0.0")

        for acct in target_accounts:
            for curr, bal in acct.balances.items():
                if bal == Decimal("0.0"):
                    continue
                if curr == base_currency:
                    total += bal
                else:
                    pair = (curr, base_currency)
                    if pair not in fx_rates:
                        raise ValueError(f"Missing FX rate for conversion pair {pair}")
                    total += bal * fx_rates[pair]
        return total

    def get_unhedged_fx_exposure(
        self,
        base_currency: Currency,
        account_id: Optional[str] = None
    ) -> dict[Currency, Decimal]:
        """Returns exposure by currency to detect and isolate foreign exchange risk."""
        target_accounts = [self.accounts[account_id]] if account_id else list(self.accounts.values())
        exposure: dict[Currency, Decimal] = {}

        for acct in target_accounts:
            for curr, bal in acct.balances.items():
                if curr == base_currency or bal == Decimal("0.0"):
                    continue
                exposure[curr] = exposure.get(curr, Decimal("0.0")) + bal
        return exposure

