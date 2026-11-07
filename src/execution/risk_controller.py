"""Statutory NSE Circuit Damper, ATR Sizing, and Micro-Capital Controller (Ponytail MVC)."""
import numpy as np
import pandas as pd
from ..domain.models import CircuitBands


class CircuitDamper:
    @staticmethod
    def calculate_bands(current_price: float, prev_close: float, band_pct: float = 0.10) -> CircuitBands:
        if not np.isfinite(current_price) or not np.isfinite(prev_close) or prev_close <= 0 or current_price <= 0:
            return CircuitBands(0.0, 0.0, 0.0, 0.0)
        lower, upper = round(prev_close * (1.0 - band_pct), 2), round(prev_close * (1.0 + band_pct), 2)
        dist_l = max(0.0, (current_price - lower) / lower) if lower > 0 else 1.0
        dist_u = max(0.0, (upper - current_price) / current_price) if current_price > 0 else 1.0
        return CircuitBands(lower, upper, dist_l, dist_u)


class PositionSizer:
    def __init__(self, total_capital: float = 50000.0, max_risk_pct: float = 1.0):
        self.total_capital, self.max_risk_pct = total_capital, max_risk_pct

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        h, l, c_prev = df["High"], df["Low"], df["Close_Raw"].shift(1)
        tr = np.maximum(h - l, np.maximum((h - c_prev).abs(), (l - c_prev).abs()))
        atr = tr.rolling(period).mean().iloc[-1]
        return float(atr) if np.isfinite(atr) else float(df["Close_Raw"].iloc[-1] * 0.02)

    def size_order(self, current_price: float, atr: float, circuit_factor: float = 1.0) -> tuple[int, float, float]:
        if not np.isfinite(current_price) or current_price <= 0 or not np.isfinite(atr) or atr < 0:
            return 0, 0.0, 0.0
        stop_dist = max(current_price * 0.015, 1.5 * atr)
        stop_loss, target = round(current_price - stop_dist, 2), round(current_price + 2.0 * stop_dist, 2)
        risk_per_share = current_price - stop_loss
        if risk_per_share <= 0:
            return 0, stop_loss, target

        raw_qty = int(((self.max_risk_pct / 100.0) * self.total_capital) // risk_per_share)
        max_shares = int((self.total_capital * 0.30) // current_price)
        return int(min(raw_qty, max_shares) * circuit_factor), stop_loss, target


class MicroCapitalController:
    """Capital Allocation & Crash Protection Controller tuned for ₹10,000 Bootstrapping (0% Leverage)."""
    def __init__(self, total_capital: float = 10000.0, max_drawdown_limit_pct: float = 3.5, buffer_reserve_pct: float = 0.02):
        self.total_capital = self.current_cash = self.peak_capital = float(total_capital)
        self.max_drawdown_limit_pct = max_drawdown_limit_pct
        self.buffer_reserve_pct = buffer_reserve_pct

    @property
    def current_drawdown_pct(self) -> float:
        return max(0.0, float((self.peak_capital - self.current_cash) / self.peak_capital * 100.0)) if self.peak_capital > 0 else 0.0

    def is_crash_protective_active(self, regime_id: int) -> tuple[bool, str]:
        if regime_id == -1:
            return True, "REGIME_CRASH_PROTECT_NOISE"
        if self.current_drawdown_pct >= self.max_drawdown_limit_pct:
            return True, f"MAX_DRAWDOWN_BREACHED_{self.current_drawdown_pct:.1f}PCT"
        return False, "REGIME_SAFE"

    def allocate(self, current_price: float, is_etf: bool = True, regime_id: int = 0, circuit_bands: CircuitBands | None = None) -> tuple[int, str]:
        if not np.isfinite(current_price) or current_price <= 0:
            return 0, "INVALID_PRICE"
        if circuit_bands and circuit_bands.is_locked:
            return 0, "CIRCUIT_LOCKED"
        is_active, reason = self.is_crash_protective_active(regime_id)
        if is_active:
            return 0, reason

        spend = min(self.current_cash * (1.0 - self.buffer_reserve_pct) * (1.0 if is_etf else 0.30), self.current_cash)
        units = int(spend // current_price) if spend > 0 else 0
        return (units, "ALLOCATED_OK") if units > 0 else (0, "INSUFFICIENT_INVESTABLE_CASH" if spend <= 0 else "UNIT_CAPITAL_DEFICIT")

    def update_balance(self, new_balance: float) -> None:
        self.current_cash = max(0.0, float(new_balance))
        self.peak_capital = max(self.peak_capital, self.current_cash)
