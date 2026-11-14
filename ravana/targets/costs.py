"""Indian Statutory Cash Delivery Friction and Flat DP Charge Drag Model."""

from typing import Optional


class DeliveryFrictionModel:
    """Calculates realistic Indian equity delivery transaction costs.

    Includes:
    - Variable statutory friction (STT, Exchange, SEBI, GST, Stamp Duty): 2.0 bps (0.02%).
    - Flat Depository Participant (CDSL) debit charge: Rs. 17.00 per sell transaction.
    """

    def __init__(
        self,
        variable_bps: float = 2.0,
        slippage_bps: float = 0.0,
        flat_dp_charge_inr: float = 17.0,
        default_capital_inr: float = 200000.0,
        min_capital_inr: float = 200000.0,
    ) -> None:
        self.variable_bps = variable_bps
        self.slippage_bps = slippage_bps
        self.flat_dp_charge_inr = flat_dp_charge_inr
        self.default_capital_inr = default_capital_inr
        self.min_capital_inr = min_capital_inr

    def compute_total_drag_bps(self, capital_inr: Optional[float] = None) -> float:
        """Compute total roundtrip drag in basis points for given capital."""
        cap = capital_inr or self.default_capital_inr
        fixed_drag_bps = (self.flat_dp_charge_inr / max(1000.0, cap)) * 10000.0
        return self.variable_bps + self.slippage_bps + fixed_drag_bps

    def apply_friction(
        self, gross_return: float, capital_inr: Optional[float] = None
    ) -> float:
        """Subtract total fractional drag from gross return."""
        drag_bps = self.compute_total_drag_bps(capital_inr)
        fractional_drag = drag_bps / 10000.0
        return float(gross_return - fractional_drag)

    def is_capital_viable(self, capital_inr: float) -> bool:
        """Check whether position sizing meets minimum institutional capital gate."""
        return capital_inr >= self.min_capital_inr


class IntradayFrictionModel:
    """Calculates realistic Indian intraday transaction costs.

    Under Indian tax law (Finance Act), for intraday trading:
    - STT on ETF is 0.0% (both buy and sell).
    - STT on equity is 0.025% (sell only = 2.5 bps).
    - CDSL DP debit charge is Rs 0.00 (shares never enter demat).
    - Exchange turnover + SEBI + Stamp Duty + Brokerage = default ~0.9 bps.
    """

    def __init__(
        self,
        variable_bps: float = 0.9,
        slippage_bps: float = 0.0,
        default_capital_inr: float = 200000.0,
        min_capital_inr: float = 200000.0,
        is_etf: bool = True,
    ) -> None:
        self.variable_bps = variable_bps
        self.slippage_bps = slippage_bps
        self.flat_dp_charge_inr = 0.0
        self.default_capital_inr = default_capital_inr
        self.min_capital_inr = min_capital_inr
        self.is_etf = is_etf

    def compute_total_drag_bps(self, capital_inr: Optional[float] = None) -> float:
        """Compute total roundtrip drag in basis points for given capital."""
        cap = capital_inr or self.default_capital_inr
        fixed_drag_bps = (self.flat_dp_charge_inr / max(1000.0, cap)) * 10000.0
        stt_bps = 0.0 if self.is_etf else 2.5  # 0.025% on sell
        return self.variable_bps + self.slippage_bps + fixed_drag_bps + stt_bps

    def apply_friction(
        self, gross_return: float, capital_inr: Optional[float] = None
    ) -> float:
        """Subtract total fractional drag from gross return."""
        drag_bps = self.compute_total_drag_bps(capital_inr)
        fractional_drag = drag_bps / 10000.0
        return float(gross_return - fractional_drag)

    def is_capital_viable(self, capital_inr: float) -> bool:
        """Check whether position sizing meets minimum institutional capital gate."""
        return capital_inr >= self.min_capital_inr
