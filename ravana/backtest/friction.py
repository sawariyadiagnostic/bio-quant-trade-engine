"""Institutional Indian statutory tax and friction ledger for NSE equity and ETFs."""
from __future__ import annotations

from dataclasses import dataclass
from ravana.backtest.config import AssetType, TradingRegime


@dataclass(frozen=True)
class FrictionBreakdown:
    """Exact rupee breakdown of all statutory and clearing charges for a trade."""
    turnover_inr: float
    stt_inr: float
    exchange_charges_inr: float
    sebi_charges_inr: float
    stamp_duty_inr: float
    gst_inr: float
    brokerage_inr: float
    dp_charges_inr: float
    total_friction_inr: float
    effective_bps: float


class IndianFrictionModel:
    """Computes exact rupee statutory taxes, SEBI fees, and clearing costs for Indian trades."""

    EXCHANGE_TURNOVER_RATE = 0.0000345  # 0.00345% NSE turnover fee
    SEBI_RATE = 0.0000010               # ₹10 per crore (0.0001%)
    GST_RATE = 0.18                     # 18% GST on brokerage + transaction + SEBI fees

    @classmethod
    def calculate_friction(
        cls,
        side: str,
        price: float,
        shares: int,
        asset_type: AssetType = AssetType.ETF,
        regime: TradingRegime = TradingRegime.OVERNIGHT,
        fixed_dp_fee_inr: float = 17.0,
        brokerage_per_order_inr: float = 0.0,
    ) -> FrictionBreakdown:
        """Calculate complete line-item friction for a buy or sell transaction."""
        side = side.upper()
        turnover = float(price * shares)
        if turnover <= 0:
            return cls._empty_breakdown()

        is_delivery = regime in (TradingRegime.OVERNIGHT, TradingRegime.SWING)
        is_buy = side == "BUY"

        stt = cls._calc_stt(turnover, is_buy, is_delivery, asset_type)
        stamp_duty = cls._calc_stamp_duty(turnover, is_buy, is_delivery)
        exch_fee = turnover * cls.EXCHANGE_TURNOVER_RATE
        sebi_fee = turnover * cls.SEBI_RATE
        brokerage = cls._calc_brokerage(turnover, is_delivery, brokerage_per_order_inr)
        gst = (brokerage + exch_fee + sebi_fee) * cls.GST_RATE
        dp_fee = fixed_dp_fee_inr if (is_delivery and not is_buy) else 0.0

        total_friction = stt + stamp_duty + exch_fee + sebi_fee + brokerage + gst + dp_fee
        effective_bps = (total_friction / turnover) * 10000.0

        return FrictionBreakdown(
            turnover_inr=round(turnover, 2),
            stt_inr=round(stt, 2),
            exchange_charges_inr=round(exch_fee, 2),
            sebi_charges_inr=round(sebi_fee, 2),
            stamp_duty_inr=round(stamp_duty, 2),
            gst_inr=round(gst, 2),
            brokerage_inr=round(brokerage, 2),
            dp_charges_inr=round(dp_fee, 2),
            total_friction_inr=round(total_friction, 2),
            effective_bps=round(effective_bps, 2),
        )

    @classmethod
    def _calc_stt(cls, turnover: float, is_buy: bool, is_delivery: bool, asset_type: AssetType) -> float:
        """Compute Securities Transaction Tax according to Central Board of Direct Taxes rules."""
        if is_delivery:
            if asset_type == AssetType.ETF:
                return turnover * 0.00001 if not is_buy else 0.0  # 0.001% on sell only
            elif asset_type == AssetType.EQUITY_STOCK:
                return turnover * 0.001                            # 0.1% on buy and sell
            return 0.0
        else:
            return turnover * 0.00025 if not is_buy else 0.0      # 0.025% on intraday sell

    @classmethod
    def _calc_stamp_duty(cls, turnover: float, is_buy: bool, is_delivery: bool) -> float:
        """Stamp duty is levied only on buyers under the Indian Stamp Act."""
        if not is_buy:
            return 0.0
        return turnover * 0.00015 if is_delivery else turnover * 0.00003

    @classmethod
    def _calc_brokerage(cls, turnover: float, is_delivery: bool, per_order_cap: float) -> float:
        """Compute discount brokerage with standard ₹20 cap."""
        if is_delivery and per_order_cap == 0.0:
            return 0.0
        return min(20.0, max(per_order_cap, turnover * 0.0005))

    @staticmethod
    def _empty_breakdown() -> FrictionBreakdown:
        return FrictionBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
