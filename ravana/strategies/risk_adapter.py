"""Muscle 18 Interlock, Risk-at-Stop Leveraged Sizing & HRP Correlation Bridge.

Connects alpha execution strategies with RAVANA's portfolio risk safeguards:
1. Muscle 18 Circuit Breaker & Kill Switch interlock (blocks BUY, permits exits).
2. Institutional Risk-at-Stop sizing (0.25% risk cap, 2x MIS leverage, 0.98x CNC cash cap).
3. Hard physical notional floor (>= ₹50,000 to cap DP fee drag <= 3.0 bps).
4. HRP sector correlation ceiling (caps pairwise correlated assets > 0.80 to 40% equity).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from ravana.risk.circuit_breaker import CircuitBreakerKillSwitch, CircuitBreakerStatus


@dataclass(frozen=True)
class SizingResult:
    """Output contract for Institutional Risk-at-Stop Leveraged Sizing."""

    shares: int
    notional: float
    risk_at_stop: float
    risk_pct: float
    max_leverage_allowed: float
    leverage_applied: float
    is_approved: bool
    rejection_reason: str = ""


@dataclass(frozen=True)
class CorrelationCheckResult:
    """Telemetry emitted by HRP Sector Correlation Ceiling verification."""

    is_approved: bool
    candidate_ticker: str
    correlated_tickers: List[str]
    current_cluster_notional: float
    proposed_total_notional: float
    cluster_allocation_pct: float
    max_allowed_notional: float
    reason: str


@dataclass(frozen=True)
class RiskBridgeDecision:
    """Consolidated risk approval ticket evaluated prior to broker dispatch."""

    is_allowed: bool
    action: str  # 'PERMITTED' | 'REJECTED'
    order_side: str
    ticker: str
    shares: int
    notional: float
    reason: str
    sizing: Optional[SizingResult] = None
    correlation: Optional[CorrelationCheckResult] = None


class StrategyRiskBridge:
    """Institutional Risk Bridge bridging execution strategies with Muscle 18 & Risk Controls."""

    LIQUIDATION_SIDES = frozenset({
        "SELL",
        "EXIT_ALL",
        "EXIT_TRANCHE_1",
        "SCALE_OUT",
        "LIQUIDATE",
    })

    BUY_SIDES = frozenset({"BUY", "LONG", "ENTER"})

    def __init__(
        self,
        kill_switch: Optional[CircuitBreakerKillSwitch] = None,
        risk_budget_pct: float = 0.0025,             # 0.25% equity risk per trade (₹500 / ₹2L)
        max_mis_leverage: float = 2.0,               # 2.0x maximum leverage for INTRADAY_MIS
        max_cnc_leverage: float = 0.98,              # 0.98x cash ceiling for OVERNIGHT_CNC
        min_trade_notional: float = 50000.0,         # Hard physical notional floor (<= 3.0 bps DP fee)
        correlation_threshold: float = 0.80,         # Pairwise correlation ceiling threshold
        max_correlated_allocation_pct: float = 0.40, # 40% capital ceiling across correlated cluster
        default_slippage_bps: float = 5.0,           # 5.0 bps default modeled slippage
    ) -> None:
        self.kill_switch = kill_switch or CircuitBreakerKillSwitch()
        self.risk_budget_pct = float(risk_budget_pct)
        self.max_mis_leverage = float(max_mis_leverage)
        self.max_cnc_leverage = float(max_cnc_leverage)
        self.min_trade_notional = float(min_trade_notional)
        self.correlation_threshold = float(correlation_threshold)
        self.max_correlated_allocation_pct = float(max_correlated_allocation_pct)
        self.default_slippage_bps = float(default_slippage_bps)

    def validate_order_gate(
        self,
        side: str,
        breaker_status: Optional[CircuitBreakerStatus] = None,
    ) -> Tuple[bool, str]:
        """Interlock order dispatch against Muscle 18 Circuit Breakers and Kill Switch.

        Strictly blocks BUY orders if circuit breaker tripped or kill switch engaged.
        Explicitly permits liquidation and scale-out orders (SELL, EXIT_ALL, EXIT_TRANCHE_1).
        """
        norm_side = str(side).upper().strip()
        is_blocked = False
        block_reason = ""

        if self.kill_switch.is_killed:
            is_blocked = True
            block_reason = f"Kill Switch engaged: {getattr(self.kill_switch, '_kill_reason', 'Emergency Halt')}"
        elif breaker_status is not None and not breaker_status.is_trading_allowed:
            is_blocked = True
            block_reason = f"Circuit Breaker tripped: {breaker_status.reason}"

        if not is_blocked:
            return True, "Trading allowed: circuit breaker and kill switch green."

        if norm_side in self.LIQUIDATION_SIDES:
            return (
                True,
                f"Liquidation order '{norm_side}' explicitly permitted under active Circuit Breaker.",
            )

        if norm_side in self.BUY_SIDES:
            return False, f"BUY order strictly blocked by Circuit Breaker: {block_reason}"

        return False, f"Unrecognized order side '{norm_side}' blocked under active Circuit Breaker."

    def calculate_risk_at_stop_size(
        self,
        ticker: str,
        entry_price: float,
        stop_loss_price: float,
        account_equity: float,
        product_type: str = "MIS",
        expected_slippage_inr: Optional[float] = None,
    ) -> SizingResult:
        """Institutional Risk-at-Stop Leveraged Sizing Engine.

        Formula:
            Shares = min( (0.0025 * Equity) / (|P_entry - S_0| + Slippage),
                          (Leverage_max * Equity) / P_entry )

        Enforces:
        - 0.25% equity risk cap at stop loss.
        - 2.0x max leverage for INTRADAY_MIS; 0.98x cash cap for OVERNIGHT_CNC.
        - Hard physical minimum notional >= ₹50,000 (caps DP fee drag <= 3.0 bps).
        """
        valid_input, err = self._validate_sizing_inputs(entry_price, stop_loss_price, account_equity)
        if not valid_input:
            return SizingResult(0, 0.0, 0.0, 0.0, 0.0, 0.0, False, err)

        leverage_cap = self._get_leverage_cap(product_type)
        stop_dist = abs(entry_price - stop_loss_price)
        slippage = self._get_slippage_buffer(entry_price, expected_slippage_inr)
        total_risk_per_share = stop_dist + slippage

        if total_risk_per_share <= 0:
            return SizingResult(0, 0.0, 0.0, 0.0, leverage_cap, 0.0, False, "Total risk per share <= 0")

        risk_budget = self.risk_budget_pct * account_equity
        shares_by_risk = risk_budget / total_risk_per_share
        shares_by_leverage = (leverage_cap * account_equity) / entry_price

        raw_shares = min(shares_by_risk, shares_by_leverage)
        shares = int(np.floor(raw_shares))

        if shares <= 0:
            return SizingResult(0, 0.0, 0.0, 0.0, leverage_cap, 0.0, False, "Sizing resolved to 0 shares")

        notional = shares * entry_price
        if notional < self.min_trade_notional:
            return SizingResult(
                0,
                round(notional, 2),
                round(shares * total_risk_per_share, 2),
                round((shares * total_risk_per_share) / account_equity, 6),
                leverage_cap,
                0.0,
                False,
                f"Notional ₹{notional:,.2f} below minimum ₹{self.min_trade_notional:,.2f} (DP fee drag > 3.0 bps)",
            )

        actual_risk = shares * total_risk_per_share
        return SizingResult(
            shares=shares,
            notional=round(notional, 2),
            risk_at_stop=round(actual_risk, 2),
            risk_pct=round(actual_risk / account_equity, 6),
            max_leverage_allowed=leverage_cap,
            leverage_applied=round(notional / account_equity, 4),
            is_approved=True,
            rejection_reason="",
        )

    def check_hrp_correlation_ceiling(
        self,
        candidate_ticker: str,
        candidate_notional: float,
        current_holdings_notional: Dict[str, float],
        correlation_matrix: Union[pd.DataFrame, Dict[str, Dict[str, float]], np.ndarray],
        account_equity: float,
        asset_names: Optional[Sequence[str]] = None,
    ) -> CorrelationCheckResult:
        """HRP Sector Correlation Ceiling Guard.

        Caps maximum capital allocation to 40% across pairwise correlated assets
        (correlation > 0.80, e.g. BANKBEES + PSUBNKBEES).
        """
        if account_equity <= 0:
            return CorrelationCheckResult(
                is_approved=False,
                candidate_ticker=candidate_ticker,
                correlated_tickers=[],
                current_cluster_notional=0.0,
                proposed_total_notional=candidate_notional,
                cluster_allocation_pct=0.0,
                max_allowed_notional=0.0,
                reason="Account equity must be positive.",
            )

        corr_series = self._extract_correlation_series(
            candidate_ticker, correlation_matrix, asset_names
        )
        correlated = self._find_correlated_assets(
            candidate_ticker, corr_series, current_holdings_notional
        )

        cluster_existing_notional = sum(
            float(current_holdings_notional.get(sym, 0.0))
            for sym in correlated
            if sym != candidate_ticker
        )
        # Also include existing notional in candidate ticker itself if already held
        existing_candidate_notional = float(current_holdings_notional.get(candidate_ticker, 0.0))
        cluster_existing_notional += existing_candidate_notional

        proposed_cluster_notional = cluster_existing_notional + candidate_notional
        max_allowed_cluster = self.max_correlated_allocation_pct * account_equity
        headroom = max(0.0, max_allowed_cluster - cluster_existing_notional)
        cluster_pct = proposed_cluster_notional / account_equity

        if proposed_cluster_notional > max_allowed_cluster:
            all_cluster_syms = sorted(set(correlated + [candidate_ticker]))
            return CorrelationCheckResult(
                is_approved=False,
                candidate_ticker=candidate_ticker,
                correlated_tickers=all_cluster_syms,
                current_cluster_notional=round(cluster_existing_notional, 2),
                proposed_total_notional=round(proposed_cluster_notional, 2),
                cluster_allocation_pct=round(cluster_pct, 4),
                max_allowed_notional=round(headroom, 2),
                reason=(
                    f"HRP Correlation Ceiling ({self.max_correlated_allocation_pct * 100:.1f}%) breached: "
                    f"Cluster notional ₹{proposed_cluster_notional:,.2f} ({cluster_pct * 100:.1f}%) "
                    f"exceeds ceiling ₹{max_allowed_cluster:,.2f} across {all_cluster_syms}."
                ),
            )

        all_cluster_syms = sorted(set(correlated + [candidate_ticker]))
        return CorrelationCheckResult(
            is_approved=True,
            candidate_ticker=candidate_ticker,
            correlated_tickers=all_cluster_syms,
            current_cluster_notional=round(cluster_existing_notional, 2),
            proposed_total_notional=round(proposed_cluster_notional, 2),
            cluster_allocation_pct=round(cluster_pct, 4),
            max_allowed_notional=round(headroom, 2),
            reason="HRP correlation ceiling cleared.",
        )

    def evaluate_order_proposal(
        self,
        ticker: str,
        side: str,
        entry_price: float,
        stop_loss_price: float,
        account_equity: float,
        product_type: str = "MIS",
        current_holdings_notional: Optional[Dict[str, float]] = None,
        correlation_matrix: Optional[Union[pd.DataFrame, Dict[str, Dict[str, float]]]] = None,
        breaker_status: Optional[CircuitBreakerStatus] = None,
        expected_slippage_inr: Optional[float] = None,
    ) -> RiskBridgeDecision:
        """Evaluate full risk gate chain: Breaker interlock, Leveraged sizing, and HRP ceiling."""
        gate_allowed, gate_reason = self.validate_order_gate(side, breaker_status)
        norm_side = str(side).upper().strip()

        if not gate_allowed:
            return RiskBridgeDecision(
                is_allowed=False,
                action="REJECTED",
                order_side=norm_side,
                ticker=ticker,
                shares=0,
                notional=0.0,
                reason=gate_reason,
            )

        if norm_side in self.LIQUIDATION_SIDES:
            return RiskBridgeDecision(
                is_allowed=True,
                action="PERMITTED",
                order_side=norm_side,
                ticker=ticker,
                shares=0,
                notional=0.0,
                reason=gate_reason,
            )

        sizing = self.calculate_risk_at_stop_size(
            ticker=ticker,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            account_equity=account_equity,
            product_type=product_type,
            expected_slippage_inr=expected_slippage_inr,
        )

        if not sizing.is_approved:
            return RiskBridgeDecision(
                is_allowed=False,
                action="REJECTED",
                order_side=norm_side,
                ticker=ticker,
                shares=0,
                notional=0.0,
                reason=f"Sizing rejected: {sizing.rejection_reason}",
                sizing=sizing,
            )

        corr_res: Optional[CorrelationCheckResult] = None
        if correlation_matrix is not None and current_holdings_notional is not None:
            corr_res = self.check_hrp_correlation_ceiling(
                candidate_ticker=ticker,
                candidate_notional=sizing.notional,
                current_holdings_notional=current_holdings_notional,
                correlation_matrix=correlation_matrix,
                account_equity=account_equity,
            )
            if not corr_res.is_approved:
                return RiskBridgeDecision(
                    is_allowed=False,
                    action="REJECTED",
                    order_side=norm_side,
                    ticker=ticker,
                    shares=0,
                    notional=0.0,
                    reason=f"Correlation ceiling rejected: {corr_res.reason}",
                    sizing=sizing,
                    correlation=corr_res,
                )

        return RiskBridgeDecision(
            is_allowed=True,
            action="PERMITTED",
            order_side=norm_side,
            ticker=ticker,
            shares=sizing.shares,
            notional=sizing.notional,
            reason="All risk gates, leveraged sizing, and correlation ceilings cleared.",
            sizing=sizing,
            correlation=corr_res,
        )

    # -------------------------------------------------------------------------
    # Helper & Validation Subroutines (< 50 lines each)
    # -------------------------------------------------------------------------

    def _validate_sizing_inputs(
        self, entry_price: float, stop_loss_price: float, equity: float
    ) -> Tuple[bool, str]:
        """Validate numeric sizing inputs for finite positive numbers."""
        if not np.isfinite(entry_price) or entry_price <= 0:
            return False, f"Invalid entry price: {entry_price}"
        if not np.isfinite(stop_loss_price) or stop_loss_price <= 0:
            return False, f"Invalid stop loss price: {stop_loss_price}"
        if not np.isfinite(equity) or equity <= 0:
            return False, f"Invalid account equity: {equity}"
        return True, ""

    def _get_leverage_cap(self, product_type: str) -> float:
        """Resolve maximum permitted leverage based on product type."""
        pt = str(product_type).upper().strip()
        if pt in {"MIS", "INTRADAY", "INTRADAY_MIS"}:
            return self.max_mis_leverage
        return self.max_cnc_leverage

    def _get_slippage_buffer(
        self, entry_price: float, expected_slippage_inr: Optional[float]
    ) -> float:
        """Compute per-share slippage buffer."""
        if expected_slippage_inr is not None and expected_slippage_inr >= 0:
            return float(expected_slippage_inr)
        return float(entry_price * (self.default_slippage_bps / 10000.0))

    def _extract_correlation_series(
        self,
        ticker: str,
        corr_matrix: Union[pd.DataFrame, Dict[str, Dict[str, float]], np.ndarray],
        asset_names: Optional[Sequence[str]],
    ) -> Dict[str, float]:
        """Normalize various correlation matrix representations into a ticker map."""
        if isinstance(corr_matrix, pd.DataFrame):
            if ticker in corr_matrix.columns:
                return corr_matrix[ticker].to_dict()
            return {ticker: 1.0}
        if isinstance(corr_matrix, dict):
            if ticker in corr_matrix:
                return corr_matrix[ticker]
            return {ticker: 1.0}
        if isinstance(corr_matrix, np.ndarray) and asset_names is not None:
            names = list(asset_names)
            if ticker in names:
                idx = names.index(ticker)
                return {names[j]: float(corr_matrix[idx, j]) for j in range(len(names))}
        return {ticker: 1.0}

    def _find_correlated_assets(
        self,
        candidate_ticker: str,
        corr_series: Dict[str, float],
        holdings: Dict[str, float],
    ) -> List[str]:
        """Identify assets currently held that exceed the correlation ceiling threshold."""
        correlated: List[str] = []
        for held_sym, held_notional in holdings.items():
            if held_notional <= 0:
                continue
            corr_val = corr_series.get(held_sym, 0.0)
            if held_sym == candidate_ticker or corr_val > self.correlation_threshold:
                correlated.append(held_sym)
        return correlated
