"""Decision Risk Shield for RAVANA Quantitative Engine.

Integrates Gemini Decision API categorical routing and scoring with
institutional risk management gates:
1. Pre-Dump Exit Shield (14:35 - 14:50 IST retail square-off cascade protection).
2. Adaptive Volatility Regime Stop-Loss Calculation (Tight, Normal, Wide, Shock).
3. Overnight Continuation Gate at morning open (09:15 IST).
4. Relational ACID audit logging with SQLiteStateLedger.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Tuple
import uuid

import pandas as pd

from ravana.decisions.client import GeminiDecisionClient
from ravana.decisions.schema import BatchResponse, DecisionResult, DecisionType, Question
from ravana.strategies.state_ledger import DecisionAuditRecord, SQLiteStateLedger

logger = logging.getLogger(__name__)


class DecisionRiskShield:
    """Institutional Risk Shield powered by Gemini categorical decisions."""

    def __init__(
        self,
        client: Optional[GeminiDecisionClient] = None,
        ledger: Optional[SQLiteStateLedger] = None,
    ) -> None:
        """Initialize DecisionRiskShield with optional client and state ledger."""
        self.client = client if client is not None else GeminiDecisionClient()
        self.ledger = ledger

    def _is_pre_dump_window(self, ts: pd.Timestamp) -> bool:
        """Check if timestamp falls strictly within the 14:35 to 14:50 IST window."""
        if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
            try:
                t_str = ts.tz_convert("Asia/Kolkata").strftime("%H:%M")
            except Exception:
                t_str = ts.strftime("%H:%M")
        else:
            t_str = ts.strftime("%H:%M")
        return "14:35" <= t_str <= "14:50"

    def _record_audit(
        self,
        ticker: str,
        cluster_id: str,
        timeframe: str,
        res: Optional[DecisionResult],
        status: str,
        latency_ms: float,
        ts: Optional[str] = None,
    ) -> None:
        """Record a single decision result into the ACID state ledger."""
        if self.ledger is None or res is None:
            return

        record_ts = ts or datetime.now(timezone.utc).isoformat()
        prob_or_score = (
            res.probability
            if res.probability is not None
            else (float(res.score) if res.score is not None else None)
        )
        rec = DecisionAuditRecord(
            decision_id=f"{cluster_id}_{ticker}_{res.key}_{uuid.uuid4().hex[:8]}",
            timestamp=record_ts,
            timeframe=timeframe,
            cluster_id=cluster_id,
            symbol=ticker,
            question_key=res.key,
            selected_choice=res.selected_choice or (str(res.score) if res.score is not None else None),
            probability_score=prob_or_score,
            latency_ms=latency_ms,
            status=status,
        )
        self.ledger.record_decision(rec)

    def evaluate_pre_dump_exit(
        self,
        timestamp: pd.Timestamp,
        active_positions: List[Dict[str, Any]],
        current_bars: Dict[str, pd.Series],
    ) -> List[Tuple[str, str]]:
        """Evaluate retail MIS square-off dump threat during 14:35 to 14:50 IST.

        Returns list of (ticker, 'PRE_DUMP_SHIELD_EXIT') if exit is recommended,
        otherwise returns an empty list.
        """
        if not self._is_pre_dump_window(timestamp) or not active_positions:
            return []

        exits: List[Tuple[str, str]] = []

        for pos in active_positions:
            ticker = pos.get("ticker") or pos.get("symbol")
            if not ticker:
                continue

            bar = current_bars.get(ticker)
            bar_dict = (
                bar.to_dict()
                if (bar is not None and hasattr(bar, "to_dict"))
                else (dict(bar) if bar is not None else {})
            )
            context = (
                f"Pre-dump risk assessment at {timestamp} for {ticker}. "
                f"Active Position: {pos}. Current Bar: {bar_dict}"
            )
            questions = [
                Question(
                    key="retail_mis_dump_intensity",
                    question="Assess retail MIS square-off dump intensity (0=None, 1=Mild, 2=Moderate, 3=Severe).",
                    decision_type=DecisionType.SCORE,
                    score_range=(0, 3),
                ),
                Question(
                    key="should_force_exit_at_1440",
                    question="Should the intraday position be force-exited before the 14:40 broker square-off cascade?",
                    decision_type=DecisionType.NOUL,
                ),
            ]

            resp: BatchResponse = self.client.evaluate(context, questions)
            score_res = resp.results.get("retail_mis_dump_intensity")
            noul_res = resp.results.get("should_force_exit_at_1440")

            score = score_res.score if (score_res and score_res.score is not None) else 0
            prob = noul_res.probability if (noul_res and noul_res.probability is not None) else 0.0

            is_exit = (prob >= 0.65) or (score >= 2)
            audit_status = "APPLIED" if is_exit else "BYPASSED"

            if is_exit:
                exits.append((ticker, "PRE_DUMP_SHIELD_EXIT"))

            ts_str = str(timestamp)
            self._record_audit(ticker, "pre_dump_shield", "5m", score_res, audit_status, resp.latency_ms, ts_str)
            self._record_audit(ticker, "pre_dump_shield", "5m", noul_res, audit_status, resp.latency_ms, ts_str)

        return exits

    def compute_adaptive_stop(
        self,
        ticker: str,
        entry_price: float,
        default_stop: float,
        bar: pd.Series,
    ) -> float:
        """Compute regime-adjusted stop loss using Gemini volatility categorization.

        Regimes:
        - 0 (Tight): Low noise, tightens stop towards entry.
        - 1 (Normal): Normal market chop, keeps default stop.
        - 2 (Wide): Expanded noise, widens stop distance.
        - 3 (Shock): Volatility shock, flattens position to current close.
        """
        bar_dict = bar.to_dict() if hasattr(bar, "to_dict") else dict(bar)
        context = (
            f"Adaptive stop calculation for {ticker}. "
            f"Entry: {entry_price}, Default Stop: {default_stop}, Bar: {bar_dict}"
        )
        questions = [
            Question(
                key="volatility_regime",
                question=f"Classify current volatility regime for {ticker} into (0=Tight, 1=Normal, 2=Wide, 3=Shock).",
                decision_type=DecisionType.SCORE,
                score_range=(0, 3),
            )
        ]

        resp: BatchResponse = self.client.evaluate(context, questions)
        res = resp.results.get("volatility_regime")
        score = res.score if (res and res.score is not None) else 1

        risk_dist = abs(entry_price - default_stop)
        if risk_dist <= 0.0:
            risk_dist = entry_price * 0.01

        is_long = default_stop <= entry_price

        if score == 0:  # Tight (low noise)
            adaptive_stop = entry_price - (0.5 * risk_dist) if is_long else entry_price + (0.5 * risk_dist)
        elif score == 1:  # Normal chop
            adaptive_stop = default_stop
        elif score == 2:  # Wide noise
            adaptive_stop = entry_price - (1.5 * risk_dist) if is_long else entry_price + (1.5 * risk_dist)
        elif score == 3:  # Shock
            # Flattens position immediately by matching current close price
            adaptive_stop = float(bar.get("Close", bar.get("close", entry_price)))
        else:
            adaptive_stop = default_stop

        self._record_audit(ticker, "adaptive_stop_shield", "5m", res, "APPLIED", resp.latency_ms)
        return float(adaptive_stop)

    def evaluate_overnight_continuation(
        self,
        ticker: str,
        overnight_pos: Dict[str, Any],
        today_bar: pd.Series,
    ) -> Tuple[bool, float, str]:
        """Evaluate holding overnight position through morning open (09:15 IST).

        Returns:
            (True, ratcheted_stop, 'CONTINUATION_HOLD_DP_SAVED') if continuation holds.
            (False, today_bar['Open'], 'STANDARD_MORNING_EXIT') if exit is recommended.
        """
        bar_dict = today_bar.to_dict() if hasattr(today_bar, "to_dict") else dict(today_bar)
        context = (
            f"Overnight continuation evaluation for {ticker} at morning open. "
            f"Overnight Pos: {overnight_pos}. Today Bar: {bar_dict}"
        )
        questions = [
            Question(
                key="should_hold_through_0915_open",
                question="Should the overnight position be held through 09:15 open based on momentum continuation?",
                decision_type=DecisionType.NOUL,
            ),
            Question(
                key="morning_gap_fill_risk",
                question="Is there elevated risk of adverse morning gap fill against the overnight position?",
                decision_type=DecisionType.NOUL,
            ),
        ]

        resp: BatchResponse = self.client.evaluate(context, questions)
        hold_res = resp.results.get("should_hold_through_0915_open")
        gap_res = resp.results.get("morning_gap_fill_risk")

        hold_prob = hold_res.probability if (hold_res and hold_res.probability is not None) else 0.0
        gap_risk = gap_res.probability if (gap_res and gap_res.probability is not None) else 0.0

        is_held = (hold_prob >= 0.65) and (gap_risk < 0.40)
        status = "APPLIED" if is_held else "BYPASSED"

        self._record_audit(ticker, "overnight_continuation", "1d", hold_res, status, resp.latency_ms)
        self._record_audit(ticker, "overnight_continuation", "1d", gap_res, status, resp.latency_ms)

        if is_held:
            current_stop = float(overnight_pos.get("trailing_stop_loss", overnight_pos.get("stop_loss", 0.0)))
            entry_price = float(overnight_pos.get("entry_price", current_stop))
            open_price = float(today_bar.get("Open", today_bar.get("close", entry_price)))

            ratcheted = overnight_pos.get("ratcheted_stop")
            if ratcheted is not None:
                ratcheted_stop = float(ratcheted)
            else:
                ratcheted_stop = max(current_stop, entry_price) if open_price >= entry_price else current_stop

            return True, ratcheted_stop, "CONTINUATION_HOLD_DP_SAVED"

        open_px = float(today_bar.get("Open", 0.0))
        return False, open_px, "STANDARD_MORNING_EXIT"

    def evaluate_cross_asset_macro_shock(
        self,
        macro_snapshot: Dict[str, Any],
    ) -> Tuple[bool, str, float]:
        """Evaluate systemic cross-asset macro shock via deterministic tripwires & Gemini veto.

        Returns (is_shock, action_reason, severity_prob).
        """
        vix_delta = float(macro_snapshot.get("india_vix_delta_pct", 0.0))
        nifty_ret = float(macro_snapshot.get("nifty_5m_ret", 0.0))
        gold_ret = float(macro_snapshot.get("gold_ret", 0.0))

        # 1. Deterministic zero-latency tripwires
        if vix_delta >= 0.15:
            return True, "MACRO_VIX_SPIKE_PANIC", 1.0
        if nifty_ret <= -0.012:
            return True, "MACRO_INDEX_GAP_DOWN", 1.0
        if gold_ret >= 0.015 and nifty_ret <= -0.008:
            return True, "MACRO_SAFE_HAVEN_DIVERGENCE", 1.0

        # 2. Gemini Decision API categorical evaluation
        context = (
            f"Macro volatility assessment: VIX 30m delta: {vix_delta*100:.1f}%, "
            f"Nifty 5m ret: {nifty_ret*100:.2f}%, Gold ret: {gold_ret*100:.2f}%."
        )
        questions = [
            Question(
                key="is_systemic_macro_shock",
                question="Is there an imminent systemic macroeconomic or liquidity shock triggering flight to safety?",
                decision_type=DecisionType.NOUL,
            ),
        ]
        resp = self.client.evaluate(context, questions)
        res = resp.results.get("is_systemic_macro_shock")
        prob = res.probability if (res and res.probability is not None) else 0.0

        is_shock = prob >= 0.70
        status = "APPLIED" if is_shock else "BYPASSED"
        self._record_audit("SYSTEM", "macro_regime_shield", "5m", res, status, resp.latency_ms)

        reason = "GEMINI_MACRO_CRISIS_VETO" if is_shock else "MACRO_REGIME_STABLE"
        return is_shock, reason, prob
