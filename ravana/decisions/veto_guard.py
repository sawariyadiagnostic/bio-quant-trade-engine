"""Decision Veto Guard for RAVANA Quantitative Engine.

Applies Gemini Decisions API evaluations to validate or veto structural trigger signals
based on false breakout traps, dead money stagnation, and genuine volume absorption.
"""

from __future__ import annotations

from datetime import datetime
import logging
from typing import Any, Dict, List, Optional, Tuple
import uuid

import pandas as pd

from ravana.decisions.client import GeminiDecisionClient
from ravana.decisions.schema import DecisionType, Question
from ravana.strategies.base import TriggerSignal
from ravana.strategies.state_ledger import DecisionAuditRecord, SQLiteStateLedger

logger = logging.getLogger(__name__)


def _extract_bar_val(bar: pd.Series, keys: List[str], default: float = 0.0) -> float:
    """Safely extract float metric from series matching any key."""
    for k in keys:
        if k in bar.index:
            try:
                val = bar[k]
                if pd.notna(val):
                    return float(val)
            except (ValueError, TypeError):
                pass
    return default


class DecisionVetoGuard:
    """Veto filter evaluating microstructure signals against Gemini Decision models."""

    def __init__(
        self,
        client: Optional[GeminiDecisionClient] = None,
        ledger: Optional[SQLiteStateLedger] = None,
        dead_money_threshold: float = 0.75,
        false_breakout_threshold: float = 0.72,
    ) -> None:
        self.client = client if client is not None else GeminiDecisionClient(mock=True)
        self.ledger = ledger
        self.dead_money_threshold = float(dead_money_threshold)
        self.false_breakout_threshold = float(false_breakout_threshold)

    def _compile_market_context(
        self,
        signal: TriggerSignal,
        bar: pd.Series,
        context_summary: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Compile structured market context string from signal and bar data."""
        symbol = signal.symbol
        raw_ts = getattr(signal, "timestamp", None)
        if raw_ts is None and hasattr(bar, "name") and bar.name is not None:
            raw_ts = bar.name
        time_str = raw_ts.isoformat() if isinstance(raw_ts, (pd.Timestamp, datetime)) else str(raw_ts or "N/A")

        price = _extract_bar_val(bar, ["Close", "close", "price", "LastPrice"], 0.0)
        vwap = _extract_bar_val(bar, ["VWAP", "vwap", "Vwap"], price)
        atr = _extract_bar_val(bar, ["ATR_14", "ATR", "atr", "atr_14"], 0.0)
        spread_z = _extract_bar_val(bar, ["SPREAD_Z", "spread_z", "spreadZ"], 0.0)
        if spread_z == 0.0 and getattr(signal, "metadata", None):
            spread_z = float(signal.metadata.get("spread_z", signal.metadata.get("z_score", 0.0)))
        ofi = _extract_bar_val(bar, ["OFI_5", "OFI", "ofi", "ofi_5"], 0.0)
        if ofi == 0.0 and getattr(signal, "metadata", None):
            ofi = float(signal.metadata.get("ofi", signal.metadata.get("ofi_5", 0.0)))
        vr = _extract_bar_val(bar, ["VARIANCE_RATIO_20", "VR_20", "VR", "vr", "vr_20"], 1.0)
        if vr == 1.0 and getattr(signal, "metadata", None):
            vr = float(signal.metadata.get("vr", signal.metadata.get("variance_ratio", 1.0)))
        nifty_ret = _extract_bar_val(bar, ["NIFTY_RET", "nifty_ret", "Index_Ret"], 0.0)
        rel_str = _extract_bar_val(bar, ["REL_STRENGTH", "RS", "relative_strength"], 0.0)

        parts = [
            f"Symbol: {symbol}",
            f"Time: {time_str}",
            f"Price: {price:.2f}",
            f"VWAP: {vwap:.2f}",
            f"ATR: {atr:.4f}",
            f"Spread_Z: {spread_z:.2f}",
            f"OFI: {ofi:.4f}",
            f"VR: {vr:.4f}",
            f"Regime: {'Chop' if vr < 0.85 else 'Trending'}",
            f"NIFTY_Ret: {nifty_ret:+.2f}%",
            f"Rel_Strength: {rel_str:+.2f}%",
            f"Trigger: {signal.trigger_name}",
            f"Action: {signal.action.value}",
            f"Conviction: {signal.conviction:.2f}",
        ]
        if context_summary:
            parts.append(f"Context Summary: {context_summary}")

        return " | ".join(parts), time_str

    def _build_questions(self) -> List[Question]:
        """Construct the 3 standard institutional veto check questions."""
        return [
            Question(
                key="is_dead_money_risk",
                question="Is there a high risk price stalls with <0.20 ATR progress over next 6 bars?",
                decision_type=DecisionType.NOUL,
            ),
            Question(
                key="is_false_breakout_trap",
                question="Is this breakout an institutional bull-trap / false breakout?",
                decision_type=DecisionType.NOUL,
            ),
            Question(
                key="is_volume_absorption_genuine",
                question="Does order flow confirm genuine institutional volume absorption?",
                decision_type=DecisionType.NOUL,
            ),
        ]

    def _log_audit_records(
        self,
        signal: TriggerSignal,
        time_str: str,
        questions: List[Question],
        batch_resp: Any,
        status: str,
    ) -> None:
        """Persist decision audit records into SQLite state ledger if present."""
        if self.ledger is None:
            return

        timeframe = str(signal.metadata.get("timeframe", "5m"))
        records: List[DecisionAuditRecord] = []
        for q in questions:
            res = batch_resp.results.get(q.key)
            choice = res.selected_choice if res else None
            prob = res.probability if res else None
            unique_suffix = uuid.uuid4().hex[:6]
            rec_id = f"veto_{signal.symbol}_{q.key}_{unique_suffix}"

            records.append(
                DecisionAuditRecord(
                    decision_id=rec_id,
                    timestamp=time_str,
                    timeframe=timeframe,
                    cluster_id="veto",
                    symbol=signal.symbol,
                    question_key=q.key,
                    selected_choice=choice,
                    probability_score=prob,
                    latency_ms=float(batch_resp.latency_ms),
                    status=status,
                )
            )
        self.ledger.record_decision_batch(records)

    def evaluate_signal(
        self,
        signal: TriggerSignal,
        bar: pd.Series,
        context_summary: Optional[str] = None,
    ) -> Tuple[bool, str, float]:
        """Evaluate trigger signal against Gemini decisions veto guard.

        Returns:
            Tuple of (is_approved, decision_reason, probability)
        """
        context_str, time_str = self._compile_market_context(signal, bar, context_summary)
        questions = self._build_questions()
        batch_resp = self.client.evaluate(context=context_str, questions=questions)

        trap_res = batch_resp.results.get("is_false_breakout_trap")
        dead_res = batch_resp.results.get("is_dead_money_risk")
        abs_res = batch_resp.results.get("is_volume_absorption_genuine")

        p_trap = trap_res.probability if (trap_res and trap_res.probability is not None) else 0.0
        p_dead = dead_res.probability if (dead_res and dead_res.probability is not None) else 0.0
        p_abs = abs_res.probability if (abs_res and abs_res.probability is not None) else 1.0

        if p_trap > self.false_breakout_threshold:
            outcome = (False, "VETO_FALSE_BREAKOUT_TRAP", p_trap)
            audit_status = "VETOED"
        elif p_dead > self.dead_money_threshold:
            outcome = (False, "VETO_DEAD_MONEY_RISK", p_dead)
            audit_status = "VETOED"
        else:
            outcome = (True, "APPROVED", p_abs)
            audit_status = "APPLIED"

        self._log_audit_records(signal, time_str, questions, batch_resp, audit_status)
        return outcome
