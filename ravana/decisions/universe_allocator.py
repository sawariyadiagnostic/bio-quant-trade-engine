"""Universe Alpha Crown Allocator for RAVANA Quantitative Engine.

Selects the single highest-probability alpha signal from multiple concurrent candidates
and assigns conviction-based capital allocation fraction using Gemini Decisions API.
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


class UniverseAlphaCrown:
    """Evaluates concurrent candidate signals and selects single winning ETF allocation."""

    SCORE_FRACTION_MAP: Dict[int, float] = {
        0: 0.0,
        1: 0.25,
        2: 0.50,
        3: 0.75,
        4: 1.0,
    }

    def __init__(
        self,
        client: Optional[GeminiDecisionClient] = None,
        ledger: Optional[SQLiteStateLedger] = None,
    ) -> None:
        self.client = client if client is not None else GeminiDecisionClient(mock=True)
        self.ledger = ledger

    def _compile_comparison_table(
        self,
        candidate_signals: List[TriggerSignal],
        current_bars: Dict[str, pd.Series],
        capital: float,
    ) -> str:
        """Compile tabular comparison of candidate ETF signals and microstructure metrics."""
        lines = [
            f"Universe Allocation Evaluation (Available Capital: INR {capital:,.2f}):",
            "Symbol | Trigger | Action | Conviction | Price | VWAP | ATR | Spread_Z | OFI | VR",
        ]
        for sig in candidate_signals:
            bar = current_bars.get(sig.symbol, pd.Series(dtype=float))
            px = _extract_bar_val(bar, ["Close", "close", "price", "LastPrice"], 0.0)
            vwap = _extract_bar_val(bar, ["VWAP", "vwap", "Vwap"], px)
            atr = _extract_bar_val(bar, ["ATR_14", "ATR", "atr", "atr_14"], 0.0)
            sz = _extract_bar_val(bar, ["SPREAD_Z", "spread_z", "spreadZ"], 0.0)
            ofi = _extract_bar_val(bar, ["OFI_5", "OFI", "ofi", "ofi_5"], 0.0)
            vr = _extract_bar_val(bar, ["VARIANCE_RATIO_20", "VR_20", "VR", "vr", "vr_20"], 1.0)
            lines.append(
                f"{sig.symbol} | {sig.trigger_name} | {sig.action.value} | "
                f"{sig.conviction:.2f} | {px:.2f} | {vwap:.2f} | {atr:.4f} | "
                f"{sz:.2f} | {ofi:.4f} | {vr:.4f}"
            )
        return "\n".join(lines)

    def _build_questions(self, candidate_symbols: List[str]) -> List[Question]:
        """Construct the choice and allocation score questions."""
        choices = candidate_symbols + ["CASH"]
        return [
            Question(
                key="best_etf_to_buy_now",
                question="Which candidate ETF has the highest probability of continuation, or choose CASH?",
                decision_type=DecisionType.CHOICE,
                choices=choices,
            ),
            Question(
                key="capital_allocation_fraction",
                question="Rate conviction capital allocation fraction on a 0-4 scale (0=0%, 1=25%, 2=50%, 3=75%, 4=100%).",
                decision_type=DecisionType.SCORE,
                score_range=(0, 4),
            ),
        ]

    def _resolve_score_fraction(self, score_res: Optional[Any]) -> float:
        """Resolve score result into discrete capital fraction [0.0, 1.0]."""
        if score_res is None:
            return 1.0

        if score_res.score is not None:
            score_int = int(score_res.score)
            return self.SCORE_FRACTION_MAP.get(score_int, max(0.0, min(1.0, score_int / 4.0)))

        if score_res.selected_choice is not None:
            try:
                score_int = int(round(float(score_res.selected_choice)))
                return self.SCORE_FRACTION_MAP.get(score_int, max(0.0, min(1.0, score_int / 4.0)))
            except (ValueError, TypeError):
                pass

        return 1.0

    def _log_audit_records(
        self,
        questions: List[Question],
        batch_resp: Any,
        winning_signal: Optional[TriggerSignal],
        time_str: str,
    ) -> None:
        """Persist universe allocation audit records into SQLite state ledger if present."""
        if self.ledger is None:
            return

        timeframe = "5m"
        if winning_signal and winning_signal.metadata:
            timeframe = str(winning_signal.metadata.get("timeframe", "5m"))

        status = "APPLIED" if winning_signal is not None else "VETOED"
        target_symbol = winning_signal.symbol if winning_signal is not None else "CASH"

        records: List[DecisionAuditRecord] = []
        for q in questions:
            res = batch_resp.results.get(q.key)
            choice = res.selected_choice if res else None
            prob = res.probability if res else None
            unique_suffix = uuid.uuid4().hex[:6]
            rec_id = f"alpha_crown_{q.key}_{unique_suffix}"

            records.append(
                DecisionAuditRecord(
                    decision_id=rec_id,
                    timestamp=time_str,
                    timeframe=timeframe,
                    cluster_id="alpha_crown",
                    symbol=target_symbol,
                    question_key=q.key,
                    selected_choice=choice,
                    probability_score=prob,
                    latency_ms=float(batch_resp.latency_ms),
                    status=status,
                )
            )
        self.ledger.record_decision_batch(records)

    def select_top_allocation(
        self,
        candidate_signals: List[TriggerSignal],
        current_bars: Dict[str, pd.Series],
        capital: float = 200000.0,
    ) -> Tuple[Optional[TriggerSignal], float]:
        """Select single top candidate signal and allocation fraction.

        Returns:
            Tuple of (winning_signal or None, allocation_fraction)
        """
        if not candidate_signals:
            return None, 0.0

        if len(candidate_signals) == 1:
            return candidate_signals[0], 1.0

        signal_by_symbol: Dict[str, TriggerSignal] = {}
        candidate_symbols: List[str] = []
        for s in candidate_signals:
            if s.symbol not in signal_by_symbol:
                candidate_symbols.append(s.symbol)
                signal_by_symbol[s.symbol] = s
            elif s.conviction > signal_by_symbol[s.symbol].conviction:
                signal_by_symbol[s.symbol] = s

        context_str = self._compile_comparison_table(candidate_signals, current_bars, capital)
        questions = self._build_questions(candidate_symbols)
        batch_resp = self.client.evaluate(context=context_str, questions=questions)

        choice_res = batch_resp.results.get("best_etf_to_buy_now")
        score_res = batch_resp.results.get("capital_allocation_fraction")

        chosen_symbol = choice_res.selected_choice if choice_res else None

        if chosen_symbol == "CASH" or chosen_symbol is None or chosen_symbol not in signal_by_symbol:
            winning_signal = None
            fraction = 0.0
        else:
            winning_signal = signal_by_symbol[chosen_symbol]
            fraction = self._resolve_score_fraction(score_res)

        raw_ts = winning_signal.timestamp if winning_signal else candidate_signals[0].timestamp
        time_str = raw_ts.isoformat() if isinstance(raw_ts, (pd.Timestamp, datetime)) else str(raw_ts or "N/A")

        self._log_audit_records(questions, batch_resp, winning_signal, time_str)
        return winning_signal, fraction
