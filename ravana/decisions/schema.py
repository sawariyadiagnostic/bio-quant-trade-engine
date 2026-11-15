"""Gemini Decisions API Schemas for RAVANA Quantitative Engine.

Defines question types, structured queries, decision outputs,
and batch response containers matching Google's Gemini Decision API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class DecisionType(str, Enum):
    """Enumeration of Gemini Decision question types."""
    NOUL = "noul"
    CHOICE = "choice"
    SCORE = "score"


@dataclass
class Question:
    """Represents a single query asked to the Gemini Decisions API."""
    key: str
    question: str
    decision_type: DecisionType
    choices: Optional[List[str]] = None
    score_range: Optional[Tuple[int, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize question into verified Gemini Decision API format."""
        base: Dict[str, Any] = {
            "key": self.key,
            "question": self.question,
        }
        if self.decision_type == DecisionType.NOUL:
            base["noul"] = {}
        elif self.decision_type == DecisionType.CHOICE:
            base["choice"] = {
                "choices": list(self.choices or [])
            }
        elif self.decision_type == DecisionType.SCORE:
            min_score, max_score = self.score_range if self.score_range is not None else (0, 3)
            base["score"] = {
                "min_score": min_score,
                "max_score": max_score,
            }
        return base

    def to_google_format(self) -> Dict[str, Any]:
        """Serialize question into Google Generative Language REST criteria format."""
        if self.decision_type == DecisionType.NOUL:
            return {
                "type": "noul",
                "instructions": self.question,
            }
        elif self.decision_type == DecisionType.CHOICE:
            opts = list(self.choices or [])
            return {
                "type": "choice",
                "instructions": self.question,
                "criteria": {opt: opt for opt in opts},
            }
        elif self.decision_type == DecisionType.SCORE:
            min_score, max_score = self.score_range if self.score_range is not None else (0, 3)
            return {
                "type": "score",
                "instructions": self.question,
                "criteria": [str(i) for i in range(min_score, max_score + 1)],
            }
        return {}


@dataclass
class DecisionResult:
    """Represents the evaluated outcome for a single question."""
    key: str
    decision_type: DecisionType
    selected_choice: Optional[str] = None
    score: Optional[int] = None
    probability: Optional[float] = None
    raw_data: Optional[Dict[str, Any]] = None

    @classmethod
    def from_decision_item(
        cls,
        item: Dict[str, Any],
        question: Optional[Question] = None,
    ) -> DecisionResult:
        """Parse from verified decision_output decisions item."""
        key = item.get("key", "")
        if "noul" in item:
            sub = item["noul"]
            prob = float(sub.get("score", 0.0)) if sub.get("score") is not None else None
            return cls(
                key=key,
                decision_type=DecisionType.NOUL,
                selected_choice=str(sub.get("choice", "")),
                score=None,
                probability=prob,
                raw_data=sub,
            )
        if "choice" in item:
            sub = item["choice"]
            prob = float(sub.get("score", 0.0)) if sub.get("score") is not None else None
            return cls(
                key=key,
                decision_type=DecisionType.CHOICE,
                selected_choice=str(sub.get("choice", "")),
                score=None,
                probability=prob,
                raw_data=sub,
            )
        if "score" in item:
            sub = item["score"]
            c_val = sub.get("choice")
            score_val = int(c_val) if c_val is not None else None
            prob = float(sub.get("score", 0.0)) if sub.get("score") is not None else None
            return cls(
                key=key,
                decision_type=DecisionType.SCORE,
                selected_choice=str(c_val) if c_val is not None else None,
                score=score_val,
                probability=prob,
                raw_data=sub,
            )
        # Fallback to question type if known
        dtype = question.decision_type if question else DecisionType.NOUL
        return cls(key=key, decision_type=dtype, raw_data=item)

    @classmethod
    def from_answer_item(
        cls,
        key: str,
        item: Dict[str, Any],
        question: Optional[Question] = None,
    ) -> DecisionResult:
        """Parse from Google Generative Language answers item."""
        type_str = item.get("type", "")
        dtype = DecisionType(type_str) if type_str in ("noul", "choice", "score") else (
            question.decision_type if question else DecisionType.NOUL
        )

        conf = item.get("confidence")
        prob = float(conf) if conf is not None else None

        if dtype == DecisionType.NOUL:
            noul_val = item.get("noul")
            if prob is None and noul_val is not None:
                prob = float(noul_val)
            choice_val = "true" if (prob is not None and prob >= 0.5) else "false"
            return cls(
                key=key,
                decision_type=dtype,
                selected_choice=choice_val,
                score=None,
                probability=prob,
                raw_data=item,
            )

        if dtype == DecisionType.CHOICE:
            choice_val = str(item.get("choice", ""))
            return cls(
                key=key,
                decision_type=dtype,
                selected_choice=choice_val,
                score=None,
                probability=prob,
                raw_data=item,
            )

        if dtype == DecisionType.SCORE:
            score_raw = item.get("score")
            score_val = int(round(float(score_raw))) if score_raw is not None else None
            choice_str = str(score_val) if score_val is not None else None
            return cls(
                key=key,
                decision_type=dtype,
                selected_choice=choice_str,
                score=score_val,
                probability=prob,
                raw_data=item,
            )

        return cls(key=key, decision_type=dtype, raw_data=item)


@dataclass
class BatchResponse:
    """Contains batch query outcome, latency, and status."""
    results: Dict[str, DecisionResult] = field(default_factory=dict)
    latency_ms: float = 0.0
    status: str = "OK"
    error_message: Optional[str] = None
