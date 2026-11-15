"""Gemini Decisions API Client for RAVANA Quantitative Engine.

Provides high-throughput, low-latency categorical routing and scoring
against Gemini Decisions API with rate limiting, retries, and offline fallback.
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from ravana.decisions.schema import (
    BatchResponse,
    DecisionResult,
    DecisionType,
    Question,
)

logger = logging.getLogger(__name__)


class RateLimiter:
    """Sliding window rate-limiter enforcing RPM and TPM constraints."""

    def __init__(self, max_rpm: int = 200, max_tpm: int = 50000) -> None:
        self.max_rpm = max_rpm
        self.max_tpm = max_tpm
        self._lock = threading.Lock()
        self._requests: List[float] = []
        self._tokens: List[Tuple[float, int]] = []

    def _purge_old_entries(self, now: float) -> None:
        cutoff = now - 60.0
        self._requests = [t for t in self._requests if t > cutoff]
        self._tokens = [(t, count) for (t, count) in self._tokens if t > cutoff]

    def current_usage(self) -> Tuple[int, int]:
        """Return (requests_in_last_minute, tokens_in_last_minute)."""
        with self._lock:
            now = time.time()
            self._purge_old_entries(now)
            return len(self._requests), sum(c for _, c in self._tokens)

    def acquire(self, estimated_tokens: int = 100) -> None:
        """Block until rate limit headroom is available, then record usage."""
        while True:
            with self._lock:
                now = time.time()
                self._purge_old_entries(now)
                current_rpm = len(self._requests)
                current_tpm = sum(c for _, c in self._tokens)

                if current_rpm < self.max_rpm and (current_tpm + estimated_tokens) <= self.max_tpm:
                    self._requests.append(now)
                    self._tokens.append((now, estimated_tokens))
                    return

                sleep_time = 0.1
                if self._requests:
                    sleep_time = max(0.05, 60.0 - (now - self._requests[0]) + 0.01)

            time.sleep(min(sleep_time, 1.0))


class GeminiDecisionClient:
    """Client for evaluating structured questions via Gemini Decisions API."""

    DEFAULT_MODEL = "models/gemini-decision-1"
    DEFAULT_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/decisions"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout: float = 10.0,
        max_rpm: int = 200,
        max_tpm: int = 50000,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        mock: bool = False,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.mock = mock or (not self.api_key)
        self.rate_limiter = RateLimiter(max_rpm=max_rpm, max_tpm=max_tpm)
        self._session = requests.Session()

    def estimate_tokens(self, context: str, questions: List[Question]) -> int:
        """Estimate token consumption for context and questions batch."""
        char_count = len(context)
        for q in questions:
            char_count += len(q.key) + len(q.question)
            if q.choices:
                char_count += sum(len(c) for c in q.choices)
        return max(30, (char_count // 4) + (len(questions) * 20))

    def build_payload(self, context: str, questions: List[Question]) -> Dict[str, Any]:
        """Build JSON payload matching verified Gemini Decision schema."""
        return {
            "model": self.model,
            "decision_input": {
                "context": context,
                "questions": [q.to_dict() for q in questions],
            },
        }

    def build_google_payload(self, context: str, questions: List[Question]) -> Dict[str, Any]:
        """Build JSON payload matching Google Generative Language REST schema."""
        return {
            "model": self.model,
            "input": context,
            "questions": {q.key: q.to_google_format() for q in questions},
        }

    def _mock_response(self, questions: List[Question], latency_ms: float = 1.0) -> BatchResponse:
        """Generate deterministic synthetic response for offline or test mode."""
        results: Dict[str, DecisionResult] = {}
        for q in questions:
            if q.decision_type == DecisionType.NOUL:
                results[q.key] = DecisionResult(
                    key=q.key,
                    decision_type=DecisionType.NOUL,
                    selected_choice="true",
                    score=None,
                    probability=0.85,
                    raw_data={"choice": "true", "score": 0.85},
                )
            elif q.decision_type == DecisionType.CHOICE:
                choice_val = q.choices[0] if q.choices else "CHOICE_A"
                results[q.key] = DecisionResult(
                    key=q.key,
                    decision_type=DecisionType.CHOICE,
                    selected_choice=choice_val,
                    score=None,
                    probability=0.75,
                    raw_data={"choice": choice_val, "score": 0.75},
                )
            elif q.decision_type == DecisionType.SCORE:
                min_s, max_s = q.score_range if q.score_range is not None else (0, 3)
                s_val = (min_s + max_s) // 2
                results[q.key] = DecisionResult(
                    key=q.key,
                    decision_type=DecisionType.SCORE,
                    selected_choice=str(s_val),
                    score=s_val,
                    probability=0.80,
                    raw_data={"choice": s_val, "score": 0.80},
                )
        return BatchResponse(results=results, latency_ms=latency_ms, status="OK")

    def _parse_response(
        self,
        data: Dict[str, Any],
        questions: List[Question],
    ) -> Dict[str, DecisionResult]:
        """Parse decisions from verified decision_output or answers structure."""
        q_map = {q.key: q for q in questions}
        results: Dict[str, DecisionResult] = {}

        if "decision_output" in data:
            decisions = data.get("decision_output", {}).get("decisions", [])
            for item in decisions:
                key = item.get("key", "")
                q = q_map.get(key)
                results[key] = DecisionResult.from_decision_item(item, q)
            return results

        if "answers" in data:
            answers = data.get("answers", {})
            for key, item in answers.items():
                q = q_map.get(key)
                results[key] = DecisionResult.from_answer_item(key, item, q)
            return results

        return results

    def _execute_http_post(
        self,
        url: str,
        payload: Dict[str, Any],
    ) -> requests.Response:
        """Execute single HTTP POST request with standard headers and timeout."""
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key or "",
        }
        return self._session.post(url, json=payload, headers=headers, timeout=self.timeout)

    def _request_with_backoff(
        self,
        context: str,
        questions: List[Question],
    ) -> Dict[str, Any]:
        """Execute HTTP request with exponential backoff and schema fallback."""
        url = f"{self.endpoint}?key={self.api_key}" if self.api_key else self.endpoint
        payload = self.build_payload(context, questions)
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self._execute_http_post(url, payload)

                # Live Google API rejects decision_input with 400; fallback to Google REST format
                if response.status_code == 400 and "decision_input" in response.text:
                    logger.debug("Falling back to Google Generative Language REST payload")
                    alt_payload = self.build_google_payload(context, questions)
                    response = self._execute_http_post(url, alt_payload)

                if response.status_code == 200:
                    return response.json()

                if response.status_code in (429, 503):
                    backoff = self.retry_delay * (2 ** attempt) + random.uniform(0.0, 0.2)
                    logger.warning("HTTP %d, retrying in %.2fs (attempt %d)", response.status_code, backoff, attempt)
                    time.sleep(backoff)
                    continue

                response.raise_for_status()

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                last_error = exc
                backoff = self.retry_delay * (2 ** attempt) + random.uniform(0.0, 0.2)
                logger.warning("Network error %s, retrying in %.2fs", exc, backoff)
                time.sleep(backoff)

            except Exception as exc:
                last_error = exc
                break

        if last_error:
            raise last_error
        raise RuntimeError("Max retries exceeded without successful response")

    def evaluate(self, context: str, questions: List[Question]) -> BatchResponse:
        """Evaluate a batch of questions against context, returning BatchResponse."""
        if not questions:
            return BatchResponse(results={}, latency_ms=0.0, status="OK")

        start_time = time.perf_counter()

        if self.mock:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return self._mock_response(questions, latency_ms=elapsed_ms)

        try:
            tokens = self.estimate_tokens(context, questions)
            self.rate_limiter.acquire(estimated_tokens=tokens)

            data = self._request_with_backoff(context, questions)
            results = self._parse_response(data, questions)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            return BatchResponse(
                results=results,
                latency_ms=elapsed_ms,
                status="OK",
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error("GeminiDecisionClient evaluation failed: %s", exc)
            return BatchResponse(
                results={},
                latency_ms=elapsed_ms,
                status="ERROR",
                error_message=str(exc),
            )
