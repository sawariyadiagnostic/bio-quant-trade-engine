"""Unit tests for Gemini Decisions API Client and Schema.

Validates schema serialization, mock mode responses, parsing logic,
rate limiting, retries, and live API connectivity when GEMINI_API_KEY is available.
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Ensure ravana package is importable regardless of runner invocation
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from ravana.decisions.client import GeminiDecisionClient, RateLimiter
from ravana.decisions.schema import (
    BatchResponse,
    DecisionResult,
    DecisionType,
    Question,
)


class TestGeminiDecisionSchema(unittest.TestCase):
    """Test schema dataclasses and serialization."""

    def test_noul_question_serialization(self) -> None:
        q = Question(
            key="q_bull",
            question="Will BANKBEES outperform?",
            decision_type=DecisionType.NOUL,
        )
        data = q.to_dict()
        self.assertEqual(data["key"], "q_bull")
        self.assertEqual(data["question"], "Will BANKBEES outperform?")
        self.assertIn("noul", data)
        self.assertEqual(data["noul"], {})

    def test_choice_question_serialization(self) -> None:
        choices = ["BANKBEES", "ITBEES", "CASH"]
        q = Question(
            key="q_lead",
            question="Which asset leads momentum?",
            decision_type=DecisionType.CHOICE,
            choices=choices,
        )
        data = q.to_dict()
        self.assertEqual(data["key"], "q_lead")
        self.assertIn("choice", data)
        self.assertEqual(data["choice"]["choices"], choices)

    def test_score_question_serialization(self) -> None:
        q = Question(
            key="q_vol",
            question="Rate shock risk 0 to 3",
            decision_type=DecisionType.SCORE,
            score_range=(0, 3),
        )
        data = q.to_dict()
        self.assertEqual(data["key"], "q_vol")
        self.assertIn("score", data)
        self.assertEqual(data["score"]["min_score"], 0)
        self.assertEqual(data["score"]["max_score"], 3)

    def test_payload_builder(self) -> None:
        client = GeminiDecisionClient(mock=True)
        questions = [
            Question(key="q1", question="Outperform?", decision_type=DecisionType.NOUL),
            Question(key="q2", question="Sector?", decision_type=DecisionType.CHOICE, choices=["IT", "AUTO"]),
            Question(key="q3", question="Score?", decision_type=DecisionType.SCORE, score_range=(0, 3)),
        ]
        payload = client.build_payload("Nifty up 1.5%", questions)

        self.assertEqual(payload["model"], "models/gemini-decision-1")
        self.assertIn("decision_input", payload)
        self.assertEqual(payload["decision_input"]["context"], "Nifty up 1.5%")
        q_list = payload["decision_input"]["questions"]
        self.assertEqual(len(q_list), 3)
        self.assertEqual(q_list[0]["key"], "q1")
        self.assertIn("noul", q_list[0])
        self.assertEqual(q_list[1]["key"], "q2")
        self.assertIn("choice", q_list[1])
        self.assertEqual(q_list[2]["key"], "q3")
        self.assertIn("score", q_list[2])


class TestGeminiDecisionParsing(unittest.TestCase):
    """Test response parsing across verified and standard formats."""

    def setUp(self) -> None:
        self.client = GeminiDecisionClient(mock=True)
        self.questions = [
            Question(key="q1", question="Outperform?", decision_type=DecisionType.NOUL),
            Question(key="q2", question="Leader?", decision_type=DecisionType.CHOICE, choices=["ITBEES", "BANKBEES"]),
            Question(key="q3", question="Shock?", decision_type=DecisionType.SCORE, score_range=(0, 3)),
        ]

    def test_parse_verified_decision_output(self) -> None:
        raw_api_response = {
            "decision_output": {
                "decisions": [
                    {
                        "key": "q1",
                        "noul": {"choice": "true", "score": 0.8125},
                    },
                    {
                        "key": "q2",
                        "choice": {"choice": "ITBEES", "score": 0.654},
                    },
                    {
                        "key": "q3",
                        "score": {"choice": 1, "score": 0.92},
                    },
                ]
            }
        }
        results = self.client._parse_response(raw_api_response, self.questions)

        self.assertEqual(len(results), 3)

        # q1: noul
        r1 = results["q1"]
        self.assertEqual(r1.decision_type, DecisionType.NOUL)
        self.assertEqual(r1.selected_choice, "true")
        self.assertAlmostEqual(r1.probability or 0.0, 0.8125)
        self.assertIsNone(r1.score)

        # q2: choice
        r2 = results["q2"]
        self.assertEqual(r2.decision_type, DecisionType.CHOICE)
        self.assertEqual(r2.selected_choice, "ITBEES")
        self.assertAlmostEqual(r2.probability or 0.0, 0.654)
        self.assertIsNone(r2.score)

        # q3: score
        r3 = results["q3"]
        self.assertEqual(r3.decision_type, DecisionType.SCORE)
        self.assertEqual(r3.score, 1)
        self.assertEqual(r3.selected_choice, "1")
        self.assertAlmostEqual(r3.probability or 0.0, 0.92)

    def test_parse_answers_format(self) -> None:
        raw_api_response = {
            "answers": {
                "q1": {"confidence": 0.85, "noul": 0.85, "type": "noul"},
                "q2": {"confidence": 0.99, "choice": "ITBEES", "type": "choice"},
                "q3": {"confidence": 0.70, "score": 2.0, "type": "score"},
            }
        }
        results = self.client._parse_response(raw_api_response, self.questions)

        self.assertEqual(len(results), 3)
        self.assertEqual(results["q1"].selected_choice, "true")
        self.assertAlmostEqual(results["q1"].probability or 0.0, 0.85)

        self.assertEqual(results["q2"].selected_choice, "ITBEES")
        self.assertAlmostEqual(results["q2"].probability or 0.0, 0.99)

        self.assertEqual(results["q3"].score, 2)
        self.assertAlmostEqual(results["q3"].probability or 0.0, 0.70)


class TestGeminiDecisionMockMode(unittest.TestCase):
    """Test offline and mock mode evaluation."""

    def test_mock_mode_evaluation(self) -> None:
        client = GeminiDecisionClient(mock=True)
        questions = [
            Question(key="q1", question="Outperform?", decision_type=DecisionType.NOUL),
            Question(key="q2", question="Which asset?", decision_type=DecisionType.CHOICE, choices=["BANKBEES", "ITBEES"]),
            Question(key="q3", question="Risk score?", decision_type=DecisionType.SCORE, score_range=(0, 3)),
        ]

        resp = client.evaluate("Context text", questions)

        self.assertEqual(resp.status, "OK")
        self.assertIsNone(resp.error_message)
        self.assertEqual(len(resp.results), 3)

        self.assertEqual(resp.results["q1"].decision_type, DecisionType.NOUL)
        self.assertIn(resp.results["q1"].selected_choice, ("true", "false"))

        self.assertEqual(resp.results["q2"].decision_type, DecisionType.CHOICE)
        self.assertIn(resp.results["q2"].selected_choice, ["BANKBEES", "ITBEES"])

        self.assertEqual(resp.results["q3"].decision_type, DecisionType.SCORE)
        self.assertIn(resp.results["q3"].score, [0, 1, 2, 3])

    def test_empty_questions_evaluation(self) -> None:
        client = GeminiDecisionClient(mock=True)
        resp = client.evaluate("Empty questions", [])
        self.assertEqual(resp.status, "OK")
        self.assertEqual(resp.results, {})


class TestRateLimiterAndRetries(unittest.TestCase):
    """Test rate limiting and retry behavior."""

    def test_rate_limiter_usage_tracking(self) -> None:
        limiter = RateLimiter(max_rpm=10, max_tpm=1000)
        limiter.acquire(estimated_tokens=50)
        limiter.acquire(estimated_tokens=75)

        rpm, tpm = limiter.current_usage()
        self.assertEqual(rpm, 2)
        self.assertEqual(tpm, 125)

    @patch("requests.Session.post")
    def test_retry_on_429(self, mock_post: MagicMock) -> None:
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.text = "Rate limit exceeded"

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.json.return_value = {
            "decision_output": {
                "decisions": [
                    {"key": "q1", "noul": {"choice": "true", "score": 0.9}}
                ]
            }
        }

        mock_post.side_effect = [resp_429, resp_200]

        client = GeminiDecisionClient(
            api_key="fake-test-key",
            mock=False,
            max_retries=2,
            retry_delay=0.01,
            enable_cache=False,
        )
        questions = [Question(key="q1", question="Outperform?", decision_type=DecisionType.NOUL)]
        resp = client.evaluate("Market context", questions)

        self.assertEqual(resp.status, "OK")
        self.assertIn("q1", resp.results)
        self.assertEqual(mock_post.call_count, 2)


class TestGeminiDecisionLiveApi(unittest.TestCase):
    """Test live API execution against Gemini Decision model when key is present."""

    def test_live_api_call_if_key_available(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            self.skipTest("GEMINI_API_KEY environment variable not available for live call")

        client = GeminiDecisionClient(api_key=api_key, mock=False)
        context = (
            "NIFTY 50 is trending strongly upwards (+1.2%), led by IT sector (+2.5%). "
            "Banking index is flat (+0.1%) with moderate volume. Volatility index (India VIX) is low at 12.4."
        )
        questions = [
            Question(
                key="outperform_it",
                question="Will ITBEES outperform BANKBEES in this regime?",
                decision_type=DecisionType.NOUL,
            ),
            Question(
                key="momentum_leader",
                question="Which asset has the strongest momentum?",
                decision_type=DecisionType.CHOICE,
                choices=["ITBEES", "BANKBEES", "CASH"],
            ),
            Question(
                key="volatility_shock_score",
                question="Rate volatility shock probability from 0 to 3",
                decision_type=DecisionType.SCORE,
                score_range=(0, 3),
            ),
        ]

        resp = client.evaluate(context, questions)

        self.assertEqual(resp.status, "OK", f"Expected OK status, got {resp.error_message}")
        self.assertGreater(resp.latency_ms, 0.0)
        self.assertEqual(len(resp.results), 3)

        # Check outperform_it
        r_noul = resp.results["outperform_it"]
        self.assertEqual(r_noul.decision_type, DecisionType.NOUL)
        self.assertIsNotNone(r_noul.probability)
        self.assertIn(r_noul.selected_choice, ("true", "false"))

        # Check momentum_leader
        r_choice = resp.results["momentum_leader"]
        self.assertEqual(r_choice.decision_type, DecisionType.CHOICE)
        self.assertIn(r_choice.selected_choice, ["ITBEES", "BANKBEES", "CASH"])
        self.assertIsNotNone(r_choice.probability)

        # Check volatility_shock_score
        r_score = resp.results["volatility_shock_score"]
        self.assertEqual(r_score.decision_type, DecisionType.SCORE)
        self.assertIn(r_score.score, [0, 1, 2, 3])
        self.assertIsNotNone(r_score.probability)


if __name__ == "__main__":
    unittest.main()
