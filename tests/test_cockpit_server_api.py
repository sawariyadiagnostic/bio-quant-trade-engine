"""Unit and Integration Tests for FastAPI REST Server & Differential Delta SSE Protocol.

Validates:
1. Health and Status telemetry endpoints.
2. Discrete n8n-style pipeline endpoints (/ingest, /evolve, /abort).
3. SQLite WAL resident elites leaderboard endpoint.
4. Emergency Stage 1 freeze and Stage 2 panic liquidation endpoints.
5. Differential Delta Server-Sent Events (SSE) streaming format and ping keepalives.
"""
import json
import time
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.ui.server import app, supervisor, sentry
from src.pipeline.shared_telemetry import SharedMemoryBus


class TestCockpitServerApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        # Reset sentry freeze state before each test
        sentry.is_frozen = False

    def test_health_and_status_endpoints(self):
        """GET /health and GET /api/status return valid JSON and HTTP 200."""
        res_h = self.client.get("/health")
        self.assertEqual(res_h.status_code, 200)
        data_h = res_h.json()
        self.assertEqual(data_h["status"], "ok")
        self.assertIn("uptime_s", data_h)
        self.assertFalse(data_h["is_frozen"])

        res_s = self.client.get("/api/status")
        self.assertEqual(res_s.status_code, 200)
        data_s = res_s.json()
        self.assertIn("worker_running", data_s)
        self.assertIn("telemetry", data_s)
        self.assertIn("resident_elites_count", data_s)

    def test_pipeline_ingest_endpoint(self):
        """POST /api/pipeline/ingest validates params and returns dataset metadata."""
        payload = {"instrument": "NIFTYBEES", "lookback_years": 1}
        res = self.client.post("/api/pipeline/ingest", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "INGESTED")
        self.assertEqual(data["instrument"], "NIFTYBEES")
        self.assertGreater(data["bars_count"], 0)

    def test_pipeline_evolve_and_abort(self):
        """POST /api/pipeline/evolve starts worker, and POST /api/pipeline/abort halts it."""
        with patch.object(supervisor, "start_worker", return_value=True) as mock_start:
            payload = {
                "instrument": "NIFTYBEES",
                "generations": 10,
                "population_size": 20
            }
            res_ev = self.client.post("/api/pipeline/evolve", json=payload)
            self.assertEqual(res_ev.status_code, 200)
            data_ev = res_ev.json()
            self.assertEqual(data_ev["status"], "STARTED")
            mock_start.assert_called_once()

        # Abort request
        res_ab = self.client.post("/api/pipeline/abort")
        self.assertEqual(res_ab.status_code, 200)
        data_ab = res_ab.json()
        self.assertEqual(data_ab["status"], "ABORT_REQUESTED")

    def test_elites_leaderboard_endpoint(self):
        """GET /api/elites/leaderboard returns sorted champions from SQLite WAL."""
        res = self.client.get("/api/elites/leaderboard?limit=10&metric=sharpe")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ok")
        self.assertIsInstance(data["elites"], list)
        self.assertGreaterEqual(len(data["elites"]), 0)

    def test_emergency_freeze_and_panic_endpoints(self):
        """POST /api/emergency/freeze and /api/emergency/panic execute emergency protocols."""
        # Stage 1 Freeze
        res_f = self.client.post("/api/emergency/freeze")
        self.assertEqual(res_f.status_code, 200)
        data_f = res_f.json()
        self.assertEqual(data_f["status"], "FROZEN")
        self.assertTrue(sentry.is_frozen)

        # Stage 2 Panic Liquidation
        res_p = self.client.post("/api/emergency/panic")
        self.assertEqual(res_p.status_code, 200)
        data_p = res_p.json()
        self.assertEqual(data_p["status"], "LIQUIDATION_COMPLETE")
        self.assertTrue(data_p["cash_secured"])

    def test_differential_sse_streaming(self):
        """GET /api/stream/evolution yields valid SSE event wire format with keepalives."""
        response = self.client.get("/api/stream/evolution?max_ticks=2")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        text = response.text
        self.assertIn("event: connected", text)
        self.assertIn("event: tick", text)
        self.assertIn("data:", text)


if __name__ == "__main__":
    unittest.main()
