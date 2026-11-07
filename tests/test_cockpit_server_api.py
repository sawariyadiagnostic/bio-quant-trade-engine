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

    def test_sse_handles_bytes_without_type_error(self):
        """Proves SSE event generator safely serializes BLOB/bytes fields without TypeError."""
        raw_elite_with_bytes = {
            "ticker": "NIFTYBEES",
            "lookback": "1y",
            "niche_row": 5,
            "niche_col": 7,
            "fitness": 12.34,
            "net_return_pct": 56.78,
            "cagr_pct": 56.78,
            "max_drawdown": 4.5,
            "win_rate": 65.0,
            "turnover": 0.5,
            "volatility": 0.15,
            "sortino": 2.1,
            "profit_factor": 1.8,
            "trades_count": 42,
            "formula_str": "RSI_14 + EMA_10",
            "behavior_signature": b"\x01\x02\x03\x04\xff\xfe"  # raw bytes
        }
        with patch.object(supervisor.archive_store, "count_champions", side_effect=[0, 1, 1]):
            with patch.object(supervisor.archive_store, "get_recent_elites", return_value=[raw_elite_with_bytes]):
                response = self.client.get("/api/stream/evolution?max_ticks=2")
                self.assertEqual(response.status_code, 200)
                text = response.text
                self.assertIn("event: elite_added", text)
                self.assertIn("data:", text)
                # Ensure no unhandled 500 error occurred
                self.assertNotIn("Internal Server Error", text)

    def test_elites_grid_endpoint(self):
        """GET /api/elites/grid returns dense 20x20 matrix with metrics and hover metadata."""
        res = self.client.get("/api/elites/grid?ticker=NIFTYBEES&metric=fitness")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["ticker"], "NIFTYBEES")
        self.assertEqual(len(data["z"]), 20)
        self.assertEqual(len(data["z"][0]), 20)
        self.assertEqual(len(data["hover_text"]), 20)
        self.assertEqual(len(data["hover_text"][0]), 20)
        self.assertEqual(len(data["x_bins"]), 20)
        self.assertEqual(len(data["y_bins"]), 20)
        self.assertGreaterEqual(data["occupied_count"], 0)
        self.assertGreaterEqual(data["occupancy_pct"], 0.0)

    def test_universe_assets_endpoint(self):
        """GET /api/universe/assets returns categorized instruments from lake and database."""
        res = self.client.get("/api/universe/assets")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ok")
        self.assertGreater(data["total_count"], 50)
        self.assertIn("categories", data)
        self.assertIn("resident_tickers", data)
        self.assertIn("NIFTYBEES", data["resident_tickers"])

    def test_elites_manifold_endpoint(self):
        """GET /api/elites/manifold returns 3D points, clusters, and pairwise cosine matrix."""
        res = self.client.get("/api/elites/manifold?ticker=NIFTYBEES&method=fast&limit=20")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("points", data)
        self.assertIn("cosine_matrix", data)
        self.assertIn("compression_ratio", data)
        self.assertIn("clusters_count", data)
        if len(data["points"]) > 0:
            first = data["points"][0]
            self.assertIn("x", first)
            self.assertIn("y", first)
            self.assertIn("z", first)
            self.assertIn("cluster", first)

    def test_evolve_with_seed(self):
        """POST /api/pipeline/evolve accepts deterministic/random seed and passes it to worker."""
        with patch.object(supervisor, "start_worker", return_value=True) as mock_start:
            payload = {
                "instrument": "BANKBEES",
                "generations": 5,
                "population_size": 20,
                "seed": 12345
            }
            res = self.client.post("/api/pipeline/evolve", json=payload)
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["status"], "STARTED")
            self.assertEqual(data["seed"], 12345)
            mock_start.assert_called_once_with(
                instrument="BANKBEES",
                generations=5,
                population_size=20,
                lookback_years=1.0,
                seed=12345
            )


if __name__ == "__main__":
    unittest.main()
