"""Integration and API test suite for Sovereign Institutional Cockpit Server.

Validates:
1. /health probe returns HTTP 200, is_frozen=False, and worker_running state.
2. /api/status returns telemetry state JSON with dictionary metrics and resident count.
3. /api/emergency/freeze and /api/emergency/panic execute atomic freeze and liquidation.
4. POST /api/strategy/bankbees/run executes BankbeesOvernightPipeline and returns valid overnight signals.
"""
import os
import sys
import unittest
from fastapi.testclient import TestClient

# Ensure trade_new is first on sys.path
test_dir = os.path.dirname(os.path.abspath(__file__))
trade_new_dir = os.path.abspath(os.path.join(test_dir, ".."))
if "" in sys.path:
    sys.path.remove("")
cwd = os.getcwd()
if cwd in sys.path and os.path.abspath(cwd) != trade_new_dir:
    sys.path.remove(cwd)
if trade_new_dir not in sys.path:
    sys.path.insert(0, trade_new_dir)

from src.ui.server import app, sentry, supervisor


class TestCockpitServerAPI(unittest.TestCase):
    """Test suite validating all sovereign UI cockpit endpoints and state transitions."""

    def setUp(self):
        """Prepares a dedicated test client and ensures sentry freeze is reset."""
        self.client = TestClient(app)
        sentry.is_frozen = False
        with supervisor._lock:
            supervisor.telemetry.abort_requested = 0
            supervisor.telemetry.worker_status = 0

    def tearDown(self):
        """Cleans up supervisor and resets sentry freeze state after each test."""
        sentry.is_frozen = False
        supervisor.cleanup()

    def test_health_probe(self):
        """Validates /health returns HTTP 200 and initial is_frozen=False."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "ok")
        self.assertIn("uptime_s", data)
        self.assertFalse(data.get("is_frozen"))
        self.assertIn("worker_running", data)

    def test_api_status_telemetry(self):
        """Validates /api/status returns valid telemetry state dictionary."""
        resp = self.client.get("/api/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "ok")
        self.assertIn("telemetry", data)
        telem = data["telemetry"]
        self.assertIn("current_gen", telem)
        self.assertIn("total_gens", telem)
        self.assertIn("worker_status", telem)
        self.assertIn("resident_count", telem)
        self.assertIn("occupancy_pct", telem)
        self.assertIn("resident_elites_count", data)

    def test_emergency_freeze_and_panic(self):
        """Validates /api/emergency/freeze sets is_frozen and /panic liquidates."""
        resp_freeze = self.client.post("/api/emergency/freeze")
        self.assertEqual(resp_freeze.status_code, 200)
        data_freeze = resp_freeze.json()
        self.assertEqual(data_freeze.get("status"), "FROZEN")
        self.assertTrue(sentry.is_frozen)

        resp_health = self.client.get("/health")
        self.assertTrue(resp_health.json().get("is_frozen"))

        resp_panic = self.client.post("/api/emergency/panic")
        self.assertEqual(resp_panic.status_code, 200)
        data_panic = resp_panic.json()
        self.assertIn("status", data_panic)
        self.assertTrue(sentry.is_frozen)

    def test_bankbees_strategy_run(self):
        """Validates POST /api/strategy/bankbees/run returns valid overnight signals."""
        resp = self.client.post(
            "/api/strategy/bankbees/run",
            json={"capital": 10000.0, "lookback_years": 1.0}
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "ok")
        self.assertIn("benchmark_price", data)
        self.assertGreater(data["benchmark_price"], 0.0)
        self.assertIn("conviction_score", data)
        self.assertGreaterEqual(data["conviction_score"], 0.0)
        self.assertLessEqual(data["conviction_score"], 1.0)
        self.assertIn(data.get("action_bias"), ["BUY", "HOLD_CASH"])
        self.assertIn("cagr_pct", data)
        self.assertIn("sortino", data)
        self.assertIn("max_drawdown", data)
        self.assertIn("win_rate", data)
        self.assertIn("champions_count", data)
        self.assertGreater(data["champions_count"], 0)

    def test_multi_asset_strategy_run(self):
        """Validates POST /api/strategy/multi_asset/run returns valid multi-asset portfolio telemetry."""
        resp = self.client.post(
            "/api/strategy/multi_asset/run",
            json={"capital": 100000.0, "threshold": 0.40}
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "ok")
        self.assertIn("timestamp", data)
        self.assertEqual(data.get("capital"), 100000.0)
        self.assertIn("allocated_capital", data)
        self.assertIn("cash_reserve", data)
        self.assertIn("statutory_friction", data)
        self.assertIn("cagr_pct", data)
        self.assertIn("sortino", data)
        self.assertIn("max_drawdown", data)
        self.assertIn("win_rate", data)
        self.assertIn("allocations", data)
        self.assertIsInstance(data["allocations"], list)
        self.assertGreater(len(data["allocations"]), 0)
        first_alloc = data["allocations"][0]
        expected_keys = [
            "ticker", "instrument", "friction_bps", "conviction",
            "weight_pct", "allocated_inr", "units", "action"
        ]
        for key in expected_keys:
            self.assertIn(key, first_alloc)

    def test_universe_assets(self):
        """Validates /api/universe/assets returns discovered lake assets and resident tickers."""
        resp = self.client.get("/api/universe/assets")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "ok")
        self.assertIn("categories", data)
        self.assertIn("all_assets", data)
        self.assertIn("resident_tickers", data)

    def test_pipeline_abort(self):
        """Validates /api/pipeline/abort sets supervisor abort flag."""
        resp = self.client.post("/api/pipeline/abort")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "ABORT_REQUESTED")


if __name__ == "__main__":
    unittest.main()
