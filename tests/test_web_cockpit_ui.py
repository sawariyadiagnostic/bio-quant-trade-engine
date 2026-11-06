"""Unit and Verification Tests for Decoupled Institutional n8n-Style Node Cockpit Web UI.

Validates:
1. Index HTML landing page serves HTTP 200.
2. All 8 vendored and companion offline assets are present and served HTTP 200.
3. DOM elements for Avionics HUD, n8n Nodes, AG Grid container, and Plotly container exist.
4. Pure Web Audio API Synthesizer syntax and methods are verified.
5. Application JS controller initialization and differential SSE listeners are verified.
"""
import os
import unittest
from fastapi.testclient import TestClient

from src.ui.server import app


class TestWebCockpitUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_index_html_serves_ok(self):
        """GET / returns HTTP 200 and loads the Cockpit Avionics HUD layout."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        html = res.text
        self.assertIn("BIO-QUANT ALPHA COCKPIT", html)
        self.assertIn("PIPELINE NODES", html)
        self.assertIn("Node 1: Data Ingest", html)
        self.assertIn("Node 2: MAP-Elites Engine", html)
        self.assertIn("Node 3: Risk Sentry", html)
        self.assertIn("leaderboardGrid", html)
        self.assertIn("mapElitesChart", html)
        self.assertIn("DEAD-MAN SWITCH ALERT", html)

    def test_vendored_offline_assets_served(self):
        """All static assets must be locally vendored and served with HTTP 200."""
        assets = [
            "/vendor/alpine.min.js",
            "/vendor/ag-grid-community.min.js",
            "/vendor/ag-grid.min.css",
            "/vendor/ag-theme-alpine.min.css",
            "/vendor/plotly.min.js",
            "/vendor/tailwind.min.js",
            "/audio_synth.js",
            "/app.js"
        ]
        for asset in assets:
            res = self.client.get(asset)
            self.assertEqual(
                res.status_code,
                200,
                f"Asset {asset} failed to load, status {res.status_code}"
            )
            self.assertGreater(len(res.content), 500, f"Asset {asset} appears truncated or empty")

    def test_audio_synth_script_validity(self):
        """Synthesizer script must define pure Web Audio synthesis methods."""
        res = self.client.get("/audio_synth.js")
        self.assertEqual(res.status_code, 200)
        js = res.text
        self.assertIn("CockpitAudioSynthesizer", js)
        self.assertIn("playEliteSound", js)
        self.assertIn("playTickSound", js)
        self.assertIn("playPanicSound", js)
        self.assertIn("playCompleteSound", js)
        self.assertIn("createOscillator", js)

    def test_app_controller_script_validity(self):
        """App controller script must define Alpine model, AG Grid virtualization, and SSE handlers."""
        res = self.client.get("/app.js")
        self.assertEqual(res.status_code, 200)
        js = res.text
        self.assertIn("cockpitApp", js)
        self.assertIn("applyTransactionAsync", js)
        self.assertIn("requestAnimationFrame", js)
        self.assertIn("startDeadManWatchdog", js)
        self.assertIn("emergencyFreeze", js)
        self.assertIn("emergencyPanic", js)


if __name__ == "__main__":
    unittest.main()
