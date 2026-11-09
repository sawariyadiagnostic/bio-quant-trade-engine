"""
Comprehensive TDD Verification Suite for ADHD Batch 5:
Cockpit Avionics, API Security & UI Sensory (ADHD-40 to ADHD-47).
"""
import os
import sys
import unittest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ui.server import app, DropTailQueue, ALLOWED_ORIGINS
from src.ui.manifold_service import compute_manifold_points


class TestADHDBatch5CockpitAvionics(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_adhd40_adhd41_audio_synth_synthesizer_invariants(self):
        """ADHD-40 & ADHD-41: audio_synth.js implements arming gate and master dynamics limiter."""
        synth_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ui", "static", "audio_synth.js"))
        self.assertTrue(os.path.exists(synth_path))
        with open(synth_path, "r", encoding="utf-8") as f:
            code = f.read()

        # ADHD-40: User gesture arming gate
        self.assertIn("this.isArmed = false;", code)
        self.assertIn("armAudio()", code)
        self.assertIn("audio-armed-pill", code)

        # ADHD-41: Master limiter and node cleanup
        self.assertIn("createDynamicsCompressor", code)
        self.assertIn("this.limiter", code)
        self.assertIn("osc.onended", code)
        self.assertIn("osc.disconnect()", code)

    def test_adhd42_droptail_queue_retains_critical_terminal_events(self):
        """ADHD-42: DropTailQueue prioritizes terminal/emergency events when full."""
        q = DropTailQueue(maxsize=3)

        # Fill with 3 regular telemetry ticks
        self.assertTrue(q.put_nowait("data: tick 1\n\n"))
        self.assertTrue(q.put_nowait("data: tick 2\n\n"))
        self.assertTrue(q.put_nowait("data: tick 3\n\n"))
        self.assertEqual(q.qsize(), 3)

        # Non-critical 4th tick is dropped
        self.assertFalse(q.put_nowait("data: tick 4\n\n"))
        self.assertEqual(q.dropped_count, 1)

        # Critical complete event MUST be accepted by discarding oldest item
        critical_event = 'event: complete\ndata: {"status": "COMPLETE", "gen": 10}\n\n'
        accepted = q.put_nowait(critical_event)
        self.assertTrue(accepted)
        self.assertEqual(q.qsize(), 3)

        # Confirm queue still yields items and contains the terminal event
        items = []
        while not q.empty():
            items.append(q.queue.get_nowait())
        self.assertIn(critical_event, items)

    def test_adhd43_ag_grid_upsert_logic(self):
        """ADHD-43: app.js performs true upsert (update if exists, add if new)."""
        app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ui", "static", "app.js"))
        self.assertTrue(os.path.exists(app_path))
        with open(app_path, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()

        self.assertIn("existingNode", code)
        self.assertIn("applyTransactionAsync({ update:", code)
        self.assertIn("applyTransactionAsync({ add:", code)

    def test_adhd44_panic_button_confirmation_latch(self):
        """ADHD-44: Panic button requires two-step confirmation and is visually segregated."""
        app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ui", "static", "app.js"))
        with open(app_path, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()

        # Confirmation guard in app.js
        self.assertIn("confirm(", code)
        self.assertIn("CONFIRM EMERGENCY PANIC LIQUIDATION", code)

        html_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ui", "static", "index.html"))
        with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()

        # Visual separation in index.html
        self.assertIn("bg-red-950/40", html)
        self.assertIn("PANIC (100% CASH)", html)

    def test_adhd45_cors_and_csrf_emergency_auth(self):
        """ADHD-45: Emergency endpoints require CSRF verification and reject unauthorized cross-origin requests."""
        # 1. CSRF token retrieval endpoint
        token_resp = self.client.get("/api/csrf-token")
        self.assertEqual(token_resp.status_code, 200)
        token = token_resp.json().get("token")
        self.assertTrue(bool(token))

        # 2. Freeze with valid token
        freeze_resp = self.client.post("/api/emergency/freeze", headers={"X-BioQuant-Token": token})
        self.assertEqual(freeze_resp.status_code, 200)
        self.assertTrue(freeze_resp.json().get("status") in ("FROZEN", "ALREADY_FROZEN"))

        # 3. Panic with valid token
        panic_resp = self.client.post("/api/emergency/panic", headers={"X-BioQuant-Token": token})
        self.assertEqual(panic_resp.status_code, 200)
        self.assertEqual(panic_resp.json().get("status"), "LIQUIDATION_COMPLETE")

        # 4. Freeze with malicious external origin and no token should be rejected (HTTP 403)
        malicious_resp = self.client.post(
            "/api/emergency/freeze",
            headers={"Origin": "http://malicious-phishing-site.com", "X-BioQuant-Token": "bad-token"}
        )
        self.assertEqual(malicious_resp.status_code, 403)

    def test_adhd46_responsive_hud_media_queries(self):
        """ADHD-46: index.html implements responsive flex-wrap and media queries for high-DPI viewports."""
        html_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ui", "static", "index.html"))
        with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()

        self.assertIn("@media (max-width: 1440px)", html)
        self.assertIn("flex-wrap", html)
        self.assertIn("telemetry-ribbon", html)

    def test_adhd47_manifold_service_candidate_capping(self):
        """ADHD-47: compute_manifold_points caps large candidate pools to top 30 champions."""
        # Generate 60 dummy elite dictionaries
        elites = []
        for i in range(60):
            elites.append({
                "ticker": "NIFTYBEES",
                "lookback": "1y",
                "fitness": 1.0 + (i * 0.05),
                "formula_str": f"ADD(FEATURE(Close), CONST({i}))"
            })

        res = compute_manifold_points(elites, ticker="NIFTYBEES")
        self.assertEqual(res["status"], "ok")
        # Candidate count must be capped to at most 30
        self.assertLessEqual(res["count"], 30)
        self.assertLessEqual(len(res["points"]), 30)
        self.assertLessEqual(len(res["cosine_matrix"]), 30)


if __name__ == "__main__":
    unittest.main()
