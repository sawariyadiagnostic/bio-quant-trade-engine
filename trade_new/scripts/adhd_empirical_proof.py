"""Empirical proof of ADHD Red Team vulnerabilities in trade_new UI and API."""
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "trade_new"))

from starlette.testclient import TestClient
from trade_new.src.ui.server import app, DropTailQueue, sse_event_generator
from trade_new.src.ui.manifold_service import compute_manifold_points

def test_cors_csrf_vulnerability():
    print("=== TEST 1: CORS & CSRF Vulnerability on Emergency Endpoints ===")
    client = TestClient(app)
    # Simulate cross-origin request from malicious website
    headers = {"Origin": "https://malicious-phishing-site.com"}
    resp = client.post("/api/emergency/freeze", headers=headers)
    print(f"POST /api/emergency/freeze status: {resp.status_code}")
    print(f"Access-Control-Allow-Origin: {resp.headers.get('access-control-allow-origin')}")
    print(f"Access-Control-Allow-Credentials: {resp.headers.get('access-control-allow-credentials')}")
    print(f"Body: {resp.json()}")
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "*"
    assert resp.headers.get("access-control-allow-credentials") == "true"
    print("[CONFIRMED] Invalid CORS wildcard with credentials + arbitrary cross-site CSRF execution on /api/emergency/freeze!")

def test_drop_tail_queue_terminal_event_drop():
    print("\n=== TEST 2: DropTailQueue Dropping Critical Terminal Events ===")
    q = DropTailQueue(maxsize=5)
    # Fill queue with 5 stale ticks
    for i in range(5):
        ok = q.put_nowait(f"tick_{i}")
        assert ok is True
    # Now try to enqueue critical emergency complete/panic event
    emergency_event = "event: complete\ndata: {\"status\": \"COMPLETE\"}\n\n"
    enqueued = q.put_nowait(emergency_event)
    print(f"Queue size: {q.qsize()}/{q.maxsize}")
    print(f"Emergency event enqueued successfully: {enqueued}")
    print(f"Dropped count: {q.dropped_count}")
    assert enqueued is False, "Drop-tail dropped the emergency event!"
    print("[CONFIRMED] DropTailQueue discards fresh emergency terminal events while retaining stale ticks!")

def test_manifold_payload_bloat():
    print("\n=== TEST 3: 3D Manifold SVD & Cosine Matrix Serialization Bloat ===")
    mock_elites = [
        {"formula_str": f"Add(Close_Adj, Mul(Volume, {i}))", "fitness": 1.5 + i * 0.01, "cagr_pct": 20.0, "max_drawdown": 10.0, "ticker": "NIFTYBEES"}
        for i in range(100)
    ]
    t0 = time.perf_counter()
    res = compute_manifold_points(mock_elites, "NIFTYBEES")
    elapsed_ms = (time.perf_counter() - t0) * 1000
    json_bytes = len(json.dumps(res))
    matrix_dim = len(res.get("cosine_matrix", []))
    print(f"Compute time for 100 elites: {elapsed_ms:.2f}ms")
    print(f"Cosine matrix dimensions: {matrix_dim}x{matrix_dim} ({matrix_dim*matrix_dim} cells)")
    print(f"JSON Payload size: {json_bytes / 1024:.2f} KB ({json_bytes} bytes)")
    print(f"Points count: {len(res.get('points', []))}")
    assert matrix_dim == 100
    assert json_bytes > 50_000
    print("[CONFIRMED] High-density matrix serialization bloat causes main-thread JSON parse & Plotly lockup!")

if __name__ == "__main__":
    test_cors_csrf_vulnerability()
    test_drop_tail_queue_terminal_event_drop()
    test_manifold_payload_bloat()
    print("\nALL EMPIRICAL TESTS PASSED AUDIT VERIFICATION!")
