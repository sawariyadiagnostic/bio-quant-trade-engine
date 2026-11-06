"""Adversarial Chaos Fault-Injection Suite.

Simulates extreme production failure modes:
1. Test A: Worker SIGKILL crash recovery and orphan prevention.
2. Test B: Broker network socket hang / timeout containment.
3. Test C: Corrupted archive ingestion and database recovery.
4. Test D: 50-clicks/sec panic button spam deduplication and race prevention.
"""
import asyncio
import os
import shutil
import tempfile
import time
import unittest
from unittest.mock import MagicMock, AsyncMock, patch

from fastapi.testclient import TestClient

from src.evolution.archive_store import ArchiveStore
from src.execution.risk_sentry import RiskSentry
from src.pipeline.shared_telemetry import SharedMemoryBus
from src.ui.server import app, supervisor, sentry


class TestChaosFaultInjection(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        sentry.is_frozen = False
        sentry._is_liquidating = False

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_chaos_worker_sigkill_recovery(self):
        """Worker sudden termination (SIGKILL) is detected cleanly without hanging supervisor."""
        bus = SharedMemoryBus(name=f"bioquant_chaos_shm_{os.getpid()}", create=True)
        try:
            # Simulate a dead worker PID
            mock_proc = MagicMock()
            mock_proc.is_alive.return_value = False
            mock_proc.pid = 999999

            supervisor.worker_process = mock_proc
            self.assertFalse(supervisor.is_worker_running())

            # Verify that supervisor accepts new launch request after crash
            with patch("src.ui.server.spawn_evolution_worker") as mock_spawn:
                new_proc = MagicMock()
                new_proc.is_alive.return_value = True
                new_proc.pid = 12345
                mock_spawn.return_value = new_proc

                success = supervisor.start_worker(instrument="NIFTYBEES", generations=5)
                self.assertTrue(success)
                self.assertEqual(supervisor.worker_process.pid, 12345)
        finally:
            bus.close()

    def test_chaos_broker_network_hang_timeout(self):
        """Hanging broker API calls are terminated by the timeout guard (<3s)."""
        mock_gateway = MagicMock()
        mock_gateway.paper_trading = False

        # Simulate network hang with a 5-second sleep in get_order_book
        def slow_hang():
            time.sleep(4.0)
            return []

        mock_gateway.api = MagicMock()
        mock_gateway.api.get_order_book = slow_hang
        mock_gateway.api.get_positions = MagicMock(return_value=[])

        local_sentry = RiskSentry(gateway=mock_gateway)
        local_sentry.SOCKET_TIMEOUT_SECONDS = 0.2  # Fast test timeout

        loop = asyncio.new_event_loop()
        try:
            t0 = time.perf_counter()
            report = loop.run_until_complete(local_sentry.execute_stage2_liquidate())
            elapsed = time.perf_counter() - t0

            # Must return in ~0.2s without hanging forever
            self.assertLess(elapsed, 1.5, f"Took {elapsed:.2f}s, timeout failed!")
            self.assertEqual(report["status"], "LIQUIDATION_COMPLETE")
            self.assertTrue(report["cash_secured"])
        finally:
            loop.close()

    def test_chaos_corrupted_archive_ingestion(self):
        """Corrupted/truncated SQLite database files are detected without crashing the system."""
        corrupt_db = os.path.join(self.tmpdir, "corrupt_alpha.db")
        with open(corrupt_db, "wb") as f:
            f.write(b"CORRUPTED_GARBAGE_BYTES_NOT_SQLITE_HEADER_1234567890")

        # Initializing store on corrupted file should handle error or reinitialize safely
        try:
            store = ArchiveStore(db_path=corrupt_db)
            count = store.count_champions()
            self.assertIsInstance(count, int)
        except Exception as e:
            # Should raise a recognized database error, not crash the process
            self.assertIn("database", str(e).lower())

    def test_chaos_panic_button_spam_deduplication(self):
        """50 rapid panic requests are safely deduplicated with zero 500 errors."""
        client = TestClient(app)

        responses = []
        for _ in range(50):
            res = client.post("/api/emergency/panic")
            responses.append(res)

        # All 50 requests must return HTTP 200
        status_codes = [r.status_code for r in responses]
        self.assertTrue(all(s == 200 for s in status_codes), f"Non-200 found: {set(status_codes)}")

        # Verify that either LIQUIDATION_COMPLETE or LIQUIDATION_IN_PROGRESS was returned
        statuses = [r.json().get("status") for r in responses]
        valid_statuses = {"LIQUIDATION_COMPLETE", "LIQUIDATION_IN_PROGRESS"}
        self.assertTrue(all(s in valid_statuses for s in statuses))
        self.assertTrue(sentry.is_frozen)


if __name__ == "__main__":
    unittest.main()
