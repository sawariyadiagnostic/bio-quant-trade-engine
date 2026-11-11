"""Tests for Kaggle Burst Compute Dispatcher using standard unittest."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from ravana.compute.kaggle_runner import (
    KaggleBurstConfig,
    KaggleBurstDispatcher,
    KaggleQuotaInfo,
)


class TestKaggleRunner(unittest.TestCase):
    """Test suite for Kaggle headless burst compute runner."""

    def test_parse_seconds(self):
        self.assertEqual(KaggleBurstDispatcher._parse_seconds("3600s"), 3600.0)
        self.assertEqual(KaggleBurstDispatcher._parse_seconds("21896.99.0s"), 21896.99)
        self.assertEqual(KaggleBurstDispatcher._parse_seconds("0s"), 0.0)
        self.assertEqual(KaggleBurstDispatcher._parse_seconds("invalid"), 0.0)

    def test_prepare_bundle(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_path = Path(tmp_dir)
            cfg = KaggleBurstConfig(kernel_slug="test-slug", kernel_title="Test Title")
            dispatcher = KaggleBurstDispatcher(config=cfg)
            dispatcher._username = "testuser"
            dispatcher._api = MagicMock()

            code = "print('hello world')"
            out_dir = dispatcher.prepare_bundle(bundle_path, code, "run.py")

            self.assertTrue((out_dir / "run.py").exists())
            self.assertEqual((out_dir / "run.py").read_text(encoding="utf-8"), code)

            meta_file = out_dir / "kernel-metadata.json"
            self.assertTrue(meta_file.exists())
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            self.assertEqual(meta["id"], "testuser/test-slug")
            self.assertEqual(meta["kernel_type"], "script")
            self.assertTrue(meta["enable_gpu"])
            self.assertEqual(meta["code_file"], "run.py")

    def test_generate_burst_script_template(self):
        script = KaggleBurstDispatcher.generate_burst_script_template(
            population_size=1000,
            generations=50,
            target_metric="rank_ic",
        )
        self.assertIn("sys.exit(0)", script)
        self.assertIn("pop_size = 1000", script)
        self.assertIn("generations = 50", script)
        self.assertIn("torch.cuda.is_available()", script)

    def test_quota_retrieval_mocked(self):
        mock_api = MagicMock()
        mock_api.quota_view.return_value = json.dumps({
            "quotaRefreshTime": "2026-11-14T00:00:00Z",
            "gpuQuota": {
                "timeUsed": "7200s",
                "totalTimeAllowed": "108000s",
            },
            "tpuQuota": {
                "timeUsed": "0s",
                "totalTimeAllowed": "72000s",
            },
        })

        dispatcher = KaggleBurstDispatcher()
        dispatcher._api = mock_api
        dispatcher._username = "testuser"

        quota = dispatcher.get_quota()
        self.assertIsInstance(quota, KaggleQuotaInfo)
        self.assertEqual(quota.gpu_used_hours, 2.0)
        self.assertEqual(quota.gpu_total_hours, 30.0)
        self.assertEqual(quota.gpu_remaining_hours, 28.0)
        self.assertEqual(quota.refresh_at, "2026-11-14T00:00:00Z")

    def test_live_quota_query(self):
        """Verify live authentication with real Kaggle token if available."""
        dispatcher = KaggleBurstDispatcher()
        if dispatcher.is_authenticated:
            quota = dispatcher.get_quota()
            self.assertGreater(quota.gpu_total_hours, 0)
            self.assertGreaterEqual(quota.gpu_remaining_hours, 0)
            self.assertGreaterEqual(quota.gpu_used_hours, 0)
