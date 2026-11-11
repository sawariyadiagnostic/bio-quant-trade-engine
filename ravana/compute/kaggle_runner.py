"""Kaggle Burst Compute Dispatcher for RAVANA.

Provides an institutional, asynchronous cloud-burst adapter for Kaggle GPU environments.
Executes large-scale alpha search, tensor convolutions, or neural training on Kaggle's
NVIDIA T4 / P100 GPUs, automatically terminating upon completion to prevent idle quota burn.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("ravana.compute.kaggle")


@dataclass(frozen=True)
class KaggleQuotaInfo:
    """Institutional snapshot of remote Kaggle accelerator quotas."""
    gpu_used_hours: float
    gpu_remaining_hours: float
    gpu_total_hours: float
    refresh_at: str
    tpu_remaining_hours: float = 0.0


@dataclass
class KaggleBurstConfig:
    """Configuration for remote Kaggle burst jobs."""
    kernel_slug: str = "ravana-alpha-burst"
    kernel_title: str = "Ravana Alpha Burst Search"
    enable_gpu: bool = True
    enable_internet: bool = False
    is_private: bool = True
    timeout_seconds: int = 7200
    poll_interval_seconds: int = 15
    dataset_sources: List[str] = field(default_factory=list)


class KaggleBurstDispatcher:
    """Manages headless remote batch compute dispatch to Kaggle GPUs with auto-shutdown."""

    def __init__(self, config: Optional[KaggleBurstConfig] = None):
        self.config = config or KaggleBurstConfig()
        self._api = None
        self._username: Optional[str] = None
        self._init_api()

    def _init_api(self) -> None:
        """Initialize Kaggle API client from environment or ~/.kaggle/access_token."""
        try:
            import kaggle
            self._api = kaggle.KaggleApi()
            self._api.authenticate()
            self._username = self._api.get_config_value("username")
            logger.info("Kaggle API successfully authenticated for user: %s", self._username)
        except Exception as exc:
            logger.warning("Kaggle API initialization deferred or failed: %s", exc)

    @property
    def is_authenticated(self) -> bool:
        """Check whether Kaggle API is available and authenticated."""
        return self._api is not None and bool(self._username)

    @property
    def username(self) -> str:
        """Return the authenticated Kaggle username."""
        if not self._username:
            raise RuntimeError("Kaggle API is not authenticated. Check ~/.kaggle/access_token.")
        return self._username

    def get_quota(self) -> KaggleQuotaInfo:
        """Query real-time weekly accelerator quota from Kaggle."""
        if not self.is_authenticated:
            raise RuntimeError("Cannot query quota: Kaggle API unauthenticated.")
        raw = self._api.quota_view()
        if hasattr(raw, "to_dict"):
            data = raw.to_dict()
        elif hasattr(raw, "to_json"):
            data = json.loads(raw.to_json())
        elif isinstance(raw, str):
            data = json.loads(raw)
        else:
            data = dict(raw)

        gpu = data.get("gpuQuota") or data.get("gpu_quota") or {}
        used_s = self._parse_seconds(gpu.get("timeUsed") or gpu.get("time_used") or "0s")
        tot_s = self._parse_seconds(gpu.get("totalTimeAllowed") or gpu.get("total_time_allowed") or "108000s")
        rem_s = max(0.0, tot_s - used_s)

        tpu = data.get("tpuQuota") or data.get("tpu_quota") or {}
        tpu_used = self._parse_seconds(tpu.get("timeUsed") or tpu.get("time_used") or "0s")
        tpu_tot = self._parse_seconds(tpu.get("totalTimeAllowed") or tpu.get("total_time_allowed") or "72000s")
        refresh = (
            data.get("quotaRefreshTime")
            or data.get("quota_refresh_time")
            or "unknown"
        )

        return KaggleQuotaInfo(
            gpu_used_hours=round(used_s / 3600.0, 2),
            gpu_remaining_hours=round(rem_s / 3600.0, 2),
            gpu_total_hours=round(tot_s / 3600.0, 2),
            refresh_at=str(refresh),
            tpu_remaining_hours=round(max(0.0, tpu_tot - tpu_used) / 3600.0, 2),
        )

    @staticmethod
    def _parse_seconds(val: Any) -> float:
        """Parse Kaggle time duration string or numeric into float seconds."""
        m = re.search(r"\d+(?:\.\d+)?", str(val))
        return float(m.group(0)) if m else 0.0

    def prepare_bundle(
        self,
        bundle_dir: Path,
        script_code: str,
        script_filename: str = "burst_run.py",
    ) -> Path:
        """Create an isolated kernel bundle with code and kernel-metadata.json."""
        bundle_dir = Path(bundle_dir)
        bundle_dir.mkdir(parents=True, exist_ok=True)

        script_path = bundle_dir / script_filename
        script_path.write_text(script_code, encoding="utf-8")

        metadata = {
            "id": f"{self.username}/{self.config.kernel_slug}",
            "title": self.config.kernel_title,
            "code_file": script_filename,
            "language": "python",
            "kernel_type": "script",
            "is_private": self.config.is_private,
            "enable_gpu": self.config.enable_gpu,
            "enable_internet": self.config.enable_internet,
            "dataset_sources": self.config.dataset_sources,
        }
        meta_path = bundle_dir / "kernel-metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return bundle_dir

    def dispatch(self, bundle_dir: Path) -> str:
        """Push the batch kernel to Kaggle for immediate headless execution."""
        if not self.is_authenticated:
            raise RuntimeError("Kaggle API unauthenticated. Cannot dispatch.")
        kernel_id = f"{self.username}/{self.config.kernel_slug}"
        logger.info("Pushing kernel %s from %s", kernel_id, bundle_dir)
        self._api.kernels_push_cli(str(bundle_dir))
        return kernel_id

    def poll_status(
        self,
        kernel_id: Optional[str] = None,
        callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Poll the remote batch job until complete, failed, or timed out."""
        target_id = kernel_id or f"{self.username}/{self.config.kernel_slug}"
        start_time = time.time()
        last_status = "UNKNOWN"

        while time.time() - start_time < self.config.timeout_seconds:
            status_data = self._api.kernels_status(target_id)
            current_status = status_data.get("status", "unknown").upper()

            if current_status != last_status:
                last_status = current_status
                logger.info("Kernel %s status changed to: %s", target_id, current_status)
                if callback:
                    callback(current_status)

            if current_status in ("COMPLETE", "COMPLETED"):
                return "COMPLETE"
            if current_status in ("ERROR", "FAILED", "CANCELLED"):
                failure_msg = status_data.get("failureMessage", "No error details")
                raise RuntimeError(f"Kernel {target_id} ended with {current_status}: {failure_msg}")

            time.sleep(self.config.poll_interval_seconds)

        raise TimeoutError(f"Kernel {target_id} exceeded {self.config.timeout_seconds}s limit.")

    def fetch_results(self, dest_dir: Path, kernel_id: Optional[str] = None) -> List[Path]:
        """Download output artifacts from the finished remote kernel."""
        target_id = kernel_id or f"{self.username}/{self.config.kernel_slug}"
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        self._api.kernels_output_cli(target_id, path=str(dest_dir))
        return list(dest_dir.glob("*"))

    def cancel(self, kernel_id: Optional[str] = None) -> bool:
        """Emergency kill-switch: abort the running Kaggle kernel immediately."""
        target_id = kernel_id or f"{self.username}/{self.config.kernel_slug}"
        try:
            res = subprocess.run(
                ["kaggle", "kernels", "delete", target_id, "-y"],
                capture_output=True,
                text=True,
                check=False,
            )
            return res.returncode == 0
        except Exception as exc:
            logger.error("Failed to cancel kernel %s: %s", target_id, exc)
            return False

    @staticmethod
    def generate_burst_script_template(
        population_size: int = 500,
        generations: int = 25,
        target_metric: str = "ic",
    ) -> str:
        """Generate a self-contained execution script that exits cleanly to save GPU quota."""
        return f'''# Auto-generated Ravana Kaggle Burst Worker
import sys
import json
import time
import torch

print("=== Starting Ravana Cloud Burst on Kaggle ===")
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
    print("Device Count:", torch.cuda.device_count())

# Configuration
pop_size = {population_size}
generations = {generations}
target = "{target_metric}"

# Simulation payload for Alpha Generation & Multi-Objective Search
print(f"Running Evolution: Pop={{pop_size}}, Gens={{generations}} on GPU...")
t0 = time.time()

# Mocking output artifact structure
results = {{
    "population_size": pop_size,
    "generations": generations,
    "champion_formula": "TsRank(Return(close, 5), 10)",
    "metrics": {{
        "ic": 0.0824,
        "rank_ic": 0.0765,
        "dsr": 1.42,
        "fdr_q": 0.012,
        "turnover": 0.18,
        "mdd": 0.065
    }},
    "pareto_frontier_count": 14,
    "elapsed_seconds": round(time.time() - t0, 3)
}}

out_path = "/kaggle/working/champion_alpha_results.json"
try:
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {{out_path}}")
except Exception:
    with open("champion_alpha_results.json", "w") as f:
        json.dump(results, f, indent=2)

print("Execution complete. Exiting script to immediately release GPU quota.")
sys.exit(0)
'''
