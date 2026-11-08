"""Cross-platform process governor and hardware pinning via psutil.

Configures execution priorities and CPU affinities for high-frequency sentry,
API server, and background evolutionary worker processes across Windows, Linux,
and containerized cloud environments with graceful degradation.
"""
from typing import Any, Dict, List, Optional
import os
import sys
import psutil


def set_priority(proc: psutil.Process, priority_level: str) -> bool:
    """Sets process priority class or nice value based on operating system."""
    level = priority_level.lower()
    is_windows = os.name == "nt" or sys.platform == "win32"
    try:
        if is_windows and hasattr(psutil, "NORMAL_PRIORITY_CLASS"):
            win_map = {
                "idle": getattr(psutil, "IDLE_PRIORITY_CLASS", 64),
                "below_normal": getattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS", 16384),
                "normal": getattr(psutil, "NORMAL_PRIORITY_CLASS", 32),
                "above_normal": getattr(psutil, "ABOVE_NORMAL_PRIORITY_CLASS", 32768),
                "high": getattr(psutil, "HIGH_PRIORITY_CLASS", 128),
                "realtime": getattr(psutil, "REALTIME_PRIORITY_CLASS", 256),
            }
            proc.nice(win_map.get(level, getattr(psutil, "NORMAL_PRIORITY_CLASS", 32)))
        else:
            posix_map = {
                "idle": 19,
                "below_normal": 10,
                "normal": 0,
                "above_normal": -5,
                "high": -10,
                "realtime": -20,
            }
            proc.nice(posix_map.get(level, 0))
        return True
    except (psutil.AccessDenied, psutil.NoSuchProcess, OSError, ValueError):
        return False


def set_affinity(proc: psutil.Process, cpu_cores: Optional[List[int]]) -> bool:
    """Sets CPU affinity mask with graceful degradation if unsupported or denied."""
    if cpu_cores is None:
        return True
    if not hasattr(proc, "cpu_affinity"):
        return True
    total_cpus = psutil.cpu_count() or 1
    valid_cores = [int(c) for c in cpu_cores if 0 <= c < total_cpus]
    if not valid_cores:
        valid_cores = [0]
    try:
        proc.cpu_affinity(valid_cores)
        return True
    except (AttributeError, NotImplementedError, psutil.AccessDenied, psutil.NoSuchProcess, OSError):
        return False


def configure_process(
    pid: int,
    priority_level: str = "normal",
    cpu_cores: Optional[List[int]] = None,
) -> bool:
    """Sets process priority level and CPU affinity with graceful fallback."""
    try:
        proc = psutil.Process(pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
        return False
    p_ok = set_priority(proc, priority_level)
    a_ok = set_affinity(proc, cpu_cores)
    return p_ok or a_ok


def get_core_topology() -> Dict[str, List[int]]:
    """Calculates standard core topology allocation for sentry, server, and workers."""
    total_cpus = psutil.cpu_count() or 1
    sentry_cores = [0, 1] if total_cpus >= 2 else [0]
    worker_cores = list(range(2, total_cpus)) if total_cpus > 2 else list(range(total_cpus))
    return {
        "sentry": sentry_cores,
        "server": sentry_cores,
        "worker": worker_cores,
    }


def configure_worker(pid: int, num_cores: Optional[int] = None) -> bool:
    """Configures worker process to below normal priority pinned to worker cores."""
    topology = get_core_topology()
    worker_cores = topology["worker"]
    if num_cores is not None and num_cores > 0:
        worker_cores = worker_cores[:num_cores]
    return configure_process(pid, priority_level="below_normal", cpu_cores=worker_cores)


def configure_sentry(pid: int) -> bool:
    """Configures risk sentry to high priority pinned to Core 0/1."""
    topology = get_core_topology()
    return configure_process(pid, priority_level="high", cpu_cores=topology["sentry"])


def configure_server(pid: int) -> bool:
    """Configures server process to normal priority pinned to Core 0/1."""
    topology = get_core_topology()
    return configure_process(pid, priority_level="normal", cpu_cores=topology["server"])


def get_process_info(pid: int) -> Dict[str, Any]:
    """Retrieves current process priority, affinity, and resource diagnostics."""
    try:
        proc = psutil.Process(pid)
        affinity = proc.cpu_affinity() if hasattr(proc, "cpu_affinity") else []
        return {
            "pid": pid,
            "nice": proc.nice(),
            "cpu_affinity": affinity,
            "cpu_percent": proc.cpu_percent(interval=None),
            "memory_mb": proc.memory_info().rss / (1024 * 1024),
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        return {"pid": pid, "error": "not_accessible"}


class ProcessGovernor:
    """Cross-platform hardware process governor and affinity manager."""

    @staticmethod
    def configure_process(
        pid: int,
        priority_level: str = "normal",
        cpu_cores: Optional[List[int]] = None,
    ) -> bool:
        """Sets process priority level and CPU affinity with graceful fallback."""
        return configure_process(pid, priority_level=priority_level, cpu_cores=cpu_cores)

    @classmethod
    def configure_worker(cls, pid: int, num_cores: Optional[int] = None) -> bool:
        """Configures worker process to below normal priority pinned to worker cores."""
        return configure_worker(pid, num_cores=num_cores)

    @classmethod
    def configure_sentry(cls, pid: int) -> bool:
        """Configures risk sentry to high priority pinned to Core 0/1."""
        return configure_sentry(pid)

    @classmethod
    def configure_server(cls, pid: int) -> bool:
        """Configures server process to normal priority pinned to Core 0/1."""
        return configure_server(pid)

    @staticmethod
    def get_process_info(pid: int) -> Dict[str, Any]:
        """Retrieves current process priority, affinity, and resource diagnostics."""
        return get_process_info(pid)

