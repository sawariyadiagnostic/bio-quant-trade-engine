"""Unit tests for Cross-Platform Process Governor."""
import os
import sys
import psutil

test_dir = os.path.dirname(os.path.abspath(__file__))
trade_new_dir = os.path.abspath(os.path.join(test_dir, ".."))
if "" in sys.path:
    sys.path.remove("")
cwd = os.getcwd()
if cwd in sys.path and os.path.abspath(cwd) != trade_new_dir:
    sys.path.remove(cwd)
if trade_new_dir not in sys.path:
    sys.path.insert(0, trade_new_dir)


def test_configure_process_current_pid():
    """Verify configure_process successfully sets priority on current process."""
    from src.pipeline.process_governor import configure_process

    pid = os.getpid()
    ok = configure_process(pid, priority_level="below_normal")
    assert ok is True

    # Revert to normal
    ok_normal = configure_process(pid, priority_level="normal")
    assert ok_normal is True
    print("[PASS] test_configure_process_current_pid")


def test_configure_worker_sentry_server():
    """Verify configure_worker, configure_sentry, and configure_server dispatch properly."""
    from src.pipeline.process_governor import (
        ProcessGovernor,
        configure_worker,
        configure_sentry,
        configure_server,
    )

    pid = os.getpid()
    was_main = ProcessGovernor.is_main_process(pid)
    if was_main:
        ProcessGovernor.unregister_main_process(pid)

    try:
        # Configure as worker
        worker_ok = configure_worker(pid, num_cores=1)
        assert worker_ok is True

        # Configure as sentry (high priority)
        sentry_ok = configure_sentry(pid)
        assert sentry_ok is True

        # Configure as server (normal priority)
        server_ok = configure_server(pid)
        assert server_ok is True
    finally:
        if was_main:
            ProcessGovernor.register_main_process(pid)
        # Reset affinity to all cores for clean test state
        proc = psutil.Process(pid)
        if hasattr(proc, "cpu_affinity"):
            try:
                total_cpus = psutil.cpu_count() or 1
                proc.cpu_affinity(list(range(total_cpus)))
            except Exception:
                pass

    print("[PASS] test_configure_worker_sentry_server")


def test_graceful_degradation_invalid_pid():
    """Verify graceful handling of non-existent PID or OS restrictions without crashing."""
    from src.pipeline.process_governor import (
        configure_process,
        get_process_info,
        set_affinity,
    )

    # PID 99999999 does not exist
    invalid_pid = 99999999
    res = configure_process(invalid_pid, priority_level="normal")
    assert res is False

    info = get_process_info(invalid_pid)
    assert "error" in info

    # Set affinity with out-of-range cores on real process should clamp/fallback gracefully
    proc = psutil.Process(os.getpid())
    aff_ok = set_affinity(proc, [999, 1000])
    assert isinstance(aff_ok, bool)
    print("[PASS] test_graceful_degradation_invalid_pid")


def test_core_topology_and_process_info():
    """Verify core topology mapping and diagnostic inspection."""
    from src.pipeline.process_governor import get_core_topology, get_process_info

    topology = get_core_topology()
    assert "sentry" in topology
    assert "server" in topology
    assert "worker" in topology

    total = psutil.cpu_count() or 1
    assert len(topology["sentry"]) <= 2
    assert all(0 <= c < total for c in topology["sentry"])
    assert all(0 <= c < total for c in topology["worker"])

    info = get_process_info(os.getpid())
    assert info["pid"] == os.getpid()
    assert "nice" in info
    assert "memory_mb" in info
    assert info["memory_mb"] > 0
    print("[PASS] test_core_topology_and_process_info")


def load_tests(loader, standard_tests, pattern):
    import inspect
    import unittest
    suite = unittest.TestSuite()
    current_module = sys.modules[__name__]
    for name, func in inspect.getmembers(current_module, inspect.isfunction):
        if name.startswith("test_"):
            suite.addTest(unittest.FunctionTestCase(func))
    return suite


if __name__ == "__main__":
    import unittest
    unittest.main()
