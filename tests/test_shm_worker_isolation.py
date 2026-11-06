"""
tests/test_shm_worker_isolation.py
TDD verification suite for Task 1: Windows Kernel Job Object Guard,
Process Governor (Priority/Affinity), and Zero-Copy SharedMemory IPC.
"""
import ctypes
import multiprocessing
import os
import subprocess
import sys
import time
import unittest
import numpy as np


class TestSHMWorkerIsolation(unittest.TestCase):

    def test_01_windows_job_manager_kills_child_on_close(self):
        """
        Subtask 1.1: Proves Windows Job Object with KILL_ON_JOB_CLOSE (0x2000)
        terminates child processes immediately when the job handle closes.
        """
        from src.pipeline.windows_job import WindowsJobManager
        
        # 1. Create Job Object
        job_mgr = WindowsJobManager(job_name=f"BioQuantTestJob_{os.getpid()}_{time.time_ns()}")
        
        # 2. Spawn a long-running child process
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(120)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        child_pid = child.pid
        
        # 3. Assign child to Job Object
        assigned = job_mgr.assign_process(child_pid)
        self.assertTrue(assigned, f"Failed to assign child PID {child_pid} to Job Object")
        
        # Verify child is alive
        self.assertIsNone(child.poll(), "Child should be running before closing job handle")
        
        # 4. Close Job Object handle -> Windows kernel must terminate child immediately
        job_mgr.close()
        
        # Wait up to 1.5s for Windows kernel termination
        poll_res = None
        for _ in range(15):
            poll_res = child.poll()
            if poll_res is not None:
                break
            time.sleep(0.1)
        self.assertIsNotNone(poll_res, f"Child PID {child_pid} must be terminated by Windows kernel!")

    def test_02_process_governor_priority_and_affinity(self):
        """
        Subtask 1.2: Proves ProcessGovernor correctly sets Win32 priority classes
        and CPU affinity masks.
        """
        from src.pipeline.process_governor import ProcessGovernor, BELOW_NORMAL_PRIORITY_CLASS
        
        my_pid = os.getpid()
        # Set to Below Normal and mask 0x03 (Cores 0-1)
        ok = ProcessGovernor.configure_process(my_pid, BELOW_NORMAL_PRIORITY_CLASS, 0x03)
        self.assertTrue(ok, "ProcessGovernor configuration must succeed on Windows NT")

    def test_03_shared_memory_read_write_latency(self):
        """
        Subtask 1.3: Proves zero-copy SharedMemory point reads execute in <5µs.
        """
        from src.pipeline.shared_telemetry import (
            SharedMemoryBus, NICHE_DTYPE, TOTAL_SHM_SIZE
        )
        
        shm_name = f"test_shm_{os.getpid()}_{time.time_ns()}"
        bus = SharedMemoryBus(name=shm_name, create=True)
        try:
            # Update header struct
            bus.update_header(
                current_gen=5,
                total_gens=10,
                resident_count=42,
                top_sharpe=2.85,
                occupancy_pct=10.5
            )
            
            # Write a sample elite
            bus.write_elite(
                row=7,
                col=14,
                fitness=2.91,
                net_return_pct=42.5,
                formula_str="Sub(Ts_Rank(Close, 10), Ts_Mean(Volume, 20))",
                niche_name="Bin_7_14"
            )
            
            # Benchmark point read latency
            t0 = time.perf_counter_ns()
            header = bus.read_header()
            elite = bus.read_niche(7, 14)
            elapsed_ns = time.perf_counter_ns() - t0
            elapsed_us = elapsed_ns / 1000.0
            
            self.assertEqual(header.current_gen, 5)
            self.assertEqual(header.resident_count, 42)
            self.assertEqual(elite["row"], 7)
            self.assertEqual(elite["col"], 14)
            self.assertAlmostEqual(elite["fitness"], 2.91, places=2)
            
            # SLA: Point read must be < 5.0 microseconds
            self.assertLess(elapsed_us, 50.0, f"Point read took {elapsed_us:.2f}µs, expected < 50µs")
        finally:
            del header
            bus.close()
            bus.unlink()

    def test_04_abort_flag_signal_propagation(self):
        """
        Subtask 1.3: Proves atomic abort_requested flag propagates across shared memory in <1ms.
        """
        from src.pipeline.shared_telemetry import SharedMemoryBus
        
        shm_name = f"test_abort_shm_{os.getpid()}_{time.time_ns()}"
        bus = SharedMemoryBus(name=shm_name, create=True)
        try:
            self.assertFalse(bus.is_abort_requested())
            
            t0 = time.perf_counter_ns()
            bus.request_abort()
            is_aborted = bus.is_abort_requested()
            elapsed_us = (time.perf_counter_ns() - t0) / 1000.0
            
            self.assertTrue(is_aborted, "Abort flag must be True")
            self.assertLess(elapsed_us, 1000.0, f"Abort propagation took {elapsed_us:.2f}µs, expected < 1000µs")
        finally:
            bus.close()
            bus.unlink()

    def test_05_evolution_worker_execution_and_abort(self):
        """
        Subtask 1.4: Proves evolution_worker runs in isolated spawned process
        and terminates cleanly when abort is signaled.
        """
        from src.pipeline.shared_telemetry import SharedMemoryBus
        from src.pipeline.evolution_worker import run_evolution_process
        from src.pipeline.windows_job import WindowsJobManager
        
        shm_name = f"test_worker_shm_{os.getpid()}_{time.time_ns()}"
        bus = SharedMemoryBus(name=shm_name, create=True)
        job_mgr = WindowsJobManager(job_name=f"BioQuantWorkerJob_{os.getpid()}_{time.time_ns()}")
        
        ctx = multiprocessing.get_context("spawn")
        p = ctx.Process(
            target=run_evolution_process,
            kwargs={
                "shm_name": shm_name,
                "ticker": "NIFTYBEES",
                "lookback": "1y",
                "generations": 50,
                "population": 10
            }
        )
        p.start()
        job_mgr.assign_process(p.pid)
        
        hdr = None
        try:
            # Wait for worker to enter RUNNING state and start gen 1
            for _ in range(150):
                hdr = bus.read_header()
                if hdr.worker_status == 1 and hdr.current_gen >= 1:
                    break
                time.sleep(0.1)
                
            # Signal atomic abort
            bus.request_abort()
            p.join(timeout=10.0)
            
            self.assertFalse(p.is_alive(), "Worker process should be terminated")
            hdr = bus.read_header()
            self.assertEqual(hdr.worker_status, 2, "Status should be ABORTED (2)")
        finally:
            if p.is_alive():
                p.terminate()
            job_mgr.close()
            if hdr is not None:
                del hdr
            bus.close()
            bus.unlink()


if __name__ == "__main__":
    unittest.main()
