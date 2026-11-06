"""
src/pipeline/process_governor.py
Manages CPU Affinity and Win32 OS Priority Classes on Windows NT Kernel.
Guarantees that compute-heavy math processes never starve the UI or Emergency Panic Kill-Switch.
"""
import ctypes
from ctypes import wintypes
import os
from loguru import logger

kernel32 = ctypes.windll.kernel32

# Win32 Priority Constants
HIGH_PRIORITY_CLASS = 0x00000080
ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
NORMAL_PRIORITY_CLASS = 0x00000020
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
PROCESS_ALL_ACCESS = 0x1FFFFF

# Strict 64-bit AMD64 argtypes definitions
kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel32.OpenProcess.restype = wintypes.HANDLE

kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
kernel32.SetPriorityClass.restype = wintypes.BOOL

kernel32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
kernel32.SetProcessAffinityMask.restype = wintypes.BOOL

kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
kernel32.CloseHandle.restype = wintypes.BOOL


class ProcessGovernor:
    """Manages CPU Affinity and OS Priority Classes on Windows NT Kernel."""

    @staticmethod
    def configure_process(pid: int, priority_class: int, affinity_mask: int) -> bool:
        """Sets priority class and affinity mask for target PID."""
        hProcess = kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, pid)
        if not hProcess:
            logger.error(f"[ProcessGovernor] OpenProcess failed for PID {pid}: {kernel32.GetLastError()}")
            return False

        try:
            ok_prio = kernel32.SetPriorityClass(hProcess, priority_class)
            ok_aff = kernel32.SetProcessAffinityMask(hProcess, ctypes.c_size_t(affinity_mask))
            logger.info(
                f"[ProcessGovernor] PID {pid} configured: Priority=0x{priority_class:X} ({bool(ok_prio)}), "
                f"AffinityMask=0x{affinity_mask:X} ({bool(ok_aff)})"
            )
            return bool(ok_prio and ok_aff)
        finally:
            kernel32.CloseHandle(hProcess)

    @classmethod
    def configure_worker(cls, pid: int, num_cores: int = 4) -> bool:
        """Worker: BELOW_NORMAL priority, pinned to compute cores (Cores 2..N)."""
        mask = 0x0C if num_cores == 4 else (2**num_cores - 1) ^ 0x03
        return cls.configure_process(pid, BELOW_NORMAL_PRIORITY_CLASS, mask)

    @classmethod
    def configure_sentry(cls, pid: int) -> bool:
        """Risk Sentry: HIGH priority, pinned to Core 0 & Core 1."""
        return cls.configure_process(pid, HIGH_PRIORITY_CLASS, 0x03)

    @classmethod
    def configure_server(cls, pid: int) -> bool:
        """FastAPI Server: NORMAL priority, pinned to Core 0 & Core 1."""
        return cls.configure_process(pid, NORMAL_PRIORITY_CLASS, 0x03)
