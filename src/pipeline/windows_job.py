"""
src/pipeline/windows_job.py
Win32 Kernel Job Object Manager.
Enforces JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE (0x2000) so that all child worker
processes are forcibly and immediately terminated by the Windows NT kernel
whenever the parent process handle closes or crashes.
Eliminates orphan zombie processes pegging CPU cores.
"""
import ctypes
from ctypes import wintypes
import os
from loguru import logger

kernel32 = ctypes.windll.kernel32

# Win32 Constants
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
JobObjectExtendedLimitInformation = 9
PROCESS_ALL_ACCESS = 0x1FFFFF

# Strict 64-bit AMD64 argtypes definitions
kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
kernel32.CreateJobObjectW.restype = wintypes.HANDLE

kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
kernel32.SetInformationJobObject.restype = wintypes.BOOL

kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel32.OpenProcess.restype = wintypes.HANDLE

kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
kernel32.AssignProcessToJobObject.restype = wintypes.BOOL

kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
kernel32.CloseHandle.restype = wintypes.BOOL


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ('ReadOperationCount', ctypes.c_uint64),
        ('WriteOperationCount', ctypes.c_uint64),
        ('OtherOperationCount', ctypes.c_uint64),
        ('ReadTransferCount', ctypes.c_uint64),
        ('WriteTransferCount', ctypes.c_uint64),
        ('OtherTransferCount', ctypes.c_uint64),
    ]


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ('PerProcessUserTimeLimit', ctypes.c_int64),
        ('PerJobUserTimeLimit', ctypes.c_int64),
        ('LimitFlags', wintypes.DWORD),
        ('MinimumWorkingSetSize', ctypes.c_size_t),
        ('MaximumWorkingSetSize', ctypes.c_size_t),
        ('ActiveProcessLimit', wintypes.DWORD),
        ('Affinity', ctypes.c_size_t),
        ('PriorityClass', wintypes.DWORD),
        ('SchedulingClass', wintypes.DWORD),
    ]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ('BasicLimitInformation', JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ('IoCounters', IO_COUNTERS),
        ('ProcessMemoryLimit', ctypes.c_size_t),
        ('JobMemoryLimit', ctypes.c_size_t),
        ('PeakProcessMemoryLimit', ctypes.c_size_t),
        ('PeakJobMemoryLimit', ctypes.c_size_t),
    ]


class WindowsJobManager:
    """Guarantees zero-orphan process lifecycle using Windows Kernel Job Objects."""

    def __init__(self, job_name: str = "BioQuantEngineJob"):
        self.hJob = kernel32.CreateJobObjectW(None, job_name)
        if not self.hJob:
            raise OSError(f"Failed to create Windows Job Object: {kernel32.GetLastError()}")

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

        success = kernel32.SetInformationJobObject(
            self.hJob,
            JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info)
        )
        if not success:
            kernel32.CloseHandle(self.hJob)
            raise OSError(f"Failed to set Job Object Limits: {kernel32.GetLastError()}")

        logger.info(f"[WindowsJobManager] Initialized Job Object '{job_name}' with KILL_ON_JOB_CLOSE (0x2000).")

    def assign_process(self, pid: int) -> bool:
        """Assigns a child PID to this Job Object using a verified real process handle."""
        hProcess = kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, pid)
        if not hProcess:
            logger.error(f"[WindowsJobManager] OpenProcess failed for PID {pid}: {kernel32.GetLastError()}")
            return False

        try:
            res = kernel32.AssignProcessToJobObject(self.hJob, hProcess)
            if not res:
                logger.error(f"[WindowsJobManager] AssignProcessToJobObject failed for PID {pid}: {kernel32.GetLastError()}")
                return False
            logger.info(f"[WindowsJobManager] PID {pid} successfully bound to Job Object.")
            return True
        finally:
            kernel32.CloseHandle(hProcess)

    def close(self):
        if self.hJob:
            kernel32.CloseHandle(self.hJob)
            self.hJob = None
