"""Token Bucket Rate Limiter enforcing broker request rate ceilings."""
import threading
import time
from typing import Optional


class TokenBucketRateLimiter:
    """Thread-safe token bucket rate limiter.
    
    Default parameters enforce Finvasia Shoonya's 20 requests/second API ceiling.
    """

    def __init__(self, rate: float = 20.0, capacity: float = 20.0) -> None:
        self.rate = float(rate)
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.last_update = time.perf_counter()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        """Replenishes tokens based on elapsed wall-clock time."""
        now = time.perf_counter()
        elapsed = now - self.last_update
        if elapsed > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """Non-blocking token consumption. Returns True if tokens were consumed."""
        with self._lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def acquire(self, tokens: float = 1.0) -> None:
        """Blocks until sufficient tokens are available, enforcing the rate ceiling."""
        while True:
            with self._lock:
                self._refill()
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return
                sleep_time = (tokens - self.tokens) / self.rate
            time.sleep(max(0.001, sleep_time))
