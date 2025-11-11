# app/ai/cost_monitor/rate_limiter.py
import time
from threading import Lock

class RateLimiter:
    """
    Simple in-memory rate limiter to restrict LLM API calls per window.
    Example: max 60 calls per minute.
    """

    def __init__(self, max_calls: int = 60, window_seconds: int = 60):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self.calls = []
        self._lock = Lock()

    def allow(self) -> bool:
        """
        Return True if a call is allowed, else False.
        """
        now = time.time()
        with self._lock:
            # Remove expired timestamps
            self.calls = [t for t in self.calls if now - t < self.window_seconds]

            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return True
            else:
                return False

    def wait_until_allowed(self):
        """
        Block until a new call can be made (safely throttles requests).
        """
        while not self.allow():
            time.sleep(0.5)
