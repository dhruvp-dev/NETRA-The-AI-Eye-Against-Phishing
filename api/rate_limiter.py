"""
NETRA - Sliding-window rate limiter
===================================
Per-key limiter used by the /predict and /admin endpoints.

Memory is bounded:
  * Keys are kept in least-recently-used order. The table never exceeds `max_keys`;
    inserting a new key over the cap evicts the least recently active key.
  * A periodic sweep drops keys whose whole window has expired.
"""

import time
from collections import OrderedDict, deque
from typing import Callable, Deque, Tuple


class SlidingWindowRateLimiter:
    def __init__(
        self,
        limit: int,
        window_seconds: float,
        max_keys: int = 100_000,
        sweep_interval: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.limit = limit
        self.window = window_seconds
        self.max_keys = max_keys
        self.sweep_interval = sweep_interval
        self._clock = clock
        self._hits: "OrderedDict[str, Deque[float]]" = OrderedDict()
        self._last_sweep = clock()

    def __len__(self) -> int:
        return len(self._hits)

    def _sweep(self, now: float) -> None:
        """Front of the OrderedDict is the least recently active key, so expired keys are contiguous there."""
        cutoff = now - self.window
        while self._hits:
            _, q = next(iter(self._hits.items()))
            if q and q[-1] > cutoff:
                break
            self._hits.popitem(last=False)
        self._last_sweep = now

    def check(self, key: str) -> Tuple[bool, int]:
        """Record a hit for `key`. Returns (allowed, retry_after_seconds)."""
        now = self._clock()
        if now - self._last_sweep >= self.sweep_interval:
            self._sweep(now)

        q = self._hits.get(key)
        if q is None:
            q = deque()
            self._hits[key] = q
            while len(self._hits) > self.max_keys:
                self._hits.popitem(last=False)
        else:
            self._hits.move_to_end(key)

        cutoff = now - self.window
        while q and q[0] <= cutoff:
            q.popleft()

        if len(q) >= self.limit:
            retry_after = int(self.window - (now - q[0])) + 1
            return False, max(retry_after, 1)

        q.append(now)
        return True, 0
