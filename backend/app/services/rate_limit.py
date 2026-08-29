from __future__ import annotations

import asyncio
import time
from collections import OrderedDict, deque
from dataclasses import dataclass


WINDOW_SECONDS = 60.0
MAX_TRACKED_CLIENTS = 2_048


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    allowed: bool
    retry_after_seconds: int


class SlidingWindowRateLimiter:
    """Per-client request cap over a rolling one-minute window.

    In-memory and per-process, which matches how Orion is deployed today (a
    single instance). If it ever runs multiple replicas this needs to move to
    shared storage, since each replica would otherwise allow the full quota.
    """

    def __init__(self, *, window_seconds: float = WINDOW_SECONDS) -> None:
        self._window = window_seconds
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def check(self, client_id: str, *, limit: int) -> RateLimitDecision:
        now = time.monotonic()
        cutoff = now - self._window
        async with self._lock:
            hits = self._hits.get(client_id)
            if hits is None:
                hits = deque()
                self._hits[client_id] = hits
            self._hits.move_to_end(client_id)

            while hits and hits[0] <= cutoff:
                hits.popleft()

            if len(hits) >= limit:
                retry_after = max(1, int(hits[0] + self._window - now) + 1)
                return RateLimitDecision(allowed=False, retry_after_seconds=retry_after)

            hits.append(now)

            # Bound memory: drop the least recently seen clients. Evicting a
            # client only resets its window, it can't grant extra allowance
            # beyond the limit for anyone currently being tracked.
            while len(self._hits) > MAX_TRACKED_CLIENTS:
                self._hits.popitem(last=False)

            return RateLimitDecision(allowed=True, retry_after_seconds=0)

    async def reset(self) -> None:
        async with self._lock:
            self._hits.clear()


chat_rate_limiter = SlidingWindowRateLimiter()
upload_rate_limiter = SlidingWindowRateLimiter()
