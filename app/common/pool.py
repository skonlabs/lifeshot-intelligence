"""Off-event-loop execution pool + backpressure.

Heavy work (DeepFace inference, OpenAI vision over multi-page PDFs, OCR) must
never block the async event loop, and must be bounded so a burst can't OOM the
worker. We run such work in a small thread pool and gate admission with a
semaphore; when saturated we reject with 429 instead of queueing unboundedly.

TensorFlow is not reliably thread-safe, so DeepFace calls are additionally
serialized per model inside face/service.py (a separate lock there). This pool
just moves blocking work off the loop and caps concurrency.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Awaitable, Callable, Optional, TypeVar

from app.common.errors import APIError
from app.common.logging import get_logger
from app.config import get_settings

log = get_logger("pool")

T = TypeVar("T")

_executor: Optional[ThreadPoolExecutor] = None
_semaphore: Optional[asyncio.Semaphore] = None
_max_inflight: int = 0
_inflight: int = 0


def init_pool() -> None:
    global _executor, _semaphore, _max_inflight
    settings = get_settings()
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=max(1, settings.face_pool_workers),
            thread_name_prefix="infer",
        )
    if _semaphore is None:
        _max_inflight = max(1, settings.max_inflight_heavy)
        _semaphore = asyncio.Semaphore(_max_inflight)


def shutdown_pool() -> None:
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=True, cancel_futures=False)
        _executor = None


def saturation() -> float:
    if _max_inflight == 0:
        return 0.0
    return round(_inflight / _max_inflight, 3)


class _Saturated(APIError):
    def __init__(self):
        super().__init__(
            503,
            "overloaded",
            "Server is at capacity; retry shortly",
            {"Retry-After": "5"},
        )


async def run_heavy(fn: Callable[..., T], *args, **kwargs) -> T:
    """Run a blocking callable off the loop, bounded by the inflight semaphore.

    Non-blocking admission: if the pool is saturated we raise 503 immediately
    rather than growing an unbounded queue.
    """
    global _inflight
    if _executor is None or _semaphore is None:
        init_pool()
    assert _semaphore is not None and _executor is not None

    if _semaphore.locked() and _inflight >= _max_inflight:
        raise _Saturated()

    acquired = False
    try:
        # Bounded wait to admit; if we can't get a slot quickly, shed load.
        try:
            await asyncio.wait_for(_semaphore.acquire(), timeout=2.0)
            acquired = True
        except asyncio.TimeoutError:
            raise _Saturated()

        _inflight += 1
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_executor, lambda: fn(*args, **kwargs))
    finally:
        if acquired:
            _inflight -= 1
            _semaphore.release()


async def gather_bounded(coros: list[Awaitable[T]]) -> list[T]:
    """Await a list of coroutines together (they each self-limit via run_heavy)."""
    return await asyncio.gather(*coros)
