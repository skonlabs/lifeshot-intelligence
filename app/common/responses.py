"""Response envelope + timing + idempotency helpers shared by routers.

Every compute response carries ``request_id`` and ``timing_ms`` and echoes what
ran (model/detector/provider). Paid endpoints use ``idempotent`` to short-circuit
a repeated Idempotency-Key to the stored result.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional

from fastapi import Request

from app.common.cache import idempotency_cache
from app.common.logging import get_request_id, hash_value


class Timer:
    def __init__(self):
        self._marks: Dict[str, float] = {}
        self._start = time.perf_counter()
        self._last = self._start

    @contextmanager
    def stage(self, name: str):
        s = time.perf_counter()
        try:
            yield
        finally:
            self._marks[name] = round((time.perf_counter() - s) * 1000, 2)

    def mark(self, name: str) -> None:
        now = time.perf_counter()
        self._marks[name] = round((now - self._last) * 1000, 2)
        self._last = now

    def total_ms(self) -> float:
        return round((time.perf_counter() - self._start) * 1000, 2)

    def as_dict(self) -> Dict[str, Any]:
        d = dict(self._marks)
        d["total"] = self.total_ms()
        return d


def finalize(payload: Dict[str, Any], timer: Optional[Timer] = None) -> Dict[str, Any]:
    """Attach request_id + timing_ms to a success payload."""
    payload["request_id"] = get_request_id()
    if timer is not None:
        payload["timing_ms"] = timer.as_dict()
    return payload


def idempotency_key(request: Request) -> Optional[str]:
    return request.headers.get("idempotency-key")


def idempotent_lookup(request: Request, scope: str) -> Optional[Dict[str, Any]]:
    key = idempotency_key(request)
    if not key:
        return None
    # scope prevents cross-endpoint collisions on the same key
    return idempotency_cache.get(f"{scope}:{hash_value(key)}")


def idempotent_store(request: Request, scope: str, response: Dict[str, Any]) -> None:
    key = idempotency_key(request)
    if not key:
        return
    idempotency_cache.set(f"{scope}:{hash_value(key)}", response)
