"""Bounded, per-worker in-memory caches (LRU).

Cache scope is explicit and bounded so memory can't grow without limit
(acceptance criteria #10). For multi-worker / multi-host deployments swap the
backing store for Redis behind this same interface (note in README).

Two users:
  * ``classification_cache`` / description caches keyed by image content hash.
  * ``idempotency_cache`` for paid endpoints: a repeated Idempotency-Key returns
    the stored response instead of re-calling OpenAI.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Optional

from app.config import get_settings


class LRUCache:
    def __init__(self, max_items: int):
        self._max = max(1, max_items)
        self._data: "OrderedDict[str, Any]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


_max = get_settings().cache_max_items
# Classification / scene-description results keyed by content hash (safe to cache).
classification_cache = LRUCache(_max)
scene_cache = LRUCache(_max)
# Idempotency: key -> stored response envelope. Do NOT store raw PII values here
# (responses may contain masked values only).
idempotency_cache = LRUCache(_max)
