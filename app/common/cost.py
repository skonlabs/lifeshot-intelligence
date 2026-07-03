"""Cost controls: per-key rate limits + global/per-key spend guard.

OpenAI is real money, so paid endpoints must fail closed on breach
(acceptance criterion #11). This is an in-process implementation (per worker);
for a fleet, back it with Redis so limits are shared.

  * ``rate_limit(principal)`` — token-bucket-ish fixed-window request cap.
  * ``check_spend()`` / ``record_spend()`` — daily global + per-key USD caps;
    ``check_spend`` is called BEFORE a paid call and raises 429 when over budget.

Spend is tracked in USD estimated from token usage returned by OpenAI.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Dict, Tuple

from app.common.auth import Principal
from app.common.errors import rate_limited
from app.common.logging import get_logger
from app.config import get_settings

log = get_logger("cost")

_lock = threading.Lock()

# fixed-window request counters: key_id -> (window_start_epoch, count)
_req_windows: Dict[str, Tuple[int, int]] = {}

# spend accounting: reset when the UTC day changes
_spend_day: int = -1
_global_spend_usd: float = 0.0
_per_key_spend_usd: Dict[str, float] = defaultdict(float)


def _utc_day() -> int:
    return int(time.time() // 86400)


def _roll_day_locked() -> None:
    global _spend_day, _global_spend_usd, _per_key_spend_usd
    today = _utc_day()
    if today != _spend_day:
        _spend_day = today
        _global_spend_usd = 0.0
        _per_key_spend_usd = defaultdict(float)


def rate_limit(principal: Principal) -> None:
    """Fixed 60s window request cap per key. Raises 429 on breach."""
    settings = get_settings()
    limit = settings.per_key_rate_per_min
    now = int(time.time())
    window = now - (now % 60)
    with _lock:
        start, count = _req_windows.get(principal.key_id, (window, 0))
        if start != window:
            start, count = window, 0
        count += 1
        _req_windows[principal.key_id] = (start, count)
        remaining = max(0, limit - count)
    if count > limit:
        retry_after = 60 - (now % 60)
        raise rate_limited(
            "Per-key request rate limit exceeded",
            retry_after=retry_after,
            extra_headers={
                "X-RateLimit-Limit": str(limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(window + 60),
            },
        )


def check_spend(principal: Principal, estimated_usd: float = 0.0) -> None:
    """Fail closed if we're already over (or would exceed) a spend cap."""
    settings = get_settings()
    with _lock:
        _roll_day_locked()
        g = _global_spend_usd
        k = _per_key_spend_usd[principal.key_id]
    if g + estimated_usd > settings.global_spend_cap_usd:
        log.warning("global_spend_cap_hit", extra={"global_usd": round(g, 4)})
        raise rate_limited(
            "Daily global spend cap reached; try again tomorrow",
            retry_after=3600,
        )
    if k + estimated_usd > settings.per_key_daily_spend_usd:
        log.warning("per_key_spend_cap_hit", extra={"key_id": principal.key_id, "key_usd": round(k, 4)})
        raise rate_limited(
            "Daily spend cap for this API key reached",
            retry_after=3600,
        )


def record_spend(principal: Principal, usd: float) -> None:
    global _global_spend_usd
    if usd <= 0:
        return
    with _lock:
        _roll_day_locked()
        _global_spend_usd += usd
        _per_key_spend_usd[principal.key_id] += usd
    # alert before the cap (best-effort log signal for ops alerting)
    settings = get_settings()
    if _global_spend_usd > 0.8 * settings.global_spend_cap_usd:
        log.warning("spend_alert_80pct", extra={"global_usd": round(_global_spend_usd, 4)})


def snapshot() -> Dict[str, float]:
    with _lock:
        _roll_day_locked()
        return {
            "global_spend_usd": round(_global_spend_usd, 4),
            "keys_tracked": len(_per_key_spend_usd),
        }


def reset_for_tests() -> None:
    global _global_spend_usd, _per_key_spend_usd, _req_windows, _spend_day
    with _lock:
        _global_spend_usd = 0.0
        _per_key_spend_usd = defaultdict(float)
        _req_windows = {}
        _spend_day = -1
