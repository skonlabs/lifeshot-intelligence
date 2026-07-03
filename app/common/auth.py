"""API-key auth dependency.

Every /v1/intelligence/* endpoint depends on ``require_api_key``. Keys are
compared in constant time. Production stores sha256 hashes (API_KEY_HASHES
"keyid:hexdigest,..."); development may pass plaintext keys (API_KEYS) which we
hash on load. Keys are never logged — only the resolved key *id*.
"""
from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Dict, Optional

from fastapi import Request

from app.common.errors import unauthorized
from app.config import Settings, get_settings


@dataclass(frozen=True)
class Principal:
    """The authenticated caller — a stable id used for quotas and logs."""

    key_id: str


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _build_registry(settings: Settings) -> Dict[str, str]:
    """Return {sha256hex: key_id}. Hashes take precedence over plaintext keys."""
    registry: Dict[str, str] = {}
    for entry in settings.api_key_hashes:
        if ":" in entry:
            key_id, digest = entry.split(":", 1)
        else:
            key_id, digest = entry[:8], entry
        registry[digest.lower().strip()] = key_id.strip()
    for i, raw in enumerate(settings.api_keys):
        registry.setdefault(_sha256(raw), f"key{i}")
    return registry


_registry_cache: Optional[Dict[str, str]] = None


def _registry(settings: Settings) -> Dict[str, str]:
    global _registry_cache
    if _registry_cache is None:
        _registry_cache = _build_registry(settings)
    return _registry_cache


def reset_registry_cache() -> None:  # for tests
    global _registry_cache
    _registry_cache = None


def _extract_key(request: Request) -> Optional[str]:
    # Accept "Authorization: Bearer <k>", "X-API-Key: <k>", or "?api_key="
    auth = request.headers.get("authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth[7:].strip()
    header_key = request.headers.get("x-api-key")
    if header_key:
        return header_key.strip()
    return None


async def require_api_key(request: Request) -> Principal:
    settings = get_settings()
    registry = _registry(settings)
    if not registry:
        # No keys configured at all — fail closed.
        raise unauthorized("Service has no API keys configured")

    presented = _extract_key(request)
    if not presented:
        raise unauthorized()

    digest = _sha256(presented)
    # Constant-time membership check across all known digests.
    matched_id: Optional[str] = None
    for known_digest, key_id in registry.items():
        if hmac.compare_digest(digest, known_digest):
            matched_id = key_id
    if matched_id is None:
        raise unauthorized()

    principal = Principal(key_id=matched_id)
    request.state.principal = principal
    return principal
