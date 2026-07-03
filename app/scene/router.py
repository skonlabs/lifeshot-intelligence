"""Scene HTTP route (no SDK imports; logic in service.py)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.common.auth import Principal, require_api_key
from app.common.cost import check_spend, rate_limit, record_spend
from app.common.errors import unprocessable
from app.common.images import resolve_input
from app.common.input import as_bool, read_input
from app.common.logging import hash_value
from app.common.pool import run_heavy
from app.common.responses import (
    Timer,
    finalize,
    idempotent_lookup,
    idempotent_store,
)
from app.scene import service
from app.scene.schemas import SceneResponse

router = APIRouter(tags=["scene"])


@router.post("/describe", response_model=SceneResponse)
async def describe(request: Request, principal: Principal = Depends(require_api_key)):
    rate_limit(principal)
    cached = idempotent_lookup(request, "scene")
    if cached is not None:
        return cached

    timer = Timer()
    file_bytes, params = await read_input(request)
    loaded = await resolve_input(file_bytes, params.get("url"), params.get("base64"))

    detail = (params.get("detail") or "short").lower()
    if detail not in ("short", "long"):
        raise unprocessable("detail must be 'short' or 'long'")
    include_embedding = as_bool(params.get("include_embedding"), False)
    known_facets = params.get("known_facets") if isinstance(params.get("known_facets"), dict) else None

    check_spend(principal, estimated_usd=0.01)
    with timer.stage("describe"):
        result = await run_heavy(
            service.describe,
            loaded.data,
            loaded.mime,
            detail=detail,
            include_embedding=include_embedding,
            known_facets=known_facets,
            content_hash=hash_value(loaded.data),
        )
    record_spend(principal, result.get("cost_usd", 0.0))

    body = {k: v for k, v in result.items() if k not in ("cost_usd", "cached")}
    body.setdefault("provider", "openai")
    payload = finalize(body, timer)
    idempotent_store(request, "scene", payload)
    return payload
