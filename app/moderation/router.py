"""Moderation HTTP route (no SDK imports; logic in service.py)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.common.auth import Principal, require_api_key
from app.common.cost import check_spend, rate_limit
from app.common.images import resolve_input
from app.common.input import as_float, read_input
from app.common.pool import run_heavy
from app.common.responses import (
    Timer,
    finalize,
    idempotent_lookup,
    idempotent_store,
)
from app.moderation import service
from app.moderation.schemas import ModerationResponse

router = APIRouter(tags=["moderation"])


@router.post("/scan", response_model=ModerationResponse)
async def scan(request: Request, principal: Principal = Depends(require_api_key)):
    rate_limit(principal)
    cached = idempotent_lookup(request, "moderation")
    if cached is not None:
        return cached

    timer = Timer()
    file_bytes, params = await read_input(request)
    loaded = await resolve_input(file_bytes, params.get("url"), params.get("base64"))
    threshold = None
    if params.get("threshold") is not None:
        threshold = as_float(params.get("threshold"), None)  # type: ignore[arg-type]

    check_spend(principal, estimated_usd=0.0)  # omni-moderation is free but keep the gate
    with timer.stage("moderation"):
        result = await run_heavy(service.scan, loaded.data, loaded.mime, threshold)

    payload = finalize(dict(result), timer)
    idempotent_store(request, "moderation", payload)
    return payload
