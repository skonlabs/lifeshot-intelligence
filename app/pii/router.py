"""PII scan HTTP route (no SDK imports; composition lives in service.py)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.common.auth import Principal, require_api_key
from app.common.cost import check_spend, rate_limit, record_spend
from app.common.images import resolve_input
from app.common.input import as_bool, as_list, read_input
from app.common.logging import hash_value
from app.common.pool import run_heavy
from app.common.responses import (
    Timer,
    finalize,
    idempotent_lookup,
    idempotent_store,
)
from app.pii import service
from app.pii.schemas import PIIScanResponse

router = APIRouter(tags=["pii"])


@router.post("/scan", response_model=PIIScanResponse)
async def scan(request: Request, principal: Principal = Depends(require_api_key)):
    rate_limit(principal)
    cached = idempotent_lookup(request, "pii")
    if cached is not None:
        return cached

    timer = Timer()
    file_bytes, params = await read_input(request)
    loaded = await resolve_input(file_bytes, params.get("url"), params.get("base64"))

    pii_types = as_list(params.get("pii_types")) or None
    include_faces = as_bool(params.get("include_faces"), True)
    redact = as_bool(params.get("redact"), False)
    include_full_text = as_bool(params.get("include_full_text"), False)
    nsfw_scan = as_bool(params.get("nsfw_scan"), False)

    check_spend(principal, estimated_usd=0.02)
    with timer.stage("pii_scan"):
        result = await run_heavy(
            service.scan,
            loaded.data,
            loaded.mime,
            pii_types=pii_types,
            include_faces=include_faces,
            redact=redact,
            include_full_text=include_full_text,
            nsfw_scan=nsfw_scan,
            content_hash=hash_value(loaded.data),
        )
    # rough spend record (2 OpenAI calls if nsfw+pii); refined in service via cost_usd if surfaced
    record_spend(principal, 0.02 + (0.0 if not nsfw_scan else 0.0))

    payload = finalize(dict(result), timer)
    idempotent_store(request, "pii", payload)
    return payload
