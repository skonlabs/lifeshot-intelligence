"""Documents HTTP routes (no SDK imports; logic in service.py)."""
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
from app.documents import service
from app.documents.schemas import ClassifyResponse, ExtractResponse
from app.pii import service as pii_service

router = APIRouter(tags=["documents"])


@router.post("/classify", response_model=ClassifyResponse)
async def classify(request: Request, principal: Principal = Depends(require_api_key)):
    rate_limit(principal)
    cached = idempotent_lookup(request, "doc_classify")
    if cached is not None:
        return cached

    timer = Timer()
    file_bytes, params = await read_input(request)
    loaded = await resolve_input(file_bytes, params.get("url"), params.get("base64"))

    check_spend(principal, estimated_usd=0.01)
    with timer.stage("classify"):
        result = await run_heavy(service.classify, loaded.data, loaded.mime, hash_value(loaded.data))
    record_spend(principal, result.get("cost_usd", 0.0))

    payload = finalize(
        {
            "document_type": result["document_type"],
            "confidence": result["confidence"],
            "candidates": result["candidates"],
            "pages": result.get("pages"),
            "model": result["model"],
            "provider": "openai",
        },
        timer,
    )
    idempotent_store(request, "doc_classify", payload)
    return payload


@router.post("/extract", response_model=ExtractResponse)
async def extract(request: Request, principal: Principal = Depends(require_api_key)):
    rate_limit(principal)
    cached = idempotent_lookup(request, "doc_extract")
    if cached is not None:
        return cached

    timer = Timer()
    file_bytes, params = await read_input(request)
    loaded = await resolve_input(file_bytes, params.get("url"), params.get("base64"))

    do_classify = as_bool(params.get("classify"), True)
    fields = as_list(params.get("fields")) or None
    include_full_text = as_bool(params.get("include_full_text"), False)
    pii_scan = as_bool(params.get("pii_scan"), False)
    nsfw_scan = as_bool(params.get("nsfw_scan"), False)
    redact = as_bool(params.get("redact"), False)

    chash = hash_value(loaded.data)
    doc_type = None
    total_cost = 0.0

    check_spend(principal, estimated_usd=0.03)

    if do_classify:
        with timer.stage("classify"):
            cls = await run_heavy(service.classify, loaded.data, loaded.mime, chash)
        doc_type = cls["document_type"]
        total_cost += cls.get("cost_usd", 0.0)

    with timer.stage("extract"):
        ext = await run_heavy(
            service.extract,
            loaded.data,
            loaded.mime,
            fields=fields,
            include_full_text=include_full_text,
            content_hash=chash,
        )
    total_cost += ext.get("cost_usd", 0.0)

    payload_body = {
        "document_type": doc_type or ext.get("document_type"),
        "fields": ext["fields"],
        "full_text": ext.get("full_text"),
        "model": ext["model"],
        "provider": "openai",
    }

    # Optional PII / NSFW passes — shapes identical to the dedicated endpoints.
    if pii_scan:
        with timer.stage("pii_scan"):
            pii = await run_heavy(
                pii_service.scan,
                loaded.data,
                loaded.mime,
                redact=redact,
                include_full_text=False,
                nsfw_scan=False,
                content_hash=chash,
            )
        payload_body["pii_found"] = pii["pii_found"]
        payload_body["pii_entities"] = pii["entities"]
        if redact:
            payload_body["redacted_image"] = pii.get("redacted_image")
        total_cost += 0.02

    if nsfw_scan:
        from app.moderation import service as moderation_service

        with timer.stage("nsfw_scan"):
            mod = await run_heavy(moderation_service.scan, loaded.data, loaded.mime, None)
        payload_body["nsfw"] = mod["nsfw"]
        payload_body["nsfw_score"] = mod["nsfw_score"]

    record_spend(principal, total_cost)
    payload = finalize(payload_body, timer)
    idempotent_store(request, "doc_extract", payload)
    return payload
