"""Face HTTP routes (no SDK imports here — all DeepFace work is in service.py).

Mounted under /v1/intelligence/face by main.py.
"""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, Request

from app.common.auth import Principal, require_api_key
from app.common.cost import rate_limit
from app.common.errors import unprocessable
from app.common.images import decode_to_bgr, resolve_input
from app.common.input import as_bool, as_int, as_list, read_input
from app.common.metadata import read_metadata
from app.common.pool import run_heavy
from app.common.responses import Timer, finalize
from app.config import get_settings
from app.face import service
from app.face.schemas import (
    AnalyzeResponse,
    DetectResponse,
    RepresentResponse,
    VerifyResponse,
)

router = APIRouter(tags=["face"])

_ALL_ACTIONS = ["age", "gender", "emotion", "race"]


async def _load_single(request: Request):
    """Resolve one image from the request → (LoadedImage, bgr, metadata, params)."""
    file_bytes, params = await read_input(request)
    loaded = await resolve_input(file_bytes, params.get("url"), params.get("base64"))
    if loaded.kind == "pdf":
        raise unprocessable("Face endpoints require an image, not a PDF")
    bgr = decode_to_bgr(loaded.data)
    meta = read_metadata(loaded.data, loaded.mime)
    return loaded, bgr, meta, params


@router.post("/detect", response_model=DetectResponse)
async def detect(request: Request, principal: Principal = Depends(require_api_key)):
    rate_limit(principal)
    timer = Timer()
    settings = get_settings()
    loaded, bgr, meta, params = await _load_single(request)

    detector = params.get("detector_backend") or settings.face_detector
    service.validate_options(None, detector, None)
    align = as_bool(params.get("align"), True)
    enforce = as_bool(params.get("enforce_detection"), False)
    return_faces = as_bool(params.get("return_faces"), False)
    padding = as_int(params.get("padding"), 0)

    with timer.stage("inference"):
        faces = await run_heavy(
            service.detect,
            bgr,
            detector_backend=detector,
            align=align,
            enforce_detection=enforce,
            return_faces=return_faces,
            padding=padding,
        )

    if enforce and not faces:
        raise unprocessable("No face detected (enforce_detection=true)")

    return finalize(
        {"image": meta, "count": len(faces), "faces": faces, "detector": detector},
        timer,
    )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: Request, principal: Principal = Depends(require_api_key)):
    rate_limit(principal)
    timer = Timer()
    settings = get_settings()
    loaded, bgr, meta, params = await _load_single(request)

    detector = params.get("detector_backend") or settings.face_detector
    service.validate_options(None, detector, None)
    align = as_bool(params.get("align"), True)
    enforce = as_bool(params.get("enforce_detection"), False)
    actions = as_list(params.get("actions")) or list(_ALL_ACTIONS)
    invalid = [a for a in actions if a not in _ALL_ACTIONS]
    if invalid:
        raise unprocessable(f"Invalid actions: {invalid}")

    with timer.stage("inference"):
        faces = await run_heavy(
            service.analyze,
            bgr,
            actions=actions,
            detector_backend=detector,
            align=align,
            enforce_detection=enforce,
        )

    if enforce and not faces:
        raise unprocessable("No face detected (enforce_detection=true)")

    return finalize(
        {"image": meta, "count": len(faces), "faces": faces, "detector": detector, "actions": actions},
        timer,
    )


@router.post("/represent", response_model=RepresentResponse)
async def represent(request: Request, principal: Principal = Depends(require_api_key)):
    rate_limit(principal)
    timer = Timer()
    settings = get_settings()
    loaded, bgr, meta, params = await _load_single(request)

    model = params.get("model_name") or settings.face_model
    detector = params.get("detector_backend") or settings.face_detector
    service.validate_options(model, detector, None)
    align = as_bool(params.get("align"), True)
    enforce = as_bool(params.get("enforce_detection"), True)

    with timer.stage("inference"):
        embeddings = await run_heavy(
            service.represent,
            bgr,
            model_name=model,
            detector_backend=detector,
            align=align,
            enforce_detection=enforce,
        )

    if enforce and not embeddings:
        raise unprocessable("No face detected (enforce_detection=true)")

    dim = len(embeddings[0]["embedding"]) if embeddings else 0
    return finalize(
        {
            "count": len(embeddings),
            "dimension": dim,
            "model": model,
            "detector": detector,
            "embeddings": embeddings,
        },
        timer,
    )


@router.post("/verify", response_model=VerifyResponse)
async def verify(request: Request, principal: Principal = Depends(require_api_key)):
    """1:N verification. Embed the reference (img1) once; embed candidates
    (img2[]) in parallel; compare each vs the per-model threshold."""
    rate_limit(principal)
    timer = Timer()
    settings = get_settings()
    file_bytes, params = await read_input(request)

    model = params.get("model_name") or settings.face_model
    detector = params.get("detector_backend") or settings.face_detector
    metric = params.get("distance_metric") or settings.face_metric
    service.validate_options(model, detector, metric)
    align = as_bool(params.get("align"), True)
    enforce = as_bool(params.get("enforce_detection"), True)

    # --- reference (img1): file upload, or url/base64 in img1 field ---
    img1_ref = file_bytes or params.get("img1")
    if isinstance(img1_ref, list):
        img1_ref = img1_ref[0] if img1_ref else None
    ref_file = img1_ref if isinstance(img1_ref, (bytes, bytearray)) else None
    ref_url = img1_ref if isinstance(img1_ref, str) and img1_ref.startswith("http") else None
    ref_b64 = img1_ref if isinstance(img1_ref, str) and not img1_ref.startswith("http") else None
    ref_loaded = await resolve_input(ref_file, ref_url, ref_b64)
    ref_bgr = decode_to_bgr(ref_loaded.data)
    ref_hash = service.content_hash_of(ref_loaded.data)

    with timer.stage("embed_reference"):
        ref_embeddings = await run_heavy(
            service.represent_cached,
            ref_bgr,
            model_name=model,
            detector_backend=detector,
            align=align,
            enforce_detection=enforce,
            content_hash=ref_hash,
        )
    ref = service.highest_confidence(ref_embeddings)
    if ref is None:
        raise unprocessable("No face detected in reference image (img1)")

    # --- candidates (img2[]) ---
    candidates = params.get("img2")
    if candidates is None:
        raise unprocessable("img2 is required (array of candidate images)")
    if not isinstance(candidates, list):
        candidates = [candidates]
    if len(candidates) > settings.max_verify_candidates:
        raise unprocessable(
            f"Too many candidates: {len(candidates)} > max {settings.max_verify_candidates}"
        )

    threshold = service.find_threshold(model, metric)

    async def _embed_candidate(idx: int, cand) -> dict:
        try:
            c_file = cand if isinstance(cand, (bytes, bytearray)) else None
            c_url = cand if isinstance(cand, str) and cand.startswith("http") else None
            c_b64 = cand if isinstance(cand, str) and not (cand or "").startswith("http") else None
            loaded = await resolve_input(c_file, c_url, c_b64)
            bgr = decode_to_bgr(loaded.data)
            embs = await run_heavy(
                service.represent_cached,
                bgr,
                model_name=model,
                detector_backend=detector,
                align=align,
                enforce_detection=False,
                content_hash=service.content_hash_of(loaded.data),
            )
            best = service.highest_confidence(embs)
            if best is None:
                return {"index": idx, "error": "no_face_detected"}
            distance = service.find_distance(ref["embedding"], best["embedding"], metric)
            return {
                "index": idx,
                "verified": bool(distance <= threshold),
                "distance": round(distance, 6),
                "threshold": round(threshold, 6),
            }
        except Exception as exc:  # one bad candidate errors that item only
            return {"index": idx, "error": type(exc).__name__}

    import asyncio

    with timer.stage("embed_candidates"):
        results = await asyncio.gather(
            *[_embed_candidate(i, c) for i, c in enumerate(candidates)]
        )

    return finalize(
        {
            "model": model,
            "detector": detector,
            "distance_metric": metric,
            "reference": {"facial_area": ref["facial_area"], "confidence": ref["face_confidence"]},
            "results": results,
        },
        timer,
    )
