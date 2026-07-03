"""Application entrypoint.

Creates the FastAPI app, mounts every feature router under
``/v1/intelligence``, runs model warm-up on startup, and drains cleanly on
SIGTERM. Health/ready/root are unversioned and open; everything else requires
auth.
"""
from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.common import openai_client
from app.common.errors import register_exception_handlers
from app.common.logging import configure_logging, get_logger, set_request_id
from app.common.pool import init_pool, saturation, shutdown_pool
from app.config import get_settings

# Feature routers (each imports only its own service — no SDKs at router level).
from app.documents.router import router as documents_router
from app.face.router import router as face_router
from app.moderation.router import router as moderation_router
from app.pii.router import router as pii_router
from app.scene.router import router as scene_router

settings = get_settings()
configure_logging(settings.log_level)
log = get_logger("main")

# Ensure DeepFace stores/reads weights from the configured location.
os.environ.setdefault("DEEPFACE_HOME", os.path.abspath(settings.deepface_home))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- startup ----
    init_pool()
    log.info("startup_begin", extra={"env": settings.app_env, "version": settings.app_version})

    if settings.face_warmup:
        try:
            from app.face import service as face_service

            face_service.warmup()
        except Exception as exc:  # keep booting; /ready will report not-warm
            log.error("face_warmup_failed", extra={"error_type": type(exc).__name__})

    if settings.moderation_provider == "local":
        try:
            from app.moderation import service as moderation_service

            moderation_service.warmup_local()
        except Exception as exc:
            log.error("local_nsfw_warmup_failed", extra={"error_type": type(exc).__name__})

    log.info("startup_complete")
    try:
        yield
    finally:
        # ---- graceful shutdown (SIGTERM): drain + close pools/clients ----
        log.info("shutdown_begin")
        shutdown_pool()
        openai_client.close_client()
        log.info("shutdown_complete")


def create_app() -> FastAPI:
    docs_url = "/docs" if (settings.enable_docs and not settings.is_production) else None
    app = FastAPI(
        title="LIFESHOT Intelligence API",
        version=settings.app_version,
        lifespan=lifespan,
        docs_url=docs_url,
        redoc_url=None,
        openapi_url=("/openapi.json" if docs_url else None),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins if settings.cors_origins else [],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "X-API-Key", "Content-Type", "Idempotency-Key"],
    )

    register_exception_handlers(app)

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        rid = request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex[:24]}"
        set_request_id(rid)
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-Id"] = rid
        return response

    # ---- feature routers under /v1/intelligence ----
    prefix = "/v1/intelligence"
    app.include_router(face_router, prefix=f"{prefix}/face")
    app.include_router(documents_router, prefix=f"{prefix}/documents")
    app.include_router(pii_router, prefix=f"{prefix}/pii")
    app.include_router(moderation_router, prefix=f"{prefix}/moderation")
    app.include_router(scene_router, prefix=f"{prefix}/scene")

    # ---- open health/ready/root ----
    @app.get("/health")
    async def health():
        return {"status": "ok", "version": settings.app_version}

    @app.get("/ready")
    async def ready():
        try:
            from app.face import service as face_service

            face_warm = face_service.is_warm()
        except Exception:
            face_warm = False
        from app.moderation import service as moderation_service

        is_ready = face_warm or not settings.face_warmup
        body = {
            "ready": bool(is_ready),
            "face_models_warm": bool(face_warm),
            "openai_configured": openai_client.is_configured(),
            "moderation_provider": moderation_service.provider_name(),
            "pool_saturation": saturation(),
        }
        return JSONResponse(status_code=200 if is_ready else 503, content=body)

    @app.get("/")
    async def root():
        return {
            "service": settings.app_name,
            "version": settings.app_version,
            "docs": docs_url or "disabled",
            "endpoints": [
                f"{prefix}/face/detect",
                f"{prefix}/face/analyze",
                f"{prefix}/face/verify",
                f"{prefix}/face/represent",
                f"{prefix}/documents/classify",
                f"{prefix}/documents/extract",
                f"{prefix}/pii/scan",
                f"{prefix}/moderation/scan",
                f"{prefix}/scene/describe",
            ],
        }

    return app


app = create_app()
