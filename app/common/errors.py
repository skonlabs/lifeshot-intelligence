"""Uniform error envelope + exception handlers.

Every error the client sees is ``{"error": {"code", "message", "request_id"}}``
with the matching HTTP status. Stack traces never leak. Codes:
  400 bad input · 401/403 auth · 404 url fetch failed · 413 too large ·
  422 validation / no face when enforced · 429 rate/quota/spend · 500 internal.
"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.common.logging import get_logger, get_request_id

log = get_logger("errors")


class APIError(Exception):
    """Raised anywhere in the app to produce a clean error envelope."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        headers: Optional[dict] = None,
    ):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.headers = headers or {}
        super().__init__(message)


# convenience constructors -------------------------------------------------
def bad_request(msg: str) -> APIError:
    return APIError(400, "bad_request", msg)


def unauthorized(msg: str = "Missing or invalid API key") -> APIError:
    return APIError(401, "unauthorized", msg, {"WWW-Authenticate": "ApiKey"})


def forbidden(msg: str) -> APIError:
    return APIError(403, "forbidden", msg)


def not_found(msg: str = "Resource could not be fetched") -> APIError:
    return APIError(404, "not_found", msg)


def too_large(msg: str) -> APIError:
    return APIError(413, "payload_too_large", msg)


def unprocessable(msg: str) -> APIError:
    return APIError(422, "unprocessable", msg)


def rate_limited(msg: str, retry_after: int = 60, extra_headers: Optional[dict] = None) -> APIError:
    headers = {"Retry-After": str(retry_after)}
    if extra_headers:
        headers.update(extra_headers)
    return APIError(429, "rate_limited", msg, headers)


def internal(msg: str = "Internal server error") -> APIError:
    return APIError(500, "internal_error", msg)


def _envelope(status: int, code: str, message: str) -> JSONResponse:
    request_id = get_request_id()
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "request_id": request_id}},
        headers={"X-Request-Id": request_id},
    )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(APIError)
    async def _api_error(_: Request, exc: APIError):
        resp = _envelope(exc.status_code, exc.code, exc.message)
        for k, v in exc.headers.items():
            resp.headers[k] = v
        return resp

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        # Summarize without echoing submitted values (could be PII).
        fields = ", ".join(".".join(str(p) for p in e["loc"][1:]) or "body" for e in exc.errors()[:5])
        return _envelope(422, "validation_error", f"Invalid request parameters: {fields}")

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException):
        code_map = {401: "unauthorized", 403: "forbidden", 404: "not_found", 413: "payload_too_large"}
        code = code_map.get(exc.status_code, "http_error")
        return _envelope(exc.status_code, code, str(exc.detail))

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        # Log full detail server-side (scrubbed), return generic message.
        log.exception("unhandled_exception", extra={"error_type": type(exc).__name__})
        return _envelope(500, "internal_error", "Internal server error")
