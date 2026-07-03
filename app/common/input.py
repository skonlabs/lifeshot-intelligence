"""Normalize the three image-input modes across all routers.

Every capability endpoint accepts EITHER ``multipart/form-data`` (a ``file``
upload plus form fields) OR ``application/json`` (``url``/``base64`` plus
params). This helper reads whichever was sent and returns a uniform
``(file_bytes, params)`` pair so routers don't repeat the plumbing.

Keeping this in one place also means the "exactly one input mode" rule and the
raw-bytes size handling live in ``images.resolve_input`` alone.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from fastapi import Request

from app.common.errors import bad_request


async def read_input(request: Request) -> Tuple[Optional[bytes], Dict[str, Any]]:
    ctype = (request.headers.get("content-type") or "").lower()
    file_bytes: Optional[bytes] = None
    params: Dict[str, Any] = {}

    if ctype.startswith("multipart/form-data"):
        form = await request.form()
        for key, value in form.multi_items():
            if hasattr(value, "read"):  # UploadFile
                if key in ("file", "img1"):
                    file_bytes = await value.read()
                else:
                    # extra files (e.g. img2[] candidates) collected as bytes list
                    params.setdefault(key, []).append(await value.read())
            else:
                if key in params and isinstance(params[key], list):
                    params[key].append(value)
                elif key in params:
                    params[key] = [params[key], value]
                else:
                    params[key] = value
    elif ctype.startswith("application/json"):
        try:
            params = await request.json()
        except Exception:
            raise bad_request("Invalid JSON body")
        if not isinstance(params, dict):
            raise bad_request("JSON body must be an object")
    else:
        # tolerate empty/other bodies (e.g. url-only via query string)
        params = dict(request.query_params)

    # merge query params (params in body win)
    for k, v in request.query_params.items():
        params.setdefault(k, v)

    return file_bytes, params


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [value]
