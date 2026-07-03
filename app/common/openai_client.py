"""The one configured OpenAI client — the ONLY module that imports ``openai``.

Centralizes timeouts, bounded retries with backoff, 429 handling, refusal
handling, and rough USD cost estimation so documents / pii-text / moderation /
scene all inherit identical resilience (acceptance: OpenAI resilience is
centralized here).

The client is created lazily and only when ``settings.openai_ready()`` — i.e.
the operator has set OPENAI_ENABLED=true (data-handling ack) AND provided a key.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel

from app.common.errors import APIError, internal, rate_limited, unprocessable
from app.common.logging import get_logger
from app.config import Settings, get_settings

log = get_logger("openai")

_client = None  # lazily created singleton


class OpenAIDisabled(APIError):
    def __init__(self):
        super().__init__(
            503,
            "openai_disabled",
            "OpenAI-backed features are disabled. Set OPENAI_ENABLED=true and provide OPENAI_API_KEY "
            "after acknowledging the data-handling posture (see README).",
        )


class OpenAIRefusal(APIError):
    def __init__(self, message: str = "The model refused to process this content"):
        super().__init__(422, "model_refusal", message)


# Rough per-1M-token USD prices for spend estimation only (update to match your
# contract; the spend guard is intentionally conservative).
_PRICES_PER_MTOK: Dict[str, Tuple[float, float]] = {
    # model_prefix: (input_usd_per_mtok, output_usd_per_mtok)
    "gpt-5.5": (5.0, 15.0),
    "gpt-5.5-mini": (0.6, 2.4),
    "gpt-5": (5.0, 15.0),
    "gpt-4o": (2.5, 10.0),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
    "omni-moderation": (0.0, 0.0),  # moderation endpoint is free
}


def _price_for(model: str) -> Tuple[float, float]:
    for prefix in sorted(_PRICES_PER_MTOK, key=len, reverse=True):
        if model.startswith(prefix):
            return _PRICES_PER_MTOK[prefix]
    return (5.0, 15.0)  # conservative default


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pin, pout = _price_for(model)
    return (prompt_tokens / 1_000_000) * pin + (completion_tokens / 1_000_000) * pout


@dataclass
class VisionResult:
    parsed: BaseModel
    model: str
    cost_usd: float
    prompt_tokens: int
    completion_tokens: int


def _get_client():
    global _client
    settings = get_settings()
    if not settings.openai_ready():
        raise OpenAIDisabled()
    if _client is None:
        from openai import OpenAI

        kwargs: Dict[str, Any] = {
            "api_key": settings.openai_api_key,
            "timeout": settings.openai_timeout_seconds,
            "max_retries": settings.openai_max_retries,
        }
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        _client = OpenAI(**kwargs)
    return _client


def is_configured() -> bool:
    return get_settings().openai_ready()


def close_client() -> None:
    """Close the shared client on shutdown."""
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
        _client = None


def _data_uri(image_bytes: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"


def _wrap_openai_error(exc: Exception) -> APIError:
    # Import here to avoid a hard dependency at module import time.
    try:
        from openai import APIStatusError, RateLimitError
    except Exception:  # pragma: no cover
        return internal("OpenAI call failed")

    if isinstance(exc, RateLimitError):
        return rate_limited("Upstream model provider is rate limiting; retry shortly", retry_after=30)
    if isinstance(exc, APIStatusError):
        if exc.status_code == 429:
            return rate_limited("Upstream model provider quota exceeded", retry_after=60)
        if 400 <= exc.status_code < 500:
            return unprocessable("The model provider rejected the request")
    return internal("OpenAI call failed")


def structured_vision(
    *,
    system_prompt: str,
    user_text: str,
    images: List[Tuple[bytes, str]],
    schema_model: Type[BaseModel],
    model: Optional[str] = None,
    max_output_tokens: int = 2048,
) -> VisionResult:
    """One vision → strict-JSON call using Structured Outputs.

    ``images`` is a list of (bytes, mime). Retries/timeouts come from the SDK
    client; refusals raise ``OpenAIRefusal`` (422).
    """
    settings = get_settings()
    client = _get_client()
    model = model or settings.openai_extract_model

    content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
    for img_bytes, mime in images:
        content.append(
            {"type": "image_url", "image_url": {"url": _data_uri(img_bytes, mime), "detail": "high"}}
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]

    try:
        completion = client.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=schema_model,
            max_tokens=max_output_tokens,
        )
    except APIError:
        raise
    except Exception as exc:  # network/status/etc → mapped envelope
        log.warning("openai_call_failed", extra={"error_type": type(exc).__name__, "model": model})
        raise _wrap_openai_error(exc)

    choice = completion.choices[0]
    msg = choice.message
    if getattr(msg, "refusal", None):
        raise OpenAIRefusal()
    parsed = msg.parsed
    if parsed is None:
        raise unprocessable("Model returned no structured output")

    usage = completion.usage
    pt = getattr(usage, "prompt_tokens", 0) or 0
    ct = getattr(usage, "completion_tokens", 0) or 0
    return VisionResult(
        parsed=parsed,
        model=model,
        cost_usd=estimate_cost_usd(model, pt, ct),
        prompt_tokens=pt,
        completion_tokens=ct,
    )


def moderate_image(image_bytes: bytes, mime: str = "image/png") -> Tuple[Dict[str, float], str]:
    """Call the omni-moderation endpoint on an image. Returns (category_scores, model)."""
    settings = get_settings()
    client = _get_client()
    model = settings.moderation_openai_model
    try:
        resp = client.moderations.create(
            model=model,
            input=[{"type": "image_url", "image_url": {"url": _data_uri(image_bytes, mime)}}],
        )
    except APIError:
        raise
    except Exception as exc:
        log.warning("openai_moderation_failed", extra={"error_type": type(exc).__name__})
        raise _wrap_openai_error(exc)

    result = resp.results[0]
    scores = result.category_scores
    # Normalize to a plain dict of floats.
    if hasattr(scores, "model_dump"):
        scores = scores.model_dump()
    elif not isinstance(scores, dict):
        scores = dict(scores)
    return {k: float(v) for k, v in scores.items() if v is not None}, model


def embed_text(text: str, model: Optional[str] = None) -> Tuple[List[float], float]:
    """Return (embedding, cost_usd)."""
    settings = get_settings()
    client = _get_client()
    model = model or settings.openai_embed_model
    try:
        resp = client.embeddings.create(model=model, input=text)
    except APIError:
        raise
    except Exception as exc:
        raise _wrap_openai_error(exc)
    vec = resp.data[0].embedding
    pt = getattr(resp.usage, "prompt_tokens", 0) or 0
    return list(vec), estimate_cost_usd(model, pt, 0)
