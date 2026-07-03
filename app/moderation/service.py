"""NSFW moderation.

Default provider = OpenAI ``omni-moderation`` (image-capable, free, no extra
dependency). Alternative = a local ONNX classifier (the ONLY place a local NSFW
model would be imported). Output is normalized to a ``sexual`` score compared
against a configurable threshold.

CSAM is explicitly OUT OF SCOPE for this classifier — it flags ADULT sexual
content only. A ``sexual_minors`` signal is surfaced separately and must route
to the dedicated legal/operational workflow (NCMEC / hash-matching), never
treated as a normal result. See README compliance section.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from app.common import openai_client
from app.common.errors import APIError, internal
from app.common.logging import get_logger
from app.common.pdf import is_pdf, rasterize_pdf
from app.config import get_settings

log = get_logger("moderation")

# local model singleton (only initialized when provider=local)
_local_session = None


def provider_name() -> str:
    settings = get_settings()
    if settings.moderation_provider == "local":
        return "local"
    return "openai-omni-moderation"


def is_ready() -> bool:
    settings = get_settings()
    if settings.moderation_provider == "local":
        return bool(settings.nsfw_model)
    return openai_client.is_configured()


def warmup_local() -> None:
    """Load the local ONNX model at startup if configured."""
    global _local_session
    settings = get_settings()
    if settings.moderation_provider != "local" or not settings.nsfw_model:
        return
    import onnxruntime as ort  # imported only here, only if local

    _local_session = ort.InferenceSession(settings.nsfw_model, providers=["CPUExecutionProvider"])
    log.info("local_nsfw_model_loaded")


def _normalize_openai_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """Map omni-moderation categories to our normalized category set."""
    sexual = float(scores.get("sexual", 0.0))
    minors = float(scores.get("sexual/minors", scores.get("sexual_minors", 0.0)))
    # "suggestive" isn't a native omni category; approximate from sexual context.
    return {
        "sexual": sexual,
        "suggestive": round(min(1.0, sexual * 0.6), 4) if sexual < 0.5 else round(sexual, 4),
        "sexual_minors": minors,
    }


def _score_local(image_bytes: bytes) -> Dict[str, float]:
    """Run the local ONNX NSFW classifier → {sexual, suggestive, sexual_minors}."""
    global _local_session
    if _local_session is None:
        warmup_local()
    if _local_session is None:
        raise internal("Local NSFW model is not configured")
    import io

    import numpy as np
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    arr = (np.asarray(img, dtype="float32") / 255.0)[None, ...]
    inp = _local_session.get_inputs()[0].name
    out = _local_session.run(None, {inp: arr})[0][0]
    # Convention: model outputs [safe, nsfw] probabilities.
    nsfw = float(out[-1]) if len(out) > 1 else float(out[0])
    return {"sexual": nsfw, "suggestive": round(nsfw * 0.6, 4), "sexual_minors": 0.0}


def _pages(data: bytes, mime: str) -> List[Tuple[bytes, str]]:
    if is_pdf(data):
        return [(p, "image/png") for p in rasterize_pdf(data)]
    return [(data, mime)]


def scan(data: bytes, mime: str, threshold: Optional[float] = None) -> Dict:
    settings = get_settings()
    thr = settings.nsfw_threshold if threshold is None else threshold

    best_scores: Dict[str, float] = {"sexual": 0.0, "suggestive": 0.0, "sexual_minors": 0.0}
    for img_bytes, page_mime in _pages(data, mime):
        if settings.moderation_provider == "local":
            scores = _score_local(img_bytes)
        else:
            raw, _model = openai_client.moderate_image(img_bytes, page_mime)
            scores = _normalize_openai_scores(raw)
        # take the worst page
        if scores.get("sexual", 0.0) >= best_scores["sexual"]:
            best_scores = scores

    sexual = float(best_scores.get("sexual", 0.0))
    minors = float(best_scores.get("sexual_minors", 0.0))
    nsfw = sexual >= thr

    if minors >= thr:
        # Do NOT persist/forward casually; surface a strong ops signal.
        log.error("sexual_minors_flagged_route_to_legal_workflow")

    return {
        "nsfw": nsfw,
        "nsfw_score": round(sexual, 4),
        "categories": {k: round(float(v), 4) for k, v in best_scores.items()},
        "provider": provider_name(),
        "threshold": thr,
        "sexual_minors_flagged": bool(minors >= thr),
    }
