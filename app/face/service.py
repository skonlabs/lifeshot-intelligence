"""Face service — the ONLY module that imports ``deepface``.

Responsibilities:
  * lazy import + one-time model load + warm-up (called from app startup)
  * serialize inference (TensorFlow is not reliably thread-safe → one lock)
  * detect / analyze / represent / verify(1:N) primitives
  * an embedding cache so verify embeds the reference once and reuses candidates

Public functions here are also imported by ``pii/service.py`` (the intentional
cross-feature composer) to add faces as biometric PII.

Design notes:
  * ``facial_area`` is returned in ORIGINAL-image pixel coordinates.
  * ``return_faces`` crops the RAW image (not DeepFace's aligned/normalized
    array). Crops are biometric PII — never persisted.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.common.cache import LRUCache
from app.common.errors import unprocessable
from app.common.images import crop_bgr, encode_bgr_to_base64
from app.common.logging import get_logger, hash_value
from app.config import get_settings

log = get_logger("face")

# TF is not thread-safe: serialize ALL deepface calls behind one lock. Combined
# with the bounded thread pool in common/pool.py this keeps inference safe.
_tf_lock = threading.Lock()

_warm = False
_embedding_cache = LRUCache(get_settings().cache_max_items)

_VALID_MODELS = {
    "Facenet512", "Facenet", "VGG-Face", "ArcFace", "Dlib", "SFace",
    "GhostFaceNet", "OpenFace", "DeepFace", "DeepID",
}
_VALID_DETECTORS = {
    "yunet", "opencv", "ssd", "dlib", "mtcnn", "retinaface",
    "mediapipe", "yolov8", "centerface", "skip",
}
_VALID_METRICS = {"cosine", "euclidean", "euclidean_l2"}


def validate_options(model: Optional[str], detector: Optional[str], metric: Optional[str]) -> None:
    if model is not None and model not in _VALID_MODELS:
        raise unprocessable(f"Invalid model_name '{model}'")
    if detector is not None and detector not in _VALID_DETECTORS:
        raise unprocessable(f"Invalid detector_backend '{detector}'")
    if metric is not None and metric not in _VALID_METRICS:
        raise unprocessable(f"Invalid distance_metric '{metric}'")


def is_warm() -> bool:
    return _warm


def warmup() -> None:
    """Load recognition + detection models once and prime them. Called at startup
    (blocking) so /ready flips only when models are usable."""
    global _warm
    settings = get_settings()
    from deepface import DeepFace  # lazy, heavy

    with _tf_lock:
        try:
            DeepFace.build_model(settings.face_model)
        except Exception as exc:  # pragma: no cover - depends on weights
            log.error("face_model_load_failed", extra={"error_type": type(exc).__name__})
            raise
        # prime the detector + representation path on a synthetic image
        dummy = np.zeros((160, 160, 3), dtype=np.uint8)
        try:
            DeepFace.represent(
                img_path=dummy,
                model_name=settings.face_model,
                detector_backend=settings.face_detector,
                enforce_detection=False,
                align=True,
            )
        except Exception:
            pass  # no face in a blank image is fine; models are now loaded
    _warm = True
    log.info("face_models_warm", extra={"model": settings.face_model, "detector": settings.face_detector})


# ---- low-level helpers (run inside the TF lock) --------------------------
def _facial_area_dict(fa: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "x": int(fa.get("x", 0)),
        "y": int(fa.get("y", 0)),
        "w": int(fa.get("w", 0)),
        "h": int(fa.get("h", 0)),
    }
    le, re = fa.get("left_eye"), fa.get("right_eye")
    out["left_eye"] = [int(le[0]), int(le[1])] if le else None
    out["right_eye"] = [int(re[0]), int(re[1])] if re else None
    return out


def _is_whole_image_noise(fa: Dict[str, Any], conf: float, shape: Tuple[int, int]) -> bool:
    """When enforce_detection=False and no face is found, DeepFace returns one
    'face' covering the whole image with confidence 0 — filter that out."""
    h, w = shape
    return conf == 0 and fa.get("w", 0) >= w * 0.98 and fa.get("h", 0) >= h * 0.98


# ---- primitives (each acquires the lock; call via pool.run_heavy) --------
def detect(
    bgr: np.ndarray,
    *,
    detector_backend: str,
    align: bool,
    enforce_detection: bool,
    return_faces: bool,
    padding: int = 0,
) -> List[Dict[str, Any]]:
    from deepface import DeepFace

    with _tf_lock:
        results = DeepFace.extract_faces(
            img_path=bgr,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
        )
    faces: List[Dict[str, Any]] = []
    for r in results:
        fa = r["facial_area"]
        conf = float(r.get("confidence", 0.0))
        if _is_whole_image_noise(fa, conf, bgr.shape[:2]):
            continue
        item: Dict[str, Any] = {"facial_area": _facial_area_dict(fa), "confidence": conf}
        if return_faces:
            fad = item["facial_area"]
            crop = crop_bgr(bgr, fad["x"], fad["y"], fad["w"], fad["h"], padding=padding)
            item["face"] = encode_bgr_to_base64(crop, ".png")
        faces.append(item)
    return faces


def analyze(
    bgr: np.ndarray,
    *,
    actions: List[str],
    detector_backend: str,
    align: bool,
    enforce_detection: bool,
) -> List[Dict[str, Any]]:
    from deepface import DeepFace

    with _tf_lock:
        results = DeepFace.analyze(
            img_path=bgr,
            actions=actions,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
        )
    if isinstance(results, dict):
        results = [results]
    out: List[Dict[str, Any]] = []
    for r in results:
        fa = r.get("region", {})
        conf = float(r.get("face_confidence", 0.0))
        if _is_whole_image_noise(fa, conf, bgr.shape[:2]):
            continue
        attributes = {
            "age": r.get("age"),
            "gender": r.get("gender"),
            "dominant_gender": r.get("dominant_gender"),
            "emotion": r.get("emotion"),
            "dominant_emotion": r.get("dominant_emotion"),
            "race": r.get("race"),
            "dominant_race": r.get("dominant_race"),
        }
        out.append(
            {
                "facial_area": _facial_area_dict(fa),
                "confidence": conf,
                "attributes": {k: v for k, v in attributes.items() if v is not None},
            }
        )
    return out


def represent(
    bgr: np.ndarray,
    *,
    model_name: str,
    detector_backend: str,
    align: bool,
    enforce_detection: bool,
) -> List[Dict[str, Any]]:
    from deepface import DeepFace

    with _tf_lock:
        results = DeepFace.represent(
            img_path=bgr,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
        )
    out: List[Dict[str, Any]] = []
    for r in results:
        fa = r.get("facial_area", {})
        conf = float(r.get("face_confidence", 0.0))
        if _is_whole_image_noise(fa, conf, bgr.shape[:2]):
            continue
        out.append(
            {
                "embedding": [float(x) for x in r["embedding"]],
                "facial_area": _facial_area_dict(fa),
                "face_confidence": conf,
            }
        )
    return out


def represent_cached(
    bgr: np.ndarray,
    *,
    model_name: str,
    detector_backend: str,
    align: bool,
    enforce_detection: bool,
    content_hash: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """represent() with an embedding cache keyed by content hash + options."""
    key = None
    if content_hash:
        key = f"{content_hash}:{model_name}:{detector_backend}:{align}"
        cached = _embedding_cache.get(key)
        if cached is not None:
            return cached
    result = represent(
        bgr,
        model_name=model_name,
        detector_backend=detector_backend,
        align=align,
        enforce_detection=enforce_detection,
    )
    if key is not None:
        _embedding_cache.set(key, result)
    return result


# ---- distance / threshold (mirror DeepFace's own math) -------------------
def _find_distance(a: List[float], b: List[float], metric: str) -> float:
    va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if metric == "cosine":
        denom = (np.linalg.norm(va) * np.linalg.norm(vb)) or 1e-12
        return float(1 - np.dot(va, vb) / denom)
    if metric == "euclidean":
        return float(np.linalg.norm(va - vb))
    if metric == "euclidean_l2":
        na = va / (np.linalg.norm(va) or 1e-12)
        nb = vb / (np.linalg.norm(vb) or 1e-12)
        return float(np.linalg.norm(na - nb))
    raise unprocessable(f"Invalid distance_metric '{metric}'")


def find_distance(a: List[float], b: List[float], metric: str) -> float:
    try:
        from deepface.modules import verification

        return float(verification.find_distance(np.asarray(a), np.asarray(b), metric))
    except Exception:
        return _find_distance(a, b, metric)


def find_threshold(model_name: str, metric: str) -> float:
    try:
        from deepface.modules import verification

        return float(verification.find_threshold(model_name, metric))
    except Exception:
        # DeepFace's documented defaults for Facenet512.
        defaults = {
            ("Facenet512", "cosine"): 0.30,
            ("Facenet512", "euclidean"): 23.56,
            ("Facenet512", "euclidean_l2"): 1.04,
        }
        return defaults.get((model_name, metric), 0.40)


def highest_confidence(faces: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not faces:
        return None
    return max(faces, key=lambda f: f.get("face_confidence", f.get("confidence", 0.0)))


def content_hash_of(data: bytes) -> str:
    return hash_value(data)
