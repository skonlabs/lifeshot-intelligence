"""Image input handling: load / decode / guard / redact.

Single entry point for turning one of {multipart file, url, base64} into raw
bytes and then a decoded array. Enforces the per-request hard caps (file size,
megapixels) and verifies file type by MAGIC BYTES, not the declared
content-type (security hardening). Also provides pixel redaction (blur boxes)
used by the PII redaction path.

cv2 is used here for decode/blur; the array we hand DeepFace is BGR uint8, which
is what its detectors expect.
"""
from __future__ import annotations

import base64 as b64
import io
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from app.common.errors import APIError, bad_request, not_found, too_large, unprocessable
from app.common.ssrf import validate_url
from app.config import get_settings

# Magic-byte signatures for allowed types.
_MAGIC = {
    b"\xff\xd8\xff": ("jpeg", "image/jpeg"),
    b"\x89PNG\r\n\x1a\n": ("png", "image/png"),
    b"GIF87a": ("gif", "image/gif"),
    b"GIF89a": ("gif", "image/gif"),
    b"BM": ("bmp", "image/bmp"),
    b"%PDF": ("pdf", "application/pdf"),
}


@dataclass
class LoadedImage:
    data: bytes           # raw file bytes (also used by metadata reader)
    kind: str             # detected type: jpeg/png/webp/heic/pdf/...
    mime: str


def _sniff(data: bytes) -> Tuple[str, str]:
    for sig, (kind, mime) in _MAGIC.items():
        if data.startswith(sig):
            return kind, mime
    # WEBP: "RIFF....WEBP"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp", "image/webp"
    # HEIC/HEIF: ftyp box with heic/heix/mif1 brands
    if data[4:8] == b"ftyp" and data[8:12] in (b"heic", b"heix", b"hevc", b"mif1", b"msf1"):
        return "heic", "image/heic"
    # TIFF
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff", "image/tiff"
    raise unprocessable("Unsupported or unrecognized file type (magic-byte check failed)")


def _enforce_size(data: bytes) -> None:
    settings = get_settings()
    if len(data) > settings.max_file_bytes:
        raise too_large(f"File exceeds max size of {settings.max_file_mb} MB")


async def fetch_url(url: str) -> bytes:
    """SSRF-guarded, bounded remote fetch."""
    import httpx

    validate_url(url)
    settings = get_settings()
    limit = settings.max_file_bytes
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=False) as client:
            async with client.stream("GET", url) as resp:
                if resp.status_code >= 400:
                    raise not_found(f"URL fetch failed with status {resp.status_code}")
                chunks = []
                total = 0
                async for chunk in resp.aiter_bytes():
                    total += len(chunk)
                    if total > limit:
                        raise too_large(f"Remote file exceeds max size of {settings.max_file_mb} MB")
                    chunks.append(chunk)
                return b"".join(chunks)
    except APIError:
        raise
    except httpx.HTTPError:
        raise not_found("Could not fetch the provided URL")


async def resolve_input(
    file_bytes: Optional[bytes],
    url: Optional[str],
    base64_str: Optional[str],
) -> LoadedImage:
    """Exactly one of the three input modes must be provided."""
    provided = [x for x in (file_bytes, url, base64_str) if x]
    if len(provided) != 1:
        raise unprocessable("Provide exactly one of: file upload, url, or base64")

    if file_bytes:
        data = file_bytes
    elif url:
        data = await fetch_url(url)
    else:
        raw = base64_str.split(",", 1)[-1] if "," in base64_str else base64_str
        try:
            data = b64.b64decode(raw, validate=True)
        except Exception:
            raise bad_request("Invalid base64 input")

    _enforce_size(data)
    kind, mime = _sniff(data)
    return LoadedImage(data=data, kind=kind, mime=mime)


def decode_to_bgr(data: bytes) -> np.ndarray:
    """Decode image bytes to a BGR uint8 numpy array, enforcing the megapixel
    decompression-bomb guard. Uses PIL (broad format support incl. HEIC) then
    converts to the BGR layout DeepFace/cv2 expect."""
    settings = get_settings()
    try:
        with Image.open(io.BytesIO(data)) as img:
            mp = (img.width * img.height) / 1_000_000
            if mp > settings.max_megapixels:
                raise too_large(
                    f"Image {round(mp,1)}MP exceeds max of {settings.max_megapixels}MP (decompression-bomb guard)"
                )
            rgb = img.convert("RGB")
            arr = np.asarray(rgb)
    except too_large:
        raise
    except Exception:
        raise unprocessable("Could not decode image")
    # RGB -> BGR
    return arr[:, :, ::-1].copy()


def encode_bgr_to_base64(bgr: np.ndarray, fmt: str = ".png") -> str:
    """Encode a BGR array back to base64 (for redacted images / crops)."""
    import cv2

    ok, buf = cv2.imencode(fmt, bgr)
    if not ok:
        raise unprocessable("Failed to encode image")
    return b64.b64encode(buf.tobytes()).decode("ascii")


def crop_bgr(bgr: np.ndarray, x: int, y: int, w: int, h: int, padding: int = 0) -> np.ndarray:
    H, W = bgr.shape[:2]
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(W, x + w + padding)
    y1 = min(H, y + h + padding)
    return bgr[y0:y1, x0:x1].copy()


def redact_boxes(bgr: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
    """Return a copy with each (x,y,w,h) box heavily blurred (pixel redaction).

    Blur (not black boxes) keeps the image human-readable while removing the PII
    signal; kernel scales with box size so small text is fully destroyed.
    """
    import cv2

    out = bgr.copy()
    H, W = out.shape[:2]
    for (x, y, w, h) in boxes:
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(W, int(x + w)), min(H, int(y + h))
        if x1 <= x0 or y1 <= y0:
            continue
        region = out[y0:y1, x0:x1]
        k = max(11, (min(region.shape[0], region.shape[1]) // 2) | 1)
        blurred = cv2.GaussianBlur(region, (k, k), 0)
        # pixelate on top for irreversibility
        small = cv2.resize(blurred, (max(1, (x1 - x0) // 12), max(1, (y1 - y0) // 12)), interpolation=cv2.INTER_LINEAR)
        out[y0:y1, x0:x1] = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    return out
