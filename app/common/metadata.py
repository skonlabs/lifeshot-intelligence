"""Local image metadata / EXIF reader — NO external calls.

Produces the ``image`` block returned by /face/detect and reused elsewhere:
dimensions, EXIF-corrected orientation, best-effort camera/EXIF, and GPS.

GPS is sensitive location PII: callers get it, but it is scrubbed from logs and
never persisted by default (see PII rules). Reverse-geocoding to a place name is
a separate, opt-in external call and is NOT done here.

HEIC support requires ``pillow-heif`` to be registered (done lazily on import).
"""
from __future__ import annotations

import io
from typing import Any, Dict, Optional

from PIL import ExifTags, Image

try:  # HEIC/HEIF (iPhone) — optional
    import pillow_heif  # type: ignore

    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover - optional dependency
    pass

# EXIF orientation values 5-8 imply a 90/270 rotation (swap w/h for "true" orientation).
_ROTATED_ORIENTATIONS = {5, 6, 7, 8}

_TAG_NAME = {v: k for k, v in ExifTags.TAGS.items()}
_GPS_TAG_NAME = {v: k for k, v in ExifTags.GPSTAGS.items()}


def _rational(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            return value[0] / value[1]
        except Exception:
            return None


def _dms_to_deg(dms, ref) -> Optional[float]:
    try:
        d = _rational(dms[0]) or 0.0
        m = _rational(dms[1]) or 0.0
        s = _rational(dms[2]) or 0.0
        deg = d + m / 60.0 + s / 3600.0
        if ref in ("S", "W"):
            deg = -deg
        return round(deg, 7)
    except Exception:
        return None


def _orientation_label(width: int, height: int) -> str:
    if width > height:
        return "landscape"
    if height > width:
        return "portrait"
    return "square"


def read_metadata(data: bytes, declared_mime: Optional[str] = None) -> Dict[str, Any]:
    """Return the metadata block for an image byte string. Never raises for a
    decodable image; unknown/absent fields are ``null``."""
    meta: Dict[str, Any] = {
        "format": None,
        "mime_type": declared_mime,
        "size_bytes": len(data),
        "width": None,
        "height": None,
        "orientation": None,
        "megapixels": None,
        "has_alpha": None,
        "dpi": None,
        "taken_at": None,
        "gps": None,
        "camera": None,
        "exif_present": False,
    }

    with Image.open(io.BytesIO(data)) as img:
        img_format = img.format
        width, height = img.size
        has_alpha = img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)
        dpi = img.info.get("dpi")

        exif = None
        try:
            exif = img.getexif()
        except Exception:
            exif = None

    # EXIF orientation may swap displayed w/h
    orient_val = None
    if exif:
        orient_val = exif.get(_TAG_NAME.get("Orientation"))
    true_w, true_h = width, height
    if orient_val in _ROTATED_ORIENTATIONS:
        true_w, true_h = height, width

    meta["format"] = img_format
    if not meta["mime_type"] and img_format:
        meta["mime_type"] = Image.MIME.get(img_format)
    meta["width"] = width
    meta["height"] = height
    meta["orientation"] = _orientation_label(true_w, true_h)
    meta["megapixels"] = round((width * height) / 1_000_000, 3)
    meta["has_alpha"] = bool(has_alpha)
    if dpi:
        try:
            meta["dpi"] = [round(float(dpi[0])), round(float(dpi[1]))]
        except Exception:
            meta["dpi"] = None

    if not exif:
        return meta

    meta["exif_present"] = True

    def tag(name: str):
        return exif.get(_TAG_NAME.get(name))

    meta["taken_at"] = tag("DateTimeOriginal") or tag("DateTime")

    camera = {
        "make": tag("Make"),
        "model": tag("Model"),
        "lens": tag("LensModel"),
        "iso": tag("ISOSpeedRatings"),
        "f_number": _rational(tag("FNumber")),
        "exposure_time": None,
        "focal_length": _rational(tag("FocalLength")),
    }
    et = tag("ExposureTime")
    if et is not None:
        val = _rational(et)
        camera["exposure_time"] = f"1/{round(1/val)}" if val and val < 1 else (str(val) if val else None)
    if any(v is not None for v in camera.values()):
        meta["camera"] = {k: (v.strip() if isinstance(v, str) else v) for k, v in camera.items()}

    # GPS
    try:
        gps_ifd = exif.get_ifd(_TAG_NAME.get("GPSInfo")) if hasattr(exif, "get_ifd") else None
    except Exception:
        gps_ifd = None
    if gps_ifd:
        def g(name: str):
            return gps_ifd.get(_GPS_TAG_NAME.get(name))

        lat = _dms_to_deg(g("GPSLatitude"), g("GPSLatitudeRef")) if g("GPSLatitude") else None
        lon = _dms_to_deg(g("GPSLongitude"), g("GPSLongitudeRef")) if g("GPSLongitude") else None
        alt = _rational(g("GPSAltitude"))
        if lat is not None or lon is not None:
            meta["gps"] = {"lat": lat, "lon": lon, "altitude": alt}

    return meta
