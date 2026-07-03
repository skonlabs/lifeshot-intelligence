"""Unit tests for shared plumbing: images loader, metadata, pdf, ssrf, logging."""
from __future__ import annotations

import pytest

from app.common import images, logging as app_logging, ssrf
from app.common.errors import APIError


# ---- image loader / magic bytes ----
@pytest.mark.asyncio
async def test_resolve_input_requires_exactly_one_mode(png_image):
    with pytest.raises(APIError) as e:
        await images.resolve_input(None, None, None)
    assert e.value.status_code == 422
    with pytest.raises(APIError):
        await images.resolve_input(png_image, "http://x", None)


@pytest.mark.asyncio
async def test_resolve_input_sniffs_png(png_image):
    loaded = await images.resolve_input(png_image, None, None)
    assert loaded.kind == "png"
    assert loaded.mime == "image/png"


@pytest.mark.asyncio
async def test_resolve_input_rejects_unknown_magic():
    with pytest.raises(APIError) as e:
        await images.resolve_input(b"this is not an image", None, None)
    assert e.value.status_code == 422


def test_decode_to_bgr_shape(png_image):
    arr = images.decode_to_bgr(png_image)
    assert arr.shape[2] == 3
    assert arr.dtype.name == "uint8"


# ---- metadata ----
def test_metadata_dimensions_and_orientation(jpeg_image):
    from app.common.metadata import read_metadata

    meta = read_metadata(jpeg_image, "image/jpeg")
    assert meta["width"] == 64 and meta["height"] == 48
    assert meta["orientation"] == "landscape"
    assert meta["megapixels"] is not None
    # No EXIF in a freshly created image → nullable fields are null.
    assert meta["gps"] is None
    assert meta["exif_present"] in (False, True)


# ---- pdf ----
def test_pdf_rasterize_and_page_cap():
    import fitz

    from app.common import pdf

    doc = fitz.open()
    doc.new_page()
    doc.new_page()
    data = doc.tobytes()
    doc.close()
    assert pdf.is_pdf(data)
    pages = pdf.rasterize_pdf(data, dpi=72, max_pages=5)
    assert len(pages) == 2
    with pytest.raises(APIError):
        pdf.rasterize_pdf(data, dpi=72, max_pages=1)


# ---- ssrf ----
def test_ssrf_blocks_private_and_loopback():
    for bad in ("http://127.0.0.1/x", "http://169.254.169.254/latest", "http://10.0.0.5/", "http://[::1]/"):
        with pytest.raises(APIError):
            ssrf.validate_url(bad)


def test_ssrf_rejects_non_http_scheme():
    with pytest.raises(APIError):
        ssrf.validate_url("file:///etc/passwd")


# ---- logging scrubber ----
def test_logging_scrubs_pii():
    assert "[SSN]" in app_logging.scrub("value is 123-45-6789 ok")
    assert "[EMAIL]" in app_logging.scrub("mail john@example.com now")
    assert app_logging.hash_value("secret").startswith("sha256:")
