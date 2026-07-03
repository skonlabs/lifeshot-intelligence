"""PDF rasterization → page images, with a max-page guard.

Used by documents / pii / scene when the input is a PDF and we need pixels for
a vision model or OCR. PyMuPDF (fitz) renders without shelling out. The page cap
keeps every synchronous request bounded (true bulk belongs on the async path).
"""
from __future__ import annotations

from typing import List

from app.common.errors import too_large, unprocessable
from app.config import get_settings


def is_pdf(data: bytes) -> bool:
    return data[:5].startswith(b"%PDF")


def rasterize_pdf(data: bytes, dpi: int = 150, max_pages: int | None = None) -> List[bytes]:
    """Render each page to PNG bytes. Enforces the configured max-page cap."""
    import fitz  # PyMuPDF

    settings = get_settings()
    cap = max_pages if max_pages is not None else settings.max_pdf_pages

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception:
        raise unprocessable("Could not open PDF")

    try:
        if doc.page_count > cap:
            raise too_large(f"PDF has {doc.page_count} pages; max allowed is {cap}")
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pages: List[bytes] = []
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            pages.append(pix.tobytes("png"))
        return pages
    finally:
        doc.close()
