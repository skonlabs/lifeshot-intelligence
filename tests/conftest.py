"""Shared test fixtures. No network calls: OpenAI + DeepFace are mocked.

All PII in fixtures is SYNTHETIC. No real explicit imagery is used anywhere.
"""
from __future__ import annotations

import io
import os

# Configure the app via env BEFORE importing anything that reads settings.
os.environ.setdefault("API_KEYS", "test-key-123")
os.environ.setdefault("OPENAI_ENABLED", "true")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-real")
os.environ.setdefault("FACE_WARMUP", "false")  # don't load TF in tests
os.environ.setdefault("CORS_ORIGINS", "http://localhost:3000")
os.environ.setdefault("APP_ENV", "development")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402


API_KEY = "test-key-123"
AUTH = {"X-API-Key": API_KEY}


@pytest.fixture(autouse=True)
def _reset_caches():
    """Reset settings + auth + cost singletons between tests."""
    from app.common import auth, cost
    from app.common.cache import classification_cache, idempotency_cache, scene_cache
    from app.config import get_settings

    get_settings.cache_clear()
    auth.reset_registry_cache()
    cost.reset_for_tests()
    for c in (classification_cache, scene_cache, idempotency_cache):
        c.clear()
    yield
    get_settings.cache_clear()
    auth.reset_registry_cache()
    cost.reset_for_tests()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from app.main import create_app

    # Use context manager so lifespan (pool init) runs.
    with TestClient(create_app()) as c:
        yield c


def _png_bytes(width=64, height=48, color=(120, 130, 140)) -> bytes:
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes(width=64, height=48) -> bytes:
    img = Image.new("RGB", (width, height), (200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def png_image() -> bytes:
    return _png_bytes()


@pytest.fixture
def jpeg_image() -> bytes:
    return _jpeg_bytes()


@pytest.fixture
def bgr_array() -> "np.ndarray":
    return np.zeros((48, 64, 3), dtype=np.uint8)
