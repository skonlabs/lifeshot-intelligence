"""Moderation tests: threshold → nsfw boolean with a mocked provider.

No real explicit imagery. The provider (OpenAI omni-moderation) is mocked to
return category scores; we assert threshold behavior and sexual_minors surfacing.
"""
from __future__ import annotations

from tests.conftest import AUTH


def _mock_scores(monkeypatch, scores):
    from app.common import openai_client

    monkeypatch.setattr(openai_client, "moderate_image", lambda b, m="image/png": (scores, "omni-moderation-latest"))


def test_nsfw_true_above_threshold(client, monkeypatch, png_image):
    _mock_scores(monkeypatch, {"sexual": 0.97, "sexual/minors": 0.01})
    resp = client.post(
        "/v1/intelligence/moderation/scan",
        headers=AUTH,
        data={"threshold": "0.5"},
        files={"file": ("i.png", png_image)},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["nsfw"] is True
    assert body["nsfw_score"] == 0.97
    assert body["categories"]["sexual"] == 0.97
    assert body["provider"] == "openai-omni-moderation"
    assert body["sexual_minors_flagged"] is False


def test_nsfw_false_below_threshold(client, monkeypatch, png_image):
    _mock_scores(monkeypatch, {"sexual": 0.10, "sexual/minors": 0.0})
    resp = client.post(
        "/v1/intelligence/moderation/scan",
        headers=AUTH,
        data={"threshold": "0.5"},
        files={"file": ("i.png", png_image)},
    )
    assert resp.status_code == 200
    assert resp.json()["nsfw"] is False


def test_sexual_minors_surfaces(client, monkeypatch, png_image):
    _mock_scores(monkeypatch, {"sexual": 0.9, "sexual/minors": 0.8})
    resp = client.post(
        "/v1/intelligence/moderation/scan",
        headers=AUTH,
        data={"threshold": "0.5"},
        files={"file": ("i.png", png_image)},
    )
    assert resp.status_code == 200
    assert resp.json()["sexual_minors_flagged"] is True


def test_moderation_requires_auth(client, png_image):
    resp = client.post("/v1/intelligence/moderation/scan", files={"file": ("i.png", png_image)})
    assert resp.status_code == 401
