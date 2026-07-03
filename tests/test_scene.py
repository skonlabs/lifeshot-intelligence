"""Scene tests: describe → caption + facets with mocked OpenAI."""
from __future__ import annotations

from tests.conftest import AUTH


def _mock_scene(monkeypatch, is_new_event=False):
    from app.common import openai_client
    from app.scene.schemas import IsNewFacet, SceneFacets

    class R:
        parsed = SceneFacets(
            caption="A family gathered around a lit birthday cake as a child blows out candles.",
            event="birthday_party",
            setting="indoor_home",
            occasion="celebration",
            activities=["blowing out candles", "cheering"],
            objects=["birthday cake", "party hats"],
            people_count=5,
            mood="joyful",
            time_of_day="evening",
            tags=["birthday", "cake", "family", "celebration", "indoor"],
            is_new_facet=IsNewFacet(event=is_new_event, setting=False, occasion=False, mood=False),
        )
        model = "gpt-5.5-mini"
        cost_usd = 0.001

    monkeypatch.setattr(openai_client, "structured_vision", lambda **k: R())


def test_scene_describe(client, monkeypatch, png_image):
    _mock_scene(monkeypatch)
    resp = client.post("/v1/intelligence/scene/describe", headers=AUTH, files={"file": ("p.png", png_image)})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "birthday cake" in body["caption"]
    assert body["event"] == "birthday_party"
    assert "birthday" in body["tags"]
    assert body["is_new_facet"]["event"] is False
    assert body["embedding"] is None
    assert body["provider"] == "openai"


def test_scene_new_facet_flagged(client, monkeypatch, png_image):
    _mock_scene(monkeypatch, is_new_event=True)
    resp = client.post("/v1/intelligence/scene/describe", headers=AUTH, files={"file": ("p.png", png_image)})
    assert resp.status_code == 200
    assert resp.json()["is_new_facet"]["event"] is True


def test_scene_with_embedding(client, monkeypatch, png_image):
    _mock_scene(monkeypatch)
    from app.common import openai_client

    monkeypatch.setattr(openai_client, "embed_text", lambda text, model=None: ([0.1, 0.2, 0.3], 0.0))
    resp = client.post(
        "/v1/intelligence/scene/describe",
        headers=AUTH,
        data={"include_embedding": "true"},
        files={"file": ("p.png", png_image)},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["embedding"] == [0.1, 0.2, 0.3]


def test_scene_requires_auth(client, png_image):
    resp = client.post("/v1/intelligence/scene/describe", files={"file": ("p.png", png_image)})
    assert resp.status_code == 401
