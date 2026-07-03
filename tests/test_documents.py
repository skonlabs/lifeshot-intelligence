"""Documents tests: classify + extract with mocked OpenAI Structured Outputs.

Also verifies extract runs PII/NSFW only when its flags are set, with shapes
matching the dedicated endpoints. Synthetic docs only.
"""
from __future__ import annotations

from tests.conftest import AUTH


def _mock_classify(monkeypatch):
    from app.common import openai_client
    from app.documents.schemas import DocCandidate, DocClassification

    class R:
        parsed = DocClassification(
            document_type="drivers_license",
            confidence=0.94,
            candidates=[DocCandidate(document_type="drivers_license", confidence=0.94), DocCandidate(document_type="id_card", confidence=0.05)],
        )
        model = "gpt-5.5-mini"
        cost_usd = 0.001

    monkeypatch.setattr(openai_client, "structured_vision", lambda **k: R())


def test_classify(client, monkeypatch, png_image):
    _mock_classify(monkeypatch)
    resp = client.post("/v1/intelligence/documents/classify", headers=AUTH, files={"file": ("d.png", png_image)})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["document_type"] == "drivers_license"
    assert body["confidence"] == 0.94
    assert body["candidates"][0]["document_type"] == "drivers_license"
    assert body["provider"] == "openai"


def test_extract_with_pii_and_nsfw_flags(client, monkeypatch, png_image):
    """extract merges pii/nsfw only when flags set; shapes match dedicated endpoints."""
    from app.common import openai_client
    from app.documents import service as doc_service
    from app.face import service as face_service
    from app.moderation import service as mod_service
    from app.pii import service as pii_service

    # extract() → fields
    monkeypatch.setattr(
        doc_service, "extract", lambda *a, **k: {"document_type": "invoice", "fields": {"total": "42.00"}, "full_text": None, "model": "gpt-5.5", "cost_usd": 0.002}
    )
    monkeypatch.setattr(doc_service, "classify", lambda *a, **k: {"document_type": "invoice", "confidence": 0.9, "candidates": [], "pages": 1, "model": "gpt-5.5-mini", "cost_usd": 0.001})
    # pii scan mock
    monkeypatch.setattr(
        pii_service,
        "scan",
        lambda *a, **k: {"pii_found": True, "counts_by_type": {"email_address": 1}, "entities": [{"type": "email_address", "masked_value": "j***@x.com", "confidence": 0.9, "valid": True}], "redacted_image": None, "full_text": None, "nsfw": None, "nsfw_score": None, "provider": "openai"},
    )
    monkeypatch.setattr(mod_service, "scan", lambda *a, **k: {"nsfw": False, "nsfw_score": 0.02, "categories": {}, "provider": "openai-omni-moderation", "threshold": 0.5, "sexual_minors_flagged": False})

    resp = client.post(
        "/v1/intelligence/documents/extract",
        headers=AUTH,
        data={"pii_scan": "true", "nsfw_scan": "true"},
        files={"file": ("d.png", png_image)},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["document_type"] == "invoice"
    assert body["fields"]["total"] == "42.00"
    # pii shape matches dedicated endpoint
    assert body["pii_found"] is True
    assert body["pii_entities"][0]["type"] == "email_address"
    # nsfw shape matches dedicated endpoint
    assert body["nsfw"] is False
    assert body["nsfw_score"] == 0.02


def test_extract_without_flags_has_no_pii(client, monkeypatch, png_image):
    from app.documents import service as doc_service

    monkeypatch.setattr(
        doc_service, "extract", lambda *a, **k: {"document_type": "receipt", "fields": {}, "full_text": None, "model": "gpt-5.5", "cost_usd": 0.0}
    )
    monkeypatch.setattr(doc_service, "classify", lambda *a, **k: {"document_type": "receipt", "confidence": 0.8, "candidates": [], "pages": 1, "model": "gpt-5.5-mini", "cost_usd": 0.0})
    resp = client.post("/v1/intelligence/documents/extract", headers=AUTH, files={"file": ("d.png", png_image)})
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("pii_found") is None
    assert body.get("nsfw") is None
