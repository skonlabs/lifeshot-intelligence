"""PII tests: validators, masking, pii_found, redaction, no-PII-in-logs.

No network — OpenAI classification and DeepFace faces are mocked. Synthetic PII
only.
"""
from __future__ import annotations

import logging

from tests.conftest import AUTH


# ---- validators / masking (pure, no network) ----
def test_validate_ssn():
    from app.pii.service import validate_and_mask

    valid, masked = validate_and_mask("ssn", "123-45-6789")
    assert valid is True
    assert masked == "***-**-6789"
    assert "6789" in masked and "123" not in masked


def test_validate_credit_card_luhn():
    from app.pii.service import validate_and_mask

    valid, masked = validate_and_mask("credit_card_number", "4242 4242 4242 4242")
    assert valid is True
    assert masked.endswith("4242")
    bad, _ = validate_and_mask("credit_card_number", "4242 4242 4242 4241")
    assert bad is False


def test_validate_email_and_phone():
    from app.pii.service import validate_and_mask

    ve, me = validate_and_mask("email_address", "jane.doe@example.com")
    assert ve is True and me.startswith("j") and "@example.com" in me
    vp, mp = validate_and_mask("phone_number", "+1 (415) 555-0142")
    assert vp is True and mp.endswith("0142")


def test_validate_routing_number_aba():
    from app.pii.service import validate_and_mask

    valid, _ = validate_and_mask("routing_number", "021000021")  # valid ABA
    assert valid is True
    bad, _ = validate_and_mask("routing_number", "123456789")
    assert bad is False


# ---- endpoint: pii_found + masking + no raw values in response ----
def _mock_pii(monkeypatch, entities):
    """Mock the OpenAI text-PII classification to return given entities."""
    from app.common import openai_client
    from app.pii.schemas import PIIDetection, PIITextEntity

    class R:
        parsed = PIIDetection(
            entities=[PIITextEntity(type=t, value=v, confidence=c) for (t, v, c) in entities],
            full_text="",
        )
        model = "gpt-5.5"
        cost_usd = 0.0
        prompt_tokens = 10
        completion_tokens = 5

    monkeypatch.setattr(openai_client, "structured_vision", lambda **k: R())


def test_pii_scan_found_and_masked(client, monkeypatch, png_image):
    from app.face import service as face_service

    _mock_pii(monkeypatch, [("ssn", "123-45-6789", 0.98), ("email_address", "jane@example.com", 0.95)])
    monkeypatch.setattr(face_service, "detect", lambda *a, **k: [])  # no faces

    resp = client.post(
        "/v1/intelligence/pii/scan",
        headers=AUTH,
        data={"include_faces": "false"},
        files={"file": ("doc.png", png_image)},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["pii_found"] is True
    assert body["counts_by_type"]["ssn"] == 1
    # raw values never leave the service; only masked
    dumped = resp.text
    assert "123-45-6789" not in dumped
    assert "jane@example.com" not in dumped
    assert any(e["masked_value"] == "***-**-6789" for e in body["entities"])


def test_pii_scan_adds_faces_as_biometric(client, monkeypatch, png_image):
    from app.face import service as face_service

    _mock_pii(monkeypatch, [])
    monkeypatch.setattr(
        face_service,
        "detect",
        lambda *a, **k: [{"facial_area": {"x": 1, "y": 2, "w": 10, "h": 12}, "confidence": 0.9}],
    )
    resp = client.post("/v1/intelligence/pii/scan", headers=AUTH, files={"file": ("p.png", png_image)})
    assert resp.status_code == 200
    body = resp.json()
    face_entities = [e for e in body["entities"] if e["type"] == "face"]
    assert len(face_entities) == 1
    assert face_entities[0]["location"]["w"] == 10


def test_pii_scan_redact_returns_image(client, monkeypatch, png_image):
    from app.face import service as face_service
    from app.pii import service as pii_service

    _mock_pii(monkeypatch, [("ssn", "123-45-6789", 0.98)])
    monkeypatch.setattr(
        face_service,
        "detect",
        lambda *a, **k: [{"facial_area": {"x": 1, "y": 2, "w": 10, "h": 12}, "confidence": 0.9}],
    )
    # OCR locates the SSN token (VLM coords are never trusted for geometry).
    monkeypatch.setattr(
        pii_service, "_ocr_words", lambda bgr: [{"text": "123-45-6789", "x": 3, "y": 4, "w": 30, "h": 8}]
    )
    resp = client.post(
        "/v1/intelligence/pii/scan",
        headers=AUTH,
        data={"redact": "true"},
        files={"file": ("doc.png", png_image)},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["redacted_image"] is not None


def test_no_pii_in_logs(client, monkeypatch, png_image, caplog):
    from app.face import service as face_service

    _mock_pii(monkeypatch, [("ssn", "123-45-6789", 0.98)])
    monkeypatch.setattr(face_service, "detect", lambda *a, **k: [])
    with caplog.at_level(logging.DEBUG):
        client.post(
            "/v1/intelligence/pii/scan",
            headers=AUTH,
            data={"include_faces": "false"},
            files={"file": ("doc.png", png_image)},
        )
    for record in caplog.records:
        assert "123-45-6789" not in record.getMessage()
