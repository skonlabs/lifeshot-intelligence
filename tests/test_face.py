"""Face endpoint tests. DeepFace is mocked at the service boundary (no TF load).

Covers: detect + 1:N verify happy paths, no-face-candidate item error, and
invalid-option 422.
"""
from __future__ import annotations

from tests.conftest import AUTH


def _fake_face(x=5, y=6, w=20, h=25, conf=0.99):
    return {
        "facial_area": {"x": x, "y": y, "w": w, "h": h, "left_eye": [10, 12], "right_eye": [18, 12]},
        "confidence": conf,
    }


def test_detect_happy_path(client, monkeypatch, png_image):
    from app.face import service

    monkeypatch.setattr(service, "detect", lambda *a, **k: [_fake_face()])
    resp = client.post("/v1/intelligence/face/detect", headers=AUTH, files={"file": ("i.png", png_image)})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] == 1
    assert body["faces"][0]["facial_area"]["w"] == 20
    assert body["image"]["width"] == 64  # metadata block present, original-pixel coords
    assert body["detector"] == "yunet"
    assert body["request_id"]
    assert resp.headers["X-Request-Id"]


def test_detect_requires_auth(client, png_image):
    resp = client.post("/v1/intelligence/face/detect", files={"file": ("i.png", png_image)})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


def test_detect_invalid_detector_422(client, png_image):
    resp = client.post(
        "/v1/intelligence/face/detect",
        headers=AUTH,
        data={"detector_backend": "not_a_detector"},
        files={"file": ("i.png", png_image)},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "unprocessable"


def test_detect_enforce_no_face_422(client, monkeypatch, png_image):
    from app.face import service

    monkeypatch.setattr(service, "detect", lambda *a, **k: [])
    resp = client.post(
        "/v1/intelligence/face/detect",
        headers=AUTH,
        data={"enforce_detection": "true"},
        files={"file": ("i.png", png_image)},
    )
    assert resp.status_code == 422


def test_verify_1n_with_candidate_error(client, monkeypatch, png_image):
    """Reference embeds once; one candidate has a face, one has none (item error).

    Candidates are distinguished by image size (not call order) so the result is
    deterministic regardless of concurrent embedding.
    """
    import io

    from PIL import Image

    from app.face import service

    emb = {"embedding": [1.0, 0.0, 0.0], "facial_area": _fake_face()["facial_area"], "face_confidence": 0.98}

    def fake_represent_cached(bgr, **k):
        # 64x48 images (reference + candidate 0) carry a face; the 32x32 one does not.
        if bgr.shape[0] == 48:
            return [dict(emb, face_confidence=0.99)]
        return []

    monkeypatch.setattr(service, "represent_cached", fake_represent_cached)
    monkeypatch.setattr(service, "find_threshold", lambda m, met: 0.30)

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (10, 20, 30)).save(buf, format="PNG")
    noface_png = buf.getvalue()

    resp = client.post(
        "/v1/intelligence/face/verify",
        headers=AUTH,
        files=[
            ("img1", ("ref.png", png_image)),
            ("img2", ("c1.png", png_image)),      # 64x48 → has face
            ("img2", ("c2.png", noface_png)),     # 32x32 → no face → item error
        ],
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["model"] == "Facenet512"
    assert body["reference"]["confidence"] == 0.99
    assert len(body["results"]) == 2
    verified = {r["index"]: r for r in body["results"]}
    assert verified[0]["verified"] is True
    assert verified[1]["error"] == "no_face_detected"


def test_verify_no_reference_face_422(client, monkeypatch, png_image):
    from app.face import service

    monkeypatch.setattr(service, "represent_cached", lambda bgr, **k: [])
    resp = client.post(
        "/v1/intelligence/face/verify",
        headers=AUTH,
        files=[("img1", ("ref.png", png_image)), ("img2", ("c1.png", png_image))],
    )
    assert resp.status_code == 422


def test_verify_too_many_candidates_422(client, monkeypatch, png_image):
    from app.face import service

    monkeypatch.setattr(
        service,
        "represent_cached",
        lambda bgr, **k: [{"embedding": [1.0], "facial_area": _fake_face()["facial_area"], "face_confidence": 0.9}],
    )
    files = [("img1", ("ref.png", png_image))] + [("img2", (f"c{i}.png", png_image)) for i in range(51)]
    resp = client.post("/v1/intelligence/face/verify", headers=AUTH, files=files)
    assert resp.status_code == 422
