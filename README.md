# LIFESHOT Intelligence API

A performance-critical HTTP API for multimodal image intelligence:

| Capability | What it does | Engine |
|---|---|---|
| **Face** | detection · attribute analysis · 1:N verify · embeddings | `deepface` (Facenet512 / yunet) |
| **Documents** | document-type classification · content/field extraction | OpenAI vision + Structured Outputs |
| **PII scan** | detect text PII **and** faces (biometric PII) in any image/PDF, optional pixel redaction | composes documents + face + OCR |
| **Moderation** | flag adult sexual content with a score + category breakdown | OpenAI `omni-moderation` (or local ONNX) |
| **Scene** | natural-language caption + structured facets for library search | OpenAI vision + Structured Outputs |

Runtime: **Python 3.11 · FastAPI + Uvicorn · Pydantic v2**. Public base URL
`https://api.lifeshot.ai`; every capability is namespaced under
`/v1/intelligence/`.

---

## 1. Feature-based layout

Everything for one capability lives in one folder — open it to understand it.

```
app/
├── main.py            # creates the app, mounts routers under /v1/intelligence, warm-up + lifecycle
├── config.py          # all settings (pydantic-settings)
├── common/            # shared plumbing (auth, logging+PII scrubber, errors, images, pdf, ssrf,
│                      #   openai_client [the ONLY openai import], pool, cost, cache, responses)
├── face/              # router.py · service.py [the ONLY deepface import] · schemas.py
├── documents/         # router.py · service.py · schemas.py
├── pii/               # router.py · service.py [the ONLY OCR import; composes face+documents] · schemas.py
├── moderation/        # router.py · service.py [the ONLY local-NSFW import] · schemas.py
└── scene/             # router.py · service.py · schemas.py
```

**Rules that keep it clean**

* Each feature folder = `router.py` (HTTP only, **no SDKs**) + `service.py`
  (logic + external calls) + `schemas.py` (Pydantic models).
* A given SDK/model is imported in **exactly one** place:
  * `deepface` → `app/face/service.py`
  * `openai` → `app/common/openai_client.py`
  * OCR (`pytesseract`/`paddleocr`) → `app/pii/service.py`
  * local NSFW model (`onnxruntime`) → `app/moderation/service.py`
* `app/pii/service.py` is the one intentional cross-feature composer — it reuses
  `face/service.py` and `common/openai_client` because PII scanning genuinely
  uses both.

The `intelligence` in the URL is a router **mount prefix** set once in
`main.py`; there is deliberately no `intelligence/` code folder.

---

## 2. Quick start (local dev)

```bash
cd lifeshot-intelligence

# project-local virtualenv (git-ignored)
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # runtime
pip install -r requirements-dev.txt      # + pytest for the test suite

# config
cp .env.example .env
#   set API_KEYS=... ; to enable OpenAI features set OPENAI_ENABLED=true and OPENAI_API_KEY=...

# pre-download DeepFace weights into ./weights (DEEPFACE_HOME)
python scripts/download_weights.py

# run (dev)
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Then:

```bash
curl -s localhost:8000/health
curl -s localhost:8000/ready
curl -s -H "X-API-Key: $API_KEY" -F file=@face.jpg \
     localhost:8000/v1/intelligence/face/detect | jq
```

**System packages** (OpenCV + OCR):

```bash
sudo apt-get install -y libgl1 libglib2.0-0 tesseract-ocr
```

> OpenAI-backed features (documents, pii text, moderation-openai, scene) are
> **disabled** until you set `OPENAI_ENABLED=true` **and** provide
> `OPENAI_API_KEY`. This is an explicit acknowledgement of the data-handling
> posture — see [Compliance](#7-compliance--data-handling).

---

## 3. Endpoints

Auth: every `/v1/intelligence/*` endpoint requires an API key
(`X-API-Key: <key>` or `Authorization: Bearer <key>`). Only `/health`,
`/ready`, `/` are open.

Image input for every capability endpoint: **exactly one** of a multipart `file`
upload, a `url`, or a `base64` string (none / more than one → `422`). Send
either `multipart/form-data` (file + form fields) or `application/json`
(`url`/`base64` + params).

| Method & path | Purpose |
|---|---|
| `POST /v1/intelligence/face/detect` | boxes (original-pixel coords) + `image` metadata block; `return_faces=true` → raw cv2 crop |
| `POST /v1/intelligence/face/analyze` | per-face age/gender/emotion/race (`actions` subset for latency) |
| `POST /v1/intelligence/face/verify` | 1:N — `img1` reference + `img2[]` candidates; embed-once reference |
| `POST /v1/intelligence/face/represent` | embeddings (Facenet512 → 512-dim) |
| `POST /v1/intelligence/documents/classify` | `document_type` + confidence + candidates |
| `POST /v1/intelligence/documents/extract` | fields/content; optional `pii_scan`, `nsfw_scan`, `redact` |
| `POST /v1/intelligence/pii/scan` | text PII + faces; validate/mask; optional `redact`, `nsfw_scan` |
| `POST /v1/intelligence/moderation/scan` | `nsfw` bool + score + categories; `threshold` override |
| `POST /v1/intelligence/scene/describe` | caption + facets (`detail`, `include_embedding`, `known_facets`) |
| `GET /health` · `GET /ready` · `GET /` | liveness · readiness · service info |

### Conventions (all capability endpoints)

* **Request ID** on every response (`request_id` field + `X-Request-Id` header),
  matching the logs.
* **Success envelope** includes `timing_ms` (per-stage) and echoes what ran
  (`model`/`detector`/`provider`).
* **Error envelope** (uniform): `{"error": {"code","message","request_id"}}`.
  Codes: `400` bad input · `401/403` auth · `404` url fetch failed · `413` too
  large · `422` validation / no-face-when-enforced · `429` rate/quota/spend
  (`Retry-After` + `X-RateLimit-*`) · `500` internal. Stack traces never leak.
* **Per-request hard caps** (config): max file size, max megapixels, max PDF
  pages, max `img2` candidates, max scene/PII pages. Exceeding → `413`/`422`.
* **Idempotency**: OpenAI-backed endpoints accept an `Idempotency-Key` header;
  a repeated key returns the stored result instead of re-calling OpenAI.
* **No-face behavior**: single-image endpoints with `enforce_detection=true` +
  no face → `422`; in `verify`, a missing **reference** face → `422`, a missing
  **candidate** face errors that item only.
* **CORS** locked to configured origins in prod (never `*`).

### Example responses

`face/detect`:

```jsonc
{
  "image": { "format":"JPEG","mime_type":"image/jpeg","size_bytes":81234,
             "width":4032,"height":3024,"orientation":"landscape","megapixels":12.19,
             "has_alpha":false,"dpi":[72,72],"taken_at":"2024:07:01 18:22:10",
             "gps":{"lat":48.8584,"lon":2.2945,"altitude":35.0},
             "camera":{"make":"Apple","model":"iPhone 15","iso":80,"f_number":1.78,
                       "exposure_time":"1/120","focal_length":6.86,"lens":null},
             "exif_present":true },
  "count": 1,
  "faces": [{ "facial_area":{"x":1180,"y":760,"w":540,"h":540,
                             "left_eye":[1320,900],"right_eye":[1560,905]},
              "confidence": 0.998 }],
  "detector": "yunet",
  "request_id": "req_ab12…", "timing_ms": {"inference": 210.4, "total": 245.1}
}
```

> `facial_area` is in **original-image pixel coordinates** so the caller can
> crop/overlay the image it already holds — no payload bloat, no server-side
> storage of biometric crops. `return_faces=true` returns a **raw cv2 crop**
> (base64), not DeepFace's aligned/normalized array. Crops and EXIF **GPS** are
> **never persisted**.

`moderation/scan`:

```json
{ "nsfw": true, "nsfw_score": 0.97,
  "categories": { "sexual": 0.97, "suggestive": 0.40, "sexual_minors": 0.01 },
  "provider": "openai-omni-moderation", "threshold": 0.5,
  "sexual_minors_flagged": false }
```

`pii/scan` (values are **masked**; raw values never leave the service):

```json
{ "pii_found": true,
  "counts_by_type": { "ssn": 1, "email_address": 1, "face": 1 },
  "entities": [
    { "type":"ssn","masked_value":"***-**-6789","confidence":0.98,"valid":true,
      "location":{"x":120,"y":88,"w":180,"h":22,"page":0} },
    { "type":"face","confidence":0.99,"valid":true,
      "location":{"x":40,"y":30,"w":110,"h":130,"page":0} }
  ],
  "redacted_image": null }
```

`scene/describe` — see the task spec; `is_new_facet` flags facets the model
coined outside the supplied `known_facets` vocabulary so you can review + adopt
them. Index `caption` + `tags` for search and store a caption vector in pgvector
for semantic retrieval.

---

## 4. Production deployment (bare Linux, no Docker)

Serving stack: **Gunicorn + Uvicorn workers → nginx → systemd**.

```bash
# 1. lay down code + venv under /opt
sudo mkdir -p /opt/lifeshot-intelligence && sudo chown lifeshot:lifeshot /opt/lifeshot-intelligence
sudo -u lifeshot git -C /opt/lifeshot-intelligence clone <repo> .   # or rsync this folder
cd /opt/lifeshot-intelligence
sudo -u lifeshot python3.11 -m venv .venv
sudo -u lifeshot .venv/bin/pip install -r requirements.txt
sudo apt-get install -y libgl1 libglib2.0-0 tesseract-ocr

# 2. secrets (chmod 600, owned by the service user)
sudo -u lifeshot cp .env.example .env && sudo chmod 600 .env   # then edit
#    OPENAI_API_KEY comes from your secret manager, NOT git.

# 3. pre-download weights (readable by the serving user)
sudo -u lifeshot DEEPFACE_HOME=/opt/lifeshot-intelligence/weights .venv/bin/python scripts/download_weights.py

# 4. systemd
sudo cp deploy/lifeshot-intelligence.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now lifeshot-intelligence

# 5. nginx + TLS
sudo cp deploy/nginx.conf /etc/nginx/sites-available/lifeshot-intelligence
sudo ln -s /etc/nginx/sites-available/lifeshot-intelligence /etc/nginx/sites-enabled/
sudo certbot --nginx -d api.lifeshot.ai         # Let's Encrypt, auto-renews
sudo nginx -t && sudo systemctl reload nginx
```

The unit runs as a **dedicated non-root user** (`lifeshot`) with sandboxing
(`NoNewPrivileges`, `ProtectSystem=strict`, `PrivateTmp`; only `weights/`
writable). The app binds `127.0.0.1` only — **nginx is the sole public
listener**; firewall accordingly. `deploy/gunicorn.conf.py` sets `timeout=180s`
(well above the worst-case multi-page vision call — the default 30s would kill
workers mid-request), `graceful_timeout`, and `max_requests` (+jitter) to
recycle workers.

**Deploy / rollback runbook.** The app is stateless. Deploy = update code +
`systemctl reload lifeshot-intelligence` (Gunicorn graceful `HUP` — near-zero
downtime), or run two instances behind nginx and swap. Rollback = check out the
previous pinned build and reload. `/ready` returns `503` until models are warm,
so the load balancer only routes to a ready worker.

**Workers vs RAM.** Each worker loads its **own** DeepFace/TensorFlow models, so
memory scales with `GUNICORN_WORKERS`. Start at 2, measure worker RSS, then
raise. Preload is intentionally **off** (TF + fork don't mix pre-fork).

**GPU.** GPU is the biggest latency lever. Swap `tensorflow` for the CUDA build
(`pip install tensorflow[and-cuda]`), keep worker count low (GPU memory is the
limit, not CPU), and verify `nvidia-smi` shows the process.

---

## 5. Configuration (env)

All settings live in `app/config.py` (pydantic-settings) and are read from env /
`.env`. Highlights (full list in `.env.example`):

| Var | Meaning |
|---|---|
| `API_KEYS` / `API_KEY_HASHES` | auth keys (plaintext dev / `keyid:sha256` prod) |
| `DEEPFACE_HOME` | face weights dir (`./weights`) |
| `FACE_MODEL` / `FACE_DETECTOR` / `FACE_METRIC` | `Facenet512` / `yunet` / `cosine` |
| `FACE_WARMUP` / `FACE_POOL_WORKERS` | warm at startup / inference threads |
| `OPENAI_ENABLED` | **data-handling ack** — gates all OpenAI features |
| `OPENAI_API_KEY` | from secret manager only |
| `OPENAI_EXTRACT_MODEL` / `OPENAI_CLASSIFY_MODEL` / `OPENAI_SCENE_MODEL` | model ids |
| `MODERATION_PROVIDER` | `openai` \| `local` |
| `NSFW_THRESHOLD` / `NSFW_MODEL` | threshold / local ONNX path |
| `OCR_ENGINE` | `tesseract` \| `paddle` |
| `MAX_FILE_MB` / `MAX_MEGAPIXELS` / `MAX_PDF_PAGES` / `MAX_VERIFY_CANDIDATES` / `MAX_SCENE_PII_PAGES` | hard caps |
| `MAX_INFLIGHT_HEAVY` / `CACHE_MAX_ITEMS` | backpressure / cache bound |
| `GLOBAL_SPEND_CAP_USD` / `PER_KEY_RATE_PER_MIN` / `PER_KEY_DAILY_SPEND_USD` | cost guard |
| `CORS_ORIGINS` | allowed origins (never `*` in prod) |

> **Model IDs.** Defaults are `gpt-5.5` (extract/PII), `gpt-5.5-mini`
> (classify/scene), `omni-moderation-latest`. **Confirm current model IDs
> against your OpenAI account** and adjust — they're config, not code.

---

## 6. Performance

* **Face:** models loaded once + warmed at startup (`main.py` → `face/service`);
  `yunet` default; **embed-once** for verify with an embedding cache; inference
  runs off the event loop in a bounded thread pool; TF calls serialized per
  model (TF isn't reliably thread-safe); per-stage `timing_ms`. GPU is the
  biggest lever. Use pgvector for stable galleries (stretch → `/face/find`).
* **OpenAI/PII/moderation/scene:** external calls are the cost/latency center —
  tight strict schemas, a smaller model for classification, PDF page caps,
  classification/description caching by image hash, client timeouts + bounded
  retries (centralized in `common/openai_client`). Scene descriptions at ingest
  are a good **Batch API** fit for backlogs.
* **Backpressure:** heavy work is bounded by a semaphore; when saturated the API
  returns `503` (no unbounded queue / OOM). Caches are **per-worker, bounded
  LRU**; note Redis as the shared path once you run multiple workers/hosts.
* **Targets:** log p50/p95 per endpoint; single-image face detect/analyze well
  under ~1s on suitable hardware. `scripts/benchmark.py` measures a path.

---

## 7. Compliance & data-handling

**PII is never logged.** The structured logger scrubs values and logs only
entity **types**, **counts**, and salted **hashes**. No PII appears in
client-facing errors. Asserted by `tests/test_pii.py::test_no_pii_in_logs`.

**OpenAI data path.** Documents/PII-text/moderation-openai/scene send image (and
for PII, text) content to OpenAI. Inputs aren't trained on by default; for
regulated PII use **Zero-Data-Retention / an enterprise DPA**. Features are
gated behind `OPENAI_ENABLED` (an explicit ack). **Verify current OpenAI
terms.** The privacy-sensitive nature of moderation is a reason to consider the
**local** NSFW provider.

**No raw persistence by default.** Classifications/descriptions may be cached by
image hash; raw PII values are **not** cached (if you enable it: encrypt +
short TTL). Face **crops** and EXIF **GPS** are never persisted. GPS is location
PII (often someone's home) — scrubbed from logs; reverse-geocoding to a place
name is a separate, opt-in external call (`GEOCODING_ENABLED`, off by default).

**Biometric + PII regimes.** Faces and inferred age/gender/race are
biometric/special-category data (BIPA, GDPR, EU AI Act). SSN/DL/financial
extraction implicates GLBA, PCI-DSS, and state laws. Obtain **consent**, apply
**minimization** and **retention** limits, and document the **OpenAI
processing** relationship for your deployment.

**Face model licensing.** DeepFace is MIT, but some wrapped weights carry their
own (sometimes non-commercial) terms. The shipped default **Facenet512** is a
reasonable commercial choice — record the license of whatever model you ship.

**⚠️ CSAM is out of scope.** The NSFW classifier detects **adult** sexual
content only. A `sexual_minors` signal is surfaced (`sexual_minors_flagged`) but
must route to a dedicated legal/operational workflow — in the US, mandatory
**NCMEC** reporting, handled via hash-matching programs (e.g. PhotoDNA) under
strict agreements with counsel. **Do not** treat it as a normal result and
**do not** persist or forward such content casually. This service intentionally
does not implement CSAM detection.

---

## 8. Testing

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q
```

The suite runs **offline** — DeepFace and OpenAI are mocked at their service
boundaries; all fixtures are synthetic (no real PII, no explicit imagery). It
covers: image loader/magic-bytes, metadata, PDF rasterize + page cap, SSRF
blocking, PII validators/masking/`pii_found`/redaction, "no PII in logs",
moderation threshold → `nsfw` (+ `sexual_minors` surfacing), face detect + 1:N
verify (incl. no-face-candidate item error + invalid-option `422`), document
classify/extract (+ `pii_scan`/`nsfw_scan` merge), and scene caption/facets.

---

## 9. Operations checklist

* **Health/observability:** `/health` (process up) vs `/ready` (models warm) are
  distinct; the LB routes on `/ready`. Structured JSON logs to journald carry a
  per-request id (PII-scrubbed). Wire latency (p50/p95), error-rate, and OpenAI
  **spend** metrics + alerts (Prometheus/APM); optional Sentry with PII
  scrubbing on.
* **Cost guard:** per-key rate limits + a daily global/per-key USD cap that
  **fails closed** (`429`) on breach so a bug or abuser can't run up the bill;
  alerts fire before the cap.
* **Security:** app binds loopback; systemd non-root + sandboxed; TLS
  auto-renews; API keys stored hashed + constant-time compared + rotatable;
  `.env` git-ignored (`chmod 600`); uploads verified by **magic bytes** (not
  content-type); `/docs` gated/disabled in prod.
* **Reproducibility:** hash-pin deps (`pip-compile --generate-hashes` →
  `requirements.lock`) and pin the Python patch version.

---

## 10. Stretch goals (not yet implemented)

* pgvector/FAISS face gallery → `/v1/intelligence/face/find`.
* NudeNet region detection to blur nudity (reuse the redaction path).
* Provider interface so OpenAI is swappable for an on-prem VLM.
