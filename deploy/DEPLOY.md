# Step-by-Step Deployment — `dev-api.blueokra.ai`

End-to-end deployment of the LIFESHOT Intelligence API. The recommended path is
the automated Ubuntu provisioner (`provision-ubuntu.sh`); a container path is at
the end. For reference material (config variables, troubleshooting) see
[`README.md`](./README.md).

---

## Phase 1 — Prerequisites (the script cannot do these)

1. **VM** — Ubuntu 22.04 / 24.04, **≥ 4 GB RAM**. TensorFlow + DeepFace weights
   are memory-heavy; workers OOM on smaller instances.
2. **DNS** — an `A` record for `dev-api.blueokra.ai` pointing at the VM's public
   IP. TLS will not be issued until this resolves to the box.
3. **Firewall / security group** — inbound TCP **80**, **443**, and **22** (SSH).
4. **Private repo** — a GitHub PAT to pass as `GIT_TOKEN` (used only for the
   clone, never written to disk).
5. **OpenAI key** — only if you need `documents/*`, `pii` text, `moderation`, or
   `scene`. These stay disabled without `OPENAI_ENABLED=true` + a valid key.

---

## Phase 2 — Provision (one command)

```bash
ssh ubuntu@<vm-public-ip>
git clone https://github.com/skonlabs/lifeshot-intelligence.git
cd lifeshot-intelligence
```

Run the provisioner with the domain set to `dev-api.blueokra.ai`:

```bash
sudo DOMAIN=dev-api.blueokra.ai \
     REPO_URL=https://github.com/skonlabs/lifeshot-intelligence.git \
     BRANCH=main \
     GIT_TOKEN=ghp_xxx \
     APP_ENV=production \
     API_KEYS=$(python3 -c 'import secrets;print(secrets.token_urlsafe(32))') \
     OPENAI_ENABLED=true OPENAI_API_KEY=sk-... \
     LETSENCRYPT_EMAIL=skonlabs@gmail.com \
     ENABLE_TLS=auto \
     bash deploy/provision-ubuntu.sh
```

- Drop `GIT_TOKEN` if the repo is public.
- Drop the two `OPENAI_*` lines if you only need the face endpoints.
- Use `APP_ENV=development` to keep interactive `/docs` enabled.
- **Save the `API key` printed at the end** — required for every capability call.

### What the script does, in order

1. Installs base packages (`git`, `nginx`, `curl`).
2. Installs **Python 3.11** (+ venv, + dev headers) via deadsnakes if absent.
3. Installs native libs: `libgl1`, `libglib2.0-0` (OpenCV) and `tesseract-ocr` (OCR).
4. Creates the non-root **`lifeshot`** service account.
5. Clones/updates the repo and checks out `main`.
6. Creates a **virtualenv** and installs `requirements.txt` (pulls TensorFlow —
   this is the slow step, several minutes).
7. Writes **`.env`** (`chmod 600`, owned by `lifeshot`) with your keys/config.
8. Prefetches **DeepFace weights** so the first request isn't a cold model load.
9. Installs the **systemd unit** (gunicorn + uvicorn workers on `127.0.0.1:8000`,
   hardened sandbox) and starts it.
10. Writes the **nginx site** for `dev-api.blueokra.ai` → the loopback app.
11. Waits for `http://127.0.0.1:8000/health` to answer.
12. Runs **certbot** for TLS (only if DNS already resolves to this host).

---

## Phase 3 — Verify

```bash
# liveness — unversioned path, NOT under /v1/intelligence
curl -s https://dev-api.blueokra.ai/health          # {"status":"ok",...}

# readiness — 200 once models are warm, 503 while warming
curl -s https://dev-api.blueokra.ai/ready

# a real capability (face detection)
curl -s -H "X-API-Key: <your-key>" -F file=@face.jpg \
     https://dev-api.blueokra.ai/v1/intelligence/face/detect | jq
```

> Health checks live at `/health` and `/ready` — **not** under the
> `/v1/intelligence` prefix. Point load balancers at `/ready` (it only reports
> healthy once models are warm); use `/health` for liveness/restart decisions.

If TLS was skipped because DNS wasn't ready, the API is live over **HTTP**.
Re-run once DNS propagates:

```bash
sudo DOMAIN=dev-api.blueokra.ai ENABLE_TLS=true bash deploy/provision-ubuntu.sh
```

---

## Phase 4 — Operate & redeploy

```bash
# status & logs
systemctl status lifeshot-intelligence
journalctl -u lifeshot-intelligence -f

# apply an .env change
sudo systemctl restart lifeshot-intelligence

# graceful reload (near-zero downtime)
sudo systemctl reload lifeshot-intelligence

# redeploy latest code — re-run the provisioner (idempotent: pulls latest,
# reinstalls deps, restarts the service)
sudo DOMAIN=dev-api.blueokra.ai bash deploy/provision-ubuntu.sh
```

---

## Container path (App Runner / ECS Fargate)

For a container host instead of a VM:

```bash
docker build -t lifeshot-intelligence .
docker run -p 8000:8000 \
  -e APP_ENV=production \
  -e API_KEYS=<your-key> \
  -e OPENAI_ENABLED=true -e OPENAI_API_KEY=sk-... \
  lifeshot-intelligence
```

- The image binds `0.0.0.0:8000` (no nginx inside) — front it with the
  platform's load balancer / TLS.
- Point the platform **health check at `/ready`** with a generous start-period
  so warmup completes before traffic is routed.
- Model weights are baked in at build time, so the first request isn't a cold load.

---

## Troubleshooting quick reference

| Symptom | Likely cause / fix |
|---|---|
| `404` on `/v1/intelligence/health` | Health routes are unversioned — use `/health` and `/ready`. |
| All endpoints `404` | Wrong server, or app not running. `systemctl status lifeshot-intelligence`; `curl 127.0.0.1:8000/health`. |
| `502 Bad Gateway` from nginx | App not up on `127.0.0.1:8000` — check `journalctl -u lifeshot-intelligence`. |
| certbot fails | DNS for `dev-api.blueokra.ai` not pointing here, or 80/443 closed. Fix, then re-run with `ENABLE_TLS=true`. |
| Workers OOM-killed | Instance has too little RAM — use ≥ 4 GB, or lower `GUNICORN_WORKERS`. |
| `401 unauthorized` | Missing/wrong API key, or `API_KEYS` empty in `.env`. |
| `/ready` returns `503` | Models still warming at startup — expected; wait for warm-up to finish. |
| documents/scene errors | Require `OPENAI_ENABLED=true` + a valid `OPENAI_API_KEY`. |
