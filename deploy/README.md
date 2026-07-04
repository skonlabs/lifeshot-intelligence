# Deploying LIFESHOT Intelligence

This directory holds everything needed to run the API as a long-lived service
behind nginx. There are two supported shapes:

- **`provision-ubuntu.sh`** — one-shot installer for a fresh Ubuntu VM/instance
  (EC2, DigitalOcean, bare metal). **Recommended for `dev-api.blueokra.ai`.**
- **`../Dockerfile`** — container image for App Runner / ECS Fargate / any
  container host. (Amplify *Hosting* cannot run this app — it is static/SSR only.)

Supporting files: `gunicorn.conf.py` (worker/timeouts), `nginx.conf` (reference
site for `api.blueokra.ai`), `lifeshot-intelligence.service` (reference systemd
unit). `provision-ubuntu.sh` generates its own site + unit from these patterns.

---

## Quick start (Ubuntu)

On a fresh Ubuntu 22.04 / 24.04 instance:

```bash
git clone https://github.com/skonlabs/lifeshot-intelligence.git
cd lifeshot-intelligence

sudo API_KEYS=my-dev-key \
     LETSENCRYPT_EMAIL=skonlabs@gmail.com \
     bash deploy/provision-ubuntu.sh
```

When it finishes, test it:

```bash
curl -s https://dev-api.blueokra.ai/health
curl -s -H "X-API-Key: my-dev-key" -F file=@face.jpg \
     https://dev-api.blueokra.ai/v1/intelligence/face/detect | jq
```

The script is **idempotent** — re-running pulls the latest code, reinstalls
dependencies, and restarts the service (i.e. it doubles as your redeploy).

---

## Prerequisites you MUST set up first

The script cannot do these for you:

1. **DNS** — an `A` record for your domain (default `dev-api.blueokra.ai`)
   pointing at the instance's public IP.
2. **Firewall / security group** — inbound TCP **80** and **443** open
   (plus **22** for SSH).
3. **RAM** — **≥ 4 GB**. TensorFlow + DeepFace weights are memory-heavy; workers
   will OOM on micro instances.
4. **Private repo access** — if the repo is private, pass `GIT_TOKEN` (below).

TLS is `auto`: certbot runs only once the domain resolves to this host. If DNS
isn't ready yet, the script leaves the API working over **HTTP** and tells you
to re-run with `ENABLE_TLS=true` after DNS propagates.

---

## Configuration variables

All are environment variables passed to the script (`sudo VAR=value bash …`).

| Variable | Default | Purpose |
|---|---|---|
| `DOMAIN` | `dev-api.blueokra.ai` | Public hostname; nginx `server_name` + cert domain. |
| `REPO_URL` | `https://github.com/skonlabs/lifeshot-intelligence.git` | Git repo to deploy. |
| `BRANCH` | `main` | Branch to check out. |
| `API_KEYS` | `my-dev-key` | Comma-separated API keys. **Change for real use.** |
| `LETSENCRYPT_EMAIL` | `skonlabs@gmail.com` | Contact email for the TLS cert. |
| `ENABLE_TLS` | `auto` | `auto` \| `true` \| `false`. Whether to run certbot. |
| `APP_ENV` | `development` | `development` enables `/docs`; use `production` to disable. |
| `CORS_ORIGINS` | `http://localhost:3000,https://$DOMAIN` | Allowed CORS origins. |
| `OPENAI_ENABLED` | `false` | `true` to enable documents/pii-text/moderation/scene. |
| `OPENAI_API_KEY` | *(empty)* | Required when `OPENAI_ENABLED=true`. |
| `GIT_TOKEN` | *(empty)* | PAT for cloning a private repo (not stored on disk). |
| `APP_DIR` | `/opt/lifeshot-intelligence` | Install location. |
| `APP_USER` / `APP_GROUP` | `lifeshot` | Dedicated non-root service account. |
| `SERVICE_NAME` | `lifeshot-intelligence` | systemd unit + nginx site name. |
| `PYTHON_VERSION` | `3.11` | Installed via deadsnakes if absent. |

### Examples

Private repo + OpenAI features, production mode:

```bash
sudo DOMAIN=dev-api.blueokra.ai \
     GIT_TOKEN=ghp_xxx \
     BRANCH=main \
     APP_ENV=production \
     API_KEYS=$(python3 -c 'import secrets;print(secrets.token_urlsafe(32))') \
     OPENAI_ENABLED=true OPENAI_API_KEY=sk-... \
     LETSENCRYPT_EMAIL=skonlabs@gmail.com \
     bash deploy/provision-ubuntu.sh
```

Force TLS after DNS is ready:

```bash
sudo ENABLE_TLS=true bash deploy/provision-ubuntu.sh
```

---

## What the script installs / creates

- **Packages**: Python 3.11 (+venv, +dev), `libgl1`, `libglib2.0-0`,
  `tesseract-ocr` (OpenCV + OCR), `nginx`, `git`, `certbot` + nginx plugin.
- **Service account**: `lifeshot` (system user, no login).
- **App**: cloned to `/opt/lifeshot-intelligence`, venv at `.venv`, DeepFace
  weights prefetched into `weights/`.
- **`.env`**: written at `chmod 600`, owned by `lifeshot` (holds API keys /
  OpenAI key). Edit it and `systemctl restart lifeshot-intelligence` to apply.
- **systemd unit** `lifeshot-intelligence.service`: gunicorn/uvicorn on
  `127.0.0.1:8000`, hardened sandbox, warm-up-aware start timeout.
- **nginx site**: public listener for `$DOMAIN` → the loopback app, with
  `client_max_body_size 10m` and vision-friendly proxy timeouts.

---

## Operating it

```bash
# status & logs
systemctl status lifeshot-intelligence
journalctl -u lifeshot-intelligence -f

# restart after editing .env
sudo systemctl restart lifeshot-intelligence

# graceful reload (near-zero downtime)
sudo systemctl reload lifeshot-intelligence

# redeploy latest code
sudo bash deploy/provision-ubuntu.sh
```

### Endpoints

All capabilities are `POST` under `/v1/intelligence` and require an API key
(`X-API-Key: <key>` or `Authorization: Bearer <key>`):

```
face/detect   face/analyze   face/represent   face/verify
documents/classify   documents/extract        # need OPENAI_ENABLED=true
pii/scan   moderation/scan   scene/describe
```

Open (no auth): `GET /health`, `GET /ready`, `GET /`.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| All endpoints `404` | You're hitting the wrong server, or the app isn't running. Check `systemctl status lifeshot-intelligence` and `curl 127.0.0.1:8000/health`. |
| `502 Bad Gateway` from nginx | App not up on `127.0.0.1:8000` — see `journalctl -u lifeshot-intelligence`. |
| certbot fails | DNS for `$DOMAIN` isn't pointing here yet, or ports 80/443 are closed. Fix, then `sudo ENABLE_TLS=true bash deploy/provision-ubuntu.sh`. |
| Workers OOM-killed | Instance has too little RAM — use ≥ 4 GB, or lower `GUNICORN_WORKERS`. |
| `401 unauthorized` | Missing/wrong API key, or `API_KEYS` empty in `.env`. |
| documents/scene return errors | `OPENAI_ENABLED=true` + a valid `OPENAI_API_KEY` are required. |
