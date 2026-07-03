# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# LIFESHOT Intelligence API — container image
#
# Runs the FastAPI app under gunicorn+uvicorn workers (see deploy/gunicorn.conf.py).
# Built for a long-running container target (AWS App Runner / ECS Fargate / EC2),
# NOT Amplify Hosting (static/SSR only) and NOT plain zip-based Lambda.
#
# System deps mirror the README "System packages" note:
#   libgl1, libglib2.0-0  -> OpenCV runtime (libGL.so.1, libgthread)
#   tesseract-ocr         -> pytesseract (PII redaction geometry)
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # DeepFace reads/writes model weights here; baked into the image below.
    DEEPFACE_HOME=/app/weights

# --- OS-level runtime deps (OpenCV + OCR) ---
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python deps (cached layer: only re-runs when requirements change) ---
COPY requirements.txt ./
RUN pip install -r requirements.txt

# --- App source ---
COPY . .

# --- Pre-download DeepFace weights so the first request isn't a cold model load
#     and the running container never needs a writable network path.
#     Needs network at build time (CodeBuild/App Runner build have it). ---
RUN python scripts/download_weights.py || echo "weight prefetch skipped"

# App Runner/ECS route traffic to this port; gunicorn must bind 0.0.0.0 (NOT the
# loopback default in deploy/gunicorn.conf.py, which is meant for nginx-fronted VMs).
ENV GUNICORN_BIND=0.0.0.0:8000 \
    APP_ENV=production \
    HOST=0.0.0.0 \
    PORT=8000
EXPOSE 8000

# Warm models at boot so /ready flips true once loaded.
CMD ["gunicorn", "-c", "deploy/gunicorn.conf.py", "app.main:app"]
