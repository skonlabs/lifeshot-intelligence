#!/usr/bin/env bash
# =============================================================================
# LIFESHOT Intelligence API — one-shot Ubuntu provisioner
#
# Fresh Ubuntu 22.04 / 24.04 instance -> a working HTTPS API at:
#     https://dev-api.lifeshot.ai/v1/intelligence/<capability>
#
# It installs every prerequisite (Python 3.11, OpenCV/OCR system libs, nginx,
# certbot), clones + builds the app into /opt/lifeshot-intelligence, runs it
# under systemd (gunicorn/uvicorn on 127.0.0.1:8000), and puts nginx in front
# as the public TLS listener. Safe to re-run: it pulls latest, reinstalls
# deps, and restarts (a redeploy).
#
# USAGE
#   sudo DOMAIN=dev-api.lifeshot.ai \
#        REPO_URL=https://github.com/skonlabs/lifeshot-intelligence.git \
#        BRANCH=main \
#        API_KEYS=my-dev-key \
#        LETSENCRYPT_EMAIL=skonlabs@gmail.com \
#        bash provision-ubuntu.sh
#
# Private repo? Pass a token (used only for the clone, not stored):
#        GIT_TOKEN=ghp_xxx ...
#
# OpenAI-backed endpoints (documents/pii-text/moderation/scene)? Add:
#        OPENAI_ENABLED=true OPENAI_API_KEY=sk-...
#
# PREREQUISITES YOU MUST DO FIRST
#   1. A DNS A record for $DOMAIN pointing at THIS instance's public IP.
#   2. Security group / firewall open on TCP 80 and 443 (and 22 for SSH).
#   3. >= 4 GB RAM (TensorFlow + DeepFace weights are memory-heavy).
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Config (override any of these via environment variables)
# ---------------------------------------------------------------------------
DOMAIN="${DOMAIN:-dev-api.lifeshot.ai}"
REPO_URL="${REPO_URL:-https://github.com/skonlabs/lifeshot-intelligence.git}"
BRANCH="${BRANCH:-main}"
APP_DIR="${APP_DIR:-/opt/lifeshot-intelligence}"
APP_USER="${APP_USER:-lifeshot}"
APP_GROUP="${APP_GROUP:-lifeshot}"
SERVICE_NAME="${SERVICE_NAME:-lifeshot-intelligence}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

# App config baked into .env
APP_ENV="${APP_ENV:-development}"                 # development enables /docs for testing
API_KEYS="${API_KEYS:-my-dev-key}"               # CHANGE THIS for anything real
CORS_ORIGINS="${CORS_ORIGINS:-http://localhost:3000,https://${DOMAIN}}"
OPENAI_ENABLED="${OPENAI_ENABLED:-false}"
OPENAI_API_KEY="${OPENAI_API_KEY:-}"

# TLS: auto = try certbot only if $DOMAIN already resolves to this host.
ENABLE_TLS="${ENABLE_TLS:-auto}"                 # auto | true | false
LETSENCRYPT_EMAIL="${LETSENCRYPT_EMAIL:-skonlabs@gmail.com}"

# Private-repo clone auth (optional; never written to disk)
GIT_TOKEN="${GIT_TOKEN:-}"

PYBIN="python${PYTHON_VERSION}"

log()  { echo -e "\n\033[1;34m==>\033[0m \033[1m$*\033[0m"; }
warn() { echo -e "\033[1;33m[warn]\033[0m $*" >&2; }
die()  { echo -e "\033[1;31m[fatal]\033[0m $*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || die "Run as root (use sudo)."
. /etc/os-release 2>/dev/null || true
[ "${ID:-}" = "ubuntu" ] || warn "This script targets Ubuntu; ID=${ID:-unknown}. Continuing anyway."

export DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# 1. Base packages
# ---------------------------------------------------------------------------
log "Installing base packages (git, nginx, curl, ...)"
apt-get update -y
apt-get install -y --no-install-recommends \
    ca-certificates curl git software-properties-common \
    nginx

# ---------------------------------------------------------------------------
# 2. Python 3.11 (via deadsnakes if the distro doesn't ship it)
# ---------------------------------------------------------------------------
if ! command -v "$PYBIN" >/dev/null 2>&1; then
    log "Python ${PYTHON_VERSION} not found — adding deadsnakes PPA"
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update -y
fi
log "Installing Python ${PYTHON_VERSION} + venv + dev headers"
apt-get install -y --no-install-recommends \
    "python${PYTHON_VERSION}" "python${PYTHON_VERSION}-venv" "python${PYTHON_VERSION}-dev"
command -v "$PYBIN" >/dev/null 2>&1 || die "Could not install $PYBIN."

# ---------------------------------------------------------------------------
# 3. Native runtime libs: OpenCV (libGL, glib) + OCR (tesseract)
#    (matches the README "System packages" note)
# ---------------------------------------------------------------------------
log "Installing OpenCV + OCR system libraries"
apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 tesseract-ocr

# ---------------------------------------------------------------------------
# 4. Dedicated non-root service account
# ---------------------------------------------------------------------------
if ! id -u "$APP_USER" >/dev/null 2>&1; then
    log "Creating service user '$APP_USER'"
    useradd --system --create-home --shell /usr/sbin/nologin "$APP_USER"
fi

# ---------------------------------------------------------------------------
# 5. Clone or update the repo (deploy latest code)
# ---------------------------------------------------------------------------
CLONE_URL="$REPO_URL"
if [ -n "$GIT_TOKEN" ]; then
    # Inject token only for this clone/fetch; strip it from stored remote after.
    CLONE_URL="$(echo "$REPO_URL" | sed -E "s#https://#https://${GIT_TOKEN}@#")"
fi

if [ -d "$APP_DIR/.git" ]; then
    log "Repo exists — fetching latest ($BRANCH)"
    git -C "$APP_DIR" remote set-url origin "$CLONE_URL"
    git -C "$APP_DIR" fetch --depth 1 origin "$BRANCH"
    git -C "$APP_DIR" checkout -B "$BRANCH" "origin/$BRANCH"
else
    log "Cloning $REPO_URL ($BRANCH) -> $APP_DIR"
    git clone --depth 1 --branch "$BRANCH" "$CLONE_URL" "$APP_DIR"
fi
# Never leave the token in the stored remote.
git -C "$APP_DIR" remote set-url origin "$REPO_URL"
chown -R "$APP_USER:$APP_GROUP" "$APP_DIR"

# ---------------------------------------------------------------------------
# 6. Python virtualenv + dependencies (+ prefetch DeepFace weights)
# ---------------------------------------------------------------------------
log "Creating virtualenv and installing requirements (this pulls TensorFlow — slow)"
sudo -u "$APP_USER" bash -euo pipefail <<EOF
cd "$APP_DIR"
[ -d .venv ] || $PYBIN -m venv .venv
./.venv/bin/pip install --upgrade pip wheel
./.venv/bin/pip install -r requirements.txt
EOF

# ---------------------------------------------------------------------------
# 7. .env (secrets/config, chmod 600, owned by the service user)
# ---------------------------------------------------------------------------
log "Writing $APP_DIR/.env"
ENV_FILE="$APP_DIR/.env"
cat > "$ENV_FILE" <<EOF
# Generated by provision-ubuntu.sh — edit and 'systemctl restart ${SERVICE_NAME}' to apply.
APP_ENV=${APP_ENV}
LOG_LEVEL=INFO
HOST=127.0.0.1
PORT=8000
CORS_ORIGINS=${CORS_ORIGINS}
ENABLE_DOCS=true

# Auth
API_KEYS=${API_KEYS}

# Face / DeepFace
DEEPFACE_HOME=${APP_DIR}/weights
FACE_WARMUP=true

# OpenAI-backed features (documents/pii-text/moderation/scene)
OPENAI_ENABLED=${OPENAI_ENABLED}
OPENAI_API_KEY=${OPENAI_API_KEY}

# Moderation defaults to OpenAI; without OPENAI_ENABLED it stays disabled.
MODERATION_PROVIDER=openai
EOF
chown "$APP_USER:$APP_GROUP" "$ENV_FILE"
chmod 600 "$ENV_FILE"

log "Prefetching DeepFace model weights (avoids a slow first request)"
sudo -u "$APP_USER" env DEEPFACE_HOME="$APP_DIR/weights" \
    "$APP_DIR/.venv/bin/python" "$APP_DIR/scripts/download_weights.py" \
    || warn "Weight prefetch failed; models will download lazily on first call."

# ---------------------------------------------------------------------------
# 8. systemd unit (gunicorn/uvicorn, bound to loopback)
# ---------------------------------------------------------------------------
log "Installing systemd service '$SERVICE_NAME'"
cat > "/etc/systemd/system/${SERVICE_NAME}.service" <<EOF
[Unit]
Description=LIFESHOT Intelligence API (FastAPI + Gunicorn/Uvicorn)
After=network-online.target
Wants=network-online.target

[Service]
Type=notify
User=${APP_USER}
Group=${APP_GROUP}
WorkingDirectory=${APP_DIR}
EnvironmentFile=${APP_DIR}/.env
Environment=DEEPFACE_HOME=${APP_DIR}/weights
Environment=OMP_NUM_THREADS=2
Environment=TF_CPP_MIN_LOG_LEVEL=2
ExecStart=${APP_DIR}/.venv/bin/gunicorn -c deploy/gunicorn.conf.py app.main:app
ExecReload=/bin/kill -s HUP \$MAINPID
KillSignal=SIGTERM
# Generous: first boot warms TensorFlow/DeepFace models before READY.
TimeoutStartSec=600
TimeoutStopSec=45
Restart=on-failure
RestartSec=3

# --- hardening ---
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
PrivateDevices=true
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectControlGroups=true
RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX
LockPersonality=true
ReadWritePaths=${APP_DIR}/weights
LimitNOFILE=8192

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl restart "$SERVICE_NAME"

# ---------------------------------------------------------------------------
# 9. Wait for the app to become healthy on the loopback
# ---------------------------------------------------------------------------
log "Waiting for the app to answer on http://127.0.0.1:8000/health"
ok=false
for i in $(seq 1 60); do
    if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then ok=true; break; fi
    sleep 5
done
if [ "$ok" != true ]; then
    warn "App did not become healthy in time. Recent logs:"
    journalctl -u "$SERVICE_NAME" -n 40 --no-pager || true
    die "Aborting before nginx wiring — fix the service first (journalctl -u ${SERVICE_NAME})."
fi
log "App is healthy."

# ---------------------------------------------------------------------------
# 10. nginx site for $DOMAIN (public listener -> loopback app)
# ---------------------------------------------------------------------------
log "Configuring nginx site for $DOMAIN"
SITE="/etc/nginx/sites-available/${SERVICE_NAME}"
cat > "$SITE" <<EOF
upstream ${SERVICE_NAME}_upstream {
    server 127.0.0.1:8000 fail_timeout=0;
}

server {
    listen 80;
    listen [::]:80;
    server_name ${DOMAIN};

    # Let certbot solve HTTP-01 challenges here.
    location /.well-known/acme-challenge/ { root /var/www/certbot; }

    # Must match the app's MAX_FILE_MB (10 MB) — reject larger at the edge.
    client_max_body_size 10m;

    # Vision-over-PDF calls take tens of seconds; stay under gunicorn's 180s.
    proxy_connect_timeout 10s;
    proxy_send_timeout    170s;
    proxy_read_timeout    170s;

    location / {
        proxy_pass http://${SERVICE_NAME}_upstream;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_set_header X-Request-Id \$request_id;
        proxy_http_version 1.1;
    }

    location = /health {
        proxy_pass http://${SERVICE_NAME}_upstream;
        access_log off;
    }
}
EOF
mkdir -p /var/www/certbot
ln -sf "$SITE" "/etc/nginx/sites-enabled/${SERVICE_NAME}"
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl reload nginx
log "nginx is serving $DOMAIN over HTTP."

# ---------------------------------------------------------------------------
# 11. TLS via Let's Encrypt (certbot) — auto/true/false
# ---------------------------------------------------------------------------
should_tls=false
case "$ENABLE_TLS" in
    true) should_tls=true ;;
    false) should_tls=false ;;
    auto)
        # Only attempt if $DOMAIN already resolves to one of this host's IPs.
        resolved="$(getent hosts "$DOMAIN" | awk '{print $1}' | head -n1 || true)"
        myips="$(hostname -I 2>/dev/null) $(curl -fsS --max-time 5 https://checkip.amazonaws.com 2>/dev/null || true)"
        if [ -n "$resolved" ] && echo "$myips" | grep -qw "$resolved"; then
            should_tls=true
        else
            warn "DNS for $DOMAIN does not resolve to this host yet (resolved='$resolved'). Skipping TLS."
            warn "Point the DNS A record here, then re-run with ENABLE_TLS=true."
        fi
        ;;
esac

if [ "$should_tls" = true ]; then
    log "Obtaining Let's Encrypt certificate for $DOMAIN"
    apt-get install -y --no-install-recommends certbot python3-certbot-nginx
    if certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos \
            -m "$LETSENCRYPT_EMAIL" --redirect; then
        systemctl reload nginx
        log "TLS enabled. certbot installed a renewal timer."
        SCHEME=https
    else
        warn "certbot failed — leaving HTTP working. Check DNS/firewall then re-run with ENABLE_TLS=true."
        SCHEME=http
    fi
else
    SCHEME=http
fi

# ---------------------------------------------------------------------------
# 12. Done — how to test
# ---------------------------------------------------------------------------
cat <<EOF

========================================================================
 Deploy complete.
========================================================================
 Service : ${SERVICE_NAME}  (systemctl status ${SERVICE_NAME})
 Code    : ${APP_DIR}  (branch ${BRANCH})
 URL     : ${SCHEME}://${DOMAIN}
 API key : ${API_KEYS}   <-- change API_KEYS in ${APP_DIR}/.env for real use

 Quick tests:
   curl -s ${SCHEME}://${DOMAIN}/health
   curl -s ${SCHEME}://${DOMAIN}/ | jq

   # a real capability (face detection):
   curl -s -H "X-API-Key: ${API_KEYS}" -F file=@face.jpg \\
        ${SCHEME}://${DOMAIN}/v1/intelligence/face/detect | jq

 Other endpoints (all POST, all under /v1/intelligence):
   face/detect  face/analyze  face/represent  face/verify
   documents/classify  documents/extract   (need OPENAI_ENABLED=true)
   pii/scan     moderation/scan     scene/describe

 Logs / redeploy:
   journalctl -u ${SERVICE_NAME} -f
   sudo bash $(basename "$0")        # re-run to pull latest & restart
EOF
