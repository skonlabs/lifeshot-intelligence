"""Gunicorn config for the LIFESHOT Intelligence API.

Uvicorn workers, bound to loopback (nginx is the public listener). Timeouts are
sized ABOVE the worst-case request (a vision call over a multi-page PDF can take
tens of seconds — the default 30s would kill workers mid-request).

Sizing: each worker loads its OWN copy of the DeepFace/TensorFlow models, so
memory scales with worker count. Start with 2 workers and measure RSS before
raising. On GPU, keep worker count low (GPU memory, not CPU, is the limit).

    gunicorn -c deploy/gunicorn.conf.py app.main:app
"""
import multiprocessing
import os

# --- binding: loopback only ---
bind = os.getenv("GUNICORN_BIND", "127.0.0.1:8000")

# --- workers ---
# Models are heavy; do NOT use the (2*cpu+1) rule blindly — it will OOM.
workers = int(os.getenv("GUNICORN_WORKERS", "2"))
worker_class = "uvicorn.workers.UvicornWorker"

# Async workers with a bounded in-process thread pool (see app/common/pool.py);
# threads here mainly help I/O. Keep modest.
threads = int(os.getenv("GUNICORN_THREADS", "1"))

# --- lifecycle / timeouts ---
# Worst-case = multi-page PDF vision call; keep well above it.
timeout = int(os.getenv("GUNICORN_TIMEOUT", "180"))
graceful_timeout = int(os.getenv("GUNICORN_GRACEFUL_TIMEOUT", "30"))
keepalive = int(os.getenv("GUNICORN_KEEPALIVE", "5"))

# Recycle workers to bound memory growth / leaks (jitter avoids thundering herd).
max_requests = int(os.getenv("GUNICORN_MAX_REQUESTS", "500"))
max_requests_jitter = int(os.getenv("GUNICORN_MAX_REQUESTS_JITTER", "50"))

# --- logging: structured JSON already goes to stdout via the app logger ---
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("GUNICORN_LOGLEVEL", "info")

# Preload is OFF: each worker loads its own models AFTER fork (TF + fork don't
# mix well when models are loaded pre-fork).
preload_app = False


def on_starting(server):
    server.log.info("LIFESHOT Intelligence starting with %s worker(s)", workers)


def worker_int(worker):
    worker.log.info("worker received INT/QUIT — draining")
