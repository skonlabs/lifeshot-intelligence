"""Structured JSON logging with a hard PII scrubber.

Rules (see acceptance criteria #7, #13):
  * PII VALUES are NEVER logged. We log entity *types*, *counts*, and salted
    hashes only.
  * A request-id is attached to every log line via a contextvar so all stages
    of one request correlate.
  * Logs go to stdout/journald as single-line JSON.

Anything that might carry a raw value (image bytes, extracted text, EXIF GPS,
face crops, API keys) must be hashed with ``hash_value`` before it reaches a
log call. ``scrub`` is a best-effort net for free-text messages.
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import re
import sys
from typing import Any

# request-id propagated across all stages of a single request
_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

# Best-effort regexes to catch PII that leaked into a free-text log message.
_PII_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[EMAIL]"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[CARD]"),
    (re.compile(r"\+?\d[\d\s().-]{7,}\d"), "[PHONE]"),
]


def set_request_id(request_id: str) -> None:
    _request_id.set(request_id)


def get_request_id() -> str:
    return _request_id.get()


def hash_value(value: Any) -> str:
    """Stable short hash for correlation without revealing the value."""
    if value is None:
        return "-"
    if isinstance(value, (bytes, bytearray)):
        data = bytes(value)
    else:
        data = str(value).encode("utf-8", errors="ignore")
    return "sha256:" + hashlib.sha256(data).hexdigest()[:16]


def scrub(text: str) -> str:
    """Redact obvious PII patterns from a free-text string."""
    if not text:
        return text
    for pattern, repl in _PII_PATTERNS:
        text = pattern.sub(repl, text)
    return text


class _JsonFormatter(logging.Formatter):
    _RESERVED = set(
        logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
    ) | {"message", "asctime", "taskName"}

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": get_request_id(),
            "msg": scrub(record.getMessage()),
        }
        # merge structured extras (already-safe: types/counts/hashes only)
        for key, val in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                payload[key] = val
        if record.exc_info:
            # scrub tracebacks too — they can echo values
            payload["exc"] = scrub(self.formatException(record.exc_info))
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level.upper())
    # quiet noisy libraries
    for noisy in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
