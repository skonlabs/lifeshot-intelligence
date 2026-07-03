"""Moderation / NSFW schemas."""
from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel


class ModerationResponse(BaseModel):
    nsfw: bool
    nsfw_score: float
    categories: Dict[str, float]
    provider: str
    threshold: float
    # sexual_minors is surfaced but routes to a legal workflow, not a normal flag.
    sexual_minors_flagged: bool = False
    request_id: Optional[str] = None
    timing_ms: Optional[dict] = None
