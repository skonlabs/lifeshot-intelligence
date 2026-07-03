"""Scene description schemas (image understanding for library search)."""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class IsNewFacet(BaseModel):
    event: bool = False
    setting: bool = False
    occasion: bool = False
    mood: bool = False


class SceneFacets(BaseModel):
    """Strict Structured-Outputs schema handed to OpenAI.

    ``caption`` + ``tags`` are the free-form primary search signal. The facet
    fields are the clickable filters: the model reuses an existing vocabulary
    label when one fits, else proposes a new label and flags it in is_new_facet.
    """

    caption: str
    event: str
    setting: str
    occasion: str
    activities: List[str]
    objects: List[str]
    people_count: int
    mood: str
    time_of_day: str
    tags: List[str]
    is_new_facet: IsNewFacet


class SceneResponse(BaseModel):
    caption: str
    event: str
    setting: str
    occasion: str
    activities: List[str]
    objects: List[str]
    people_count: int
    mood: str
    time_of_day: str
    tags: List[str]
    is_new_facet: IsNewFacet
    embedding: Optional[List[float]] = None
    model: str
    provider: str = "openai"
    request_id: Optional[str] = None
    timing_ms: Optional[dict] = None
