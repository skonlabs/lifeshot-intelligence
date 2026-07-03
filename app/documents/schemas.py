"""Document classification + extraction schemas.

The Structured-Outputs models (``DocClassification``, ``DocExtraction``) are the
strict JSON schemas we hand OpenAI. Keep them tight — smaller schemas = cheaper,
more reliable calls.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

# Configurable taxonomy. Kept as a plain tuple so it can be swapped/extended.
DOCUMENT_TYPES = (
    "passport",
    "drivers_license",
    "ssn_card",
    "bank_statement",
    "pay_stub",
    "w2",
    "invoice",
    "receipt",
    "utility_bill",
    "insurance_card",
    "medical_record",
    "contract",
    "resume",
    "id_card",
    "generic",
    "unknown",
)


# ---- Structured Outputs (strict schema handed to OpenAI) ----
class DocCandidate(BaseModel):
    document_type: str
    confidence: float


class DocClassification(BaseModel):
    """Strict schema for classify."""

    document_type: str = Field(description="single best document type from the taxonomy")
    confidence: float = Field(description="0..1 confidence in document_type")
    candidates: List[DocCandidate] = Field(description="ranked alternatives incl. the top choice")


class DocExtraction(BaseModel):
    """Strict schema for extract. ``fields`` is a flat map of found values;
    the model is instructed to use 'not found' rather than guessing."""

    document_type: str
    fields: Dict[str, str]
    full_text: str = Field(description="verbatim visible text, or empty string if include_full_text was not requested")


# ---- HTTP response models ----
class ClassifyResponse(BaseModel):
    document_type: str
    confidence: float
    candidates: List[DocCandidate]
    pages: Optional[int] = None
    model: str
    provider: str = "openai"
    request_id: Optional[str] = None
    timing_ms: Optional[dict] = None


class ExtractResponse(BaseModel):
    document_type: Optional[str] = None
    fields: Dict[str, str] = {}
    full_text: Optional[str] = None
    pii_found: Optional[bool] = None
    pii_entities: Optional[List[dict]] = None
    redacted_image: Optional[str] = None
    nsfw: Optional[bool] = None
    nsfw_score: Optional[float] = None
    model: str
    provider: str = "openai"
    request_id: Optional[str] = None
    timing_ms: Optional[dict] = None
