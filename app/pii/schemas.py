"""PII scan schemas.

The Structured-Outputs model (``PIIDetection``) is what OpenAI returns: it
CLASSIFIES visible text PII only. Locations (boxes) are added later by OCR /
DeepFace — we never trust VLM coordinates.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

PII_TYPES = (
    "person_name",
    "address",
    "phone_number",
    "email_address",
    "ssn",
    "drivers_license_number",
    "passport_number",
    "date_of_birth",
    "national_id",
    "bank_account_number",
    "routing_number",
    "credit_card_number",
    "other",
)
# 'face' is not a text type — it's added from DeepFace.
ALL_TYPES = PII_TYPES + ("face",)


# ---- Structured Outputs (strict schema handed to OpenAI) ----
class PIITextEntity(BaseModel):
    type: str = Field(description="one of the PII types (text types only, never 'face')")
    value: str = Field(description="the exact visible text of the PII item")
    confidence: float = Field(description="0..1 confidence this is PII of that type")


class PIIDetection(BaseModel):
    entities: List[PIITextEntity]
    full_text: str = Field(description="verbatim visible text, or empty string if not requested")


# ---- HTTP response models ----
class Location(BaseModel):
    x: int
    y: int
    w: int
    h: int
    page: int = 0


class PIIEntity(BaseModel):
    type: str
    value: Optional[str] = None  # omitted/masked for output safety
    masked_value: Optional[str] = None
    confidence: float
    valid: bool
    location: Optional[Location] = None


class PIIScanResponse(BaseModel):
    pii_found: bool
    counts_by_type: dict
    entities: List[PIIEntity]
    redacted_image: Optional[str] = None
    full_text: Optional[str] = None
    nsfw: Optional[bool] = None
    nsfw_score: Optional[float] = None
    provider: str = "openai"
    request_id: Optional[str] = None
    timing_ms: Optional[dict] = None
