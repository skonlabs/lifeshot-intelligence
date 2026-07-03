"""Central application settings (pydantic-settings).

Everything configurable lives here so the rest of the code reads config, never
os.environ. Loaded once and cached via ``get_settings()``.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Annotated, List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# CSV env strings for list fields: disable pydantic-settings' JSON pre-decode so
# our own comma-splitting validator runs instead.
CSVList = Annotated[List[str], NoDecode]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- App ---
    app_env: str = "development"
    app_name: str = "lifeshot-intelligence"
    app_version: str = "1.0.0"
    log_level: str = "INFO"
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: CSVList = Field(default_factory=lambda: ["http://localhost:3000"])
    enable_docs: bool = True

    # --- Auth ---
    api_keys: CSVList = Field(default_factory=list)
    # "keyid:sha256hex,keyid2:sha256hex"
    api_key_hashes: CSVList = Field(default_factory=list)

    # --- Face / DeepFace ---
    deepface_home: str = "./weights"
    face_model: str = "Facenet512"
    face_detector: str = "yunet"
    face_metric: str = "cosine"
    face_warmup: bool = True
    face_pool_workers: int = 2

    # --- OpenAI ---
    openai_enabled: bool = False
    openai_api_key: str = ""
    openai_extract_model: str = "gpt-5.5"
    openai_classify_model: str = "gpt-5.5-mini"
    openai_scene_model: str = "gpt-5.5-mini"
    openai_embed_model: str = "text-embedding-3-small"
    openai_timeout_seconds: float = 60.0
    openai_max_retries: int = 3
    openai_base_url: str = ""

    # --- Moderation ---
    moderation_provider: str = "openai"  # openai | local
    moderation_openai_model: str = "omni-moderation-latest"
    nsfw_threshold: float = 0.5
    nsfw_model: str = ""

    # --- OCR ---
    ocr_engine: str = "tesseract"  # tesseract | paddle

    # --- Input caps ---
    max_file_mb: float = 10.0
    max_megapixels: float = 40.0
    max_pdf_pages: int = 10
    max_verify_candidates: int = 50
    max_scene_pii_pages: int = 5

    # --- Geocoding ---
    geocoding_enabled: bool = False

    # --- Capacity / backpressure ---
    max_inflight_heavy: int = 8
    cache_max_items: int = 1024

    # --- Cost controls ---
    global_spend_cap_usd: float = 50.0
    per_key_rate_per_min: int = 120
    per_key_daily_spend_usd: float = 10.0

    # ---- validators: allow comma-separated env strings for list fields ----
    @field_validator("cors_origins", "api_keys", "api_key_hashes", mode="before")
    @classmethod
    def _split_csv(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"

    @property
    def max_file_bytes(self) -> int:
        return int(self.max_file_mb * 1024 * 1024)

    def openai_ready(self) -> bool:
        """True only when the operator has acknowledged the data path AND a key exists."""
        return self.openai_enabled and bool(self.openai_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
