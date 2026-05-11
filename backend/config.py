"""
backend/config.py
==================
Application configuration — loaded from environment variables via
pydantic-settings BaseSettings. All secrets live in .env file only.

Usage:
    from backend.config import get_settings
    settings = get_settings()
    api_key = settings.groq_api_key
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Field names map directly to .env variable names (case-insensitive).
    """

    model_config = SettingsConfigDict(
        env_file        = ".env",
        env_file_encoding = "utf-8",
        case_sensitive  = False,
        extra           = "ignore",
    )

    # ── Groq LLM ──────────────────────────────────────────────────────
    groq_api_key: str   = ""
    groq_model:   str   = "llama-3.3-70b-versatile"

    # ── ChromaDB ──────────────────────────────────────────────────────
    chroma_db_path: str = "data/chromadb"

    # ── ML Model ──────────────────────────────────────────────────────
    model_path:     str = "data/models"

    # ── Market Data ───────────────────────────────────────────────────
    vnindex_ticker:  str = "^VNINDEX"
    lookback_days:   int = 504

    # ── Scraper ───────────────────────────────────────────────────────
    max_articles_per_source: int = 8

    # ── FastAPI ───────────────────────────────────────────────────────
    api_host:    str = "0.0.0.0"
    api_port:    int = 8000
    cors_origins: str = "http://localhost:8501"   # Comma-separated list

    # ── Frontend ──────────────────────────────────────────────────────
    backend_url: str = "http://localhost:8000"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return cached Settings instance.
    lru_cache ensures .env is read only once per process lifetime.
    """
    return Settings()
