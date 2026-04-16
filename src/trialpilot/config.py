"""Configuration management for TrialPilot."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model_name: str = field(default_factory=lambda: os.getenv("TRIALPILOT_MODEL", "gpt-4o"))
    temperature: float = field(
        default_factory=lambda: float(os.getenv("TRIALPILOT_TEMPERATURE", "0.2"))
    )
    ctgov_base_url: str = field(
        default_factory=lambda: os.getenv(
            "CTGOV_BASE_URL", "https://clinicaltrials.gov/api/v2"
        )
    )
    log_level: str = field(default_factory=lambda: os.getenv("TRIALPILOT_LOG_LEVEL", "INFO"))
    max_trials_to_fetch: int = 20
    synthetic_cohort_size: int = 200

    def __post_init__(self) -> None:
        if not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required. Set it in your .env file or environment."
            )


def get_settings() -> Settings:
    return Settings()
