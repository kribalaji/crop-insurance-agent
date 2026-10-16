"""Application settings, loaded from environment variables and `.env`."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["gemini", "anthropic", "ollama"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: LLMProvider = "gemini"
    llm_model: str | None = None
    llm_temperature: float = 0.0

    google_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434"

    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_enabled: bool = True  # tracing also needs both Langfuse keys

    data_dir: Path = Path("data")
    cache_dir: Path = Path("cache")
    policies_dir: Path = Path("policies")
    chroma_dir: Path | None = None  # defaults to $CACHE_DIR/chroma

    @property
    def chroma_path(self) -> Path:
        return self.chroma_dir or self.cache_dir / "chroma"


@lru_cache
def get_settings() -> Settings:
    return Settings()
