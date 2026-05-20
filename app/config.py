import os
from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY", "False")


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env."""

    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", env_file_encoding="utf-8")

    app_name: str = "AI Buscador RAG"
    database_url: str = f"sqlite:///{BASE_DIR / 'storage' / 'rag.db'}"
    upload_dir: Path = BASE_DIR / "storage" / "uploads"
    chroma_dir: Path = BASE_DIR / "storage" / "chroma"
    chroma_collection: str = "documents"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 900
    chunk_overlap: int = 150
    max_context_tokens: int = Field(default=8000, validation_alias="MAX_CONTEXT_TOKENS")
    default_top_k: int = 5

    llm_provider: str = Field(default="ollama", validation_alias="LLM_PROVIDER")
    model_name: str = Field(default="llama3.1", validation_alias="MODEL_NAME")
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    groq_api_key: str | None = Field(default=None, validation_alias="GROQ_API_KEY")
    ollama_base_url: str = Field(default="http://localhost:11434", validation_alias="OLLAMA_BASE_URL")
    lmstudio_base_url: str = Field(default="http://localhost:1234/v1", validation_alias="LMSTUDIO_BASE_URL")


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return settings
