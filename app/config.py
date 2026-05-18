from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias=AliasChoices("OLLAMA_BASE_URL", "OLLAMA_HOST"),
    )
    model: str = Field(
        default="qwen3.5",
        validation_alias=AliasChoices("MODEL", "OLLAMA_MODEL"),
    )
    app_name: str = "AI Chatbot"
    debug: bool = False
    upload_dir: str = "uploads"
    chroma_db_dir: str = "chroma_db"
    chunk_size: int = 1000
    chunk_overlap: int = 100
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_cache_dir: str = "models"
    rag_top_k: int = 3
    rag_min_relevance_score: float = 0.06
    history_max_turns: int = 6

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
