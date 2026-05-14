from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    groq_api_key: str = ""
    model: str = Field(
        default="llama3-70b-8192",
        validation_alias=AliasChoices("MODEL", "GROQ_MODEL"),
    )
    agent_temperature: float = 0.35
    app_name: str = "AI Chatbot"
    debug: bool = False
    upload_dir: str = "uploads"
    chroma_db_dir: str = "chroma_db"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_top_k: int = 4
    rag_min_relevance_score: float = 0.06

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
