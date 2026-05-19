from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias=AliasChoices("OLLAMA_BASE_URL", "OLLAMA_HOST"),
    )
    model: str = Field(
        default="qwen2.5:1.5b",
        validation_alias=AliasChoices("MODEL", "OLLAMA_MODEL"),
    )
    app_name: str = "AI Chatbot"
    debug: bool = False
    upload_dir: str = "uploads"
    chroma_db_dir: str = "chroma_db"
    chunk_size: int = 400
    chunk_overlap: int = 50
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_cache_dir: str = "models"
    rag_top_k: int = 1 
    rag_rerank_pool: int = 5
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rag_min_relevance_score: float = 0.06
    history_max_turns: int = 3
    llm_num_ctx: int = 2048
    llm_num_predict: int = 512
    llm_keep_alive: str = "30m"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
