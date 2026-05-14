from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.services.embeddings import get_embeddings
from app.services.rag_service import RAGService
from app.services.vector_store import DocumentIndex


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    settings = get_settings()
    return RAGService(DocumentIndex(settings, get_embeddings()))
