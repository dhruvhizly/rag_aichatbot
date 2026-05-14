from __future__ import annotations

import hashlib

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import Settings


def _collection_name(session_id: str) -> str:
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:48]
    return f"rag_{digest}"


class DocumentIndex:
    def __init__(self, settings: Settings, embedding_fn) -> None:
        self._persist_dir = settings.chroma_db_dir
        self._embedding_fn = embedding_fn

    def _vectorstore(self, session_id: str) -> Chroma:
        return Chroma(
            collection_name=_collection_name(session_id),
            embedding_function=self._embedding_fn,
            persist_directory=self._persist_dir,
        )

    def add_documents(self, session_id: str, texts: list[str], metadatas: list[dict]) -> None:
        docs = [Document(page_content=text, metadata=meta) for text, meta in zip(texts, metadatas)]
        self._vectorstore(session_id).add_documents(docs)

    def similarity_search(self, session_id: str, query: str, k: int) -> list[str]:
        hits = self._vectorstore(session_id).similarity_search(query, k=k)
        return [d.page_content for d in hits]

    def similarity_search_with_relevance_scores(
        self, session_id: str, query: str, k: int
    ) -> list[tuple[str, float]]:
        pairs = self._vectorstore(session_id).similarity_search_with_relevance_scores(query, k=k)
        return [(doc.page_content, float(score)) for doc, score in pairs]

    def chunk_count(self, session_id: str) -> int:
        try:
            return int(self._vectorstore(session_id)._collection.count())
        except Exception:
            return 0
