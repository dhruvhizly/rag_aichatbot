from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import Settings

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "rag_main"

class DocumentIndex:
    def __init__(self, settings: Settings, embedding_fn) -> None:
        self._persist_dir = settings.chroma_db_dir
        self._embedding_fn = embedding_fn
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chroma")
        self._client: Any = None
        self._collection: Any = None
        self._sessions_with_docs: set[str] = set()
        self._executor.submit(self._ensure_collection).result()
        self._executor.submit(self._load_existing_sessions).result()

    def _ensure_collection(self):
        if self._collection is not None:
            return self._collection
        self._client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:search_ef": 40,
                "hnsw:construction_ef": 100,
                "hnsw:M": 16,
            },
        )
        return self._collection

    def _load_existing_sessions(self) -> None:
        try:
            data = self._collection.get(include=["metadatas"])
            for md in data.get("metadatas") or []:
                sid = (md or {}).get("session_id")
                if sid:
                    self._sessions_with_docs.add(sid)
            if self._sessions_with_docs:
                logger.info("Loaded %d existing session(s) from index", len(self._sessions_with_docs))
        except Exception:
            logger.exception("Failed to enumerate existing sessions; starting empty")

    def _add_sync(self, session_id: str, texts: list[str], metadatas: list[dict]) -> None:
        col = self._ensure_collection()
        vectors = self._embedding_fn.embed_documents(texts)
        ids = [uuid.uuid4().hex for _ in texts]
        md = [{"session_id": session_id, **(m or {})} for m in metadatas]
        col.add(ids=ids, documents=texts, embeddings=vectors, metadatas=md)
        self._sessions_with_docs.add(session_id)

    def _search_sync(self, session_id: str, query: str, k: int) -> list[str]:
        col = self._ensure_collection()
        from app.services.embeddings import embed_query_cached

        query_vec = embed_query_cached(query)
        res = col.query(
            query_embeddings=[query_vec],
            n_results=k,
            where={"session_id": session_id},
        )
        docs_outer = res.get("documents") or [[]]
        return list(docs_outer[0])

    def add_documents(self, session_id: str, texts: list[str], metadatas: list[dict]) -> None:
        self._executor.submit(self._add_sync, session_id, texts, metadatas).result()

    def similarity_search(self, session_id: str, query: str, k: int) -> list[str]:
        return self._executor.submit(self._search_sync, session_id, query, k).result()

    def has_session(self, session_id: str) -> bool:
        return session_id in self._sessions_with_docs

    def prewarm(self) -> None:
        self._executor.submit(self._ensure_collection).result()
