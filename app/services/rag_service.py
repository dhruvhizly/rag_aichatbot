from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.services.vector_store import DocumentIndex
from app.utils.file_loader import extract_pages

_NO_FILES = (
    "[Session context: no documents are indexed for this session. "
    "Tell the user that you don't know the answer.]"
)

_NO_CHUNKS = (
    "[Retrieval result: no excerpts were returned for this query. "
    "Tell the user that you don't know the answer, they can try rephrasing.]"
)


class RAGService:
    def __init__(self, index: DocumentIndex) -> None:
        settings = get_settings()
        self._index = index
        self._top_k = settings.rag_top_k
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        self._retrieval_cache: "OrderedDict[tuple[str, str], tuple[str, bool]]" = OrderedDict()
        self._retrieval_cache_max = 128

    def index_file(self, session_id: str, file_path: Path, source_name: str) -> int:
        pages = extract_pages(file_path)
        if not pages:
            raise ValueError("No extractable text in document.")
        chunks: list[str] = []
        metadatas: list[dict] = []
        for page_no, page_text in enumerate(pages, start=1):
            for chunk in self._splitter.split_text(page_text):
                chunks.append(chunk)
                metadatas.append({"source": source_name, "page": page_no})
        if not chunks:
            raise ValueError("Document produced zero chunks after splitting.")
        self._index.add_documents(session_id, chunks, metadatas)
        self._invalidate_cache(session_id)
        return len(chunks)

    def session_has_indexed_documents(self, session_id: str) -> bool:
        return self._index.has_session(session_id)

    def retrieval_for_chat(self, session_id: str, query: str) -> str:
        block, _ = self.retrieval_and_presence(session_id, query)
        return block

    def retrieval_and_presence(self, session_id: str, query: str) -> tuple[str, bool]:
        key = (session_id, query)
        cached = self._retrieval_cache.get(key)
        if cached is not None:
            self._retrieval_cache.move_to_end(key)
            return cached

        has_docs = self._index.has_session(session_id)
        if not has_docs:
            result = (_NO_FILES, False)
        else:
            docs = self._index.similarity_search(session_id, query, self._top_k)
            if not docs:
                result = (_NO_CHUNKS, True)
            else:
                parts = [f"[Excerpt {i}]\n{block}" for i, block in enumerate(docs, start=1)]
                result = ("\n\n".join(parts), True)

        self._retrieval_cache[key] = result
        if len(self._retrieval_cache) > self._retrieval_cache_max:
            self._retrieval_cache.popitem(last=False)
        return result

    def _invalidate_cache(self, session_id: str) -> None:
        for key in [k for k in self._retrieval_cache if k[0] == session_id]:
            self._retrieval_cache.pop(key, None)
