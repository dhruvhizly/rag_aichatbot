from __future__ import annotations

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.services.vector_store import DocumentIndex
from app.utils.file_loader import extract_text

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

    def index_file(self, session_id: str, file_path: Path, source_name: str) -> int:
        text = extract_text(file_path)
        if not text:
            raise ValueError("No extractable text in document.")
        chunks = self._splitter.split_text(text)
        if not chunks:
            raise ValueError("Document produced zero chunks after splitting.")
        metadatas = [{"source": source_name} for _ in chunks]
        self._index.add_documents(session_id, chunks, metadatas)
        return len(chunks)

    def session_has_indexed_documents(self, session_id: str) -> bool:
        return self._index.chunk_count(session_id) > 0

    def retrieval_for_chat(self, session_id: str, query: str) -> str:
        block, _ = self.retrieval_and_presence(session_id, query)
        return block

    def retrieval_and_presence(self, session_id: str, query: str) -> tuple[str, bool]:
        has_docs = self._index.chunk_count(session_id) > 0
        if not has_docs:
            return _NO_FILES, False

        pairs = self._index.similarity_search_with_relevance_scores(session_id, query, self._top_k)
        if not pairs:
            return _NO_CHUNKS, True

        texts = [text for text, _ in pairs]
        parts = [f"[Excerpt {i}]\n{block}" for i, block in enumerate(texts, start=1)]
        return "\n\n".join(parts), True
