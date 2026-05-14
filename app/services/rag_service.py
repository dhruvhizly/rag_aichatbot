from __future__ import annotations

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.services.vector_store import DocumentIndex
from app.utils.file_loader import extract_text

_NO_FILES = (
    "[Session context: no documents are indexed for this session. "
    "Answer from general knowledge; use web search for current or external facts when helpful.]"
)

_NO_CHUNKS = (
    "[Retrieval result: no excerpts were returned for this query. "
    "If the user needs live or external facts, use web search. "
    "If they expected content from uploads, say clearly that nothing matched in the indexed files.]"
)

_WEAK_MATCH = (
    "[Retrieval: this session has uploaded files, but their content is not semantically close to this question "
    "(low relevance scores). Do not quote unrelated upload text as an answer. "
    "For weather, news, sports, or other live/general questions, use web search. "
    "Use the calculator only for math the user asks for.]"
)


class RAGService:
    def __init__(self, index: DocumentIndex) -> None:
        settings = get_settings()
        self._index = index
        self._top_k = settings.rag_top_k
        self._min_relevance = settings.rag_min_relevance_score
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

    def retrieval_for_chat(self, session_id: str, query: str) -> tuple[str, bool]:
        """Returns (retrieval_block, anchor_to_uploads)."""
        if self._index.chunk_count(session_id) <= 0:
            return _NO_FILES, False

        pairs = self._index.similarity_search_with_relevance_scores(session_id, query, self._top_k)
        if not pairs:
            return _NO_CHUNKS, False

        best = max(score for _, score in pairs)
        anchor = best >= self._min_relevance
        if not anchor:
            return _WEAK_MATCH, False

        texts = [text for text, _ in pairs]
        parts = [f"[Excerpt {i}]\n{block}" for i, block in enumerate(texts, start=1)]
        return "\n\n".join(parts), True
