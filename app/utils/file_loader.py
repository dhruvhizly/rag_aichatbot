from __future__ import annotations

from pathlib import Path


class UnsupportedDocumentError(ValueError):
    pass


def extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        raw = file_path.read_text(encoding="utf-8", errors="replace")
        return raw.strip()
    if suffix == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        pages = PyPDFLoader(str(file_path)).load()
        parts = [p.page_content.strip() for p in pages if p.page_content and p.page_content.strip()]
        return "\n\n".join(parts).strip()
    raise UnsupportedDocumentError(f"Unsupported type {suffix!r}; use .pdf or .txt.")
