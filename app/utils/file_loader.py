from __future__ import annotations

from pathlib import Path


class UnsupportedDocumentError(ValueError):
    pass


def extract_pages(file_path: Path) -> list[str]:
    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        raw = file_path.read_text(encoding="utf-8", errors="replace").strip()
        return [raw] if raw else []
    if suffix == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        pages = PyPDFLoader(str(file_path)).load()
        return [
            p.page_content.strip()
            for p in pages
            if p.page_content and p.page_content.strip()
        ]
    raise UnsupportedDocumentError(f"Unsupported type {suffix!r}; use .pdf or .txt.")


def extract_text(file_path: Path) -> str:
    return "\n\n".join(extract_pages(file_path)).strip()
