from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import get_settings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    settings = get_settings()
    cache_dir = Path(settings.embedding_cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    allow_download = os.environ.get("ALLOW_EMBEDDING_DOWNLOAD") == "1"
    model_present = any(cache_dir.glob("models--*"))
    local_only = not (allow_download or not model_present)

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        cache_folder=str(cache_dir),
        model_kwargs={"device": "cpu", "local_files_only": local_only},
        encode_kwargs={"normalize_embeddings": True},
    )
