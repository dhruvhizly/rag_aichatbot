from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import get_settings

logger = logging.getLogger(__name__)


def _detect_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    settings = get_settings()
    cache_dir = Path(settings.embedding_cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    allow_download = os.environ.get("ALLOW_EMBEDDING_DOWNLOAD") == "1"
    model_present = any(cache_dir.glob("models--*"))
    local_only = not (allow_download or not model_present)

    device = _detect_device()
    batch_size = 128 if device == "cuda" else 64

    logger.info("Loading embeddings %s on %s (batch=%d)", settings.embedding_model, device, batch_size)

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        cache_folder=str(cache_dir),
        model_kwargs={"device": device, "local_files_only": local_only},
        encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size},
    )


@lru_cache(maxsize=512)
def _cached_query_vector(text: str) -> tuple[float, ...]:
    return tuple(get_embeddings().embed_query(text))


def embed_query_cached(text: str) -> list[float]:
    return list(_cached_query_vector(text))


def prewarm() -> None:
    emb = get_embeddings()
    emb.embed_query("warmup")
