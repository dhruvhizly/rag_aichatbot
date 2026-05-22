from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from sentence_transformers import CrossEncoder

from app.config import get_settings

logger = logging.getLogger(__name__)


def _detect_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    settings = get_settings()
    cache_dir = Path(settings.embedding_cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = _detect_device()
    logger.info("Loading reranker %s on %s", settings.reranker_model, device)
    return CrossEncoder(
        settings.reranker_model,
        device=device,
        max_length=512,
        model_kwargs={"cache_dir": str(cache_dir)},
        local_files_only=True,
    )


def rerank(query: str, candidates: list[str], top_k: int) -> list[tuple[str, float]]:
    if not candidates or top_k <= 0:
        return []
    model = get_reranker()
    pairs = [(query, c) for c in candidates]
    scores = model.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [(c, float(s)) for c, s in ranked[:top_k]]


def prewarm() -> None:
    model = get_reranker()
    model.predict([("warmup query", "warmup document")])
