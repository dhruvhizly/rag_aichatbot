"""One-time bootstrap: download the embedding model into ./models/.

Run this once after `pip install -r requirements.txt`. After it completes
the application runs fully offline.

Usage:
    python scripts/prefetch_embeddings.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ["ALLOW_EMBEDDING_DOWNLOAD"] = "1"

from app.config import get_settings  # noqa: E402
from app.services.embeddings import get_embeddings  # noqa: E402


def main() -> None:
    settings = get_settings()
    cache_dir = Path(settings.embedding_cache_dir).resolve()
    print(f"Downloading embedding model '{settings.embedding_model}' into {cache_dir} ...")
    embeddings = get_embeddings()
    vec = embeddings.embed_query("warmup")
    print(f"Done. Vector dimension: {len(vec)}")
    print("The application will now run offline.")


if __name__ == "__main__":
    main()
