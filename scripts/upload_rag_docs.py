"""Upload every file in rag_docs/ to the running /api/upload endpoint."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

SUPPORTED_SUFFIXES = {".pdf", ".txt"}


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    default_docs = repo_root / "rag_docs"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/api/upload")
    parser.add_argument("--session-id", default="default")
    parser.add_argument("--docs-dir", type=Path, default=default_docs)
    args = parser.parse_args()

    if not args.docs_dir.is_dir():
        print(f"Docs directory not found: {args.docs_dir}", file=sys.stderr)
        return 1

    files = sorted(p for p in args.docs_dir.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)
    if not files:
        print(f"No .pdf or .txt files found in {args.docs_dir}", file=sys.stderr)
        return 1

    failures = 0
    for path in files:
        print(f"Uploading {path.name} ...", flush=True)
        with path.open("rb") as fh:
            response = requests.post(
                args.url,
                data={"session_id": args.session_id},
                files={"file": (path.name, fh, "application/octet-stream")},
                timeout=600,
            )
        if response.ok:
            payload = response.json()
            print(f"  ok — {payload.get('chunks_indexed')} chunks")
        else:
            failures += 1
            print(f"  failed ({response.status_code}): {response.text}", file=sys.stderr)

    print(f"\nDone. {len(files) - failures}/{len(files)} uploaded.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
