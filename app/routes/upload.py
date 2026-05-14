from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import get_settings
from app.models import DocumentUploadResponse
from app.services.rag_registry import get_rag_service
from app.utils.file_loader import UnsupportedDocumentError

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Upload"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    session_id: str = Form(default="default"),
    file: UploadFile = File(...),
):
    settings = get_settings()
    upload_root = Path(settings.upload_dir)
    chroma_root = Path(settings.chroma_db_dir)
    upload_root.mkdir(parents=True, exist_ok=True)
    chroma_root.mkdir(parents=True, exist_ok=True)

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".txt"}:
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are accepted.")

    safe_original = Path(file.filename).name
    temp_path = upload_root / f"{uuid.uuid4().hex}_{safe_original}"

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")

    temp_path.write_bytes(raw)

    try:
        chunk_count = await asyncio.to_thread(
            get_rag_service().index_file,
            session_id,
            temp_path,
            safe_original,
        )
    except UnsupportedDocumentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)

    logger.info("Indexed %s chunks session=%s file=%s", chunk_count, session_id, safe_original)
    return DocumentUploadResponse(
        session_id=session_id,
        filename=safe_original,
        chunks_indexed=chunk_count,
    )
