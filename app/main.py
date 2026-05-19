import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.routes import chat, upload

STATIC_DIR = Path(__file__).resolve().parent / "static"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio

    from app.services.embeddings import prewarm as prewarm_embeddings
    from app.services.llm_service import LLMService
    from app.services.rag_registry import get_rag_service

    logger.info("Pre-warming embeddings...")
    prewarm_embeddings()
    logger.info("Pre-warming Chroma index...")
    get_rag_service()
    logger.info("Pre-warming Ollama model...")
    try:
        await asyncio.wait_for(LLMService().prewarm(), timeout=120)
    except Exception:
        logger.exception("LLM prewarm failed (continuing)")
    logger.info("Startup complete.")
    yield


app = FastAPI(
    title=settings.app_name,
    description="Conversational AI chatbot with RAG capabilities",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api")
app.include_router(upload.router, prefix="/api")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok", "app": settings.app_name}
