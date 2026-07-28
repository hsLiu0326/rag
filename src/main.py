"""FastAPI application factory."""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Load .env immediately so HF_ENDPOINT and other vars are available
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: startup / shutdown."""
    settings = get_settings()
    import os
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    # ── Embedding (Local CPU) ──
    print(f"[startup] Loading embedding model: {settings.embedding_model_name}")
    from src.core.preprocessor.embedder import EmbeddingService
    embedder = EmbeddingService(
        model_name=settings.embedding_model_name,
        device=settings.embedding_device,
    )
    print(f"[startup] Embedding model ready (dim={embedder.dim}).")
    app.state.embedder = embedder

    # ── Connect Milvus ──
    from src.storage.milvus_store import MilvusStore
    milvus = MilvusStore(uri=settings.milvus_uri, token=settings.milvus_token or None)
    try:
        milvus.ensure_collection(dim=1024)
        app.state.milvus = milvus
        print(f"[startup] Milvus connected: {settings.milvus_uri}")
    except Exception as e:
        print(f"[startup] WARNING: Milvus not available ({e}). "
              f"Retrieval endpoints will return 503.")
        app.state.milvus = None

    # ── Connect Redis ──
    from src.storage.redis_store import RedisStore
    try:
        redis = RedisStore.from_url(settings.redis_url)
        await redis.ping()
        app.state.redis = redis
        print(f"[startup] Redis connected: {settings.redis_url}")
    except Exception as e:
        print(f"[startup] WARNING: Redis not available ({e}). "
              f"Parent retrieval will be unavailable.")
        app.state.redis = None

    print("[startup] Application ready.")

    yield

    # Shutdown
    print("[shutdown] Closing connections...")
    if app.state.redis:
        await app.state.redis.close()
    if app.state.milvus:
        app.state.milvus.close()
    print("[shutdown] Done.")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Enterprise RAG Platform",
        version="1.0.0",
        description="企业级智能文档检索与问答平台",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    from src.api.routes import health, documents, qa

    app.include_router(health.router, tags=["health"])
    app.include_router(documents.router, prefix="/api/v1", tags=["documents"])
    app.include_router(qa.router, prefix="/api/v1", tags=["qa"])

    return app


app = create_app()
