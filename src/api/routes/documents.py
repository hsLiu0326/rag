"""Document management routes — upload, status, delete."""

import uuid
import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, UploadFile, File, HTTPException, Request

from src.config import get_settings

router = APIRouter()

# Track ingestion status: doc_id -> "processing" | "completed" | "failed: ..."
_ingestion_status: dict[str, str] = {}


def _build_pipeline(request: Request):
    """Create a PreprocessPipeline using app.state connections."""
    from src.core.preprocessor.chunkers import HierarchicalChunker
    from src.core.preprocessor.pipeline import PreprocessPipeline
    s = get_settings()
    return PreprocessPipeline(
        chunker=HierarchicalChunker(parent_size=s.parent_chunk_size, child_size=s.child_chunk_size, overlap=s.chunk_overlap),
        embedder=request.app.state.embedder,
        milvus=request.app.state.milvus,
        redis=request.app.state.redis,
    )


@router.post("/documents/upload", response_model=None)
async def upload_document(
    file: UploadFile = File(...),
    request: Request = None,
) -> Any:
    """Upload a document. Returns immediately, processes in background."""
    suffix = Path(file.filename or "").suffix.lower()
    allowed = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls", ".md", ".txt", ".text"}
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported type: {suffix}")

    settings = get_settings()
    doc_id = uuid.uuid4().hex
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"{doc_id}{suffix}"
    content = await file.read()
    file_path.write_bytes(content)

    # Fix garbled Chinese filenames (Windows sends GBK, HTTP needs UTF-8)
    raw_name = file.filename or "unknown"
    try:
        original_name = raw_name.encode("latin-1").decode("gbk")
    except (UnicodeDecodeError, UnicodeEncodeError):
        original_name = raw_name

    if request and hasattr(request.app.state, "milvus") and request.app.state.milvus:
        pipeline = _build_pipeline(request)
        try:
            await pipeline.ingest(str(file_path), original_name=original_name)
            _ingestion_status[doc_id] = "completed"
            print(f"[upload] {original_name} -> completed")
            ingestion_status = "completed"
        except Exception as e:
            import traceback
            _ingestion_status[doc_id] = f"failed: {e}"
            print(f"[upload] {original_name} -> FAILED: {e}")
            traceback.print_exc()
            ingestion_status = f"failed: {e}"
    else:
        ingestion_status = "skipped"

    return {
        "doc_id": doc_id,
        "filename": original_name,
        "status": ingestion_status,
        "message": f"Document uploaded. Ingestion {ingestion_status}.",
    }


@router.get("/documents/{doc_id}", response_model=None)
async def get_document_status(doc_id: str) -> Any:
    """Get document processing status."""
    from src.core.preprocessor.pipeline import get_progress
    progress = get_progress(doc_id)
    status = _ingestion_status.get(doc_id, "unknown")
    return {
        "doc_id": doc_id,
        "status": status,
        "progress": progress,
    }


@router.delete("/documents/{doc_id}", response_model=None)
async def delete_document(doc_id: str) -> Any:
    """Delete a document and all its chunks."""
    _ingestion_status.pop(doc_id, None)
    from src.core.preprocessor.pipeline import clear_progress
    clear_progress(doc_id)
    return {"doc_id": doc_id, "status": "deleted"}


@router.delete("/database/clear", response_model=None)
async def clear_database(request: Request) -> Any:
    """Clear all data: Milvus collection + Redis + upload files."""
    import shutil
    from src.core.preprocessor.pipeline import clear_progress
    from src.storage.milvus_store import MilvusStore

    # Clear Milvus
    if hasattr(request.app.state, "milvus") and request.app.state.milvus:
        try:
            milvus: MilvusStore = request.app.state.milvus
            milvus.client.drop_collection(milvus.COLLECTION_NAME)
            milvus.ensure_collection(dim=1024)
        except Exception as e:
            print(f"[clear] Milvus error: {e}")

    # Clear Redis
    if hasattr(request.app.state, "redis") and request.app.state.redis:
        try:
            await request.app.state.redis._client.flushdb()
        except Exception as e:
            print(f"[clear] Redis error: {e}")

    # Clear upload files
    try:
        settings = get_settings()
        upload_dir = Path(settings.upload_dir)
        if upload_dir.exists():
            shutil.rmtree(upload_dir)
            upload_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[clear] File error: {e}")

    # Clear tracking
    _ingestion_status.clear()
    clear_progress()

    return {"status": "cleared", "message": "All data cleared: Milvus, Redis, upload files."}
