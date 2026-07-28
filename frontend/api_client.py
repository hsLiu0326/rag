"""Backend API client for the Streamlit frontend."""

from __future__ import annotations

import json
from typing import Any, Generator

import requests

# Default backend URL (configurable)
API_BASE = "http://127.0.0.1:8001"


def upload_document(file_bytes: bytes, filename: str) -> dict[str, Any]:
    """Upload a document to the backend."""
    resp = requests.post(
        f"{API_BASE}/api/v1/documents/upload",
        files={"file": (filename, file_bytes)},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def stream_qa(
    question: str,
    history: list[dict[str, str]] | None = None,
    filters: dict[str, Any] | None = None,
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> Generator[dict[str, Any], None, None]:
    """
    Stream Q&A response via SSE.
    Yields dicts: {"type": "status"|"token"|"sources"|"done", "data": ...}
    """
    body = {
        "question": question,
        "history": history or [],
        "filters": filters or {},
        "temperature": temperature,
        "max_tokens": max_tokens,
        "enable_rerank": False,
    }

    resp = requests.post(
        f"{API_BASE}/api/v1/qa/ask",
        json=body,
        stream=True,
        timeout=120,
    )
    resp.raise_for_status()

    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        # SSE format: "event: <name>\ndata: <payload>"
        if line.startswith("event: "):
            event_type = line[7:].strip()
        elif line.startswith("data: "):
            data_str = line[6:].strip()
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                data = data_str  # token is raw string, not JSON
            yield {"type": event_type, "data": data}
        # else: continuation or comment line, skip


def get_document_status(doc_id: str) -> dict[str, Any]:
    """Check document processing status."""
    resp = requests.get(f"{API_BASE}/api/v1/documents/{doc_id}", timeout=5)
    resp.raise_for_status()
    return resp.json()


def clear_database() -> dict[str, Any]:
    """Clear all data in Milvus, Redis, and uploads."""
    resp = requests.delete(f"{API_BASE}/api/v1/database/clear", timeout=30)
    resp.raise_for_status()
    return resp.json()


def health_check() -> dict[str, Any]:
    """Check backend health."""
    resp = requests.get(f"{API_BASE}/health", timeout=5)
    resp.raise_for_status()
    return resp.json()
