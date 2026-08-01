"""API-level tests running against the FastAPI app with services stubbed out."""

from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from src.main import app


@pytest.fixture
async def client(tmp_path, monkeypatch):
    # Stub out backing services so tests need no Docker/Milvus/Redis
    app.state.embedder = None
    app.state.milvus = None
    app.state.redis = None
    fake_settings = SimpleNamespace(upload_dir=str(tmp_path / "uploads"))
    monkeypatch.setattr("src.api.routes.documents.get_settings", lambda: fake_settings)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "version": "1.0.0"}


@pytest.mark.asyncio
async def test_upload_unsupported_type(client):
    resp = await client.post(
        "/api/v1/documents/upload",
        files={"file": ("evil.exe", b"MZ")},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_upload_skipped_without_backend_services(client, tmp_path):
    resp = await client.post(
        "/api/v1/documents/upload",
        files={"file": ("test.txt", "你好世界".encode("utf-8"))},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["doc_id"]
    assert body["status"] == "skipped"
    assert body["filename"] == "test.txt"
    saved = list((tmp_path / "uploads").iterdir())
    assert len(saved) == 1


@pytest.mark.asyncio
async def test_document_status_and_delete(client):
    resp = await client.get("/api/v1/documents/abc123")
    assert resp.status_code == 200
    assert resp.json()["status"] == "unknown"

    resp = await client.delete("/api/v1/documents/abc123")
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"


@pytest.mark.asyncio
async def test_qa_returns_error_event_when_services_unavailable(client):
    async with client.stream(
        "POST",
        "/api/v1/qa/ask",
        json={"question": "测试问题", "max_tokens": 16},
    ) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        lines = [line async for line in resp.aiter_lines()]
    text = "\n".join(lines)
    assert "phase" in text
    assert "error" in text
    assert "done" in text
