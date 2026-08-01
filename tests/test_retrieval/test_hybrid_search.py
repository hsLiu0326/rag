"""Tests for hybrid retrieval (dense + sparse, RRF via Milvus)."""

import pytest

from src.core.retrieval.hybrid_search import HybridRetriever


class FakeEmbedder:
    """Minimal embedder double returning a fixed vector."""

    async def aembed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class DictHit:
    """A pymilvus-2.x style result dict with a .score attribute."""

    def __init__(self, data: dict):
        self._data = data
        self.score = data.get("score", 0.0)

    def get(self, key, default=None):
        return self._data.get(key, default)


class FakeMilvus:
    def __init__(self, hits: list):
        self.hits = hits
        self.calls = []

    def build_filter_expr(self, filters):
        # Mirrors the real implementation for filter forwarding checks
        if not filters:
            return None
        parts = []
        for key, value in filters.items():
            if isinstance(value, str):
                parts.append(f'{key} == "{value}"')
            elif isinstance(value, int):
                parts.append(f"{key} == {value}")
        return " and ".join(parts) if parts else None

    def hybrid_search(self, **kwargs):
        self.calls.append(kwargs)
        # MilvusStore.hybrid_search() unwraps results[0] before returning,
        # so HybridRetriever expects a flat list of hits here.
        return self.hits


@pytest.mark.asyncio
async def test_retrieve_maps_hits_to_search_results():
    hits = [
        DictHit(
            {
                "chunk_id": "c1",
                "parent_id": "p1",
                "text": "文本",
                "title_path": "标题",
                "source": "a.pdf",
                "page": 1,
                "chunk_index": 2,
                "chunk_type": "child",
                "score": 0.42,
            }
        )
    ]
    retriever = HybridRetriever(FakeMilvus(hits), FakeEmbedder())
    results = await retriever.retrieve("查询", top_k=8)
    assert len(results) == 1
    r = results[0]
    assert r.chunk_id == "c1"
    assert r.parent_id == "p1"
    assert r.text == "文本"
    assert r.title_path == "标题"
    assert r.source == "a.pdf"
    assert r.page == 1
    assert r.chunk_index == 2
    assert r.chunk_type == "child"
    assert r.score == 0.42


@pytest.mark.asyncio
async def test_retrieve_passes_filters_to_milvus():
    milvus = FakeMilvus([])
    retriever = HybridRetriever(milvus, FakeEmbedder())
    await retriever.retrieve("查询", filters={"source": "a.pdf", "page": 2})
    call = milvus.calls[0]
    assert call["filter_expr"] == 'source == "a.pdf" and page == 2'
    assert call["top_k"] == 8
    assert call["query_text"] == "查询"


@pytest.mark.asyncio
async def test_retrieve_without_filters_uses_none():
    milvus = FakeMilvus([])
    retriever = HybridRetriever(milvus, FakeEmbedder())
    await retriever.retrieve("查询")
    assert milvus.calls[0]["filter_expr"] is None


@pytest.mark.asyncio
async def test_retrieve_empty_results():
    retriever = HybridRetriever(FakeMilvus([]), FakeEmbedder())
    assert await retriever.retrieve("查询") == []
