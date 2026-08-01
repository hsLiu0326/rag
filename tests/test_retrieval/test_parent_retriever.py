"""Tests for parent-document expansion after child retrieval."""

import pytest

from src.core.retrieval.parent_retriever import ParentRetriever
from src.models.document import SearchResult


class FakeRedis:
    def __init__(self, data: dict):
        self.data = data

    async def get_parents_batch(self, parent_ids: list[str]):
        return [self.data.get(pid) for pid in parent_ids]


def make_child(chunk_id: str, parent_id: str, score: float) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        parent_id=parent_id,
        text=f"child-{chunk_id}",
        score=score,
    )


@pytest.mark.asyncio
async def test_expand_dedupes_parents_and_keeps_max_score():
    redis = FakeRedis(
        {
            "p1": {
                "text": "父文档1",
                "title_path": "H1 > H2",
                "source": "a.docx",
                "pages": [1],
                "child_ids": ["c1", "c2"],
            },
            "p2": {
                "text": "父文档2",
                "title_path": "",
                "source": "b.pdf",
                "pages": [0],
                "child_ids": ["c3"],
            },
        }
    )
    retriever = ParentRetriever(redis)
    results = await retriever.expand_to_parents(
        [
            make_child("c1", "p1", 0.2),
            make_child("c2", "p1", 0.5),
            make_child("c3", "p2", 0.9),
        ],
        top_k=2,
    )
    assert len(results) == 2
    # Sorted by score descending
    assert results[0].parent_id == "p2"
    assert results[0].score == 0.9
    assert results[1].parent_id == "p1"
    assert results[1].score == 0.5  # max child score, not 0.2
    assert results[1].matched_children == ["c1", "c2"]
    assert results[0].text == "父文档2"


@pytest.mark.asyncio
async def test_expand_skips_missing_parents():
    redis = FakeRedis({"p1": {"text": "存在", "title_path": "", "source": "", "pages": [], "child_ids": []}})
    retriever = ParentRetriever(redis)
    results = await retriever.expand_to_parents(
        [make_child("c1", "p1", 0.5), make_child("c2", "missing", 0.8)],
        top_k=5,
    )
    assert len(results) == 1
    assert results[0].parent_id == "p1"


@pytest.mark.asyncio
async def test_expand_empty_input():
    retriever = ParentRetriever(FakeRedis({}))
    assert await retriever.expand_to_parents([]) == []
