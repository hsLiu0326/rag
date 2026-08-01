"""Tests for the reranker (currently a pass-through stub)."""

import pytest

from src.core.retrieval.reranker import Reranker
from src.models.document import ParentContext


@pytest.mark.asyncio
async def test_reranker_passes_through_unchanged():
    parents = [
        ParentContext(parent_id="p1", text="内容", score=0.4),
        ParentContext(parent_id="p2", text="内容2", score=0.7),
    ]
    reranked = await Reranker().rerank("问题", parents)
    assert reranked == parents
