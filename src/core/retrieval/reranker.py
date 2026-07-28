"""Optional Cross-Encoder reranker (stub — extend with BGE-Reranker)."""

from __future__ import annotations

from src.models.document import ParentContext


class Reranker:
    """Cross-encoder reranker for post-retrieval re-scoring.

    Currently a pass-through. Extend with FlagEmbedding BGE-Reranker-v2-m3
    for production use.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name

    async def rerank(
        self, query: str, parents: list[ParentContext]
    ) -> list[ParentContext]:
        """Re-score parent contexts against query."""
        # Stub: return unchanged for now.
        # Production: load BGE-Reranker-v2-m3 and compute relevance scores.
        return parents
