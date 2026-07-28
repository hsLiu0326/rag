"""Hybrid retriever — dense (COSINE) + sparse (BM25) with RRF fusion."""

from __future__ import annotations

from src.models.document import SearchResult
from src.storage.milvus_store import MilvusStore
from src.core.preprocessor.embedder import EmbeddingService


class HybridRetriever:
    """Orchestrates dense + sparse retrieval with RRF fusion via Milvus."""

    def __init__(self, milvus: MilvusStore, embedder: EmbeddingService) -> None:
        self.milvus = milvus
        self.embedder = embedder

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 8,
        dense_limit: int = 20,
        sparse_limit: int = 20,
        rrf_k: int = 60,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Run hybrid search and return fused results."""
        # 1. Embed query
        query_embedding = await self.embedder.aembed_query(query)

        # 2. Build filter expression
        filter_expr = self.milvus.build_filter_expr(filters)

        # 3. Hybrid search (dense + sparse with RRF)
        hits = self.milvus.hybrid_search(
            query_embedding=query_embedding,
            query_text=query,
            top_k=top_k,
            dense_limit=dense_limit,
            sparse_limit=sparse_limit,
            rrf_k=rrf_k,
            filter_expr=filter_expr,
        )

        # 4. Map to SearchResult (compatible with pymilvus 2.x dict and 3.x Hit)
        results: list[SearchResult] = []
        for hit in hits:
            # pymilvus 3.x Hit: .get(key) works like dict.get; attributes via .score
            _get = lambda k, d="": (hit.get(k) or d)

            results.append(SearchResult(
                chunk_id=str(_get("chunk_id")),
                parent_id=str(_get("parent_id")),
                text=str(_get("text")),
                score=float(getattr(hit, "score", 0.0) or hit.get("score", 0.0) or 0.0),
                title_path=str(_get("title_path")),
                source=str(_get("source")),
                page=int(_get("page") or 0),
                chunk_index=int(_get("chunk_index") or 0),
                chunk_type=str(_get("chunk_type") or "child"),
            ))

        return results
