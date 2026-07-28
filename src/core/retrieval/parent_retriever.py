"""Parent document retriever — expand child hits to full parent context via Redis."""

from __future__ import annotations

from src.models.document import SearchResult, ParentContext
from src.storage.redis_store import RedisStore


class ParentRetriever:
    """After hybrid search returns child chunks, fetch their parent documents."""

    def __init__(self, redis: RedisStore) -> None:
        self.redis = redis

    async def expand_to_parents(
        self,
        child_results: list[SearchResult],
        top_k: int = 3,
    ) -> list[ParentContext]:
        """
        1. Collect unique parent_ids from child results
        2. Batch-fetch parents from Redis
        3. Merge scores (max child score per parent)
        4. Sort by score, return top_k
        """
        if not child_results:
            return []

        # Unique parent_ids ordered by first occurrence (preserve score order)
        seen: set[str] = set()
        parent_ids: list[str] = []
        parent_scores: dict[str, float] = {}

        for r in child_results:
            if not r.parent_id:
                continue
            if r.parent_id not in parent_scores:
                parent_scores[r.parent_id] = r.score
                seen.add(r.parent_id)
                parent_ids.append(r.parent_id)
            else:
                parent_scores[r.parent_id] = max(parent_scores[r.parent_id], r.score)

        # Batch fetch from Redis
        parents_data = await self.redis.get_parents_batch(parent_ids)

        # Assemble results
        results: list[ParentContext] = []
        for pid, data in zip(parent_ids, parents_data):
            if data is None:
                continue
            results.append(ParentContext(
                parent_id=pid,
                text=data.get("text", ""),
                title_path=data.get("title_path", ""),
                source=data.get("source", ""),
                pages=data.get("pages", []),
                score=parent_scores.get(pid, 0.0),
                matched_children=data.get("child_ids", []),
            ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
