"""Async Redis client — parent document store and embedding cache."""

from __future__ import annotations

import json
from typing import Any

import redis.asyncio as aioredis


class RedisStore:
    """Async Redis wrapper for parent document store + query cache."""

    PARENT_KEY_PREFIX: str = "parent:"
    EMBEDDING_KEY_PREFIX: str = "emb:"
    DEFAULT_TTL: int = 0  # 0 = no expiry (permanent)

    def __init__(self, client: aioredis.Redis) -> None:
        self._client = client

    @classmethod
    def from_url(cls, url: str, **kwargs: Any) -> "RedisStore":
        client = aioredis.from_url(url, decode_responses=False, **kwargs)
        return cls(client)

    async def close(self) -> None:
        await self._client.aclose()

    async def ping(self) -> bool:
        return await self._client.ping()

    # ── Parent Document Storage ────────────────────────────────────

    async def put_parent(self, parent_id: str, data: dict[str, Any], ttl: int | None = None) -> None:
        """Store parent document JSON with optional TTL. 0 = no expiry."""
        key = f"{self.PARENT_KEY_PREFIX}{parent_id}"
        value = json.dumps(data, ensure_ascii=False).encode("utf-8")
        ex = ttl if ttl is not None else self.DEFAULT_TTL
        if ex and ex > 0:
            await self._client.set(key, value, ex=ex)
        else:
            await self._client.set(key, value)

    async def get_parent(self, parent_id: str) -> dict[str, Any] | None:
        """Fetch a single parent document."""
        key = f"{self.PARENT_KEY_PREFIX}{parent_id}"
        data = await self._client.get(key)
        if data is None:
            return None
        return json.loads(data if isinstance(data, str) else data.decode("utf-8"))

    async def get_parents_batch(self, parent_ids: list[str]) -> list[dict[str, Any] | None]:
        """Batch fetch parent documents via pipeline."""
        if not parent_ids:
            return []
        keys = [f"{self.PARENT_KEY_PREFIX}{pid}" for pid in parent_ids]
        pipe = self._client.pipeline()
        for key in keys:
            pipe.get(key)
        results = await pipe.execute()
        parsed: list[dict[str, Any] | None] = []
        for r in results:
            if r is None:
                parsed.append(None)
            else:
                parsed.append(json.loads(r if isinstance(r, str) else r.decode("utf-8")))
        return parsed

    async def delete_parents(self, parent_ids: list[str]) -> int:
        """Delete parent documents. Returns count deleted."""
        if not parent_ids:
            return 0
        keys = [f"{self.PARENT_KEY_PREFIX}{pid}" for pid in parent_ids]
        return await self._client.delete(*keys)

    # ── Embedding Cache ────────────────────────────────────────────

    async def cache_embedding(self, text_hash: str, embedding: list[float], ttl: int = 3600) -> None:
        """Cache a query embedding by hash."""
        key = f"{self.EMBEDDING_KEY_PREFIX}{text_hash}"
        value = json.dumps(embedding).encode("utf-8")
        await self._client.set(key, value, ex=ttl)

    async def get_cached_embedding(self, text_hash: str) -> list[float] | None:
        """Retrieve cached embedding if exists."""
        key = f"{self.EMBEDDING_KEY_PREFIX}{text_hash}"
        data = await self._client.get(key)
        if data is None:
            return None
        return json.loads(data if isinstance(data, str) else data.decode("utf-8"))
