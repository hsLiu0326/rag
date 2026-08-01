"""Tests for the Redis parent-document store using an in-memory fake client."""

import pytest

from src.storage.redis_store import RedisStore


class FakePipeline:
    def __init__(self, store):
        self.store = store
        self.keys = []

    def get(self, key):
        self.keys.append(key)
        return self

    async def execute(self):
        return [self.store.data.get(k) for k in self.keys]


class FakeRedis:
    def __init__(self):
        self.data = {}

    async def set(self, key, value, ex=None):
        self.data[key] = value

    async def get(self, key):
        return self.data.get(key)

    async def delete(self, *keys):
        count = 0
        for key in keys:
            if key in self.data:
                del self.data[key]
                count += 1
        return count

    def pipeline(self):
        return FakePipeline(self)


@pytest.fixture
def store():
    return RedisStore(FakeRedis())


@pytest.mark.asyncio
async def test_put_and_get_parent_roundtrip(store):
    data = {
        "text": "父文档内容（中文）",
        "title_path": "第一章",
        "source": "a.docx",
        "doc_id": "doc-1",
        "pages": [1, 2],
        "child_ids": ["c1", "c2"],
    }
    await store.put_parent("p1", data)
    loaded = await store.get_parent("p1")
    assert loaded == data


@pytest.mark.asyncio
async def test_get_missing_parent_returns_none(store):
    assert await store.get_parent("nope") is None


@pytest.mark.asyncio
async def test_get_parents_batch(store):
    await store.put_parent("p1", {"text": "一"})
    await store.put_parent("p2", {"text": "二"})
    result = await store.get_parents_batch(["p1", "missing", "p2"])
    assert result[0] == {"text": "一"}
    assert result[1] is None
    assert result[2] == {"text": "二"}


@pytest.mark.asyncio
async def test_delete_parents_returns_count(store):
    await store.put_parent("p1", {"text": "一"})
    await store.put_parent("p2", {"text": "二"})
    assert await store.delete_parents(["p1", "p2", "p3"]) == 2
    assert await store.get_parent("p1") is None


@pytest.mark.asyncio
async def test_delete_empty_list(store):
    assert await store.delete_parents([]) == 0
