"""Local Embedding service using Qwen3-Embedding-0.6B (CPU).

Runs on CPU to leave GPU free for MinerU (Ollama/PyTorch CUDA context conflict).
For single-query embeddings (~50-100ms), CPU inference is fast enough.
"""

from __future__ import annotations

import asyncio
import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Local Qwen3-Embedding-0.6B via sentence-transformers (CPU).

    Output dimension: 1024 (Qwen3-0.6B hidden_size).
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", device: str = "cpu") -> None:
        self._model_name = model_name
        self._device = device
        self._model: SentenceTransformer | None = None
        self._pool = ThreadPoolExecutor(max_workers=1)

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(
                self._model_name,
                device=self._device,
                trust_remote_code=True,
            )
        return self._model

    @property
    def dim(self) -> int:
        try:
            return self.model.get_embedding_dimension()
        except AttributeError:
            return self.model.get_sentence_embedding_dimension()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Batch embed documents."""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    async def aembed_query(self, text: str) -> list[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._pool, self.embed_query, text)

    async def aembed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._pool, lambda: self.embed_documents(texts, batch_size),
        )

    def hash_text(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()
