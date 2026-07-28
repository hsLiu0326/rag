"""Document preprocessing pipeline orchestrator.

Simplified flow (MinerU):
  load (MinerU PDF / python-docx / MD / TXT) → clean → chunk → embed → store

MinerU handles layout detection, reading order, tables, formulas,
and OCR internally — no separate OCR step needed.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from src.models.document import ChunkPair
from src.storage.milvus_store import MilvusStore
from src.storage.redis_store import RedisStore
from src.core.preprocessor.loaders import load_document
from src.core.preprocessor.chunkers import HierarchicalChunker
from src.core.preprocessor.embedder import EmbeddingService
from src.core.preprocessor.cleaner import TextCleaner

# Module-level progress tracking: doc_id -> {"stage": str, "pct": int, "message": str}
_progress: dict[str, dict] = {}


def get_progress(doc_id: str) -> dict:
    """Get processing progress for a document."""
    return _progress.get(doc_id, {"stage": "unknown", "pct": 0, "message": "No record"})


def clear_progress(doc_id: str | None = None) -> None:
    """Clear progress records. If doc_id is None, clears all."""
    if doc_id:
        _progress.pop(doc_id, None)
    else:
        _progress.clear()


class PreprocessPipeline:
    """Orchestrates: load → chunk → embed → store(children+parents)"""

    def __init__(
        self,
        chunker: HierarchicalChunker,
        embedder: EmbeddingService,
        milvus: MilvusStore,
        redis: RedisStore,
    ) -> None:
        self.chunker = chunker
        self.embedder = embedder
        self.milvus = milvus
        self.redis = redis

    async def ingest(self, file_path: str | Path, original_name: str = "") -> str:
        """
        Full ingestion pipeline. Returns doc_id.

        Steps:
        1. Load document (PDF→MinerU, DOCX→python-docx, etc.)
        2. Hierarchical chunk → ChunkPairs
        3. Embed all child chunks
        4. Insert children into Milvus
        5. Store parents in Redis
        """
        file_path = Path(file_path)
        doc_id = uuid.uuid4().hex
        display_name = original_name or file_path.name
        suffix = file_path.suffix.lower()
        print(f"[pipeline] Ingesting doc_id={doc_id} type={suffix} from {display_name}")

        _progress[doc_id] = {"stage": "loading", "pct": 5, "message": f"正在加载 {display_name}..."}

        # 1. Load — PDFLoader auto-decides fast(pypdf) vs slow(MinerU GPU)
        docs = load_document(file_path)
        # Override source with original filename
        for doc in docs:
            doc.metadata["source"] = display_name

        # 1.5 Clean
        _progress[doc_id] = {"stage": "cleaning", "pct": 30, "message": "正在清洗文本..."}
        cleaner = TextCleaner()
        for doc in docs:
            doc.page_content = cleaner.clean(doc.page_content)

        # 2. Chunk
        _progress[doc_id] = {"stage": "chunking", "pct": 40, "message": "正在分层切片..."}
        all_pairs: list[ChunkPair] = []
        for doc in docs:
            if not doc.page_content.strip():
                continue

            base_meta = {
                "source": display_name,
                "doc_id": doc_id,
                "format": doc.metadata.get("format", "unknown"),
                "page": doc.metadata.get("page", 0),
            }
            pairs = self.chunker.split(doc.page_content, base_meta)
            all_pairs.extend(pairs)

        if not all_pairs:
            print(f"[pipeline] WARNING: No content extracted from {file_path.name}")
            _progress[doc_id] = {"stage": "failed", "pct": 0, "message": "未能提取到文本内容"}
            return doc_id

        # 3-5. Embed + Store
        _progress[doc_id] = {"stage": "embedding", "pct": 60, "message": f"正在向量化 {sum(len(p.children) for p in all_pairs)} 个文本块..."}
        await self._store_pairs(all_pairs, doc_id)

        parent_count = len(all_pairs)
        child_count = sum(len(p.children) for p in all_pairs)
        print(f"[pipeline] doc_id={doc_id} done: {parent_count} parents, {child_count} children")
        return doc_id

    async def _store_pairs(self, pairs: list[ChunkPair], doc_id: str) -> None:
        """Embed children → insert into Milvus → store parents in Redis."""

        # Collect all child texts
        all_children: list[tuple[str, str, dict]] = []
        for pair in pairs:
            for child in pair.children:
                all_children.append((child.chunk_id, child.text, child.metadata))

        if not all_children:
            return

        # Batch embed children (async, in thread pool)
        child_texts = [c[1] for c in all_children]
        embeddings = await self.embedder.aembed_documents(child_texts)

        # Build Milvus insert rows
        milvus_rows: list[dict] = []
        for (chunk_id, text, meta), emb in zip(all_children, embeddings):
            milvus_rows.append({
                "chunk_id": chunk_id,
                "parent_id": meta.get("parent_id", ""),
                "doc_id": doc_id,
                "text": text[:4096],
                "dense_vector": emb,
                "title_path": meta.get("title_path", ""),
                "source": meta.get("source", ""),
                "chunk_type": "child",
                "page": meta.get("page", 0) if isinstance(meta.get("page"), int) else 0,
                "chunk_index": meta.get("chunk_index", 0),
            })

        self.milvus.insert_batch(milvus_rows)

        # Store parents in Redis
        for pair in pairs:
            parent_data = {
                "text": pair.parent_text,
                "title_path": pair.parent_metadata.get("title_path", ""),
                "source": pair.parent_metadata.get("source", ""),
                "doc_id": doc_id,
                "pages": [pair.parent_metadata.get("page", 0)],
                "child_ids": [c.chunk_id for c in pair.children],
            }
            await self.redis.put_parent(pair.parent_id, parent_data)
