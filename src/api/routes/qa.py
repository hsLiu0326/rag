"""Q&A routes — SSE streaming and sync endpoints."""

from __future__ import annotations

import json
import time

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from src.models.schemas import QuestionRequest
from src.core.retrieval.hybrid_search import HybridRetriever
from src.core.retrieval.parent_retriever import ParentRetriever
from src.core.retrieval.reranker import Reranker
from src.core.llm.qwen_client import QwenLLMClient
from src.core.llm.prompts import build_rag_prompt

router = APIRouter()


@router.post("/qa/ask")
async def ask_question_stream(
    req: QuestionRequest,
    request: Request,
):
    """SSE streaming Q&A endpoint.

    Events:
      - status: {"phase": "retrieving" | "generating"}
      - token: "文本片段"
      - sources: [{"title": ..., "source": ..., "pages": [...]}]
      - done: {"token_count": N}
    """
    # Dependency-like: extract from app state
    milvus = request.app.state.milvus
    redis = request.app.state.redis
    embedder = request.app.state.embedder

    services_ready = milvus is not None and redis is not None

    if services_ready:
        retriever = HybridRetriever(milvus, embedder)
        parent_retriever = ParentRetriever(redis)
    reranker = Reranker()
    llm = QwenLLMClient()

    async def event_generator():
        t0 = time.perf_counter()

        if not services_ready:
            yield ServerSentEvent(
                event="status",
                data=json.dumps({"phase": "error", "error": "Milvus/Redis not available. Please start Docker services first."}, ensure_ascii=False),
            )
            yield ServerSentEvent(event="done", data=json.dumps({"token_count": 0}))
            return

        # Phase 1: Retrieval
        yield ServerSentEvent(event="status", data=json.dumps({"phase": "retrieving"}, ensure_ascii=False))

        child_results = await retriever.retrieve(
            query=req.question,
            top_k=8,
            filters=req.filters,
        )
        parents = await parent_retriever.expand_to_parents(child_results, top_k=3)

        if req.enable_rerank:
            parents = await reranker.rerank(req.question, parents)

        t1 = time.perf_counter()
        retrieval_ms = (t1 - t0) * 1000

        # Phase 2: Generation
        yield ServerSentEvent(
            event="status",
            data=json.dumps({
                "phase": "generating",
                "retrieval_ms": round(retrieval_ms, 1),
            }, ensure_ascii=False),
        )

        history = [{"role": h.role, "content": h.content} for h in (req.history or [])]
        system_prompt, user_prompt = build_rag_prompt(req.question, parents, history=history)

        token_count = 0
        first_token = True
        async for token in llm.stream_chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        ):
            if first_token:
                first_token = False
                ttft_ms = (time.perf_counter() - t0) * 1000
                print(f"[qa] first_token_latency={ttft_ms:.0f}ms, "
                      f"retrieval={retrieval_ms:.0f}ms")
            yield ServerSentEvent(event="token", data=token)
            token_count += 1

        # Phase 3: Sources
        sources = [
            {
                "title": p.title_path,
                "source": p.source,
                "pages": p.pages,
                "score": round(p.score, 4),
            }
            for p in parents
        ]
        yield ServerSentEvent(
            event="sources",
            data=json.dumps(sources, ensure_ascii=False),
        )

        # Phase 4: Done
        total_ms = (time.perf_counter() - t0) * 1000
        yield ServerSentEvent(
            event="done",
            data=json.dumps({
                "token_count": token_count,
                "total_ms": round(total_ms, 1),
            }, ensure_ascii=False),
        )

    return EventSourceResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
