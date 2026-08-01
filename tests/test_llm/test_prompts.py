"""Tests for RAG prompt assembly."""

from src.core.llm.prompts import SYSTEM_PROMPT, build_rag_prompt
from src.models.document import ParentContext


def test_system_prompt_grounds_answer_in_documents():
    assert "基于提供的文档上下文" in SYSTEM_PROMPT


def test_build_rag_prompt_includes_context_metadata():
    parents = [
        ParentContext(
            parent_id="p1",
            text="文档内容正文",
            title_path="第一章 > 1.1",
            source="manual.docx",
            pages=[1, 2],
            score=0.5,
        )
    ]
    system, user = build_rag_prompt("问题是什么？", parents)
    assert "第一章 > 1.1" in user
    assert "manual.docx" in user
    assert "第1,2页" in user
    assert "文档内容正文" in user
    assert "问题是什么？" in user
    assert "参考文档" in system or "文档上下文" in system


def test_empty_parents_placeholder():
    _, user = build_rag_prompt("问题", [])
    assert "（暂无上传文档）" in user


def test_history_is_formatted_and_truncated_to_last_ten():
    history = []
    for i in range(12):
        history.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"消息{i}"})

    _, user = build_rag_prompt("新问题", [], history=history)
    # First two messages should be dropped, last ten kept
    assert "消息0" not in user
    assert "消息2" in user
    assert "消息11" in user
    assert "用户：消息2" in user
    assert "助手：消息3" in user


def test_no_history_placeholder():
    _, user = build_rag_prompt("问题", [], history=[])
    assert "（无历史对话）" in user
