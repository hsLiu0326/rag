"""RAG prompt templates for the enterprise Q&A platform."""

from __future__ import annotations

from src.models.document import ParentContext

SYSTEM_PROMPT = """你是一个专业的企业文档问答助手，底层使用DeepSeek大模型。

你的回答必须基于提供的文档上下文。请遵循以下规则：
1. 如果文档上下文中包含答案，请准确引用并给出详细解释。
2. 如果文档上下文中没有相关信息，请明确说"文档中未找到相关信息"，不要编造答案。
3. 回答时请引用具体的来源章节和页码（如果可用）。
4. 使用中文回答，保持专业、简洁、准确。
5. 如果涉及技术术语，请给出必要的解释。
6. 关于你自己的身份、能力、模型版本等简单问题可以直接回答，不需要依赖文档。
7. 如果用户的问题是对上一轮的追问，请结合对话历史和文档上下文一起理解。"""

USER_PROMPT_TEMPLATE = """## 对话历史

{history}

## 参考文档

{context}

## 用户问题

{question}

请基于上述参考文档回答问题。如果参考文档为空或无相关内容，对于简单问题可以直接回答。"""


def build_rag_prompt(
    question: str,
    parents: list[ParentContext],
    history: list[dict] | None = None,
) -> tuple[str, str]:
    """Build system + user prompts for RAG Q&A.

    Args:
        question: Current user question.
        parents: Retrieved parent document contexts.
        history: Previous chat messages [{"role": "user/assistant", "content": "..."}].

    Returns (system_prompt, user_prompt).
    """
    # Assemble context from parent documents
    context_parts: list[str] = []
    for i, parent in enumerate(parents, 1):
        header = f"【文档{i}】"
        if parent.title_path:
            header += f" 章节：{parent.title_path}"
        if parent.source:
            header += f" 来源：{parent.source}"
        if parent.pages:
            header += f" 第{','.join(str(p) for p in parent.pages)}页"
        context_parts.append(f"{header}\n{parent.text}")

    context_text = "\n\n---\n\n".join(context_parts) if context_parts else "（暂无上传文档）"

    # Assemble chat history (last 10 turns to stay within context window)
    history_text = "（无历史对话）"
    if history:
        recent = history[-10:]  # Keep last 5 Q&A pairs
        lines = []
        for msg in recent:
            role_label = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role_label}：{msg['content']}")
        history_text = "\n".join(lines)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        history=history_text,
        context=context_text,
        question=question,
    )

    return SYSTEM_PROMPT, user_prompt
