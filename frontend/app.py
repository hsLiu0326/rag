"""Streamlit frontend — Enterprise RAG Platform.

Run: streamlit run frontend/app.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import streamlit as st

# Add project root to path so we can share schemas if needed
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frontend.api_client import upload_document, stream_qa, health_check, API_BASE

# ── Page Config ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="企业智能文档问答平台",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Force sidebar to never auto-collapse
st.markdown("""
<style>
    [data-testid="stSidebar"] { min-width: 300px !important; }
    button[data-testid="baseButton-headerNoPadding"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Custom CSS ───────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Chat message styling */
    .stChatMessage {
        border-radius: 12px;
        padding: 8px 16px;
    }
    /* Source citation card */
    .source-card {
        background: #f0f2f6;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 4px 0;
        font-size: 0.88em;
        border-left: 3px solid #4a90d9;
    }
    .source-card .title { font-weight: 600; color: #1a3a5c; }
    .source-card .meta { color: #666; font-size: 0.85em; }
    /* Sidebar upload area */
    .upload-area {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-bottom: 16px;
    }
    /* Status badges */
    .badge-ok {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #4caf50;
        margin-right: 6px;
    }
    .badge-off {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #f44336;
        margin-right: 6px;
    }
    /* Footer */
    .footer {
        text-align: center;
        color: #999;
        font-size: 0.8em;
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State ────────────────────────────────────────────────────

import json as _json
from pathlib import Path as _Path
_STATE_FILE = _Path(__file__).resolve().parent / ".session_state.json"

def init_state():
    defaults = {
        "messages": [],
        "backend_ok": False,
        "documents": [],
        "last_uploaded_name": "",
    }
    # Load persisted state
    if _STATE_FILE.exists():
        try:
            saved = _json.loads(_STATE_FILE.read_text(encoding="utf-8"))
            for k, v in saved.items():
                if k in defaults:
                    defaults[k] = v
        except Exception:
            pass
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def save_state():
    data = {
        "messages": st.session_state.get("messages", []),
        "documents": st.session_state.get("documents", []),
        "last_uploaded_name": st.session_state.get("last_uploaded_name", ""),
    }
    _STATE_FILE.write_text(_json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

init_state()

# ── Sidebar ──────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/color/96/open-book.png", width=48) if False else None
    st.title("📚 文档管理")

    # ── Backend Status ──
    st.subheader("🔌 服务状态")
    try:
        h = health_check()
        st.session_state.backend_ok = True
        st.success(f"✅ 后端已连接 (v{h.get('version', '?')})")
    except Exception:
        st.session_state.backend_ok = False
        st.error("❌ 后端未连接")
        st.caption(f"请确保后端运行在 {API_BASE}")
        if st.button("🔄 重新检查连接", use_container_width=True):
            st.rerun()

    st.divider()

    # ── Document Upload ──
    st.subheader("📤 上传文档")

    if "last_uploaded_name" not in st.session_state:
        st.session_state.last_uploaded_name = ""

    uploaded_file = st.file_uploader(
        "支持 PDF / DOCX / Markdown / TXT",
        type=["pdf", "docx", "doc", "pptx", "ppt", "xlsx", "xls", "md", "txt"],
        accept_multiple_files=False,
        disabled=not st.session_state.backend_ok,
        label_visibility="collapsed",
    )

    if uploaded_file and uploaded_file.name != st.session_state.last_uploaded_name:
        st.session_state.last_uploaded_name = uploaded_file.name
        try:
            result = upload_document(uploaded_file.getvalue(), uploaded_file.name)
            doc_id = result.get("doc_id", "")
            status = result.get("status", "")
            if status == "processing":
                st.success(f"✅ {uploaded_file.name} 上传成功，后台处理中...")
            else:
                st.info(f"📄 {uploaded_file.name}: {status}")
            st.caption(f"文档ID: {doc_id[:12]}...")
            st.session_state.documents.append({
                "id": doc_id,
                "name": uploaded_file.name,
                "status": status,
            })
            save_state()
        except Exception as e:
            st.error(f"上传失败: {e}")
            st.session_state.last_uploaded_name = ""

    # Show processing status with progress bar
    from frontend.api_client import get_document_status
    import time as _time
    has_processing = False
    for doc in st.session_state.documents:
        if doc["status"] == "processing":
            has_processing = True
            try:
                s = get_document_status(doc["id"])
                prog = s.get("progress", {})
                pct = prog.get("pct", 0)
                msg = prog.get("message", "处理中...")
                st.caption(f"📄 {doc['name']}")
                st.progress(pct / 100, text=f"{msg} ({pct}%)")
                if s["status"] == "completed":
                    doc["status"] = "completed"
                    st.success(f"✅ {doc['name']} 处理完成")
                    _time.sleep(0.5)
                    st.rerun()
                elif s["status"].startswith("failed"):
                    doc["status"] = s["status"]
                    st.error(f"❌ {doc['name']} 处理失败")
            except Exception:
                pass

    # Auto-refresh while documents are processing
    if has_processing and st.session_state.get("_auto_refresh", True):
        st.session_state._auto_refresh = False
        _time.sleep(3)
        st.session_state._auto_refresh = True
        st.rerun()

    st.divider()

    # ── Settings ──
    st.subheader("⚙️ 问答设置")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.3, 0.1, key="temp")
    max_tokens = st.slider("最大输出长度", 256, 4096, 2048, 256, key="max_tok")

    st.divider()

    # ── Upload History ──
    if st.session_state.documents:
        st.subheader("📋 已上传文档")
        for doc in st.session_state.documents:
            icon = "✅" if doc["status"] == "completed" else ("⏳" if doc["status"] == "processing" else "📄")
            st.caption(f"{icon} {doc['name']}")
    else:
        st.caption("尚未上传任何文档")

    # ── Danger Zone ──
    st.divider()
    with st.expander("⚠️ 危险操作"):
        if st.button("🗑️ 清空所有数据", use_container_width=True, type="secondary"):
            from frontend.api_client import clear_database
            try:
                clear_database()
                st.session_state.documents = []
                st.session_state.messages = []
                save_state()
                st.success("数据库已清空")
                _time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"清空失败: {e}")

    # ── Footer ──
    st.markdown(
        '<div class="footer">Enterprise RAG Platform v1.0<br>'
        'Powered by Qwen2.5-7B + Milvus</div>',
        unsafe_allow_html=True,
    )


# ── Main Chat Area ───────────────────────────────────────────────────

st.title("💬 企业智能文档问答")

# Welcome message
if not st.session_state.messages:
    with st.chat_message("assistant", avatar="📚"):
        st.markdown("""
        ### 欢迎使用企业智能文档问答平台 👋

        请先上传您的文档（左侧边栏），然后开始提问。

        **支持的功能：**
        - 🔍 多格式文档智能检索（PDF、DOCX、Markdown、TXT）
        - 🧠 基于文档上下文的精准问答
        - 📎 回答附带引用来源
        - ⚡ 流式逐字输出，快速响应
        """)

# Display chat history
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "📚"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        # Show sources for assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📎 参考来源"):
                for src in msg["sources"]:
                    title = src.get("title", "未知章节")
                    source = src.get("source", "")
                    pages = src.get("pages", [])
                    score = src.get("score", 0)
                    st.markdown(
                        f'<div class="source-card">'
                        f'<div class="title">{title}</div>'
                        f'<div class="meta">📄 {source} | 📖 第{",".join(str(p) for p in pages) if pages else "?"}页 '
                        f'| 🎯 相关度: {1 - score:.1%}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

# ── Chat Input ───────────────────────────────────────────────────────

if prompt := st.chat_input(
    "输入您的问题...",
    disabled=not st.session_state.backend_ok,
):
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "sources": [],
    })
    save_state()

    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Stream assistant response
    with st.chat_message("assistant", avatar="📚"):
        response_placeholder = st.empty()
        status_placeholder = st.empty()

        full_response = ""
        sources = []
        retrieval_ms = 0

        try:
            with status_placeholder.status("🔍 正在检索相关文档...", expanded=False) as status_box:
                # Build history from previous messages (last 10 turns)
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[-10:]
                ]
                for event in stream_qa(
                    question=prompt,
                    history=history,
                    temperature=st.session_state.get("temp", 0.3),
                    max_tokens=st.session_state.get("max_tok", 2048),
                ):
                    etype = event["type"]
                    edata = event["data"]

                    if etype == "status":
                        if isinstance(edata, dict) and edata.get("phase") == "generating":
                            retrieval_ms = edata.get("retrieval_ms", 0)
                            status_box.update(label=f"✅ 检索完成 ({retrieval_ms:.0f}ms)，正在生成回答...")

                    elif etype == "token":
                        full_response += edata if isinstance(edata, str) else ""
                        response_placeholder.markdown(full_response + "▌")

                    elif etype == "sources":
                        sources = edata if isinstance(edata, list) else []
                        status_box.update(label="✅ 回答完成", state="complete")

                    elif etype == "done":
                        status_box.update(label=f"✅ 完成 | 检索 {retrieval_ms:.0f}ms", state="complete")

        except Exception as e:
            full_response = f"❌ 请求失败: {str(e)}"
            response_placeholder.error(full_response)

        # Final render without cursor
        if full_response:
            response_placeholder.markdown(full_response)

            # Show sources
            if sources:
                with st.expander("📎 参考来源", expanded=True):
                    for src in sources:
                        title = src.get("title", "未知章节")
                        source = src.get("source", "")
                        pages = src.get("pages", [])
                        score = src.get("score", 0)
                        st.markdown(
                            f'<div class="source-card">'
                            f'<div class="title">📖 {title}</div>'
                            f'<div class="meta">📄 {source} | 📖 第{",".join(str(p) for p in pages) if pages else "?"}页 '
                            f'| 🎯 相关度: {1 - score:.1%}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": sources,
    })
    save_state()

# ── Clear Chat Button ────────────────────────────────────────────────

col1, col2, col3 = st.columns([1, 1, 1])
with col3:
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
