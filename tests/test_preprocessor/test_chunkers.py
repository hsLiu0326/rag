"""Tests for the hierarchical parent-child chunker."""

import pytest

from src.core.preprocessor.chunkers import HierarchicalChunker


@pytest.fixture
def chunker() -> HierarchicalChunker:
    return HierarchicalChunker(parent_size=1024, child_size=256, overlap=32)


def test_heading_tree_splits_into_leaf_sections(chunker, sample_markdown):
    pairs = chunker.split(sample_markdown, {"source": "manual.md", "doc_id": "doc-1"})
    # 产品手册(H1) > 第一章/第二章(H2) > 1.1/1.2/2.1/2.2(H3)
    assert len(pairs) == 4
    for pair in pairs:
        assert pair.parent_text
        assert pair.children
        for child in pair.children:
            assert child.metadata["parent_id"] == pair.parent_id
            assert child.metadata["chunk_type"] == "child"
            assert child.metadata["source"] == "manual.md"
            assert child.metadata["doc_id"] == "doc-1"


def test_title_path_is_breadcrumb(chunker, sample_markdown):
    pairs = chunker.split(sample_markdown, {})
    title_paths = {p.parent_metadata["title_path"] for p in pairs}
    assert "产品手册 > 第一章 系统概述 > 1.1 核心特性" in title_paths
    assert "产品手册 > 第二章 安装部署 > 2.1 环境要求" in title_paths


def test_uniform_fallback_without_headings(chunker):
    text = "第一段。\n\n第二段。\n\n第三段。"
    pairs = chunker.split(text, {"source": "plain.txt"})
    assert len(pairs) == 1
    assert pairs[0].parent_text
    assert pairs[0].children
    assert pairs[0].parent_metadata["title_path"] == ""


def test_large_section_splits_into_multiple_parents(chunker):
    paragraphs = "\n\n".join(
        ["这是一段用于测试分片逻辑的中文文本内容，包含足够多的词语以便累计 token 数量。"]
        * 120
    )
    text = f"# 大章节\n\n{paragraphs}"
    pairs = chunker.split(text, {})
    assert len(pairs) >= 2
    # Parent chunks should stay within the configured budget
    for pair in pairs:
        assert chunker.count_tokens(pair.parent_text) <= chunker.parent_size
        assert pair.children


def test_children_stay_within_child_budget(chunker):
    paragraphs = "\n\n".join(
        ["这是一段用于测试子块大小的中文文本内容。"] * 80
    )
    text = f"# 主题\n\n{paragraphs}"
    pairs = chunker.split(text, {})
    assert pairs
    for pair in pairs:
        for child in pair.children:
            assert chunker.count_tokens(child.text) <= chunker.child_size
            assert child.metadata["chunk_index"] >= 0


def test_empty_content_returns_no_pairs(chunker):
    assert chunker.split("", {}) == []
    assert chunker.split("   \n\n  ", {}) == []


def test_child_indexes_are_sequential(chunker, sample_chinese_text):
    pairs = chunker.split(sample_chinese_text, {})
    assert pairs
    for pair in pairs:
        indexes = [c.metadata["chunk_index"] for c in pair.children]
        assert indexes == list(range(len(indexes)))
