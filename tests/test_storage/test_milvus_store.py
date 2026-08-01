"""Tests for Milvus filter expression building (pure logic, no live server)."""

from src.storage.milvus_store import MilvusStore


def build_filter(filters):
    # Skip __init__ (which would create a client) — build_filter_expr is stateless
    store = object.__new__(MilvusStore)
    return store.build_filter_expr(filters)


def test_none_filters():
    assert build_filter(None) is None
    assert build_filter({}) is None


def test_string_value_quoted():
    assert build_filter({"source": "a.pdf"}) == 'source == "a.pdf"'


def test_int_value_unquoted():
    assert build_filter({"page": 3}) == "page == 3"


def test_multiple_filters_joined_with_and():
    expr = build_filter({"source": "a.pdf", "page": 3})
    assert expr == 'source == "a.pdf" and page == 3'


def test_unsupported_value_types_skipped():
    assert build_filter({"source": ["a", "b"], "page": 1}) == "page == 1"
