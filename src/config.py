"""Enterprise RAG Platform - Application Configuration."""

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── 阿里云百炼 API (Qwen2.5-7B-Instruct) ──
    bailian_api_key: str = Field(..., alias="BAILIAN_API_KEY")
    bailian_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        alias="BAILIAN_BASE_URL",
    )
    bailian_model: str = Field(default="qwen2.5-7b-instruct", alias="BAILIAN_MODEL")

    # ── 本地 Embedding 模型 ──
    embedding_model_name: str = Field(
        default="Qwen/Qwen3-Embedding-0.6B",
        alias="EMBEDDING_MODEL_NAME",
    )
    embedding_device: str = Field(default="cpu", alias="EMBEDDING_DEVICE")

    # ── Milvus ──
    milvus_uri: str = Field(default="http://localhost:19530", alias="MILVUS_URI")
    milvus_token: str = Field(default="", alias="MILVUS_TOKEN")

    # ── Redis ──
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    # ── Application ──
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8001, alias="APP_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    upload_dir: str = Field(default="./data/uploads", alias="UPLOAD_DIR")

    # ── Retrieval ──
    retrieval_top_k: int = Field(default=8, alias="RETRIEVAL_TOP_K")
    dense_limit: int = Field(default=20, alias="DENSE_LIMIT")
    sparse_limit: int = Field(default=20, alias="SPARSE_LIMIT")
    rrf_k: int = Field(default=60, alias="RRF_K")
    parent_top_k: int = Field(default=3, alias="PARENT_TOP_K")

    # ── Chunking ──
    parent_chunk_size: int = Field(default=1024, alias="PARENT_CHUNK_SIZE")
    child_chunk_size: int = Field(default=256, alias="CHILD_CHUNK_SIZE")
    chunk_overlap: int = Field(default=32, alias="CHUNK_OVERLAP")

    # ── MinerU ──
    mineru_backend: str = Field(default="pipeline", alias="MINERU_BACKEND")
    mineru_timeout: int = Field(default=1800, alias="MINERU_TIMEOUT")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
