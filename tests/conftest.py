"""Pytest fixtures for the RAG platform tests."""

import pytest


@pytest.fixture
def sample_markdown() -> str:
    return """# 产品手册

## 第一章 系统概述

本系统是一个企业级智能文档检索与问答平台。它支持多种文档格式的接入和处理。

### 1.1 核心特性

- 多格式文档支持：PDF、DOCX、Markdown、TXT
- 混合检索：融合稠密向量和稀疏关键词检索
- 流式问答：基于SSE的实时逐字输出

### 1.2 技术架构

技术栈包括Python、FastAPI、Milvus、Redis和Qwen大模型。

## 第二章 安装部署

本章介绍系统的安装和部署步骤。

### 2.1 环境要求

需要Python 3.12以上版本，以及Docker环境用于运行Milvus和Redis。

### 2.2 Docker部署

使用docker-compose一键启动所有依赖服务。
"""


@pytest.fixture
def sample_chinese_text() -> str:
    return """深度学习是机器学习的一个分支，它使用多层神经网络来学习数据的表示。

Transformer架构由Vaswani等人在2017年提出，它完全基于注意力机制，
摒弃了传统的循环神经网络结构。

大语言模型（LLM）是近年来人工智能领域最重要的突破之一。
"""
