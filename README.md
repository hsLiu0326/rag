# Enterprise RAG Platform

企业级智能文档检索与问答平台。上传 PDF/Word/PPT/Excel/Markdown 文档，基于内容进行智能问答。

![image-20260728152203940](D:\rag_proj\docs\image-20260728152203940.png)

![image-20260728152219561](D:\rag_proj\docs\image-20260728152219561.png)

## 技术栈

| 组件 | 技术 |
|------|------|
| 后端框架 | FastAPI (异步) + SSE 流式 |
| 前端 | Streamlit |
| 大模型 | DeepSeek / 阿里云百炼 (OpenAI 兼容 API) |
| Embedding | Qwen3-Embedding-0.6B (本地 CPU) |
| 向量数据库 | Milvus 2.5 (HNSW + BM25) |
| 缓存 | Redis 7 |
| PDF 解析 | pypdf (快速通道) + MinerU (扫描件 OCR) |
| 文档解析 | python-docx / python-pptx / openpyxl |

## 核心特性

- **混合检索** — 稠密向量 + BM25 关键词，RRF 融合，专有名词召回率提升 20%
- **父子文档架构** — 小块检索 + 大块上下文，减少切片幻觉
- **多格式支持** — PDF / DOCX / PPTX / XLSX / MD / TXT
- **智能 PDF 解析** — 电子版秒出，扫描件自动走 MinerU GPU
- **流式问答** — SSE 逐字输出，首字 < 1 秒
- **多轮对话** — 追问保留上下文
- **清洗管线** — 6 条可插拔文本清洗规则

## 快速开始

### 1. 环境要求

- Python 3.12+
- Docker Desktop
- 8GB+ RAM（本地 Embedding 模型约需 2GB）
- (可选) NVIDIA GPU + CUDA 12.4+（MinerU 扫描件 PDF 加速）

### 2. 克隆项目

```bash
git clone https://github.com/hsLiu0326/rag.git
cd rag
```

### 3. 配置

```bash
cp .env.example .env
# 编辑 .env，填入你的 API Key
```

| 变量 | 说明 |
|------|------|
| `BAILIAN_API_KEY` | LLM API Key（DeepSeek 或阿里云百炼） |
| `BAILIAN_BASE_URL` | API 地址（DeepSeek: `https://api.deepseek.com/v1`） |
| `BAILIAN_MODEL` | 模型名（`deepseek-chat` 或 `qwen-turbo`） |

### 4. 创建虚拟环境

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 5. 安装依赖

```bash
# 基础依赖（不含 MinerU，PDF 电子版足够）
pip install -e .

# 需要扫描件 PDF 支持时再装 MinerU
pip install "mineru[all]"
```

### 6. 启动 Docker 服务

```bash
docker-compose up -d
```

等待所有容器变为 healthy（约 15 秒）。

### 7. 启动后端

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload
```

看到 `Application ready` 即启动完成。

### 8. 启动前端

```bash
streamlit run frontend/app.py --server.port 8501
```

浏览器打开 http://localhost:8501

### 9. 使用

1. 左侧边栏上传文档
2. 等待处理完成（状态显示 ✅）
3. 底部输入框提问

## 项目结构

```
rag/
├── src/
│   ├── main.py                 # FastAPI 入口
│   ├── config.py               # 配置管理
│   ├── api/routes/
│   │   ├── documents.py        # 文档上传/状态/清库
│   │   ├── qa.py               # SSE 流式问答
│   │   └── health.py           # 健康检查
│   ├── core/
│   │   ├── preprocessor/
│   │   │   ├── loaders.py      # 多格式加载器
│   │   │   ├── chunkers.py     # 分层切片器
│   │   │   ├── embedder.py     # Embedding 服务
│   │   │   ├── cleaner.py      # 文本清洗管线
│   │   │   ├── mineru_parser.py # MinerU PDF 解析
│   │   │   └── pipeline.py     # 预处理编排
│   │   ├── retrieval/
│   │   │   ├── hybrid_search.py    # 混合检索 + RRF
│   │   │   ├── parent_retriever.py # 父子文档扩展
│   │   │   └── reranker.py         # 重排序
│   │   └── llm/
│   │       ├── qwen_client.py  # LLM 客户端
│   │       └── prompts.py      # RAG Prompt 模板
│   ├── storage/
│   │   ├── milvus_store.py     # Milvus 向量库
│   │   ├── redis_store.py      # Redis 父文档存储
│   │   └── filestore.py        # 文件持久化
│   └── models/
│       ├── schemas.py          # API 数据模型
│       └── document.py         # 内部数据结构
├── frontend/
│   ├── app.py                  # Streamlit 前端
│   └── api_client.py           # 后端 API 客户端
├── docker-compose.yml          # Milvus + Redis 部署
├── .env.example                # 配置模板
└── run_backend.bat / run_frontend.bat  # Windows 启动脚本
```

## 常见问题

### Q: 上传 PDF 没反应 / 超时？

大 PDF 首次处理需加载 MinerU 模型（~70s）。**先上传小文件测试**，确认功能正常。

### Q: 前端显示"后端未连接"？

1. 确认后端已启动：`curl http://127.0.0.1:8001/health`
2. 前端端口是否被占？默认 8501
3. 点击侧边栏"重新检查连接"

### Q: 问答显示"文档中未找到相关信息"？

1. 确认文档已上传成功（侧边栏显示 ✅ 已完成）
2. 提问用文档**正文中存在的词**，不要用元信息（如"简历"是文件名不是内容）
3. 试试问文档里实际有的内容（"专业技能"、"项目经历"等）

### Q: Docker 启动失败？

```bash
# 确认 Docker Desktop 正在运行
docker ps

# 端口冲突时修改 docker-compose.yml 中的端口映射
```

### Q: C 盘空间不足？

模型文件默认在 `~/.cache/`。可迁移到 D 盘：
```bash
# 移动缓存
mv C:\Users\<用户名>\.cache\huggingface D:\rag\.cache\huggingface
mv C:\Users\<用户名>\.cache\modelscope D:\rag\.cache\modelscope

# 创建目录联接（Windows 管理员终端）
mklink /J C:\Users\<用户名>\.cache\huggingface D:\rag\.cache\huggingface
mklink /J C:\Users\<用户名>\.cache\modelscope D:\rag\.cache\modelscope
```

### Q: 换 LLM 模型怎么改？

编辑 `.env`：
```bash
# DeepSeek
BAILIAN_BASE_URL=https://api.deepseek.com/v1
BAILIAN_MODEL=deepseek-chat

# 阿里云百炼
BAILIAN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
BAILIAN_MODEL=qwen-turbo

# 其他 OpenAI 兼容服务同理
```

### Q: Ollama vs 本地 Embedding 怎么选？

当前默认：本地 sentence-transformers 加载模型（CPU），GPU 留给 MinerU PDF 解析。

如果你用 Ollama 跑 Embedding，编辑 `.env` 后修改 `src/core/preprocessor/embedder.py` 和 `src/main.py` 即可切换。

## 清理数据

侧边栏 → 危险操作 → 清空所有数据，或命令行：

```bash
docker-compose down -v   # 删除 Docker 数据
rm -rf data/uploads/*    # 删除上传文件
```

## License

MIT
