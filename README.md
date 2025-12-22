# healthcare agent

本仓库实现一个“风险分层 + 就医建议”的医疗对话系统，包含后端服务、RAG 证据检索与前端 UI。

## 技术栈

- 后端：FastAPI + Uvicorn
- Agent 编排：LangGraph（将对话流程编排为状态机）
- LLM 接入：LangChain + OpenAI 兼容接口（例如 DeepSeek）
- RAG：Chroma（本地持久化向量库）+ SentenceTransformers（Embedding）
- 前端：React + Vite

## 系统能力（概览）

- 多轮问诊：收集关键信息，决定是否进入分诊
- 分诊：检索证据、生成结构化结果（风险等级/红旗症状/就医建议等）
- 引用强制：回答中的 `[E1]` 等引用必须能定位到证据块
- 安全审查：`mode=safe` 会启用更严格的安全审查链
- 可观测：返回 `trace`（`INQUIRY → RETRIEVE → ASSESS → SAFETY`）及耗时，便于调试与问题定位

## HTTP API

- `GET /health`：健康检查
- `POST /v1/chat`：多轮对话入口（前端使用）
- `POST /v1/triage`：单次分诊入口（便于独立调试）

---

## 项目结构与模块边界

### 后端（目录：app/）

- `app/api_server.py`
	- **职责**：HTTP 服务入口；实现 `/health`、`/v1/chat`、`/v1/triage`；把“问诊→检索→评估→安全审查”编排为 LangGraph 状态机。
	- **边界**：只做编排与 I/O（请求/响应 schema、session 落盘、trace 汇总），不在这里实现检索细节/业务规则。

- `app/triage_service.py`
	- **职责**：分诊引擎的主实现（RAG 检索、证据引用校验、结构化输出、guardrails、安全审查链等）。
	- **边界**：对外提供可复用步骤（供 LangGraph 节点调用），并保证单次分诊的接口行为稳定。

- `app/triage_protocol.py`
	- **职责**：分诊协议/输出 schema（例如风险等级、红旗症状、就医建议等字段的定义与约束）。
	- **边界**：只定义 schema 与规则，不处理外部依赖（LLM、向量库）。

- `app/config_llm.py`
	- **职责**：LLM 配置与实例化（从 `.env` 读取 key/base_url/model 等）。
	- **边界**：只负责“如何获得可用的 LLM 客户端”，不包含业务流程与分诊策略。

### RAG（目录：app/rag/）

- `app/rag/retriever.py`
	- **职责**：从本地 Chroma 向量库中检索证据块。
	- **边界**：只负责“给定 query → 返回 evidence”，不生成最终回答。

- `app/rag/ingest_kb.py`
	- **职责**：从 `app/rag/kb_docs/` 构建/更新向量库到 `app/rag/kb_store/`。
	- **边界**：这是离线数据工程脚本；运行成本较高（模型下载与 embedding 计算），与在线服务解耦。

- `app/rag/kb_store/`
	- **职责**：持久化向量库（Chroma）。
	- **边界**：可直接拷贝到新电脑使用，避免重建。

### 前端（React + Vite，目录：frontend/）

- `frontend/src/App.tsx`
	- **职责**：对话 UI；右侧面板展示 `rag_query / evidence / trace`。
	- **边界**：只消费后端返回字段并展示，不做任何分诊推理。

### 测试（目录：tests/）

- `tests/test_api_auth.py`
	- **职责**：接口鉴权与基础 API 行为的回归测试。

---

## 启动方式

前置条件：
- Miniconda/Anaconda
- Node.js（建议 18+）

### 1) 创建 Python 环境

```powershell
conda env create -f environment.yml
conda activate healthcare-agent
```

如果不使用 conda，也可以直接用 pip：

```powershell
python -m pip install -r requirements.txt
```

### 2) 配置 .env

```powershell
copy .env.example .env
```

然后编辑 `.env`，至少填写：
- `DEEPSEEK_API_KEY=...`

### 3) 启动后端

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
uvicorn app.api_server:app --host 127.0.0.1 --port 8000
```

健康检查：打开 `http://127.0.0.1:8000/health`，应返回 `{"status":"ok"}`。

### 4) 启动前端

```powershell
cd frontend
npm install
npm run dev
```

打开：`http://127.0.0.1:5173/`。

---

## 环境变量（常用）

- `DEEPSEEK_API_KEY`：必填
- `TRIAGE_API_KEY`：可选，开启接口鉴权（请求需带 `X-API-Key`）
- `OUTPUT_DIR`：默认 `outputs`，保存 `/v1/chat` 的 session 轨迹
- `ALLOW_SAVE_SESSION_RAW_TEXT=1`：可选，落盘保存原文（不推荐，注意隐私）
- `CHAT_SLOT_EXTRACTOR=rules`：可选，强制不用 LLM 抽槽（用于离线测试/稳定性）

Windows 兼容性：
- 若出现 OpenMP 冲突（`libiomp5md.dll already initialized`），可设置 `KMP_DUPLICATE_LIB_OK=TRUE` 再启动后端。

---

## 清理与安全建议（重要）

- 不要把 API Key 写进代码或提交到仓库；推荐放在本地 `.env`。
- `outputs/`、`__pycache__/`、`.pytest_cache/` 都是生成物/缓存（已在 `.gitignore` 中忽略）。