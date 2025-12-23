# MedCaht 后端项目总报告（LLM + RAG + LangGraph Agent + 评测闭环）

版本：2025-12-23

本报告仅覆盖后端（FastAPI / RAG / LangGraph Agent / 评测脚本与产物），不包含前端。

---

## 1. 背景与目标

目标：构建一个面向中文医患对话场景的“风险分层 + 证据检索 + 可追问编排”的后端服务。

- LLM：通过 LangChain 的 OpenAI 兼容接口对接（默认 DeepSeek 兼容端点），用于回答生成与槽位抽取（可降级）。见 [app/config_llm.py](../app/config_llm.py)。
- RAG：使用 Chroma 本地持久化向量库，默认优先使用 BCEmbedding 的 BCE embedding；可选 BCE reranker 两阶段重排。见 [app/rag/rag_core.py](../app/rag/rag_core.py) 与 [app/rag/utils/rag_shared.py](../app/rag/utils/rag_shared.py)。
- LangGraph Agent：提供多轮“Ask / Answer / Escalate”状态机、结构化追问、session 持久化与 trace。见 [app/agent/graph.py](../app/agent/graph.py)。
- 评测闭环：提供端到端多轮回放、RAG 离线质量评估、轻量并发压测脚本与 reports 产物说明。见 [scripts/](../scripts/) 与 [reports/README.md](../reports/README.md)。

---

## 2. 需求与约束

### 2.1 安全

- 非诊断：输出为风险分层与就医建议，不替代医生面诊。
- 红旗分流：高风险症状直接 escalate（建议尽快就医/急救）。见 [app/agent/graph.py](../app/agent/graph.py) 的函数 `_node_safety_gate`。
- API 鉴权：对 `/v1/triage`、`/v1/chat`、`/v1/rag/retrieve` 提供可选 `X-API-Key` 鉴权（通过环境变量 `TRIAGE_API_KEY`）。见 [app/api_server.py](../app/api_server.py) 的函数 `_auth_guard`。

### 2.2 可解释

- 强制引用契约：M2 Answer 模式回答必须包含“引用：[...]”行与免责声明。见 [app/agent/graph.py](../app/agent/graph.py) 的函数 `_ensure_answer_contract`。
- RAG evidence 统一字段：M1 `retrieve()` 返回的每条证据包含 `eid/text/source/score/rerank_score/metadata/chunk_id`。见 [app/rag/rag_core.py](../app/rag/rag_core.py) 的函数 `retrieve` 与 `_normalize_evidence_item`，以及测试 [tests/test_rag_retrieve_contract.py](../tests/test_rag_retrieve_contract.py)。

### 2.3 可观测

- 请求 trace_id：所有请求经 middleware 自动分配 trace_id 并注入日志。见 [app/api_server.py](../app/api_server.py) 的 middleware `add_trace_id_middleware`。
- Agent trace：`/v1/agent/chat_v2` 返回 `trace.node_order` 与 `trace.timings_ms`，并在 RAG 节点记录 `trace.rag_stats`。见 [app/agent/graph.py](../app/agent/graph.py) 的 `_trace_start/_trace_end` 与 `_node_rag_retrieve`。

### 2.4 可复现

- 评测脚本必须打印 MedDG 读取样本条数与字段名，避免“假设格式正确”。见 [scripts/eval_meddg_e2e.py](../scripts/eval_meddg_e2e.py)、[scripts/eval_rag_quality.py](../scripts/eval_rag_quality.py)、[scripts/eval_perf.py](../scripts/eval_perf.py)。
- reports 目录给出可复现命令。见 [reports/README.md](../reports/README.md)。

---

## 3. 总体架构

### 3.1 模块图（Mermaid）

~~~mermaid
flowchart LR
  subgraph API[FastAPI 服务]
    A[app/api_server.py\nFastAPI app + middleware + /v1/*]
    AR[app/agent/router.py\n/v1/agent/chat_v2]
  end

  subgraph M2[LangGraph Agent (M2)]
    G[app/agent/graph.py\nStateGraph: Safety->Memory->Plan->RAG->Compose->Persist]
    S[app/agent/state.py\nSlots + Summary]
    ST[app/agent/storage_sqlite.py\nSQLite sessions]
  end

  subgraph M1[RAG Service (M1)]
    RC[app/rag/rag_core.py\nChroma + Embedding + optional rerank]
    RS[app/rag/utils/rag_shared.py\nembedding/device/config]
    KB[(app/rag/kb_store/\nChroma persist)]
  end

  subgraph M0[Ingest (M0)]
    IN[app/rag/ingest_kb.py\nKB CSV -> chunk -> Chroma]
    KBD[(app/rag/kb_docs/\nCSV knowledge base)]
  end

  subgraph M4[Eval Loop (M4)]
    E1[scripts/eval_meddg_e2e.py\nE2E replay]
    E2[scripts/eval_rag_quality.py\nRAG offline]
    E3[scripts/eval_perf.py\nPerf]
    REP[(reports/\nJSON/CSV artifacts)]
  end

  A --> AR --> G
  G --> S
  G --> ST
  G --> RC
  RC --> RS
  RC --> KB
  IN --> RS
  IN --> KB
  IN --> KBD
  E1 --> A
  E2 --> A
  E3 --> A
  E1 --> REP
  E2 --> REP
  E3 --> REP
~~~

### 3.2 时序图（Mermaid）

以 `/v1/agent/chat_v2` 一轮为例：

~~~mermaid
sequenceDiagram
  participant C as Client
  participant API as FastAPI (/v1/agent/chat_v2)
  participant LG as LangGraph (app/agent/graph.py)
  participant DB as SQLite (app/data/agent_sessions.sqlite3)
  participant RAG as RAG Core (app/rag/rag_core.py)
  participant LLM as LLM (DeepSeek via LangChain)

  C->>API: POST /v1/agent/chat_v2 {session_id?, user_message, top_k, top_n, use_rerank}
  API->>LG: run_chat_v2_turn(...)
  LG->>DB: load_session(session_id)
  LG->>LG: SafetyGate
  alt red_flag_hits
    LG->>LG: mode=escalate + answer=ESCALATE_ANSWER_TEMPLATE
  else no red flags
    LG->>LG: MemoryUpdate (slot extractor: LLM or rules)
    LG->>LG: TriagePlanner (mode=ask or answer)
    alt mode=answer
      LG->>RAG: retrieve(rag_query, top_k, top_n, department, use_rerank)
      alt evidence>0
        LG->>LLM: compose answer with evidence block
      else evidence=0
        LG->>LG: deterministic fallback answer
      end
    else mode=ask
      LG->>LG: build structured follow-up questions
    end
  end
  LG->>DB: save_session(updated state)
  API-->>C: {mode, ask_text, next_questions, answer, citations, slots, summary, trace}
~~~

---

## 4. 模块设计

### 4.1 M0 数据处理与入库（GPU/编码统一/分科合并）

关键实现：

- 入库脚本：将 [app/rag/kb_docs/](../app/rag/kb_docs/) 下 CSV 读取、清洗、切分并写入 Chroma 持久化目录 [app/rag/kb_store/](../app/rag/kb_store/)。见 [app/rag/ingest_kb.py](../app/rag/ingest_kb.py)。
- 统一 embedding/device：入库与检索共用 `make_embeddings()` 与 `resolve_embedding_device()`，确保一致性并支持 GPU 优先。见 [app/rag/utils/rag_shared.py](../app/rag/utils/rag_shared.py)。
- Windows OpenMP 兼容：入库与检索都调用 `apply_windows_openmp_workaround()` 进行兼容处理。见 [app/rag/utils/rag_shared.py](../app/rag/utils/rag_shared.py)。

交付建议：

- 在运行入库前固定环境变量（embedding provider、device、模型名）以保证可复现。
- 对 CSV 的编码与非法行做拒绝记录，避免 “silent bad data”。见 [app/rag/ingest_kb.py](../app/rag/ingest_kb.py) 的 `_append_bad_row`。

### 4.2 M1 RAGService（Chroma + BCEEmbedding + 可选 BCE rerank）

在线接口：

- `/v1/rag/stats`：返回 `collection/count/persist_dir/device/embed_model/rerank_model/updated_at`。见 [app/api_server.py](../app/api_server.py) 的 `rag_stats()`。
- `/v1/rag/retrieve`：返回 `evidence` 列表与 `stats`。见 [app/api_server.py](../app/api_server.py) 的 `rag_retrieve()`。

核心实现：

- `retrieve(query, top_k, top_n, department, use_rerank)`：两阶段检索（top_n 向量召回 + 可选 rerank），返回固定 evidence 契约。见 [app/rag/rag_core.py](../app/rag/rag_core.py) 的 `retrieve()`。
- evidence 字段契约由单测保障。见 [tests/test_rag_retrieve_contract.py](../tests/test_rag_retrieve_contract.py)。

### 4.3 M2 LangGraphAgent（/v1/agent/chat_v2：Ask/Answer/Escalate）

在线接口：

- `/v1/agent/chat_v2`：多轮编排入口，返回 mode、结构化追问、回答、引用、槽位、摘要与 trace。见 [app/agent/router.py](../app/agent/router.py) 的 `agent_chat_v2()`。

核心状态机：

- 节点：`SafetyGate`（红旗分流）、`MemoryUpdate`（槽位抽取与摘要）、`TriagePlanner`（Ask/Answer 决策）、`RAGRetrieve`（调用 M1）、`AnswerCompose`（LLM 或模板）、`PersistState`（SQLite 落库）。见 [app/agent/graph.py](../app/agent/graph.py)。
- Ask/Answer/Escalate 条件：
  - Escalate：`SafetyGate` 命中红旗关键词，直接返回。见 `_node_safety_gate()`。
  - Ask：槽位不足则构造结构化追问。见 `_node_triage_planner()`。
  - Answer：槽位满足或知识问答直达，进入检索与回答。见 `_node_triage_planner()` 与 `_node_rag_retrieve()`。
- Session 持久化：SQLite 表 `sessions(session_id PRIMARY KEY, state_json, updated_at)`，默认路径为 [app/data/agent_sessions.sqlite3](../app/data/agent_sessions.sqlite3)。见 [app/agent/storage_sqlite.py](../app/agent/storage_sqlite.py)。
- Trace：`trace.node_order` 与 `trace.timings_ms` 为验收字段，单测覆盖。见 [tests/test_agent_graph_trace.py](../tests/test_agent_graph_trace.py)。

### 4.4 M4 评测闭环（MedDG 端到端 + RAG 离线 + 性能）

- 端到端多轮回放：通过 HTTP 调用 `/v1/agent/chat_v2`，统计 Answer/Ask/Escalate/引用率/hit_rate/P95 等，输出 `reports/meddg_eval_summary.json` 与 `reports/meddg_eval_cases.csv`（并生成兼容别名 `reports/cases.csv`）。见 [scripts/eval_meddg_e2e.py](../scripts/eval_meddg_e2e.py)。
- RAG 离线质量：从 MedDG 抽取 (patient, doctor) 对，调用 `/v1/rag/retrieve`，用字符 bigram Jaccard 作为近似可解释相似度，输出 `reports/rag_eval_summary.json` 与 `reports/rag_eval_details.csv`（并生成兼容别名 `reports/details.csv`）。见 [scripts/eval_rag_quality.py](../scripts/eval_rag_quality.py)。
- 轻量性能压测：对 `/v1/rag/retrieve` 与 `/v1/agent/chat_v2` 进行并发 1/5/10 请求，输出 `reports/perf_eval.json`。见 [scripts/eval_perf.py](../scripts/eval_perf.py)。

---

## 5. 关键实现亮点（有据可查）

- 统一证据契约与单测回归：M1 `retrieve()` 固定 evidence 字段并由 [tests/test_rag_retrieve_contract.py](../tests/test_rag_retrieve_contract.py) 校验，降低“上游改动导致下游崩”的风险。
- Agent 可观测 trace：`run_chat_v2_turn()` 初始化 `trace` 并在每个节点记录 node_order/timings_ms；`RAGRetrieve` 还记录 `trace.rag_stats`。见 [app/agent/graph.py](../app/agent/graph.py) 的 `_trace_start/_trace_end/_node_rag_retrieve/run_chat_v2_turn`。
- 安全分流与回答护栏：红旗命中直接 escalate（`_node_safety_gate`），Answer 模式输出强制包含引用与免责声明（`_ensure_answer_contract`）。见 [app/agent/graph.py](../app/agent/graph.py)。
- API 层 fast->safe 强制策略：默认不允许绕过安全链，只有 localhost + `ALLOW_FAST_MODE=1` 才允许 fast。见 [app/api_server.py](../app/api_server.py) 的 `_apply_mode_policy()` 与单测 [tests/test_api_auth.py](../tests/test_api_auth.py)。
- 数据隐私最小化：
  - `/v1/chat` 默认落盘仅保存 meta（present/length/sha256），除非显式 `ALLOW_SAVE_SESSION_RAW_TEXT=1`。见 [app/api_server.py](../app/api_server.py) 的 `chat()`。
  - Agent 日志对用户文本仅打印前 100 字或 hash。见 [app/agent/router.py](../app/agent/router.py) 的 `_safe_for_log()` 与 [app/agent/graph.py](../app/agent/graph.py) 的 `_safe_text_for_log()`。

---

## 6. 风险与局限（基于现状，不虚构结果）

- `/v1/agent/chat_v2` 当前未接入 `X-API-Key` 鉴权（仅 triage/chat/rag_retrieve 有 `_auth_guard`）。若面向公网需补齐鉴权与限流。
- RAG 质量依赖本地知识库 CSV 与入库参数；目前仓库未提供“黄金标准”标注，因此离线质量评估使用近似指标（bigram Jaccard），只能用于横向对比与回归。
- LLM 可用性与成本不可控：DeepSeek key 缺失或网络异常时会触发降级路径；不同 LLM 版本可能影响回答风格与可解释性。

---

## 7. 未来工作

- 安全与合规：为 `/v1/agent/chat_v2` 增加鉴权、限流、审计日志；引入更严格的隐私脱敏策略（PII 检测与日志擦除）。
- 评测质量：建立标注集与更严格的 RAG 评价（例如基于标准答案的 Recall@k、引用准确率），并将评测接入 CI。
- 可观测性：将 trace 规范化为 OpenTelemetry span（或至少统一字段字典），并输出到集中式日志系统。
