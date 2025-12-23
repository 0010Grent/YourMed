# MedCaht 后端文件索引（M0/M1/M2/M4）

版本：2025-12-23

说明：本索引面向交付/答辩展示，按模块分层列出关键文件、职责、关键入口（函数/路由）与关键环境变量。

---

## M0 基础设施（API/配置/错误处理/鉴权）

- [app/api_server.py](../app/api_server.py)
  - 职责：FastAPI 入口；挂载路由；鉴权与模式策略；暴露 `/v1/rag/*`、`/v1/triage`、`/v1/chat`。
  - 关键入口：`app = FastAPI(...)`、`rag_stats()`、`rag_retrieve()`、`triage()`、`chat()`、`_auth_guard()`、`_apply_mode_policy()`
  - 关键 env：`TRIAGE_API_KEY`、`ALLOW_FAST_MODE`、`ALLOW_SAVE_SESSION_RAW_TEXT`

- [app/config_llm.py](../app/config_llm.py)
  - 职责：LLM 客户端配置（OpenAI-compatible DeepSeek）。
  - 关键入口：`get_llm()`（或同类工厂函数/单例初始化）
  - 关键 env：`DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL`

- [requirements.txt](../requirements.txt) / [environment.yml](../environment.yml)
  - 职责：依赖锁定与环境配置。

---

## M1 RAG（知识库、检索、证据契约）

- [app/rag/rag_core.py](../app/rag/rag_core.py)
  - 职责：核心检索逻辑与 evidence 标准化；可选 rerank。
  - 关键入口：`retrieve()`、`get_stats()`、`_normalize_evidence_item()`

- [app/rag/ingest_kb.py](../app/rag/ingest_kb.py)
  - 职责：CSV 清洗与 Chroma 入库；写入 ingest 进度。
  - 关键入口：命令行 main / ingest 相关函数
  - 产物：`app/rag/kb_store/chroma.sqlite3`、`app/rag/kb_store/ingest_progress.json`

- [app/rag/retriever.py](../app/rag/retriever.py)
  - 职责：对外/兼容封装层（调用 rag_core）。

- [app/rag/utils/rag_shared.py](../app/rag/utils/rag_shared.py)
  - 职责：embedding/rerank provider 选择；device 与模型名解析；fallback。
  - 关键 env：`RAG_DEVICE`、`RAG_EMBEDDING_DEVICE`、`RAG_EMBED_MODEL`、`RAG_USE_RERANKER`、`RAG_PERSIST_DIR`

- [tests/test_rag_retrieve_contract.py](../tests/test_rag_retrieve_contract.py)
  - 职责：验证 `/v1/rag/retrieve` 的 evidence 字段契约（交付证据）。

---

## M2 LangGraph Agent v2（Ask/Answer/Escalate）

- [app/agent/router.py](../app/agent/router.py)
  - 职责：`/v1/agent/chat_v2` 路由与请求/响应 schema。
  - 关键入口：`agent_chat_v2()`、`AgentChatV2Request`、`AgentChatV2Response`

- [app/agent/graph.py](../app/agent/graph.py)
  - 职责：LangGraph 状态机编排；实现 Ask/Answer/Escalate。
  - 关键入口：`run_chat_v2_turn()`、各 `_node_*` 节点、`_trace_start/_trace_end`、路由函数 `_route_after_*`

- [app/agent/state.py](../app/agent/state.py)
  - 职责：会话状态与槽位定义；summary 规则。
  - 关键入口：`Slots`、`AgentSessionState`、`build_summary_from_slots()`

- [app/agent/storage_sqlite.py](../app/agent/storage_sqlite.py)
  - 职责：SQLite 会话存储（WAL + upsert）；state_json 保存最近轮次。
  - 产物：`app/data/agent_sessions.sqlite3`

- [tests/test_agent_graph_trace.py](../tests/test_agent_graph_trace.py)
  - 职责：验证 trace 字段/节点顺序（交付证据）。

- [tests/test_agent_kb_qa_bypass.py](../tests/test_agent_kb_qa_bypass.py)
  - 职责：验证 planner 策略分支（如 kb_qa）行为（交付证据）。

---

## M4 评测闭环（可复现脚本 + 报告产物）

- [scripts/eval_meddg_e2e.py](../scripts/eval_meddg_e2e.py)
  - 职责：MedDG 端到端多轮回放（调用 `/v1/agent/chat_v2`）。
  - 特性：运行前打印样本条数与字段 keys；输出 `reports/meddg_eval_cases.csv`。

- [scripts/eval_rag_quality.py](../scripts/eval_rag_quality.py)
  - 职责：RAG 检索质量评估（调用 `/v1/rag/retrieve`）。
  - 特性：兼容 `evidence/evidences`；输出 `reports/rag_eval_details.csv`。

- [scripts/eval_perf.py](../scripts/eval_perf.py)
  - 职责：并发性能评测（RAG + Agent）。

- [reports/README.md](../reports/README.md)
  - 职责：报告产物说明与复现命令汇总。

---

## 附：旧接口与分诊模块

- [app/triage_service.py](../app/triage_service.py)
  - 职责：单次分诊（triage_once）与 RAG/LLM 调度。

- [app/triage_protocol.py](../app/triage_protocol.py)
  - 职责：分诊 payload 协议定义与构建。
