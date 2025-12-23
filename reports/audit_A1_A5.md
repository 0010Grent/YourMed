# 企业课程设计验收审计（A1-A5）

日期：2025-12-23

本审计以“仓库内可复现实证”为准：优先绑定到可执行命令、可点击代码证据与产物文件。

## 运行校验（本次已执行）

- 单元/契约测试：`python -m pytest`（Exit=0）
- RAG 进程内自检：`python scripts/selftest_rag.py`（Exit=0）
- 评测脚本语法校验：`python -m py_compile scripts/eval_meddg_e2e.py scripts/eval_rag_quality.py scripts/eval_perf.py`（无语法错误）

说明：
- Agent v2 的 HTTP 自检脚本 [scripts/selftest_agent_v2.py](scripts/selftest_agent_v2.py) 需要先启动后端 `uvicorn app.api_server:app`。

## A1-A5 审计结论表

| 维度 | 结论 | 可复现证据（代码/文档/命令） |
|---|---|---|
| A1 前后端分离与模块边界 | PASS | 后端入口 [app/api_server.py](app/api_server.py)；Agent 路由 [app/agent/router.py](app/agent/router.py)；前端工程 [frontend/package.json](frontend/package.json)；项目结构说明 [README.md](README.md) |
| A2 配置、安全与隐私（最小合规） | PASS（最小实现） | `.env` 模板与敏感信息不入库 [\.env.example](.env.example)；API Key 鉴权回归测试 [tests/test_api_auth.py](tests/test_api_auth.py)；日志脱敏（最多 100 字符或 hash）[app/agent/router.py](app/agent/router.py) 与 [app/api_server.py](app/api_server.py)；LLM 超时/重试配置 [app/config_llm.py](app/config_llm.py) |
| A3 RAG 检索与证据契约 | PASS | RAG 核心实现（两阶段检索/可选 rerank）[app/rag/rag_core.py](app/rag/rag_core.py)；RAG 自检脚本 [scripts/selftest_rag.py](scripts/selftest_rag.py)；RAG 评测脚本 [scripts/eval_rag_quality.py](scripts/eval_rag_quality.py) 与产物说明 [reports/README.md](reports/README.md) |
| A4 LangGraph Agent（多轮/追问/可观测） | PASS | Agent 路由 [app/agent/router.py](app/agent/router.py)；Graph 编排与 trace（node_order/timings_ms/rag_stats）[app/agent/graph.py](app/agent/graph.py)；契约测试 [tests/test_agent_graph_trace.py](tests/test_agent_graph_trace.py) 与科普直达测试 [tests/test_agent_kb_qa_bypass.py](tests/test_agent_kb_qa_bypass.py) |
| A5 交付物（文档/自检/评测闭环） | PASS | 评测说明 [app/eval/README_EVAL.md](app/eval/README_EVAL.md)；评测产物目录说明 [reports/README.md](reports/README.md)；三类评测脚本（C1/C2/C3）[scripts/eval_meddg_e2e.py](scripts/eval_meddg_e2e.py)、[scripts/eval_rag_quality.py](scripts/eval_rag_quality.py)、[scripts/eval_perf.py](scripts/eval_perf.py)；测试目录 [tests/](tests/) |

## 用 MedDG_UTF8 测试集的“完整流程”（可直接贴进验收报告）

测试集位置：
- `app/MedDG_UTF8/test.pk`

1) 启动后端：
- `uvicorn app.api_server:app --host 127.0.0.1 --port 8000`

2) C1 端到端多轮回放（Agent v2）：
- `python scripts/eval_meddg_e2e.py --meddg_path app/MedDG_UTF8/test.pk --n 100 --base_url http://127.0.0.1:8000`

3) C2 RAG 离线质量（不依赖 LLM 文本质量，只评检索返回）：
- `python scripts/eval_rag_quality.py --meddg_path app/MedDG_UTF8/test.pk --n 200 --top_k 5 --base_url http://127.0.0.1:8000`

4) C3 并发性能（rag + agent 两条链路）：
- `python scripts/eval_perf.py --base_url http://127.0.0.1:8000 --concurrency 1,5,10 --requests 20 --meddg_path app/MedDG_UTF8/test.pk`

5) 产物检查：
- `reports/meddg_eval_summary.json` / `reports/meddg_eval_cases.csv`
- `reports/rag_eval_summary.json` / `reports/rag_eval_details.csv`
- `reports/perf_eval.json`

## 备注（边界与前置条件）

- C1 的 Agent v2 回放会触发 LLM（回答生成），因此需要配置 `DEEPSEEK_API_KEY`（参考 [\.env.example](.env.example) 与 [app/config_llm.py](app/config_llm.py)）。
- C2/C3 中的 `/v1/rag/retrieve` 依赖本地 Chroma 持久化库（默认目录 `app/rag/kb_store/`），若不存在需先运行入库脚本 `app/rag/ingest_kb.py`（见 [README.md](README.md) 与 RAG 文档）。
