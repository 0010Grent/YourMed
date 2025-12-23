# MedDG 评测闭环（课程设计/企业验收版）

本目录描述“如何用 MedDG 对 MedCaht 做可复现评测”，覆盖：

- C1 端到端多轮对话评测（调用后端 `/v1/agent/chat_v2`）
- C2 RAG 离线质量评估（调用 `/v1/rag/retrieve`，不依赖 LLM 文本质量）
- C3 轻量性能评测（并发=1/5/10）

## 数据格式（MedDG pickle 版）

MedDG 的 `train.pk/test.pk/dev.pk` 在本仓库里表现为：

- 顶层是 `list`，每个元素是一段对话（`dialogue`）
- 每段对话是 `list[dict]`，每个 dict 是一轮，典型字段：
  - `id`: `Patients` / `Doctor`
  - `Sentence`: 中文文本
  - 其他字段：`Symptom/Medicine/Examination/Attribute/Disease`（可用于扩展指标）

评测脚本会：

- 端到端：只用 `Patients` 的 `Sentence` 作为用户输入逐轮回放
- RAG 离线：用 `Patients.Sentence` 作为 query，用紧随其后的 `Doctor.Sentence` 作为 reference

## 指标解释（最小闭环）

### C1 端到端

- `answer_rate`：`mode=answer` 的轮次占比
- `ask_rate`：`mode=ask` 的轮次占比
- `avg_ask_turns_before_answer`：每段对话中从开始到首次 answer 之前的 ask 轮次数平均值（仅统计出现过 answer 的对话）
- `citation_rate`：answer 轮次中 `citations>0` 的占比
- `hit_rate`：所有轮次中 `trace.rag_stats.hits>0` 的占比（主要看进入 answer 的轮次）
- `avg_latency_ms / p95_latency_ms`：客户端测得的 HTTP 往返耗时
- `escalate_rate`：`mode=escalate` 的轮次占比

### C2 RAG 离线

- `recall_at_k`：对每个 query，若任一 evidence.text 与 reference 的相似度 >= 阈值，则记为命中
- `coverage_rate`：evidence.text 非空且长度 >= 阈值的比例
- `avg_max_similarity`：每个 query 的 max 相似度平均值

相似度实现为“字符 bigram Jaccard”，不依赖第三方分词，适合课程设计可复现。

### C3 性能

- 对 `/v1/rag/retrieve` 与 `/v1/agent/chat_v2` 分别压测
- 并发=1/5/10；每个并发发起请求数=20（可配）
- 统计 `avg_ms/p95_ms/error_rate`

## 复现步骤

1) 启动后端：

```powershell
uvicorn app.api_server:app --host 127.0.0.1 --port 8000 --reload
```

2) 运行评测：

```powershell
python scripts/eval_meddg_e2e.py --meddg_path app/MedDG_UTF8/test.pk --n 100 --base_url http://127.0.0.1:8000
python scripts/eval_rag_quality.py --meddg_path app/MedDG_UTF8/test.pk --n 200 --top_k 5 --base_url http://127.0.0.1:8000
python scripts/eval_perf.py --base_url http://127.0.0.1:8000 --concurrency 1,5,10 --requests 20
```

3) 查看产物：

- `reports/meddg_eval_summary.json`
- `reports/meddg_eval_cases.csv`
- `reports/rag_eval_summary.json`
- `reports/rag_eval_details.csv`
- `reports/perf_eval.json`
