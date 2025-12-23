# MedCaht 后端 API 文档（仅后端）

版本：2025-12-23

说明：

- 本文档根据仓库真实代码生成，接口以 FastAPI 实现为准。
- 鉴权：当环境变量 `TRIAGE_API_KEY` 设置后，部分接口需要请求头 `X-API-Key` 匹配（见 [app/api_server.py](../app/api_server.py) 的 `_auth_guard()`）。
- `/v1/agent/chat_v2` 当前不走 `_auth_guard`（见 [app/agent/router.py](../app/agent/router.py)），如需公网暴露建议补齐。

---

## 0. 统一约定

### 0.1 Base URL

- 本地默认：`http://127.0.0.1:8000`

### 0.2 通用错误响应

当触发错误处理器时，响应形如：

- `400 BAD_REQUEST`
- `401 UNAUTHORIZED`
- `500 INTERNAL_ERROR`

响应结构（示例）：

~~~json
{
  "code": "BAD_REQUEST",
  "message": "请求参数不合法",
  "trace_id": "<uuid>"
}
~~~

证据：见 [app/api_server.py](../app/api_server.py) 的 exception handlers。

---

## 1) POST /v1/agent/chat_v2（LangGraph Agent v2）

实现位置：

- 路由：`agent_chat_v2()`，见 [app/agent/router.py](../app/agent/router.py)
- 业务：`run_chat_v2_turn()`，见 [app/agent/graph.py](../app/agent/graph.py)

### 1.1 请求（JSON）

Schema（来自 Pydantic 模型 `AgentChatV2Request`）：

- `session_id`：string，可选；不传则后端生成
- `user_message`：string，必填；用户输入
- `top_k`：int，默认 5，范围 [1,20]；最终返回证据条数
- `top_n`：int，默认 30，范围 [1,200]；第一阶段召回条数
- `use_rerank`：bool，默认 true；是否启用 rerank

注意：

- `user_message` 不能为空；否则 `run_chat_v2_turn()` 会抛 ValueError（见 [app/agent/graph.py](../app/agent/graph.py)）。

请求示例：

Windows PowerShell：

~~~powershell
$body = @{ session_id = "demo_s1"; user_message = "我头疼两天了"; top_k = 5; top_n = 30; use_rerank = $true } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/v1/agent/chat_v2" -ContentType "application/json" -Body $body
~~~

bash：

~~~bash
curl -sS "http://127.0.0.1:8000/v1/agent/chat_v2" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo_s1","user_message":"我头疼两天了","top_k":5,"top_n":30,"use_rerank":true}'
~~~

### 1.2 响应（JSON）

Schema（来自 Pydantic 模型 `AgentChatV2Response`）：

- `session_id`：string，必有
- `mode`：string，必有；取值为 `ask | answer | escalate`
- `ask_text`：string；Ask 模式下为非空自然追问开场，否则可能为空字符串
- `questions`：array[object]；Ask 模式下为非空，Answer/Escalate 为空数组
- `next_questions`：array[string]；Ask 模式下为非空，Answer/Escalate 为空数组
- `answer`：string；Answer/Escalate 模式下为非空，Ask 模式下为空字符串
- `citations`：array[object]；Answer 且 evidence>0 时通常非空；否则为空数组
- `slots`：object；结构化槽位快照（见 [app/agent/state.py](../app/agent/state.py) 的 `Slots`）
- `summary`：string；由槽位规则生成的短摘要
- `trace`：object；可观测字段（node_order/timings_ms/rag_stats 等）

字段契约（何时为空）：

- `mode=ask`：
  - `ask_text` 非空
  - `questions` 非空
  - `next_questions` 非空
  - `answer` 为空字符串
  - `citations` 为空数组
- `mode=answer`：
  - `answer` 非空
  - `ask_text` 为空字符串
  - `next_questions` 为空数组
  - `citations`：若 RAG hits>0 则通常非空；若 RAG 失败/无证据则为空
- `mode=escalate`：
  - `answer` 非空（红旗分流模板）
  - `questions/next_questions/citations` 为空

证据：见 [app/agent/graph.py](../app/agent/graph.py) 的 `_node_safety_gate/_node_triage_planner/_node_rag_retrieve/_node_answer_compose`。

响应示例（精简示意）：

~~~json
{
  "session_id": "demo_s1",
  "mode": "ask",
  "ask_text": "为了更准确判断，我想再确认几个关键信息。",
  "questions": [{"slot":"age","question":"请问你大概多大年龄？","type":"text"}],
  "next_questions": ["请问你大概多大年龄？"],
  "answer": "",
  "citations": [],
  "slots": {"age": null, "sex": "", "symptoms": [], "red_flags": []},
  "summary": "",
  "trace": {"node_order": ["SafetyGate","MemoryUpdate","TriagePlanner","PersistState"], "timings_ms": {"SafetyGate": 0}}
}
~~~

---

## 2) GET /v1/rag/stats（RAG 底座状态）

实现位置：见 [app/api_server.py](../app/api_server.py) 的 `rag_stats()`，内部调用 [app/rag/rag_core.py](../app/rag/rag_core.py) 的 `get_stats()`。

### 2.1 请求

- 无请求体

Windows PowerShell：

~~~powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/v1/rag/stats"
~~~

bash：

~~~bash
curl -sS "http://127.0.0.1:8000/v1/rag/stats"
~~~

### 2.2 响应

- `collection`：string
- `count`：int
- `persist_dir`：string
- `device`：string（例如 cpu/cuda/cuda:0）
- `embed_model`：string
- `rerank_model`：string|null
- `updated_at`：string

---

## 3) POST /v1/rag/retrieve（独立 RAG 检索）

实现位置：见 [app/api_server.py](../app/api_server.py) 的 `rag_retrieve()`，内部调用 [app/rag/rag_core.py](../app/rag/rag_core.py) 的 `retrieve()`。

### 3.1 鉴权（可选）

- 如果设置了 `TRIAGE_API_KEY`：必须带 `X-API-Key`。

### 3.2 请求（JSON）

来自 `RagRetrieveRequest`（见 [app/api_server.py](../app/api_server.py)）：

- `query`：string，必填
- `top_k`：int，默认 5，范围 [1,20]
- `top_n`：int，默认 30，范围 [1,200]
- `department`：string|null，可选；科室过滤（严格等值匹配）
- `use_rerank`：bool|null，可选；为 null 时按环境变量 `RAG_USE_RERANKER` 决定

Windows PowerShell（带鉴权示例）：

~~~powershell
$headers = @{ "X-API-Key" = "<your_key>" }
$body = @{ query = "咳嗽发热怎么办"; top_k = 5; top_n = 30; department = $null; use_rerank = $true } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/v1/rag/retrieve" -Headers $headers -ContentType "application/json" -Body $body
~~~

bash：

~~~bash
curl -sS "http://127.0.0.1:8000/v1/rag/retrieve" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your_key>" \
  -d '{"query":"咳嗽发热怎么办","top_k":5,"top_n":30,"department":null,"use_rerank":true}'
~~~

### 3.3 响应（JSON）

来自 `rag_retrieve()` 的返回字典（见 [app/api_server.py](../app/api_server.py)）：

- `query`：string
- `top_k`：int
- `top_n`：int
- `use_rerank`：bool（若请求未显式传，则由环境变量推导）
- `evidence`：array[object]，证据列表
- `stats`：object（collection/count/device/embed_model/rerank_model）

`evidence` 单条字段契约（由单测保障）：

- `eid`：string（E1..Ek 连续）
- `text`：string（非空）
- `source`：string（非空）
- `score`：number
- `rerank_score`：number|null
- `chunk_id`：string
- `metadata`：object（至少包含 department/title/row/source_file）

证据：见 [app/rag/rag_core.py](../app/rag/rag_core.py) 的 `_normalize_evidence_item()` 与 `retrieve()`；单测见 [tests/test_rag_retrieve_contract.py](../tests/test_rag_retrieve_contract.py)。

---

## 4) 兼容/旧接口

### 4.1 POST /v1/triage（单次分诊）

实现位置：见 [app/api_server.py](../app/api_server.py) 的 `triage()`，内部调用 `triage_once()`（见 [app/triage_service.py](../app/triage_service.py)）。

请求（`TriageRequest`）：

- `user_text`：string，必填
- `top_k`：int，默认 5
- `mode`：`fast|safe`，默认 fast，但 API 层默认强制为 safe（除非 localhost 且 `ALLOW_FAST_MODE=1`）。见 [app/api_server.py](../app/api_server.py) 的 `_apply_mode_policy()`。
- `clinical_record_path`：string|null，可选

响应：遵循统一分诊协议（见 [app/triage_protocol.py](../app/triage_protocol.py) 的 `build_triage_payload()`）：

- `answer`：object
- `evidence`：array
- `rag_query`：string
- `meta`：object（mode/created_at；可能额外包含 forced_safe）

### 4.2 POST /v1/chat（多轮对话旧接口）

实现位置：见 [app/api_server.py](../app/api_server.py) 的 `chat()`。

请求（`ChatRequest`）：

- `session_id`：string|null，可选
- `patient_message`：string，必填
- `top_k`：int，默认 5
- `mode`：`fast|safe`，默认 fast，但可能被 API 层强制改写为 safe

响应（真实返回字段）：

- `session_id`：string
- `act`：string（INQUIRY/DIAGNOSIS/EXPLANATION/RECOMMENDATION 等，见 [app/api_server.py](../app/api_server.py) 的 `_select_chat_act()`）
- `doctor_reply`：string
- `intake_slots`：object
- `triage`：object|null（为分诊协议 payload 或 null）
- `trace_id`：string
- `debug.trace`：array（LangGraph 编排 trace）

证据：见 [app/api_server.py](../app/api_server.py) 的 `chat()` return 结构。
