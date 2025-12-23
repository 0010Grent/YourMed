
# MedCaht 差距审计报告（基于仓库代码取证）

日期：2025-12-22

> 说明：本报告逐条对照“目标规格”，所有结论均附【证据：文件路径 + 行号链接 + 关键片段】；若未发现实现，会注明“未发现”并给出已搜索关键词范围。

---

## A. 仓库现状总览（入口、依赖、可运行性、启动方式）（附证据）

### A1) 目录树（2层深度）

- 根目录
	- .env
	- .env.example
	- README.md
	- requirements.txt
	- environment.yml
	- app/
	- frontend/
	- outputs/
	- tests/

- app/（2层）
	- api_server.py
	- config_llm.py
	- triage_protocol.py
	- triage_service.py
	- rag/
		- ingest_kb.py
		- retriever.py
		- kb_docs/
		- kb_store/

- frontend/（2层）
	- package.json
	- src/
		- main.tsx
		- App.tsx

### A2) 入口定位

- 后端入口（FastAPI）：app对象与路由定义在 [app/api_server.py](app/api_server.py#L980-L1292)
	- 已实现路由：`GET /health`、`POST /v1/triage`、`POST /v1/chat`
	- 证据片段：
		- [app/api_server.py](app/api_server.py#L1010-L1018)：`@app.get("/health")`
		- [app/api_server.py](app/api_server.py#L1152-L1183)：`@app.post("/v1/triage")`
		- [app/api_server.py](app/api_server.py#L1184-L1292)：`@app.post("/v1/chat")`

- 前端入口（Vite+React）：挂载点在 [frontend/src/main.tsx](frontend/src/main.tsx#L1-L11)，主要UI逻辑在 [frontend/src/App.tsx](frontend/src/App.tsx#L1-L549)
	- 证据片段：
		- [frontend/src/main.tsx](frontend/src/main.tsx#L1-L11)：`createRoot(...).render(<App />)`

### A3) 配置文件（.env/README/示例配置）

- 环境变量示例：DeepSeek key/base_url/model、鉴权、输出目录、RAG配置见 [.env.example](.env.example#L1-L22)
	- 证据片段：
		- [.env.example](.env.example#L1-L4)：`DEEPSEEK_API_KEY/DEEPSEEK_BASE_URL/DEEPSEEK_MODEL`
		- [.env.example](.env.example#L6-L8)：`TRIAGE_API_KEY`
		- [.env.example](.env.example#L10-L18)：`OUTPUT_DIR/ALLOW_SAVE_SESSION_RAW_TEXT/CHAT_SLOT_EXTRACTOR`
		- [.env.example](.env.example#L20-L22)：`RAG_COLLECTION/HF_EMBEDDING_MODEL`

- LLM客户端配置：用LangChain ChatOpenAI走OpenAI兼容环境变量 `OPENAI_API_KEY/OPENAI_API_BASE`，由`DEEPSEEK_*`注入，见 [app/config_llm.py](app/config_llm.py#L70-L125)
	- 证据片段：
		- [app/config_llm.py](app/config_llm.py#L93-L112)：读取 `DEEPSEEK_API_KEY/DEEPSEEK_BASE_URL/DEEPSEEK_MODEL`
		- [app/config_llm.py](app/config_llm.py#L115-L121)：写入 `OPENAI_API_KEY/OPENAI_API_BASE`

### A4) 依赖清单（后端requirements / 前端package.json）

- 后端依赖：FastAPI/Uvicorn/Pydantic/LangChain/LangGraph/Chroma/Sentence-Transformers等见 [requirements.txt](requirements.txt#L1-L26)
	- 证据片段：
		- [requirements.txt](requirements.txt#L1-L14)：`fastapi/uvicorn/pydantic/openai/langchain/langgraph`
		- [requirements.txt](requirements.txt#L16-L20)：`chromadb/sentence-transformers`
	- 关键差距：未包含BCEmbedding / bce-embedding相关依赖（目标规格要求 BCEmbedding）

- 前端依赖：React + Vite见 [frontend/package.json](frontend/package.json#L1-L34)

### A5) 可运行性与启动方式

- README提供conda与uvicorn启动方式见 [README.md](README.md#L70-L120)
- `TRIAGE_API_KEY`鉴权与`fast→safe`策略在测试中有覆盖见 [tests/test_api_auth.py](tests/test_api_auth.py#L1-L132)

---

## B. 模块审计（M0~M5：现状→证据→缺口→建议）

> 目标规格（摘要）：DeepSeek API + RAG（BCEmbedding+Chroma持久化+citations）+ Agent（LangChain/LangGraph编排，Planner/Inquiry/Retrieve/Safety/Doctor/Memory，trace回放）+ 数据入库（Toyhom+Zhongjing→DataGateway→normalized/*.jsonl→ingest，UTF-8/清洗/NFKC/去\x00）+ 评测（MedDG仅评测）+ 后端接口（/v1/chat/send、/v1/kb/ingest、/v1/admin/trace/{trace_id}、/health）+ 前端展示（对话、citations、risk、follow_up、trace入口）+ 合规。

### M0：DataGateway（编码统一+清洗+科室合并+去重+脱敏+normalized输出）

#### 现状

- **未形成独立DataGateway与normalized产物**：仓库未发现任何`normalized/`目录或写入`*.jsonl`的实现。
	- 证据：仓库结构中无normalized目录；且业务入库脚本直接从`kb_dir`读取数据源文件并写入Chroma，见 [app/rag/ingest_kb.py](app/rag/ingest_kb.py#L560-L740)
	- 关键片段：`load_raw_docs(kb_dir)`遍历`kb_dir.rglob("*")`直接读取`.csv/.pdf/.md/.txt`

- **编码处理为启发式选编码 + errors=replace**：会把乱码替换字符`\ufffd`纳入文本（不满足“严禁乱码进入SQLite/Chroma”的硬约束）。
	- 证据：编码选择与打开文件方式见 [app/rag/ingest_kb.py](app/rag/ingest_kb.py#L90-L170) 与 [app/rag/ingest_kb.py](app/rag/ingest_kb.py#L300-L360)
	- 关键片段：`s = sample_bytes.decode(enc, errors="replace")`；`open(..., errors="replace")`

- **科室合并（仅对Chinese-medical-dialogue-data的目录结构）**：推断`department_group`并写入metadata。
	- 证据：`_infer_department_group`与`md["department_group"] = dept`见 [app/rag/ingest_kb.py](app/rag/ingest_kb.py#L763-L910)

- **脱敏（会话落盘默认不保存原文）**：`/v1/chat`落盘仅保存`present/length/sha256`，默认`ALLOW_SAVE_SESSION_RAW_TEXT=0`。
	- 证据：会话落盘与元信息计算见 [app/api_server.py](app/api_server.py#L132-L210) 与 [app/api_server.py](app/api_server.py#L1184-L1292)

#### 缺口（对照目标规格）

- **缺失DataGateway硬门槛**：
	- UTF-8统一、控制字符清洗、去`\x00`、统一换行、Unicode NFKC
	- 严禁乱码/bytes进入SQLite或Chroma
	- 所有原始数据先落`normalized/*.jsonl`再入库
	- 证据：未发现`unicodedata`/`NFKC`相关实现（关键词：`unicodedata|NFKC|normalize(`，范围：`**/*.py, **/*.md`；结果：未发现）。

- **缺失Zhongjing多轮对话入库与科室推断**：仓库未出现`Zhongjing/SupritYoung`相关实现。
	- 证据：入库进度只出现Chinese-medical-dialogue-data路径，见 [app/rag/kb_store/ingest_progress.json](app/rag/kb_store/ingest_progress.json#L1-L38)

#### 建议

- 新增`app/datagateway/`：
	- 统一读Toyhom+Zhongjing → 严格UTF-8文本化与清洗（含NFKC、\x00、控制字符、换行统一） → 输出`normalized/{dataset}/{dept}.jsonl`
	- `POST /v1/kb/ingest`只允许从`normalized/`消费并入库

---

### M1：RAGService（BCEmbedding+Chroma+检索契约+citations）

#### 现状

- **Chroma持久化**：persist目录固定为`app/rag/kb_store`，collection默认`medical_kb`。
	- 证据：[app/rag/retriever.py](app/rag/retriever.py#L35-L80)：`persist_directory=str(persist_dir)`

- **Embedding当前为HuggingFaceEmbeddings**，默认模型`intfloat/multilingual-e5-small`。
	- 证据：检索侧默认模型 [app/rag/retriever.py](app/rag/retriever.py#L12-L25)
	- 证据：入库侧`HuggingFaceEmbeddings(model_name=model_name)` [app/rag/ingest_kb.py](app/rag/ingest_kb.py#L660-L740)

- **检索输出契约（evidence项字段）**：`eid/source/page/section/chunk_id/score/text`。
	- 证据：[app/rag/retriever.py](app/rag/retriever.py#L80-L120)

- **citations校验与重算存在**：禁止引用不存在的EID；最终重算`citations_used`。
	- 证据：引用校验与剥离逻辑见 [app/triage_service.py](app/triage_service.py#L650-L770)

#### 缺口

- **未满足BCEmbedding(bce-embedding-base_v1)**：当前使用E5-small且requirements无BCEmbedding。
	- 证据：依赖中无BCEmbedding [requirements.txt](requirements.txt#L1-L26)
	- 证据：实现使用`HuggingFaceEmbeddings` [app/rag/retriever.py](app/rag/retriever.py#L12-L25)

- **API层缺少独立`citations`字段**：当前协议为`answer/evidence/rag_query/meta`。
	- 证据：协议构建见 [app/triage_protocol.py](app/triage_protocol.py#L18-L55)

#### 建议

- 引入BCEmbedding并保证入库/检索一致；把“证据块(evidence)”与“引用(citations)”在API层显式分离：
	- `citations[]`用于前端展示与追溯
	- `evidence[]`可作为debug/高级模式保留

---

### M2：AgentOrchestrator（LangGraph节点、短路安全、trace回放）

#### 现状

- **LangGraph编排已存在**：chat状态机节点包含`preprocess/inquiry/triage_retrieve/triage_assess/triage_safety/triage_build_payload/format`。
	- 证据：节点注册见 [app/api_server.py](app/api_server.py#L939-L969)

- **安全链存在**：safe模式会跑`safety.state/serious/errors/modify_json`，并回到`ensure_answer_json`再校验。
	- 证据：[app/triage_service.py](app/triage_service.py#L520-L690)

- **trace存在（轻量、避免泄漏原文）**：trace条目为`{step,status,ms,...}`。
	- 证据：chat trace append函数 [app/api_server.py](app/api_server.py#L666-L700)
	- 证据：triage trace append函数 [app/triage_service.py](app/triage_service.py#L452-L510)

#### 缺口

- **节点集合不满足目标的6类节点**：未见Planner/Memory节点；Doctor节点不是独立“生成节点”（当前format是确定性格式化）。
	- 证据：现有节点清单见 [app/api_server.py](app/api_server.py#L939-L969)

- **缺失trace回放能力（接口+持久化）**：
	- 目标要求：trace_id + 每节点input/output/latency/citations_used
	- 当前：trace仅少量标量、且没有`GET /v1/admin/trace/{trace_id}`接口
	- 证据：路由仅`/health,/v1/triage,/v1/chat`见 [app/api_server.py](app/api_server.py#L1010-L1292)

#### 建议

- 引入TraceStore（SQLite）并在LangGraph每节点写入：
	- node_name
	- input_summary（脱敏摘要）
	- output_summary（脱敏摘要）
	- latency_ms
	- citations_used
	- citations_used建议复用现有`citations_used`提取逻辑（证据：见 [app/triage_service.py](app/triage_service.py#L50-L90)）

---

### M3：BackendAPI（FastAPI接口、DeepSeekClient封装、SQLite表、日志/审计）

#### 现状

- **已有接口**：
	- `GET /health` [app/api_server.py](app/api_server.py#L1010-L1018)
	- `POST /v1/triage` [app/api_server.py](app/api_server.py#L1152-L1183)
	- `POST /v1/chat` [app/api_server.py](app/api_server.py#L1184-L1292)

- **DeepSeek接入**：通过LangChain ChatOpenAI走OpenAI兼容环境变量。
	- 证据：[app/config_llm.py](app/config_llm.py#L70-L125)

- **会话存储**：落盘到`OUTPUT_DIR/sessions/{session_id}.json`，默认不保存原文。
	- 证据：[app/api_server.py](app/api_server.py#L1184-L1292)

#### 缺口

- **接口契约不匹配目标规格**：
	- 目标：`POST /v1/chat/send`返回`reply,follow_up_questions,risk_level,citations,trace_id,memory_write`
	- 当前：`POST /v1/chat`返回`session_id,act,doctor_reply,intake_slots,triage,trace_id,debug`
	- 证据：返回体见 [app/api_server.py](app/api_server.py#L1265-L1292)

- **缺失`POST /v1/kb/ingest`（从normalized入库）**：当前入库仅离线脚本。
	- 证据：入库脚本入口见 [app/rag/ingest_kb.py](app/rag/ingest_kb.py#L960-L987)

- **缺失`GET /v1/admin/trace/{trace_id}`**：未发现相关路由。
	- 证据：路由清单仅见 [app/api_server.py](app/api_server.py#L1010-L1292)

#### 建议

- 新增v1路由层（保持兼容或迁移）：
	- `POST /v1/chat/send`
	- `POST /v1/kb/ingest`
	- `GET /v1/admin/trace/{trace_id}`
	- 并引入SQLite作为TraceStore（业务侧），用于回放与审计

---

### M4：FrontendUI（证据、风险、追问、trace入口）

#### 现状

- 已展示：对话、trace_id、trace步骤列表、RAG查询、证据列表。
	- 证据：发送请求与消息渲染见 [frontend/src/App.tsx](frontend/src/App.tsx#L300-L560)

- 前端调用：`fetch('/v1/chat')`
	- 证据：[frontend/src/App.tsx](frontend/src/App.tsx#L313-L350)

#### 缺口

- 目标要求展示：risk提示、follow_up问题卡片、trace入口（回放接口）。当前未实现。
	- 证据：UI仅呈现`act/trace_id/trace/evidence/rag_query`，未出现`risk_level/follow_up_questions`等字段，见 [frontend/src/App.tsx](frontend/src/App.tsx#L360-L560)

#### 建议

- 迁移到`/v1/chat/send`契约后：
	- risk banner（紧急/高风险短路提示）
	- follow-up卡片（点击可直接填入输入框或发送）
	- trace回放入口（调用`/v1/admin/trace/{trace_id}`展示节点回放）

---

### M5：Evaluation（MedDG评测runner、指标输出、case导出）

#### 现状

- 未发现MedDG评测runner；tests仅做鉴权与chat落盘回归。
	- 证据：[tests/test_api_auth.py](tests/test_api_auth.py#L1-L220)

#### 缺口

- 目标要求：MedDG仅用于评测（不入库），逐轮回放→调用/chat接口→生成reports(json/csv)，并支持红旗症状用例集。未发现实现。
	- 未发现（关键词：`MedDG|meddg|evaluate|metric|replay|report|csv`，范围：`**/*.{py,md}`；结果：未发现）

#### 建议

- 新增`eval/`目录：
	- `eval/runner_meddg.py`：逐轮回放并写`reports/*.json`与`reports/*.csv`
	- `eval/red_flag_cases.jsonl`：红旗症状安全用例集

---

## C. 差距矩阵表（必须有）

| 模块&关键子能力 | 旧项目状态 | 证据 | 差距描述 | 建议改造 | 优先级 |
|---|---|---|---|---|---|
| M0：normalized/*.jsonl落盘 | 缺失 | 入库直读kb_dir [app/rag/ingest_kb.py](app/rag/ingest_kb.py#L560-L740) | 不满足“先normalized再入库”硬要求 | 新增DataGateway产物目录 + 改`/v1/kb/ingest`只消费normalized | P0 |
| M0：UTF-8/控制字符/\x00/NFKC统一 | 部分具备 | `errors="replace"`读CSV [app/rag/ingest_kb.py](app/rag/ingest_kb.py#L300-L360)；NFKC未发现 | 可能把乱码写入Chroma | DataGateway严格清洗与拒绝坏行 + NFKC | P0 |
| M0：Toyhom单轮QA按科室合并 | 部分具备 | 仅看到Chinese-medical-dialogue-data入库进度 [app/rag/kb_store/ingest_progress.json](app/rag/kb_store/ingest_progress.json#L1-L38) | 无“Toyhom→normalized→ingest”流水线 | 新增Toyhom解析与科室合并到normalized | P0 |
| M0：Zhongjing多轮对话科室推断 | 缺失 | 未见Zhongjing实现；进度文件无该数据源 [app/rag/kb_store/ingest_progress.json](app/rag/kb_store/ingest_progress.json#L1-L38) | 缺少多轮与科室推断规则 | 新增Zhongjing解析与规则推断科室 | P0 |
| M1：Chroma持久化 | 已具备 | persist_dir固定 [app/rag/retriever.py](app/rag/retriever.py#L35-L80) | 满足持久化要求 | 保留 | P0 |
| M1：BCEmbedding(bce-embedding-base_v1) | 缺失 | 当前E5-small [app/rag/retriever.py](app/rag/retriever.py#L12-L25)；依赖无BCEmbedding [requirements.txt](requirements.txt#L1-L26) | 不满足目标embedding | 引入BCEmbedding并统一入库/检索 | P0 |
| M1：citations输出（字段级） | 部分具备 | evidence + `[E#]` + `citations_used` [app/triage_service.py](app/triage_service.py#L650-L770) | API缺少独立`citations`字段 | 在`/v1/chat/send`返回规范化citations数组 | P0 |
| M2：LangGraph编排 | 部分具备 | 节点链路 [app/api_server.py](app/api_server.py#L939-L969) | 不满足Planner/Doctor/Memory节点集合 | 新增Planner/Doctor/Memory节点（可最小实现） | P0 |
| M2：trace回放（节点I/O/latency/citations_used） | 部分具备 | trace仅step/status/ms [app/triage_service.py](app/triage_service.py#L452-L510) | 缺I/O、citations_used落库与回放接口 | SQLite TraceStore + `/v1/admin/trace/{trace_id}` | P0 |
| M3：API契约（/v1/chat/send等） | 缺失 | 现为`/v1/chat`返回结构 [app/api_server.py](app/api_server.py#L1265-L1292) | 不匹配目标字段 | 新增`/v1/chat/send`并迁移前端 | P0 |
| M3：`/v1/kb/ingest` | 缺失 | 未见路由；仅离线脚本 [app/rag/ingest_kb.py](app/rag/ingest_kb.py#L960-L987) | 入库无法服务化且不走normalized | 增加FastAPI入库接口消费normalized | P0 |
| M3：日志脱敏 | 已具备 | 默认只落sha256 meta [app/api_server.py](app/api_server.py#L1184-L1292) | 符合脱敏方向 | 复用策略到trace落库 | P0 |
| M4：对话+证据展示 | 已具备 | evidence/trace渲染 [frontend/src/App.tsx](frontend/src/App.tsx#L470-L560) | 已有基础展示 | 保留 | P0 |
| M4：risk提示+follow_up+trace回放入口 | 缺失 | UI无risk/follow_up [frontend/src/App.tsx](frontend/src/App.tsx#L360-L560) | 不满足目标UI | 增加risk banner、follow-up卡片、trace回放入口 | P0 |
| M5：MedDG评测runner与报告 | 缺失 | 未见meddg实现（关键词见上） | 无评测闭环 | 新增runner与reports(json/csv) | P1 |

---

## D. 三天MVP改造任务清单（按模块拆：每条含验收标准）

> P0=三天MVP必须；P1=加分项；P2=未来扩展。

### Agent（P0）

1) 新增`Planner/Inquiry/Retrieve/Safety/Doctor/Memory`节点并在图里串联
	 - 验收标准：`GET /v1/admin/trace/{trace_id}`回放至少包含上述6类节点名（可最小实现，Planner/Memory允许先做stub）。
	 - 现状对照证据：当前节点链路为`preprocess/inquiry/.../format`，[app/api_server.py](app/api_server.py#L939-L969)

2) 节点级trace落库（含input/output摘要 + latency + citations_used）
	 - 验收标准：回放结果每节点都有`latency_ms`；Safety/Doctor节点回放包含`citations_used`。
	 - 现状对照证据：trace仅`{step,status,ms,...}`，[app/triage_service.py](app/triage_service.py#L452-L510)

### RAG（P0）

1) 替换Embedding为BCEmbedding（bce-embedding-base_v1）并保证入库/检索一致
	 - 验收标准：入库与检索使用同一模型名；requirements新增对应依赖；本地可成功检索。
	 - 现状对照证据：当前E5-small，[app/rag/retriever.py](app/rag/retriever.py#L12-L25)

2) 明确citations契约（API字段级）
	 - 验收标准：`/v1/chat/send`响应包含`citations: []`且可追溯到`chunk_id/source/page/score`。
	 - 现状对照证据：当前仅`evidence`列表与`citations_used`，[app/triage_protocol.py](app/triage_protocol.py#L18-L55)

### 后端（P0）

1) 新增`POST /v1/chat/send`（或将`/v1/chat`迁移并保留兼容层）
	 - 验收标准：响应字段严格为`reply,follow_up_questions,risk_level,citations,trace_id,memory_write`。
	 - 现状对照证据：当前`/v1/chat`返回结构 [app/api_server.py](app/api_server.py#L1265-L1292)

2) 新增`POST /v1/kb/ingest`（从normalized入库）
	 - 验收标准：调用接口后Chroma持久化目录更新，且入库来源仅来自`normalized/`。
	 - 现状对照证据：当前仅离线脚本main [app/rag/ingest_kb.py](app/rag/ingest_kb.py#L960-L987)

3) 新增`GET /v1/admin/trace/{trace_id}`（trace回放）
	 - 验收标准：按trace_id返回节点日志数组（含latency与citations_used）。
	 - 现状对照证据：当前无该路由 [app/api_server.py](app/api_server.py#L1010-L1292)

### 前端（P0）

1) 对接`/v1/chat/send`并新增：risk提示、follow_up卡片、trace回放入口
	 - 验收标准：
		 - 出现risk banner（高风险/红旗症状时提示线下就医/急救）
		 - follow_up_questions以卡片展示并可点击触发输入
		 - trace_id可点击并拉取回放接口展示节点回放
	 - 现状对照证据：当前仅调用`/v1/chat` [frontend/src/App.tsx](frontend/src/App.tsx#L313-L350)

### 评测（P1）

1) 新增MedDG评测runner（不入库）+ reports(json/csv) + 红旗用例集
	 - 验收标准：运行`eval/runner_meddg.py`产出`reports/*.json`与`reports/*.csv`，且包含红旗用例集结果。
	 - 现状对照证据：未发现meddg实现（关键词见B/M5）。

---

## E. 高风险问题与必修复项（含证据）

1) 数据质量：CSV读取`errors="replace"`可能将乱码写入向量库（违反“严禁乱码进入Chroma”的目标约束）
	 - 证据：[app/rag/ingest_kb.py](app/rag/ingest_kb.py#L300-L360)

2) 缺失“DataGateway强制规范化（NFKC/去\x00/控制字符/UTF-8/jsonl）”导致无法保证入库一致性
	 - 证据：未发现`unicodedata/NFKC`实现；入库直接读取原始文件 [app/rag/ingest_kb.py](app/rag/ingest_kb.py#L560-L740)

3) 缺失trace回放接口与节点I/O记录，难以满足审计/可观测目标
	 - 证据：仅有`/health,/v1/triage,/v1/chat` [app/api_server.py](app/api_server.py#L1010-L1292)

4) License/数据许可风险未落地
	 - 证据：仓库未发现`LICENSE*`文件（工作区文件扫描结果为未发现）。

---

## F. 建议的下一步提交顺序（模块顺序）

M0 DataGateway → M1 RAGService → M2 AgentOrchestrator+TraceStore → M3 BackendAPI → M4 FrontendUI → M5 Evaluation

