#RAG入库改造对比表格版(旧项目vs当前项目)


##1.总体结论(一句话版)
旧项目存在“向量库规模异常小(例如count=81)”与“入库可观测性不足”的问题；当前项目完成BCEmbedding+GPU入库、白名单数据隔离、流式批量入库与可验收日志链路，抽样入库后collectioncount提升至35654并可稳定检索返回证据。

---

##2.核心能力对比表

|维度|旧项目(改造前)|当前项目(改造后)|收益/影响|验收证据(示例)|
|---|---|---|---|---|
|数据源组织|原始数据与评测/说明混放风险较高|白名单目录入库：`dataset-v2/合并数据-CSV格式/*.csv`；评测集`MedDG_UTF8`独立隔离|避免评测集污染KB，便于复现与上线治理|入库日志显示`kb_dir=...合并数据-CSV格式`且`files_csv=17 files_txt=0`|
|字段标准化|字段命名/格式可能不统一导致解析困难|统一为`department,title,ask,answer`；UTF-8编码|入库脚本可稳定解析，降低脏数据概率|抽样检查：`df.columns=['department','title','ask','answer']`|
|编码与文本清洗|缺少统一清洗策略，存在乱码/控制字符风险|统一清洗：NFKC、统一换行、去控制字符、去\x00；硬闸门拒绝含U+FFFD(�)文本|防“乱码污染数据库/向量库”，提升线上稳定性|bad_rows.log(如启用)；入库过程无Unicode报错|
|入库覆盖率|出现count极小(例如81)，与真实数据规模不匹配|显式支持`RAG_RESET=1`重建；支持抽样与全量模式|确保KB规模与数据规模一致，RAG命中率显著提升|抽样入库后：`count=35654 chunks_ingested=35654`|
|入库性能|CPU向量化为主，吞吐受限|BCEmbedding在GPU执行embedding；batch可调|缩短入库时间，三天MVP可在开发期反复重建KB|日志：`provider=bce device=cuda`；`nvidia-smi`可观察显存占用|
|大文件处理|可能一次性加载导致内存压力或不可控|流式/批量入库：逐行→batch写入→定期persist；支持progress打印|支持25万行级CSV持续入库，过程可观测可恢复|日志：`[PROGRESS] last_row=2000/4000...`、`[PERSIST]`|
|可观测性(Logs)|缺少关键参数打印，难定位扫到哪些文件/入了多少|启动打印provider/model/device/kb_dir/persist_dir/collection；过程打印文件数、batch、进度；结束打印count|问题定位速度显著提升，适合企业交付|`[RAG_INGEST] provider=... device=... files_csv=... count=...`|
|检索一致性|入库embedding与检索embedding不一致风险|retriever与ingest统一embeddingprovider/model/device参数|避免“入库一套、检索一套”导致检索失效|`RAG_DEBUG`打印`device=cuda count=... hits=...`|
|运行门槛|依赖与环境提示不足，容易踩坑|提供README_GPU_INGEST.md：安装torch-cuda、开启GPU入库、验证GPU步骤|降低接手成本，利于团队协作与答辩展示|README中包含PowerShell与bash命令+验收步骤|
|风险控制|缺少隔离与闸门，后续易被“顺手放文件”污染|目录白名单+txt默认禁入库+硬闸门+可追溯日志|降低长期维护成本，利于后续评测与迭代|入库仅统计CSV文件，评测pk不参与|

---

##3.关键指标快照(可贴PPT)
-数据目录：`app/rag/kb_docs/dataset-v2/合并数据-CSV格式`
-科室CSV数量：17
-字段：`department,title,ask,answer`
-内科数据规模示例：`(259752,4)`
-抽样入库策略：`RAG_CSV_MAX_ROWS=5000`
-抽样入库结果：`collection=medical_kb,count=35654,chunks_ingested=35654`
-GPU执行证据：入库日志`device=cuda`；BCEmbedding日志`Execute device:cuda`；`nvidia-smi`可观测python进程占显存

---

##4.已识别风险与缓解策略(企业风控写法)

|风险|表现|影响|缓解策略|
|---|---|---|---|
|旧库残留导致误判|count异常偏小(如81)|误以为入库失败/检索差|固定流程：入库前`RAG_RESET=1`重建；日志打印`initial_count`与最终count|
|显存不足|CUDAoutofmemory|入库中断|降低`RAG_INGEST_BATCH_SIZE`(512→256→128)，或临时切`device=cpu`应急|
|数据污染|评测集/说明文件误入库|检索证据不可信、评测失真|白名单目录入库；默认禁用txt；评测目录独立维护|
|模型下载/网络不稳定|首次加载模型慢或失败|影响演示|提前离线缓存HF模型；仓库文档写清楚镜像/代理策略(可选)|
|文本异常字符|出现\x00或U+FFFD|向量库写入失败或检索质量下降|清洗+硬闸门+bad_rows.log追踪|

---

##5.后续可选增强(不改架构也能显著提升体验)
-证据可解释：回答强制引用E1/E2/E3并前端可展开snippet
-科室路由检索：先按department过滤检索，召回不足再回退全库
-评测闭环：用MedDG回放多轮对话，输出reports/eval_summary.json与cases.csv
