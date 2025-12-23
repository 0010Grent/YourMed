# -*- coding: utf-8 -*-
"""M2：基于 LangGraph 的多轮问诊 Agent 编排模块。

设计目标：
- 最小侵入式：不破坏既有 /v1/triage 与 /v1/chat。
- 前后端分离友好：新增 /v1/agent/chat_v2，前端只需传 session_id 即可多轮。
- 可复现可验收：提供 SQLite 持久化、trace 输出与自测脚本。
"""
