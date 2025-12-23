# -*- coding: utf-8 -*-
"""router.py

FastAPI 路由：/v1/agent/chat_v2

设计：
- 最小侵入式：只新增路由，不影响 /v1/triage 与 /v1/chat。
- 前后端分离：前端每次带 session_id 即可延续会话。
- 不泄漏隐私：日志最多前 100 字符或 hash。
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.agent.graph import run_chat_v2_turn


logger = logging.getLogger(__name__)

router = APIRouter()


def _sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="replace")).hexdigest()


def _safe_for_log(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return "(empty)"
    prefix = s[:100]
    if len(s) <= 100:
        return prefix
    return f"{prefix}…(sha256={_sha256_text(s)[:12]})"


class AgentChatV2Request(BaseModel):
    session_id: Optional[str] = Field(default=None, description="可选，不传则后端生成")
    user_message: str = Field(..., description="用户输入")
    top_k: int = Field(5, ge=1, le=20, description="最终返回证据条数")
    top_n: int = Field(30, ge=1, le=200, description="第一阶段召回条数")
    use_rerank: bool = Field(True, description="是否启用 rerank")


class AgentChatV2Response(BaseModel):
    session_id: str
    mode: str
    ask_text: str = ""
    questions: List[Dict[str, Any]] = Field(default_factory=list)
    next_questions: List[str]
    answer: str
    citations: List[Dict[str, Any]]
    slots: Dict[str, Any]
    summary: str
    trace: Dict[str, Any]


@router.post("/v1/agent/chat_v2", response_model=AgentChatV2Response)
def agent_chat_v2(req: AgentChatV2Request, request: Request) -> Dict[str, Any]:
    # 不打印完整 user_message
    logger.info(
        "AGENT_CHAT_V2 start session_id=%s msg=%s",
        (req.session_id or "(new)"),
        _safe_for_log(req.user_message),
    )

    out = run_chat_v2_turn(
        session_id=req.session_id,
        user_message=req.user_message,
        top_k=req.top_k,
        top_n=req.top_n,
        use_rerank=req.use_rerank,
    )

    return out
