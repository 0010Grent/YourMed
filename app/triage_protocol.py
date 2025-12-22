# -*- coding: utf-8 -*-
"""triage_protocol.py

统一“分诊输出协议”（线上API与离线跑批共用）。

协议目标：
- answer: 结构化分诊JSON（包含 triage_level/red_flags/immediate_actions/.../citations_used）
- evidence: RAG召回证据列表（包含 eid/source/page/chunk_id/text 等字段）
- rag_query: 本次检索使用的 query
- meta: 运行元信息（mode/created_at 等）
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional


def now_str(dt: Optional[datetime] = None) -> str:
    d = dt or datetime.now()
    return d.strftime("%Y-%m-%d %H:%M:%S")


def build_triage_payload(
    *,
    answer: Dict[str, Any],
    evidence: List[Dict[str, Any]],
    rag_query: str,
    mode: str,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the canonical triage output payload."""

    payload: Dict[str, Any] = {
        "answer": answer if isinstance(answer, dict) else {},
        "evidence": evidence if isinstance(evidence, list) else [],
        "rag_query": (rag_query or "").strip(),
        "meta": {
            "mode": (mode or "").strip() or "fast",
            "created_at": (created_at or "").strip() or now_str(),
        },
    }

    return payload
