# -*- coding: utf-8 -*-
"""selftest_agent_v2.py

命令行自测 M2：/v1/agent/chat_v2

用法：
  python scripts/selftest_agent_v2.py

说明：
- 该脚本通过 HTTP 调用 FastAPI 服务，因此需要你先启动：
  uvicorn app.api_server:app --reload
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict

import requests


BASE = "http://127.0.0.1:8000"


def _post(payload: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(f"{BASE}/v1/agent/chat_v2", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def main() -> int:
    print("[M2] Step1: 第1轮（new session），头痛两天 -> 应 ask（结构化追问）")
    out1 = _post({"user_message": "我头疼两天了怎么办"})
    print(json.dumps(out1, ensure_ascii=False, indent=2))
    sid = out1.get("session_id")
    if not sid:
        print("[M2] FAIL: 未返回 session_id")
        return 1
    if out1.get("mode") != "ask":
        print("[M2] FAIL: Step1 期望 mode=ask")
        return 1
    if not (out1.get("ask_text") or "").strip():
        print("[M2] FAIL: Step1 期望 ask_text 非空")
        return 1
    if not (out1.get("questions") or out1.get("next_questions")):
        print("[M2] FAIL: Step1 期望 questions 或 next_questions 非空")
        return 1
    tr1 = out1.get("trace") or {}
    if not (tr1.get("node_order") or []):
        print("[M2] FAIL: Step1 期望 trace.node_order 非空")
        return 1
    if not isinstance(tr1.get("timings_ms"), dict):
        print("[M2] FAIL: Step1 期望 trace.timings_ms 为 dict")
        return 1

    print("\n[M2] Step2: 第2轮（same session），补充年龄/性别/疼痛分 -> 仍可能 ask")
    out2 = _post({"session_id": sid, "user_message": "我24岁男，疼痛程度6/10", "top_k": 3, "top_n": 30, "use_rerank": True})
    print(json.dumps(out2, ensure_ascii=False, indent=2))
    if out2.get("session_id") != sid:
        print("[M2] FAIL: Step2 session_id 未延续")
        return 1
    if out2.get("mode") != "ask":
        print("[M2] FAIL: Step2 期望 mode=ask")
        return 1
    if not (out2.get("questions") or out2.get("next_questions")):
        print("[M2] FAIL: Step2 期望 questions 或 next_questions 非空")
        return 1

    print("\n[M2] Step3: 第3轮（same session），补充是否发烧 -> 应进入 answer")
    out3 = _post({"session_id": sid, "user_message": "没有发烧", "top_k": 3, "top_n": 30, "use_rerank": True})
    print(json.dumps(out3, ensure_ascii=False, indent=2))
    if out3.get("session_id") != sid:
        print("[M2] FAIL: Step3 session_id 未延续")
        return 1
    if out3.get("mode") != "answer" or not (out3.get("answer") or "").strip():
        print("[M2] FAIL: Step3 期望 mode=answer 且 answer 非空")
        return 1

    tr3 = out3.get("trace") or {}
    if not (tr3.get("node_order") or []):
        print("[M2] FAIL: Step3 期望 trace.node_order 非空")
        return 1
    if not isinstance(tr3.get("timings_ms"), dict):
        print("[M2] FAIL: Step3 期望 trace.timings_ms 为 dict")
        return 1

    rag_stats = tr3.get("rag_stats") or {}
    if isinstance(rag_stats, dict):
        hits = int(rag_stats.get("hits") or 0)
        citations = out3.get("citations") or []
        if hits > 0 and not citations:
            print("[M2] FAIL: Step3 期望 hits>0 时 citations 非空")
            return 1

    print("\n[M2] Step4: 第4轮（same session），最剧烈头痛 + 呕吐 -> 应 escalate")
    out4 = _post({"session_id": sid, "user_message": "这是我经历过最剧烈的头痛，而且一直在呕吐"})
    print(json.dumps(out4, ensure_ascii=False, indent=2))
    if out4.get("mode") != "escalate" or not (out4.get("answer") or "").strip():
        print("[M2] FAIL: Step4 期望 mode=escalate 且 answer 非空")
        return 1

    tr4 = out4.get("trace") or {}
    order4 = tr4.get("node_order") or []
    must = {"SafetyGate", "MemoryUpdate", "TriagePlanner", "PersistState"}
    if not must.issubset(set(order4)):
        print(f"[M2] FAIL: Step4 trace.node_order 缺少关键节点：{must}，实际={order4}")
        return 1

    print("\n[M2] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
