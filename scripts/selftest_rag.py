# -*- coding: utf-8 -*-
"""selftest_rag.py

命令行快速自测 RAGService：
- 打印 device / collection count / 模型信息
- 打印前 3 条证据的 eid/score/rerank_score/前 50 字

用法：
  python scripts/selftest_rag.py
  python scripts/selftest_rag.py "咳嗽发热怎么办"

注意：
- 该脚本不涉及会话，不会读取/输出用户历史对话。
"""

from __future__ import annotations

from pathlib import Path
import sys


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    query = "咳嗽发热怎么办"
    if len(sys.argv) >= 2:
        query = (sys.argv[1] or "").strip() or query

    try:
        from app.rag.rag_core import get_stats, retrieve

        st = get_stats()
        print("[RAG] collection=", st.collection)
        print("[RAG] count=", st.count)
        print("[RAG] persist_dir=", st.persist_dir)
        print("[RAG] device=", st.device)
        print("[RAG] embed_model=", st.embed_model)
        print("[RAG] rerank_model=", st.rerank_model)
        if st.updated_at:
            print("[RAG] updated_at=", st.updated_at)

        ev = retrieve(query, top_k=3)
        print("[RAG] query=", query)
        print("[RAG] hits=", len(ev))
        for e in ev[:3]:
            text = str(e.get("text") or "").strip().replace("\n", " ")
            if len(text) > 50:
                text = text[:50] + "…"
            print(
                "- ",
                e.get("eid"),
                "score=", e.get("score"),
                "rerank=", e.get("rerank_score"),
                "source=", e.get("source"),
                "text=", text,
            )
        return 0
    except Exception as e:
        print("[RAG] 自测失败：", type(e).__name__, str(e))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
