# -*- coding: utf-8 -*-
"""
本地RAG检索器：从rag/kb_store(Chroma持久化库)中召回证据chunk。
与rag/ingest_kb.py保持同一路径与collection默认值。
"""

from __future__ import annotations

import os
import platform
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

DEFAULT_COLLECTION = "medical_kb"
# Default to multilingual embeddings.
# NOTE: Must match the embedding model used during ingest (see rag/ingest_kb.py).
DEFAULT_EMBED_MODEL = "intfloat/multilingual-e5-small"


def _apply_windows_openmp_workaround() -> None:
    """Avoid common OpenMP runtime conflicts on Windows.

    Some ML stacks on Windows may load multiple OpenMP runtimes (libiomp5md.dll),
    causing a hard crash. For classroom demo stability, we enable the documented
    workaround unless the user explicitly configured it.
    """

    try:
        if platform.system().lower() != "windows":
            return
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    except Exception:
        return


def _get_persist_dir() -> Path:
    # retriever.py位于app/rag/retriever.py
    # parents[0]=rag, parents[1]=app
    base_dir = Path(__file__).resolve().parents[1]
    return base_dir / "rag/kb_store"


@lru_cache(maxsize=1)
def _get_vectorstore() -> Chroma:
    _apply_windows_openmp_workaround()

    persist_dir = _get_persist_dir()
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"找不到Chroma持久化目录：{persist_dir}。请先运行rag/ingest_kb.py完成入库。"
        )

    collection_name = (os.getenv("RAG_COLLECTION") or "").strip() or DEFAULT_COLLECTION
    model_name = (os.getenv("HF_EMBEDDING_MODEL") or "").strip() or DEFAULT_EMBED_MODEL

    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vs = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    # 可选调试：RAG_DEBUG=1时打印count，帮助排查“连空库”
    if (os.getenv("RAG_DEBUG") or "").strip() == "1":
        try:
            print(f"[RAG_DEBUG]persist_dir={persist_dir}")
            print(f"[RAG_DEBUG]collection={collection_name}, count={vs._collection.count()}")
        except Exception:
            pass

    return vs


def retrieve(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    返回证据列表，每条包含：
    eid(E1..)、source、page、section、chunk_id、score、text
    """
    query = (query or "").strip()
    if not query:
        return []

    vs = _get_vectorstore()
    k = int(top_k) if top_k and int(top_k) > 0 else 5

    # 关键：不做阈值过滤，先保证一定返回top_k
    docs_scores = vs.similarity_search_with_score(query, k=k)

    out: List[Dict[str, Any]] = []
    for i, (doc, score) in enumerate(docs_scores, start=1):
        md = dict(doc.metadata or {})
        out.append(
            {
                "eid": f"E{i}",
                "source": md.get("source", ""),
                "page": md.get("page", None),
                "section": md.get("section", ""),
                "chunk_id": md.get("chunk_id", ""),
                "score": float(score) if score is not None else None,
                "text": (doc.page_content or "").strip(),
            }
        )
    return out
