# -*- coding: utf-8 -*-
"""storage_sqlite.py

最小可用的 SQLite 会话存储。

表结构：
- sessions(session_id PRIMARY KEY, state_json TEXT, updated_at)

设计说明：
- 使用 stdlib sqlite3，避免额外依赖。
- 线程安全：写入使用进程内锁；使用 WAL 提升并发读写能力。
- 容错：读写失败会抛 RuntimeError，便于定位。
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from app.agent.state import AgentSessionState, utc_now_iso


def _find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "README.md").exists() and (p / "app").is_dir():
            return p
    return start.parent


def default_db_path() -> Path:
    """默认存储位置：<repo>/app/data/agent_sessions.sqlite3"""

    repo_root = _find_repo_root(Path(__file__))
    data_dir = (repo_root / "app" / "data")
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "agent_sessions.sqlite3"


class SqliteSessionStore:
    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = (db_path or default_db_path()).resolve()
        self._lock = threading.Lock()
        self._init_db()

    @property
    def db_path(self) -> str:
        return str(self._db_path)

    def _connect(self) -> sqlite3.Connection:
        # check_same_thread=False 允许跨线程使用；我们用锁控制写入。
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                      session_id TEXT PRIMARY KEY,
                      state_json TEXT NOT NULL,
                      updated_at TEXT NOT NULL
                    )
                    """
                )
        except Exception as e:
            raise RuntimeError(f"初始化 SQLite 失败：{type(e).__name__}: {e}") from e

    def load_session(self, session_id: str) -> Optional[AgentSessionState]:
        sid = (session_id or "").strip()
        if not sid:
            return None

        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "SELECT state_json FROM sessions WHERE session_id = ?",
                    (sid,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                raw = row[0]
        except Exception as e:
            raise RuntimeError(f"读取 session 失败：{type(e).__name__}: {e}") from e

        try:
            # 兼容：state_json 可能不是严格 JSON 字符串
            if isinstance(raw, str):
                return AgentSessionState.model_validate_json(raw)
            return AgentSessionState.model_validate(raw)
        except Exception as e:
            raise RuntimeError(f"解析 session_json 失败：{type(e).__name__}: {e}") from e

    def save_session(self, state: AgentSessionState) -> None:
        if not isinstance(state, AgentSessionState):
            raise RuntimeError("save_session 入参不是 AgentSessionState")

        # 只保留最近 20 轮
        state.trim_messages(max_turns=20)
        state.last_update_ts = utc_now_iso()

        payload = state.model_dump()
        state_json = json.dumps(payload, ensure_ascii=False)

        with self._lock:
            try:
                with self._connect() as conn:
                    conn.execute(
                        """
                        INSERT INTO sessions(session_id, state_json, updated_at)
                        VALUES(?, ?, ?)
                        ON CONFLICT(session_id) DO UPDATE SET
                          state_json=excluded.state_json,
                          updated_at=excluded.updated_at
                        """,
                        (state.session_id, state_json, state.last_update_ts),
                    )
                    conn.commit()
            except Exception as e:
                raise RuntimeError(f"保存 session 失败：{type(e).__name__}: {e}") from e

    def delete_session(self, session_id: str) -> None:
        sid = (session_id or "").strip()
        if not sid:
            return
        with self._lock:
            try:
                with self._connect() as conn:
                    conn.execute("DELETE FROM sessions WHERE session_id = ?", (sid,))
                    conn.commit()
            except Exception as e:
                raise RuntimeError(f"删除 session 失败：{type(e).__name__}: {e}") from e
