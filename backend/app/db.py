"""SQLite session storage (standard-library sqlite3, no external services).

Tables
------
messages  - the minimal data model from the assignment, plus an index.
sessions  - one row per session; holds the session's system-prompt override.

A chat exchange (user message + assistant reply) is written in ONE transaction
and only after the LLM call succeeded, so failed requests (422/503) never leave
a half-written turn in history.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id     TEXT PRIMARY KEY,
    system_prompt  TEXT,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    metadata    TEXT,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages (session_id, id);
"""


def now_iso() -> str:
    """Server time, ISO-8601, UTC."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def sqlite_path_from_url(database_url: str) -> str:
    """Turn `sqlite:////data/app.db` into `/data/app.db` and `sqlite:///./app.db` into `./app.db`.

    A bare path is accepted too. `:memory:` is supported for tests.
    """
    url = database_url.strip()
    if url.startswith("sqlite://"):
        path = url[len("sqlite://"):]
        # sqlite:///relative -> "/relative" -> "relative"; sqlite:////abs -> "//abs" -> "/abs"
        if path.startswith("/"):
            path = path[1:]
        return path or ":memory:"
    return url


class Database:
    def __init__(self, database_url: str) -> None:
        self.path = sqlite_path_from_url(database_url)
        if self.path != ":memory:":
            Path(self.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        # One shared connection guarded by a lock: simple, safe, and fast enough for this workload.
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(SCHEMA)

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def ping(self) -> bool:
        try:
            with self._lock:
                self._conn.execute("SELECT 1").fetchone()
            return True
        except sqlite3.Error:
            return False

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---- sessions ---------------------------------------------------------

    def session_exists(self, session_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return row is not None

    def get_system_prompt(self, session_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT system_prompt FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return row["system_prompt"] if row else None

    # ---- messages ---------------------------------------------------------

    def get_history(self, session_id: str) -> list[dict[str, Any]]:
        """All stored messages of a session, oldest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content, metadata, created_at FROM messages "
                "WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        history = []
        for row in rows:
            item: dict[str, Any] = {"role": row["role"], "content": row["content"]}
            if row["role"] == "assistant" and row["metadata"]:
                try:
                    item["metadata"] = json.loads(row["metadata"])
                except json.JSONDecodeError:
                    item["metadata"] = None
            item["created_at"] = row["created_at"]
            history.append(item)
        return history

    def save_exchange(
        self,
        session_id: str,
        user_message: str,
        assistant_reply: str,
        metadata: dict[str, Any],
        system_prompt: str | None,
    ) -> None:
        """Persist one successful turn atomically (creates the session on first use)."""
        user_ts = now_iso()
        with self._transaction() as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, system_prompt, created_at, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET "
                "system_prompt = COALESCE(excluded.system_prompt, sessions.system_prompt), "
                "updated_at = excluded.updated_at",
                (session_id, system_prompt, user_ts, user_ts),
            )
            conn.execute(
                "INSERT INTO messages (session_id, role, content, metadata, created_at) "
                "VALUES (?, 'user', ?, NULL, ?)",
                (session_id, user_message, user_ts),
            )
            conn.execute(
                "INSERT INTO messages (session_id, role, content, metadata, created_at) "
                "VALUES (?, 'assistant', ?, ?, ?)",
                (session_id, assistant_reply, json.dumps(metadata), now_iso()),
            )
