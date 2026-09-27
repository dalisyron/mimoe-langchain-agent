"""Conversation history for the web UI.

``serve`` keeps every conversation in one SQLite file: LangGraph's ``AsyncSqliteSaver`` stores the
messages (the checkpoints), and :class:`ConversationStore` adds the little the sidebar needs to
list them without loading them: a title and when each conversation was started and last used.
The default file is in the per-user data directory (:func:`default_history_path`);
``--history off`` keeps both in memory for the life of the server.

:func:`transcript` turns a thread's checkpointed messages back into the turns the web UI draws,
so a reopened conversation looks as it did while it streamed.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mimoe_agent.stream import ToolCallIds

if TYPE_CHECKING:
    import aiosqlite

TITLE_CHARS = 60
"""A new conversation's title: its first message, cut to this many characters."""
MAX_TITLE_CHARS = 200
RUN_ENDED = "no result: the run ended before the tool reported back"
"""Result of a tool call that never answered; ``web/src/reducer.ts`` uses the same text."""
REJECTED_PREFIX = "User rejected the tool call"
"""How ``HumanInTheLoopMiddleware`` begins the result of a declined call."""

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    thread_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
)
"""


def default_history_path() -> Path:
    """The per-user conversations file: ``%LOCALAPPDATA%`` on Windows, ``~/Library/Application
    Support`` on macOS, ``$XDG_DATA_HOME`` (``~/.local/share``) elsewhere."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "mimoe-agent" / "conversations.sqlite"


def prepare_history_file(path: Path) -> None:
    """Create ``path`` (and its folder) readable by the user only: conversations hold file
    contents and tool output. Existing files keep their permissions."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not path.exists():
        path.touch(mode=0o600)


def title_from(message: str) -> str:
    """The title of a conversation that starts with ``message``: one line, at most
    :data:`TITLE_CHARS` characters."""
    text = " ".join(message.split())
    if len(text) <= TITLE_CHARS:
        return text or "New conversation"
    return text[: TITLE_CHARS - 1].rstrip() + "…"


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Conversation:
    """One row of the sidebar."""

    id: str
    title: str
    created_at: float
    updated_at: float

    def summary(self) -> dict[str, Any]:
        """The JSON the web UI gets (times as ISO 8601 in UTC)."""
        return {
            "id": self.id,
            "title": self.title,
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
        }


class ConversationStore:
    """Titles and times of the web UI's conversations: in SQLite when there is a file, in a dict
    otherwise (no database thread to outlive the server, which matters for ``--history off`` and
    for tests).

    The SQLite connection opens on first use, from the event loop that uses it, and must be
    closed (:meth:`close`): an open ``aiosqlite`` connection keeps its worker thread, and with it
    the process, alive.
    """

    def __init__(self, path: Path | None) -> None:
        """Args:
        path: The SQLite file (shared with the checkpoints), or ``None`` for memory only.
        """
        self.path = path
        self._rows: dict[str, Conversation] = {}
        self._conn: aiosqlite.Connection | None = None
        self._opening = asyncio.Lock()
        self._last = 0.0

    async def _db(self) -> aiosqlite.Connection:
        assert self.path is not None
        if self._conn is not None:
            return self._conn
        async with self._opening:
            if self._conn is None:
                import aiosqlite

                prepare_history_file(self.path)
                conn = await aiosqlite.connect(self.path)
                await conn.execute(_SCHEMA)
                await conn.commit()
                self._conn = conn
        return self._conn

    async def close(self) -> None:
        if self._conn is not None:
            conn, self._conn = self._conn, None
            await conn.close()

    async def touch(self, thread_id: str, first_message: str | None = None) -> Conversation:
        """Record that ``thread_id`` was used now; a new conversation takes its title from
        ``first_message``."""
        # strictly increasing, so the order holds where the clock is coarse (15 ms on Windows)
        now = self._last = max(time.time(), self._last + 0.001)
        title = title_from(first_message or "")
        if self.path is None:
            old = self._rows.get(thread_id)
            self._rows[thread_id] = Conversation(
                thread_id, old.title if old else title, old.created_at if old else now, now
            )
            return self._rows[thread_id]
        db = await self._db()
        await db.execute(
            "INSERT INTO conversations (thread_id, title, created_at, updated_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(thread_id) DO UPDATE SET updated_at = excluded.updated_at",
            (thread_id, title, now, now),
        )
        await db.commit()
        conversation = await self.get(thread_id)
        assert conversation is not None
        return conversation

    async def list(self) -> list[Conversation]:
        """Every conversation, the most recently used first."""
        if self.path is None:
            return sorted(
                self._rows.values(), key=lambda c: (c.updated_at, c.created_at), reverse=True
            )
        db = await self._db()
        async with db.execute(
            "SELECT thread_id, title, created_at, updated_at FROM conversations "
            "ORDER BY updated_at DESC, created_at DESC"
        ) as cursor:
            return [Conversation(*row) async for row in cursor]

    async def get(self, thread_id: str) -> Conversation | None:
        if self.path is None:
            return self._rows.get(thread_id)
        db = await self._db()
        async with db.execute(
            "SELECT thread_id, title, created_at, updated_at FROM conversations "
            "WHERE thread_id = ?",
            (thread_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return Conversation(*row) if row is not None else None

    async def rename(self, thread_id: str, title: str) -> Conversation | None:
        """Set the title (one line, at most :data:`MAX_TITLE_CHARS`); ``None`` if unknown."""
        clean = " ".join(title.split())[:MAX_TITLE_CHARS]
        if self.path is None:
            old = self._rows.get(thread_id)
            if old is None:
                return None
            self._rows[thread_id] = Conversation(thread_id, clean, old.created_at, old.updated_at)
            return self._rows[thread_id]
        db = await self._db()
        await db.execute(
            "UPDATE conversations SET title = ? WHERE thread_id = ?", (clean, thread_id)
        )
        await db.commit()
        return await self.get(thread_id)

    async def delete(self, thread_id: str) -> None:
        if self.path is None:
            self._rows.pop(thread_id, None)
            return
        db = await self._db()
        await db.execute("DELETE FROM conversations WHERE thread_id = ?", (thread_id,))
        await db.commit()


# -- transcript ------------------------------------------------------------------------------------


def _text(content: Any) -> str:
    """The text of message content, a string or a list of blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        return "".join(
            str(block.get("text", "")) if isinstance(block, Mapping) else str(block)
            for block in content
        )
    return str(content or "")


def _injected(message: Any) -> bool:
    """A message a middleware put in the thread (the model-call limit), not a model reply: the
    stream shows those as notices. Model replies carry response metadata and usage."""
    return (
        not message.tool_calls and message.usage_metadata is None and not message.response_metadata
    )


def transcript(
    messages: Sequence[Any], pending: Mapping[str, Any] | None = None
) -> tuple[list[dict[str, Any]], ToolCallIds]:
    """The turns the web UI draws for a thread's checkpointed ``messages``.

    Mirrors the live stream: a user turn per ``HumanMessage``, and an assistant turn holding, per
    model reply, its reasoning, its text and its tool calls, with tool-call ids aliased per turn
    exactly as :class:`mimoe_agent.stream.ToolCallIds` does while streaming. A tool call without a
    result is ``awaiting_approval`` when ``pending`` (the thread's interrupt value) asks about it,
    ``running`` when it waits behind that approval, and otherwise ended without a result.

    Args:
        messages: ``HumanMessage``/``AIMessage``/``ToolMessage`` objects in thread order.
        pending: The value of the interrupt the thread is waiting on, if any.

    Returns:
        The turns, and the tool-call aliases of the last turn (a resume continues them).
    """
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    turns: list[dict[str, Any]] = []
    ids = ToolCallIds()
    turn: dict[str, Any] | None = None
    calls: dict[str, dict[str, Any]] = {}  # raw tool-call id -> its block in this turn

    def assistant() -> dict[str, Any]:
        nonlocal turn
        if turn is None:
            turn = {"role": "assistant", "blocks": []}
            turns.append(turn)
        return turn

    for message in messages:
        if isinstance(message, HumanMessage):
            turns.append({"role": "user", "text": _text(message.content)})
            turn, ids, calls = None, ToolCallIds(), {}
        elif isinstance(message, AIMessage):
            blocks = assistant()["blocks"]
            text = _text(message.content)
            if _injected(message):
                if text.strip():
                    blocks.append({"kind": "notice", "text": text.strip()})
                continue
            reasoning = message.additional_kwargs.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning.strip():
                blocks.append({"kind": "thinking", "text": reasoning.strip()})
            if text.strip():
                blocks.append({"kind": "text", "text": text})
            for call in message.tool_calls:
                raw = str(call.get("id") or f"call_{len(calls)}")
                block = {
                    "kind": "tool",
                    "id": ids.announce(raw),
                    "name": call.get("name") or "tool",
                    "args": call.get("args") or {},
                    "status": "error",
                    "result": RUN_ENDED,
                }
                blocks.append(block)
                calls[raw] = block
            _add_stats(assistant(), message)
        elif isinstance(message, ToolMessage):
            raw = str(message.tool_call_id)
            block = calls.get(raw)
            if block is None:  # a result without its call: show it rather than lose it
                block = {"kind": "tool", "id": ids.resolve(raw), "name": message.name or "tool"}
                block["args"] = {}
                assistant()["blocks"].append(block)
                calls[raw] = block
            content = _text(message.content)
            block["result"] = content
            if message.status != "error":
                block["status"] = "done"
            else:
                block["status"] = "denied" if content.startswith(REJECTED_PREFIX) else "error"
    if pending is not None and turn is not None:
        _mark_pending(turn, pending)
    return turns, ids


def _add_stats(turn: dict[str, Any], message: Any) -> None:
    """Add one model reply to the turn's stats (model calls, tokens, the model that spoke)."""
    stats = turn.setdefault(
        "stats", {"model": None, "usage": {"input_tokens": 0, "output_tokens": 0, "llm_calls": 0}}
    )
    usage = message.usage_metadata or {}
    stats["usage"]["llm_calls"] += 1
    stats["usage"]["input_tokens"] += int(usage.get("input_tokens") or 0)
    stats["usage"]["output_tokens"] += int(usage.get("output_tokens") or 0)
    model = message.response_metadata.get("model_name") or message.response_metadata.get("model")
    if model:
        stats["model"] = str(model)


def _mark_pending(turn: dict[str, Any], pending: Mapping[str, Any]) -> None:
    """Calls without a result: the k-th one of a name asked about is ``awaiting_approval``
    (the order ``HumanInTheLoopMiddleware`` keeps), the rest wait behind it."""
    asked = [
        str(r.get("name")) for r in pending.get("action_requests") or [] if isinstance(r, Mapping)
    ]
    for block in turn["blocks"]:
        if block.get("kind") != "tool" or block.get("result") != RUN_ENDED:
            continue
        block.pop("result")
        if block["name"] in asked:
            asked.remove(block["name"])
            block["status"] = "awaiting_approval"
        else:
            block["status"] = "running"
