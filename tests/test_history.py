"""Conversation history: titles, the per-user file, the store in both modes, and the transcript
rebuilt from checkpointed messages (what the web UI draws when a conversation is reopened)."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from mimoe_agent import history
from mimoe_agent.config import Settings, history_path, load_settings
from mimoe_agent.history import (
    REJECTED_PREFIX,
    RUN_ENDED,
    ConversationStore,
    default_history_path,
    title_from,
    transcript,
)

POSIX = sys.platform != "win32"
USAGE = {"input_tokens": 100, "output_tokens": 7, "total_tokens": 107}
META = {"model_name": "qwen3-4b", "finish_reason": "stop"}


def reply(content: str = "", *calls: dict[str, Any], **extra: Any) -> AIMessage:
    """A model reply as the checkpoint holds it (usage and response metadata included)."""
    return AIMessage(
        content=content,
        tool_calls=[{"type": "tool_call", **call} for call in calls],
        usage_metadata=USAGE,
        response_metadata=META,
        **extra,
    )


def call(name: str, raw_id: str = "tool_0", **args: Any) -> dict[str, Any]:
    return {"name": name, "args": args, "id": raw_id}


# -- titles and files ----------------------------------------------------------------------------


def test_title_is_the_first_message_on_one_line_and_cut() -> None:
    assert title_from("  What is the sum\n of 10 to 20?  ") == "What is the sum of 10 to 20?"
    long = title_from("word " * 40)
    assert len(long) == history.TITLE_CHARS and long.endswith("…")
    assert title_from("   ") == "New conversation"


@pytest.mark.parametrize(
    ("platform", "env", "expected"),
    [
        ("win32", {"LOCALAPPDATA": "/L"}, Path("/L/mimoe-agent/conversations.sqlite")),
        (
            "darwin",
            {},
            Path.home() / "Library/Application Support/mimoe-agent/conversations.sqlite",
        ),
        ("linux", {"XDG_DATA_HOME": "/X"}, Path("/X/mimoe-agent/conversations.sqlite")),
        ("linux", {}, Path.home() / ".local/share/mimoe-agent/conversations.sqlite"),
    ],
)
def test_default_history_path_per_platform(
    monkeypatch: pytest.MonkeyPatch, platform: str, env: dict[str, str], expected: Path
) -> None:
    monkeypatch.setattr(history.sys, "platform", platform)
    for name in ("LOCALAPPDATA", "XDG_DATA_HOME"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert default_history_path() == expected


def test_history_setting(settings_tmp: Settings, tmp_path: Path) -> None:
    assert settings_tmp.history is None  # tests and the REPL never write a history file
    assert history_path(settings_tmp) == default_history_path()  # what `serve` uses
    for off in ("off", "OFF", "0", "false", "no"):
        assert history_path(dataclasses.replace(settings_tmp, history=off)) is None
    chosen = history_path(dataclasses.replace(settings_tmp, history=str(tmp_path / "c.db")))
    assert chosen == (tmp_path / "c.db").resolve()


def test_history_is_read_from_the_environment_but_not_from_dotenv(
    workspace_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (workspace_tmp.parent / ".env").write_text("MIMOE_HISTORY=/somewhere/else.db\n")
    settings = load_settings({"workspace": workspace_tmp}, cwd=workspace_tmp.parent)
    assert settings.history is None  # a .env next to the workspace cannot redirect it
    monkeypatch.setenv("MIMOE_HISTORY", "off")
    settings = load_settings({"workspace": workspace_tmp}, cwd=workspace_tmp.parent)
    assert settings.history == "off" and settings.sources["history"] == "env"


# -- the store -----------------------------------------------------------------------------------


@pytest.fixture(params=["memory", "sqlite"])
def store_path(request: pytest.FixtureRequest, tmp_path: Path) -> Path | None:
    return None if request.param == "memory" else tmp_path / "data" / "conversations.sqlite"


async def test_store_records_lists_renames_and_deletes(store_path: Path | None) -> None:
    store = ConversationStore(store_path)
    try:
        first = await store.touch("a", "Summarize notes.md")
        await store.touch("b", "What files are in this workspace?")
        again = await store.touch("a", "a later message does not rename it")
        assert again.title == "Summarize notes.md" and again.created_at == first.created_at
        assert again.updated_at >= first.updated_at
        assert [c.id for c in await store.list()] == ["a", "b"]  # most recently used first
        renamed = await store.rename("b", "  Files\n in   the workspace ")
        assert renamed is not None and renamed.title == "Files in the workspace"
        assert await store.rename("nope", "x") is None
        await store.delete("a")
        assert [c.id for c in await store.list()] == ["b"] and await store.get("a") is None
        summary = renamed.summary()
        assert set(summary) == {"id", "title", "created_at", "updated_at"}
        assert summary["created_at"].endswith("+00:00")
    finally:
        await store.close()


async def test_sqlite_store_survives_a_restart_and_is_private(tmp_path: Path) -> None:
    path = tmp_path / "data" / "conversations.sqlite"
    store = ConversationStore(path)
    await store.touch("a", "hello")
    await store.close()
    reopened = ConversationStore(path)
    try:
        assert [c.title for c in await reopened.list()] == ["hello"]
    finally:
        await reopened.close()
    if POSIX:  # conversations hold file contents and tool output
        assert (path.stat().st_mode & 0o777) == 0o600
        assert (path.parent.stat().st_mode & 0o777) == 0o700


def test_memory_store_opens_no_file(tmp_path: Path) -> None:
    ConversationStore(None)
    assert list(tmp_path.iterdir()) == []


# -- transcript ----------------------------------------------------------------------------------


def test_transcript_mirrors_the_stream() -> None:
    messages = [
        HumanMessage("What is the sum of 10 to 20?"),
        reply(
            "I'll compute it.",
            call("calculator", expression="sum_between(10, 20)"),  # a function it has not
            additional_kwargs={"reasoning_content": " The user wants a sum. "},
        ),
        ToolMessage(
            "ERROR: unknown function 'sum_between'; the calculator has Python's built-in "
            "functions such as sum, len, range, sorted, min, max, round and abs.",
            tool_call_id="tool_0",
            status="error",
        ),
        reply("", call("run_python", code="print(sum(range(10, 21)))")),
        ToolMessage("exit_code: 0\nstdout:\n165", tool_call_id="tool_0", name="run_python"),
        reply("The sum is 165."),
    ]
    turns, _ = transcript(messages)
    assert turns[0] == {"role": "user", "text": "What is the sum of 10 to 20?"}
    assistant = turns[1]
    kinds = [(b["kind"], b.get("status")) for b in assistant["blocks"]]
    assert kinds == [
        ("thinking", None),
        ("text", None),
        ("tool", "error"),
        ("tool", "done"),
        ("text", None),
    ]
    thinking, _, failed, python, answer = assistant["blocks"]
    assert thinking["text"] == "The user wants a sum."
    assert failed["id"] == "tool_0" and failed["result"].startswith("ERROR: unknown function")
    assert python["id"] == "tool_0#2"  # the engine's repeated id, aliased as the stream does
    assert python["args"] == {"code": "print(sum(range(10, 21)))"}
    assert "165" in python["result"] and answer["text"] == "The sum is 165."
    usage = {"input_tokens": 300, "output_tokens": 21, "llm_calls": 3}
    assert assistant["stats"] == {"model": "qwen3-4b", "usage": usage}


def test_tool_ids_restart_with_every_user_turn_and_notices_stay_notices() -> None:
    notice = AIMessage("Model call limits exceeded: thread limit (8/8)")  # injected, no metadata
    messages = [
        HumanMessage("one"),
        reply("", call("list_files", path=".")),
        ToolMessage("notes.md", tool_call_id="tool_0"),
        notice,
        HumanMessage("two"),
        reply("", call("list_files", path="src")),
        ToolMessage("app.py", tool_call_id="tool_0"),
        reply("Done."),
    ]
    turns, ids = transcript(messages)
    assert [b["kind"] for b in turns[1]["blocks"]] == ["tool", "notice"]
    assert turns[1]["blocks"][1]["text"].startswith("Model call limits exceeded")
    assert turns[3]["blocks"][0]["id"] == "tool_0"  # not tool_0#2: a new turn
    assert ids.resolve("tool_0") == "tool_0"


def test_declined_and_unanswered_calls() -> None:
    declined = f"{REJECTED_PREFIX} for `run_python` with reason: The user declined."
    messages = [
        HumanMessage("run it"),
        reply("", call("run_python", code="print(1)")),
        ToolMessage(declined, tool_call_id="tool_0", status="error"),
        reply("I did not run it."),
        HumanMessage("list"),
        reply("", call("list_files", path=".")),  # the run ended before the result
    ]
    turns, _ = transcript(messages)
    assert turns[1]["blocks"][0]["status"] == "denied"
    lost = turns[3]["blocks"][0]
    assert lost["status"] == "error" and lost["result"] == RUN_ENDED


def test_a_pending_approval_is_awaiting_and_the_rest_waits_behind_it() -> None:
    messages = [
        HumanMessage("list and run"),
        reply("", call("list_files", "tool_0", path="."), call("run_python", "tool_1", code="1")),
    ]
    pending = {"action_requests": [{"name": "run_python", "args": {"code": "1"}}]}
    turns, ids = transcript(messages, pending)
    listing, python = turns[1]["blocks"]
    assert python["status"] == "awaiting_approval" and "result" not in python
    assert listing["status"] == "running" and "result" not in listing
    assert ids.resolve("tool_1") == "tool_1"  # the aliases a resume continues


def test_a_result_without_its_call_is_still_shown() -> None:
    turns, _ = transcript([HumanMessage("x"), ToolMessage("orphan", tool_call_id="t9", name="now")])
    block = turns[1]["blocks"][0]
    assert block["name"] == "now" and block["result"] == "orphan" and block["status"] == "done"


def test_block_content_is_joined() -> None:
    blocks = [{"type": "text", "text": "Hel"}, {"type": "text", "text": "lo"}]
    turns, _ = transcript([HumanMessage(content=blocks), reply(content=blocks)])  # type: ignore[arg-type]
    assert turns[0]["text"] == "Hello" and turns[1]["blocks"][0]["text"] == "Hello"
