"""``iter_events`` / ``aiter_events`` on small ``create_agent`` graphs, plus ``ThinkSplitter``.

Every agent scenario runs through the sync and the async iterator (the ``run`` fixture) against
:class:`conftest.FakeMimoe`, which streams the chunk shapes verified on Studio 0.6.5 and the
1.0.27 runtime.
"""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from conftest import FakeMimoe
from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    InterruptOnConfig,
    ModelCallLimitMiddleware,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from mimoe_agent.config import load_settings
from mimoe_agent.llm import make_model
from mimoe_agent.mimoe import (
    HINT_CONTEXT,
    HINT_NOT_REACHABLE,
    HINT_SERVER_ERROR,
    MimoeClient,
    preflight,
)
from mimoe_agent.stream import ThinkSplitter, aiter_events, iter_events, visible_text

Event = dict[str, Any]
Runner = Callable[[Any, Any, dict[str, Any]], list[Event]]

REPO_WORKSPACE = Path(__file__).resolve().parent.parent / "workspace"
LIVE_BASE_URL = "http://127.0.0.1:8083/mimik-ai/openai/v1"
LIVE = os.environ.get("MIMOE_LIVE") == "1"
"""Read at import: the autouse ``_clean_mimoe_env`` fixture strips ``MIMOE_*`` before tests run."""


@tool
def add(a: int, b: int) -> str:
    """Add two integers."""
    return str(a + b)


@tool
def echo(text: str) -> str:
    """Echo text back."""
    return f"ECHO: {text}"


ECHO_APPROVAL = InterruptOnConfig(allowed_decisions=["approve", "reject"], description="run echo?")


def build_agent(
    fake: FakeMimoe,
    workspace: Path,
    *,
    hitl: bool = True,
    thread_limit: int | None = None,
    llm: BaseChatModel | None = None,
    tools: Sequence[BaseTool] | None = None,
) -> Any:
    """A two-tool agent on the fake; script the fake only after this (preflight runs the probe)."""
    settings = load_settings(
        {"workspace": workspace, "base_url": fake.base_url}, cwd=workspace.parent
    )
    if llm is None:
        client = MimoeClient(settings.base_url, settings.api_key, client=fake.client())
        pre = preflight(settings, client=client)
        llm = make_model(
            settings, pre, http_client=fake.client(), http_async_client=fake.async_client()
        )
    middleware: list[Any] = []
    if thread_limit is not None:
        middleware.append(ModelCallLimitMiddleware(thread_limit=thread_limit, exit_behavior="end"))
    if hitl:
        middleware.append(HumanInTheLoopMiddleware(interrupt_on={"echo": ECHO_APPROVAL}))
    return create_agent(
        llm,
        tools=list(tools) if tools is not None else [add, echo],
        system_prompt="You are terse.",
        middleware=middleware,
        checkpointer=InMemorySaver(),
    )


def cfg(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def names(events: list[Event]) -> list[str]:
    return [event["event"] for event in events]


def text_of(events: list[Event], kind: str) -> str:
    return "".join(event["text"] for event in events if event["event"] == kind)


def usage_of(events: list[Event]) -> dict[str, int]:
    assert events[-1]["event"] == "done"
    return events[-1]["usage_total"]


class ScriptedGraph:
    """Stand-in agent that replays canned ``(mode, data)`` items and records a closed stream."""

    def __init__(self, *items: tuple[str, Any]) -> None:
        self.items = list(items)
        self.closed = False

    def stream(
        self, payload: Any, config: Any, *, stream_mode: list[str]
    ) -> Iterator[tuple[str, Any]]:
        assert stream_mode == ["messages", "updates"]
        try:
            yield from self.items
        finally:
            self.closed = True

    async def astream(
        self, payload: Any, config: Any, *, stream_mode: list[str]
    ) -> AsyncIterator[tuple[str, Any]]:
        assert stream_mode == ["messages", "updates"]
        try:
            for item in self.items:
                yield item
        finally:
            self.closed = True


def model_chunk(text: str, run_id: str = "lc_run--1", **extra: Any) -> tuple[str, Any]:
    """A ``messages``-mode item of the ``model`` node (every chunk of a call shares its run id)."""
    chunk = AIMessageChunk(content=text, id=run_id, **extra)
    return ("messages", (chunk, {"langgraph_node": "model", "ls_model_name": "scripted"}))


@pytest.fixture(params=["sync", "async"])
def run(request: pytest.FixtureRequest) -> Iterator[Runner]:
    """Collect the events of one run through ``iter_events`` or ``aiter_events``."""
    if request.param == "sync":
        yield lambda agent, payload, config: list(iter_events(agent, payload, config))
        return
    loop = asyncio.new_event_loop()

    async def collect(agent: Any, payload: Any, config: dict[str, Any]) -> list[Event]:
        return [event async for event in aiter_events(agent, payload, config)]

    yield lambda agent, payload, config: loop.run_until_complete(collect(agent, payload, config))
    loop.close()


# -- plain answers and reasoning -------------------------------------------------------------------


def test_plain_answer(fake_mimoe: FakeMimoe, workspace_tmp: Path, run: Runner) -> None:
    agent = build_agent(fake_mimoe, workspace_tmp)
    fake_mimoe.script({"content": "Hello there, friend."})
    events = run(agent, {"messages": [HumanMessage("hi")]}, cfg("plain"))
    assert events[:-1] == [
        {"event": "token", "text": "Hello"},
        {"event": "token", "text": " there,"},
        {"event": "token", "text": " friend."},
    ]
    done = events[-1]
    assert done["event"] == "done"
    assert done["status"] == "completed"
    assert done["model"] == "qwen3-4b"
    assert done["usage_total"] == {"input_tokens": 120, "output_tokens": 3, "llm_calls": 1}
    assert isinstance(done["elapsed_s"], float) and done["elapsed_s"] >= 0
    assert "<think>" not in text_of(events, "token")


def test_inline_think_becomes_thinking_events(
    fake_mimoe: FakeMimoe, workspace_tmp: Path, run: Runner
) -> None:
    agent = build_agent(fake_mimoe, workspace_tmp)
    fake_mimoe.script({"content": "Hello there, friend.", "reasoning": "I should greet back"})
    events = run(agent, {"messages": [HumanMessage("hi")]}, cfg("think"))
    assert names(events) == ["thinking", "token", "token", "token", "done"]
    assert events[0] == {"event": "thinking", "text": "I should greet back"}
    assert text_of(events, "token") == "Hello there, friend."


class _ReasoningChatOpenAI(ChatOpenAI):
    """Stand-in for ``llm.MimoeChatOpenAI``: copies ``delta.reasoning_content`` (1.0 engines) into
    ``additional_kwargs["reasoning_content"]`` of the streamed chunk."""

    def _convert_chunk_to_generation_chunk(
        self, chunk: dict, default_chunk_class: type, base_generation_info: dict | None
    ) -> Any:
        generation = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        choices = chunk.get("choices") or []
        delta = choices[0].get("delta") if choices else None
        reasoning = delta.get("reasoning_content") if isinstance(delta, dict) else None
        if generation is not None and isinstance(reasoning, str) and reasoning:
            generation.message.additional_kwargs["reasoning_content"] = reasoning
        return generation


def test_reasoning_content_v10(fake_mimoe_v10: FakeMimoe, workspace_tmp: Path, run: Runner) -> None:
    llm = _ReasoningChatOpenAI(
        model="qwen3-4b",
        base_url=fake_mimoe_v10.base_url,
        api_key="1234",
        use_responses_api=False,
        temperature=0,
        max_retries=0,
        stream_usage=True,
        extra_body={"max_tokens": 300, "enable_thinking": True},
        http_client=fake_mimoe_v10.client(),
        http_async_client=fake_mimoe_v10.async_client(),
    )
    agent = build_agent(fake_mimoe_v10, workspace_tmp, llm=llm)
    fake_mimoe_v10.script({"content": "Hello there, friend.", "reasoning": "I should greet back"})
    events = run(agent, {"messages": [HumanMessage("hi")]}, cfg("v10"))
    assert names(events) == ["thinking"] * 4 + ["token"] * 3 + ["done"]
    assert text_of(events, "thinking") == "I should greet back"
    assert text_of(events, "token") == "Hello there, friend."
    assert events[-1]["model"] == "qwen3-4b"
    assert usage_of(events)["llm_calls"] == 1


def test_v10_through_make_model_streams_clean_tokens(
    fake_mimoe_v10: FakeMimoe, workspace_tmp: Path, run: Runner
) -> None:
    """The stock ``ChatOpenAI`` drops ``reasoning_content``; ``MimoeChatOpenAI`` (llm.py) keeps
    it. Either way the visible answer must be clean."""
    agent = build_agent(fake_mimoe_v10, workspace_tmp)
    fake_mimoe_v10.script({"content": "Hello there, friend.", "reasoning": "I should greet back"})
    events = run(agent, {"messages": [HumanMessage("hi")]}, cfg("v10-stock"))
    assert text_of(events, "token") == "Hello there, friend."
    assert "<think>" not in text_of(events, "token")
    assert text_of(events, "thinking") in ("", "I should greet back")
    assert events[-1]["status"] == "completed" and usage_of(events)["llm_calls"] == 1


def test_feed_chunk_reads_additional_kwargs() -> None:
    splitter = ThinkSplitter()
    reasoning = AIMessageChunk(content="", additional_kwargs={"reasoning_content": "\nBecause"})
    assert splitter.feed_chunk(reasoning) == [{"event": "thinking", "text": "Because"}]
    more = AIMessageChunk(content="", additional_kwargs={"reasoning_content": " of that\n"})
    assert splitter.feed_chunk(more) == [{"event": "thinking", "text": " of that"}]
    assert splitter.feed_chunk(AIMessageChunk(content="\n\nHi")) == [
        {"event": "token", "text": "Hi"}
    ]
    assert splitter.feed_chunk(AIMessageChunk(content="")) == []
    assert splitter.flush() == []


def test_empty_status_chunks_emit_nothing() -> None:
    splitter = ThinkSplitter()
    assert splitter.feed_chunk(AIMessageChunk(content="")) == []
    assert splitter.feed_chunk(AIMessageChunk(content="", additional_kwargs={})) == []
    assert splitter.flush() == []


# -- tools, approvals, notices ---------------------------------------------------------------------


def test_tool_round_trip(fake_mimoe: FakeMimoe, workspace_tmp: Path, run: Runner) -> None:
    agent = build_agent(fake_mimoe, workspace_tmp)
    fake_mimoe.script(
        {"tool_calls": [{"name": "add", "args": {"a": 2, "b": 3}}]}, {"content": "It is 5."}
    )
    events = run(agent, {"messages": [HumanMessage("add 2 and 3")]}, cfg("tool"))
    assert names(events) == ["tool_call", "tool_result", "token", "token", "token", "done"]
    assert events[0] == {
        "event": "tool_call",
        "id": "tool_0",
        "name": "add",
        "args": {"a": 2, "b": 3},
    }
    assert events[1] == {
        "event": "tool_result",
        "id": "tool_0",
        "name": "add",
        "content": "5",
        "is_error": False,
    }
    assert text_of(events, "token") == "It is 5."
    assert usage_of(events) == {"input_tokens": 240, "output_tokens": 9, "llm_calls": 2}


def test_approval_then_approve(fake_mimoe: FakeMimoe, workspace_tmp: Path, run: Runner) -> None:
    agent = build_agent(fake_mimoe, workspace_tmp)
    fake_mimoe.script(
        {"tool_calls": [{"name": "echo", "args": {"text": "yo"}}]}, {"content": "Done: ECHO: yo"}
    )
    first = run(agent, {"messages": [HumanMessage("echo yo")]}, cfg("approve"))
    assert names(first) == ["tool_call", "approval_required", "done"]
    assert first[0] == {
        "event": "tool_call",
        "id": "tool_0",
        "name": "echo",
        "args": {"text": "yo"},
    }
    approval = first[1]
    assert re.fullmatch(r"[0-9a-f]{32}", approval["interrupt_id"])
    assert approval["action_requests"] == [
        {"name": "echo", "args": {"text": "yo"}, "description": "run echo?"}
    ]
    assert approval["review_configs"] == [
        {"action_name": "echo", "allowed_decisions": ["approve", "reject"]}
    ]
    assert first[-1]["status"] == "awaiting_approval"
    assert first[-1]["usage_total"]["llm_calls"] == 1

    second = run(agent, Command(resume={"decisions": [{"type": "approve"}]}), cfg("approve"))
    assert names(second) == ["tool_result", "token", "token", "token", "done"]
    assert second[0] == {
        "event": "tool_result",
        "id": "tool_0",
        "name": "echo",
        "content": "ECHO: yo",
        "is_error": False,
    }
    assert text_of(second, "token") == "Done: ECHO: yo"
    assert second[-1]["status"] == "completed"
    assert usage_of(second) == {"input_tokens": 120, "output_tokens": 3, "llm_calls": 1}


def test_approval_then_reject(fake_mimoe: FakeMimoe, workspace_tmp: Path, run: Runner) -> None:
    agent = build_agent(fake_mimoe, workspace_tmp)
    fake_mimoe.script(
        {"tool_calls": [{"name": "echo", "args": {"text": "no"}}]},
        {"content": "OK, not running it."},
    )
    first = run(agent, {"messages": [HumanMessage("echo no")]}, cfg("reject"))
    assert names(first) == ["tool_call", "approval_required", "done"]
    decision = {"type": "reject", "message": "The user declined to run this code."}
    second = run(agent, Command(resume={"decisions": [decision]}), cfg("reject"))
    assert names(second) == ["tool_result", "token", "token", "token", "token", "done"]
    result = second[0]
    assert result["id"] == "tool_0" and result["name"] == "echo" and result["is_error"] is True
    assert "rejected" in result["content"] and "The user declined" in result["content"]
    assert text_of(second, "token") == "OK, not running it."
    assert second[-1]["status"] == "completed"


def test_model_call_limit_notice(fake_mimoe: FakeMimoe, workspace_tmp: Path, run: Runner) -> None:
    agent = build_agent(fake_mimoe, workspace_tmp, hitl=False, thread_limit=3)
    for _ in range(4):
        fake_mimoe.script({"tool_calls": [{"name": "add", "args": {"a": 1, "b": 1}}]})
    events = run(agent, {"messages": [HumanMessage("loop")]}, cfg("loop"))
    assert names(events) == ["tool_call", "tool_result"] * 3 + ["notice", "done"]
    # mimOE reuses id tool_0 on every reply: one tool_call event per AI message, not per id
    assert [event["id"] for event in events if event["event"] == "tool_call"] == ["tool_0"] * 3
    assert events[-2] == {
        "event": "notice",
        "text": "Model call limits exceeded: thread limit (3/3)",
    }
    assert events[-1]["status"] == "completed"
    assert usage_of(events)["llm_calls"] == 3


def test_notice_only_turn_uses_configured_model_name(
    fake_mimoe: FakeMimoe, workspace_tmp: Path, run: Runner
) -> None:
    agent = build_agent(fake_mimoe, workspace_tmp, hitl=False, thread_limit=1)
    fake_mimoe.script({"content": "First."})
    first = run(agent, {"messages": [HumanMessage("one")]}, cfg("limit1"))
    assert names(first) == ["token", "done"] and first[-1]["model"] == "qwen3-4b"
    second = run(agent, {"messages": [HumanMessage("two")]}, cfg("limit1"))
    assert second == [
        {"event": "notice", "text": "Model call limits exceeded: thread limit (1/1)"},
        second[-1],
    ]
    assert second[-1]["event"] == "done"
    assert second[-1]["model"] == "unknown"
    assert second[-1]["usage_total"] == {"input_tokens": 0, "output_tokens": 0, "llm_calls": 0}
    config = {**cfg("limit1"), "metadata": {"model": "from-config"}}
    third = run(agent, {"messages": [HumanMessage("three")]}, config)
    assert names(third) == ["notice", "done"] and third[-1]["model"] == "from-config"


def test_non_streamed_reply_is_emitted_from_the_update(
    fake_mimoe: FakeMimoe, workspace_tmp: Path, run: Runner
) -> None:
    """A model that does not stream (or a middleware that rewrote the reply) still yields text."""
    settings = load_settings(
        {"workspace": workspace_tmp, "base_url": fake_mimoe.base_url}, cwd=workspace_tmp.parent
    )
    client = MimoeClient(settings.base_url, settings.api_key, client=fake_mimoe.client())
    pre = preflight(settings, client=client)
    llm = make_model(
        settings, pre, http_client=fake_mimoe.client(), http_async_client=fake_mimoe.async_client()
    )
    llm.disable_streaming = True
    agent = build_agent(fake_mimoe, workspace_tmp, llm=llm)
    fake_mimoe.script({"content": "Whole reply.", "reasoning": "quietly"})
    events = run(agent, {"messages": [HumanMessage("hi")]}, cfg("nostream"))
    assert events[:-1] == [
        {"event": "thinking", "text": "quietly"},
        {"event": "token", "text": "Whole reply."},
    ]
    assert events[-1]["status"] == "completed" and usage_of(events)["llm_calls"] == 1


def test_two_tool_calls_in_one_message(
    fake_mimoe: FakeMimoe, workspace_tmp: Path, run: Runner
) -> None:
    agent = build_agent(fake_mimoe, workspace_tmp, hitl=False)
    fake_mimoe.script(
        {
            "tool_calls": [
                {"name": "add", "args": {"a": 1, "b": 1}},
                {"name": "add", "args": {"a": 2, "b": 2}},
            ]
        },
        {"content": "2 and 4."},
    )
    events = run(agent, {"messages": [HumanMessage("add both")]}, cfg("two-calls"))
    assert names(events) == ["tool_call"] * 2 + ["tool_result"] * 2 + ["token"] * 3 + ["done"]
    assert [(e["id"], e["args"]) for e in events[:2]] == [
        ("tool_0", {"a": 1, "b": 1}),
        ("tool_1", {"a": 2, "b": 2}),
    ]
    assert {e["id"]: e["content"] for e in events[2:4]} == {"tool_0": "2", "tool_1": "4"}
    assert usage_of(events)["llm_calls"] == 2


def test_one_interrupt_for_two_gated_calls(
    fake_mimoe: FakeMimoe, workspace_tmp: Path, run: Runner
) -> None:
    """HITL batches every gated call of one AI message into a single interrupt."""
    agent = build_agent(fake_mimoe, workspace_tmp)
    fake_mimoe.script(
        {
            "tool_calls": [
                {"name": "echo", "args": {"text": "a"}},
                {"name": "echo", "args": {"text": "b"}},
            ]
        },
        {"content": "Did a only."},
    )
    first = run(agent, {"messages": [HumanMessage("echo a and b")]}, cfg("two-gated"))
    assert names(first) == ["tool_call", "tool_call", "approval_required", "done"]
    approval = first[2]
    assert [r["args"] for r in approval["action_requests"]] == [{"text": "a"}, {"text": "b"}]
    assert [r["action_name"] for r in approval["review_configs"]] == ["echo", "echo"]
    assert first[-1]["status"] == "awaiting_approval"

    decisions = [{"type": "approve"}, {"type": "reject", "message": "not b"}]
    second = run(agent, Command(resume={"decisions": decisions}), cfg("two-gated"))
    assert names(second) == ["tool_result", "tool_result", "token", "token", "token", "done"]
    results = {e["id"]: e for e in second[:2]}
    assert results["tool_0"]["content"] == "ECHO: a" and results["tool_0"]["is_error"] is False
    assert results["tool_1"]["is_error"] is True and "not b" in results["tool_1"]["content"]
    assert text_of(second, "token") == "Did a only."
    assert second[-1]["status"] == "completed"


def test_later_turn_does_not_replay_earlier_tool_results(
    fake_mimoe: FakeMimoe, workspace_tmp: Path, run: Runner
) -> None:
    agent = build_agent(fake_mimoe, workspace_tmp, hitl=False)
    fake_mimoe.script(
        {"tool_calls": [{"name": "add", "args": {"a": 2, "b": 3}}]},
        {"content": "5"},
        {"content": "Hello again."},
    )
    first = run(agent, {"messages": [HumanMessage("add 2 and 3")]}, cfg("turns"))
    assert names(first) == ["tool_call", "tool_result", "token", "done"]
    second = run(agent, {"messages": [HumanMessage("hi")]}, cfg("turns"))
    assert names(second) == ["token", "token", "done"]
    assert text_of(second, "token") == "Hello again."


def test_stopping_early_keeps_the_thread_usable(fake_mimoe: FakeMimoe, workspace_tmp: Path) -> None:
    agent = build_agent(fake_mimoe, workspace_tmp)
    fake_mimoe.script(*[{"content": "one two three"}, {"content": "Still fine."}] * 2)
    for event in iter_events(agent, {"messages": [HumanMessage("go")]}, cfg("early-sync")):
        assert event == {"event": "token", "text": "one"}
        break
    events = list(iter_events(agent, {"messages": [HumanMessage("again")]}, cfg("early-sync")))
    assert text_of(events, "token") == "Still fine." and events[-1]["status"] == "completed"

    async def scenario() -> None:
        config = cfg("early-async")
        stream = aiter_events(agent, {"messages": [HumanMessage("go")]}, config)
        assert await anext(stream) == {"event": "token", "text": "one"}
        await stream.aclose()
        # the run was torn down before aclose() returned: nothing of it is still pending
        assert [t for t in asyncio.all_tasks() if t is not asyncio.current_task()] == []
        events = [
            e async for e in aiter_events(agent, {"messages": [HumanMessage("again")]}, config)
        ]
        assert text_of(events, "token") == "Still fine." and events[-1]["status"] == "completed"

    asyncio.run(scenario())


# -- mapper on canned graph output -----------------------------------------------------------------


def test_usage_without_usage_metadata_counts_the_call(run: Runner) -> None:
    update = ("updates", {"model": {"messages": [AIMessage(content="No usage here.")]}})
    graph = ScriptedGraph(update)
    events = run(graph, {"messages": [HumanMessage("hi")]}, cfg("nousage"))
    assert events[:-1] == [{"event": "token", "text": "No usage here."}]
    done = events[-1]
    assert done["usage_total"] == {"input_tokens": 0, "output_tokens": 0, "llm_calls": 1}
    assert isinstance(done["elapsed_s"], float) and done["elapsed_s"] >= 0
    assert done["model"] == "unknown"
    assert graph.closed


def test_tool_call_event_comes_from_the_update_only(run: Runner) -> None:
    """0.6 streams a call as id+name then argument fragments; the event waits for the complete
    call and the HITL node's re-emission of the same message on resume adds nothing."""

    def fragment(**call: Any) -> tuple[str, Any]:
        return model_chunk("", tool_call_chunks=[{"index": 0, "type": "tool_call_chunk", **call}])

    complete = AIMessage(
        content="<think>",
        id="lc_run--1",
        tool_calls=[{"name": "add", "args": {"a": 1, "b": 2}, "id": "tool_0", "type": "tool_call"}],
    )
    graph = ScriptedGraph(
        model_chunk("<think>"),
        fragment(name="add", args="", id="tool_0"),
        fragment(name=None, args='{"a": 1', id=None),
        fragment(name=None, args=', "b": 2}', id=None),
        ("updates", {"model": {"messages": [complete]}}),
        ("updates", {"HumanInTheLoopMiddleware.after_model": {"messages": [complete]}}),
    )
    events = run(graph, {"messages": [HumanMessage("add")]}, cfg("fragments"))
    assert events[:-1] == [
        {"event": "tool_call", "id": "tool_0", "name": "add", "args": {"a": 1, "b": 2}}
    ]
    assert events[-1]["model"] == "scripted" and usage_of(events)["llm_calls"] == 1


def test_stopping_early_closes_the_graph_stream() -> None:
    def graph() -> ScriptedGraph:
        return ScriptedGraph(model_chunk("one"), model_chunk(" two"), model_chunk(" three"))

    sync_graph = graph()
    events = iter_events(sync_graph, {"messages": [HumanMessage("go")]}, cfg("early"))
    assert next(events) == {"event": "token", "text": "one"}
    events.close()
    assert sync_graph.closed

    async def scenario() -> None:
        async_graph = graph()
        stream = aiter_events(async_graph, {"messages": [HumanMessage("go")]}, cfg("early"))
        assert await anext(stream) == {"event": "token", "text": "one"}
        await stream.aclose()
        assert async_graph.closed  # closed before aclose() returned, not left to the GC hook

    asyncio.run(scenario())


# -- errors ----------------------------------------------------------------------------------------


def test_error_after_tool_events_keeps_them(
    fake_mimoe: FakeMimoe, workspace_tmp: Path, run: Runner
) -> None:
    """A failure on the second model call still reports the tool round trip before it."""

    @tool
    def unplug(kind: str) -> str:
        """Break the engine for the next request."""
        if kind == "down":
            fake_mimoe.down = True
        else:
            fake_mimoe.fail_next(500, {"message": "boom", "statusCode": 500})
        return "unplugged"

    agent = build_agent(fake_mimoe, workspace_tmp, hitl=False, tools=[unplug])
    fake_mimoe.script({"tool_calls": [{"name": "unplug", "args": {"kind": "500"}}]})
    events = run(agent, {"messages": [HumanMessage("go")]}, cfg("mid-500"))
    assert names(events) == ["tool_call", "tool_result", "error"]
    assert events[1]["content"] == "unplugged" and events[1]["is_error"] is False
    assert events[-1] == {
        "event": "error",
        "message": "mimOE returned a server error (boom)",
        "hint": HINT_SERVER_ERROR,
    }

    fake_mimoe.script({"tool_calls": [{"name": "unplug", "args": {"kind": "down"}}]})
    events = run(agent, {"messages": [HumanMessage("go")]}, cfg("mid-down"))
    assert names(events) == ["tool_call", "tool_result", "error"]
    assert events[-1]["message"] == "mimOE Studio is not reachable"


def test_error_from_mimoe_500_body(fake_mimoe: FakeMimoe, workspace_tmp: Path, run: Runner) -> None:
    agent = build_agent(fake_mimoe, workspace_tmp)
    fake_mimoe.fail_next(500, {"message": "llama_decode() failed", "statusCode": 500})
    events = run(agent, {"messages": [HumanMessage("boom")]}, cfg("err-context"))
    assert events == [
        {
            "event": "error",
            "message": "the conversation exceeded the model's context window",
            "hint": HINT_CONTEXT,
        }
    ]
    fake_mimoe.fail_next(500, {"message": "boom", "statusCode": 500})
    events = run(agent, {"messages": [HumanMessage("boom")]}, cfg("err-500"))
    assert events == [
        {
            "event": "error",
            "message": "mimOE returned a server error (boom)",
            "hint": HINT_SERVER_ERROR,
        }
    ]


def test_error_when_engine_unreachable(
    fake_mimoe: FakeMimoe, workspace_tmp: Path, run: Runner
) -> None:
    agent = build_agent(fake_mimoe, workspace_tmp)
    fake_mimoe.down = True
    events = run(agent, {"messages": [HumanMessage("hi")]}, cfg("down"))
    assert events == [
        {"event": "error", "message": "mimOE Studio is not reachable", "hint": HINT_NOT_REACHABLE}
    ]
    assert "done" not in names(events)


# -- ThinkSplitter ---------------------------------------------------------------------------------


def test_splitter_tags_split_across_chunks() -> None:
    splitter = ThinkSplitter()
    assert splitter.feed("<thi") == []
    assert splitter.feed("nk>abc</th") == [{"event": "thinking", "text": "abc"}]
    assert splitter.feed("ink>") == []
    assert splitter.feed("\n\nHi") == [{"event": "token", "text": "Hi"}]
    assert splitter.feed(" there") == [{"event": "token", "text": " there"}]
    assert splitter.flush() == []


def test_splitter_empty_block_emits_nothing() -> None:
    splitter = ThinkSplitter()
    events: list[Event] = []
    for piece in ("<think>", "\n\n</think>", "\n\n", "Hello", " there"):
        events += splitter.feed(piece)
    events += splitter.flush()
    assert events == [{"event": "token", "text": "Hello"}, {"event": "token", "text": " there"}]


def test_splitter_unclosed_block_stays_thinking() -> None:
    splitter = ThinkSplitter()
    assert splitter.feed("<think>") == []
    assert splitter.feed("\nI was cut") == [{"event": "thinking", "text": "I was cut"}]
    assert splitter.feed(" off</thi") == [{"event": "thinking", "text": " off"}]
    assert splitter.in_think is True
    assert splitter.flush() == [{"event": "thinking", "text": "</thi"}]
    assert splitter.flush() == []


def test_splitter_keeps_interior_whitespace_and_trims_ends() -> None:
    splitter = ThinkSplitter()
    events: list[Event] = []
    for piece in ("<think>", "\n\n", "one\n", "\n", "two", "\n</think>", "\n\n", "Answer"):
        events += splitter.feed(piece)
    events += splitter.flush()
    assert events == [
        {"event": "thinking", "text": "one"},
        {"event": "thinking", "text": "\n\ntwo"},
        {"event": "token", "text": "Answer"},
    ]


def test_splitter_dangling_angle_bracket_is_visible_text() -> None:
    """Once the answer has started nothing is held back: a "<" cannot open a block any more."""
    splitter = ThinkSplitter()
    assert splitter.feed("1 <") == [{"event": "token", "text": "1 <"}]
    assert splitter.feed("2 and a <t") == [{"event": "token", "text": "2 and a <t"}]
    assert splitter.flush() == []
    fresh = ThinkSplitter()
    assert fresh.feed("\n<t") == []  # still possibly a leading tag
    assert fresh.feed("ext>") == [{"event": "token", "text": "<text>"}]


def test_splitter_think_tag_inside_the_answer_stays_visible() -> None:
    """Only a leading block is reasoning (what QwenMiddleware strips); a quoted tag is text."""
    splitter = ThinkSplitter()
    assert splitter.feed("<think>\n\n</think>\n\nUse ") == [{"event": "token", "text": "Use "}]
    assert splitter.feed("<thi") == [{"event": "token", "text": "<thi"}]
    assert splitter.feed("nk> tags.") == [{"event": "token", "text": "nk> tags."}]
    assert splitter.flush() == []
    one_chunk = ThinkSplitter()
    assert one_chunk.feed("1 <think>2</think>3") == [
        {"event": "token", "text": "1 <think>2</think>3"}
    ]
    assert visible_text(AIMessage(content="Write <think>x</think> literally")) == (
        "Write <think>x</think> literally"
    )


def test_splitter_reasoning_deltas() -> None:
    splitter = ThinkSplitter()
    assert splitter.feed_reasoning("\n") == []
    assert splitter.feed_reasoning("Okay,") == [{"event": "thinking", "text": "Okay,"}]
    assert splitter.feed_reasoning(" so\n\n") == [{"event": "thinking", "text": " so"}]
    assert splitter.feed_reasoning("then") == [{"event": "thinking", "text": "\n\nthen"}]
    assert splitter.feed("The answer") == [{"event": "token", "text": "The answer"}]
    assert splitter.flush() == []


def test_visible_text_of_final_message() -> None:
    assert visible_text(AIMessage(content="<think>\n\n</think>\n\nHello")) == "Hello"
    assert visible_text(AIMessage(content="<think>\n\n</think>")) == ""
    assert (
        visible_text(AIMessage(content="Model call limits exceeded"))
        == "Model call limits exceeded"
    )
    assert visible_text(AIMessage(content=[{"type": "text", "text": "block"}])) == "block"


# -- live ------------------------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.skipif(not LIVE, reason="set MIMOE_LIVE=1 with mimOE Studio running on 8083")
def test_live_list_files_round_trip() -> None:
    """One real run on Studio 0.6.5 / qwen3-4b (``MIMOE_LIVE=1``): prints the event names."""
    from mimoe_agent.tools.workspace import Workspace, make_workspace_tools

    settings = load_settings({"workspace": REPO_WORKSPACE, "base_url": LIVE_BASE_URL})
    pre = preflight(settings)
    llm = make_model(settings, pre)
    llm.extra_body = {**(llm.extra_body or {}), "max_tokens": 300}
    list_files = [
        t for t in make_workspace_tools(Workspace(settings.workspace)) if t.name == "list_files"
    ]
    agent = create_agent(
        llm,
        tools=list_files,
        system_prompt=(
            "/no_think\nYou are a terse assistant. Use the list_files tool to answer questions "
            "about the workspace files, then answer in one sentence."
        ),
        middleware=[ModelCallLimitMiddleware(thread_limit=4, exit_behavior="end")],
        checkpointer=InMemorySaver(),
    )
    payload = {"messages": [HumanMessage("Which files are in the workspace? /no_think")]}
    events = list(iter_events(agent, payload, cfg("live")))
    print("\nlive events:", " ".join(names(events)))
    print("live answer:", text_of(events, "token")[:200])
    assert "tool_call" in names(events)
    assert "tool_result" in names(events)
    assert events[-1]["event"] == "done" and events[-1]["status"] == "completed"
    assert events[-1]["model"] == "qwen3-4b"
    assert "<think>" not in text_of(events, "token")
