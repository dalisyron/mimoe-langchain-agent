"""``QwenMiddleware`` and ``GuardrailMiddleware``: hook-level tests, then their effect inside the
agent built against the fake engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from conftest import FakeMimoe
from langchain.agents.middleware import ModelRequest, ModelResponse, ToolCallRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from mimoe_agent.agent import build_agent
from mimoe_agent.config import Settings, load_settings
from mimoe_agent.llm import make_model
from mimoe_agent.middleware import (
    MALFORMED_TOOL_CALL,
    OLD_RESULT_TAIL,
    REASONING_KEY,
    GuardrailMiddleware,
    QwenMiddleware,
    clean_ai_message,
    split_think,
)
from mimoe_agent.mimoe import MimoeClient, preflight

CFG = {"configurable": {"thread_id": "t1"}}
TOOL_CALL = {"name": "list_files", "args": {"path": "."}, "id": "tool_0", "type": "tool_call"}
INVALID_CALL = {
    "name": "run_python",
    "args": '{"code": "print(1)',
    "id": "tool_0",
    "error": "Unterminated string",
    "type": "invalid_tool_call",
}


class Recorder:
    """A ``wrap_model_call`` handler that records the request it received."""

    def __init__(self, message: AIMessage) -> None:
        self.message = message
        self.seen: ModelRequest | None = None

    def __call__(self, request: ModelRequest) -> ModelResponse:
        self.seen = request
        return ModelResponse(result=[self.message])

    async def acall(self, request: ModelRequest) -> ModelResponse:
        return self(request)


def _request(llm: ChatOpenAI, *messages: Any) -> ModelRequest:
    return ModelRequest(model=llm, messages=list(messages))


def _tool_request(name: str = "read_file") -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": name, "args": {"path": "x"}, "id": "tool_0", "type": "tool_call"},
        tool=None,
        state={},
        runtime=None,  # type: ignore[arg-type]
    )


def _agent(fake: FakeMimoe, settings: Settings, tools: list[Any]) -> Any:
    client = MimoeClient(settings.base_url, settings.api_key, client=fake.client())
    pre = preflight(settings, client=client)
    llm = make_model(
        settings, pre, http_client=fake.client(), http_async_client=fake.async_client()
    )
    agent = build_agent(settings, pre, llm=llm, tools=tools)
    fake.calls.clear()
    return agent


@tool
def big(path: str = ".") -> str:
    """Return a 100 KB result."""
    return "x" * 100_000


@tool
def boom(path: str = ".") -> str:
    """Always raise."""
    raise RuntimeError("tool bug")


@tool
def medium(path: str = ".") -> str:
    """Return a 5,000-character result (under the cap, over the old-result cap)."""
    return "m" * 5_000


# -- split_think / clean_ai_message --------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "reasoning", "visible"),
    [
        ("<think>\nplan the answer\n</think>\n\nThe answer.", "plan the answer", "The answer."),
        ("<think>\n\n</think>\n\nThe answer.", "", "The answer."),
        ("<think>\n</think>\n\nThe answer.", "", "The answer."),
        ("<think>\n\n</think>", "", ""),
        ("<think>\nstill thinking", "still thinking", ""),
        ("The answer.", "", "The answer."),
        ("", "", ""),
    ],
)
def test_split_think(text: str, reasoning: str, visible: str) -> None:
    assert split_think(text) == (reasoning, visible)


THINK_VARIANTS = [
    ("<think>\nplan the answer\n</think>\n\nThe answer.", "plan the answer", "The answer."),
    ("<think>\n\n</think>\n\nThe answer.", None, "The answer."),
    ("<think>\nstill thinking", "still thinking", ""),
    ("The answer.", None, "The answer."),
]


@pytest.mark.parametrize(("content", "reasoning", "visible"), THINK_VARIANTS)
async def test_think_variants_sync_and_async(
    llm_fake: ChatOpenAI, content: str, reasoning: str | None, visible: str
) -> None:
    middleware = QwenMiddleware(soft_no_think=False)
    original = AIMessage(content=content, id="ai-1")
    handler = Recorder(original)

    for response in (
        middleware.wrap_model_call(_request(llm_fake, HumanMessage("hi")), handler),
        await middleware.awrap_model_call(_request(llm_fake, HumanMessage("hi")), handler.acall),
    ):
        message = response.result[0]
        assert isinstance(message, AIMessage)
        assert message.content == visible
        assert message.id == "ai-1"
        assert message.additional_kwargs.get(REASONING_KEY) == reasoning
        if content == visible:
            assert message is original  # untouched messages are passed through as-is


def test_think_block_plus_tool_call_keeps_the_call(llm_fake: ChatOpenAI) -> None:
    message = AIMessage(content="<think>\n\n</think>", tool_calls=[TOOL_CALL])
    response = QwenMiddleware(soft_no_think=True).wrap_model_call(
        _request(llm_fake, HumanMessage("list")), Recorder(message)
    )
    cleaned = response.result[0]
    assert isinstance(cleaned, AIMessage)
    assert cleaned.content == ""
    assert cleaned.tool_calls == [TOOL_CALL]
    assert REASONING_KEY not in cleaned.additional_kwargs


def test_server_reasoning_drops_the_templates_leading_whitespace() -> None:
    # 1.0 engines with thinking on: content arrives as "\n\n<answer>" after the reasoning
    # (observed live with smollm3-3b); the checkpoint should hold the same text as the 0.6 path
    thinking = AIMessage(content="\n\n7 times 8 is 56.", additional_kwargs={REASONING_KEY: "7*8"})
    cleaned = clean_ai_message(thinking)
    assert cleaned.content == "7 times 8 is 56."
    assert cleaned.additional_kwargs == {REASONING_KEY: "7*8"}

    plain = AIMessage(content="\n\nno reasoning here")  # thinking off: content is left alone
    assert clean_ai_message(plain) is plain
    empty = AIMessage(content="\n\nx", additional_kwargs={REASONING_KEY: ""})
    assert clean_ai_message(empty) is empty


def test_inline_reasoning_appends_to_server_reasoning() -> None:
    message = AIMessage(
        content="<think>\ninline\n</think>\n\nA",
        additional_kwargs={REASONING_KEY: "server"},
    )
    cleaned = clean_ai_message(message)
    assert cleaned.content == "A"
    assert cleaned.additional_kwargs[REASONING_KEY] == "server\ninline"
    assert cleaned.content_blocks[0] == {"type": "reasoning", "reasoning": "server\ninline"}


async def test_soft_no_think_patches_latest_human_message_in_request_only(
    llm_fake: ChatOpenAI,
) -> None:
    human = HumanMessage("list the files")
    messages = [
        HumanMessage("earlier question"),
        AIMessage(content="earlier answer"),
        human,
        AIMessage(content="", tool_calls=[TOOL_CALL]),
        ToolMessage(content="notes.md", tool_call_id="tool_0", name="list_files"),
    ]
    request = _request(llm_fake, *messages)
    handler = Recorder(AIMessage(content="ok"))

    QwenMiddleware(soft_no_think=True).wrap_model_call(request, handler)
    assert handler.seen is not None
    assert handler.seen.messages[2].content == "list the files /no_think"
    assert handler.seen.messages[0].content == "earlier question"  # only the latest one
    assert human.content == "list the files"  # the original (checkpointed) message is untouched
    assert request.messages[2] is human

    await QwenMiddleware(soft_no_think=True).awrap_model_call(request, handler.acall)
    assert handler.seen.messages[2].content == "list the files /no_think"

    QwenMiddleware(soft_no_think=False).wrap_model_call(request, handler)
    assert handler.seen is request  # nothing to patch: the request object passes through


def test_invalid_tool_call_guard_keeps_the_next_request_clean(llm_fake: ChatOpenAI) -> None:
    malformed = AIMessage(content="<think>\n\n</think>", invalid_tool_calls=[INVALID_CALL])
    dangling = llm_fake._get_request_payload([HumanMessage("hi"), malformed])["messages"][-1]
    assert "tool_calls" in dangling  # what ChatOpenAI would send without the guard

    response = QwenMiddleware(soft_no_think=False).wrap_model_call(
        _request(llm_fake, HumanMessage("hi")), Recorder(malformed)
    )
    fixed = response.result[0]
    assert isinstance(fixed, AIMessage)
    assert fixed.content == MALFORMED_TOOL_CALL
    assert fixed.invalid_tool_calls == [] and fixed.tool_calls == []
    payload = llm_fake._get_request_payload([HumanMessage("hi"), fixed])
    assistant = payload["messages"][-1]
    assert assistant == {"role": "assistant", "content": MALFORMED_TOOL_CALL}
    assert "tool_calls" not in assistant


def test_invalid_next_to_valid_tool_calls_is_left_alone() -> None:
    message = AIMessage(
        content="<think>\n\n</think>", tool_calls=[TOOL_CALL], invalid_tool_calls=[INVALID_CALL]
    )
    cleaned = clean_ai_message(message)
    assert cleaned.content == ""
    assert cleaned.tool_calls == [TOOL_CALL]
    assert cleaned.invalid_tool_calls == [INVALID_CALL]


# -- GuardrailMiddleware hooks -------------------------------------------------------------------


async def test_tool_result_capped_with_tail() -> None:
    guard = GuardrailMiddleware()
    request = _tool_request()
    huge = ToolMessage(content="x" * 100_000, tool_call_id="tool_0", name="read_file")

    async def ahandler(req: ToolCallRequest) -> ToolMessage:
        return huge

    for result in (
        guard.wrap_tool_call(request, lambda req: huge),
        await guard.awrap_tool_call(request, ahandler),
    ):
        assert isinstance(result, ToolMessage)
        head, tail = result.content.rsplit("\n", 1)
        assert head == "x" * 8000
        assert tail == "[truncated: 100,000 chars total; use offset/limit]"
        assert result.tool_call_id == "tool_0" and result.status == "success"

    small = ToolMessage(content="x" * 8000, tool_call_id="tool_0", name="read_file")
    assert guard.wrap_tool_call(request, lambda req: small) is small
    command = Command(update={"messages": []})
    assert guard.wrap_tool_call(request, lambda req: command) is command


async def test_tool_exception_becomes_error_message() -> None:
    guard = GuardrailMiddleware()
    request = _tool_request()

    def handler(req: ToolCallRequest) -> ToolMessage:
        raise RuntimeError("boom")

    async def ahandler(req: ToolCallRequest) -> ToolMessage:
        raise RuntimeError("boom")

    for result in (
        guard.wrap_tool_call(request, handler),
        await guard.awrap_tool_call(request, ahandler),
    ):
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert result.content == "ERROR: read_file failed with RuntimeError: boom"
        assert result.tool_call_id == "tool_0" and result.name == "read_file"


async def test_interrupts_propagate_through_the_tool_hook() -> None:
    guard = GuardrailMiddleware()

    def handler(req: ToolCallRequest) -> ToolMessage:
        raise GraphInterrupt()

    async def ahandler(req: ToolCallRequest) -> ToolMessage:
        raise GraphInterrupt()

    with pytest.raises(GraphInterrupt):
        guard.wrap_tool_call(_tool_request(), handler)
    with pytest.raises(GraphInterrupt):
        await guard.awrap_tool_call(_tool_request(), ahandler)


async def test_old_tool_messages_trimmed_in_request_only(llm_fake: ChatOpenAI) -> None:
    old_result = ToolMessage(content="x" * 5000, tool_call_id="tool_0", name="list_files")
    new_result = ToolMessage(content="y" * 5000, tool_call_id="tool_0", name="list_files")
    request = _request(
        llm_fake,
        HumanMessage("turn 1"),
        AIMessage(content="", tool_calls=[TOOL_CALL]),
        old_result,
        AIMessage(content="done"),
        HumanMessage("turn 2"),
        AIMessage(content="", tool_calls=[TOOL_CALL]),
        new_result,
    )
    handler = Recorder(AIMessage(content="ok"))
    guard = GuardrailMiddleware()

    guard.wrap_model_call(request, handler)
    assert handler.seen is not None and handler.seen is not request
    seen = handler.seen.messages
    assert seen[2].content == "x" * 400 + "\n" + OLD_RESULT_TAIL
    assert seen[6] is new_result  # current turn: untouched
    assert request.messages[2] is old_result and old_result.content == "x" * 5000

    await guard.awrap_model_call(request, handler.acall)
    assert handler.seen.messages[2].content == "x" * 400 + "\n" + OLD_RESULT_TAIL

    short = _request(llm_fake, HumanMessage("only turn"), new_result)
    guard.wrap_model_call(short, handler)
    assert handler.seen is short


def test_before_agent_resets_the_thread_counter() -> None:
    guard = GuardrailMiddleware(thread_limit=5)
    assert guard.thread_limit == 5
    assert guard.before_agent({"messages": [], "thread_model_call_count": 7}, None) == {  # type: ignore[arg-type]
        "thread_model_call_count": 0
    }


# -- inside the agent ----------------------------------------------------------------------------


def test_agent_counter_resets_per_turn_across_a_resume(
    fake_mimoe: FakeMimoe, settings_tmp: Settings
) -> None:
    from mimoe_agent.tools import build_tools

    agent = _agent(fake_mimoe, settings_tmp, build_tools(settings_tmp))
    fake_mimoe.script(
        {"tool_calls": [{"name": "run_python", "args": {"code": "print(6*7)"}}]},
        {"content": "42"},
        {"content": "Hi!"},
    )
    result = agent.invoke({"messages": [HumanMessage("run")]}, CFG)
    assert "__interrupt__" in result
    agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), CFG)
    assert agent.get_state(CFG).values["thread_model_call_count"] == 2

    agent.invoke({"messages": [HumanMessage("hi")]}, CFG)
    assert agent.get_state(CFG).values["thread_model_call_count"] == 1  # reset, then one call
    assert len(fake_mimoe.calls) == 3


def test_agent_caps_a_100kb_tool_result(fake_mimoe: FakeMimoe, settings_tmp: Settings) -> None:
    agent = _agent(fake_mimoe, settings_tmp, [big])
    fake_mimoe.script({"tool_calls": [{"name": "big", "args": {}}]}, {"content": "done"})
    result = agent.invoke({"messages": [HumanMessage("go")]}, CFG)
    tool_message = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert len(tool_message.content) == 8000 + len(
        "\n[truncated: 100,000 chars total; use offset/limit]"
    )
    assert tool_message.content.endswith("[truncated: 100,000 chars total; use offset/limit]")
    sent = fake_mimoe.calls[-1]["messages"][-1]
    assert sent["role"] == "tool" and sent["content"] == tool_message.content


def test_agent_turns_a_tool_exception_into_an_error_message(
    fake_mimoe: FakeMimoe, settings_tmp: Settings
) -> None:
    agent = _agent(fake_mimoe, settings_tmp, [boom])
    fake_mimoe.script({"tool_calls": [{"name": "boom", "args": {}}]}, {"content": "it failed"})
    result = agent.invoke({"messages": [HumanMessage("go")]}, CFG)
    tool_message = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert tool_message.status == "error"
    assert tool_message.content == "ERROR: boom failed with RuntimeError: tool bug"
    assert result["messages"][-1].content == "it failed"


async def test_agent_trims_old_tool_messages_but_keeps_the_checkpoint(
    workspace_tmp: Path,
) -> None:
    fake = FakeMimoe("0.6")
    settings = load_settings(
        {"workspace": workspace_tmp, "base_url": fake.base_url}, cwd=workspace_tmp.parent
    )
    agent = _agent(fake, settings, [medium])
    fake.script({"tool_calls": [{"name": "medium", "args": {}}]}, {"content": "done"})
    await agent.ainvoke({"messages": [HumanMessage("turn 1")]}, CFG)
    same_turn = [m for m in fake.calls[-1]["messages"] if m["role"] == "tool"][0]
    assert len(same_turn["content"]) == 5000  # the current turn sees the full result

    fake.script({"content": "again"})
    await agent.ainvoke({"messages": [HumanMessage("turn 2")]}, CFG)
    older = [m for m in fake.calls[-1]["messages"] if m["role"] == "tool"][0]
    assert older["content"] == "m" * 400 + "\n" + OLD_RESULT_TAIL
    checkpointed = next(
        m for m in agent.get_state(CFG).values["messages"] if isinstance(m, ToolMessage)
    )
    assert checkpointed.content == "m" * 5000
