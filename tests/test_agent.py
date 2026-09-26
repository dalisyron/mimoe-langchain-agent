"""``build_agent`` end to end against the fake engine: answers, tool round trips, the
``run_python`` approval flow, chat-only mode, the per-turn model-call budget, sync and async."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from conftest import FakeMimoe
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from mimoe_agent.agent import APPROVAL_WARNING, build_agent, describe_run_python
from mimoe_agent.config import Settings, load_settings
from mimoe_agent.llm import MimoeChatOpenAI, make_model
from mimoe_agent.middleware import REASONING_KEY
from mimoe_agent.mimoe import MimoeClient, Preflight, preflight
from mimoe_agent.stream import aiter_events, iter_events
from mimoe_agent.tools import TOOL_NAMES, build_tools

CFG = {"configurable": {"thread_id": "t1"}}
LIST_FILES = {"tool_calls": [{"name": "list_files", "args": {"path": "."}}]}
RUN_PYTHON = {"tool_calls": [{"name": "run_python", "args": {"code": "print(6*7)"}}]}
APPROVE = Command(resume={"decisions": [{"type": "approve"}]})
DECLINED = (
    "The user declined to run this code. Tell the user it was not executed and stop; do not retry."
)
NOTICE = "Model call limits exceeded: thread limit (8/8)"


def _build(
    fake: FakeMimoe, settings: Settings, pre: Preflight | None = None
) -> tuple[Any, Preflight]:
    client = MimoeClient(settings.base_url, settings.api_key, client=fake.client())
    if pre is None:
        pre = preflight(settings, client=client)
    llm = make_model(
        settings, pre, http_client=fake.client(), http_async_client=fake.async_client()
    )
    agent = build_agent(settings, pre, llm=llm, tools=build_tools(settings, client))
    fake.calls.clear()
    return agent, pre


def _tool_messages(result: dict[str, Any]) -> list[ToolMessage]:
    return [m for m in result["messages"] if isinstance(m, ToolMessage)]


# -- assembly ------------------------------------------------------------------------------------


def test_build_agent_defaults_and_middleware_nodes(
    settings_tmp: Settings, preflight_fake: Preflight
) -> None:
    agent = build_agent(settings_tmp, preflight_fake)  # real clients are built, nothing is called
    nodes = set(agent.nodes)
    assert {
        "model",
        "tools",
        "GuardrailMiddleware.before_agent",
        "ModelCallLimitMiddleware.before_model",
        "ModelCallLimitMiddleware.after_model",
        "HumanInTheLoopMiddleware.after_model",
    } <= nodes
    assert agent.checkpointer is not None

    trusting = build_agent(dataclasses.replace(settings_tmp, auto_approve=True), preflight_fake)
    assert "HumanInTheLoopMiddleware.after_model" not in set(trusting.nodes)


def test_describe_run_python_renders_the_code() -> None:
    call = {"name": "run_python", "args": {"code": "print(6*7)"}, "id": "tool_0"}
    text = describe_run_python(call, {"messages": []}, None)  # type: ignore[arg-type]
    assert text == (
        "run_python wants to run this code:\n\n```python\nprint(6*7)\n```\n\n" + APPROVAL_WARNING
    )
    nested = {"name": "run_python", "args": {"code": 's = """```"""'}, "id": "tool_0"}
    fenced = describe_run_python(nested, {"messages": []}, None)  # type: ignore[arg-type]
    assert '````python\ns = """```"""\n````' in fenced


# -- sync paths ----------------------------------------------------------------------------------


def test_plain_answer(fake_mimoe: FakeMimoe, settings_tmp: Settings) -> None:
    agent, _ = _build(fake_mimoe, settings_tmp)
    fake_mimoe.script({"content": "Hello!"})
    result = agent.invoke({"messages": [HumanMessage("hi")]}, CFG)

    final = result["messages"][-1]
    assert isinstance(final, AIMessage)
    assert final.content == "Hello!"  # the empty <think> block the 0.6 engine adds is gone
    assert REASONING_KEY not in final.additional_kwargs
    assert "__interrupt__" not in result

    body = fake_mimoe.calls[-1]
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"].startswith("/no_think\n")
    assert body["messages"][-1] == {"role": "user", "content": "hi /no_think"}
    assert [t["function"]["name"] for t in body["tools"]] == list(TOOL_NAMES)
    assert body["max_tokens"] == 1024 and body["stream"] is False
    assert agent.get_state(CFG).values["messages"][0].content == "hi"  # checkpoint unpatched


def test_inline_think_is_moved_out_of_the_checkpoint(
    fake_mimoe: FakeMimoe, settings_tmp: Settings
) -> None:
    agent, _ = _build(fake_mimoe, settings_tmp)
    fake_mimoe.script({"content": "Four.", "reasoning": "two plus two is four"})
    result = agent.invoke({"messages": [HumanMessage("2+2?")]}, CFG)
    final = result["messages"][-1]
    assert final.content == "Four."
    assert final.additional_kwargs[REASONING_KEY] == "two plus two is four"
    assert "<think>" not in agent.get_state(CFG).values["messages"][-1].content


def test_tool_round_trip(fake_mimoe: FakeMimoe, settings_tmp: Settings) -> None:
    agent, _ = _build(fake_mimoe, settings_tmp)
    fake_mimoe.script(LIST_FILES, {"content": "There are three files and a src folder."})
    result = agent.invoke({"messages": [HumanMessage("what is here?")]}, CFG)

    tool_message = _tool_messages(result)[0]
    assert tool_message.name == "list_files" and tool_message.status == "success"
    assert "notes.md" in tool_message.content and "sales.csv" in tool_message.content
    assert result["messages"][-1].content == "There are three files and a src folder."
    assert "__interrupt__" not in result  # list_files needs no approval

    assert len(fake_mimoe.calls) == 2
    second = fake_mimoe.calls[-1]["messages"]
    assert second[-2]["role"] == "assistant"
    assert second[-2]["tool_calls"][0]["id"] == "tool_0"
    assert second[-2]["content"] is None  # the stripped think block is not echoed back
    assert second[-1]["role"] == "tool" and second[-1]["content"] == tool_message.content


def test_run_python_approval_approve(fake_mimoe: FakeMimoe, settings_tmp: Settings) -> None:
    agent, _ = _build(fake_mimoe, settings_tmp)
    fake_mimoe.script(RUN_PYTHON, {"content": "6*7 is 42."})
    result = agent.invoke({"messages": [HumanMessage("Use run_python to print 6*7")]}, CFG)

    assert _tool_messages(result) == []  # nothing ran yet
    interrupt = result["__interrupt__"][0]
    request = interrupt.value["action_requests"][0]
    assert request["name"] == "run_python" and request["args"] == {"code": "print(6*7)"}
    assert "```python\nprint(6*7)\n```" in request["description"]
    assert request["description"].endswith(APPROVAL_WARNING)
    assert interrupt.value["review_configs"] == [
        {"action_name": "run_python", "allowed_decisions": ["approve", "reject"]}
    ]
    assert agent.get_state(CFG).next == ("HumanInTheLoopMiddleware.after_model",)

    resumed = agent.invoke(APPROVE, CFG)
    tool_message = _tool_messages(resumed)[0]
    assert tool_message.status == "success"
    assert "stdout:\n42" in tool_message.content
    assert resumed["messages"][-1].content == "6*7 is 42."
    assert "__interrupt__" not in resumed
    assert len(fake_mimoe.calls) == 2


def test_run_python_approval_reject(fake_mimoe: FakeMimoe, settings_tmp: Settings) -> None:
    agent, _ = _build(fake_mimoe, settings_tmp)
    fake_mimoe.script(RUN_PYTHON, {"content": "I did not run the code."})
    agent.invoke({"messages": [HumanMessage("Use run_python to print 6*7")]}, CFG)

    resumed = agent.invoke(
        Command(resume={"decisions": [{"type": "reject", "message": DECLINED}]}), CFG
    )
    tool_message = _tool_messages(resumed)[0]
    assert tool_message.status == "error" and tool_message.name == "run_python"
    assert "not executed" in tool_message.content and DECLINED in tool_message.content
    assert resumed["messages"][-1].content == "I did not run the code."
    # the model was told about the rejection and nothing was executed
    sent = fake_mimoe.calls[-1]["messages"][-1]
    assert sent["role"] == "tool" and "not executed" in sent["content"]

    fake_mimoe.script(RUN_PYTHON, {"content": "Skipped."})
    agent.invoke({"messages": [HumanMessage("again")]}, {"configurable": {"thread_id": "t2"}})
    bare = agent.invoke(
        Command(resume={"decisions": [{"type": "reject"}]}), {"configurable": {"thread_id": "t2"}}
    )
    assert "The tool was not executed." in _tool_messages(bare)[0].content
    assert bare["messages"][-1].content == "Skipped."


def test_auto_approve_runs_without_interrupt(fake_mimoe: FakeMimoe, settings_tmp: Settings) -> None:
    agent, _ = _build(fake_mimoe, dataclasses.replace(settings_tmp, auto_approve=True))
    fake_mimoe.script(RUN_PYTHON, {"content": "42"})
    result = agent.invoke({"messages": [HumanMessage("Use run_python to print 6*7")]}, CFG)
    assert "__interrupt__" not in result
    assert "stdout:\n42" in _tool_messages(result)[0].content
    assert result["messages"][-1].content == "42"


def test_chat_only_build_has_no_tools(
    fake_mimoe: FakeMimoe, settings_tmp: Settings, preflight_fake: Preflight
) -> None:
    chat_only = dataclasses.replace(preflight_fake, tools_enabled=False)
    agent, _ = _build(fake_mimoe, settings_tmp, chat_only)
    fake_mimoe.script({"content": "I cannot read files in this session."})
    result = agent.invoke({"messages": [HumanMessage("what is in notes.md?")]}, CFG)
    assert result["messages"][-1].content == "I cannot read files in this session."
    body = fake_mimoe.calls[-1]
    assert "tools" not in body and "tool_choice" not in body
    assert "Tools are disabled" in body["messages"][0]["content"]


def test_model_call_limit_ends_the_turn_with_a_notice(
    fake_mimoe: FakeMimoe, settings_tmp: Settings
) -> None:
    agent, _ = _build(fake_mimoe, settings_tmp)
    fake_mimoe.script(*([LIST_FILES] * 8))  # a 9th call would answer the fake's default 'OK.'
    result = agent.invoke({"messages": [HumanMessage("loop")]}, CFG)

    assert len(fake_mimoe.calls) == 8
    assert len(_tool_messages(result)) == 8
    final = result["messages"][-1]
    assert isinstance(final, AIMessage) and final.content == NOTICE
    assert agent.get_state(CFG).values["thread_model_call_count"] == 8
    assert agent.get_state(CFG).next == ()

    fake_mimoe.script({"content": "Back to normal."})  # the budget is per turn
    again = agent.invoke({"messages": [HumanMessage("hi")]}, CFG)
    assert again["messages"][-1].content == "Back to normal."
    assert agent.get_state(CFG).values["thread_model_call_count"] == 1


def test_model_call_limit_holds_across_approval_resumes(
    fake_mimoe: FakeMimoe, settings_tmp: Settings
) -> None:
    agent, _ = _build(fake_mimoe, settings_tmp)
    fake_mimoe.script(*([RUN_PYTHON] * 8))
    result = agent.invoke({"messages": [HumanMessage("loop")]}, CFG)
    approvals = 0
    while "__interrupt__" in result and approvals < 20:
        result = agent.invoke(APPROVE, CFG)
        approvals += 1

    assert approvals == 8 and len(fake_mimoe.calls) == 8
    assert result["messages"][-1].content == NOTICE
    assert all("stdout:\n42" in m.content for m in _tool_messages(result))


# -- async paths ---------------------------------------------------------------------------------


async def test_ainvoke_plain_answer_and_round_trip(
    fake_mimoe: FakeMimoe, settings_tmp: Settings
) -> None:
    agent, _ = _build(fake_mimoe, settings_tmp)
    fake_mimoe.script({"content": "Hello!"}, LIST_FILES, {"content": "Listed."})
    first = await agent.ainvoke({"messages": [HumanMessage("hi")]}, CFG)
    assert first["messages"][-1].content == "Hello!"
    second = await agent.ainvoke({"messages": [HumanMessage("files?")]}, CFG)
    assert _tool_messages(second)[0].name == "list_files"
    assert second["messages"][-1].content == "Listed."
    assert fake_mimoe.calls[-1]["messages"][-3]["content"] == "files? /no_think"


async def test_astream_approval_flow(fake_mimoe: FakeMimoe, settings_tmp: Settings) -> None:
    agent, _ = _build(fake_mimoe, settings_tmp)
    fake_mimoe.script(RUN_PYTHON, {"content": "6*7 is 42."})

    async def drain(payload: Any) -> tuple[list[Any], dict[str, Any]]:
        chunks: list[Any] = []
        updates: dict[str, Any] = {}
        async for mode, chunk in agent.astream(payload, CFG, stream_mode=["messages", "updates"]):
            if mode == "messages":
                chunks.append(chunk[0])
            else:
                updates.update(chunk)
        return chunks, updates

    chunks, updates = await drain({"messages": [HumanMessage("Use run_python to print 6*7")]})
    assert "__interrupt__" in updates
    assert updates["__interrupt__"][0].value["action_requests"][0]["name"] == "run_python"
    assert any(getattr(c, "tool_call_chunks", None) for c in chunks)
    assert updates["model"]["messages"][-1].tool_calls[0]["args"] == {"code": "print(6*7)"}
    assert updates["model"]["messages"][-1].content == ""  # cleaned before the checkpoint

    chunks, updates = await drain(APPROVE)
    assert "__interrupt__" not in updates
    tool_chunks = [c for c in chunks if isinstance(c, ToolMessage)]
    assert tool_chunks and "stdout:\n42" in tool_chunks[0].content
    assert "".join(c.content for c in chunks if isinstance(c, AIMessage)).endswith("6*7 is 42.")
    assert updates["model"]["messages"][-1].content == "6*7 is 42."
    assert fake_mimoe.calls[-1]["stream"] is True


async def test_v10_engine_reasoning_content_reaches_the_final_message(workspace_tmp: Path) -> None:
    fake = FakeMimoe("1.0")
    settings = load_settings(
        {"workspace": workspace_tmp, "base_url": fake.base_url}, cwd=workspace_tmp.parent
    )
    agent, pre = _build(fake, settings)
    assert pre.thinking_control == "native"
    fake.script({"content": "four", "reasoning": "two plus two"})

    pieces: list[str] = []
    final: AIMessage | None = None
    async for mode, chunk in agent.astream(
        {"messages": [HumanMessage("2+2?")]}, CFG, stream_mode=["messages", "updates"]
    ):
        if mode == "messages" and isinstance(chunk[0], AIMessage):
            piece = chunk[0].additional_kwargs.get(REASONING_KEY)
            if piece:
                pieces.append(piece)
        elif mode == "updates" and "model" in chunk:
            final = chunk["model"]["messages"][-1]

    assert "".join(pieces) == "two plus two"
    assert final is not None and final.content == "four"
    assert final.additional_kwargs[REASONING_KEY] == "two plus two"
    body = fake.calls[-1]
    assert body["enable_thinking"] is False and "/no_think" not in body["messages"][-1]["content"]
    assert isinstance(agent.nodes["model"], object)  # graph built around the subclass
    assert isinstance(make_model(settings, pre, http_client=fake.client()), MimoeChatOpenAI)


async def test_v10_thinking_answer_is_checkpointed_without_leading_whitespace(
    workspace_tmp: Path,
) -> None:
    fake = FakeMimoe("1.0")
    settings = load_settings(
        {"workspace": workspace_tmp, "base_url": fake.base_url, "think": True},
        cwd=workspace_tmp.parent,
    )
    agent, _ = _build(fake, settings)
    fake.script({"content": "\n\n7 times 8 is 56.", "reasoning": "seven times eight"})

    streamed: list[str] = []
    async for mode, chunk in agent.astream(
        {"messages": [HumanMessage("7*8?")]}, CFG, stream_mode=["messages", "updates"]
    ):
        if mode == "messages" and isinstance(chunk[0], AIMessage) and chunk[0].content:
            streamed.append(chunk[0].content)

    assert "".join(streamed) == "\n\n7 times 8 is 56."  # the raw token stream is untouched
    final = agent.get_state(CFG).values["messages"][-1]
    assert final.content == "7 times 8 is 56."  # the checkpoint matches the 0.6 path
    assert final.additional_kwargs[REASONING_KEY] == "seven times eight"
    assert fake.calls[-1]["enable_thinking"] is True


# -- a refused calculator call ------------------------------------------------------------------

PRIMES = "sum(1 for n in range(1000, 4501) if all(n % i != 0 for i in range(2, int(n**0.5) + 1)))"
PRIMES_CALL = {"tool_calls": [{"name": "calculator", "args": {"expression": PRIMES}}]}


def _check_refused_calculator(events: list[dict[str, Any]], agent: Any, fake: FakeMimoe) -> None:
    result = next(e for e in events if e["event"] == "tool_result")
    assert result["name"] == "calculator" and result["is_error"] is True
    assert "call run_python with code that prints the result" in result["content"]
    messages = agent.get_state(CFG).values["messages"]
    tool_message = next(m for m in messages if isinstance(m, ToolMessage))
    assert tool_message.status == "error"
    sent = fake.calls[-1]["messages"][-1]  # what the model read before its next step
    assert sent["role"] == "tool" and "call run_python" in sent["content"]


def test_a_refused_calculator_call_is_shown_as_failed_and_points_to_run_python(
    fake_mimoe: FakeMimoe, settings_tmp: Settings
) -> None:
    """The prime-number turn from a real web session: the call used to show as done."""
    agent, _ = _build(fake_mimoe, settings_tmp)
    fake_mimoe.script(PRIMES_CALL, {"content": "I will count them with run_python."})
    payload = {"messages": [HumanMessage("How many primes are there between 1000 and 4500?")]}
    _check_refused_calculator(list(iter_events(agent, payload, CFG)), agent, fake_mimoe)


async def test_a_refused_calculator_call_is_shown_as_failed_async(
    fake_mimoe: FakeMimoe, settings_tmp: Settings
) -> None:
    """Same through the async path the web server uses (``awrap_tool_call``)."""
    agent, _ = _build(fake_mimoe, settings_tmp)
    fake_mimoe.script(PRIMES_CALL, {"content": "I will count them with run_python."})
    payload = {"messages": [HumanMessage("How many primes are there between 1000 and 4500?")]}
    events = [e async for e in aiter_events(agent, payload, CFG)]
    _check_refused_calculator(events, agent, fake_mimoe)
