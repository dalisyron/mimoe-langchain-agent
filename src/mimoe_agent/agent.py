"""Agent assembly: the system prompt and ``build_agent``.

LangChain is imported inside :func:`build_agent` so that entry points can call
:func:`mimoe_agent.config.apply_tracing_env` before any LangChain module is loaded.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from mimoe_agent.config import Settings
from mimoe_agent.mimoe import Preflight

if TYPE_CHECKING:
    from langchain.agents.middleware import AgentState
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import ToolCall
    from langchain_core.tools import BaseTool
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.runtime import Runtime

NO_THINK = "/no_think"

SYSTEM_PROMPT = (
    "You are a private local assistant for a developer's workspace folder. You can inspect files, "
    "run Python, query git history and report on the mimOE runtime using the tools provided. "
    "Call a tool only when it is needed to answer; for greetings or general questions answer "
    "directly. Never invent file contents or results.\n"
    "The workspace folder is {workspace}; tool paths are relative to it.\n"
    "Answer in at most five sentences unless asked for detail. Never do arithmetic in your head, "
    "not even one multiplication: get every computed number from calculator (one expression) or "
    "run_python (ranges, counting, sums over many values) and state the result it returns. If a "
    "tool returns an error or empty output, fix the call and retry instead of guessing."
)
"""Tool-mode prompt. The first three sentences are the wording that scored 6/6 on tool selection
with qwen3-4b; ``{workspace}`` is filled in by :func:`system_prompt`.

The arithmetic rule replaced "Only state numbers that appear in a tool result", which
qwen3-4b-instruct-2507 did not take as a reason to call a tool: it worked 6 of 8 number questions
out in text, four of them wrong (a sum of multiples of 5 came out as 186,000,000 instead of
186,842,970). With the rule, 7 of 8 went to calculator or run_python, questions about facts that
contain numbers still got no tool, and the workspace questions picked the same tools as before
(first-step replays at temperature 0). A longer rule that listed sums, averages and percentages
did worse (4 of 10)."""

SYSTEM_PROMPT_CHAT_ONLY = (
    "You are a private local assistant for a developer's workspace folder at {workspace}. "
    "Tools are disabled in this session, so you cannot read files, run code or query git; answer "
    "from the conversation and say plainly when you would need a tool. Never invent file contents "
    "or results. Answer in at most five sentences unless asked for detail."
)
"""Prompt used when the preflight disabled tools (chat-only mode)."""

APPROVAL_WARNING = "this code runs as you, with your files"
"""The line every approval request ends with (CLI, web and the interrupt payload agree)."""
APPROVAL_TOOL = "run_python"
"""The only tool gated by human approval."""
_BACKTICK_RUNS = re.compile(r"`+")


def system_prompt(settings: Settings, pre: Preflight) -> str:
    """Return the system prompt for this session.

    Picks the tool or chat-only wording from ``pre.tools_enabled``, fills in the workspace path and
    puts ``/no_think`` on the first line when thinking is off and the engine only understands the
    soft switch (``pre.thinking_control == "soft"``).
    """
    template = SYSTEM_PROMPT if pre.tools_enabled else SYSTEM_PROMPT_CHAT_ONLY
    # Forward slashes even on Windows (they work there): a backslashed path in the prompt made
    # qwen3-4b write code with literal "\\n" sequences instead of line breaks.
    text = template.replace("{workspace}", settings.workspace.as_posix())
    if pre.thinking_control == "soft" and not settings.think:
        text = f"{NO_THINK}\n{text}"
    return text


def describe_run_python(tool_call: ToolCall, state: AgentState, runtime: Runtime) -> str:
    """Render a ``run_python`` approval request: the code in a fenced block plus the warning.

    Used as the ``description`` of the human-in-the-loop interrupt, i.e. what an API consumer
    reading the raw interrupt (``approval_required.action_requests[].description``) sees. The
    CLI and the web panel do not show this text: they render ``args["code"]`` themselves,
    escaped and checked for hidden characters.
    """
    code = str((tool_call.get("args") or {}).get("code", ""))
    # a snippet that itself contains backticks needs a fence longer than any run inside it
    longest_run = max((len(run) for run in _BACKTICK_RUNS.findall(code)), default=0)
    fence = "`" * max(3, longest_run + 1)
    return (
        f"{APPROVAL_TOOL} wants to run this code:\n\n"
        f"{fence}python\n{code}\n{fence}\n\n"
        f"{APPROVAL_WARNING}"
    )


def build_agent(
    settings: Settings,
    pre: Preflight,
    *,
    llm: BaseChatModel | None = None,
    tools: Sequence[BaseTool] | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Assemble the LangChain agent for one session.

    Args:
        settings: Effective settings (``think``, ``auto_approve``, ``workspace``...).
        pre: Preflight result (model id, thinking control, whether tools are enabled).
        llm: Chat model to use; defaults to :func:`mimoe_agent.llm.make_model`.
        tools: Tools to bind; defaults to :func:`mimoe_agent.tools.build_tools`. Ignored (no
            tools at all) when the preflight disabled tools.
        checkpointer: Checkpointer for threads and approval interrupts; defaults to an
            ``InMemorySaver``.

    Returns:
        The compiled agent graph. Middleware, outermost first: ``QwenMiddleware`` (think handling,
        ``/no_think``), ``GuardrailMiddleware`` (result caps, tool errors, per-turn budget reset),
        ``ModelCallLimitMiddleware(thread_limit=8, exit_behavior="end")`` and, unless
        ``settings.auto_approve``, ``HumanInTheLoopMiddleware`` on ``run_python`` with the decisions
        ``approve``/``reject``.
    """
    from langchain.agents import create_agent
    from langchain.agents.middleware import (
        HumanInTheLoopMiddleware,
        InterruptOnConfig,
        ModelCallLimitMiddleware,
    )
    from langgraph.checkpoint.memory import InMemorySaver

    from mimoe_agent.llm import make_model
    from mimoe_agent.middleware import GuardrailMiddleware, QwenMiddleware
    from mimoe_agent.tools import build_tools

    model = llm if llm is not None else make_model(settings, pre)
    agent_tools: list[BaseTool]
    if not pre.tools_enabled:
        agent_tools = []
    elif tools is not None:
        agent_tools = list(tools)
    else:
        agent_tools = build_tools(settings)

    guard = GuardrailMiddleware()
    middleware: list[Any] = [
        QwenMiddleware(soft_no_think=pre.thinking_control == "soft" and not settings.think),
        guard,
        ModelCallLimitMiddleware(thread_limit=guard.thread_limit, exit_behavior="end"),
    ]
    if not settings.auto_approve:
        middleware.append(
            HumanInTheLoopMiddleware(
                interrupt_on={
                    APPROVAL_TOOL: InterruptOnConfig(
                        allowed_decisions=["approve", "reject"],
                        description=describe_run_python,
                    )
                }
            )
        )
    return create_agent(
        model,
        tools=agent_tools,
        system_prompt=system_prompt(settings, pre),
        middleware=middleware,
        checkpointer=checkpointer if checkpointer is not None else InMemorySaver(),
    )
