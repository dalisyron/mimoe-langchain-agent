"""Agent assembly: the system prompt now; ``build_agent`` arrives in step C3."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from mimoe_agent.config import Settings
from mimoe_agent.mimoe import Preflight

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.checkpoint.base import BaseCheckpointSaver

NO_THINK = "/no_think"

SYSTEM_PROMPT = (
    "You are a private local assistant for a developer's workspace folder. You can inspect files, "
    "run Python, query git history and report on the mimOE runtime using the tools provided. "
    "Call a tool only when it is needed to answer; for greetings or general questions answer "
    "directly. Never invent file contents or results.\n"
    "The workspace folder is {workspace}; tool paths are relative to it.\n"
    "Answer in at most five sentences unless asked for detail. Only state numbers that appear in "
    "a tool result; if a tool returns an error or empty output, fix the call and retry instead of "
    "guessing."
)
"""Tool-mode prompt. The first three sentences are the wording that scored 6/6 on tool selection
with qwen3-4b; ``{workspace}`` is filled in by :func:`system_prompt`."""

SYSTEM_PROMPT_CHAT_ONLY = (
    "You are a private local assistant for a developer's workspace folder at {workspace}. "
    "Tools are disabled in this session, so you cannot read files, run code or query git; answer "
    "from the conversation and say plainly when you would need a tool. Never invent file contents "
    "or results. Answer in at most five sentences unless asked for detail."
)
"""Prompt used when the preflight disabled tools (chat-only mode)."""


def system_prompt(settings: Settings, pre: Preflight) -> str:
    """Return the system prompt for this session.

    Picks the tool or chat-only wording from ``pre.tools_enabled``, fills in the workspace path and
    puts ``/no_think`` on the first line when thinking is off and the engine only understands the
    soft switch (``pre.thinking_control == "soft"``).
    """
    template = SYSTEM_PROMPT if pre.tools_enabled else SYSTEM_PROMPT_CHAT_ONLY
    text = template.replace("{workspace}", str(settings.workspace))
    if pre.thinking_control == "soft" and not settings.think:
        text = f"{NO_THINK}\n{text}"
    return text


def build_agent(
    settings: Settings,
    pre: Preflight,
    *,
    llm: BaseChatModel | None = None,
    tools: Sequence[BaseTool] | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> Any:
    """Assemble the LangChain agent (middleware, tools, checkpointer); implemented in step C3."""
    raise NotImplementedError("step C3")
