"""Agent middleware: Qwen reasoning normalisation and guard rails.

Both classes implement the sync and the async variant of every hook they use: ``create_agent``
registers a middleware for the async path as soon as it overrides the sync hook, and the base
async implementation raises ``NotImplementedError`` under ``ainvoke``/``astream`` (verified in
langchain 1.4.2, ``factory.py`` and ``middleware/types.py``).

Reasoning arrives in two shapes: 0.6 engines put it inline as ``<think>...</think>`` at the start
of ``content`` (an empty block always precedes the answer with ``/no_think``), 1.0 engines put it
in ``reasoning_content``, which :class:`mimoe_agent.llm.MimoeChatOpenAI` stores in
``additional_kwargs["reasoning_content"]``. :class:`QwenMiddleware` folds the inline shape into the
same key, so the rest of the program sees one representation and the checkpoint holds only the
visible answer.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain.agents.middleware.model_call_limit import ModelCallLimitState
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphBubbleUp
from langgraph.runtime import Runtime
from langgraph.types import Command

THINK_RE: re.Pattern[str] = re.compile(r"^\s*<think>(.*?)</think>\s*", re.DOTALL)
"""A complete leading think block; group 1 is the reasoning text."""
THINK_OPEN_RE: re.Pattern[str] = re.compile(r"^\s*<think>(.*)\Z", re.DOTALL)
"""An unclosed leading think block (generation cut off inside the reasoning)."""
REASONING_KEY = "reasoning_content"
"""``additional_kwargs`` key that langchain-core exposes as a ``reasoning`` content block."""
NO_THINK_SUFFIX = " /no_think"
MALFORMED_TOOL_CALL = "I produced a malformed tool call; please rephrase."
TRUNCATED_TAIL = "[truncated: {total:,} chars total; use offset/limit]"
OLD_RESULT_TAIL = "[older tool output trimmed]"
ERROR_DETAIL_CAP = 2_000
"""Characters of an exception message kept in the error ``ToolMessage``."""

_EXIT_CODE_RE = re.compile(r"exit_code: (\S+)")


def reports_failure(content: str) -> bool:
    """Whether a tool's text result says the call failed.

    The tools return ``ERROR: ...`` (the workspace tools ``error: ...``) instead of raising, so
    the model can read what went wrong and the loop continues; ``run_python`` starts its result
    with ``exit_code: N``, and anything but a clean 0 is a failed or killed run.
    """
    head = content.lstrip()
    if head[:6].lower() == "error:":
        return True
    first_line = head.split("\n", 1)[0]
    match = _EXIT_CODE_RE.match(first_line)
    return bool(match) and (match.group(1) != "0" or "(killed:" in first_line)


def split_think(text: str) -> tuple[str, str]:
    """Split a leading ``<think>`` block off ``text``.

    Args:
        text: Raw assistant content.

    Returns:
        ``(reasoning, visible)``: the stripped reasoning text (``""`` when there is no block or
        the block is empty) and the content that follows the block. An unclosed block counts as
        reasoning only, so a generation cut off mid-thought yields an empty visible answer.
    """
    match = THINK_RE.match(text) or THINK_OPEN_RE.match(text)
    if match is None:
        return "", text
    return match.group(1).strip(), text[match.end() :]


def clean_ai_message(message: AIMessage) -> AIMessage:
    """Return ``message`` with inline reasoning moved to ``additional_kwargs`` and guarded.

    Inline ``<think>`` text is appended to ``additional_kwargs["reasoning_content"]`` (after any
    value the server already set) and stripped from ``content``. When the server itself supplied
    the reasoning (1.0 engines), the whitespace its chat template leaves between the reasoning
    and the answer (``"\\n\\n"``) is dropped too, as the regex does after ``</think>``, so both
    engine generations checkpoint the same visible text. A message whose only tool calls are
    malformed (``invalid_tool_calls`` and no ``tool_calls``) is replaced by a visible note and
    the invalid calls are cleared, otherwise ``ChatOpenAI`` would send them back as dangling
    ``tool_calls`` on the next request. Unchanged messages are returned as-is.
    """
    updates: dict[str, Any] = {}
    extra = dict(message.additional_kwargs)
    if isinstance(message.content, str):
        reasoning, visible = split_think(message.content)
        server_reasoning = extra.get(REASONING_KEY)
        if isinstance(server_reasoning, str) and server_reasoning:
            visible = visible.lstrip()
        if visible != message.content:
            updates["content"] = visible
            if reasoning:
                existing = extra.get(REASONING_KEY)
                if isinstance(existing, str) and existing:
                    reasoning = f"{existing}\n{reasoning}"
                extra[REASONING_KEY] = reasoning
                updates["additional_kwargs"] = extra
    if message.invalid_tool_calls and not message.tool_calls:
        extra.pop("tool_calls", None)
        extra.pop("function_call", None)
        updates.update(content=MALFORMED_TOOL_CALL, invalid_tool_calls=[], additional_kwargs=extra)
    if not updates:
        return message
    return message.model_copy(update=updates)


def _with_no_think(message: HumanMessage) -> HumanMessage:
    """Append ``/no_think`` to a human message (string content or a list of blocks)."""
    content = message.content
    if isinstance(content, str):
        if content.rstrip().endswith("/no_think"):
            return message
        return message.model_copy(update={"content": content.rstrip() + NO_THINK_SUFFIX})
    blocks = list(content)
    blocks.append({"type": "text", "text": NO_THINK_SUFFIX.strip()})
    return message.model_copy(update={"content": blocks})


class QwenMiddleware(AgentMiddleware):
    """Normalise Qwen replies from either engine generation.

    Request side (only when ``soft_no_think``): ``" /no_think"`` is appended to the latest
    ``HumanMessage`` of the request; the checkpoint never sees it. Response side: see
    :func:`clean_ai_message`.
    """

    def __init__(self, soft_no_think: bool) -> None:
        """Create the middleware.

        Args:
            soft_no_think: Whether the engine only understands the ``/no_think`` soft switch
                and thinking should be off (``pre.thinking_control == "soft"`` and not
                ``settings.think``).
        """
        super().__init__()
        self.soft_no_think = soft_no_think

    def prepare(self, request: ModelRequest) -> ModelRequest:
        """Return the request the model should see (``/no_think`` on the latest user turn)."""
        if not self.soft_no_think:
            return request
        messages: list[AnyMessage] = list(request.messages)
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if isinstance(message, HumanMessage):
                messages[index] = _with_no_think(message)
                return request.override(messages=messages)
        return request

    @staticmethod
    def clean(response: ModelResponse) -> ModelResponse:
        """Return the response with every ``AIMessage`` passed through :func:`clean_ai_message`."""
        result = [
            clean_ai_message(message) if isinstance(message, AIMessage) else message
            for message in response.result
        ]
        return ModelResponse(result=result, structured_response=response.structured_response)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Sync hook: patch the request, call the model, clean the reply."""
        return self.clean(handler(self.prepare(request)))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async hook: same as :meth:`wrap_model_call`."""
        return self.clean(await handler(self.prepare(request)))


COMPUTE_REMINDER = " (Use calculator or run_python for any arithmetic.)"
"""Added after a question that contains a digit, in the request only (see
:class:`ComputeReminderMiddleware`)."""


def _text(message: HumanMessage) -> str:
    """The text of a human message, whether its content is a string or a list of blocks."""
    if isinstance(message.content, str):
        return message.content
    return " ".join(
        str(block.get("text", "")) if isinstance(block, dict) else str(block)
        for block in message.content
    )


class ComputeReminderMiddleware(AgentMiddleware):
    """Remind the model, next to a question that contains a number, to compute with a tool.

    The system prompt's arithmetic rule moved larger calculations to calculator or run_python,
    but qwen3-4b-instruct-2507 still answered small ones from memory: "What is the sum of 10 to
    20?" came out as 155 (it is 165), none of 7 small-arithmetic questions went to a tool, and two
    rewordings of the rule moved at most one. With :data:`COMPUTE_REMINDER` after the question, 5 of
    7 went to a tool (the other two were answered correctly), the larger calculations stayed on
    tools, questions that merely contain numbers ("Who won the 2018 World Cup?") got no tool, and
    no workspace question changed tool or arguments (first-step replays at temperature 0). A
    longer reminder that began "If this needs any calculation" moved only 2 of 7.

    Request side only, on the first model call of a turn (the request ends with the user's
    message); the checkpoint never sees it. ``build_agent`` lists it before
    :class:`QwenMiddleware`, so the reminder lands before the ``/no_think`` that one appends.
    """

    def prepare(self, request: ModelRequest) -> ModelRequest:
        """Return the request with the reminder after a question that contains a digit."""
        messages: list[AnyMessage] = list(request.messages)
        if not messages or not isinstance(messages[-1], HumanMessage):
            return request
        question = messages[-1]
        text = _text(question)
        if not any(char.isdigit() for char in text) or COMPUTE_REMINDER.strip() in text:
            return request
        if isinstance(question.content, str):
            content: Any = question.content.rstrip() + COMPUTE_REMINDER
        else:
            content = [*question.content, {"type": "text", "text": COMPUTE_REMINDER.strip()}]
        messages[-1] = question.model_copy(update={"content": content})
        return request.override(messages=messages)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Sync hook: add the reminder, call the model."""
        return handler(self.prepare(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async hook: same as :meth:`wrap_model_call`."""
        return await handler(self.prepare(request))


class GuardrailMiddleware(AgentMiddleware[ModelCallLimitState[Any], Any, Any]):
    """Budget and size guard rails around tool calls and the model request.

    * ``before_agent`` resets ``thread_model_call_count`` to zero. It runs once per user message
      and not on ``Command(resume=...)``, so ``ModelCallLimitMiddleware(thread_limit=...)`` becomes
      a per-turn budget that survives approval round trips (``run_limit`` would be wiped by every
      resume, verified in the plan-v2 stack check).
    * ``wrap_tool_call`` / ``awrap_tool_call`` cap every string tool result at ``result_cap``
      characters with a ``[truncated: N chars total; use offset/limit]`` tail and turn any
      exception raised by a tool into a ``ToolMessage(status="error")`` so the loop continues.
      A result whose text reports a failure (:func:`reports_failure`) is marked
      ``status="error"`` too, so the clients show it as failed; the model only ever sees the
      text. LangGraph control-flow signals (interrupts) propagate untouched.
    * ``wrap_model_call`` / ``awrap_model_call`` shorten ``ToolMessage``s from earlier turns (before
      the latest ``HumanMessage``) to ``old_result_cap`` characters in the request only; the
      checkpoint keeps the full text.
    """

    state_schema = ModelCallLimitState  # type: ignore[assignment]

    def __init__(
        self, result_cap: int = 8000, old_result_cap: int = 400, thread_limit: int = 8
    ) -> None:
        """Create the middleware.

        Args:
            result_cap: Maximum characters of a tool result handed to the model.
            old_result_cap: Characters kept from tool results of earlier turns.
            thread_limit: Model calls allowed per user turn; ``build_agent`` passes it to
                ``ModelCallLimitMiddleware`` so the budget has a single source of truth.
        """
        super().__init__()
        self.result_cap = result_cap
        self.old_result_cap = old_result_cap
        self.thread_limit = thread_limit

    # -- per-turn budget ---------------------------------------------------------------------

    def before_agent(
        self, state: ModelCallLimitState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Reset the per-thread model-call counter at the start of every user turn."""
        return {"thread_model_call_count": 0}

    async def abefore_agent(
        self, state: ModelCallLimitState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Async variant of :meth:`before_agent`."""
        return self.before_agent(state, runtime)

    # -- tool results ------------------------------------------------------------------------

    def cap_result(self, result: ToolMessage | Command[Any]) -> ToolMessage | Command[Any]:
        """Truncate an oversized string ``ToolMessage``; everything else passes through."""
        if not isinstance(result, ToolMessage) or not isinstance(result.content, str):
            return result
        total = len(result.content)
        if total <= self.result_cap:
            return result
        tail = TRUNCATED_TAIL.format(total=total)
        return result.model_copy(update={"content": f"{result.content[: self.result_cap]}\n{tail}"})

    @staticmethod
    def mark_failure(result: ToolMessage | Command[Any]) -> ToolMessage | Command[Any]:
        """Set ``status="error"`` on a ``ToolMessage`` whose text reports a failure."""
        if (
            isinstance(result, ToolMessage)
            and result.status != "error"
            and isinstance(result.content, str)
            and reports_failure(result.content)
        ):
            return result.model_copy(update={"status": "error"})
        return result

    @staticmethod
    def error_message(request: ToolCallRequest, exc: BaseException) -> ToolMessage:
        """Build the error ``ToolMessage`` for an exception raised by a tool."""
        name = request.tool.name if request.tool is not None else request.tool_call["name"]
        detail = str(exc)[:ERROR_DETAIL_CAP] or "no details"
        return ToolMessage(
            content=f"ERROR: {name} failed with {type(exc).__name__}: {detail}",
            tool_call_id=request.tool_call["id"],
            name=name,
            status="error",
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Sync hook: run the tool, cap its result, flag reported failures, convert exceptions."""
        try:
            return self.mark_failure(self.cap_result(handler(request)))
        except GraphBubbleUp:
            raise
        except Exception as exc:
            return self.error_message(request, exc)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Async hook: same as :meth:`wrap_tool_call`."""
        try:
            return self.mark_failure(self.cap_result(await handler(request)))
        except GraphBubbleUp:
            raise
        except Exception as exc:
            return self.error_message(request, exc)

    # -- model request -----------------------------------------------------------------------

    def trim_old_results(self, request: ModelRequest) -> ModelRequest:
        """Return the request with tool results from earlier turns shortened."""
        messages: list[AnyMessage] = list(request.messages)
        last_human = next(
            (i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], HumanMessage)),
            -1,
        )
        changed = False
        for index in range(max(last_human, 0)):
            message = messages[index]
            if (
                isinstance(message, ToolMessage)
                and isinstance(message.content, str)
                and len(message.content) > self.old_result_cap
            ):
                shortened = f"{message.content[: self.old_result_cap]}\n{OLD_RESULT_TAIL}"
                messages[index] = message.model_copy(update={"content": shortened})
                changed = True
        return request.override(messages=messages) if changed else request

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Sync hook: call the model with old tool results trimmed."""
        return handler(self.trim_old_results(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async hook: same as :meth:`wrap_model_call`."""
        return await handler(self.trim_old_results(request))
