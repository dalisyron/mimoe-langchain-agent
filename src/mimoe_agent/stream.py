"""Turn an agent run into the UI event stream: ``iter_events`` / ``aiter_events`` (CONTRACTS.md).

Both functions drive ``agent.stream``/``agent.astream`` with ``stream_mode=["messages", "updates"]``
and map what langgraph 1.2 yields for a ``create_agent`` graph (verified against both engine
generations through the test fake and live):

* ``("messages", (chunk, metadata))``: ``AIMessageChunk`` deltas of the ``model`` node (every
  chunk of one model call shares an ``lc_run--...`` id), and ``ToolMessage`` results from the
  ``tools`` node or from ``HumanInTheLoopMiddleware.after_model`` (a rejected call).
* ``("updates", {node: update})``: the complete ``AIMessage`` of the ``model`` node (tool calls,
  ``usage_metadata``, ``response_metadata["model_name"]``), ``{"__interrupt__": (Interrupt, ...)}``
  for an approval, and the limit ``AIMessage`` a middleware node injects (``jump_to: "end"``).
  An update is ``None``, a dict, or a list of dicts when a node wrote several times.

Events are plain dicts with an ``"event"`` key:
``token{text}``, ``thinking{text}``, ``tool_call{id,name,args}``,
``tool_result{id,name,content,is_error}``,
``approval_required{interrupt_id,action_requests,review_configs}``, ``notice{text}``,
``done{status,elapsed_s,model,usage_total{input_tokens,output_tokens,llm_calls}}`` and
``error{message,hint}`` (no ``done`` after an error).

Reasoning reaches the stream two ways and :class:`ThinkSplitter` turns both into ``thinking``
events: inline ``<think>...</think>`` text on 0.6 engines (an empty block always precedes the
answer with ``/no_think``; tags may be split across chunks) and
``additional_kwargs["reasoning_content"]`` deltas that ``MimoeChatOpenAI`` copies from the 1.0
engines' ``delta.reasoning_content``.
"""

from __future__ import annotations

import copy
import time
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from mimoe_agent.mimoe import friendly_error

STREAM_MODES: tuple[str, ...] = ("messages", "updates")
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
MODEL_NODE = "model"
INTERRUPT_KEY = "__interrupt__"
REASONING_KEY = "reasoning_content"
UNKNOWN_MODEL = "unknown"

Event = dict[str, Any]


class ThinkSplitter:
    """Classify one model call's deltas into ``thinking`` and ``token`` events.

    Stateful and meant to live for exactly one model call: create a fresh instance when the next
    call starts (``iter_events`` does so on every new chunk id and every ``model`` update).

    Handles both reasoning transports: inline ``<think>...</think>`` text in the content stream,
    whose tags may arrive split across chunks (``"<thi"`` + ``"nk>"``), and ``reasoning_content``
    deltas (:meth:`feed_reasoning`, or :meth:`feed_chunk` reading ``additional_kwargs``). An empty
    block emits nothing, the reasoning is trimmed at both ends but keeps its interior whitespace,
    and the visible answer's leading whitespace (the ``"\\n\\n"`` after the block) is dropped.
    Only a leading block is reasoning, exactly what ``QwenMiddleware`` strips from the checkpoint:
    a ``<think>`` inside the visible answer (a model quoting the tag) stays answer text.
    """

    def __init__(self) -> None:
        self._buf = ""
        """Content not classified yet; may end with the prefix of a tag."""
        self._in_think = False
        self._think_started = False
        self._pending_ws = ""
        """Whitespace inside the reasoning held back until more text follows it."""
        self._visible_started = False

    @property
    def in_think(self) -> bool:
        """Whether the last classified text was inside a ``<think>`` block."""
        return self._in_think

    def feed(self, text: str) -> list[Event]:
        """Classify a content delta and return the events it completes (possibly none).

        Text that could be the start of a tag is held back until the next delta or :meth:`flush`;
        once visible text has started nothing is held back, because a block can no longer open.
        """
        events: list[Event] = []
        if not text:
            return events
        self._buf += text
        while self._buf:
            if self._in_think:
                at = self._buf.find(THINK_CLOSE)
                if at >= 0:
                    self._emit(events, self._buf[:at])
                    self._buf = self._buf[at + len(THINK_CLOSE) :]
                    self._in_think = False
                    self._pending_ws = ""  # the block's trailing whitespace is dropped
                    self._think_started = False
                    continue
                keep = _partial_tag_len(self._buf, THINK_CLOSE)
            else:
                at = self._buf.find(THINK_OPEN)
                if at >= 0 and self._before_answer(self._buf[:at]):
                    self._emit(events, self._buf[:at])  # whitespace only, dropped
                    self._buf = self._buf[at + len(THINK_OPEN) :]
                    self._in_think = True
                    continue
                keep = _partial_tag_len(self._buf, THINK_OPEN)
                if not self._before_answer(self._buf[: len(self._buf) - keep]):
                    keep = 0  # the answer has started: a literal "<" is text, not a tag
            self._emit(events, self._buf[: len(self._buf) - keep])
            self._buf = self._buf[len(self._buf) - keep :]
            break
        return events

    def feed_reasoning(self, text: str) -> list[Event]:
        """Return the ``thinking`` events for a ``reasoning_content`` delta."""
        events: list[Event] = []
        self._emit_thinking(events, text)
        return events

    def feed_chunk(self, chunk: BaseMessage) -> list[Event]:
        """Map one streamed message chunk: ``additional_kwargs["reasoning_content"]`` becomes
        ``thinking`` text, then the text content goes through :meth:`feed`."""
        events: list[Event] = []
        reasoning = chunk.additional_kwargs.get(REASONING_KEY)
        if isinstance(reasoning, str):
            self._emit_thinking(events, reasoning)
        events.extend(self.feed(message_text(chunk)))
        return events

    def flush(self) -> list[Event]:
        """End the model call: release held-back text (an unclosed block stays ``thinking``)."""
        events: list[Event] = []
        if self._buf:
            self._emit(events, self._buf)
            self._buf = ""
        self._pending_ws = ""
        return events

    # -- internals -----------------------------------------------------------------------------

    def _before_answer(self, preceding: str) -> bool:
        """Whether no visible text exists yet, ``preceding`` (text before a tag) included."""
        return not self._visible_started and not preceding.strip()

    def _emit(self, events: list[Event], text: str) -> None:
        if self._in_think:
            self._emit_thinking(events, text)
        else:
            self._emit_token(events, text)

    def _emit_thinking(self, events: list[Event], text: str) -> None:
        if not text:
            return
        if not self._think_started:
            text = text.lstrip()
            if not text:
                return
            self._think_started = True
        text = self._pending_ws + text
        kept = text.rstrip()
        self._pending_ws = text[len(kept) :]
        if kept:
            events.append({"event": "thinking", "text": kept})

    def _emit_token(self, events: list[Event], text: str) -> None:
        if not text:
            return
        if not self._visible_started:
            text = text.lstrip()
            if not text:
                return
            self._visible_started = True
        events.append({"event": "token", "text": text})


def message_text(message: BaseMessage) -> str:
    """Return the text of a message: its string content, or the ``text`` blocks joined."""
    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, Mapping) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def visible_text(message: BaseMessage) -> str:
    """Return a message's text with every ``<think>`` block removed (leading whitespace too)."""
    splitter = ThinkSplitter()
    events = splitter.feed(message_text(message)) + splitter.flush()
    return "".join(event["text"] for event in events if event["event"] == "token")


def iter_events(agent: Any, payload: Any, config: RunnableConfig | None = None) -> Iterator[Event]:
    """Run the agent synchronously and yield UI events in order.

    Args:
        agent: The compiled ``create_agent`` graph.
        payload: ``{"messages": [HumanMessage]}`` for a new turn or ``Command(resume=...)``.
        config: The run config (``{"configurable": {"thread_id": ...}}``).

    Yields:
        Event dicts, ending with ``done`` (``status`` ``"completed"`` or ``"awaiting_approval"``)
        or with ``error`` when anything raised (``mimoe.friendly_error`` supplies the hint).
        A consumer that stops early (``close()``) closes the graph run with it.
    """
    mapper = _EventMapper(agent, config)
    stream: Iterator[Any] | None = None
    try:
        stream = agent.stream(payload, config, stream_mode=list(STREAM_MODES))
        for mode, data in stream:
            yield from mapper.handle(mode, data)
        yield from mapper.finish()
    except Exception as exc:
        yield from mapper.fail(exc)
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            close()


async def aiter_events(
    agent: Any, payload: Any, config: RunnableConfig | None = None
) -> AsyncIterator[Event]:
    """Async twin of :func:`iter_events` built on ``agent.astream`` (same events, same order).

    ``asyncio.CancelledError`` (a client that went away) is not an ``Exception`` and propagates.
    ``aclose()`` on this generator closes the graph run before it returns, instead of leaving
    the run's teardown to the event loop's garbage-collection hook.
    """
    mapper = _EventMapper(agent, config)
    stream: AsyncIterator[Any] | None = None
    try:
        stream = agent.astream(payload, config, stream_mode=list(STREAM_MODES))
        async for mode, data in stream:
            for event in mapper.handle(mode, data):
                yield event
        for event in mapper.finish():
            yield event
    except Exception as exc:
        for event in mapper.fail(exc):
            yield event
    finally:
        aclose = getattr(stream, "aclose", None)
        if aclose is not None:
            await aclose()


# -- run state -------------------------------------------------------------------------------------


@dataclass
class _ModelCall:
    """Splitter state of the model call currently streaming."""

    id: str | None
    splitter: ThinkSplitter = field(default_factory=ThinkSplitter)
    emitted_token: bool = False
    emitted_thinking: bool = False

    def note(self, events: list[Event]) -> list[Event]:
        for event in events:
            if event["event"] == "token":
                self.emitted_token = True
            elif event["event"] == "thinking":
                self.emitted_thinking = True
        return events


class _EventMapper:
    """Shared state machine behind ``iter_events`` and ``aiter_events``."""

    def __init__(self, agent: Any, config: Mapping[str, Any] | None) -> None:
        self._started = time.perf_counter()
        self._configured_model = _configured_model(agent, config)
        self._response_model: str | None = None
        self._meta_model: str | None = None
        self._usage = {"input_tokens": 0, "output_tokens": 0, "llm_calls": 0}
        self._call: _ModelCall | None = None
        self._tool_calls_seen: set[tuple[str | None, str | None]] = set()
        self._tool_results_seen: set[str] = set()
        self._notices_seen: set[str] = set()
        self._awaiting = False

    def handle(self, mode: str, data: Any) -> list[Event]:
        """Map one ``(mode, data)`` item of the graph stream to zero or more events."""
        if mode == "messages":
            return self._on_message(data)
        if mode == "updates":
            return self._on_updates(data)
        return []

    def finish(self) -> list[Event]:
        """Events for a stream that ended normally: leftovers, then ``done``."""
        return [*self._close_call(), self._done()]

    def fail(self, exc: BaseException) -> list[Event]:
        """Events for a stream that raised: leftovers, then ``error`` (never ``done``)."""
        message, hint = friendly_error(exc)
        return [*self._close_call(), {"event": "error", "message": message, "hint": hint}]

    # -- messages mode -----------------------------------------------------------------------

    def _on_message(self, data: Any) -> list[Event]:
        if not isinstance(data, Sequence) or len(data) != 2:
            return []
        message, meta = data
        if not isinstance(meta, Mapping):
            meta = {}
        if isinstance(message, ToolMessage):
            return self._tool_result(message)
        if not isinstance(message, AIMessageChunk) or meta.get("langgraph_node") != MODEL_NODE:
            return []
        events: list[Event] = []
        if self._call is None or (message.id is not None and message.id != self._call.id):
            events.extend(self._close_call())
            self._call = _ModelCall(id=message.id)
        model_name = meta.get("ls_model_name")
        if isinstance(model_name, str) and model_name:
            self._meta_model = model_name
        events.extend(self._call.note(self._call.splitter.feed_chunk(message)))
        return events

    # -- updates mode ------------------------------------------------------------------------

    def _on_updates(self, data: Any) -> list[Event]:
        if not isinstance(data, Mapping):
            return []
        events: list[Event] = []
        for node, update in data.items():
            if node == INTERRUPT_KEY:
                events.extend(self._interrupts(update))
                continue
            for part in _update_parts(update):
                for message in _messages_of(part):
                    if isinstance(message, ToolMessage):
                        events.extend(self._tool_result(message))
                    elif isinstance(message, AIMessage) and node == MODEL_NODE:
                        events.extend(self._model_message(message))
                    elif isinstance(message, AIMessage):
                        events.extend(self._notice(message))
        return events

    def _interrupts(self, update: Any) -> list[Event]:
        items = list(update) if isinstance(update, Sequence | set) else [update]
        events: list[Event] = []
        for interrupt in items:
            value = getattr(interrupt, "value", None)
            payload = value if isinstance(value, Mapping) else {}
            events.append(
                {
                    "event": "approval_required",
                    "interrupt_id": getattr(interrupt, "id", None),
                    "action_requests": copy.deepcopy(list(payload.get("action_requests") or [])),
                    "review_configs": copy.deepcopy(list(payload.get("review_configs") or [])),
                }
            )
        self._awaiting = True
        return events

    def _model_message(self, message: AIMessage) -> list[Event]:
        """The model node finished a call: flush its splitter, account usage, emit tool calls.

        When nothing visible (or no reasoning) was streamed for the call but the final message
        carries some, it is emitted now, so a non-streaming model or a middleware that rewrote the
        reply (``QwenMiddleware``'s malformed-tool-call note) still reaches the UI.
        """
        call = self._call if self._call is not None else _ModelCall(id=message.id)
        self._call = None
        events = call.note(call.splitter.flush())
        if not (call.emitted_token and call.emitted_thinking):
            catch_up = ThinkSplitter()
            for event in catch_up.feed_chunk(message) + catch_up.flush():
                streamed = event["event"] == "token" and call.emitted_token
                streamed |= event["event"] == "thinking" and call.emitted_thinking
                if not streamed:
                    events.append(event)
        self._usage["llm_calls"] += 1
        usage = message.usage_metadata
        if usage:
            self._usage["input_tokens"] += _int(usage.get("input_tokens"))
            self._usage["output_tokens"] += _int(usage.get("output_tokens"))
        model_name = message.response_metadata.get("model_name") or message.response_metadata.get(
            "model"
        )
        if isinstance(model_name, str) and model_name:
            self._response_model = model_name
        for call_spec in message.tool_calls:
            key = (message.id, call_spec.get("id"))
            if key in self._tool_calls_seen:
                continue
            self._tool_calls_seen.add(key)
            events.append(
                {
                    "event": "tool_call",
                    "id": call_spec.get("id"),
                    "name": call_spec.get("name"),
                    "args": dict(call_spec.get("args") or {}),
                }
            )
        return events

    def _notice(self, message: AIMessage) -> list[Event]:
        """An ``AIMessage`` a middleware node injected (a limit message) becomes a ``notice``.

        The ``HumanInTheLoopMiddleware.after_model`` update re-emits the model's own tool-call
        message on resume; messages with tool calls are therefore never notices.
        """
        if message.tool_calls:
            return []
        text = visible_text(message)
        if not text:
            return []
        key = message.id or text
        if key in self._notices_seen:
            return []
        self._notices_seen.add(key)
        return [{"event": "notice", "text": text}]

    # -- shared --------------------------------------------------------------------------------

    def _tool_result(self, message: ToolMessage) -> list[Event]:
        key = message.id or f"{message.tool_call_id}\0{message_text(message)}"
        if key in self._tool_results_seen:
            return []
        self._tool_results_seen.add(key)
        return [
            {
                "event": "tool_result",
                "id": message.tool_call_id,
                "name": message.name,
                "content": message_text(message),
                "is_error": message.status == "error",
            }
        ]

    def _close_call(self) -> list[Event]:
        if self._call is None:
            return []
        call, self._call = self._call, None
        return call.note(call.splitter.flush())

    def _done(self) -> Event:
        return {
            "event": "done",
            "status": "awaiting_approval" if self._awaiting else "completed",
            "elapsed_s": round(time.perf_counter() - self._started, 3),
            "model": self._response_model or self._meta_model or self._configured_model,
            "usage_total": dict(self._usage),
        }


# -- helpers ---------------------------------------------------------------------------------------


def _partial_tag_len(text: str, tag: str) -> int:
    """Length of the longest proper prefix of ``tag`` that ``text`` ends with (0 if none)."""
    for length in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:length]):
            return length
    return 0


def _update_parts(update: Any) -> list[Mapping[str, Any]]:
    """Normalise a node update (``None``, a dict, or a list of dicts) to a list of dicts."""
    if isinstance(update, Mapping):
        return [update]
    if isinstance(update, Sequence) and not isinstance(update, str):
        return [part for part in update if isinstance(part, Mapping)]
    return []


def _messages_of(part: Mapping[str, Any]) -> list[BaseMessage]:
    messages = part.get("messages")
    if isinstance(messages, BaseMessage):
        return [messages]
    if isinstance(messages, Sequence) and not isinstance(messages, str):
        return [message for message in messages if isinstance(message, BaseMessage)]
    return []


def _configured_model(agent: Any, config: Mapping[str, Any] | None) -> str:
    """Model name from ``config["metadata"]``, ``config["configurable"]`` or the agent."""
    sources: list[Any] = []
    if isinstance(config, Mapping):
        sources.extend((config.get("metadata"), config.get("configurable")))
    sources.append(getattr(agent, "metadata", None))
    for source in sources:
        if isinstance(source, Mapping):
            name = source.get("model")
            if isinstance(name, str) and name:
                return name
    return UNKNOWN_MODEL


def _int(value: object) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return 0
