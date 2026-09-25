"""``ChatOpenAI`` factory pinned to mimOE's chat-completions endpoint.

Verified against langchain-openai 1.6.6: ``max_tokens`` (and ``model_kwargs["max_tokens"]``) is
rewritten to ``max_completion_tokens``, which mimOE ignores, so the cap travels in ``extra_body``;
``use_responses_api=False`` short-circuits every Responses-API heuristic; ``stream_usage=True`` adds
``stream_options={"include_usage": true}`` to streaming requests only.

langchain-openai also never reads ``reasoning_content`` (its class docstring says so, and neither
``_convert_dict_to_message`` nor ``_convert_delta_to_message_chunk`` looks at the key), while the
1.0 engines deliver every reasoning token there. :class:`MimoeChatOpenAI` closes that gap.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Iterator, Mapping
from typing import Any

import httpx
import openai
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI

from mimoe_agent.config import Settings
from mimoe_agent.mimoe import HINT_STREAM_CUT, MimoeError, Preflight

MAX_TOKENS = 1024
MAX_TOKENS_THINK = 4096
REQUEST_TIMEOUT_S = 200.0
"""Above mimOE's own 180 s execution timeout, so the engine's error arrives before ours."""
REASONING_KEY = "reasoning_content"
"""``additional_kwargs`` key; langchain-core renders it as a ``reasoning`` content block."""


def _reasoning_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _delta_reasoning(chunk: Mapping[str, Any]) -> str | None:
    """Return ``delta.reasoning_content`` of a streamed chunk, if any."""
    choices = chunk.get("choices") or (chunk.get("chunk") or {}).get("choices") or []
    if not choices:
        return None
    delta = choices[0].get("delta") or {}
    return _reasoning_text(delta.get(REASONING_KEY))


def _message_reasonings(response: Mapping[str, Any] | openai.BaseModel) -> Iterable[str | None]:
    """Yield ``message.reasoning_content`` per choice of a non-streamed completion."""
    if isinstance(response, Mapping):
        for choice in response.get("choices") or []:
            yield _reasoning_text((choice.get("message") or {}).get(REASONING_KEY))
        return
    for choice in getattr(response, "choices", None) or []:
        message = getattr(choice, "message", None)
        yield _reasoning_text(getattr(message, REASONING_KEY, None))


class MimoeChatOpenAI(ChatOpenAI):
    """``ChatOpenAI`` that keeps the reasoning a 1.0 engine sends in ``reasoning_content``.

    Streaming: each delta's ``reasoning_content`` lands in the chunk's
    ``additional_kwargs["reasoning_content"]``; langchain-core concatenates the key across chunks,
    so the aggregated ``AIMessage`` carries the full text. Non-streaming: the message's
    ``reasoning_content`` lands in the same key. The key is never serialised back to the server
    (``_convert_message_to_dict`` only forwards content, name, tool calls and audio).
    """

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        generation = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if generation is None:
            return None
        reasoning = _delta_reasoning(chunk)
        if reasoning:
            generation.message.additional_kwargs[REASONING_KEY] = reasoning
        return generation

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        """``ChatOpenAI._stream`` that fails loudly when the engine ends the stream early.

        Every complete completion ends with a chunk carrying ``finish_reason``. mimOE 0.6.5
        reports an error in the middle of a stream (the context filled up, the model crashed)
        without the SSE framing the client expects, so the stream just stops and the turn would
        end with a cut-off or empty answer and no explanation.
        """
        finished = False
        for chunk in super()._stream(*args, **kwargs):
            finished = finished or _finished(chunk)
            yield chunk
        if not finished:
            raise _stream_cut()

    async def _astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        """Async twin of :meth:`_stream`."""
        finished = False
        async for chunk in super()._astream(*args, **kwargs):
            finished = finished or _finished(chunk)
            yield chunk
        if not finished:
            raise _stream_cut()

    def _create_chat_result(
        self,
        response: dict | openai.BaseModel,
        generation_info: dict | None = None,
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)
        for generation, reasoning in zip(
            result.generations, _message_reasonings(response), strict=False
        ):
            if reasoning:
                generation.message.additional_kwargs[REASONING_KEY] = reasoning
        return result


def _finished(chunk: ChatGenerationChunk) -> bool:
    info = chunk.generation_info or {}
    return bool(info.get("finish_reason") or chunk.message.response_metadata.get("finish_reason"))


def _stream_cut() -> MimoeError:
    return MimoeError(
        "mimOE stopped in the middle of the answer (the stream ended without a finish reason)",
        hint=HINT_STREAM_CUT,
    )


def make_model(
    settings: Settings,
    pre: Preflight,
    *,
    http_client: httpx.Client | None = None,
    http_async_client: httpx.AsyncClient | None = None,
) -> MimoeChatOpenAI:
    """Build the chat model for the discovered engine and the chosen model.

    Args:
        settings: Effective settings (``api_key`` and ``think`` are used).
        pre: Preflight result (engine base URL, model id, thinking control).
        http_client: Injected sync client (tests pass an ``httpx.MockTransport`` client);
            defaults to ``httpx.Client(trust_env=False)`` so proxies never capture localhost.
        http_async_client: Same for the async path.
    """
    extra_body: dict[str, object] = {
        "max_tokens": MAX_TOKENS_THINK if settings.think else MAX_TOKENS,
    }
    if pre.thinking_control == "native":
        extra_body["enable_thinking"] = settings.think
    return MimoeChatOpenAI(
        model=pre.model.id,
        base_url=pre.engine.base_url,
        api_key=settings.api_key,
        use_responses_api=False,
        temperature=0,
        timeout=REQUEST_TIMEOUT_S,
        max_retries=0,
        stream_usage=True,
        model_kwargs={},
        extra_body=extra_body,
        http_client=http_client or httpx.Client(trust_env=False),
        http_async_client=http_async_client or httpx.AsyncClient(trust_env=False),
    )
