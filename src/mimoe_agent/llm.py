"""``ChatOpenAI`` factory pinned to mimOE's chat-completions endpoint.

Verified against langchain-openai 1.6.6: ``max_tokens`` (and ``model_kwargs["max_tokens"]``) is
rewritten to ``max_completion_tokens``, which mimOE ignores, so the cap travels in ``extra_body``;
``use_responses_api=False`` short-circuits every Responses-API heuristic; ``stream_usage=True`` adds
``stream_options={"include_usage": true}`` to streaming requests only.
"""

from __future__ import annotations

import httpx
from langchain_openai import ChatOpenAI

from mimoe_agent.config import Settings
from mimoe_agent.mimoe import Preflight

MAX_TOKENS = 1024
MAX_TOKENS_THINK = 4096
REQUEST_TIMEOUT_S = 200.0
"""Above mimOE's own 180 s execution timeout, so the engine's error arrives before ours."""


def make_model(
    settings: Settings,
    pre: Preflight,
    *,
    http_client: httpx.Client | None = None,
    http_async_client: httpx.AsyncClient | None = None,
) -> ChatOpenAI:
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
    return ChatOpenAI(
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
