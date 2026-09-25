"""``make_model``: constructor fields and the request body mimOE actually receives."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from conftest import FakeMimoe
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from mimoe_agent.agent import SYSTEM_PROMPT, system_prompt
from mimoe_agent.config import Settings, load_settings
from mimoe_agent.llm import MAX_TOKENS, MAX_TOKENS_THINK, REQUEST_TIMEOUT_S, make_model
from mimoe_agent.mimoe import MimoeClient, Preflight, preflight


def _build(fake: FakeMimoe, settings: Settings) -> tuple[ChatOpenAI, Preflight]:
    client = MimoeClient(settings.base_url, settings.api_key, client=fake.client())
    pre = preflight(settings, client=client)
    llm = make_model(
        settings, pre, http_client=fake.client(), http_async_client=fake.async_client()
    )
    return llm, pre


def test_constructor_fields(llm_fake: ChatOpenAI, settings_tmp: Settings) -> None:
    assert llm_fake.model_name == "qwen3-4b"
    assert llm_fake.openai_api_base == "http://fake/mimik-ai/openai/v1"
    assert llm_fake.openai_api_key is not None
    assert llm_fake.openai_api_key.get_secret_value() == settings_tmp.api_key
    assert llm_fake.use_responses_api is False
    assert llm_fake.temperature == 0
    assert llm_fake.request_timeout == REQUEST_TIMEOUT_S
    assert llm_fake.max_retries == 0
    assert llm_fake.stream_usage is True
    assert llm_fake.model_kwargs == {}
    assert llm_fake.extra_body == {"max_tokens": MAX_TOKENS}


def test_payload_keeps_max_tokens_in_extra_body(llm_fake: ChatOpenAI) -> None:
    payload = llm_fake._get_request_payload([HumanMessage("hi")])
    assert "max_completion_tokens" not in payload
    assert "max_tokens" not in payload  # only inside extra_body, which the SDK merges top-level
    assert payload["extra_body"] == {"max_tokens": MAX_TOKENS}
    assert "reasoning_effort" not in payload
    assert payload["messages"] == [{"role": "user", "content": "hi"}]


def test_request_body_soft_control(llm_fake: ChatOpenAI, fake_mimoe: FakeMimoe) -> None:
    fake_mimoe.calls.clear()
    llm_fake.invoke("hi")
    body = fake_mimoe.calls[-1]
    assert body["model"] == "qwen3-4b"
    assert body["max_tokens"] == MAX_TOKENS
    assert "max_completion_tokens" not in body
    assert "enable_thinking" not in body
    assert "reasoning_effort" not in body
    assert "stream_options" not in body
    assert body["stream"] is False
    assert body["temperature"] == 0


def test_request_body_streaming_adds_stream_options(
    llm_fake: ChatOpenAI, fake_mimoe: FakeMimoe
) -> None:
    fake_mimoe.calls.clear()
    list(llm_fake.stream("hi"))
    body = fake_mimoe.calls[-1]
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["max_tokens"] == MAX_TOKENS


async def test_request_body_astream(llm_fake: ChatOpenAI, fake_mimoe: FakeMimoe) -> None:
    fake_mimoe.calls.clear()
    async for _ in llm_fake.astream("hi"):
        pass
    body = fake_mimoe.calls[-1]
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert "max_completion_tokens" not in body


def test_native_control_sends_enable_thinking(workspace_tmp: Path) -> None:
    fake = FakeMimoe("1.0")
    settings = load_settings(
        {"workspace": workspace_tmp, "base_url": fake.base_url}, cwd=workspace_tmp.parent
    )
    llm, pre = _build(fake, settings)
    assert pre.thinking_control == "native"
    assert llm.extra_body == {"max_tokens": MAX_TOKENS, "enable_thinking": False}
    llm.invoke("hi")
    body = fake.calls[-1]
    assert body["enable_thinking"] is False
    assert body["max_tokens"] == MAX_TOKENS
    assert "max_completion_tokens" not in body


def test_think_native_raises_cap_and_enables_thinking(workspace_tmp: Path) -> None:
    fake = FakeMimoe("1.0")
    settings = load_settings(
        {"workspace": workspace_tmp, "base_url": fake.base_url, "think": True},
        cwd=workspace_tmp.parent,
    )
    llm, _ = _build(fake, settings)
    llm.invoke("hi")
    body = fake.calls[-1]
    assert body["enable_thinking"] is True
    assert body["max_tokens"] == MAX_TOKENS_THINK


def test_think_soft_control_never_sends_enable_thinking(
    fake_mimoe: FakeMimoe, settings_tmp: Settings
) -> None:
    settings = dataclasses.replace(settings_tmp, think=True)
    llm, pre = _build(fake_mimoe, settings)
    assert pre.thinking_control == "soft"
    llm.invoke("hi")
    body = fake_mimoe.calls[-1]
    assert body["max_tokens"] == MAX_TOKENS_THINK
    assert "enable_thinking" not in body


def test_system_prompt_variants(settings_tmp: Settings, preflight_fake: Preflight) -> None:
    soft = system_prompt(settings_tmp, preflight_fake)  # 0.6: soft control, thinking off
    assert soft.startswith("/no_think\n")
    assert str(settings_tmp.workspace) in soft and "{workspace}" not in soft
    assert soft.removeprefix("/no_think\n") == SYSTEM_PROMPT.replace(
        "{workspace}", str(settings_tmp.workspace)
    )
    thinking = system_prompt(dataclasses.replace(settings_tmp, think=True), preflight_fake)
    assert "/no_think" not in thinking
    native = system_prompt(
        settings_tmp, dataclasses.replace(preflight_fake, thinking_control="native")
    )
    assert "/no_think" not in native  # enable_thinking travels in the request body instead
    chat_only = system_prompt(
        settings_tmp, dataclasses.replace(preflight_fake, tools_enabled=False)
    )
    assert chat_only.startswith("/no_think\n") and "Tools are disabled" in chat_only
    assert "run Python" not in chat_only


def test_default_http_clients_ignore_proxy_env(
    settings_tmp: Settings, preflight_fake: Preflight, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:3128")
    llm = make_model(settings_tmp, preflight_fake)
    assert llm.http_client is not None and llm.http_client.trust_env is False
    assert llm.http_async_client is not None and llm.http_async_client.trust_env is False
