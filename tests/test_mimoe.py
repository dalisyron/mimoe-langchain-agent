"""Engine client: discovery, parsing of both generations, load/unload, probe, preflight, errors."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import httpx
import openai
import pytest
from conftest import NODE_ID, FakeMimoe
from langchain_core.exceptions import (
    ModelConnectionError,
    ModelPermissionDeniedError,
    ModelTimeoutError,
)
from langchain_openai import ChatOpenAI

from mimoe_agent.config import ConfigError, Settings, load_settings
from mimoe_agent.llm import make_model
from mimoe_agent.mimoe import (
    CANDIDATE_BASE_URLS,
    EngineGeneration,
    MimoeClient,
    MimoeError,
    Preflight,
    ProbeResult,
    derive_urls,
    detect_generation,
    friendly_error,
    parse_loaded_model,
    parse_version,
    preflight,
    probe_body,
    strip_node_prefix,
)


def _client(fake: FakeMimoe, *, base_url: str | None = None, api_key: str = "1234") -> MimoeClient:
    return MimoeClient(base_url or fake.base_url, api_key, client=fake.client())


def _settings(workspace: Path, **overrides: object) -> Settings:
    return load_settings(
        {"workspace": workspace, "base_url": "http://fake/mimik-ai/openai/v1", **overrides},
        cwd=workspace.parent,
    )


# -- pure helpers --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("base", "store", "rpc"),
    [
        (
            "http://localhost:8083/mimik-ai/openai/v1",
            "http://localhost:8083/mimik-ai/store/v1",
            "http://localhost:8083/jsonrpc/v1",
        ),
        (
            "http://localhost:8083/openai/v1/",
            "http://localhost:8083/store/v1",
            "http://localhost:8083/jsonrpc/v1",
        ),
        (
            "http://127.0.0.1:8093/mimik-airouter/openai/v1",
            "http://127.0.0.1:8093/mimik-ai/store/v1",
            "http://127.0.0.1:8093/jsonrpc/v1",
        ),
        ("http://host:1/weird", "http://host:1/mimik-ai/store/v1", "http://host:1/jsonrpc/v1"),
    ],
)
def test_derive_urls(base: str, store: str, rpc: str) -> None:
    assert derive_urls(base) == (store, rpc)


def test_strip_node_prefix() -> None:
    assert strip_node_prefix(f"{NODE_ID}/qwen3-4b") == "qwen3-4b"
    assert strip_node_prefix("abc/qwen3-4b", node_id="abc") == "qwen3-4b"
    assert strip_node_prefix("qwen3-4b") == "qwen3-4b"
    assert strip_node_prefix("owner/repo") == "owner/repo"  # not a node id, left alone


def test_parse_version_and_generation() -> None:
    assert parse_version("v3.22.8 (developer edition)") == (3, 22)
    assert parse_version("v3.30.26 (developer edition)") == (3, 30)
    assert parse_version(None) is None
    assert parse_version("garbage") is None
    assert detect_generation([], "v3.22.8") is EngineGeneration.V06
    assert detect_generation([], "v3.30.26") is EngineGeneration.V10
    assert detect_generation([], "v4.0.0") is EngineGeneration.V10
    assert detect_generation([], None) is EngineGeneration.V06
    v10_entry = {"id": "x", "info": {"supported_parameters": ["tools"]}}
    assert detect_generation([v10_entry], "v3.22.8") is EngineGeneration.V10
    assert (
        detect_generation([{"id": "x", "supported_parameters": []}], None) is EngineGeneration.V10
    )


def test_parse_loaded_model_v06_shape(fake_mimoe: FakeMimoe) -> None:
    model = parse_loaded_model(fake_mimoe.model_entry("qwen3-4b"))
    assert model.id == "qwen3-4b"
    assert model.kind == "llm"
    assert model.family is None
    assert model.max_context == 12000
    assert model.n_params == 4022468096
    assert model.tokens_per_second == pytest.approx(35.4)
    assert model.avg_tokens_per_second == pytest.approx(32.3)
    assert model.supports_tools is None
    assert model.thinking_supported is None
    assert model.thinking_can_disable is None
    assert model.raw["object"] == "model"


def test_parse_loaded_model_v10_shape(fake_mimoe_v10: FakeMimoe) -> None:
    model = parse_loaded_model(fake_mimoe_v10.model_entry("qwen3-4b"))
    assert model.family == "qwen3"
    assert model.supports_tools is True
    assert model.thinking_supported is True
    assert model.thinking_can_disable is True
    fake_mimoe_v10.tools_ok = False
    fake_mimoe_v10.thinking_supported = False
    model = parse_loaded_model(fake_mimoe_v10.model_entry("qwen3-4b-instruct-2507"))
    assert model.supports_tools is False
    assert model.thinking_supported is False
    assert model.thinking_can_disable is None


def test_parse_loaded_model_tolerates_junk() -> None:
    model = parse_loaded_model({"id": f"{NODE_ID}/m", "info": "nope", "metrics": None})
    assert model.id == "m"
    assert model.kind == "llm"
    assert model.max_context is None
    assert model.tokens_per_second is None
    empty = parse_loaded_model({})
    assert empty.id == ""


# -- discovery -----------------------------------------------------------------------------------


def test_discover_v06(fake_mimoe: FakeMimoe) -> None:
    engine = _client(fake_mimoe).discover()
    assert engine.base_url == "http://fake/mimik-ai/openai/v1"
    assert engine.store_url == "http://fake/mimik-ai/store/v1"
    assert engine.rpc_url == "http://fake/jsonrpc/v1"
    assert engine.node_name == "fake-node"
    assert engine.version == "v3.22.8 (developer edition)"
    assert engine.generation is EngineGeneration.V06


def test_discover_v10_by_version_with_nothing_loaded(fake_mimoe_v10: FakeMimoe) -> None:
    fake_mimoe_v10.loaded.clear()
    engine = _client(fake_mimoe_v10).discover()
    assert engine.generation is EngineGeneration.V10
    assert engine.version == "v3.30.26 (developer edition)"


def test_discover_v10_by_capabilities_when_getme_fails(fake_mimoe_v10: FakeMimoe) -> None:
    fake_mimoe_v10.rpc_path = "/elsewhere"  # getMe now hits an unknown path -> 503
    engine = _client(fake_mimoe_v10).discover()
    assert engine.version is None
    assert engine.node_name is None
    assert engine.generation is EngineGeneration.V10


def test_discover_tries_explicit_url_first(fake_mimoe: FakeMimoe) -> None:
    client = _client(fake_mimoe, base_url="http://fake/mimik-ai/openai/v1/")
    engine = client.discover()
    assert engine.base_url == "http://fake/mimik-ai/openai/v1"
    assert fake_mimoe.requests[0].url.path == "/mimik-ai/openai/v1/models"


def test_discover_falls_back_to_candidates(fake_mimoe: FakeMimoe) -> None:
    fake_mimoe.openai_path = "/openai/v1"  # an older build: the mimik-ai path answers 503
    client = MimoeClient(None, "1234", client=fake_mimoe.client())
    engine = client.discover()
    assert engine.base_url == CANDIDATE_BASE_URLS[1]
    assert engine.store_url == "http://127.0.0.1:8083/store/v1"
    tried = [str(r.url) for r in fake_mimoe.requests if r.url.path.endswith("/models")]
    assert tried == [c + "/models" for c in CANDIDATE_BASE_URLS[:2]]  # found at the second


def test_discover_unreachable_lists_urls(fake_mimoe: FakeMimoe) -> None:
    fake_mimoe.down = True
    with pytest.raises(MimoeError) as info:
        MimoeClient("http://fake/mimik-ai/openai/v1", "1234", client=fake_mimoe.client()).discover()
    assert info.value.message == "mimOE Studio is not reachable"
    assert "Open mimOE Studio" in info.value.hint
    assert "http://fake/mimik-ai/openai/v1: ConnectError" in info.value.hint
    assert all(url not in info.value.hint for url in CANDIDATE_BASE_URLS)  # explicit URL only
    with pytest.raises(MimoeError) as info:
        MimoeClient(None, "1234", client=fake_mimoe.client()).discover()
    for url in CANDIDATE_BASE_URLS:
        assert url in info.value.hint


def test_discover_explicit_url_never_falls_back(fake_mimoe: FakeMimoe) -> None:
    """An explicit --base-url that is down must not silently connect to another engine."""

    def only_port_8083(request: httpx.Request) -> httpx.Response:
        if request.url.port != 8083:
            raise httpx.ConnectError("Connection refused", request=request)
        return fake_mimoe.handler(request)

    http = httpx.Client(transport=httpx.MockTransport(only_port_8083))
    with pytest.raises(MimoeError, match="not reachable"):
        MimoeClient("http://127.0.0.1:8093/mimik-ai/openai/v1", "1234", client=http).discover()
    assert fake_mimoe.requests == []  # the default candidates on port 8083 were never tried
    assert MimoeClient(None, "1234", client=http).discover().base_url == CANDIDATE_BASE_URLS[0]


def test_discover_prefers_the_ipv4_loopback() -> None:
    """localhost resolves to ::1 first; Windows spends ~2 s per refused connect there."""
    assert CANDIDATE_BASE_URLS[0].startswith("http://127.0.0.1:8083/")
    assert CANDIDATE_BASE_URLS[-1].startswith("http://localhost:8083/")  # IPv6-only engines


def test_discover_rejects_non_model_list(fake_mimoe: FakeMimoe) -> None:
    def html(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>Studio</html>")

    http = httpx.Client(transport=httpx.MockTransport(html))
    with pytest.raises(MimoeError) as info:
        MimoeClient("http://fake/mimik-ai/openai/v1", "1234", client=http).discover()
    assert "not a model list" in info.value.hint


def test_discover_ignores_key_but_registry_validates_it(fake_mimoe_v10: FakeMimoe) -> None:
    client = _client(fake_mimoe_v10, api_key="wrong")
    assert client.discover().generation is EngineGeneration.V10  # GET /models is public
    with pytest.raises(MimoeError) as info:
        client.registry_models()
    assert info.value.message == "mimOE rejected the API key"
    assert "API button" in info.value.hint


def test_engine_property_discovers_lazily(fake_mimoe: FakeMimoe) -> None:
    client = _client(fake_mimoe)
    assert client.loaded_models()[0].id == "qwen3-4b"
    assert client.engine.generation is EngineGeneration.V06
    assert sum(r.url.path.endswith("/jsonrpc/v1") for r in fake_mimoe.requests) == 1


# -- listing -------------------------------------------------------------------------------------


def test_loaded_models_strips_airouter_prefix(fake_mimoe: FakeMimoe) -> None:
    fake_mimoe.id_prefix = f"{NODE_ID}/"
    models = _client(fake_mimoe).loaded_models()
    assert [m.id for m in models] == ["qwen3-4b"]
    assert models[0].raw["id"].startswith(NODE_ID)


def test_registry_models(fake_mimoe_v10: FakeMimoe) -> None:
    fake_mimoe_v10.registry_sizes["qwen3-4b"] = None  # registered by local path: no totalSize
    models = {m.id: m for m in _client(fake_mimoe_v10).registry_models()}
    assert set(models) == set(fake_mimoe_v10.registry)
    assert models["qwen3-4b"].size_bytes is None
    assert models["smollm2-360m"].size_bytes == 2497281120
    assert all(m.ready and m.kind == "llm" for m in models.values())


def test_registry_models_v06(fake_mimoe: FakeMimoe) -> None:
    models = _client(fake_mimoe).registry_models()
    assert {m.id for m in models} == set(fake_mimoe.registry)
    assert models[0].raw["gguf"]["initContextSize"] == 12000


def test_node_info(fake_mimoe: FakeMimoe) -> None:
    info = _client(fake_mimoe).node_info()
    assert info["nodeId"] == NODE_ID
    assert info["version"].startswith("v3.22")
    fake_mimoe.rpc_path = "/elsewhere"
    assert _client(fake_mimoe).node_info() == {}


def test_node_info_when_unreachable(fake_mimoe: FakeMimoe) -> None:
    fake_mimoe.down = True
    assert _client(fake_mimoe).node_info() == {}


# -- load / unload -------------------------------------------------------------------------------


def test_load_model_v06_streams_progress_then_bare_tail(fake_mimoe: FakeMimoe) -> None:
    seen: list[str] = []
    model = _client(fake_mimoe).load_model("smollm2-360m", on_progress=seen.append)
    assert seen == ["loading model 0%", "loading model 50%", "loading model 100%"]
    assert model.id == "smollm2-360m"
    assert "smollm2-360m" in fake_mimoe.loaded
    load_request = next(
        r for r in fake_mimoe.requests if r.method == "POST" and r.url.path.endswith("/models")
    )
    assert load_request.url.path == "/mimik-ai/openai/v1/models"
    assert load_request.headers["accept"] == "text/event-stream"


def test_load_model_accepts_full_object_final_event(fake_mimoe: FakeMimoe) -> None:
    fake_mimoe.load_final_style = "doc"
    model = _client(fake_mimoe).load_model("smollm2-360m")
    assert model.id == "smollm2-360m"
    assert model.max_context == 12000
    # no extra GET /models was needed after the stream
    assert [r.method for r in fake_mimoe.requests].count("GET") == 1


def test_load_model_unknown_id(fake_mimoe: FakeMimoe) -> None:
    with pytest.raises(MimoeError) as info:
        _client(fake_mimoe).load_model("nope")
    assert "could not find" in info.value.message
    assert "not found in store" in info.value.message


def test_load_model_v10_uses_store_action(fake_mimoe_v10: FakeMimoe) -> None:
    """1.0 loads through PUT {store}/models {id, action: load} (verified live), not a completion."""
    seen: list[str] = []
    model = _client(fake_mimoe_v10).load_model(f"{NODE_ID}/smollm2-360m", on_progress=seen.append)
    assert model.id == "smollm2-360m"
    assert "smollm2-360m" in fake_mimoe_v10.loaded
    assert seen == ["loading model 0%", "loading model 50%", "loading model 100%"]
    put = next(r for r in fake_mimoe_v10.requests if r.method == "PUT")
    assert put.url.path == "/mimik-ai/store/v1/models"
    assert json.loads(put.content) == {"id": "smollm2-360m", "action": "load"}
    assert all(r.method != "POST" or "chat" not in r.url.path for r in fake_mimoe_v10.requests)
    assert fake_mimoe_v10.calls == []  # no completion was needed


def test_load_model_v10_auto_load_when_store_route_missing(fake_mimoe_v10: FakeMimoe) -> None:
    fake_mimoe_v10.store_path = "/elsewhere/store/v1"  # store PUT -> 503 empty (route missing)
    seen: list[str] = []
    model = _client(fake_mimoe_v10).load_model("smollm2-360m", on_progress=seen.append)
    assert model.id == "smollm2-360m"
    assert seen[:3] == ["Loading model (0%)", "Loading model (50%)", "Loading model (100%)"]
    posts = [r for r in fake_mimoe_v10.requests if r.method == "POST"]
    assert [r.url.path for r in posts][-1] == "/mimik-ai/openai/v1/chat/completions"
    body = fake_mimoe_v10.calls[-1]
    assert body["model"] == "smollm2-360m"
    assert body["stream"] is True
    assert body["max_tokens"] == 1


def test_load_model_v10_unknown_id(fake_mimoe_v10: FakeMimoe) -> None:
    with pytest.raises(MimoeError) as info:
        _client(fake_mimoe_v10).load_model("nope")
    assert "could not find" in info.value.message
    assert fake_mimoe_v10.calls == []  # a real 404 is not mistaken for a missing route


def test_unload_model_v06(fake_mimoe: FakeMimoe) -> None:
    client = _client(fake_mimoe)
    client.unload_model(f"{NODE_ID}/qwen3-4b")
    assert fake_mimoe.loaded == []
    request = next(r for r in fake_mimoe.requests if r.method == "DELETE")
    assert request.url.params["modelId"] == "qwen3-4b"
    with pytest.raises(MimoeError) as info:
        client.unload_model("qwen3-4b")
    assert "is not loaded" in info.value.message


def test_unload_model_v10_uses_store_action(fake_mimoe_v10: FakeMimoe) -> None:
    """1.0 unloads through PUT {store}/models {id, action: unload} (verified live)."""
    client = _client(fake_mimoe_v10)
    client.unload_model(f"{NODE_ID}/qwen3-4b")
    assert fake_mimoe_v10.loaded == []
    put = next(r for r in fake_mimoe_v10.requests if r.method == "PUT")
    assert put.url.path == "/mimik-ai/store/v1/models"
    assert json.loads(put.content) == {"id": "qwen3-4b", "action": "unload"}
    assert all(r.method != "DELETE" for r in fake_mimoe_v10.requests)
    with pytest.raises(MimoeError, match="is not loaded"):
        client.unload_model("qwen3-4b")


def test_unload_model_without_any_route(fake_mimoe_v10: FakeMimoe) -> None:
    fake_mimoe_v10.store_path = "/elsewhere/store/v1"  # store gone, DELETE answers 404 not found
    with pytest.raises(MimoeError) as info:
        _client(fake_mimoe_v10).unload_model("qwen3-4b")
    assert "no unload endpoint" in info.value.message
    assert "Studio > Models" in info.value.hint
    assert {r.method for r in fake_mimoe_v10.requests} >= {"PUT", "DELETE"}  # both were tried
    assert fake_mimoe_v10.loaded == ["qwen3-4b"]


# -- probe and chat ------------------------------------------------------------------------------


def test_probe_body_variants() -> None:
    soft = probe_body("qwen3-4b", thinking_control="soft")
    assert soft["max_tokens"] == 48
    assert soft["tools"][0]["function"]["name"] == "ping"
    assert soft["messages"][0]["content"].startswith("/no_think\n")
    assert soft["messages"][1]["content"].endswith("/no_think")
    assert "enable_thinking" not in soft
    native = probe_body("qwen3-4b", thinking_control="native")
    assert native["enable_thinking"] is False
    assert "/no_think" not in native["messages"][0]["content"]
    none = probe_body("qwen3-4b", thinking_control="none")
    assert "enable_thinking" not in none
    assert "/no_think" not in none["messages"][1]["content"]


def test_probe_tools_ok(fake_mimoe: FakeMimoe) -> None:
    result = _client(fake_mimoe).probe_tools("qwen3-4b", thinking_control="soft")
    assert result.tools_ok is True
    assert result.latency_s >= 0
    assert "ping" in result.detail
    body = fake_mimoe.calls[-1]
    assert body["stream"] is False
    assert body["max_tokens"] == 48
    assert body["messages"][0]["content"].startswith("/no_think")


def test_probe_tools_not_ok(fake_mimoe: FakeMimoe) -> None:
    fake_mimoe.tools_ok = False
    result = _client(fake_mimoe).probe_tools("qwen3-4b", thinking_control="soft")
    assert result.tools_ok is False
    assert "finish_reason=stop" in result.detail


def test_probe_tools_ignores_ping_written_as_text(fake_mimoe: FakeMimoe) -> None:
    """smollm-style replies put the call in content; only structured tool_calls count."""
    fake_mimoe.script(
        {"content": '<tool_call>{"name": "ping", "arguments": {}}</tool_call> ping pong'},
        {"tool_calls": [{"name": "pong", "args": {}}]},
    )
    client = _client(fake_mimoe)
    text_only = client.probe_tools("qwen3-4b", thinking_control="soft")
    assert text_only.tools_ok is False
    assert "content=" in text_only.detail and "tool names=none" in text_only.detail
    wrong_tool = client.probe_tools("qwen3-4b", thinking_control="soft")
    assert wrong_tool.tools_ok is False
    assert "pong" in wrong_tool.detail


def test_probe_tools_validates_api_key(fake_mimoe: FakeMimoe) -> None:
    # 0.6: GET /models is public, so the wrong key only surfaces at the completion
    client = _client(fake_mimoe, api_key="wrong")
    assert client.discover().generation is EngineGeneration.V06
    with pytest.raises(MimoeError) as info:
        client.probe_tools("qwen3-4b", thinking_control="soft")
    assert info.value.message == "mimOE rejected the API key"
    assert "API button" in info.value.hint


def test_probe_tools_timeout_hint(fake_mimoe: FakeMimoe) -> None:
    def slow(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat/completions"):
            raise httpx.ReadTimeout("slow", request=request)
        return fake_mimoe.handler(request)

    client = MimoeClient(
        fake_mimoe.base_url, "1234", client=httpx.Client(transport=httpx.MockTransport(slow))
    )
    with pytest.raises(MimoeError) as info:
        client.probe_tools("qwen3-4b", thinking_control="soft")
    assert "timed out" in info.value.message
    assert "--force-tools" in info.value.hint


def test_chat_forces_non_streaming(fake_mimoe: FakeMimoe) -> None:
    fake_mimoe.script({"content": "hello", "reasoning": "thinking"})
    reply = _client(fake_mimoe).chat(
        {"model": "qwen3-4b", "messages": [{"role": "user", "content": "hi"}], "stream": True}
    )
    assert fake_mimoe.calls[-1]["stream"] is False
    assert reply["choices"][0]["message"]["content"] == "<think>\nthinking\n</think>\n\nhello"
    assert reply["usage"]["token_per_second"] == pytest.approx(35.4)


def test_chat_error_shapes(fake_mimoe: FakeMimoe) -> None:
    client = _client(fake_mimoe)
    body = {"model": "qwen3-4b", "messages": []}
    fake_mimoe.fail_next(404, {"message": "Model 'x' not found in mmodelstore", "statusCode": 404})
    with pytest.raises(MimoeError, match="not found in mmodelstore"):
        client.chat(body)
    fake_mimoe.fail_next(403, {"error": {"code": 403, "message": "Forbidden"}})
    with pytest.raises(MimoeError, match="rejected the API key"):
        client.chat(body)
    fake_mimoe.fail_next(500, {"message": "llama_decode() failed", "statusCode": 500})
    with pytest.raises(MimoeError, match="context window") as info:
        client.chat(body)
    assert "/new" in info.value.hint
    fake_mimoe.fail_next(503, "")
    with pytest.raises(MimoeError, match="base URL path") as info:
        client.chat(body)
    assert "--base-url" in info.value.hint
    fake_mimoe.fail_next(500, {"error": {"message": "boom", "type": "server_error"}})
    with pytest.raises(MimoeError, match="boom") as info:
        client.chat(body)
    assert "Studio > Models" in info.value.hint


# -- preflight -----------------------------------------------------------------------------------


def test_preflight_v06(fake_mimoe: FakeMimoe, settings_tmp: Settings) -> None:
    statuses: list[str] = []
    pre = preflight(settings_tmp, client=_client(fake_mimoe), on_status=statuses.append)
    assert isinstance(pre, Preflight)
    assert pre.engine.generation is EngineGeneration.V06
    assert pre.model.id == "qwen3-4b"
    assert pre.thinking_control == "soft"
    assert pre.probe is not None and pre.probe.tools_ok
    assert pre.tools_enabled is True
    assert pre.warnings == ()
    assert statuses == ["connecting to mimOE Studio", "warming up qwen3-4b (tool probe)"]
    paths = [r.url.path for r in fake_mimoe.requests]
    assert paths.index("/jsonrpc/v1") < paths.index("/mimik-ai/openai/v1/chat/completions")


def test_preflight_builds_its_own_client_when_none_given(
    settings_tmp: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = dataclasses.replace(settings_tmp, base_url="http://127.0.0.1:9/mimik-ai/openai/v1")
    monkeypatch.setattr("mimoe_agent.mimoe.CANDIDATE_BASE_URLS", ())
    with pytest.raises(MimoeError, match="not reachable"):
        preflight(settings)


def test_preflight_v10_native_skips_probe(fake_mimoe_v10: FakeMimoe, workspace_tmp: Path) -> None:
    pre = preflight(_settings(workspace_tmp), client=_client(fake_mimoe_v10))
    assert pre.thinking_control == "native"
    assert pre.probe is None
    assert pre.tools_enabled is True
    assert pre.warnings == ()
    assert fake_mimoe_v10.calls == []


def test_preflight_v10_advertised_no_tools(fake_mimoe_v10: FakeMimoe, workspace_tmp: Path) -> None:
    fake_mimoe_v10.tools_ok = False
    pre = preflight(_settings(workspace_tmp), client=_client(fake_mimoe_v10))
    assert pre.tools_enabled is False
    assert pre.probe is None
    assert any("chat-only" in w for w in pre.warnings)


def test_preflight_v10_thinking_variants(fake_mimoe_v10: FakeMimoe, workspace_tmp: Path) -> None:
    # A 1.0 engine accepts enable_thinking for every model (docs/compatibility.md), so the
    # control is "native" whatever the (unreliable) reasoning block claims.
    fake_mimoe_v10.thinking_supported = False
    pre = preflight(_settings(workspace_tmp, think=True), client=_client(fake_mimoe_v10))
    assert pre.thinking_control == "native"
    assert not any("--think has no effect" in w for w in pre.warnings)
    fake_mimoe_v10.thinking_supported = True
    fake_mimoe_v10.thinking_can_disable = False
    pre = preflight(_settings(workspace_tmp), client=_client(fake_mimoe_v10))
    assert pre.thinking_control == "native"


def test_preflight_v10_without_capabilities_runs_probe(
    fake_mimoe_v10: FakeMimoe, workspace_tmp: Path
) -> None:
    fake_mimoe_v10.advertise_capabilities = False
    pre = preflight(_settings(workspace_tmp), client=_client(fake_mimoe_v10))
    assert pre.engine.generation is EngineGeneration.V10  # from the version
    assert pre.thinking_control == "native"
    assert pre.probe is not None and pre.probe.tools_ok
    assert pre.tools_enabled is True


def test_preflight_no_model_loaded(fake_mimoe: FakeMimoe, settings_tmp: Settings) -> None:
    fake_mimoe.loaded.clear()
    with pytest.raises(MimoeError) as info:
        preflight(settings_tmp, client=_client(fake_mimoe))
    assert info.value.message == "no model is loaded"
    assert "Studio > Models > Load" in info.value.hint


def test_preflight_requested_model_not_loaded(fake_mimoe: FakeMimoe, workspace_tmp: Path) -> None:
    fake_mimoe.loaded.append("smollm2-360m")
    with pytest.raises(MimoeError) as info:
        preflight(_settings(workspace_tmp, model="qwen3-8b"), client=_client(fake_mimoe))
    assert "qwen3-8b" in info.value.message
    assert "qwen3-4b, smollm2-360m" in info.value.message
    assert "Studio > Models > Load qwen3-8b" in info.value.hint


def test_preflight_requested_model_with_node_prefix(
    fake_mimoe: FakeMimoe, workspace_tmp: Path
) -> None:
    fake_mimoe.loaded[:] = ["smollm2-360m", "qwen3-4b"]
    pre = preflight(
        _settings(workspace_tmp, model=f"{NODE_ID}/qwen3-4b"), client=_client(fake_mimoe)
    )
    assert pre.model.id == "qwen3-4b"


def test_preflight_picks_first_llm(fake_mimoe: FakeMimoe, settings_tmp: Settings) -> None:
    fake_mimoe.loaded[:] = ["nomic-embed", "qwen3-4b"]
    fake_mimoe.kinds["nomic-embed"] = "embed"
    pre = preflight(settings_tmp, client=_client(fake_mimoe))
    assert pre.model.id == "qwen3-4b"
    fake_mimoe.loaded[:] = ["nomic-embed"]
    with pytest.raises(MimoeError, match="no chat model"):
        preflight(settings_tmp, client=_client(fake_mimoe))


def test_preflight_force_tools_skips_probe(fake_mimoe: FakeMimoe, workspace_tmp: Path) -> None:
    fake_mimoe.tools_ok = False
    pre = preflight(_settings(workspace_tmp, force_tools=True), client=_client(fake_mimoe))
    assert pre.probe is None
    assert pre.tools_enabled is True
    assert any("--force-tools" in w for w in pre.warnings)
    assert fake_mimoe.calls == []


def test_preflight_probe_failure_means_chat_only(
    fake_mimoe: FakeMimoe, settings_tmp: Settings
) -> None:
    fake_mimoe.tools_ok = False
    pre = preflight(settings_tmp, client=_client(fake_mimoe))
    assert pre.probe is not None and not pre.probe.tools_ok
    assert pre.tools_enabled is False
    assert any("chat-only" in w and "--force-tools" in w for w in pre.warnings)


def test_preflight_warns_when_getme_fails(fake_mimoe: FakeMimoe, settings_tmp: Settings) -> None:
    fake_mimoe.rpc_path = "/elsewhere"
    pre = preflight(settings_tmp, client=_client(fake_mimoe))
    assert pre.engine.version is None
    assert any("getMe" in w for w in pre.warnings)


def test_preflight_v10_validates_key_via_registry(
    fake_mimoe_v10: FakeMimoe, workspace_tmp: Path
) -> None:
    with pytest.raises(MimoeError) as info:
        preflight(
            _settings(workspace_tmp, api_key="wrong"),
            client=_client(fake_mimoe_v10, api_key="wrong"),
        )
    assert info.value.message == "mimOE rejected the API key"
    assert fake_mimoe_v10.calls == []  # no completion was needed


def test_preflight_v10_registry_unavailable_only_warns(
    fake_mimoe_v10: FakeMimoe, workspace_tmp: Path
) -> None:
    fake_mimoe_v10.store_path = "/elsewhere/store/v1"
    pre = preflight(_settings(workspace_tmp), client=_client(fake_mimoe_v10))
    assert pre.tools_enabled is True and pre.probe is None
    assert any("could not verify the API key" in w for w in pre.warnings)


def test_preflight_bad_key(fake_mimoe: FakeMimoe, workspace_tmp: Path) -> None:
    with pytest.raises(MimoeError) as info:
        preflight(
            _settings(workspace_tmp, api_key="wrong"), client=_client(fake_mimoe, api_key="wrong")
        )
    assert info.value.message == "mimOE rejected the API key"


# -- friendly_error ------------------------------------------------------------------------------


def _llm(fake: FakeMimoe, settings: Settings) -> ChatOpenAI:
    pre = preflight(settings, client=_client(fake))
    return make_model(
        settings, pre, http_client=fake.client(), http_async_client=fake.async_client()
    )


def test_friendly_error_passthrough() -> None:
    assert friendly_error(MimoeError("m", hint="h")) == ("m", "h")
    assert friendly_error(ConfigError("c", hint="d")) == ("c", "d")


def test_friendly_error_httpx() -> None:
    request = httpx.Request("GET", "http://x")
    msg, hint = friendly_error(httpx.ConnectError("refused", request=request))
    assert msg == "mimOE Studio is not reachable" and "Open mimOE Studio" in hint
    msg, hint = friendly_error(httpx.ReadTimeout("slow", request=request))
    assert msg == "mimOE did not answer in time"
    response = httpx.Response(
        403, json={"error": {"code": 403, "message": "Forbidden"}}, request=request
    )
    msg, _ = friendly_error(httpx.HTTPStatusError("x", request=request, response=response))
    assert msg == "mimOE rejected the API key"


def test_friendly_error_openai_sdk_classes() -> None:
    request = httpx.Request("POST", "http://x/chat/completions")
    msg, _ = friendly_error(openai.APITimeoutError(request))
    assert msg == "mimOE did not answer in time"
    msg, _ = friendly_error(openai.APIConnectionError(request=request))
    assert msg == "mimOE Studio is not reachable"
    resp = httpx.Response(401, json={"error": {"message": "Unauthorized"}}, request=request)
    msg, _ = friendly_error(openai.AuthenticationError("x", response=resp, body=resp.json()))
    assert msg == "mimOE rejected the API key"
    resp = httpx.Response(500, json={"message": "boom", "statusCode": 500}, request=request)
    msg, hint = friendly_error(openai.InternalServerError("x", response=resp, body=resp.json()))
    assert msg == "mimOE returned a server error (boom)" and "Studio > Models" in hint
    resp = httpx.Response(429, content=b"", request=request)
    msg, _ = friendly_error(openai.RateLimitError("x", response=resp, body=None))
    assert msg == "mimOE returned HTTP 429"


@pytest.mark.parametrize(
    ("status", "body", "expected", "hint_part"),
    [
        (
            403,
            {"error": {"code": 403, "message": "Forbidden"}},
            "rejected the API key",
            "API button",
        ),
        (401, {"error": {"message": "Unauthorized"}}, "rejected the API key", "API button"),
        (
            403,  # the 1.0 chat shape
            {"error": {"message": "Forbidden", "type": "permission_error", "code": None}},
            "rejected the API key",
            "API button",
        ),
        (
            404,
            {"message": "Model 'x' not found in mmodelstore", "statusCode": 404},
            "could not find the requested model (Model 'x' not found in mmodelstore)",
            "Studio > Models > Load",
        ),
        (500, {"message": "llama_decode() failed", "statusCode": 500}, "context window", "/new"),
        (503, "", "base URL path is wrong", "--base-url"),
        (
            400,
            {"message": "Model 'x' is not ready (readyToUse: false)", "statusCode": 400},
            "the model is not ready",
            "Download",
        ),
        (500, {"message": "boom", "statusCode": 500}, "server error (boom)", "Studio > Models"),
        (400, {"message": "body.max_tokens must be integer", "statusCode": 400}, "HTTP 400", ""),
    ],
)
def test_friendly_error_through_chatopenai(
    fake_mimoe: FakeMimoe,
    settings_tmp: Settings,
    status: int,
    body: object,
    expected: str,
    hint_part: str,
) -> None:
    llm = _llm(fake_mimoe, settings_tmp)
    fake_mimoe.fail_next(status, body)
    with pytest.raises(openai.APIStatusError) as info:
        llm.invoke("hi")
    msg, hint = friendly_error(info.value)
    assert expected in msg
    assert hint_part in hint


def test_friendly_error_langchain_wrappers(fake_mimoe: FakeMimoe, settings_tmp: Settings) -> None:
    llm = _llm(fake_mimoe, settings_tmp)
    fake_mimoe.fail_next(403, {"error": {"code": 403, "message": "Forbidden"}})
    with pytest.raises(ModelPermissionDeniedError) as denied:
        llm.invoke("hi")
    assert friendly_error(denied.value)[0] == "mimOE rejected the API key"
    fake_mimoe.down = True
    with pytest.raises(ModelConnectionError) as down:
        llm.invoke("hi")
    assert friendly_error(down.value)[0] == "mimOE Studio is not reachable"
    # a langchain-core error that is not an openai subclass is recognised by name
    assert friendly_error(ModelTimeoutError("slow"))[0] == "mimOE did not answer in time"


def test_friendly_error_generic() -> None:
    msg, hint = friendly_error(ValueError("odd"))
    assert msg == "ValueError: odd"
    assert hint
    msg, _ = friendly_error(openai.OpenAIError("missing credentials"))
    assert msg.startswith("model error:")


# -- remote (cloud / provider) entries -----------------------------------------------------------

_CLOUD_06 = {"id": "gpt-4o-mini", "object": "model", "owned_by": "cloud", "info": {"kind": "llm"}}
_PROVIDER_10 = {
    "id": "ollama/llama3.2:3b",
    "object": "model",
    "owned_by": "provider:ollama",
    "attached": True,
    "info": {"kind": "llm", "cloud": False},
}


def test_is_remote_recognises_cloud_and_provider_entries(fake_mimoe: FakeMimoe) -> None:
    from mimoe_agent.mimoe import is_remote, parse_loaded_model

    assert is_remote(parse_loaded_model(_CLOUD_06))
    assert is_remote(parse_loaded_model(_PROVIDER_10))
    assert not any(is_remote(m) for m in _client(fake_mimoe).loaded_models())


def test_preflight_never_auto_picks_a_remote_model(
    fake_mimoe: FakeMimoe, workspace_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Studio lists cloud/provider models next to local ones; "nothing leaves your machine"
    only holds if the agent never picks one on its own."""
    from mimoe_agent.mimoe import parse_loaded_model

    client = _client(fake_mimoe)
    local = client.loaded_models()
    remote = [parse_loaded_model(_CLOUD_06), parse_loaded_model(_PROVIDER_10)]
    monkeypatch.setattr(client, "loaded_models", lambda: [*remote, *local])
    pre = preflight(_settings(workspace_tmp), client=client)
    assert pre.model.id == "qwen3-4b"

    monkeypatch.setattr(client, "loaded_models", lambda: remote)
    with pytest.raises(MimoeError, match="only remote models are listed: gpt-4o-mini"):
        preflight(_settings(workspace_tmp), client=client)


def test_an_explicitly_chosen_remote_model_comes_with_a_warning(
    fake_mimoe: FakeMimoe, workspace_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mimoe_agent.mimoe import parse_loaded_model

    client = _client(fake_mimoe)
    local = client.loaded_models()
    monkeypatch.setattr(client, "loaded_models", lambda: [parse_loaded_model(_CLOUD_06), *local])
    monkeypatch.setattr(
        client,
        "probe_tools",
        lambda *a, **k: ProbeResult(tools_ok=True, latency_s=0.1, detail="ok"),
    )
    pre = preflight(_settings(workspace_tmp, model="gpt-4o-mini"), client=client)
    assert pre.model.id == "gpt-4o-mini"
    assert any("leave this computer" in w for w in pre.warnings)
