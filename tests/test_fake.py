"""The fake itself: raw HTTP shapes of both generations, and what ``ChatOpenAI`` makes of them."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest
from conftest import NODE_ID, FakeMimoe
from langchain_core.messages import AIMessageChunk
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

BASE = "http://fake/mimik-ai/openai/v1"
HEADERS = {"Authorization": "Bearer 1234"}


@tool
def run_python(code: str) -> str:
    """Run a Python snippet."""
    return "2"


def _merge(chunks: list[AIMessageChunk]) -> AIMessageChunk:
    merged = chunks[0]
    for chunk in chunks[1:]:
        merged = merged + chunk
    return merged


# -- raw HTTP ------------------------------------------------------------------------------------


def test_models_v06_shape(fake_mimoe: FakeMimoe) -> None:
    with fake_mimoe.client() as http:
        data = http.get(f"{BASE}/models").json()  # no key needed on 0.6
    entry = data["data"][0]
    assert data["object"] == "list"
    assert entry["id"] == "qwen3-4b"
    assert entry["info"]["kind"] == "llm"
    assert "supported_parameters" not in entry["info"]
    assert entry["metrics"]["tokens_per_second"] == pytest.approx(35.4)


def test_models_v10_shape_and_auth(fake_mimoe_v10: FakeMimoe) -> None:
    with fake_mimoe_v10.client() as http:
        public = http.get(f"{BASE}/models")  # verified on 1.0.27: no key needed here
        store_denied = http.get("http://fake/mimik-ai/store/v1/models")
        chat_denied = http.post(
            f"{BASE}/chat/completions", json={"model": "qwen3-4b", "messages": []}
        )
        data = http.get(f"{BASE}/models", headers=HEADERS).json()
    assert public.status_code == 200
    assert store_denied.status_code == 403
    assert store_denied.json() == {"error": {"code": 403, "message": "incorrect API key"}}
    assert chat_denied.status_code == 403
    assert chat_denied.json()["error"]["type"] == "permission_error"
    info = data["data"][0]["info"]
    assert info["family"] == "qwen3"
    assert info["supported_parameters"] == ["tools", "tool_choice", "enable_thinking"]
    assert info["reasoning"] == {"supported": True, "default_enabled": True, "can_disable": True}


def test_bad_key_v06_shape(fake_mimoe: FakeMimoe) -> None:
    with fake_mimoe.client() as http:
        resp = http.post(f"{BASE}/chat/completions", json={"model": "qwen3-4b", "messages": []})
    assert resp.status_code == 403
    assert resp.json() == {"error": {"code": 403, "message": "Forbidden"}}


def test_unknown_paths(fake_mimoe: FakeMimoe) -> None:
    with fake_mimoe.client() as http:
        wrong_container = http.get("http://fake/nope/v1/models")
        wrong_route = http.get(f"{BASE}/nope", headers=HEADERS)
    assert wrong_container.status_code == 503 and wrong_container.content == b""
    assert wrong_route.status_code == 404
    assert wrong_route.json() == {"statusCode": 404, "message": "not found"}


def test_store_and_getme(fake_mimoe: FakeMimoe, fake_mimoe_v10: FakeMimoe) -> None:
    with fake_mimoe.client() as http:
        store = http.get("http://fake/mimik-ai/store/v1/models").json()
        me = http.post(
            "http://fake/jsonrpc/v1",
            json={"jsonrpc": "2.0", "id": 1, "method": "getMe", "params": []},
        ).json()
    assert {m["id"] for m in store["data"]} == set(fake_mimoe.registry)
    assert store["data"][0]["readyToUse"] is True and store["data"][0]["totalSize"] > 0
    assert me["result"]["nodeId"] == NODE_ID
    assert me["result"]["version"].startswith("v3.22")
    with fake_mimoe_v10.client() as http:
        me10 = http.post(
            "http://fake/jsonrpc/v1",
            json={"jsonrpc": "2.0", "id": 7, "method": "getMe", "params": []},
            headers=HEADERS,
        ).json()
        store10 = http.get("http://fake/mimik-ai/store/v1/models", headers=HEADERS).json()
    assert me10["id"] == 7 and me10["result"]["version"].startswith("v3.30")
    assert store10["data"][0]["_v"] == 2


def test_load_and_unload_v06(fake_mimoe: FakeMimoe) -> None:
    with fake_mimoe.client() as http:
        resp = http.post(f"{BASE}/models", json={"model": "smollm2-360m"}, headers=HEADERS)
        lines = [line for line in resp.text.splitlines() if line]
        gone = http.delete(f"{BASE}/models", params={"modelId": "smollm2-360m"}, headers=HEADERS)
        again = http.delete(f"{BASE}/models", params={"modelId": "smollm2-360m"}, headers=HEADERS)
        missing = http.post(f"{BASE}/models", json={"model": "nope"}, headers=HEADERS)
    assert resp.headers["content-type"] == "text/event-stream"
    assert lines[0] == 'data: {"progress": "<|loading_model|> 0%<br />"}'
    assert json.loads(lines[-1]) == {
        "id": "smollm2-360m",
        "object": "model",
        "kind": "llm",
        "loaded": True,
    }
    assert gone.json() == {"id": "smollm2-360m", "object": "model", "deleted": True}
    assert again.status_code == 404 and "not found in cache" in again.json()["message"]
    assert missing.status_code == 404 and missing.json()["statusCode"] == 404


def test_v10_has_no_load_or_unload_route(fake_mimoe_v10: FakeMimoe) -> None:
    with fake_mimoe_v10.client() as http:
        load = http.post(f"{BASE}/models", json={"model": "smollm2-360m"}, headers=HEADERS)
        unload = http.delete(f"{BASE}/models", params={"modelId": "qwen3-4b"}, headers=HEADERS)
    assert load.status_code == 404 and load.json()["message"] == "not found"
    assert unload.status_code == 404


def test_v10_store_action_load_and_unload(fake_mimoe_v10: FakeMimoe, fake_mimoe: FakeMimoe) -> None:
    """The verified 1.0.27 lifecycle: PUT {store}/models with action load (SSE) / unload (JSON)."""
    store = "http://fake/mimik-ai/store/v1/models"
    with fake_mimoe_v10.client() as http:
        denied = http.put(store, json={"id": "smollm2-360m", "action": "load"})
        load = http.put(store, json={"id": "smollm2-360m", "action": "load"}, headers=HEADERS)
        lines = [line for line in load.text.splitlines() if line]
        unload = http.put(store, json={"id": "smollm2-360m", "action": "unload"}, headers=HEADERS)
        again = http.put(store, json={"id": "smollm2-360m", "action": "unload"}, headers=HEADERS)
        missing = http.put(store, json={"id": "nope", "action": "load"}, headers=HEADERS)
    assert denied.status_code == 403
    assert load.headers["content-type"] == "text/event-stream"
    assert lines[0] == 'data: {"progress": "<|loading_model|> 0%<br />\\n"}'
    final = {"id": "smollm2-360m", "object": "model", "kind": "llm", "loaded": True}
    assert lines[-2:] == [f"data: {json.dumps(final)}"] * 2  # the real engine sends it twice
    assert unload.json() == {"id": "smollm2-360m", "object": "model", "unloaded": True}
    assert again.status_code == 404 and missing.status_code == 404
    assert "smollm2-360m" not in fake_mimoe_v10.loaded
    with fake_mimoe.client() as http:  # 0.6 store: only action "update" is documented
        old = http.put(store, json={"id": "smollm2-360m", "action": "load"}, headers=HEADERS)
    assert old.status_code == 400 and "action" in old.json()["error"]["message"]
    assert "smollm2-360m" not in fake_mimoe.loaded


def test_chat_non_stream_shapes(fake_mimoe: FakeMimoe, fake_mimoe_v10: FakeMimoe) -> None:
    body = {"model": "qwen3-4b", "messages": [{"role": "user", "content": "hi"}]}
    fake_mimoe.script({"content": "Hello", "reasoning": "why not"}, {"content": "Tool-free"})
    with fake_mimoe.client() as http:
        first = http.post(f"{BASE}/chat/completions", json=body, headers=HEADERS).json()
        second = http.post(f"{BASE}/chat/completions", json=body, headers=HEADERS).json()
    assert first["choices"][0]["message"]["content"] == "<think>\nwhy not\n</think>\n\nHello"
    assert first["choices"][0]["finish_reason"] == "stop"
    assert "reasoning_content" not in first["choices"][0]["message"]
    assert second["choices"][0]["message"]["content"] == "<think>\n\n</think>\n\nTool-free"
    assert (
        first["usage"]["total_tokens"]
        == first["usage"]["prompt_tokens"] + first["usage"]["completion_tokens"]
    )
    fake_mimoe_v10.script({"tool_calls": [{"name": "run_python", "args": {"code": "1"}}]})
    with fake_mimoe_v10.client() as http:
        reply = http.post(f"{BASE}/chat/completions", json=body, headers=HEADERS).json()
    message = reply["choices"][0]["message"]
    assert message["content"] == ""
    assert message["tool_calls"][0]["id"] == "tool_0"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"code": "1"}
    assert reply["choices"][0]["finish_reason"] == "tool_calls"


def test_chat_stream_chunk_order(
    fake_mimoe: FakeMimoe, sse_events: Callable[[bytes], list[dict]]
) -> None:
    fake_mimoe.script(
        {"content": "two words", "tool_calls": [{"name": "run_python", "args": {"code": "x"}}]}
    )
    body = {"model": "qwen3-4b", "messages": [{"role": "user", "content": "hi"}], "stream": True}
    with fake_mimoe.client() as http:
        resp = http.post(f"{BASE}/chat/completions", json=body, headers=HEADERS)
    assert resp.headers["content-type"] == "text/event-stream"
    assert resp.text.endswith("data: [DONE]\n\n")
    events = sse_events(resp.content)
    assert events[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert events[1]["mimoe_status"]["stage"] == "processing_prompt"
    assert events[1]["choices"][0]["delta"] == {}
    contents = [e["choices"][0]["delta"].get("content") for e in events]
    assert contents[2:4] == ["<think>", "\n\n</think>"]
    assert "".join(c for c in contents if c) == "<think>\n\n</think>\n\ntwo words"
    tool_deltas = [
        e["choices"][0]["delta"]["tool_calls"][0]
        for e in events
        if "tool_calls" in e["choices"][0]["delta"]
    ]
    assert tool_deltas[0]["id"] == "tool_0" and tool_deltas[0]["function"]["name"] == "run_python"
    assert all("id" not in d for d in tool_deltas[1:])
    assert "".join(d["function"]["arguments"] for d in tool_deltas) == '{"code": "x"}'
    assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert events[-1]["usage"]["completion_tokens"] > 0
    assert all(e["id"] == events[0]["id"] for e in events)


def test_chat_stream_v10_reasoning_and_auto_load(
    fake_mimoe_v10: FakeMimoe, sse_events: Callable[[bytes], list[dict]]
) -> None:
    fake_mimoe_v10.script({"content": "done", "reasoning": "hmm"})
    body = {
        "model": "smollm2-360m",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    with fake_mimoe_v10.client() as http:
        resp = http.post(f"{BASE}/chat/completions", json=body, headers=HEADERS)
    events = sse_events(resp.content)
    stages = [e["mimoe_status"]["stage"] for e in events if "mimoe_status" in e]
    assert stages == ["loading_model"] * 3 + ["processing_prompt"]
    assert "smollm2-360m" in fake_mimoe_v10.loaded
    deltas = [e["choices"][0]["delta"] for e in events]
    assert any(d.get("reasoning_content") == "hmm" for d in deltas)
    assert "".join(d.get("content", "") for d in deltas) == "done"
    assert not any("<think>" in d.get("content", "") for d in deltas)


def test_chat_stream_v10_tool_call_is_one_delta(
    fake_mimoe_v10: FakeMimoe, sse_events: Callable[[bytes], list[dict]]
) -> None:
    """Verified on 1.0.27: id, name and the full arguments arrive in a single delta."""
    fake_mimoe_v10.script({"tool_calls": [{"name": "run_python", "args": {"code": "print(1)"}}]})
    body = {"model": "qwen3-4b", "messages": [{"role": "user", "content": "hi"}], "stream": True}
    with fake_mimoe_v10.client() as http:
        resp = http.post(f"{BASE}/chat/completions", json=body, headers=HEADERS)
    tool_deltas = [
        e["choices"][0]["delta"]["tool_calls"][0]
        for e in sse_events(resp.content)
        if "tool_calls" in e["choices"][0]["delta"]
    ]
    assert len(tool_deltas) == 1
    assert tool_deltas[0]["id"] == "tool_0"
    assert tool_deltas[0]["function"] == {"name": "run_python", "arguments": '{"code": "print(1)"}'}


def test_unknown_model_and_fail_next(fake_mimoe: FakeMimoe) -> None:
    fake_mimoe.fail_next(500, {"message": "llama_decode() failed", "statusCode": 500})
    with fake_mimoe.client() as http:
        failed = http.post(
            f"{BASE}/chat/completions", json={"model": "qwen3-4b", "messages": []}, headers=HEADERS
        )
        missing = http.post(
            f"{BASE}/chat/completions", json={"model": "nope", "messages": []}, headers=HEADERS
        )
    assert failed.status_code == 500 and "llama_decode" in failed.json()["message"]
    assert missing.status_code == 404 and missing.json()["statusCode"] == 404
    assert fake_mimoe.calls[-1]["model"] == "nope"  # recorded even when it fails


def test_down_raises_connect_error(fake_mimoe: FakeMimoe) -> None:
    fake_mimoe.down = True
    with fake_mimoe.client() as http, pytest.raises(httpx.ConnectError):
        http.get(f"{BASE}/models")


def test_default_reply_calls_ping_when_offered(fake_mimoe: FakeMimoe) -> None:
    ping = {"type": "function", "function": {"name": "ping", "parameters": {"type": "object"}}}
    with fake_mimoe.client() as http:
        with_tool = http.post(
            f"{BASE}/chat/completions",
            json={"model": "qwen3-4b", "messages": [], "tools": [ping]},
            headers=HEADERS,
        ).json()
        without = http.post(
            f"{BASE}/chat/completions", json={"model": "qwen3-4b", "messages": []}, headers=HEADERS
        ).json()
    assert with_tool["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "ping"
    assert without["choices"][0]["message"]["content"].endswith("OK.")
    fake_mimoe.tools_ok = False
    with fake_mimoe.client() as http:
        refused = http.post(
            f"{BASE}/chat/completions",
            json={"model": "qwen3-4b", "messages": [], "tools": [ping]},
            headers=HEADERS,
        ).json()
    assert "tool_calls" not in refused["choices"][0]["message"]


# -- through ChatOpenAI --------------------------------------------------------------------------


def test_chatopenai_invoke(llm_fake: ChatOpenAI, fake_mimoe: FakeMimoe) -> None:
    fake_mimoe.script({"content": "Hello there"})
    message = llm_fake.invoke("hi")
    assert message.content == "<think>\n\n</think>\n\nHello there"
    assert message.response_metadata["finish_reason"] == "stop"
    assert message.usage_metadata is not None and message.usage_metadata["total_tokens"] > 0
    assert message.response_metadata["token_usage"]["token_per_second"] == pytest.approx(35.4)


def test_chatopenai_stream_matches_invoke(llm_fake: ChatOpenAI, fake_mimoe: FakeMimoe) -> None:
    fake_mimoe.script({"content": "Hello there"}, {"content": "Hello there"})
    invoked = llm_fake.invoke("hi")
    streamed = _merge(list(llm_fake.stream("hi")))
    assert streamed.content == invoked.content
    assert streamed.usage_metadata is not None
    assert streamed.usage_metadata["total_tokens"] == invoked.usage_metadata["total_tokens"]  # type: ignore[index]
    assert streamed.response_metadata["finish_reason"] == "stop"


async def test_chatopenai_astream_tool_calls(llm_fake: ChatOpenAI, fake_mimoe: FakeMimoe) -> None:
    fake_mimoe.script({"tool_calls": [{"name": "run_python", "args": {"code": "print(1+1)"}}]})
    chunks = [chunk async for chunk in llm_fake.bind_tools([run_python]).astream("compute 1+1")]
    merged = _merge(chunks)
    assert merged.tool_calls[0]["id"] == "tool_0"
    assert merged.tool_calls[0]["name"] == "run_python"
    assert merged.tool_calls[0]["args"] == {"code": "print(1+1)"}
    assert merged.content == "<think>\n\n</think>"
    assert merged.response_metadata["finish_reason"] == "tool_calls"
    assert fake_mimoe.calls[-1]["tools"][0]["function"]["name"] == "run_python"


def test_chatopenai_v10_reasoning_is_not_in_content(workspace_tmp: object) -> None:
    fake = FakeMimoe("1.0")
    fake.script({"content": "four", "reasoning": "2+2"})
    llm = ChatOpenAI(
        model="qwen3-4b",
        base_url=fake.base_url,
        api_key="1234",
        use_responses_api=False,
        max_retries=0,
        http_client=fake.client(),
    )
    message = llm.invoke("2+2?")
    assert message.content == "four"
    assert "<think>" not in message.content


def test_chatopenai_error_classes(llm_fake: ChatOpenAI, fake_mimoe: FakeMimoe) -> None:
    fake_mimoe.fail_next(404, {"message": "Model 'x' not found in mmodelstore", "statusCode": 404})
    with pytest.raises(Exception) as info:
        llm_fake.invoke("hi")
    assert type(info.value).__name__ == "OpenAIModelNotFoundError"
    assert getattr(info.value, "status_code", None) == 404
    assert info.value.body["statusCode"] == 404  # type: ignore[attr-defined]
