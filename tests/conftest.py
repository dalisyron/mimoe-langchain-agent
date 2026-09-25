"""Shared fixtures: :class:`FakeMimoe`, an in-process double of both mimOE generations.

The fake is an ``httpx.MockTransport`` handler, so tests run without sockets on every CI runner.
It reproduces the quirks verified against the real engines (Studio 0.6.5 / milm 1.14.7 and the
1.0.27 runtime / milm 1.17.35):

* 0.6 replies start with an inline ``<think>\\n\\n</think>`` block; 1.0 replies carry reasoning in
  ``reasoning_content`` and ``content`` is a plain string (``""`` on tool-call turns).
* streaming follows the real chunk order: a role chunk, ``mimoe_status`` progress chunks with an
  empty delta, content / ``reasoning_content`` deltas, tool-call deltas whose id (``tool_0``, ...)
  is only on the first fragment, a final chunk with ``finish_reason`` and ``usage``, ``[DONE]``.
* errors use mimOE's non-OpenAI bodies: ``{"message","statusCode"}`` for routes,
  ``{"error":{"code":403,"message":"Forbidden"}}`` (0.6) or the OpenAI-styled ``{"error":{...}}``
  (1.0) for a bad key on a completion, HTTP 503 with an empty body for an unknown container path.
  ``GET /models`` on the OpenAI path is public on both generations; the 1.0 registry ``GET``
  requires the key (``{"error":{"code":403,"message":"incorrect API key"}}``).
* ``GET /models`` shows generation-specific fields; 1.0 has no ``POST``/``DELETE /models`` route
  (404 ``not found``): it loads and unloads through ``PUT {store}/models {"id", "action"}``
  (verified live: SSE progress then ``{loaded: true}`` twice; unload answers
  ``{"id", "object": "model", "unloaded": true}``) and also loads a model on its first
  completion, streaming ``mimoe_status`` chunks.
* 1.0 streams a tool call as one complete delta (id, name, full arguments); 0.6 sends the id and
  name first and then argument fragments. Ids are ``tool_0``... on both (the real 1.0.27 uses
  ``call_0_xxxxxxxx``; per-response either way).

Unverified guesses, marked in the code: a 0.6 store rejects ``action: load/unload`` with a 400
naming the action, and a 1.0 store answers 404 when unloading a model that is not loaded.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import httpx
import pytest
from langchain_openai import ChatOpenAI

from mimoe_agent.config import ENV_PREFIX, Settings, load_settings
from mimoe_agent.llm import make_model
from mimoe_agent.mimoe import MimoeClient, Preflight, preflight, strip_node_prefix

NODE_ID = "976350e296903f45b2acc70bb52b75d7190775205d93b6226c2f0938"
REPO_WORKSPACE = Path(__file__).resolve().parent.parent / "workspace"
VERSIONS = {"0.6": "v3.22.8 (developer edition)", "1.0": "v3.30.26 (developer edition)"}
DEFAULT_REGISTRY = ("qwen3-4b", "qwen3-4b-instruct-2507", "smollm2-360m")


def _sse(obj: object) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def _pieces(text: str) -> list[str]:
    """Split text into word-sized streaming deltas (keeps the whitespace glued to the word)."""
    return re.findall(r"\s*\S+|\s+", text)


class FakeMimoe:
    """httpx.MockTransport-backed fake of both engine generations."""

    def __init__(
        self, generation: str = "0.6", model_id: str = "qwen3-4b", *, tools_ok: bool = True
    ) -> None:
        if generation not in VERSIONS:
            raise ValueError(f"generation must be one of {sorted(VERSIONS)}, got {generation!r}")
        self.generation = generation
        self.model_id = model_id
        self.tools_ok = tools_ok
        self.api_key = "1234"
        self.openai_path = "/mimik-ai/openai/v1"
        self.store_path = "/mimik-ai/store/v1"
        self.rpc_path = "/jsonrpc/v1"
        self.node_name = "fake-node"
        self.id_prefix = ""
        """Set to ``f"{NODE_ID}/"`` to emulate the airouter path, which prefixes model ids."""
        self.down = False
        """When ``True`` every request raises ``httpx.ConnectError`` (Studio not running)."""
        self.advertise_capabilities = generation == "1.0"
        """1.0 only: emit ``supported_parameters``/``reasoning``/``family`` in ``GET /models``."""
        self.thinking_supported = True
        self.thinking_can_disable = True
        self.model_kind = "llm"
        self.kinds: dict[str, str] = {}
        """Per-model ``info.kind`` overrides (default ``model_kind``)."""
        self.load_final_style = "0.6"
        """``"0.6"``: bare ``{id,object,kind,loaded}`` tail; ``"doc"``: a full model object line."""
        self.loaded: list[str] = [model_id] if model_id else []
        self.registry: list[str] = sorted({model_id, *DEFAULT_REGISTRY} - {""})
        self.registry_sizes: dict[str, int | None] = {}
        self.calls: list[dict] = []
        """Every chat body received, for assertions."""
        self.requests: list[httpx.Request] = []
        self._turns: list[dict] = []
        self._failures: list[tuple[int, object]] = []
        self._counter = 0

    # -- scripting ---------------------------------------------------------------------------

    def script(self, *turns: dict) -> None:
        """Queue assistant replies: ``{"content": str}`` or ``{"tool_calls": [{"name","args"}]}``;
        each may include ``"reasoning"``. Without a script the fake answers ``OK.``, or calls
        ``ping`` when a ``ping`` tool is offered and ``tools_ok`` is set."""
        self._turns.extend(turns)

    def fail_next(self, status: int, body: object = None) -> None:
        """Make the next ``/chat/completions`` request fail with ``status`` and ``body``
        (a dict is sent as JSON, a string verbatim, ``None`` as an empty body)."""
        self._failures.append((status, body))

    # -- clients -----------------------------------------------------------------------------

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=self.transport(), trust_env=False)

    def async_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=self.transport(), trust_env=False)

    @property
    def base_url(self) -> str:
        return "http://fake" + self.openai_path

    # -- routing -----------------------------------------------------------------------------

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.down:
            raise httpx.ConnectError("[Errno 61] Connection refused", request=request)
        path = request.url.path
        if path.startswith(self.openai_path + "/"):
            return self._openai(request, path[len(self.openai_path) :])
        if path.startswith(self.store_path + "/"):
            return self._store(request, path[len(self.store_path) :])
        if path == self.rpc_path:
            return self._rpc(request)
        return httpx.Response(503, content=b"")

    def _authorized(self, request: httpx.Request) -> bool:
        return request.headers.get("authorization") == f"Bearer {self.api_key}"

    def _forbidden(self) -> httpx.Response:
        if self.generation == "0.6":
            body: dict[str, Any] = {"error": {"code": 403, "message": "Forbidden"}}
        else:
            body = {
                "error": {
                    "message": "Forbidden",
                    "type": "permission_error",
                    "code": None,
                    "param": None,
                }
            }
        return httpx.Response(403, json=body)

    def _route_error(self, status: int, message: str) -> httpx.Response:
        if self.generation == "0.6":
            return httpx.Response(status, json={"statusCode": status, "message": message})
        return httpx.Response(
            status,
            json={
                "error": {
                    "message": message,
                    "type": "not_found_error" if status == 404 else "api_error",
                    "code": None,
                    "param": None,
                }
            },
        )

    def _openai(self, request: httpx.Request, route: str) -> httpx.Response:
        method = request.method
        # verified on 0.6.5 and 1.0.27: GET /models on the OpenAI path never checks the key
        if method != "GET" and not self._authorized(request):
            return self._forbidden()
        if route == "/models" and method == "GET":
            return httpx.Response(200, json=self.models_payload())
        if route == "/models" and method == "POST" and self.generation == "0.6":
            return self._load(request)
        if route == "/models" and method == "DELETE" and self.generation == "0.6":
            return self._unload(request)
        if route == "/chat/completions" and method == "POST":
            return self._chat(request)
        return httpx.Response(404, json={"statusCode": 404, "message": "not found"})

    def _store(self, request: httpx.Request, route: str) -> httpx.Response:
        if self.generation == "1.0" and not self._authorized(request):
            # verified on 1.0.27: the registry's GET requires the key, unlike the OpenAI path
            return httpx.Response(
                403, json={"error": {"code": 403, "message": "incorrect API key"}}
            )
        if route == "/models" and request.method == "GET":
            return httpx.Response(
                200, json={"data": [self._registry_entry(m) for m in self.registry]}
            )
        if route == "/models" and request.method == "PUT":
            return self._store_action(request)
        return httpx.Response(404, json={"error": {"code": 404, "message": "not found"}})

    def _store_action(self, request: httpx.Request) -> httpx.Response:
        """``PUT {store}/models {"id", "action"}``: load/unload on 1.0 (verified live)."""
        body = json.loads(request.content or b"{}")
        model_id = str(body.get("id") or "")
        action = str(body.get("action") or "")
        if self.generation == "0.6" or action not in ("load", "unload"):
            # guess: the 0.6 store only documents action "update"
            return httpx.Response(
                400, json={"error": {"code": 400, "message": f"unsupported action: {action}"}}
            )
        if action == "unload":
            if model_id not in self.loaded:
                # guess: not captured live
                return httpx.Response(
                    404, json={"error": {"code": 404, "message": f"model not loaded: {model_id}"}}
                )
            self.loaded.remove(model_id)
            return httpx.Response(200, json={"id": model_id, "object": "model", "unloaded": True})
        if model_id not in self.registry:
            return httpx.Response(
                404, json={"error": {"code": 404, "message": f"model not found: {model_id}"}}
            )
        if model_id not in self.loaded:
            self.loaded.append(model_id)
        lines = [_sse({"progress": f"<|loading_model|> {pct}%<br />\n"}) for pct in (0, 50, 100)]
        final = _sse({"id": model_id, "object": "model", "kind": "llm", "loaded": True})
        return httpx.Response(
            200,
            content=("".join(lines) + final + final).encode(),
            headers={"content-type": "text/event-stream"},
        )

    def _rpc(self, request: httpx.Request) -> httpx.Response:
        try:
            body = json.loads(request.content)
        except ValueError:
            body = {}
        if body.get("method") != "getMe":
            return httpx.Response(
                200,
                json={
                    "id": body.get("id"),
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": "Method not found"},
                },
            )
        result: dict[str, Any] = {
            "accountId": "",
            "linkLocalIp": "192.168.1.216",
            "name": self.node_name,
            "nodeId": NODE_ID,
            "supernodeTypeName": "_mk-v15-4996e4c2442cc796f2c0ddb4e5e1627d._tcp",
            "version": VERSIONS[self.generation],
        }
        if self.generation == "1.0":
            result["localOnly"] = "true"
        return httpx.Response(
            200, json={"id": body.get("id", 1), "jsonrpc": "2.0", "result": result}
        )

    # -- payload builders --------------------------------------------------------------------

    def model_entry(self, model_id: str) -> dict[str, Any]:
        """The ``GET /models`` entry for ``model_id`` in this generation's shape."""
        info: dict[str, Any] = {
            "kind": self.kinds.get(model_id, self.model_kind),
            "chat_template_hint": "",
            "chatTemplate": "chatml",
            "n_gpu_layers": -1,
            "max_context": 12000,
            "n_vocab": 151936,
            "n_ctx_train": 40960,
            "n_embd": 2560,
            "n_params": 4022468096,
            "model_size": 2491323904,
        }
        if self.generation == "1.0" and self.advertise_capabilities:
            params = ["tools", "tool_choice"] if self.tools_ok else ["tool_choice"]
            if self.thinking_can_disable:
                params.append("enable_thinking")
            reasoning: dict[str, Any] = {"supported": self.thinking_supported}
            if self.thinking_supported:
                reasoning.update(
                    {"default_enabled": True, "can_disable": self.thinking_can_disable}
                )
            info.update(
                {
                    "family": "qwen3",
                    "input_modalities": ["text"],
                    "supported_parameters": params,
                    "reasoning": reasoning,
                    "capability_source": {
                        "input_modalities": "discovered",
                        "supported_parameters": "discovered",
                        "reasoning": "discovered",
                    },
                }
            )
        now = int(time.time())
        metrics = {
            "inference_count": len(self.calls),
            "last_used": now,
            "loaded_at": now - 60,
            "tokens_per_second": 35.4,
            "avg_tokens_per_second": 32.3,
        }
        return {
            "id": self.id_prefix + model_id,
            "object": "model",
            "created": now - 60,
            "owned_by": "mimik",
            "info": info,
            "metrics": metrics,
        }

    def models_payload(self) -> dict[str, Any]:
        return {"data": [self.model_entry(m) for m in self.loaded], "object": "list"}

    def _registry_entry(self, model_id: str) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "id": model_id,
            "version": "1.0.0",
            "kind": "llm",
            "readyToUse": True,
            "createdAt": 1790318912697,
            "gguf": {"initContextSize": 12000, "initGpuLayerSize": 99},
        }
        size = self.registry_sizes.get(model_id, 2497281120)
        if self.generation == "1.0":
            entry.update({"_v": 2, "namespace": "", "status": "ready", "statusMessage": ""})
            if size is None:
                entry["localPath"] = f"/Users/me/.mimoe/models/{model_id}.gguf"
            else:
                entry["filePath"] = f"{model_id}/{model_id}.gguf"
        if size is not None:
            entry["totalSize"] = size
        return entry

    def _usage(self, completion_tokens: int) -> dict[str, Any]:
        return {
            "prompt_tokens": 120,
            "completion_tokens": completion_tokens,
            "total_tokens": 120 + completion_tokens,
            "completion_tokens_details": None,
            "prompt_tokens_details": None,
            "token_per_second": 35.4,
        }

    # -- model load / unload (0.6 routes) ----------------------------------------------------

    def _load(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        model_id = str(body.get("model") or "")
        if model_id not in self.registry:
            return httpx.Response(
                404, json={"message": f"Model '{model_id}' not found in store", "statusCode": 404}
            )
        lines = [_sse({"progress": f"<|loading_model|> {pct}%<br />"}) for pct in (0, 50, 100)]
        if model_id not in self.loaded:
            self.loaded.append(model_id)
        if self.load_final_style == "doc":
            tail = _sse(self.model_entry(model_id))
        else:
            tail = json.dumps({"id": model_id, "object": "model", "kind": "llm", "loaded": True})
        return httpx.Response(
            200,
            content=("".join(lines) + tail).encode(),
            headers={"content-type": "text/event-stream"},
        )

    def _unload(self, request: httpx.Request) -> httpx.Response:
        model_id = parse_qs(request.url.query.decode()).get("modelId", [""])[0]
        if model_id not in self.loaded:
            return httpx.Response(
                404, json={"statusCode": 404, "message": f"model not found in cache: {model_id}"}
            )
        self.loaded.remove(model_id)
        return httpx.Response(200, json={"id": model_id, "object": "model", "deleted": True})

    # -- chat ----------------------------------------------------------------------------------

    def _default_turn(self, body: dict) -> dict:
        names = [
            (tool.get("function") or {}).get("name")
            for tool in body.get("tools") or []
            if isinstance(tool, dict)
        ]
        if "ping" in names and self.tools_ok:
            return {"tool_calls": [{"name": "ping", "args": {}}]}
        return {"content": "OK."}

    def _chat(self, request: httpx.Request) -> httpx.Response:
        if self._failures:
            status, body = self._failures.pop(0)
            if isinstance(body, dict | list):
                return httpx.Response(status, json=body)
            return httpx.Response(status, content=(body or "").encode())
        body = json.loads(request.content)
        self.calls.append(body)
        model_id = strip_node_prefix(str(body.get("model") or ""), NODE_ID)
        loading = False
        if model_id not in self.loaded:
            if model_id not in self.registry:
                return httpx.Response(
                    404,
                    json={
                        "message": f"Model '{model_id}' not found in mmodelstore",
                        "statusCode": 404,
                    },
                )
            self.loaded.append(model_id)
            loading = True
        turn = self._turns.pop(0) if self._turns else self._default_turn(body)
        self._counter += 1
        chat_id = f"chatcmpl-{self._counter}"
        content = str(turn.get("content") or "")
        reasoning = turn.get("reasoning")
        tool_calls = [
            {
                "id": f"tool_{i}",
                "type": "function",
                "function": {
                    "name": str(call["name"]),
                    "arguments": json.dumps(call.get("args") or {}),
                },
            }
            for i, call in enumerate(turn.get("tool_calls") or [])
        ]
        finish = "tool_calls" if tool_calls else "stop"
        tokens = max(1, len(content.split()) + sum(6 for _ in tool_calls))
        if body.get("stream"):
            return self._stream_reply(
                chat_id, model_id, content, reasoning, tool_calls, finish, tokens, loading
            )
        message: dict[str, Any] = {"role": "assistant"}
        if self.generation == "0.6":
            think = f"<think>\n{reasoning}\n</think>" if reasoning else "<think>\n\n</think>"
            message["content"] = think + (f"\n\n{content}" if content else "")
        else:
            message["content"] = content
            if reasoning:
                message["reasoning_content"] = reasoning
        if tool_calls:
            message["tool_calls"] = tool_calls
        return httpx.Response(
            200,
            json={
                "id": chat_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": self._usage(tokens),
            },
        )

    def _stream_reply(
        self,
        chat_id: str,
        model_id: str,
        content: str,
        reasoning: str | None,
        tool_calls: list[dict],
        finish: str,
        tokens: int,
        loading: bool,
    ) -> httpx.Response:
        base = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_id,
        }

        def delta(d: dict[str, Any]) -> dict[str, Any]:
            return {**base, "choices": [{"index": 0, "delta": d, "finish_reason": None}]}

        def status(stage: str, progress: float | None, message: str) -> dict[str, Any]:
            return {
                **base,
                "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
                "mimoe_status": {"stage": stage, "progress": progress, "message": message},
            }

        chunks: list[dict[str, Any]] = [delta({"role": "assistant"})]
        if loading:
            for pct in (0, 50, 100):
                chunks.append(status("loading_model", pct / 100, f"Loading model ({pct}%)"))
        chunks.append(status("processing_prompt", 0.5, "Processing prompt (50%)"))
        if self.generation == "0.6":
            pieces = (
                ["<think>", f"\n{reasoning}", "\n</think>"]
                if reasoning
                else ["<think>", "\n\n</think>"]
            )
            if content:
                pieces += ["\n\n", *_pieces(content)]
            chunks.extend(delta({"content": p}) for p in pieces)
        else:
            if reasoning:
                chunks.extend(delta({"reasoning_content": p}) for p in _pieces(reasoning))
            chunks.extend(delta({"content": p}) for p in _pieces(content))
        for index, call in enumerate(tool_calls):
            args = call["function"]["arguments"]
            # verified: 1.0.27 sends the whole call in one delta, 0.6.5 fragments the arguments
            first_args = args if self.generation == "1.0" else ""
            chunks.append(
                delta(
                    {
                        "tool_calls": [
                            {
                                "index": index,
                                "id": call["id"],
                                "type": "function",
                                "function": {
                                    "name": call["function"]["name"],
                                    "arguments": first_args,
                                },
                            }
                        ]
                    }
                )
            )
            for start in range(len(first_args), len(args), 6):
                chunks.append(
                    delta(
                        {
                            "tool_calls": [
                                {
                                    "index": index,
                                    "type": "function",
                                    "function": {"arguments": args[start : start + 6]},
                                }
                            ]
                        }
                    )
                )
        chunks.append(
            {
                **base,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                "usage": self._usage(tokens),
            }
        )
        payload = "".join(_sse(c) for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(
            200, content=payload.encode(), headers={"content-type": "text/event-stream"}
        )


# -- fixtures ------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_mimoe_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's shell (MIMOE_*, OPENAI_*, LangSmith) out of every test."""
    for name in list(os.environ):
        if name.startswith((ENV_PREFIX, "OPENAI_", "LANGSMITH_", "LANGCHAIN_")):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fake_mimoe() -> FakeMimoe:
    return FakeMimoe("0.6")


@pytest.fixture
def fake_mimoe_v10() -> FakeMimoe:
    return FakeMimoe("1.0")


@pytest.fixture
def workspace_tmp(tmp_path: Path) -> Path:
    """A copy of the sample ``./workspace`` inside ``tmp_path``."""
    dest = tmp_path / "workspace"
    shutil.copytree(REPO_WORKSPACE, dest)
    return dest


@pytest.fixture
def settings_tmp(workspace_tmp: Path) -> Settings:
    return load_settings(
        {"workspace": workspace_tmp, "base_url": "http://fake/mimik-ai/openai/v1"},
        cwd=workspace_tmp.parent,
    )


@pytest.fixture
def preflight_fake(fake_mimoe: FakeMimoe, settings_tmp: Settings) -> Preflight:
    client = MimoeClient(settings_tmp.base_url, settings_tmp.api_key, client=fake_mimoe.client())
    return preflight(settings_tmp, client=client)


@pytest.fixture
def llm_fake(
    fake_mimoe: FakeMimoe, settings_tmp: Settings, preflight_fake: Preflight
) -> ChatOpenAI:
    return make_model(
        settings_tmp,
        preflight_fake,
        http_client=fake_mimoe.client(),
        http_async_client=fake_mimoe.async_client(),
    )


@pytest.fixture
def sse_events() -> Any:
    """Helper: decode an SSE body into the list of ``data:`` JSON objects (``[DONE]`` skipped)."""

    def decode(body: bytes) -> Iterator[dict]:
        for line in body.decode().splitlines():
            if line.startswith("data: ") and line[6:] != "[DONE]":
                yield json.loads(line[6:])

    return lambda body: list(decode(body))
