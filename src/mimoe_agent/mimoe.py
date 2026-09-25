"""mimOE engine client: discovery, model listing, load/unload, the tool probe and preflight.

Sync ``httpx`` only and no LangChain imports. :func:`friendly_error` recognises the openai SDK's
exception classes directly and langchain-core's ``Model*Error`` wrappers by name, so it can map
whatever ``ChatOpenAI`` raises without importing LangChain here.

Two engine generations are supported:

* **0.6** (Studio 0.6.5, engine v3.22, milm 1.14): ``GET/POST/DELETE /models`` on the OpenAI path,
  thinking arrives inline as ``<think>`` text and only ``/no_think`` switches it off.
* **1.0** (Studio 1.0.27, engine v3.30, milm 1.17): ``GET /models`` entries carry
  ``supported_parameters``, ``reasoning{supported,default_enabled,can_disable}`` and ``family``
  inside ``info``; ``enable_thinking`` is honoured; reasoning arrives as ``reasoning_content``.
  ``POST``/``DELETE /models`` on the OpenAI path answer 404; loading and unloading go through the
  store, ``PUT {store}/models {"id", "action": "load"|"unload"}`` (verified live in the
  compatibility matrix: same SSE progress stream, final ``{loaded: true}`` / ``{unloaded: true}``).
  A model also loads on its first completion (progress streamed as ``mimoe_status`` chunks), which
  :meth:`MimoeClient.load_model` uses as the last resort.

``GET /models`` on the OpenAI path never checks the API key (both generations); the 1.0 registry
``GET`` does, and completions do on both.
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
import openai

from mimoe_agent.config import ConfigError, Settings

CANDIDATE_BASE_URLS = (
    "http://127.0.0.1:8083/mimik-ai/openai/v1",
    "http://127.0.0.1:8083/openai/v1",
    "http://localhost:8083/mimik-ai/openai/v1",
)
"""Tried in order when no base URL is configured. The IPv4 loopback comes first: ``localhost``
resolves to ``::1`` first, and Windows spends about two seconds on every refused connection
there before it tries ``127.0.0.1`` (mimOE listens on IPv4)."""

GET_TIMEOUT_S = 5.0
RPC_TIMEOUT_S = 3.0
PROBE_TIMEOUT_S = 60.0
LOAD_TIMEOUT_S = 300.0
UNLOAD_TIMEOUT_S = 60.0
"""Unloading waits for in-flight requests (about 13 s observed on 1.0.27), so more than a GET."""
PROBE_MAX_TOKENS = 48
V10_MIN_VERSION = (3, 30)

_STUDIO_ICON = "system-tray icon" if sys.platform == "win32" else "menu-bar icon"
_ENGINE_DIR = r"%USERPROFILE%\.mimoe" if sys.platform == "win32" else "~/.mimoe"

HINT_NOT_REACHABLE = (
    f"Open mimOE Studio ({_STUDIO_ICON}) and wait for the status dot to turn green, then retry; "
    "if Studio listens on another port pass --base-url http://localhost:PORT/mimik-ai/openai/v1."
)
HINT_NO_MODEL = "Studio > Models > Load a chat model (for example qwen3-4b), then run again."
HINT_API_KEY = (
    "Studio shows the API key under the API button (default 1234); "
    "pass --api-key KEY or set MIMOE_API_KEY."
)
HINT_CONTEXT = (
    "Start a new conversation with /new (web UI: New conversation). If it happens early in a "
    "conversation, the engine is short of memory: close other apps or load a smaller model."
)
HINT_STREAM_CUT = (
    "The conversation may have outgrown the model's context window: start a new one with /new "
    "(web UI: New conversation). If it keeps happening, the engine may be out of memory: close "
    "other apps or load a smaller model."
)
HINT_WRONG_PATH = (
    "Use --base-url http://localhost:8083/mimik-ai/openai/v1 "
    "(older builds: http://localhost:8083/openai/v1)."
)
HINT_MODEL_NOT_READY = (
    "The model is registered but its file is missing; Studio > Models > Download it first."
)
HINT_SERVER_ERROR = (
    "Studio > Models: reload the model if it crashed, then retry; the engine log is under "
    f"{_ENGINE_DIR}."
)
HINT_TIMEOUT = "The model may be busy or still loading; wait a moment and retry."
HINT_PROBE_FAILED = (
    "The model produced no structured tool call, so the agent runs in chat-only mode (no file, "
    "Python or git tools). Use a tool-capable model such as qwen3-4b (Studio > Models > Load), "
    "or pass --force-tools to try anyway."
)
HINT_NO_UNLOAD = (
    "Neither DELETE {base}/models nor PUT {store}/models {action: unload} exists on this engine: "
    "unload from Studio > Models, or simply load the other model."
)

_NODE_PREFIX_RE = re.compile(r"^[0-9a-fA-F]{32,}/")
_VERSION_RE = re.compile(r"v?(\d+)\.(\d+)")

ThinkingControl = Literal["native", "soft", "none"]


class MimoeError(Exception):
    """An engine problem explained for humans: ``message`` says what, ``hint`` says what to do."""

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


class EngineGeneration(StrEnum):
    """Which mimOE API generation the engine speaks."""

    V06 = "0.6"
    V10 = "1.0"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class EngineInfo:
    """Where the engine lives and what it is."""

    base_url: str
    store_url: str
    rpc_url: str
    node_name: str | None
    version: str | None
    generation: EngineGeneration


@dataclass(frozen=True)
class LoadedModel:
    """One entry of ``GET /models`` (a model currently in the runtime cache)."""

    id: str
    kind: str
    family: str | None
    max_context: int | None
    n_params: int | None
    tokens_per_second: float | None
    avg_tokens_per_second: float | None
    supports_tools: bool | None
    thinking_supported: bool | None
    thinking_can_disable: bool | None
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class RegistryModel:
    """One entry of the model registry (``GET {store}/models``)."""

    id: str
    kind: str
    ready: bool
    size_bytes: int | None
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of the warm-up completion with the ``ping`` tool."""

    tools_ok: bool
    latency_s: float
    detail: str


@dataclass(frozen=True)
class Preflight:
    """Everything the agent needs to know before its first model call."""

    engine: EngineInfo
    model: LoadedModel
    probe: ProbeResult | None
    tools_enabled: bool
    thinking_control: ThinkingControl
    warnings: tuple[str, ...]


def strip_node_prefix(model_id: str, node_id: str | None = None) -> str:
    """Return the bare model id: the airouter path lists ``<nodeId>/<id>``."""
    if node_id and model_id.startswith(node_id + "/"):
        return model_id[len(node_id) + 1 :]
    return _NODE_PREFIX_RE.sub("", model_id, count=1)


def derive_urls(base_url: str) -> tuple[str, str]:
    """Return ``(store_url, rpc_url)`` for an OpenAI-compatible base URL.

    ``.../mimik-ai/openai/v1`` -> ``.../mimik-ai/store/v1``; ``.../openai/v1`` -> ``.../store/v1``;
    the airouter path has no store of its own and maps to ``mimik-ai``; the JSON-RPC endpoint is
    always ``{origin}/jsonrpc/v1``.
    """
    base = base_url.rstrip("/")
    parts = urlsplit(base)
    origin = f"{parts.scheme}://{parts.netloc}"
    match = re.match(r"^(?P<prefix>.*?)/openai/v1$", parts.path)
    prefix = match.group("prefix") if match else "/mimik-ai"
    if prefix == "/mimik-airouter":
        prefix = "/mimik-ai"
    return f"{origin}{prefix}/store/v1", f"{origin}/jsonrpc/v1"


def parse_version(version: str | None) -> tuple[int, int] | None:
    """Return ``(major, minor)`` from ``"v3.30.26 (developer edition)"``, or ``None``."""
    if not version:
        return None
    match = _VERSION_RE.search(version)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def detect_generation(
    entries: Iterable[Mapping[str, Any]], version: str | None
) -> EngineGeneration:
    """V10 when a ``/models`` entry has ``supported_parameters`` or the engine is >= v3.30."""
    for entry in entries:
        info = entry.get("info")
        if "supported_parameters" in entry or (
            isinstance(info, Mapping) and "supported_parameters" in info
        ):
            return EngineGeneration.V10
    parsed = parse_version(version)
    if parsed is not None and parsed >= V10_MIN_VERSION:
        return EngineGeneration.V10
    return EngineGeneration.V06


def is_remote(model: LoadedModel) -> bool:
    """True for ``GET /models`` entries that mimOE does not run on this machine.

    0.6 engines list a configured cloud model as ``owned_by: "cloud"``; 1.0 engines list
    provider models (Ollama, cloud APIs) with ``owned_by: "provider:<id>"``, ``attached: true``
    and ``info.cloud``. Prompts sent to those leave the machine, so they are never auto-picked.
    """
    raw = model.raw
    owner = str(raw.get("owned_by") or "").lower()
    info = raw.get("info") if isinstance(raw.get("info"), Mapping) else {}
    return (
        owner == "cloud"
        or owner.startswith("provider")
        or bool(raw.get("attached"))
        or bool(info.get("cloud"))
    )


def parse_loaded_model(entry: Mapping[str, Any], node_id: str | None = None) -> LoadedModel:
    """Parse one ``GET /models`` entry of either generation (tolerates missing fields)."""
    info = entry.get("info") if isinstance(entry.get("info"), Mapping) else {}
    metrics = entry.get("metrics") if isinstance(entry.get("metrics"), Mapping) else {}
    caps: Mapping[str, Any] = (
        info if "supported_parameters" in info or "reasoning" in info else entry
    )
    supported = caps.get("supported_parameters")
    reasoning = caps.get("reasoning") if isinstance(caps.get("reasoning"), Mapping) else None
    family = info.get("family") if info.get("family") is not None else entry.get("family")
    return LoadedModel(
        id=strip_node_prefix(str(entry.get("id") or entry.get("model") or ""), node_id),
        kind=str(info.get("kind") or entry.get("kind") or "llm"),
        family=str(family) if family else None,
        max_context=_opt_int(info.get("max_context")),
        n_params=_opt_int(info.get("n_params")),
        tokens_per_second=_opt_float(metrics.get("tokens_per_second")),
        avg_tokens_per_second=_opt_float(metrics.get("avg_tokens_per_second")),
        supports_tools=("tools" in supported) if isinstance(supported, list) else None,
        thinking_supported=_opt_bool(reasoning.get("supported")) if reasoning else None,
        thinking_can_disable=_opt_bool(reasoning.get("can_disable")) if reasoning else None,
        raw=entry,
    )


def parse_registry_model(entry: Mapping[str, Any]) -> RegistryModel:
    """Parse one registry entry; ``totalSize`` is absent for models registered by local path."""
    return RegistryModel(
        id=str(entry.get("id") or ""),
        kind=str(entry.get("kind") or "llm"),
        ready=bool(entry.get("readyToUse")),
        size_bytes=_opt_int(entry.get("totalSize")),
        raw=entry,
    )


def probe_body(model_id: str, *, thinking_control: str) -> dict[str, Any]:
    """Build the warm-up request: one ``ping`` tool, 48 tokens, thinking switched off."""
    system = "You are a tool-using assistant."
    user = "Call the ping tool now."
    body: dict[str, Any] = {
        "model": model_id,
        "temperature": 0,
        "max_tokens": PROBE_MAX_TOKENS,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "ping",
                    "description": "Reply with pong. Call it when asked to ping.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    }
    if thinking_control == "native":
        body["enable_thinking"] = False
    elif thinking_control == "soft":
        system = "/no_think\n" + system
        user += " /no_think"
    body["messages"] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return body


class MimoeClient:
    """Thin sync client for one mimOE node (OpenAI path, model registry and JSON-RPC ``getMe``)."""

    def __init__(
        self,
        base_url: str | None,
        api_key: str,
        *,
        timeout: float = GET_TIMEOUT_S,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url_hint = base_url.rstrip("/") if base_url else None
        self.api_key = api_key
        self.timeout = timeout
        self._client = client or httpx.Client(trust_env=False, timeout=timeout)
        self._owns_client = client is None
        self._engine: EngineInfo | None = None

    # -- lifecycle -----------------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying ``httpx.Client`` if this instance created it."""
        if self._owns_client:
            self._client.close()

    @property
    def engine(self) -> EngineInfo:
        """The discovered engine (runs :meth:`discover` on first use)."""
        if self._engine is None:
            self._engine = self.discover()
        return self._engine

    # -- discovery -----------------------------------------------------------------------------

    def discover(self) -> EngineInfo:
        """Find the engine: the configured base URL when one was given, else the candidates.

        An explicit URL is never silently replaced by a candidate: with two engines installed
        (Studio 0.6.5 on 8083, a 1.0 runtime on 8093) falling back would connect the agent to the
        wrong one.

        Raises:
            MimoeError: ``"mimOE Studio is not reachable"`` (hint lists every URL tried) or a
                rejected API key when an engine answered 401/403.
        """
        candidates = [self.base_url_hint] if self.base_url_hint else list(CANDIDATE_BASE_URLS)
        failures: list[str] = []
        for base in candidates:
            try:
                resp = self._client.get(
                    base + "/models", headers=self._headers(), timeout=self._timeout(GET_TIMEOUT_S)
                )
            except httpx.TimeoutException:
                failures.append(f"{base}: no answer within {GET_TIMEOUT_S:.0f} s")
                continue
            except httpx.TransportError as exc:
                failures.append(f"{base}: {exc.__class__.__name__}")
                continue
            if resp.status_code in (401, 403):
                raise MimoeError("mimOE rejected the API key", hint=HINT_API_KEY)
            entries = _model_entries(resp) if resp.status_code == 200 else None
            if entries is None:
                what = "not a model list" if resp.status_code == 200 else f"HTTP {resp.status_code}"
                failures.append(f"{base}: {what}")
                continue
            store_url, rpc_url = derive_urls(base)
            node = self._get_me(rpc_url)
            version = _opt_str(node.get("version"))
            self._engine = EngineInfo(
                base_url=base,
                store_url=store_url,
                rpc_url=rpc_url,
                node_name=_opt_str(node.get("name")),
                version=version,
                generation=detect_generation(entries, version),
            )
            return self._engine
        raise MimoeError(
            "mimOE Studio is not reachable",
            hint=f"{HINT_NOT_REACHABLE} Tried: {'; '.join(failures)}.",
        )

    def node_info(self) -> dict[str, Any]:
        """Return the JSON-RPC ``getMe`` result, or ``{}`` when the node does not answer."""
        try:
            return self._get_me(self.engine.rpc_url)
        except MimoeError:
            return {}

    # -- models --------------------------------------------------------------------------------

    def loaded_models(self) -> list[LoadedModel]:
        """List the models in the runtime cache (``GET /models``)."""
        engine = self.engine
        resp = self._request("GET", engine.base_url + "/models", timeout=GET_TIMEOUT_S)
        entries = _model_entries(resp)
        if entries is None:
            raise MimoeError(
                f"unexpected reply from {engine.base_url}/models", hint=HINT_WRONG_PATH
            )
        return [parse_loaded_model(entry) for entry in entries]

    def registry_models(self) -> list[RegistryModel]:
        """List the models registered in the model store (``GET {store}/models``)."""
        engine = self.engine
        resp = self._request("GET", engine.store_url + "/models", timeout=GET_TIMEOUT_S)
        entries = _model_entries(resp)
        if entries is None:
            raise MimoeError(
                f"unexpected reply from {engine.store_url}/models", hint=HINT_WRONG_PATH
            )
        return [parse_registry_model(entry) for entry in entries]

    def load_model(
        self, model_id: str, on_progress: Callable[[str], None] | None = None
    ) -> LoadedModel:
        """Load ``model_id`` into the runtime cache and return its ``GET /models`` entry.

        Generation 0.6 streams ``POST {base}/models {"model"}``, generation 1.0 streams
        ``PUT {store}/models {"id", "action": "load"}`` (progress lines, then a final JSON object);
        each is tried as the other's fallback when its route is missing, and as a last resort a
        one-token streaming completion triggers the engine's auto-load, with its ``mimoe_status``
        chunks feeding ``on_progress``.
        """
        engine = self.engine
        model_id = strip_node_prefix(model_id)
        notify = on_progress or (lambda _text: None)
        store_first = engine.generation is EngineGeneration.V10
        for via_store in (store_first, not store_first):
            loaded = self._load_via_route(model_id, notify, via_store=via_store)
            if loaded is not None:
                return loaded
        self._load_via_completion(model_id, notify)
        return self._find_loaded(model_id)

    def unload_model(self, model_id: str) -> None:
        """Remove ``model_id`` from the runtime cache.

        Generation 0.6: ``DELETE {base}/models?modelId=``; generation 1.0:
        ``PUT {store}/models {"id", "action": "unload"}``; each is the other's fallback.

        Raises:
            MimoeError: when the model is not loaded, or when neither unload route exists.
        """
        engine = self.engine
        model_id = strip_node_prefix(model_id)
        store_first = engine.generation is EngineGeneration.V10
        for via_store in (store_first, not store_first):
            if via_store:
                resp = self._request(
                    "PUT",
                    engine.store_url + "/models",
                    json={"id": model_id, "action": "unload"},
                    timeout=UNLOAD_TIMEOUT_S,
                    check=False,
                )
            else:
                resp = self._request(
                    "DELETE",
                    engine.base_url + "/models",
                    params={"modelId": model_id},
                    timeout=UNLOAD_TIMEOUT_S,
                    check=False,
                )
            if resp.is_success:
                return
            body = _parse_body(resp)
            if _route_missing(resp.status_code, body):
                continue
            if resp.status_code == 404:
                loaded = ", ".join(m.id for m in self.loaded_models()) or "none"
                raise MimoeError(f"model '{model_id}' is not loaded (loaded: {loaded})")
            raise _status_error(resp.status_code, body, url=str(resp.request.url))
        raise MimoeError("this mimOE build has no unload endpoint", hint=HINT_NO_UNLOAD)

    # -- completions ---------------------------------------------------------------------------

    def probe_tools(self, model_id: str, *, thinking_control: str) -> ProbeResult:
        """Run one warm-up completion with a ``ping`` tool and report whether the model called it.

        The call also validates the API key (a 403 raises :class:`MimoeError` with the key hint)
        and absorbs the cold start of a freshly loaded model.
        """
        model_id = strip_node_prefix(model_id)
        started = time.perf_counter()
        try:
            reply = self.chat(probe_body(model_id, thinking_control=thinking_control))
        except MimoeError as exc:
            if "not answer" in exc.message:
                raise MimoeError(
                    f"the warm-up completion for {model_id} timed out after "
                    f"{PROBE_TIMEOUT_S:.0f} s",
                    hint=HINT_TIMEOUT + " Or pass --force-tools to skip the probe.",
                ) from exc
            raise
        latency = time.perf_counter() - started
        choices = reply.get("choices") if isinstance(reply, Mapping) else None
        choice = choices[0] if isinstance(choices, list) and choices else {}
        message = choice.get("message") if isinstance(choice, Mapping) else None
        message = message if isinstance(message, Mapping) else {}
        names = [
            str((call.get("function") or {}).get("name"))
            for call in message.get("tool_calls") or []
            if isinstance(call, Mapping)
        ]
        if "ping" in names:
            return ProbeResult(
                True, latency, f"{model_id} answered with a structured ping call in {latency:.1f} s"
            )
        content = str(message.get("content") or "").strip()
        finish = choice.get("finish_reason") if isinstance(choice, Mapping) else None
        detail = (
            f"{model_id} answered without a tool call in {latency:.1f} s "
            f"(finish_reason={finish}, tool names={names or 'none'}, content={content[:80]!r})"
        )
        return ProbeResult(False, latency, detail)

    def chat(self, body: Mapping[str, Any]) -> dict[str, Any]:
        """``POST /chat/completions`` without streaming and return the decoded JSON reply."""
        engine = self.engine
        resp = self._request(
            "POST",
            engine.base_url + "/chat/completions",
            json={**body, "stream": False},
            timeout=PROBE_TIMEOUT_S,
        )
        try:
            data = resp.json()
        except ValueError as exc:
            raise MimoeError("mimOE returned a non-JSON chat reply", hint=HINT_WRONG_PATH) from exc
        if not isinstance(data, dict):
            raise MimoeError("mimOE returned an unexpected chat reply", hint=HINT_WRONG_PATH)
        return data

    # -- internals -----------------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    @staticmethod
    def _timeout(seconds: float) -> httpx.Timeout:
        return httpx.Timeout(seconds, connect=GET_TIMEOUT_S)

    def _request(
        self,
        method: str,
        url: str,
        *,
        timeout: float,
        json: Any = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        check: bool = True,
    ) -> httpx.Response:
        """Send one request, turning transport failures (and, if ``check``, bad statuses) into
        :class:`MimoeError`."""
        try:
            resp = self._client.request(
                method,
                url,
                json=json,
                params=params,
                headers={**self._headers(), **(headers or {})},
                timeout=self._timeout(timeout),
            )
        except httpx.TimeoutException as exc:
            raise MimoeError(
                f"mimOE did not answer within {timeout:.0f} s ({method} {url})", hint=HINT_TIMEOUT
            ) from exc
        except httpx.TransportError as exc:
            raise MimoeError(
                "mimOE Studio is not reachable",
                hint=f"{HINT_NOT_REACHABLE} ({exc.__class__.__name__} for {url})",
            ) from exc
        if check:
            self._raise_for_status(resp)
        return resp

    def _raise_for_status(self, resp: httpx.Response) -> None:
        if resp.is_success:
            return
        raise _status_error(resp.status_code, _parse_body(resp), url=str(resp.request.url))

    def _get_me(self, rpc_url: str) -> dict[str, Any]:
        payload = {"jsonrpc": "2.0", "id": 1, "method": "getMe", "params": []}
        try:
            resp = self._client.post(
                rpc_url, json=payload, headers=self._headers(), timeout=self._timeout(RPC_TIMEOUT_S)
            )
            data = resp.json() if resp.is_success else None
        except (httpx.HTTPError, ValueError):
            return {}
        result = data.get("result") if isinstance(data, Mapping) else None
        return dict(result) if isinstance(result, Mapping) else {}

    def _find_loaded(self, model_id: str) -> LoadedModel:
        for model in self.loaded_models():
            if model.id == model_id:
                return model
        raise MimoeError(
            f"'{model_id}' is not listed by GET /models after loading",
            hint="Studio > Models shows the load error; check the engine log under ~/.mimoe.",
        )

    def _load_via_route(
        self, model_id: str, notify: Callable[[str], None], *, via_store: bool
    ) -> LoadedModel | None:
        """Stream one load route (store ``PUT`` or OpenAI-path ``POST``); ``None`` if missing."""
        engine = self.engine
        if via_store:
            method, url = "PUT", engine.store_url + "/models"
            payload: dict[str, Any] = {"id": model_id, "action": "load"}
        else:
            method, url = "POST", engine.base_url + "/models"
            payload = {"model": model_id}
        final: dict[str, Any] | None = None
        try:
            with self._client.stream(
                method,
                url,
                json=payload,
                headers={**self._headers(), "Accept": "text/event-stream"},
                timeout=self._timeout(LOAD_TIMEOUT_S),
            ) as resp:
                if not resp.is_success:
                    resp.read()
                    body = _parse_body(resp)
                    if _route_missing(resp.status_code, body):
                        return None
                    raise _status_error(resp.status_code, body, url=str(resp.request.url))
                for obj in _iter_sse_json(resp.iter_lines()):
                    if "progress" in obj:
                        notify(_progress_text(str(obj["progress"])))
                    elif _is_error_body(obj):
                        raise _status_error(_body_status(obj) or 500, obj, url=str(resp.url))
                    else:
                        final = obj
        except httpx.TimeoutException as exc:
            raise MimoeError(
                f"loading {model_id} took longer than {LOAD_TIMEOUT_S:.0f} s", hint=HINT_TIMEOUT
            ) from exc
        except httpx.TransportError as exc:
            raise MimoeError("mimOE Studio is not reachable", hint=HINT_NOT_REACHABLE) from exc
        if final is not None and isinstance(final.get("info"), Mapping):
            return parse_loaded_model(final)
        return self._find_loaded(model_id)

    def _load_via_completion(self, model_id: str, notify: Callable[[str], None]) -> None:
        """Trigger the engine's auto-load with a one-token streaming completion."""
        engine = self.engine
        body = {
            "model": model_id,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
            "stream": True,
        }
        try:
            with self._client.stream(
                "POST",
                engine.base_url + "/chat/completions",
                json=body,
                headers=self._headers(),
                timeout=self._timeout(LOAD_TIMEOUT_S),
            ) as resp:
                if not resp.is_success:
                    resp.read()
                    raise _status_error(
                        resp.status_code, _parse_body(resp), url=str(resp.request.url)
                    )
                for obj in _iter_sse_json(resp.iter_lines()):
                    status = obj.get("mimoe_status")
                    if isinstance(status, Mapping):
                        notify(str(status.get("message") or status.get("stage") or "loading"))
                    elif _is_error_body(obj):
                        raise _status_error(_body_status(obj) or 500, obj, url=str(resp.url))
        except httpx.TimeoutException as exc:
            raise MimoeError(
                f"loading {model_id} took longer than {LOAD_TIMEOUT_S:.0f} s", hint=HINT_TIMEOUT
            ) from exc
        except httpx.TransportError as exc:
            raise MimoeError("mimOE Studio is not reachable", hint=HINT_NOT_REACHABLE) from exc


def preflight(
    settings: Settings,
    *,
    client: MimoeClient | None = None,
    on_status: Callable[[str], None] | None = None,
) -> Preflight:
    """Discover the engine, pick the model, decide thinking control and whether tools are on.

    Order: discover -> loaded models (none -> "no model is loaded") -> ``settings.model`` or the
    first ``kind == "llm"`` -> thinking control -> advertised tool support, else the probe (skipped
    with ``force_tools``). The probe validates the API key; when no probe runs, one registry
    ``GET`` does instead (``GET /models`` on the OpenAI path is public on both generations, the
    1.0 registry answers 403 to a wrong key). Every failure is a :class:`MimoeError` whose hint
    names a Studio click-path.
    """
    notify = on_status or (lambda _text: None)
    client = client or MimoeClient(settings.base_url, settings.api_key)
    notify("connecting to mimOE Studio")
    engine = client.discover()
    models = client.loaded_models()
    local = [m for m in models if not is_remote(m)]
    if not local and not settings.model:
        remote = ", ".join(m.id for m in models)
        detail = f" (only remote models are listed: {remote})" if remote else ""
        raise MimoeError(f"no model is loaded{detail}", hint=HINT_NO_MODEL)
    model = _pick_model(models if settings.model else local, settings.model)
    warnings: list[str] = []
    if is_remote(model):
        warnings.append(
            f"{model.id} is served by a remote provider, not by mimOE on this machine: your "
            "prompts and workspace contents leave this computer"
        )
    if engine.version is None:
        warnings.append("node info unavailable (JSON-RPC getMe failed); engine version unknown")

    # 1.0-generation engines accept ``enable_thinking`` for every model, while their
    # ``reasoning.can_disable`` flag is always false (see docs/compatibility.md), so the
    # generation decides; 0.6 engines only understand the ``/no_think`` soft switch.
    if engine.generation is EngineGeneration.V10 or model.thinking_can_disable:
        thinking_control: ThinkingControl = "native"
    elif model.thinking_supported is False:
        thinking_control = "none"
    else:
        thinking_control = "soft"
    if settings.think and thinking_control == "none":
        warnings.append(f"{model.id} does not support thinking; --think has no effect")

    probe: ProbeResult | None = None
    if settings.force_tools:
        tools_enabled = True
        warnings.append("--force-tools: the tool probe was skipped; tool calls may fail")
    elif model.supports_tools is not None:
        tools_enabled = model.supports_tools
        if not tools_enabled:
            warnings.append(f"{model.id} advertises no tool support: chat-only mode")
    else:
        notify(f"warming up {model.id} (tool probe)")
        probe = client.probe_tools(model.id, thinking_control=thinking_control)
        tools_enabled = probe.tools_ok
        if not tools_enabled:
            warnings.append(f"tool probe failed: {probe.detail}. {HINT_PROBE_FAILED}")
    if probe is None:
        warnings.extend(_check_api_key(client))
    return Preflight(
        engine=engine,
        model=model,
        probe=probe,
        tools_enabled=tools_enabled,
        thinking_control=thinking_control,
        warnings=tuple(warnings),
    )


def friendly_error(exc: BaseException) -> tuple[str, str]:
    """Map any exception to ``(message, hint)`` for the CLI, the server and the UI.

    Handles :class:`MimoeError`/:class:`ConfigError`, ``httpx`` transport errors, the openai SDK
    classes (``ChatOpenAI`` raises langchain-openai subclasses of them) and, by class name, the
    langchain-core ``Model*Error`` wrappers.
    """
    if isinstance(exc, MimoeError | ConfigError):
        return exc.message, exc.hint
    names = {cls.__name__ for cls in type(exc).__mro__}
    if "TurnCancelled" in names:
        return "turn cancelled", "Send a new message when you are ready."
    if isinstance(exc, httpx.TimeoutException | openai.APITimeoutError) or (
        "ModelTimeoutError" in names
    ):
        return "mimOE did not answer in time", HINT_TIMEOUT
    if isinstance(exc, httpx.TransportError | openai.APIConnectionError) or (
        "ModelConnectionError" in names
    ):
        return "mimOE Studio is not reachable", HINT_NOT_REACHABLE
    status, body = _status_of(exc)
    text = _body_text(body)
    server_message = _error_message(body) or ""
    if status in (401, 403) or names & {"ModelPermissionDeniedError", "ModelAuthenticationError"}:
        return "mimOE rejected the API key", HINT_API_KEY
    if status == 404 or "ModelNotFoundError" in names:
        detail = f" ({server_message})" if server_message else ""
        return (
            f"mimOE could not find the requested model{detail}",
            "Studio > Models > Load the model, or pass --model with a loaded id.",
        )
    if "ContextOverflowError" in names or "llama_decode" in text:
        return "the conversation exceeded the model's context window (llama_decode failed)", (
            HINT_CONTEXT
        )
    if status == 503 and not text.strip():
        return "mimOE answered 503 with an empty body: the base URL path is wrong", HINT_WRONG_PATH
    if status == 400 and "not ready" in text.lower():
        return f"the model is not ready ({server_message})", HINT_MODEL_NOT_READY
    if (status is not None and status >= 500) or "ModelAPIError" in names:
        detail = f" ({server_message})" if server_message else ""
        return f"mimOE returned a server error{detail}", HINT_SERVER_ERROR
    if status is not None:
        detail = f": {server_message}" if server_message else ""
        return f"mimOE returned HTTP {status}{detail}", HINT_SERVER_ERROR
    if isinstance(exc, openai.OpenAIError) or "ModelError" in names:
        return f"model error: {exc}", HINT_SERVER_ERROR
    return f"{type(exc).__name__}: {exc}", "Retry; if it persists, report the message above."


# -- helpers ------------------------------------------------------------------------------------


def _check_api_key(client: MimoeClient) -> list[str]:
    """Validate the key with a registry ``GET`` when no probe ran; unrelated failures only warn."""
    try:
        client.registry_models()
    except MimoeError as exc:
        if exc.message == "mimOE rejected the API key":
            raise
        return [f"could not verify the API key against the model registry: {exc.message}"]
    return []


def _pick_model(models: list[LoadedModel], requested: str | None) -> LoadedModel:
    ids = ", ".join(m.id for m in models)
    if requested:
        wanted = strip_node_prefix(requested)
        for model in models:
            if model.id == wanted:
                return model
        raise MimoeError(
            f"model '{wanted}' is not loaded (loaded: {ids})",
            hint=f"Studio > Models > Load {wanted}, or pass --model with one of the loaded ids.",
        )
    for model in models:
        if model.kind == "llm":
            return model
    raise MimoeError(
        f"no chat model (kind llm) is loaded (loaded: {ids})",
        hint=HINT_NO_MODEL,
    )


def _model_entries(resp: httpx.Response) -> list[Mapping[str, Any]] | None:
    """Return the ``data`` list of a ``/models`` reply, or ``None`` when it is not one."""
    try:
        data = resp.json()
    except ValueError:
        return None
    if not isinstance(data, Mapping) or not isinstance(data.get("data"), list):
        return None
    return [entry for entry in data["data"] if isinstance(entry, Mapping)]


def _parse_body(resp: httpx.Response) -> object:
    try:
        return resp.json()
    except ValueError:
        return resp.text


def _is_error_body(obj: Mapping[str, Any]) -> bool:
    return "statusCode" in obj or "error" in obj


def _route_missing(status: int, body: object) -> bool:
    """True when a status/body pair means "this engine has no such route" rather than a failure.

    Verified: a 1.0 router miss on the OpenAI path is ``404 {"statusCode":404,"message":"not
    found"}`` and an unknown container path is an empty 503; a store that only knows
    ``action: "update"`` is assumed to reject other actions with a 400 naming the action.
    """
    message = (_error_message(body) or "").strip().lower()
    if status in (404, 405, 503) and message in ("", "not found"):
        return True
    return status == 400 and "action" in message


def _body_status(obj: Mapping[str, Any]) -> int | None:
    status = obj.get("statusCode")
    if status is None and isinstance(obj.get("error"), Mapping):
        status = obj["error"].get("code")
    return status if isinstance(status, int) and not isinstance(status, bool) else None


def _error_message(body: object) -> str | None:
    """Extract the message from mimOE's error shapes: ``{message,statusCode}``,
    ``{error:{code,message}}``, ``{error:{message,type}}`` or plain text."""
    if isinstance(body, Mapping):
        err = body.get("error")
        if isinstance(err, Mapping):
            message = err.get("message")
            return str(message) if message else None
        if isinstance(err, str) and err:
            return err
        message = body.get("message")
        return str(message) if message else None
    if isinstance(body, str) and body.strip():
        return body.strip()[:300]
    return None


def _body_text(body: object) -> str:
    if body is None:
        return ""
    if isinstance(body, str):
        return body
    try:
        return json.dumps(body)
    except (TypeError, ValueError):
        return str(body)


def _status_error(status: int, body: object, *, url: str) -> MimoeError:
    """Turn an HTTP error status plus body into a :class:`MimoeError` with a Studio hint."""
    message = _error_message(body) or ""
    text = _body_text(body)
    if status in (401, 403):
        return MimoeError("mimOE rejected the API key", hint=HINT_API_KEY)
    if status == 404:
        detail = f": {message}" if message else ""
        return MimoeError(
            f"mimOE could not find the requested model{detail}",
            hint="Studio > Models > Load the model, or pass --model with a loaded id.",
        )
    if status == 400 and "not ready" in message.lower():
        return MimoeError(f"the model is not ready ({message})", hint=HINT_MODEL_NOT_READY)
    if status >= 500 and "llama_decode" in text:
        return MimoeError(
            "the conversation exceeded the model's context window (llama_decode failed)",
            hint=HINT_CONTEXT,
        )
    if status == 503 and not text.strip():
        return MimoeError(
            f"mimOE answered 503 with an empty body for {url}: the base URL path is wrong",
            hint=HINT_WRONG_PATH,
        )
    detail = f": {message}" if message else ""
    return MimoeError(f"mimOE returned HTTP {status}{detail} ({url})", hint=HINT_SERVER_ERROR)


def _status_of(exc: BaseException) -> tuple[int | None, object]:
    """Return ``(status_code, body)`` for SDK/httpx status errors, ``(None, None)`` otherwise."""
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code, exc.body
    if isinstance(exc, openai.APIError):  # an error event inside a stream (1.0 engines)
        body = exc.body
        return (_body_status(body) if isinstance(body, Mapping) else None), body
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code, _parse_body(exc.response)
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and not isinstance(status, bool):
        return status, getattr(exc, "body", None)
    return None, None


def _iter_sse_json(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Yield the JSON objects of an SSE stream (``data:`` lines, or bare JSON lines as milm 1.14
    emits for the final load event); ``[DONE]``, comments and non-JSON lines are skipped."""
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("data:"):
            line = line[5:].strip()
        if not line or line == "[DONE]" or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            yield obj


def _progress_text(progress: str) -> str:
    """``"<|loading_model|> 25%<br />"`` -> ``"loading model 25%"``."""
    text = re.sub(r"<br\s*/?>", "", progress)
    text = re.sub(r"<\|(\w+)\|>", lambda m: m.group(1).replace("_", " "), text)
    return " ".join(text.split())


def _opt_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return None


def _opt_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _opt_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _opt_str(value: object) -> str | None:
    return str(value) if isinstance(value, str) and value.strip() else None
