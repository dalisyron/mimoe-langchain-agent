"""Model registry helpers used by the CLI and the server: presets, pulls and model switching.

Pull paths, verified against the store sources shipped with each Studio (mmodelstore 1.5.5 in
0.6.5, 1.11.7 in 1.0.27) and the live download log of the six presets:

* **1.0**: ``POST {store}/hf/models/pull {"repo", "quant", "name"}`` streams SSE stages:
  ``resolving`` -> ``quant_required{available}`` (no quant given and several files; the stream
  ends) | ``resolved{modelFile, modelId}`` -> ``already_exists`` (ends) | ``{size, totalSize}``
  progress lines -> end of stream. Once the first stage is out, every failure arrives as
  ``{"stage": "error", "message"}``; a malformed body answers 400 JSON. ``name`` sets the registry
  id (the handler reads only ``repo``, ``quant``, ``name`` and ``mmproj``); without it the store
  uses the file name minus ``.gguf``.
* **0.6**: ``POST {store}/models {id, version, kind, gguf}`` registers the metadata (201 new, 200
  existing), then ``POST {store}/models/{id}/download {"url"}`` streams ``{size, totalSize}``
  lines and ends when the file is complete: both store versions mark the entry ``readyToUse``
  before they end the response and never write a ``done`` event (verified in the download proxy
  of both bundles and in the live download log). A failure after the headers went out arrives as
  one bare JSON error line (the route's ``sendError``). The 0.6 store has no HuggingFace
  resolver: for a custom ``owner/repo:QUANT`` spec the file name is derived from the repo name
  (``<repo minus -GGUF>-<QUANT>.gguf``, true for every preset) unless the spec names the file.

Neither store reacts to a client disconnect, so a stream is read to its end with no read timeout
(a pull can take many minutes; Ctrl-C is the way out) and an interrupted pull leaves the entry
not ready; running the pull again re-registers (upsert) and downloads again. Both stores validate
ids against ``^[a-zA-Z0-9._-]+$`` with no ``..`` and no leading ``.`` or ``-``;
:func:`parse_spec` enforces the same rule before anything is sent. Everything raises
:class:`MimoeError`; exceptions raised by a progress callback propagate unchanged and end the
stream where they happened.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from mimoe_agent.mimoe import (
    GET_TIMEOUT_S,
    HINT_NOT_REACHABLE,
    HINT_TIMEOUT,
    EngineGeneration,
    LoadedModel,
    MimoeClient,
    MimoeError,
    RegistryModel,
    _body_status,
    _error_message,
    _is_error_body,
    _parse_body,
    _status_error,
    strip_node_prefix,
)


@dataclass(frozen=True)
class Preset:
    """A measured model: where its GGUF lives and what the compatibility matrix found."""

    id: str
    repo: str
    file: str
    size_gb: float
    note: str


PRESETS: tuple[Preset, ...] = (
    Preset(
        "qwen3-4b-instruct-2507",
        "unsloth/Qwen3-4B-Instruct-2507-GGUF",
        "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        2.50,
        "recommended default: loads on 0.6 and 1.0 engines, 6/6 structured tool calls, clean "
        "round trip, no thinking mode to manage",
    ),
    Preset(
        "qwen3-8b",
        "Qwen/Qwen3-8B-GGUF",
        "Qwen3-8B-Q4_K_M.gguf",
        5.03,
        "both engines, 6/6 structured tool calls; thinking model (keep it off for tools); 5 GB "
        "and about 40% slower per token than the 4B",
    ),
    Preset(
        "smollm3-3b",
        "bartowski/HuggingFaceTB_SmolLM3-3B-GGUF",
        "HuggingFaceTB_SmolLM3-3B-Q4_K_M.gguf",
        1.92,
        "both engines; tool calls are structured only on 1.0 (plain text on 0.6) and the round "
        "trip breaks either way; thinking leaks through /no_think on 0.6",
    ),
    Preset(
        "smollm2-360m",
        "lmstudio-community/SmolLM2-360M-Instruct-GGUF",
        "SmolLM2-360M-Instruct-Q8_0.gguf",
        0.39,
        "both engines; chat only (no tool calling on either engine), no thinking; fastest "
        "(135-150 tok/s)",
    ),
    Preset(
        "qwen3.5-4b",
        "unsloth/Qwen3.5-4B-GGUF",
        "Qwen3.5-4B-Q4_K_M.gguf",
        2.74,
        "1.0 engines only (fails to load on 0.6.5); 6/6 structured tool calls; brief thinking "
        "(~60 tokens) when enabled",
    ),
    Preset(
        "qwen3.5-9b",
        "unsloth/Qwen3.5-9B-GGUF",
        "Qwen3.5-9B-Q4_K_M.gguf",
        5.68,
        "1.0 engines only; 6/6 structured tool calls; brief thinking when enabled; 5.7 GB, "
        "needs a 16 GB machine with nothing else open",
    ),
)
DEFAULT_PRESET = "qwen3-4b-instruct-2507"
V10_ONLY_PRESETS = frozenset({"qwen3.5-4b", "qwen3.5-9b"})
"""Presets whose architecture (qwen35) the 0.6.5 engine cannot load (matrix: HTTP 500)."""

HF_BASE_URL = "https://huggingface.co"
REGISTER_VERSION = "1.0.0"
REGISTER_GGUF = {"initContextSize": 12000, "initGpuLayerSize": 99}
PULL_TIMEOUT = httpx.Timeout(None, connect=GET_TIMEOUT_S)
"""A pull stream has a connect timeout only: the transfer takes as long as it takes and neither
store handles a client disconnect, so cutting a slow stream would only misreport it."""
MAX_ID_LENGTH = 255
WARN_FILE_GB = 4.5
OVERHEAD_GB_AT_12K = 1.5
"""KV cache plus compute buffers observed for a 4-9B model at a 12k context on Apple Silicon."""
REFERENCE_CONTEXT = 12000

HINT_SPEC = (
    "use a preset id ("
    + ", ".join(p.id for p in PRESETS)
    + ") or owner/repo:QUANT from HuggingFace, for example Qwen/Qwen3-8B-GGUF:Q4_K_M"
)
HINT_PULL_FAILED = (
    "Check the internet connection and the repo/quant spelling; Studio > Models shows the "
    "registry entry, delete it there if the download must start over."
)
HINT_NO_PULL_ROUTE = (
    "This engine's model store has no HuggingFace pull endpoint; register and download the "
    "model in Studio > Models instead."
)

_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
"""Both stores' ``validateModelId``; they also refuse ``..`` and a leading ``.`` or ``-``."""
_REPO_PART_RE = re.compile(r"^[A-Za-z0-9_](?:[A-Za-z0-9._-]*[A-Za-z0-9_])?$")
"""HuggingFace's own rule (``validate_repo_id``): each part starts and ends with a word
character; ``..`` and ``--`` are refused separately."""
_QUANT_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_QUANT_FILE_RE = re.compile(
    r"-((?:UD-)?(?:Q[0-9]+_K_[MSXL]+|Q[0-9]+_[0-9]+|IQ[0-9]+_[A-Z]+|(?:BF|F)(?:16|32)"
    r"|MXFP4[A-Z_]*))\.gguf$",
    re.IGNORECASE,
)
"""The 1.0 store's ``QUANT_REGEX``: the quantization suffix of a GGUF file name."""
_GGUF_SUFFIX_RE = re.compile(r"[-_.]?gguf$", re.IGNORECASE)


@dataclass(frozen=True)
class PullSpec:
    """A parsed pull spec: the registry id plus where the file comes from."""

    model_id: str
    repo: str
    quant: str | None
    """Quantization name as the 1.0 store expects it (``Q4_K_M``); ``None`` lets it auto-pick."""
    file: str | None
    """GGUF file name inside the repo (needed on 0.6); ``None`` when no quant was given."""
    preset: Preset | None


def preset_by_id(model_id: str) -> Preset | None:
    """Return the preset with this id (case-insensitive), or ``None``."""
    wanted = model_id.strip().lower()
    for preset in PRESETS:
        if preset.id == wanted:
            return preset
    return None


def quant_of(file: str) -> str | None:
    """Return the quantization in a GGUF file name (``...-Q4_K_M.gguf`` -> ``Q4_K_M``)."""
    match = _QUANT_FILE_RE.search(file)
    return match.group(1).upper() if match else None


def download_url(repo: str, file: str) -> str:
    """The HuggingFace URL the stores download from (``buildDownloadUrl`` in the 1.0 store)."""
    return f"{HF_BASE_URL}/{repo}/resolve/main/{file}"


def parse_spec(spec: str) -> PullSpec:
    """Parse a preset id, ``owner/repo``, ``owner/repo:QUANT`` or ``owner/repo:File.gguf``.

    A custom spec that names a preset's repo and quantization maps to that preset, so the file is
    never registered twice under different ids.

    Raises:
        MimoeError: when the spec is neither a preset nor a HuggingFace repo reference.
    """
    text = spec.strip()
    if not text:
        raise MimoeError("no model given", hint=HINT_SPEC)
    preset = preset_by_id(text)
    if preset is not None:
        return _spec_from_preset(preset)
    repo, _, token = text.partition(":")
    owner, slash, name = repo.partition("/")
    if (
        not slash
        or not _REPO_PART_RE.match(owner)
        or not _REPO_PART_RE.match(name)
        or ".." in repo
        or "--" in repo
        or (token and not _QUANT_TOKEN_RE.match(token.removesuffix(".gguf").removesuffix(".GGUF")))
        or ":" in token
    ):
        raise MimoeError(
            f"'{text}' is neither a preset nor an owner/repo[:QUANT] reference", hint=HINT_SPEC
        )
    if not token and ":" in text:
        raise MimoeError(f"'{text}' has an empty quantization after the colon", hint=HINT_SPEC)
    stem = _GGUF_SUFFIX_RE.sub("", name)
    if token.lower().endswith(".gguf"):
        file: str | None = token
        quant = quant_of(token)
    elif token:
        quant = token.upper()
        file = f"{stem}-{quant}.gguf"
    else:
        quant = None
        file = None
    for candidate in PRESETS:
        if candidate.repo.lower() == repo.lower() and (
            (file is not None and candidate.file.lower() == file.lower())
            or (quant is not None and quant_of(candidate.file) == _normalize_quant(quant))
        ):
            return _spec_from_preset(candidate)
    model_id = stem.lower() + (f"-{quant.lower()}" if quant else "")
    if not valid_model_id(model_id):
        raise MimoeError(
            f"cannot derive a registry id from '{text}' (got {model_id!r}; ids use letters, "
            "digits, '.', '_' and '-', never start with '.' or '-' and never contain '..')",
            hint=HINT_SPEC,
        )
    return PullSpec(model_id=model_id, repo=repo, quant=quant, file=file, preset=None)


def valid_model_id(model_id: str) -> bool:
    """True when both stores' ``validateModelId`` accepts ``model_id`` (so it is also a safe
    single URL path segment: no ``/``, no ``..``, no leading ``.`` or ``-``)."""
    return (
        0 < len(model_id) <= MAX_ID_LENGTH
        and _ID_RE.match(model_id) is not None
        and not model_id.startswith((".", "-"))
        and ".." not in model_id
    )


def memory_note(size_bytes: int, max_context: int = REFERENCE_CONTEXT) -> str | None:
    """Warn when a model file is too big for a 16 GB machine once the context cache is added.

    A simple estimate, no psutil: weights (the file) plus about 1.5 GB of KV cache and compute
    buffers at a 12k context, scaled with ``max_context``. Files up to 4.5 GB pass silently.

    Returns:
        A one-line warning, or ``None`` when the model is comfortably small.
    """
    if size_bytes <= 0:
        return None
    weights_gb = size_bytes / 1e9
    if weights_gb <= WARN_FILE_GB:
        return None
    overhead_gb = OVERHEAD_GB_AT_12K * max(max_context, 0) / REFERENCE_CONTEXT
    total_gb = weights_gb + overhead_gb
    verdict = "too much for a 16 GB machine" if total_gb > 12 else "tight on a 16 GB machine"
    return f"about {total_gb:.0f} GB at {max_context / 1000:.0f}k context; {verdict}"


def pull_model(
    client: MimoeClient, spec: str, *, on_progress: Callable[[str, float | None], None]
) -> RegistryModel:
    """Download ``spec`` into the engine's model registry and return its registry entry.

    Args:
        client: Engine client (its generation picks the pull path, see the module docstring).
        spec: Preset id or ``owner/repo[:QUANT]`` (``owner/repo:File.gguf`` names the file).
        on_progress: Receives ``(text, fraction)``; ``fraction`` is ``None`` outside the download.

    Raises:
        MimoeError: bad spec, unknown repo or quantization (the message lists the available
            ones), a store error, or a stream that ended without the model becoming ready.
    """
    parsed = parse_spec(spec)
    engine = client.engine
    existing = _registry_entry(client, parsed.model_id)
    if existing is not None and existing.ready:
        on_progress(f"{parsed.model_id} is already in the registry", 1.0)
        return existing
    if (
        parsed.preset is not None
        and parsed.preset.id in V10_ONLY_PRESETS
        and engine.generation is not EngineGeneration.V10
    ):
        on_progress(
            f"warning: {parsed.preset.id} does not load on 0.6-generation engines "
            f"({parsed.preset.note})",
            None,
        )
    if engine.generation is EngineGeneration.V10:
        _pull_v10(client, parsed, on_progress)
    else:
        _pull_v06(client, parsed, on_progress)
    entry = _registry_entry(client, parsed.model_id)
    if entry is None or not entry.ready:
        status = _registry_status(entry)
        raise MimoeError(
            f"the pull of {parsed.model_id} ended but the model is not ready{status}",
            hint=HINT_PULL_FAILED,
        )
    on_progress(f"{parsed.model_id} is ready", 1.0)
    return entry


def switch_model(
    client: MimoeClient,
    model_id: str,
    *,
    unload_previous: bool,
    on_status: Callable[[str], None],
) -> LoadedModel:
    """Make ``model_id`` the loaded chat model and return its ``GET /models`` entry.

    The id must be loaded already or registered with its file present. With ``unload_previous``
    every other loaded ``llm`` model is unloaded first; an unload failure is reported through
    ``on_status`` and does not stop the switch. Load progress lines go to ``on_status`` too.

    Raises:
        MimoeError: unknown or not-yet-downloaded id (the hint says to pull it), or a load error
            (its hint names the models that were already unloaded).
    """
    model_id = strip_node_prefix(model_id)
    loaded = client.loaded_models()
    current = next((m for m in loaded if m.id == model_id), None)
    if current is None:
        entry = _registry_entry(client, model_id)
        if entry is None:
            registered = ", ".join(sorted(m.id for m in client.registry_models())) or "none"
            raise MimoeError(
                f"'{model_id}' is not in the model registry (registered: {registered})",
                hint=f"pull it first: `mimoe-agent models pull {model_id}` for a preset, "
                "or `mimoe-agent models pull owner/repo:QUANT` for another HuggingFace GGUF; "
                f"presets: {', '.join(p.id for p in PRESETS)}",
            )
        if not entry.ready:
            raise MimoeError(
                f"'{model_id}' is registered but its file is not downloaded"
                f"{_registry_status(entry)}",
                hint=f"finish the download with `mimoe-agent models pull {model_id}` "
                "or Studio > Models > Download",
            )
        note = memory_note(_size_hint(entry), _context_hint(entry))
        if note:
            on_status(f"warning: {model_id} is {note}")
    unloaded: list[str] = []
    if unload_previous:
        for other in loaded:
            if other.kind != "llm" or other.id == model_id:
                continue
            on_status(f"unloading {other.id}")
            try:
                client.unload_model(other.id)
            except MimoeError as exc:
                on_status(f"could not unload {other.id}: {exc.message}")
            else:
                unloaded.append(other.id)
    if current is not None:
        on_status(f"{model_id} is already loaded")
        return next((m for m in client.loaded_models() if m.id == model_id), current)
    on_status(f"loading {model_id}")
    try:
        return client.load_model(model_id, on_progress=on_status)
    except MimoeError as exc:
        if not unloaded:
            raise
        raise MimoeError(
            exc.message,
            hint=f"{exc.hint} Note: {', '.join(unloaded)} was unloaded before the failure; "
            f"`mimoe-agent models use {unloaded[0]}` brings it back.".strip(),
        ) from exc


# -- pull paths ----------------------------------------------------------------------------------


def _pull_v10(
    client: MimoeClient, parsed: PullSpec, on_progress: Callable[[str, float | None], None]
) -> None:
    """``POST {store}/hf/models/pull``: the store resolves the file on HuggingFace itself."""
    if parsed.quant is None and parsed.file is not None:
        raise MimoeError(
            f"cannot read a quantization from '{parsed.file}'",
            hint=f"pass {parsed.repo}:QUANT instead, for example {parsed.repo}:Q4_K_M",
        )
    body: dict[str, Any] = {"repo": parsed.repo, "name": parsed.model_id}
    if parsed.quant is not None:
        body["quant"] = parsed.quant
    url = client.engine.store_url + "/hf/models/pull"
    for obj in _sse_objects(client, url, body, missing_hint=HINT_NO_PULL_ROUTE):
        stage = obj.get("stage")
        if stage == "resolving":
            on_progress(f"resolving {parsed.repo} ({parsed.quant or 'auto'})", None)
        elif stage == "quant_required":
            names = [
                str(item.get("name") or item.get("file") or "?")
                for item in obj.get("available") or []
                if isinstance(item, Mapping)
            ]
            raise MimoeError(
                f"{parsed.repo} needs a quantization (available: {', '.join(names) or 'none'})",
                hint=f"run again with {parsed.repo}:{names[0] if names else 'QUANT'}",
            )
        elif stage == "resolved":
            on_progress(
                f"resolved {obj.get('modelFile') or parsed.file or '?'} "
                f"as {obj.get('modelId') or parsed.model_id}",
                None,
            )
        elif stage == "already_exists":
            on_progress(f"{obj.get('modelId') or parsed.model_id} is already in the registry", 1.0)
        elif stage == "error":
            message = str(obj.get("message") or "unknown error")
            raise MimoeError(
                f"pull of {parsed.repo} failed: {message}",
                hint=_pull_error_hint(message, parsed.repo),
            )
        elif "size" in obj or "totalSize" in obj:
            _report_download(obj, on_progress)
        elif _is_error_body(obj):
            raise _status_error(_body_status(obj) or 500, obj, url=url)


def _pull_v06(
    client: MimoeClient, parsed: PullSpec, on_progress: Callable[[str, float | None], None]
) -> None:
    """Register the metadata, then stream ``POST {store}/models/{id}/download {"url"}``."""
    if parsed.file is None:
        raise MimoeError(
            f"0.6-generation engines need the quantization of {parsed.repo}",
            hint=f"run again with {parsed.repo}:QUANT, for example {parsed.repo}:Q4_K_M",
        )
    store = client.engine.store_url
    body = {
        "id": parsed.model_id,
        "version": REGISTER_VERSION,
        "kind": "llm",
        "gguf": dict(REGISTER_GGUF),
    }
    client._request("POST", store + "/models", json=body, timeout=GET_TIMEOUT_S)
    on_progress(f"registered {parsed.model_id}", None)
    url = f"{store}/models/{parsed.model_id}/download"
    source = download_url(parsed.repo, parsed.file)
    on_progress(f"downloading {source}", None)
    for obj in _sse_objects(client, url, {"url": source}, missing_hint=HINT_PULL_FAILED):
        if obj.get("done") is True:
            break
        if _is_error_body(obj):
            raise MimoeError(
                f"download of {parsed.model_id} failed: {_error_message(obj) or 'unknown error'}",
                hint=HINT_PULL_FAILED,
            )
        if "size" in obj or "totalSize" in obj:
            _report_download(obj, on_progress)


def _sse_objects(
    client: MimoeClient, url: str, body: Mapping[str, Any], *, missing_hint: str
) -> Iterator[dict[str, Any]]:
    """Stream one store POST and yield its JSON objects, keeping the connection open until the
    server ends the stream (or the consumer stops iterating); connect timeout only."""
    http = client._client
    try:
        with http.stream(
            "POST",
            url,
            json=dict(body),
            headers={**client._headers(), "Accept": "text/event-stream"},
            timeout=PULL_TIMEOUT,
        ) as resp:
            if not resp.is_success:
                resp.read()
                parsed = _parse_body(resp)
                if resp.status_code in (404, 405, 503):
                    detail = _error_message(parsed) or f"HTTP {resp.status_code}"
                    raise MimoeError(f"{url} answered {detail}", hint=missing_hint)
                raise _status_error(resp.status_code, parsed, url=url)
            yield from _iter_sse_objects(resp.iter_lines())
    except httpx.RemoteProtocolError as exc:
        raise MimoeError(
            f"the model store closed the stream before the pull finished ({url})",
            hint=HINT_PULL_FAILED,
        ) from exc
    except httpx.TimeoutException as exc:
        raise MimoeError(
            f"mimOE did not accept the connection within {GET_TIMEOUT_S:.0f} s ({url})",
            hint=HINT_TIMEOUT,
        ) from exc
    except httpx.TransportError as exc:
        raise MimoeError("mimOE Studio is not reachable", hint=HINT_NOT_REACHABLE) from exc


def _iter_sse_objects(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Yield the JSON objects of a store stream from its already split lines (httpx splits on
    LF, CRLF and CR).

    Handles SSE events whose ``data:`` spans several lines (joined with a newline, as the spec
    says), ignores ``event:``/``id:``/``retry:`` fields and comments, and also accepts the bare
    JSON line a route writes when a download fails after the headers went out. ``[DONE]`` and
    payloads that are not JSON objects are skipped.
    """
    data: list[str] = []
    for line in lines:
        if not line.strip():
            if data:
                yield from _json_object("\n".join(data))
                data = []
            continue
        if line.startswith(":"):
            continue
        field, sep, value = line.partition(":")
        if sep and field in ("data", "event", "id", "retry"):
            if field == "data":
                data.append(value.removeprefix(" "))
            continue
        yield from _json_object(line)
    if data:
        yield from _json_object("\n".join(data))


def _json_object(text: str) -> Iterator[dict[str, Any]]:
    text = text.strip()
    if not text.startswith("{"):
        return
    try:
        obj = json.loads(text)
    except ValueError:
        return
    if isinstance(obj, dict):
        yield obj


def _pull_error_hint(message: str, repo: str) -> str:
    """A hint for the 1.0 store's ``error`` stage (messages verified live on 1.0.27)."""
    _, sep, available = message.partition("Available: ")
    if sep:
        first = available.split(",")[0].strip().rstrip(".") or "QUANT"
        return f"run again with {repo}:{first}"
    if "HuggingFace API request failed" in message:
        return (
            f"check the spelling of {repo} on huggingface.co (a missing repo answers 401 there); "
            "gated or private repos cannot be pulled"
        )
    return HINT_PULL_FAILED


def _report_download(
    obj: Mapping[str, Any], on_progress: Callable[[str, float | None], None]
) -> None:
    size = _as_int(obj.get("size"))
    total = _as_int(obj.get("totalSize"))
    fraction = min(size / total, 1.0) if size is not None and total else None
    on_progress(f"downloading {_gb(size)} / {_gb(total)} GB", fraction)


# -- helpers -------------------------------------------------------------------------------------


def _spec_from_preset(preset: Preset) -> PullSpec:
    return PullSpec(
        model_id=preset.id,
        repo=preset.repo,
        quant=quant_of(preset.file),
        file=preset.file,
        preset=preset,
    )


def _normalize_quant(quant: str) -> str:
    """The 1.0 store compares quantizations upper-cased with ``-`` folded to ``_``."""
    return quant.upper().replace("-", "_")


def _registry_entry(client: MimoeClient, model_id: str) -> RegistryModel | None:
    for entry in client.registry_models():
        if entry.id == model_id:
            return entry
    return None


def _registry_status(entry: RegistryModel | None) -> str:
    if entry is None:
        return " (not registered)"
    status = entry.raw.get("status")
    message = entry.raw.get("statusMessage")
    if status:
        return f" (status {status}{f': {message}' if message else ''})"
    return ""


def _size_hint(entry: RegistryModel) -> int:
    if entry.size_bytes:
        return entry.size_bytes
    preset = preset_by_id(entry.id)
    return int(preset.size_gb * 1e9) if preset else 0


def _context_hint(entry: RegistryModel) -> int:
    """The context the entry loads with (``gguf.initContextSize``: 12000 for our registrations,
    32768 for a 1.0 HuggingFace pull), else the reference 12k."""
    gguf = entry.raw.get("gguf")
    context = _as_int(gguf.get("initContextSize")) if isinstance(gguf, Mapping) else None
    return context if context and context > 0 else REFERENCE_CONTEXT


def _as_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return None


def _gb(value: int | None) -> str:
    return "?" if value is None else f"{value / 1e9:.2f}"
