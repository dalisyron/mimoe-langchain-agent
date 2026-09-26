# Component contracts

Signatures and data shapes that the modules agree on. Implementation agents follow this file;
change it deliberately (and update every caller) rather than drifting from it.

## Conventions
- Python 3.13, `from __future__ import annotations`, type hints everywhere, ruff clean
  (`uvx ruff check .`, `uvx ruff format .`). Google-style docstrings on public functions.
- Tools return `str`, never raise, never return `""`. Every tool result is capped at 8,000
  characters by `GuardrailMiddleware`; tools may cap earlier with a `[truncated ...]` tail.
- Cross-platform: `pathlib`, `sys.executable`, no shell strings, no POSIX-only APIs without a
  Windows branch. Encode subprocess I/O as UTF-8 with `errors="replace"`.
- Network clients use `httpx` with `trust_env=False` so proxies never capture localhost.
- Nothing in the package prints to stdout except `cli.py`; libraries raise or return.

## config.py
```python
@dataclass(frozen=True)
class Settings:
    base_url: str | None      # None = auto-discover (mimoe.discover_engine)
    api_key: str              # default "1234"
    model: str | None         # None = first loaded llm
    workspace: Path           # resolved absolute path; must exist and be a directory
    think: bool               # default False
    auto_approve: bool        # default False
    allow_network: bool       # default False (lifts run_python's socket guard only)
    force_tools: bool         # default False (skip the tool probe result)
    trace: bool               # default False (LangSmith env vars forced off unless True)
    sources: Mapping[str, str]  # field name -> "flag" | "env" | ".env" | "default"

def load_settings(overrides: Mapping[str, object] | None = None, *, cwd: Path | None = None) -> Settings
```
Precedence: overrides (CLI flags, only keys whose value is not None) > process env
(`MIMOE_BASE_URL`, `MIMOE_API_KEY`, `MIMOE_MODEL`, `MIMOE_WORKSPACE`, `MIMOE_THINK`,
`MIMOE_AUTO_APPROVE`, `MIMOE_ALLOW_NETWORK`, `MIMOE_FORCE_TOOLS`, `MIMOE_TRACE`) > `.env` in `cwd`
only (`load_dotenv(cwd / ".env", override=False)` and only the first four keys) > defaults.
Booleans parse `1/true/yes/on` (case-insensitive). Workspace default: `cwd / "workspace"` if it
is a directory, else raise `ConfigError("no workspace", hint="pass --workspace PATH")`.
`ConfigError(message, hint)` is the only exception type here.

## mimoe.py (engine client, sync httpx; no LangChain imports)
```python
class MimoeError(Exception):      # .message: str, .hint: str
class EngineGeneration(StrEnum): V06 = "0.6"; V10 = "1.0"; UNKNOWN = "unknown"

@dataclass(frozen=True)
class EngineInfo:
    base_url: str          # e.g. http://localhost:8083/mimik-ai/openai/v1 (no trailing slash)
    store_url: str         # e.g. http://localhost:8083/mimik-ai/store/v1
    rpc_url: str           # e.g. http://localhost:8083/jsonrpc/v1
    node_name: str | None
    version: str | None    # from getMe, e.g. "v3.22.8 (developer edition)"
    generation: EngineGeneration   # V10 when GET /models entries carry supported_parameters or version >= v3.30, else V06

@dataclass(frozen=True)
class LoadedModel:
    id: str; kind: str; family: str | None; max_context: int | None; n_params: int | None
    tokens_per_second: float | None; avg_tokens_per_second: float | None
    supports_tools: bool | None        # from supported_parameters when present, else None
    thinking_supported: bool | None    # reasoning.supported when present
    thinking_can_disable: bool | None  # reasoning.can_disable when present
    raw: Mapping[str, Any]

@dataclass(frozen=True)
class RegistryModel:
    id: str; kind: str; ready: bool; size_bytes: int | None; raw: Mapping[str, Any]

@dataclass(frozen=True)
class ProbeResult:
    tools_ok: bool; latency_s: float; detail: str    # detail = short human explanation

@dataclass(frozen=True)
class Preflight:
    engine: EngineInfo
    model: LoadedModel
    probe: ProbeResult | None            # None when force_tools or capabilities were advertised
    tools_enabled: bool
    thinking_control: Literal["native", "soft", "none"]   # native = enable_thinking param, soft = /no_think
    warnings: tuple[str, ...]

class MimoeClient:
    def __init__(self, base_url: str | None, api_key: str, *, timeout: float = 5.0, client: httpx.Client | None = None) -> None
    def discover(self) -> EngineInfo            # tries base_url, else candidates; raises MimoeError("mimOE Studio is not reachable", hint=...)
    def loaded_models(self) -> list[LoadedModel]
    def registry_models(self) -> list[RegistryModel]
    def node_info(self) -> dict[str, Any]       # JSON-RPC getMe result or {}
    def load_model(self, model_id: str, on_progress: Callable[[str], None] | None = None) -> LoadedModel
    def unload_model(self, model_id: str) -> None
    def probe_tools(self, model_id: str, *, thinking_control: str) -> ProbeResult   # one completion with a `ping` tool, max_tokens 48
    def chat(self, body: Mapping[str, Any]) -> dict[str, Any]   # raw POST /chat/completions (non-streaming), used by probe and tests

CANDIDATE_BASE_URLS = ("http://localhost:8083/mimik-ai/openai/v1", "http://localhost:8083/openai/v1")

def preflight(settings: Settings, *, client: MimoeClient | None = None, on_status: Callable[[str], None] | None = None) -> Preflight
def friendly_error(exc: BaseException) -> tuple[str, str]   # (message, hint) for any exception incl. langchain-openai errors
```
Preflight order: discover -> loaded_models (empty -> MimoeError "no model is loaded", hint "Studio >
Models > Load") -> pick `settings.model` or first `kind == "llm"` (strip a `<nodeId>/` prefix) ->
thinking_control = "native" if generation is V10 or model.thinking_can_disable (1.0 engines accept
enable_thinking for every model and report can_disable=false unreliably) else "none" if
thinking_supported is False else "soft" -> tools_enabled = supports_tools if it is not None
else probe (unless force_tools). The probe also validates the API key (403 -> MimoeError with hint).
Error bodies: mimOE returns `{"message","statusCode"}` or `{"error":{"code","message"}}`; both map
to MimoeError. Timeouts: 5 s for GETs, 60 s for the probe, 300 s for load_model.

## llm.py
```python
def make_model(settings: Settings, pre: Preflight, *, http_client: httpx.Client | None = None, http_async_client: httpx.AsyncClient | None = None) -> ChatOpenAI
```
Always: `base_url=pre.engine.base_url`, `api_key=settings.api_key`, `use_responses_api=False`,
`temperature=0`, `timeout=200`, `max_retries=0`, `stream_usage=True`, `model_kwargs={}`,
`extra_body={"max_tokens": 4096 if settings.think else 1024, **({"enable_thinking": settings.think} if pre.thinking_control == "native" else {})}`,
`http_client=http_client or httpx.Client(trust_env=False)`, same for async.

## middleware.py
```python
class QwenMiddleware(AgentMiddleware):      # __init__(soft_no_think: bool)
class GuardrailMiddleware(AgentMiddleware): # __init__(result_cap: int = 8000, old_result_cap: int = 400, thread_limit: int = 8)
THINK_RE: re.Pattern   # ^\s*<think>(.*?)</think>\s* (DOTALL) plus an unclosed-block variant
def split_think(text: str) -> tuple[str, str]   # (reasoning, visible)
```
QwenMiddleware (wrap_model_call + awrap_model_call): when `soft_no_think`, append " /no_think" to
the latest HumanMessage in the request (request only, never the checkpoint); on the response move
inline think text to `additional_kwargs["reasoning_content"]` (append if the server already set it)
and strip it from `content`; if `invalid_tool_calls` and no `tool_calls`, replace the message with
`AIMessage(content="I produced a malformed tool call; please rephrase.")` and clear
`invalid_tool_calls`. GuardrailMiddleware: `before_agent` resets `thread_model_call_count`;
`wrap_tool_call`/`awrap_tool_call` cap results, convert exceptions to `ToolMessage(status="error")`
and set `status="error"` on a returned result that `reports_failure` (it starts with `ERROR:`, or its
first line is `exit_code:` with a non-zero code or a `(killed:` note); the content is unchanged;
model hooks shorten ToolMessages older than the current turn to `old_result_cap` chars in the
request only. ComputeReminderMiddleware (wrap_model_call + awrap_model_call): when the request ends
with a HumanMessage whose text contains a digit, append `COMPUTE_REMINDER` (" (Use calculator or
run_python for any arithmetic.)") to it in the request only; it runs outside QwenMiddleware, so the
reminder comes before " /no_think".

## agent.py
```python
SYSTEM_PROMPT: str
def system_prompt(settings: Settings, pre: Preflight) -> str
def build_agent(settings: Settings, pre: Preflight, *, llm: BaseChatModel | None = None, tools: Sequence[BaseTool] | None = None, checkpointer: BaseCheckpointSaver | None = None)
```
Middleware order: `[ComputeReminderMiddleware() (only when the tools include calculator and run_python), QwenMiddleware(soft_no_think=pre.thinking_control == "soft" and not settings.think), GuardrailMiddleware(), ModelCallLimitMiddleware(thread_limit=8, exit_behavior="end"), HumanInTheLoopMiddleware(interrupt_on={"run_python": InterruptOnConfig(allowed_decisions=["approve", "reject"], description=...)})]`;
the HITL entry is omitted when `settings.auto_approve`. `tools=[]` when not `pre.tools_enabled`.
Checkpointer default `InMemorySaver()`.

## stream.py
```python
def iter_events(agent, payload, config=None, *, tool_ids: ToolCallIds | None = None) -> Iterator[dict]   # sync, agent.stream(stream_mode=["messages", "updates"])
async def aiter_events(agent, payload, config=None, *, tool_ids: ToolCallIds | None = None) -> AsyncIterator[dict]
class ToolCallIds: ...   # per-conversation-turn alias state: a repeated engine id ("tool_0") is emitted as "tool_0#2", "tool_0#3"; tool_result ids follow the alias
```
`payload` is `{"messages": [HumanMessage]}` or `Command(resume={"decisions": [...]})`. Events
(dicts with `"event"` plus fields), exactly:
`token{text}`, `thinking{text}`, `tool_call{id,name,args}`, `tool_result{id,name,content,is_error}`,
`approval_required{interrupt_id,action_requests:[{name,args,description}],review_configs}`,
`notice{text}`, `done{status:"completed"|"awaiting_approval",elapsed_s,model,usage_total:{input_tokens,output_tokens,llm_calls}}`,
`error{message,hint}`. Thinking arrives either as `reasoning_content` chunks or inline `<think>`
text; `ThinkSplitter` (stateful, per model call) turns both into `thinking` events.

## tools/
```python
def build_tools(settings: Settings, client: MimoeClient | None = None) -> list[BaseTool]
# workspace.py
class Workspace:  __init__(root: Path); resolve(rel: str) -> Path  (an absolute path under the root counts as the relative path it names; raises WorkspaceError on escape)
def make_workspace_tools(ws: Workspace) -> list[BaseTool]     # list_files, read_file, search_files
# run_python.py
def make_run_python(ws: Workspace, *, allow_network: bool, approval: bool = True, timeout_s: float = 30.0, memory_limit_mb: int = MEMORY_LIMIT_MB) -> BaseTool   # build_tools passes approval=not settings.auto_approve
@dataclass class RunResult: stdout: str; stderr: str; exit_code: int | None; timed_out: bool; killed_for_size: bool; killed_for_memory: bool = False; cancelled: bool = False
def run_python_code(code: str, ws: Workspace, *, allow_network: bool, timeout_s: float, memory_limit_mb: int = MEMORY_LIMIT_MB, cancel: threading.Event | None = None) -> RunResult
CANCEL_KEY = "mimoe_cancel"   # config["configurable"][CANCEL_KEY]: a threading.Event per turn; set by Ctrl-C (CLI) or Stop/disconnect (server), it kills the snippet's tree
MEMORY_LIMIT_MB = 2048        # resident memory of the whole tree (psutil); above it the tree is killed
# system.py
calculator: BaseTool; now: BaseTool
def make_git(ws: Workspace) -> BaseTool
def make_mimoe_status(client: MimoeClient) -> BaseTool
```
Tool names and argument schemas (flat, all strings/ints, defaults as shown):
`run_python(code: str)`, `list_files(path: str = ".", max_depth: int = 4)`,
`read_file(path: str, offset: int = 1, limit: int = 200)`,
`search_files(pattern: str, path: str = ".", glob: str = "*", max_results: int = 100)`,
`calculator(expression: str)`, `now(timezone: str | None = None)`, `mimoe_status()`,
`git(command: Literal["status","log","diff"], path: str | None = None)`.
Docstrings are the tool descriptions the model sees: one sentence of purpose, one of constraints.

## models.py (registry helpers used by CLI/server)
```python
@dataclass(frozen=True) class Preset: id: str; repo: str; file: str; size_gb: float; note: str
PRESETS: tuple[Preset, ...]
def pull_model(client: MimoeClient, spec: str, *, on_progress: Callable[[str, float | None], None]) -> RegistryModel   # spec = preset id or "owner/repo:QUANT"
def switch_model(client: MimoeClient, model_id: str, *, unload_previous: bool, on_status: Callable[[str], None]) -> LoadedModel
```
Pull path per generation: V10 -> `POST {store}/hf/models/pull` (SSE); V06 -> `POST {store}/models`
then `POST {store}/models/{id}/download {"url"}` (SSE; keep the connection open until `done`).

## server.py (FastAPI)
`POST /api/chat {thread_id, message}` -> SSE of the events above; 409 if the thread is busy or an
approval is pending. `POST /api/resume {thread_id, interrupt_id, decisions:["approve"|"reject",...]}`
-> SSE; 409 stale/nothing pending; 422 bad decisions. `GET /api/health` ->
`{mimoe_reachable, model, tokens_per_second, max_context, node, engine_version, generation, workspace, approval, network, mode:"tools"|"chat_only", error}`.
`GET /api/models` -> `{loaded:[...], registry:[...], current}`. `POST /api/model {model, unload_previous}` -> `{model, probe}`.
`GET /` serves `web/dist` (StaticFiles, html=True) mounted after the API routes.
`TrustedHostMiddleware(allowed_hosts=["127.0.0.1", "localhost", "[::1]"])`; uvicorn binds 127.0.0.1.
SSE frames: `event: <name>\ndata: <json>\n\n`; keep-alive comment every 15 s.

## tests/conftest.py (A3)
```python
class FakeMimoe:
    """httpx.MockTransport-backed fake of both engine generations."""
    def __init__(self, generation: str = "0.6", model_id: str = "qwen3-4b", *, tools_ok: bool = True) -> None
    def script(self, *turns: dict) -> None          # queue of scripted assistant replies: {"content": str} or {"tool_calls": [{"name","args"}]}; each may include "reasoning"
    def handler(self, request: httpx.Request) -> httpx.Response     # routes /models (GET/POST/DELETE), /chat/completions (stream and non-stream, tool_calls as "tool_0"...), store /models, /jsonrpc/v1 getMe, 403 on bad key
    def transport(self) -> httpx.MockTransport
    def client(self) -> httpx.Client; def async_client(self) -> httpx.AsyncClient
    calls: list[dict]                              # every chat body received, for assertions
@pytest.fixture def fake_mimoe() -> FakeMimoe        # generation "0.6"
@pytest.fixture def fake_mimoe_v10() -> FakeMimoe
@pytest.fixture def workspace_tmp(tmp_path) -> Path  # copy of ./workspace
@pytest.fixture def settings_tmp(workspace_tmp) -> Settings
@pytest.fixture def preflight_fake(fake_mimoe, settings_tmp) -> Preflight
@pytest.fixture def llm_fake(fake_mimoe, settings_tmp, preflight_fake) -> ChatOpenAI  # via make_model(http_client=fake.client(), http_async_client=fake.async_client())
```
The fake emulates the real endpoint quirks: 0.6 replies start with `<think>\n\n</think>\n\n`
(inline), 1.0 replies put reasoning in `reasoning_content`; streaming chunks follow the real
delta shapes (role chunk, `mimoe_status` progress chunks, content/tool_call deltas, final chunk
with `finish_reason` and `usage`); errors use mimOE's non-OpenAI shapes.

## Live engines available during development (not in CI)
- Studio 0.6.5: `http://localhost:8083/mimik-ai/openai/v1` (key 1234). Loaded: qwen3-4b. Registered: qwen3-4b, qwen3-4b-instruct-2507, qwen3-8b, smollm3-3b, smollm2-360m, qwen3.5-4b (the last cannot load on this engine).
- Throwaway 1.0.27 runtime: `http://localhost:8093/mimik-ai/openai/v1` (key 1234), same models plus qwen3.5-9b.
- Keep live tests small: use qwen3-4b or qwen3-4b-instruct-2507, `/no_think` or `enable_thinking:false`, `max_tokens` <= 300. Do not unload models you did not load. Do not load models larger than 3 GB (a separate job runs the big-model matrix).
