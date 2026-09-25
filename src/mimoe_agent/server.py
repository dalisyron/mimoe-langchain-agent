"""FastAPI backend for the web UI: chat and resume as SSE, health, models, the static bundle.

CONTRACTS.md ("server.py") and PLAN.md section 7 define the surface; this module wires the
shared pieces together and owns only what is server-specific:

* **Lazy start.** ``serve`` always comes up. The agent is built on the first request after a
  successful :func:`mimoe_agent.mimoe.preflight` (attempted at startup, then again on every
  ``/api/chat``, ``/api/resume`` and ``/api/health`` while it keeps failing). Meanwhile
  ``/api/health`` reports the hint and a chat answers with a single ``error`` event carrying it.
* **One run per thread.** A per-thread ``asyncio.Lock`` is held for the whole SSE response; a
  second request on a locked thread gets 409. Inside the lock ``agent.aget_state`` decides
  whether an approval is pending: a new message would silently discard the interrupt (409), and
  a resume needs exactly that interrupt (409 when nothing is pending or the id is stale, 422
  when the decisions do not fit the action requests).
* **Disconnects.** The graph runs in a task of its own (:func:`_pump`) that feeds the SSE
  generator through a queue. When the browser goes away sse-starlette cancels its task group
  with a level-triggered anyio scope, in which every await raises again: thrown straight into
  the graph that cuts langgraph's teardown short and leaves the model call running (observed
  with the fake engine). Cancelling the pump task instead is delivered once, so langgraph
  cancels and awaits its node tasks; a shielded wait lets that finish, then the lock is
  released, all before the response returns. :class:`_TurnResponse` closes the generator
  after the task group is gone, because sse-starlette leaves a suspended iterator to the GC.
* **Tool-call ids.** 0.6 engines number tool calls per response (``tool_0`` every time) and the
  web UI keys its tool cards by id within one assistant turn, so the runs of a turn (the chat
  and the resumes that answer its approvals) share one :class:`mimoe_agent.stream.ToolCallIds`:
  a repeated id becomes ``tool_0#2`` and its result, arriving in the next run, follows it.
* **Loopback only.** ``TrustedHostMiddleware`` accepts ``127.0.0.1``, ``localhost`` and
  ``[::1]`` (a DNS-rebinding page cannot drive the agent), uvicorn binds ``127.0.0.1`` and
  there is no CORS middleware: the Vite dev proxy keeps ``/api`` same-origin.

LangChain is imported inside :func:`create_app`, after the entry point has called
:func:`mimoe_agent.config.apply_tracing_env`.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.types import Receive, Scope, Send

from mimoe_agent.config import Settings
from mimoe_agent.mimoe import (
    HINT_NO_MODEL,
    MimoeClient,
    Preflight,
    ProbeResult,
    friendly_error,
    preflight,
)
from mimoe_agent.models import switch_model, valid_model_id

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.types import Interrupt, StateSnapshot

    from mimoe_agent.stream import ToolCallIds

log = logging.getLogger(__name__)

ALLOWED_HOSTS: tuple[str, ...] = ("127.0.0.1", "localhost", "[::1]")
BIND_HOST = "127.0.0.1"
PING_INTERVAL_S = 15
"""Keep-alive comment interval (``: ping - <utc time>``), so proxies and browsers keep waiting."""
DECISION_TYPES: tuple[str, ...] = ("approve", "reject")
"""Decisions the web UI can express; the interrupt's ``review_configs`` narrow them further."""
REJECT_MESSAGE = (
    "The user declined to run this code. Tell the user it was not executed and stop; do not retry."
)
"""Reason attached to every rejection (a bare reject produced a confused answer)."""
DEFAULT_WEB_DIST = Path(__file__).resolve().parents[2] / "web" / "dist"
"""The committed Vite build when running from a checkout; absent from an installed wheel."""
MAX_THREAD_ID_LENGTH = 200

HINT_UI_NOT_BUILT = (
    "Build it with `cd web && npm ci && npm run build` (the repository ships web/dist); "
    "the API lives under /api."
)
HINT_RUN_IN_PROGRESS = "Wait for the current turn to finish (or stop it), then send again."
HINT_APPROVAL_PENDING = (
    "Approve or reject the pending run_python request first (POST /api/resume), "
    "or start a new conversation."
)
HINT_NOTHING_PENDING = (
    "Send a message with POST /api/chat; resume only answers an approval_required event."
)
HINT_STALE_INTERRUPT = (
    "The pending request has a different interrupt_id; reload the page or start a new conversation."
)
HINT_DECISIONS = "Send one decision per action request, in order, each 'approve' or 'reject'."
HINT_SWITCH_IN_PROGRESS = "A model switch is in progress; wait for POST /api/model to finish."
HINT_SWITCH_BUSY = "Wait for the running conversation turn(s) to finish, then switch again."
HINT_MODEL_ID = "Model ids use letters, digits, '.', '_' and '-'; GET /api/models lists them."
HINT_REQUEST_BODY = "Send a JSON body with the fields the endpoint expects (see docs/CONTRACTS.md)."
HINT_NOT_FOUND = "The API offers /api/chat, /api/resume, /api/health, /api/models and /api/model."
STATUS_STARTING = "connecting to mimOE Studio, please wait"
STATUS_SWITCHING = "a model switch is in progress, please wait"

Event = dict[str, Any]
ModelFactory = Callable[[Settings, Preflight], "BaseChatModel"]
"""Builds the chat model for a preflight; tests inject one that talks to the fake engine."""


# -- request bodies ------------------------------------------------------------------------------


class ChatBody(BaseModel):
    """``POST /api/chat``."""

    thread_id: str = Field(min_length=1, max_length=MAX_THREAD_ID_LENGTH)
    message: str


class ResumeBody(BaseModel):
    """``POST /api/resume``: one decision per action request of the pending interrupt."""

    thread_id: str = Field(min_length=1, max_length=MAX_THREAD_ID_LENGTH)
    interrupt_id: str = Field(min_length=1)
    decisions: list[str]


class ModelBody(BaseModel):
    """``POST /api/model``."""

    model: str
    unload_previous: bool = True


# -- helpers -------------------------------------------------------------------------------------


def _http(status: int, message: str, hint: str) -> HTTPException:
    """An error the UI can show: ``detail`` is always ``{"message", "hint"}``."""
    return HTTPException(status, {"message": message, "hint": hint})


def _error_text(error: tuple[str, str]) -> str:
    message, hint = error
    return f"{message.rstrip('.')}. {hint}".strip() if hint else message


def sse_frame(event: Mapping[str, Any]) -> dict[str, str]:
    """Turn a stream event into the sse-starlette frame ``event: <name>`` + ``data: <json>``."""
    payload = {key: value for key, value in event.items() if key != "event"}
    return {"event": str(event.get("event") or "message"), "data": json.dumps(payload, default=str)}


async def _aclose(source: object) -> None:
    """Close an async generator if it is one (a generator already finished or running is left)."""
    aclose = getattr(source, "aclose", None)
    if aclose is None:
        return
    try:
        await aclose()
    except RuntimeError:  # "aclose(): asynchronous generator is already running"
        log.debug("could not close %r", source, exc_info=True)


async def _error_events(error: tuple[str, str]) -> AsyncIterator[Event]:
    message, hint = error
    yield {"event": "error", "message": message, "hint": hint}


async def _pump(events: AsyncIterator[Event], queue: asyncio.Queue[Event | None]) -> None:
    """Drain ``events`` into ``queue``; ``None`` marks the end, also after a failure."""
    try:
        async for event in events:
            queue.put_nowait(event)
    except Exception as exc:  # aiter_events maps failures to events; this is the last resort
        log.exception("run failed outside the event mapper")
        message, hint = friendly_error(exc)
        queue.put_nowait({"event": "error", "message": message, "hint": hint})
    finally:
        await _aclose(events)
        queue.put_nowait(None)


def _pending_interrupt(snapshot: StateSnapshot) -> Interrupt | None:
    """The interrupt a thread is waiting on, or ``None`` (fresh thread, completed turn)."""
    for interrupt in snapshot.interrupts:
        return interrupt
    for task in snapshot.tasks:
        for interrupt in task.interrupts:
            return interrupt
    return None


def _decisions(raw: list[str], interrupt_value: Any) -> list[dict[str, str]]:
    """Validate the UI's decisions against the interrupt and build the resume decisions.

    Raises:
        HTTPException: 422 when the count differs from the action requests or a decision is not
            allowed for its request (``review_configs[i].allowed_decisions``).
    """
    value = interrupt_value if isinstance(interrupt_value, Mapping) else {}
    requests = list(value.get("action_requests") or [])
    configs = list(value.get("review_configs") or [])
    if len(raw) != len(requests):
        raise _http(422, f"expected {len(requests)} decision(s), got {len(raw)}", HINT_DECISIONS)
    decisions: list[dict[str, str]] = []
    for index, decision in enumerate(raw):
        config = (
            configs[index] if index < len(configs) and isinstance(configs[index], Mapping) else {}
        )
        allowed = config.get("allowed_decisions")
        allowed = [str(a) for a in allowed] if isinstance(allowed, list) else list(DECISION_TYPES)
        if decision not in DECISION_TYPES or decision not in allowed:
            choices = ", ".join(a for a in allowed if a in DECISION_TYPES) or "none"
            raise _http(
                422,
                f"decision {index + 1} must be one of: {choices}; got {decision!r}",
                HINT_DECISIONS,
            )
        if decision == "approve":
            decisions.append({"type": "approve"})
        else:
            decisions.append({"type": "reject", "message": REJECT_MESSAGE})
    return decisions


def _log_status(text: str) -> None:
    """Progress lines from preflight and model loads; 1.0 engines report every 1 %."""
    log.log(logging.DEBUG if "%" in text else logging.INFO, "%s", text)


# -- one run --------------------------------------------------------------------------------------


class _Turn:
    """The per-thread lock held for one SSE response; released exactly once."""

    def __init__(self, thread_id: str, lock: asyncio.Lock) -> None:
        self.thread_id = thread_id
        self._lock = lock
        self._held = True

    def release(self) -> None:
        if self._held:
            self._held = False
            self._lock.release()

    async def frames(self, events: AsyncIterator[Event]) -> AsyncGenerator[dict[str, str]]:
        """SSE frames for ``events``, which a task of their own produces (module docstring).

        However the generator ends (exhausted, cancelled by a disconnect, closed), the run
        task is cancelled if still going, its teardown awaited under a shield, and only then
        is the thread lock released.
        """
        queue: asyncio.Queue[Event | None] = asyncio.Queue()
        pump = asyncio.create_task(_pump(events, queue), name=f"mimoe-agent run {self.thread_id}")
        try:
            while (event := await queue.get()) is not None:
                yield sse_frame(event)
        finally:
            with anyio.CancelScope(shield=True):
                if not pump.done():
                    pump.cancel()
                await asyncio.gather(pump, return_exceptions=True)
            self.release()

    def response(self, events: AsyncIterator[Event]) -> EventSourceResponse:
        return _TurnResponse(self, self.frames(events))


class _TurnResponse(EventSourceResponse):
    """``EventSourceResponse`` that finishes its turn when the response is over.

    sse-starlette cancels its task group on a client disconnect but does not close the body
    iterator; a generator suspended at ``yield`` would then wait for the garbage collector to
    run its ``finally``. Closing it here (after the task group is gone, so it is never running)
    stops the run and releases the thread lock before the request returns.
    """

    def __init__(self, turn: _Turn, frames: AsyncGenerator[dict[str, str]]) -> None:
        super().__init__(frames, ping=PING_INTERVAL_S)
        self._turn = turn
        self._frames = frames

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await _aclose(self._frames)
            self._turn.release()


# -- runtime --------------------------------------------------------------------------------------


@dataclass
class _Ready:
    """The agent and the preflight it was built from."""

    pre: Preflight
    agent: CompiledStateGraph[Any, Any, Any, Any]


class AgentRuntime:
    """Server state: the lazily built agent, per-thread locks and the model switcher.

    Blocking engine calls (preflight, model loads, listings) run in a worker thread through
    ``asyncio.to_thread`` so the event loop keeps answering ``/api/health`` meanwhile; the agent
    itself is assembled on the loop thread.
    """

    def __init__(
        self, settings: Settings, client: MimoeClient, model_factory: ModelFactory
    ) -> None:
        self.settings = settings
        self.client = client
        self._model_factory = model_factory
        self.model_id: str | None = settings.model
        """The model the agent should use; ``None`` lets the preflight pick the first loaded."""
        self.ready: _Ready | None = None
        self.last_error: tuple[str, str] | None = None
        """``(message, hint)`` of the last failed preflight, for ``/api/health``."""
        self.switching = False
        self._checkpointer: BaseCheckpointSaver | None = None
        self._locks: dict[str, asyncio.Lock] = {}
        self._tool_ids: dict[str, ToolCallIds] = {}
        """Thread id -> the tool-call id aliases of its current turn (module docstring)."""
        self._build_lock = asyncio.Lock()

    # -- threads -------------------------------------------------------------------------------

    def lock(self, thread_id: str) -> asyncio.Lock:
        """The lock of ``thread_id`` (created on first use, kept for the process lifetime)."""
        return self._locks.setdefault(thread_id, asyncio.Lock())

    def busy_threads(self) -> list[str]:
        """Thread ids with a run in progress."""
        return [thread_id for thread_id, lock in self._locks.items() if lock.locked()]

    def tool_ids(self, thread_id: str, *, new_turn: bool) -> ToolCallIds:
        """The tool-call id aliases for a run on ``thread_id``.

        A chat starts a new turn with fresh aliases (the UI keys tool cards within one
        assistant turn); a resume continues the turn, so a call the model repeats with the raw
        id of an earlier run (``tool_0`` on 0.6 engines) still gets a unique id and its result
        is remapped to it.
        """
        from mimoe_agent.stream import ToolCallIds

        if new_turn or thread_id not in self._tool_ids:
            self._tool_ids[thread_id] = ToolCallIds()
        return self._tool_ids[thread_id]

    async def begin(self, thread_id: str) -> _Turn:
        """Take the thread's lock for one run.

        Raises:
            HTTPException: 409 while a model switch or another run on the thread is in progress.
        """
        if self.switching:
            raise _http(409, "a model switch is in progress", HINT_SWITCH_IN_PROGRESS)
        lock = self.lock(thread_id)
        if lock.locked():
            raise _http(409, f"a run is in progress on thread {thread_id}", HINT_RUN_IN_PROGRESS)
        await lock.acquire()  # not locked and single-threaded: returns without waiting
        return _Turn(thread_id, lock)

    def run_config(self, thread_id: str, pre: Preflight) -> dict[str, Any]:
        """The run config: the thread plus the model name for ``done`` on notice-only turns."""
        return {"configurable": {"thread_id": thread_id}, "metadata": {"model": pre.model.id}}

    # -- agent ---------------------------------------------------------------------------------

    async def ensure_ready(self) -> _Ready:
        """Return the agent, running the preflight and building it first when needed.

        Raises:
            MimoeError: (or whatever the preflight raised) when the engine is not usable yet;
                the failure is kept in :attr:`last_error` and the next call tries again.
        """
        if self.ready is not None:
            return self.ready
        async with self._build_lock:
            if self.ready is not None:
                return self.ready
            try:
                pre = await asyncio.to_thread(self._preflight)
                ready = _Ready(pre, self._assemble(pre))
            except Exception as exc:
                self.last_error = friendly_error(exc)
                raise
            self.ready = ready
            self.last_error = None
            log.info(
                "agent ready: %s on %s (%s), %s",
                pre.model.id,
                pre.engine.base_url,
                pre.engine.version or "engine version unknown",
                "tools enabled" if pre.tools_enabled else "chat-only mode",
            )
            return ready

    def _preflight(self) -> Preflight:
        settings = dataclasses.replace(self.settings, model=self.model_id)
        pre = preflight(settings, client=self.client, on_status=_log_status)
        for warning in pre.warnings:
            log.warning("%s", warning)
        if pre.probe is not None:
            log.info("%s", pre.probe.detail)
        return pre

    def _assemble(self, pre: Preflight) -> CompiledStateGraph[Any, Any, Any, Any]:
        from langgraph.checkpoint.memory import InMemorySaver

        from mimoe_agent.agent import build_agent
        from mimoe_agent.tools import build_tools

        if self._checkpointer is None:
            self._checkpointer = InMemorySaver()  # shared by every build: threads survive a switch
        settings = dataclasses.replace(self.settings, model=pre.model.id)
        return build_agent(
            settings,
            pre,
            llm=self._model_factory(settings, pre),
            tools=build_tools(settings, self.client),
            checkpointer=self._checkpointer,
        )

    # -- reports -------------------------------------------------------------------------------

    async def health(self) -> dict[str, Any]:
        """The ``/api/health`` report; never raises."""
        report: dict[str, Any] = {
            "mimoe_reachable": False,
            "model": self.model_id,
            "tokens_per_second": None,
            "max_context": None,
            "node": None,
            "engine_version": None,
            "generation": None,
            "workspace": str(self.settings.workspace),
            "approval": "auto" if self.settings.auto_approve else "manual",
            "network": "on" if self.settings.allow_network else "off",
            "mode": "chat_only",
            "error": None,
        }
        try:
            if self.switching:
                self._describe(report)
                report["error"] = STATUS_SWITCHING
                return report
            if self.ready is None and self._build_lock.locked():
                report["error"] = STATUS_STARTING
                return report
            try:
                ready = await self.ensure_ready()
            except Exception as exc:
                report["error"] = _error_text(friendly_error(exc))
                return report
            self._describe(report)
            try:
                loaded = await asyncio.to_thread(self.client.loaded_models)
            except Exception as exc:
                report["error"] = _error_text(friendly_error(exc))
                return report
            report["mimoe_reachable"] = True
            current = next((m for m in loaded if m.id == ready.pre.model.id), None)
            if current is None:
                report["error"] = _error_text(
                    (f"{ready.pre.model.id} is no longer loaded", HINT_NO_MODEL)
                )
            else:
                report["tokens_per_second"] = (
                    current.tokens_per_second
                    or current.avg_tokens_per_second
                    or report["tokens_per_second"]
                )
                report["max_context"] = current.max_context or report["max_context"]
        except Exception as exc:  # never let a bug take the badge down
            log.exception("health report failed")
            report["error"] = f"{type(exc).__name__}: {exc}"
        return report

    def _describe(self, report: dict[str, Any]) -> None:
        """Fill the engine and model fields from the preflight the agent was built with."""
        if self.ready is None:
            return
        pre = self.ready.pre
        report.update(
            model=pre.model.id,
            tokens_per_second=pre.model.tokens_per_second or pre.model.avg_tokens_per_second,
            max_context=pre.model.max_context,
            node=pre.engine.node_name,
            engine_version=pre.engine.version,
            generation=str(pre.engine.generation),
            mode="tools" if pre.tools_enabled else "chat_only",
        )

    async def models(self) -> dict[str, Any]:
        """``/api/models``: loaded and registered models plus the current id.

        Raises:
            HTTPException: 502 with the engine's hint when a listing fails.
        """
        try:
            loaded = await asyncio.to_thread(self.client.loaded_models)
            registry = await asyncio.to_thread(self.client.registry_models)
        except Exception as exc:
            message, hint = friendly_error(exc)
            raise _http(502, message, hint) from exc
        current = self.ready.pre.model.id if self.ready is not None else self.model_id
        return {
            "loaded": [dataclasses.asdict(m) for m in loaded],
            "registry": [dataclasses.asdict(m) for m in registry],
            "current": current,
        }

    async def switch(self, model_id: str, *, unload_previous: bool) -> dict[str, Any]:
        """``/api/model``: load ``model_id``, re-run the preflight and rebuild the agent.

        Raises:
            HTTPException: 422 for an invalid id, 409 while a run or another switch is in
                progress, 502 (with the engine's hint) when the load or the preflight fails.
        """
        model_id = model_id.strip()
        if not valid_model_id(model_id):
            raise _http(422, f"{model_id!r} is not a valid model id", HINT_MODEL_ID)
        if self.switching:
            raise _http(409, "a model switch is already in progress", HINT_SWITCH_IN_PROGRESS)
        busy = self.busy_threads()
        if busy:
            raise _http(409, f"a run is in progress on thread {busy[0]}", HINT_SWITCH_BUSY)
        self.switching = True
        try:
            async with self._build_lock:
                try:
                    await asyncio.to_thread(
                        switch_model,
                        self.client,
                        model_id,
                        unload_previous=unload_previous,
                        on_status=_log_status,
                    )
                except Exception as exc:
                    message, hint = friendly_error(exc)
                    raise _http(502, message, hint) from exc
                self.model_id = model_id
                self.ready = None  # the previous agent may point at an unloaded model
                try:
                    pre = await asyncio.to_thread(self._preflight)
                    self.ready = _Ready(pre, self._assemble(pre))
                except Exception as exc:
                    self.last_error = friendly_error(exc)
                    raise _http(502, *self.last_error) from exc
                self.last_error = None
        finally:
            self.switching = False
        log.info("switched to %s (%s)", pre.model.id, "tools" if pre.tools_enabled else "chat only")
        probe: ProbeResult | None = pre.probe
        return {
            "model": dataclasses.asdict(pre.model),
            "probe": dataclasses.asdict(probe) if probe is not None else None,
        }


# -- app ------------------------------------------------------------------------------------------


def create_app(
    settings: Settings,
    *,
    client: MimoeClient | None = None,
    model_factory: ModelFactory | None = None,
    web_dist: Path | None = None,
) -> FastAPI:
    """Build the FastAPI application.

    Args:
        settings: Effective settings; ``apply_tracing_env(settings)`` must already have run.
        client: Engine client shared by the preflight, the ``mimoe_status`` tool and the
            model endpoints; defaults to ``MimoeClient(settings.base_url, settings.api_key)``.
            Tests pass one built on ``FakeMimoe``.
        model_factory: Builds the chat model from ``(settings, preflight)``; defaults to
            :func:`mimoe_agent.llm.make_model`. Tests inject the fake engine's HTTP clients.
        web_dist: The built web UI to serve at ``/``; defaults to the checkout's ``web/dist``.
            When the directory is missing ``/`` answers with a JSON hint instead.

    Returns:
        The app. Its runtime state is at ``app.state.runtime`` (:class:`AgentRuntime`).
    """
    from langchain_core.messages import HumanMessage
    from langgraph.types import Command

    from mimoe_agent.stream import aiter_events

    if model_factory is None:
        from mimoe_agent.llm import make_model

        model_factory = make_model
    engine_client = client or MimoeClient(settings.base_url, settings.api_key)
    runtime = AgentRuntime(settings, engine_client, model_factory)
    dist = web_dist if web_dist is not None else DEFAULT_WEB_DIST

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            await runtime.ensure_ready()
        except Exception as exc:
            log.warning("not ready yet: %s", _error_text(friendly_error(exc)))
        yield

    app = FastAPI(title="mimoe-agent", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.runtime = runtime
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(ALLOWED_HOSTS))

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        problems = "; ".join(
            ".".join(str(part) for part in error.get("loc", ()) if part != "body")
            + f": {error.get('msg', 'invalid')}"
            for error in exc.errors()
        )
        detail = {
            "message": f"invalid request: {problems or 'malformed body'}",
            "hint": HINT_REQUEST_BODY,
        }
        return JSONResponse({"detail": detail}, status_code=422)

    @app.post("/api/chat", response_class=EventSourceResponse)
    async def chat(body: ChatBody) -> EventSourceResponse:
        if not body.message.strip():
            raise _http(422, "message is empty", "Type a message first.")
        turn = await runtime.begin(body.thread_id)
        try:
            try:
                ready = await runtime.ensure_ready()
            except Exception as exc:
                return turn.response(_error_events(friendly_error(exc)))
            config = runtime.run_config(body.thread_id, ready.pre)
            snapshot = await ready.agent.aget_state(config)
            if _pending_interrupt(snapshot) is not None:
                raise _http(409, "an approval is pending on this thread", HINT_APPROVAL_PENDING)
            payload = {"messages": [HumanMessage(body.message)]}
            ids = runtime.tool_ids(body.thread_id, new_turn=True)
            return turn.response(aiter_events(ready.agent, payload, config, tool_ids=ids))
        except BaseException:
            turn.release()
            raise

    @app.post("/api/resume", response_class=EventSourceResponse)
    async def resume(body: ResumeBody) -> EventSourceResponse:
        turn = await runtime.begin(body.thread_id)
        try:
            try:
                ready = await runtime.ensure_ready()
            except Exception as exc:
                return turn.response(_error_events(friendly_error(exc)))
            config = runtime.run_config(body.thread_id, ready.pre)
            interrupt = _pending_interrupt(await ready.agent.aget_state(config))
            if interrupt is None:
                raise _http(409, "nothing to resume on this thread", HINT_NOTHING_PENDING)
            if interrupt.id != body.interrupt_id:
                raise _http(409, "stale interrupt id", HINT_STALE_INTERRUPT)
            decisions = _decisions(body.decisions, interrupt.value)
            payload = Command(resume={"decisions": decisions})
            ids = runtime.tool_ids(body.thread_id, new_turn=False)
            return turn.response(aiter_events(ready.agent, payload, config, tool_ids=ids))
        except BaseException:
            turn.release()
            raise

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return await runtime.health()

    @app.get("/api/models")
    async def models() -> dict[str, Any]:
        return await runtime.models()

    @app.post("/api/model")
    async def switch(body: ModelBody) -> dict[str, Any]:
        return await runtime.switch(body.model, unload_previous=body.unload_previous)

    @app.api_route("/api/{rest:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def unknown(rest: str) -> None:
        raise _http(404, f"no such endpoint: /api/{rest}", HINT_NOT_FOUND)

    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="ui")
    else:

        @app.get("/")
        async def root() -> dict[str, str]:
            return {"message": "the web UI is not built", "hint": HINT_UI_NOT_BUILT}

    return app


def run(settings: Settings, port: int) -> None:
    """Serve :func:`create_app` on ``127.0.0.1:port`` (loopback only; there is no host flag)."""
    _configure_logging()
    uvicorn.run(create_app(settings), host=BIND_HOST, port=port, log_level="info")


def _configure_logging() -> None:
    """Show this package's INFO lines next to uvicorn's (uvicorn configures only its own)."""
    package = logging.getLogger("mimoe_agent")
    if package.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:     %(message)s"))
    package.addHandler(handler)
    package.setLevel(logging.INFO)
