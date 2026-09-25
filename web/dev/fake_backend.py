"""Fake mimoe-agent backend for UI development: scripted SSE turns, no inference.

Run ``uv run python web/dev/fake_backend.py`` (binds 127.0.0.1:8000) and then either
``npm run dev`` in ``web/`` (the Vite proxy forwards /api) or, after ``npm run build``, open
http://127.0.0.1:8000/ directly: the built UI is mounted at "/" exactly like the real server.

Scenarios are picked by keywords in the message (case-insensitive):

- default: thinking, a ``list_files`` call and result, a markdown answer with a table
- ``python`` / ``csv`` / ``revenue``: one ``run_python`` call that needs approval
- ``two`` / ``both``: two ``run_python`` calls in one approval (Continue button)
- ``network`` / ``flag``: ``run_python`` code that trips the red-flag hint
- ``notice`` / ``loop``: a model-call-limit notice
- ``error`` / ``boom``: partial text, then an ``error`` event

``FAKE_MIMOE_DOWN=1`` simulates an unreachable Studio (health hint, chat answers with an error).
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

WEB_DIR = Path(__file__).resolve().parents[1]
WORKSPACE = WEB_DIR.parent / "workspace"
DOWN = os.environ.get("FAKE_MIMOE_DOWN", "").lower() in {"1", "true", "yes", "on"}
DECISIONS = ("approve", "reject")
REGISTRY: tuple[tuple[str, float, bool], ...] = (  # id, size in GB, downloaded
    ("qwen3-4b", 2.5, True),
    ("qwen3-4b-instruct-2507", 2.5, True),
    ("qwen3-8b", 5.0, True),
    ("smollm3-3b", 1.9, True),
    ("smollm2-360m", 0.4, True),
    ("qwen3.5-4b", 2.7, False),
)
LISTING = "notes.md\nsales.csv\nREADME.md\nsrc/\n  app.py\n  utils.py\n"
ANSWER = (
    "The workspace holds **4 files** in two folders:\n\n"
    "| file | what it is |\n|---|---|\n| notes.md | two TODO lines |\n"
    "| sales.csv | 8 rows of sales |\n"
    "| src/app.py | a small app with one TODO |\n\n"
    "Raw HTML is shown as text: <script>alert(1)</script> and images are dropped: "
    "![beacon](http://127.0.0.1:9/x)\n"
)
CODE_SALES = (
    "import pandas as pd\n\ndf = pd.read_csv('sales.csv')\n"
    "print(len(df), round(df['revenue'].sum(), 2))\n"
)
CODE_TODOS = "import pathlib\n\nfor p in pathlib.Path('.').rglob('*.py'):\n    print(p)\n"
CODE_NET = "import requests\n\nprint(requests.get('http://example.com').status_code)\n"

app = FastAPI(title="mimoe-agent fake backend")
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]"])


@dataclass
class Pending:
    """An interrupt waiting for decisions on one thread."""

    interrupt_id: str
    codes: list[str]


class State:
    """Mutable server state: current model, per-thread locks and pending approvals."""

    model: str = "qwen3-4b"
    locks: dict[str, asyncio.Lock] = {}
    pending: dict[str, Pending] = {}

    @classmethod
    def lock(cls, thread_id: str) -> asyncio.Lock:
        return cls.locks.setdefault(thread_id, asyncio.Lock())


class ChatBody(BaseModel):
    thread_id: str
    message: str


class ResumeBody(BaseModel):
    thread_id: str
    interrupt_id: str
    decisions: list[str]


class ModelBody(BaseModel):
    model: str
    unload_previous: bool = True


Event = dict[str, str]


def ev(name: str, payload: dict[str, Any]) -> Event:
    """One SSE frame as sse-starlette wants it: ``event: <name>`` + ``data: <json>``."""
    return {"event": name, "data": json.dumps(payload)}


def loaded_model(model_id: str) -> dict[str, Any]:
    size = next((gb for mid, gb, _ in REGISTRY if mid == model_id), 2.5)
    return {
        "id": model_id,
        "kind": "llm",
        "family": "qwen3",
        "max_context": 12000,
        "n_params": int(size * 1.6e9),
        "tokens_per_second": 40.9,
        "avg_tokens_per_second": 33.3,
        "supports_tools": None,
        "thinking_supported": None,
        "thinking_can_disable": None,
    }


async def words(name: str, text: str, delay: float = 0.03) -> AsyncIterator[Event]:
    """Stream ``text`` as word-sized ``token``/``thinking`` events."""
    for piece in text.replace("\n", "\n\x00").split("\x00"):
        for word in piece.split(" "):
            yield ev(name, {"text": word + " " if not word.endswith("\n") else word})
            await asyncio.sleep(delay)


def done(status: str, elapsed: float, calls: int) -> Event:
    usage = {"input_tokens": 700 * calls, "output_tokens": 45 * calls, "llm_calls": calls}
    return ev(
        "done",
        {"status": status, "elapsed_s": elapsed, "model": State.model, "usage_total": usage},
    )


async def scenario(thread_id: str, message: str) -> AsyncIterator[Event]:
    """Pick and play one scripted turn."""
    m = message.lower()
    if DOWN:
        yield ev(
            "error",
            {
                "message": "mimOE Studio is not reachable",
                "hint": "open mimOE Studio, load a model, then try again",
            },
        )
        return
    if any(k in m for k in ("error", "boom")):
        async for e in words("token", "Working on it, "):
            yield e
        yield ev(
            "error",
            {
                "message": "mimOE returned 500: llama_decode() failed",
                "hint": "the conversation exceeded the model's context; start a new one",
            },
        )
        return
    if any(k in m for k in ("notice", "loop")):
        yield ev("tool_call", {"id": "tool_0", "name": "now", "args": {}})
        await asyncio.sleep(0.3)
        yield ev(
            "tool_result",
            {"id": "tool_0", "name": "now", "content": "2026-09-25 09:00", "is_error": False},
        )
        yield ev(
            "notice",
            {"text": "Model call limits exceeded: thread limit (8/8); stopping this turn."},
        )
        yield done("completed", 12.4, 8)
        return
    if any(k in m for k in ("two", "both")):
        codes = [CODE_SALES, CODE_TODOS]
    elif any(k in m for k in ("network", "flag")):
        codes = [CODE_NET]
    elif any(k in m for k in ("python", "csv", "revenue")):
        codes = [CODE_SALES]
    else:
        codes = []
    if codes:
        async for e in approval_turn(thread_id, codes):
            yield e
        return
    async for e in words("thinking", "The user wants a listing; list_files is the right tool.\n"):
        yield e
    yield ev(
        "tool_call", {"id": "tool_0", "name": "list_files", "args": {"path": ".", "max_depth": 4}}
    )
    await asyncio.sleep(0.4)
    yield ev(
        "tool_result", {"id": "tool_0", "name": "list_files", "content": LISTING, "is_error": False}
    )
    async for e in words("token", ANSWER):
        yield e
    yield done("completed", 3.1, 2)


async def approval_turn(thread_id: str, codes: list[str]) -> AsyncIterator[Event]:
    """The model asks for ``run_python``; the interrupt is registered before the frame goes out."""
    async for e in words("thinking", "I should compute this with pandas rather than guess.\n"):
        yield e
    requests = [{"name": "run_python", "args": {"code": c}} for c in codes]
    for i, r in enumerate(requests):
        yield ev("tool_call", {"id": f"tool_{i}", "name": "run_python", "args": r["args"]})
    interrupt_id = uuid.uuid4().hex
    State.pending[thread_id] = Pending(interrupt_id, codes)
    yield ev(
        "approval_required",
        {
            "interrupt_id": interrupt_id,
            "action_requests": [
                {**r, "description": f"Run this Python code?\n\n{r['args']['code']}"}
                for r in requests
            ],
            "review_configs": [
                {"action_name": "run_python", "allowed_decisions": list(DECISIONS)}
                for _ in requests
            ],
        },
    )
    yield done("awaiting_approval", 1.9, 1)


def fake_output(code: str) -> str:
    if "requests" in code:
        return "Traceback (most recent call last):\n  ...\nRuntimeError: network disabled"
    if "rglob" in code:
        return "src/app.py\nsrc/utils.py\n"
    return "8 1836.6\n"


async def resume_turn(p: Pending, decisions: list[str]) -> AsyncIterator[Event]:
    """Tool results in request order (rejections are error results), then the answer."""
    for i, (code, decision) in enumerate(zip(p.codes, decisions, strict=True)):
        if decision == "reject":
            content = (
                "User rejected the tool call for `run_python` with reason: The user declined "
                "to run this code. Tell the user it was not executed and stop; do not retry."
            )
            yield ev(
                "tool_result",
                {"id": f"tool_{i}", "name": "run_python", "content": content, "is_error": True},
            )
            continue
        await asyncio.sleep(0.5)
        yield ev(
            "tool_result",
            {
                "id": f"tool_{i}",
                "name": "run_python",
                "content": fake_output(code),
                "is_error": False,
            },
        )
    if all(d == "reject" for d in decisions):
        text = "Understood, I did not run the code. Ask again if you change your mind.\n"
    elif "requests" in p.codes[0]:
        text = "The code failed: run_python has no network access, so I cannot fetch that page.\n"
    else:
        text = "sales.csv has **8 rows** and the total revenue is **1836.6**.\n"
    async for e in words("token", text):
        yield e
    yield done("completed", 4.2, 2)


async def locked(thread_id: str, events: AsyncIterator[Event]) -> AsyncIterator[Event]:
    """Serialise runs per thread; a client disconnect cancels the generator and frees the lock."""
    async with State.lock(thread_id):
        async for e in events:
            yield e


@app.get("/api/health")
async def health() -> dict[str, Any]:
    if DOWN:
        return {
            "mimoe_reachable": False,
            "model": None,
            "tokens_per_second": None,
            "max_context": None,
            "node": None,
            "engine_version": None,
            "generation": None,
            "workspace": str(WORKSPACE),
            "approval": "manual",
            "network": False,
            "mode": "chat_only",
            "error": "mimOE Studio is not reachable at http://localhost:8083; "
            "open Studio and load a model",
        }
    return {
        "mimoe_reachable": True,
        "model": State.model,
        "tokens_per_second": 33.3,
        "max_context": 12000,
        "node": "fake-node",
        "engine_version": "v3.22.8 (developer edition)",
        "generation": "0.6",
        "workspace": str(WORKSPACE),
        "approval": "manual",
        "network": False,
        "mode": "chat_only" if State.model == "smollm2-360m" else "tools",
        "error": None,
    }


@app.get("/api/models")
async def models() -> dict[str, Any]:
    registry = [
        {"id": mid, "kind": "llm", "ready": ready, "size_bytes": int(gb * 1e9), "raw": {}}
        for mid, gb, ready in REGISTRY
    ]
    return {"loaded": [loaded_model(State.model)], "registry": registry, "current": State.model}


@app.post("/api/model")
async def switch_model(body: ModelBody) -> dict[str, Any]:
    if body.model not in {mid for mid, _, _ in REGISTRY}:
        raise HTTPException(
            404, {"message": f"unknown model {body.model!r}", "hint": "see GET /api/models"}
        )
    await asyncio.sleep(1.5)  # a load takes a while; the picker shows progress meanwhile
    if body.model == "qwen3.5-4b":
        raise HTTPException(
            502,
            {
                "message": f"{body.model} cannot load on this engine",
                "hint": "Studio 0.6 lacks the qwen35 architecture",
            },
        )
    State.model = body.model
    tools_ok = body.model != "smollm2-360m"
    detail = (
        "answered the warm-up with a structured tool call"
        if tools_ok
        else "no tool call in the warm-up; chat-only mode"
    )
    return {
        "model": loaded_model(body.model),
        "probe": {"tools_ok": tools_ok, "latency_s": 3.9, "detail": detail},
    }


@app.post("/api/chat")
async def chat(body: ChatBody) -> EventSourceResponse:
    if State.lock(body.thread_id).locked():
        raise HTTPException(
            409, {"message": "run in progress on this thread", "hint": "wait for it to finish"}
        )
    if (p := State.pending.get(body.thread_id)) is not None:
        raise HTTPException(
            409,
            {
                "message": "approval pending on this thread",
                "hint": "decide first",
                "interrupt_id": p.interrupt_id,
            },
        )
    return EventSourceResponse(
        locked(body.thread_id, scenario(body.thread_id, body.message)), ping=5
    )


@app.post("/api/resume")
async def resume(body: ResumeBody) -> EventSourceResponse:
    p = State.pending.get(body.thread_id)
    if p is None:
        raise HTTPException(
            409, {"message": "nothing to resume on this thread", "hint": "send a message"}
        )
    if body.interrupt_id != p.interrupt_id:
        raise HTTPException(
            409, {"message": "stale interrupt id", "hint": "start a new conversation"}
        )
    if len(body.decisions) != len(p.codes) or any(d not in DECISIONS for d in body.decisions):
        raise HTTPException(
            422,
            {
                "message": f"expected {len(p.codes)} decisions from {list(DECISIONS)}, "
                f"got {body.decisions}",
                "hint": "one decision per action request",
            },
        )
    del State.pending[body.thread_id]
    return EventSourceResponse(locked(body.thread_id, resume_turn(p, body.decisions)), ping=5)


# Like the real server: the built UI is mounted after the API routes, only when it exists.
if (WEB_DIR / "dist").is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIR / "dist", html=True), name="ui")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
