"""``create_app`` end to end through ``httpx.ASGITransport`` against the fake engine.

The fake reaches the server twice: through ``client=`` (preflight, the ``mimoe_status`` tool,
health/models/model) and through ``model_factory=`` (the chat model's HTTP clients). The model's
async client sits behind :class:`GatedTransport`, so a test can hold a model call open and act
while the run is in progress, or cut the connection while it is.

``ASGITransport`` buffers a response until the app is done with it, hence the gate: what the
browser sees as a live stream arrives here as one body of CRLF-separated frames.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
from conftest import DEFAULT_SIZE, FakeMimoe
from fastapi import FastAPI

from mimoe_agent import server
from mimoe_agent.agent import APPROVAL_WARNING
from mimoe_agent.config import Settings
from mimoe_agent.llm import make_model
from mimoe_agent.mimoe import HINT_NO_MODEL, HINT_NOT_REACHABLE, MimoeClient, Preflight
from mimoe_agent.server import (
    HINT_APPROVAL_PENDING,
    HINT_RUN_IN_PROGRESS,
    HINT_SWITCH_IN_PROGRESS,
    REJECT_MESSAGE,
    STATUS_SWITCHING,
    AgentRuntime,
    create_app,
    sse_frame,
)

BASE_URL = "http://127.0.0.1"
"""ASGITransport derives the Host header from it; ``http://testserver`` would fail TrustedHost."""
LIST_FILES = {"tool_calls": [{"name": "list_files", "args": {"path": "."}}]}
RUN_PYTHON = {"tool_calls": [{"name": "run_python", "args": {"code": "print(6*7)"}}]}
HEALTH_KEYS = {
    "mimoe_reachable",
    "model",
    "tokens_per_second",
    "max_context",
    "node",
    "engine_version",
    "generation",
    "workspace",
    "approval",
    "network",
    "mode",
    "error",
}


class GatedTransport(httpx.AsyncBaseTransport):
    """The fake engine behind a gate: requests wait while ``gate`` is clear.

    ``arrived`` is set when a request reaches the transport, so a test knows the run is inside
    its model call (the thread lock is held) before it acts.
    """

    def __init__(self, fake: FakeMimoe) -> None:
        self._inner = httpx.MockTransport(fake.handler)
        self.gate = asyncio.Event()
        self.gate.set()
        self.arrived = asyncio.Event()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.arrived.set()
        await self.gate.wait()
        return await self._inner.handle_async_request(request)


@dataclass
class Harness:
    fake: FakeMimoe
    settings: Settings
    app: FastAPI
    transport: GatedTransport

    @property
    def runtime(self) -> AgentRuntime:
        return self.app.state.runtime

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url=BASE_URL)

    async def prime(self) -> None:
        """Run the preflight now, so a test's scripted turns are not eaten by the probe."""
        await self.runtime.ensure_ready()
        self.fake.calls.clear()


def make_harness(fake: FakeMimoe, settings: Settings, dist: Path) -> Harness:
    transport = GatedTransport(fake)

    def model_factory(s: Settings, pre: Preflight) -> Any:
        return make_model(
            s,
            pre,
            http_client=fake.client(),
            http_async_client=httpx.AsyncClient(transport=transport, trust_env=False),
        )

    engine = MimoeClient(settings.base_url, settings.api_key, client=fake.client())
    app = create_app(settings, client=engine, model_factory=model_factory, web_dist=dist)
    return Harness(fake, settings, app, transport)


@pytest.fixture
def harness(fake_mimoe: FakeMimoe, settings_tmp: Settings, tmp_path: Path) -> Harness:
    """An app on the 0.6 fake with qwen3-4b loaded and no web bundle."""
    return make_harness(fake_mimoe, settings_tmp, tmp_path / "dist")


# -- helpers -------------------------------------------------------------------------------------


def sse_events(text: str) -> list[dict[str, Any]]:
    """Decode an SSE body into ``[{"event": name, **data}, ...]``; comments (pings) are skipped."""
    events: list[dict[str, Any]] = []
    for frame in text.split("\r\n\r\n"):
        name: str | None = None
        data: list[str] = []
        for line in frame.split("\r\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].strip())
        if name is not None:
            events.append({"event": name, **json.loads("\n".join(data))})
    return events


async def chat(client: httpx.AsyncClient, thread_id: str, message: str) -> httpx.Response:
    return await client.post("/api/chat", json={"thread_id": thread_id, "message": message})


async def resume(
    client: httpx.AsyncClient, thread_id: str, interrupt_id: str, decisions: list[str]
) -> httpx.Response:
    body = {"thread_id": thread_id, "interrupt_id": interrupt_id, "decisions": decisions}
    return await client.post("/api/resume", json=body)


def names(events: list[dict[str, Any]]) -> list[str]:
    return [event["event"] for event in events]


def text_of(events: list[dict[str, Any]]) -> str:
    return "".join(event["text"] for event in events if event["event"] == "token")


def detail(resp: httpx.Response) -> dict[str, Any]:
    body = resp.json()["detail"]
    assert set(body) == {"message", "hint"}, body
    return body


def leftover_tasks() -> list[str]:
    """Names of tasks other than this one and sse-starlette's process-wide shutdown watcher."""
    return sorted(
        task.get_coro().__qualname__  # type: ignore[union-attr]
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and task.get_coro().__qualname__ != "_shutdown_watcher"  # type: ignore[union-attr]
    )


# -- streaming turns -----------------------------------------------------------------------------


async def test_plain_answer_streams_tokens_then_done(harness: Harness) -> None:
    async with harness.client() as client:
        await harness.prime()
        harness.fake.script({"content": "Hello there, friend."})
        resp = await chat(client, "t1", "hi")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert resp.headers["cache-control"] == "no-store"
        assert resp.text.startswith('event: token\r\ndata: {"text": "Hello"}\r\n\r\n')
        events = sse_events(resp.text)
        assert names(events) == ["token", "token", "token", "done"]
        assert text_of(events) == "Hello there, friend."
        done = events[-1]
        assert done["status"] == "completed"
        assert isinstance(done["elapsed_s"], float) and done["elapsed_s"] >= 0
        assert done["model"] == "qwen3-4b"
        assert done["usage_total"] == {"input_tokens": 120, "output_tokens": 3, "llm_calls": 1}
        body = harness.fake.calls[-1]
        assert body["stream"] is True and body["model"] == "qwen3-4b"
        assert body["messages"][-1] == {"role": "user", "content": "hi /no_think"}
        assert not harness.runtime.lock("t1").locked()
        assert leftover_tasks() == []


async def test_tool_round_trip(harness: Harness) -> None:
    async with harness.client() as client:
        await harness.prime()
        harness.fake.script(LIST_FILES, {"content": "There are files."})
        events = sse_events((await chat(client, "t1", "what is here?")).text)
        assert names(events) == ["tool_call", "tool_result", "token", "token", "token", "done"]
        assert events[0] == {
            "event": "tool_call",
            "id": "tool_0",
            "name": "list_files",
            "args": {"path": "."},
        }
        result = events[1]
        assert result["id"] == "tool_0" and result["name"] == "list_files"
        assert result["is_error"] is False
        assert "notes.md" in result["content"] and "sales.csv" in result["content"]
        assert text_of(events) == "There are files."
        assert events[-1]["usage_total"]["llm_calls"] == 2


async def test_approval_flow_approve(harness: Harness) -> None:
    async with harness.client() as client:
        await harness.prime()
        harness.fake.script(RUN_PYTHON, {"content": "6*7 is 42."})
        first = sse_events((await chat(client, "t1", "Use run_python to print 6*7")).text)
        assert names(first) == ["tool_call", "approval_required", "done"]
        assert first[0] == {
            "event": "tool_call",
            "id": "tool_0",
            "name": "run_python",
            "args": {"code": "print(6*7)"},
        }
        approval = first[1]
        assert re.fullmatch(r"[0-9a-f]{32}", approval["interrupt_id"])
        [request] = approval["action_requests"]
        assert request["name"] == "run_python" and request["args"] == {"code": "print(6*7)"}
        assert "```python\nprint(6*7)\n```" in request["description"]
        assert request["description"].endswith(APPROVAL_WARNING)
        assert approval["review_configs"] == [
            {"action_name": "run_python", "allowed_decisions": ["approve", "reject"]}
        ]
        assert first[-1]["status"] == "awaiting_approval"
        assert first[-1]["usage_total"]["llm_calls"] == 1
        assert not harness.runtime.lock("t1").locked()  # the user may take their time

        resp = await resume(client, "t1", approval["interrupt_id"], ["approve"])
        assert resp.status_code == 200
        second = sse_events(resp.text)
        assert names(second) == ["tool_result", "token", "token", "token", "done"]
        result = second[0]
        assert result["id"] == "tool_0" and result["name"] == "run_python"
        assert result["is_error"] is False and "stdout:\n42" in result["content"]
        assert text_of(second) == "6*7 is 42."
        assert second[-1]["status"] == "completed" and second[-1]["model"] == "qwen3-4b"
        assert len(harness.fake.calls) == 2

        again = await resume(client, "t1", approval["interrupt_id"], ["approve"])
        assert again.status_code == 409
        assert detail(again)["message"] == "nothing to resume on this thread"


async def test_approval_flow_reject(harness: Harness) -> None:
    async with harness.client() as client:
        await harness.prime()
        harness.fake.script(RUN_PYTHON, {"content": "I did not run it."})
        first = sse_events((await chat(client, "t1", "run it")).text)
        assert first[-1]["status"] == "awaiting_approval"
        second = sse_events((await resume(client, "t1", first[1]["interrupt_id"], ["reject"])).text)
        assert names(second) == ["tool_result", *["token"] * 5, "done"]
        result = second[0]
        assert result["id"] == "tool_0" and result["name"] == "run_python"
        assert result["is_error"] is True
        assert "not executed" in result["content"] and REJECT_MESSAGE in result["content"]
        assert text_of(second) == "I did not run it."
        assert second[-1]["status"] == "completed"
        sent = harness.fake.calls[-1]["messages"][-1]  # the model was told, nothing ran
        assert sent["role"] == "tool" and REJECT_MESSAGE in sent["content"]


async def test_run_config_carries_thread_and_model(harness: Harness) -> None:
    await harness.prime()
    pre = harness.runtime.ready.pre  # type: ignore[union-attr]
    assert harness.runtime.run_config("abc", pre) == {
        "configurable": {"thread_id": "abc"},
        "metadata": {"model": "qwen3-4b"},
    }


# -- locks and pending approvals -----------------------------------------------------------------


async def test_409_while_a_run_is_in_progress(harness: Harness) -> None:
    async with harness.client() as client:
        await harness.prime()
        harness.fake.script({"content": "Slow answer."})
        harness.transport.gate.clear()
        first = asyncio.create_task(chat(client, "t1", "hello"))
        await asyncio.wait_for(harness.transport.arrived.wait(), 5)
        assert harness.runtime.lock("t1").locked()
        assert harness.runtime.busy_threads() == ["t1"]

        second = await chat(client, "t1", "again")
        assert second.status_code == 409
        assert detail(second) == {
            "message": "a run is in progress on thread t1",
            "hint": HINT_RUN_IN_PROGRESS,
        }
        blocked = await resume(client, "t1", "0" * 32, ["approve"])
        assert blocked.status_code == 409 and "in progress" in detail(blocked)["message"]
        switch = await client.post("/api/model", json={"model": "qwen3-4b-instruct-2507"})
        assert switch.status_code == 409 and "in progress" in detail(switch)["message"]
        health = await client.get("/api/health")  # keeps answering meanwhile
        assert health.status_code == 200 and health.json()["mimoe_reachable"] is True

        harness.transport.gate.set()
        events = sse_events((await first).text)
        assert text_of(events) == "Slow answer." and events[-1]["status"] == "completed"
        assert not harness.runtime.lock("t1").locked()
        assert harness.fake.calls[-1]["messages"][-1]["content"] == "hello /no_think"


async def test_409_chat_while_an_approval_is_pending(harness: Harness) -> None:
    async with harness.client() as client:
        await harness.prime()
        harness.fake.script(RUN_PYTHON)
        first = sse_events((await chat(client, "t1", "run it")).text)
        assert first[-1]["status"] == "awaiting_approval"
        calls = len(harness.fake.calls)

        resp = await chat(client, "t1", "something else")
        assert resp.status_code == 409
        assert detail(resp) == {
            "message": "an approval is pending on this thread",
            "hint": HINT_APPROVAL_PENDING,
        }
        assert len(harness.fake.calls) == calls  # the message never reached the graph
        assert not harness.runtime.lock("t1").locked()

        harness.fake.script({"content": "Hi from t2."})  # other threads are unaffected
        other = sse_events((await chat(client, "t2", "hi")).text)
        assert text_of(other) == "Hi from t2."

        harness.fake.script({"content": "Ran."})  # the interrupt survived the refusal
        second = sse_events(
            (await resume(client, "t1", first[1]["interrupt_id"], ["approve"])).text
        )
        assert names(second)[0] == "tool_result" and second[-1]["status"] == "completed"


async def test_409_resume_with_nothing_pending_or_a_stale_id(harness: Harness) -> None:
    async with harness.client() as client:
        await harness.prime()
        fresh = await resume(client, "new-thread", "0" * 32, ["approve"])
        assert fresh.status_code == 409
        assert detail(fresh)["message"] == "nothing to resume on this thread"

        harness.fake.script({"content": "Hello."})
        assert sse_events((await chat(client, "t1", "hi")).text)[-1]["status"] == "completed"
        completed = await resume(client, "t1", "0" * 32, ["approve"])
        assert completed.status_code == 409
        assert detail(completed)["message"] == "nothing to resume on this thread"

        harness.fake.script(RUN_PYTHON)
        first = sse_events((await chat(client, "t1", "run it")).text)
        calls = len(harness.fake.calls)
        stale = await resume(client, "t1", "0" * 32, ["approve"])
        assert stale.status_code == 409
        assert detail(stale)["message"] == "stale interrupt id"
        assert len(harness.fake.calls) == calls  # no Command(resume) went to the graph

        harness.fake.script({"content": "Ran."})
        ok = await resume(client, "t1", first[1]["interrupt_id"], ["approve"])
        assert ok.status_code == 200 and sse_events(ok.text)[-1]["status"] == "completed"


async def test_422_bad_decisions(harness: Harness) -> None:
    async with harness.client() as client:
        await harness.prime()
        harness.fake.script(RUN_PYTHON)
        first = sse_events((await chat(client, "t1", "run it")).text)
        interrupt_id = first[1]["interrupt_id"]

        too_many = await resume(client, "t1", interrupt_id, ["approve", "approve"])
        assert too_many.status_code == 422
        assert detail(too_many)["message"] == "expected 1 decision(s), got 2"
        none = await resume(client, "t1", interrupt_id, [])
        assert none.status_code == 422
        edit = await resume(client, "t1", interrupt_id, ["edit"])
        assert edit.status_code == 422
        assert detail(edit)["message"] == "decision 1 must be one of: approve, reject; got 'edit'"
        assert not harness.runtime.lock("t1").locked()

        harness.fake.script({"content": "Ran."})  # nothing was consumed
        ok = await resume(client, "t1", interrupt_id, ["approve"])
        assert ok.status_code == 200 and sse_events(ok.text)[-1]["status"] == "completed"


async def test_two_requests_in_one_interrupt_take_one_decision_each(harness: Harness) -> None:
    async with harness.client() as client:
        await harness.prime()
        two = {
            "tool_calls": [
                {"name": "run_python", "args": {"code": "print(1)"}},
                {"name": "run_python", "args": {"code": "print(2)"}},
            ]
        }
        harness.fake.script(two, {"content": "Only the first ran."})
        first = sse_events((await chat(client, "t1", "run both")).text)
        assert names(first) == ["tool_call", "tool_call", "approval_required", "done"]
        assert [c["id"] for c in first[:2]] == ["tool_0", "tool_1"]
        approval = first[2]
        assert len(approval["action_requests"]) == 2

        short = await resume(client, "t1", approval["interrupt_id"], ["approve"])
        assert short.status_code == 422 and "expected 2" in detail(short)["message"]
        second = sse_events(
            (await resume(client, "t1", approval["interrupt_id"], ["approve", "reject"])).text
        )
        results = {e["id"]: e for e in second if e["event"] == "tool_result"}
        assert "stdout:\n1" in results["tool_0"]["content"]
        assert results["tool_0"]["is_error"] is False
        assert (
            results["tool_1"]["is_error"] is True and REJECT_MESSAGE in results["tool_1"]["content"]
        )
        assert text_of(second) == "Only the first ran."


async def test_resumed_runs_keep_tool_call_ids_unique_within_the_turn(harness: Harness) -> None:
    """0.6 engines answer every model call with ``tool_0``. The web UI keys its tool cards by id
    within one assistant turn and gives a result to every card with its id, so a call repeated
    in a resumed run (the model retries after an approval) needs an alias, the alias must hold
    in the run after that (where the result arrives), and a new chat turn starts over."""
    run_again = {"tool_calls": [{"name": "run_python", "args": {"code": "print(2)"}}]}
    async with harness.client() as client:
        await harness.prime()
        harness.fake.script(RUN_PYTHON, run_again, {"content": "42 then 2."})
        first = sse_events((await chat(client, "t1", "run twice")).text)
        assert names(first) == ["tool_call", "approval_required", "done"]
        assert first[0]["id"] == "tool_0"
        second = sse_events(
            (await resume(client, "t1", first[1]["interrupt_id"], ["approve"])).text
        )
        assert names(second) == ["tool_result", "tool_call", "approval_required", "done"]
        assert second[0]["id"] == "tool_0" and "stdout:\n42" in second[0]["content"]
        assert second[1]["id"] == "tool_0#2" and second[1]["args"] == {"code": "print(2)"}
        third = sse_events(
            (await resume(client, "t1", second[2]["interrupt_id"], ["approve"])).text
        )
        assert names(third)[0] == "tool_result" and third[-1]["status"] == "completed"
        assert third[0]["id"] == "tool_0#2" and "stdout:\n2" in third[0]["content"]

        harness.fake.script(RUN_PYTHON, LIST_FILES, {"content": "Listed."})
        first = sse_events((await chat(client, "t1", "run, then list")).text)
        assert first[0]["id"] == "tool_0"  # a new turn: the UI's cards start over too
        second = sse_events(
            (await resume(client, "t1", first[1]["interrupt_id"], ["approve"])).text
        )
        assert [(e["event"], e["id"]) for e in second if "id" in e] == [
            ("tool_result", "tool_0"),
            ("tool_call", "tool_0#2"),  # an ungated call in the resumed run
            ("tool_result", "tool_0#2"),
        ]
        assert text_of(second) == "Listed."


async def test_two_threads_stream_concurrently(harness: Harness) -> None:
    """The lock is per thread: a second conversation reaches the model while the first is still
    inside its model call, and both runs complete."""
    async with harness.client() as client:
        await harness.prime()
        arrivals: list[str] = []
        inner = harness.transport.handle_async_request

        async def counting(request: httpx.Request) -> httpx.Response:
            arrivals.append(json.loads(request.content)["messages"][-1]["content"])
            return await inner(request)

        harness.transport.handle_async_request = counting  # type: ignore[method-assign]
        harness.fake.script({"content": "One."}, {"content": "Two."})
        harness.transport.gate.clear()
        first = asyncio.create_task(chat(client, "t1", "one"))
        second = asyncio.create_task(chat(client, "t2", "two"))
        for _ in range(500):
            if len(arrivals) == 2:
                break
            await asyncio.sleep(0.01)
        assert sorted(arrivals) == ["one /no_think", "two /no_think"]
        assert sorted(harness.runtime.busy_threads()) == ["t1", "t2"]
        harness.transport.gate.set()
        texts = {text_of(sse_events((await task).text)) for task in (first, second)}
        assert texts == {"One.", "Two."}
        assert harness.runtime.busy_threads() == []
        assert leftover_tasks() == []


async def test_409_while_a_model_switch_is_in_progress(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """While ``switching`` is set, chat and resume answer 409, a second switch is refused, health
    reports the switch without waiting for it, and the switch itself completes afterwards."""
    hold = threading.Event()
    real_switch = server.switch_model

    def slow_switch(client: MimoeClient, model_id: str, **kwargs: Any) -> Any:
        assert hold.wait(10)  # a worker thread, like the real load
        return real_switch(client, model_id, **kwargs)

    monkeypatch.setattr(server, "switch_model", slow_switch)
    async with harness.client() as client:
        await harness.prime()
        switching = asyncio.create_task(
            client.post("/api/model", json={"model": "qwen3-4b-instruct-2507"})
        )
        for _ in range(500):
            if harness.runtime.switching:
                break
            await asyncio.sleep(0.01)
        assert harness.runtime.switching is True
        blocked = await chat(client, "t1", "hi")
        assert blocked.status_code == 409
        assert detail(blocked) == {
            "message": "a model switch is in progress",
            "hint": HINT_SWITCH_IN_PROGRESS,
        }
        assert (await resume(client, "t1", "0" * 32, ["approve"])).status_code == 409
        again = await client.post("/api/model", json={"model": "smollm2-360m"})
        assert again.status_code == 409 and "already in progress" in detail(again)["message"]
        health = (await client.get("/api/health")).json()
        assert health["error"] == STATUS_SWITCHING and health["model"] == "qwen3-4b"
        assert not harness.runtime.lock("t1").locked()

        hold.set()
        switched = await switching
        assert switched.status_code == 200
        assert switched.json()["model"]["id"] == "qwen3-4b-instruct-2507"
        assert harness.runtime.switching is False
        harness.fake.script({"content": "Switched."})
        assert text_of(sse_events((await chat(client, "t1", "hi")).text)) == "Switched."


async def test_client_disconnect_releases_the_lock_and_closes_the_run(harness: Harness) -> None:
    """The browser goes away mid-run: sse-starlette cancels the response, the graph run is torn
    down (no task left behind) and the thread lock is free for the next message."""
    await harness.prime()
    harness.fake.script({"content": "Never delivered."})
    harness.transport.gate.clear()
    payload = json.dumps({"thread_id": "t1", "message": "hello"}).encode()
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/chat",
        "raw_path": b"/api/chat",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"127.0.0.1"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(payload)).encode()),
        ],
        "client": ("127.0.0.1", 40000),
        "server": ("127.0.0.1", 8000),
    }
    sent: list[dict[str, Any]] = []
    body_sent = False

    async def receive() -> dict[str, Any]:
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {"type": "http.request", "body": payload, "more_body": False}
        await harness.transport.arrived.wait()  # the run is inside its model call: hang up
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await asyncio.wait_for(harness.app(scope, receive, send), 10)
    assert sent[0]["type"] == "http.response.start" and sent[0]["status"] == 200
    assert not harness.runtime.lock("t1").locked()
    assert leftover_tasks() == []

    harness.transport.gate.set()  # the held request was cancelled, so its reply is still queued
    async with harness.client() as client:
        events = sse_events((await chat(client, "t1", "again")).text)
        assert text_of(events) == "Never delivered." and events[-1]["status"] == "completed"
    users = [m["content"] for m in harness.fake.calls[-1]["messages"] if m["role"] == "user"]
    assert users == ["hello", "again /no_think"]  # the cut turn's message stayed checkpointed


# -- readiness and health ------------------------------------------------------------------------


async def test_chat_before_the_engine_is_ready_answers_with_an_error_event(
    harness: Harness,
) -> None:
    harness.fake.down = True
    async with harness.client() as client:
        resp = await chat(client, "t1", "hi")
        assert resp.status_code == 200
        [error] = sse_events(resp.text)
        assert error["event"] == "error" and error["message"] == "mimOE Studio is not reachable"
        assert error["hint"].startswith(HINT_NOT_REACHABLE)
        assert harness.runtime.ready is None
        assert harness.runtime.last_error == (error["message"], error["hint"])
        assert not harness.runtime.lock("t1").locked()
        health = (await client.get("/api/health")).json()
        assert health["mimoe_reachable"] is False and health["error"].startswith(error["message"])

        harness.fake.down = False  # the next chat retries the preflight and then answers
        events = sse_events((await chat(client, "t1", "hi")).text)
        assert text_of(events) == "OK." and events[-1]["status"] == "completed"
        assert harness.runtime.ready is not None and harness.runtime.last_error is None


async def test_health_while_the_engine_is_down_and_after_it_comes_back(harness: Harness) -> None:
    harness.fake.down = True
    async with harness.client() as client:
        down = (await client.get("/api/health")).json()
        assert set(down) == HEALTH_KEYS
        assert down["mimoe_reachable"] is False and down["model"] is None
        assert down["mode"] == "chat_only" and down["generation"] is None
        assert down["error"].startswith("mimOE Studio is not reachable. ")
        assert HINT_NOT_REACHABLE in down["error"]
        assert down["workspace"] == str(harness.settings.workspace)
        assert down["approval"] == "manual" and down["network"] == "off"
        assert harness.runtime.ready is None

        harness.fake.down = False
        up = (await client.get("/api/health")).json()
        assert set(up) == HEALTH_KEYS
        assert up["mimoe_reachable"] is True and up["error"] is None
        assert up["model"] == "qwen3-4b" and up["mode"] == "tools"
        assert up["generation"] == "0.6" and up["engine_version"] == "v3.22.8 (developer edition)"
        assert up["node"] == "fake-node" and up["max_context"] == 12000
        assert isinstance(up["tokens_per_second"], float) and up["tokens_per_second"] > 0
        assert harness.runtime.ready is not None

        harness.fake.down = True  # gone again after the build: the badge says so, the agent stays
        gone = (await client.get("/api/health")).json()
        assert gone["mimoe_reachable"] is False and gone["model"] == "qwen3-4b"
        assert gone["mode"] == "tools" and "not reachable" in gone["error"]
        assert harness.runtime.ready is not None


async def test_health_reports_a_missing_model_and_never_raises(
    settings_tmp: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeMimoe("0.6")
    fake.loaded.clear()
    harness = make_harness(fake, settings_tmp, tmp_path / "dist")
    async with harness.client() as client:
        empty = (await client.get("/api/health")).json()
        assert empty["mimoe_reachable"] is False
        assert empty["error"] == f"no model is loaded. {HINT_NO_MODEL}"

        fake.loaded.append("qwen3-4b")
        assert (await client.get("/api/health")).json()["model"] == "qwen3-4b"
        fake.loaded.clear()  # unloaded behind our back
        unloaded = (await client.get("/api/health")).json()
        assert unloaded["mimoe_reachable"] is True
        assert unloaded["error"] == f"qwen3-4b is no longer loaded. {HINT_NO_MODEL}"

        def boom() -> list[Any]:
            raise RuntimeError("boom")

        monkeypatch.setattr(harness.runtime.client, "loaded_models", boom)
        broken = await client.get("/api/health")
        assert broken.status_code == 200
        assert broken.json()["error"].startswith("RuntimeError: boom")


async def test_approval_and_network_flags_are_reported(
    fake_mimoe: FakeMimoe, settings_tmp: Settings, tmp_path: Path
) -> None:
    import dataclasses

    settings = dataclasses.replace(settings_tmp, auto_approve=True, allow_network=True)
    harness = make_harness(fake_mimoe, settings, tmp_path / "dist")
    async with harness.client() as client:
        health = (await client.get("/api/health")).json()
        assert health["approval"] == "auto" and health["network"] == "on"
        harness.fake.script(RUN_PYTHON, {"content": "42"})  # trust-the-model mode: no interrupt
        events = sse_events((await chat(client, "t1", "run it")).text)
        assert names(events) == ["tool_call", "tool_result", "token", "done"]
        assert "stdout:\n42" in events[1]["content"]


# -- host check, static files, unknown routes ----------------------------------------------------


async def test_bad_host_header_is_rejected(harness: Harness) -> None:
    async with harness.client() as client:
        for host in ("evil.example", "evil.example:8000", "192.168.1.5"):
            resp = await client.get("/api/health", headers={"Host": host})
            assert resp.status_code == 400, host
            assert resp.text == "Invalid host header"
        for host in ("127.0.0.1", "127.0.0.1:8000", "localhost:8000", "[::1]:8000"):
            assert (await client.get("/api/health", headers={"Host": host})).status_code == 200


async def test_static_bundle_is_served_after_the_api_routes(
    fake_mimoe: FakeMimoe, settings_tmp: Settings, tmp_path: Path
) -> None:
    dist = tmp_path / "web" / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>ui</title>", encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    harness = make_harness(fake_mimoe, settings_tmp, dist)
    async with harness.client() as client:
        root = await client.get("/")
        assert root.status_code == 200 and "<title>ui</title>" in root.text
        assert root.headers["content-type"].startswith("text/html")
        asset = await client.get("/assets/app.js")
        assert asset.status_code == 200 and "javascript" in asset.headers["content-type"]
        assert (await client.get("/api/health")).status_code == 200
        missing = await client.get("/api/nope")
        assert missing.status_code == 404
        assert detail(missing)["message"] == "no such endpoint: /api/nope"
        assert (await client.post("/api/nope/deeper", json={})).status_code == 404
        assert (await client.get("/nope.txt")).status_code == 404


async def test_json_hint_without_a_web_bundle(harness: Harness) -> None:
    async with harness.client() as client:
        root = await client.get("/")
        assert root.status_code == 200
        assert root.json()["message"] == "the web UI is not built"
        assert "npm run build" in root.json()["hint"]
        missing = await client.get("/api/nope")
        assert missing.status_code == 404 and "no such endpoint" in detail(missing)["message"]
        assert (await client.get("/api/health/extra")).status_code == 404


async def test_validation_errors_use_the_detail_shape(harness: Harness) -> None:
    async with harness.client() as client:
        missing = await client.post("/api/chat", json={"thread_id": "t1"})
        assert missing.status_code == 422
        assert detail(missing)["message"].startswith("invalid request: message:")
        blank_thread = await client.post("/api/chat", json={"thread_id": "", "message": "hi"})
        assert blank_thread.status_code == 422
        empty = await client.post("/api/chat", json={"thread_id": "t1", "message": "   "})
        assert empty.status_code == 422 and detail(empty)["message"] == "message is empty"
        no_id = await client.post("/api/resume", json={"thread_id": "t1", "decisions": []})
        assert no_id.status_code == 422 and "interrupt_id" in detail(no_id)["message"]
        assert not harness.runtime.lock("t1").locked()


# -- models --------------------------------------------------------------------------------------


async def test_models_listing_and_switching(harness: Harness) -> None:
    async with harness.client() as client:
        await harness.prime()
        listing = (await client.get("/api/models")).json()
        assert set(listing) == {"loaded", "registry", "current"}
        assert [m["id"] for m in listing["loaded"]] == ["qwen3-4b"]
        assert listing["loaded"][0]["max_context"] == 12000
        assert listing["loaded"][0]["supports_tools"] is None  # 0.6 engines do not advertise
        registry = {m["id"]: m for m in listing["registry"]}
        assert {"qwen3-4b", "qwen3-4b-instruct-2507", "smollm2-360m"} <= set(registry)
        assert registry["smollm2-360m"] == {
            "id": "smollm2-360m",
            "kind": "llm",
            "ready": True,
            "size_bytes": DEFAULT_SIZE,
            "raw": registry["smollm2-360m"]["raw"],
        }
        assert listing["current"] == "qwen3-4b"

        harness.fake.script({"content": "Before."})
        assert text_of(sse_events((await chat(client, "t1", "remember me")).text)) == "Before."

        bad = await client.post("/api/model", json={"model": "../etc"})
        assert bad.status_code == 422 and "not a valid model id" in detail(bad)["message"]
        unknown = await client.post("/api/model", json={"model": "nope-1b"})
        assert unknown.status_code == 502
        assert "not in the model registry" in detail(unknown)["message"]
        assert "pull" in detail(unknown)["hint"]
        assert harness.runtime.ready is not None  # a failed switch leaves the agent alone
        assert harness.runtime.ready.pre.model.id == "qwen3-4b"
        assert harness.runtime.switching is False

        switched = await client.post(
            "/api/model", json={"model": "qwen3-4b-instruct-2507", "unload_previous": True}
        )
        assert switched.status_code == 200
        body = switched.json()
        assert set(body) == {"model", "probe"}
        assert body["model"]["id"] == "qwen3-4b-instruct-2507" and body["model"]["kind"] == "llm"
        assert body["probe"]["tools_ok"] is True
        assert "structured ping call" in body["probe"]["detail"]
        assert isinstance(body["probe"]["latency_s"], float)
        assert harness.fake.loaded == ["qwen3-4b-instruct-2507"]  # the previous model went
        health = (await client.get("/api/health")).json()
        assert health["model"] == "qwen3-4b-instruct-2507" and health["mode"] == "tools"
        assert (await client.get("/api/models")).json()["current"] == "qwen3-4b-instruct-2507"

        harness.fake.script({"content": "New model here."})  # rebuilt agent, same threads
        events = sse_events((await chat(client, "t1", "hi")).text)
        assert text_of(events) == "New model here."
        assert events[-1]["model"] == "qwen3-4b-instruct-2507"
        sent = harness.fake.calls[-1]
        assert sent["model"] == "qwen3-4b-instruct-2507"
        assert [m["content"] for m in sent["messages"] if m["role"] == "user"] == [
            "remember me",
            "hi /no_think",
        ]

        harness.fake.tools_ok = False  # a model that fails the probe: chat-only mode
        kept = await client.post(
            "/api/model", json={"model": "smollm2-360m", "unload_previous": False}
        )
        assert kept.status_code == 200
        assert kept.json()["probe"]["tools_ok"] is False
        assert "without a tool call" in kept.json()["probe"]["detail"]
        assert sorted(harness.fake.loaded) == ["qwen3-4b-instruct-2507", "smollm2-360m"]
        assert (await client.get("/api/health")).json()["mode"] == "chat_only"


async def test_models_reports_engine_errors_as_502(harness: Harness) -> None:
    async with harness.client() as client:
        harness.fake.down = True
        resp = await client.get("/api/models")
        assert resp.status_code == 502
        assert detail(resp)["message"] == "mimOE Studio is not reachable"
        assert detail(resp)["hint"].startswith(HINT_NOT_REACHABLE)
        switch = await client.post("/api/model", json={"model": "smollm2-360m"})
        assert switch.status_code == 502 and "not reachable" in detail(switch)["message"]
        assert harness.runtime.switching is False


# -- construction and run ------------------------------------------------------------------------


def test_create_app_defaults(settings_tmp: Settings) -> None:
    app = create_app(settings_tmp)  # a real MimoeClient; nothing is called yet
    runtime = app.state.runtime
    assert isinstance(runtime, AgentRuntime) and isinstance(runtime.client, MimoeClient)
    assert runtime.ready is None and runtime.model_id is None
    assert server.DEFAULT_WEB_DIST.name == "dist" and server.DEFAULT_WEB_DIST.parent.name == "web"


def test_run_binds_loopback_only(settings_tmp: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Any, dict[str, Any]]] = []
    monkeypatch.setattr(server.uvicorn, "run", lambda app, **kwargs: calls.append((app, kwargs)))
    server.run(settings_tmp, 8123)
    [(app, kwargs)] = calls
    assert isinstance(app, FastAPI)
    assert kwargs["host"] == "127.0.0.1" and kwargs["port"] == 8123


def test_sse_frame_shape() -> None:
    frame = sse_frame({"event": "token", "text": "hi"})
    assert frame == {"event": "token", "data": '{"text": "hi"}'}
    assert json.loads(sse_frame({"event": "done", "elapsed_s": 1.5})["data"]) == {"elapsed_s": 1.5}


def test_default_handlers_are_the_shared_ones() -> None:
    """The seams default to the package's own pieces (a typed check, no network)."""
    from mimoe_agent.llm import make_model as default_factory

    factory: Callable[..., Any] = default_factory
    assert factory is make_model
