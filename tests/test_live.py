"""Live smoke tests against a running mimOE Studio.

Skipped unless ``MIMOE_LIVE=1`` (and selected with ``-m live``, which ``addopts`` excludes by
default). Budget: at most seven completions in total: one REPL session with one prompt (probe +
tool call + answer) and two replayed steps (probe + one completion each); the preflight test
skips the probe with ``force_tools``.

``MIMOE_BASE_URL`` / ``MIMOE_API_KEY`` are read at import time because the autouse
``_clean_mimoe_env`` fixture strips every ``MIMOE_*`` variable before a test runs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from langchain_core.utils.function_calling import convert_to_openai_tool
from typer.testing import CliRunner

from mimoe_agent.agent import system_prompt
from mimoe_agent.cli import app
from mimoe_agent.config import load_settings
from mimoe_agent.mimoe import EngineGeneration, MimoeClient, preflight
from mimoe_agent.tools import build_tools
from mimoe_agent.tools.system import calculate

LIVE = os.environ.get("MIMOE_LIVE") == "1"
BASE_URL = os.environ.get("MIMOE_BASE_URL") or "http://127.0.0.1:8083/mimik-ai/openai/v1"
API_KEY = os.environ.get("MIMOE_API_KEY") or "1234"
PROMPT = "What files are in this workspace?"

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not LIVE, reason="set MIMOE_LIVE=1 and run `pytest -m live`"),
]


def test_preflight_against_the_real_engine(workspace_tmp: Path) -> None:
    """Discovery, model listing and thinking control; no completion (``force_tools``)."""
    settings = load_settings(
        {"workspace": workspace_tmp, "base_url": BASE_URL, "api_key": API_KEY, "force_tools": True},
        cwd=workspace_tmp.parent,
    )
    client = MimoeClient(settings.base_url, settings.api_key)
    try:
        pre = preflight(settings, client=client)
    finally:
        client.close()
    assert pre.engine.base_url == BASE_URL
    assert pre.engine.generation in (EngineGeneration.V06, EngineGeneration.V10)
    assert pre.model.kind == "llm" and pre.model.max_context
    assert pre.probe is None and pre.tools_enabled
    assert pre.thinking_control in ("native", "soft", "none")
    assert any("--force-tools" in warning for warning in pre.warnings)


def test_repl_session_with_one_prompt(workspace_tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Probe + one tool-using turn through the CLI (three completions at most)."""
    monkeypatch.setenv("COLUMNS", "120")
    result = CliRunner().invoke(
        app,
        [
            "--workspace",
            str(workspace_tmp),
            "--base-url",
            BASE_URL,
            "--api-key",
            API_KEY,
            "--auto-approve",
        ],
        input=f"{PROMPT}\n/quit\n",
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    out = result.output
    assert "mimoe-agent 0.1.0" in out and "tokens/s" in out
    assert f"you> {PROMPT}" in out
    # qwen3-4b answers this with list_files (6/6 in the matrix); another model loaded by a
    # concurrent job on the same engine can answer without a tool, hence the output in the message
    assert "-> " in out, f"no tool call in the session:\n{out}"
    assert "sales.csv" in out or "notes.md" in out, out
    assert "elapsed" in out and "model call" in out


PRIMES = "sum(1 for n in range(1000, 4501) if all(n % i != 0 for i in range(2, int(n**0.5) + 1)))"
PREAMBLE = (
    "I'll calculate the number of prime numbers between 1000 and 4500 inclusive. This requires a "
    "mathematical computation rather than file inspection or git operations.\n\n\n"
)
MULTIPLES = "What is the sum of every multiplier of 5 between 432 and 43229?"


def _next_step(workspace_tmp: Path, messages: list[dict[str, Any]]) -> dict[str, Any]:
    """The model's reply to ``messages`` after the agent's system prompt, with the agent's tools
    and temperature (probe + one completion)."""
    settings = load_settings(
        {"workspace": workspace_tmp, "base_url": BASE_URL, "api_key": API_KEY},
        cwd=workspace_tmp.parent,
    )
    client = MimoeClient(settings.base_url, settings.api_key)
    try:
        pre = preflight(settings, client=client)
        reply = client.chat(
            {
                "model": pre.model.id,
                "messages": [
                    {"role": "system", "content": system_prompt(settings, pre)},
                    *messages,
                ],
                "tools": [convert_to_openai_tool(t) for t in build_tools(settings, client)],
                "temperature": 0,
                "max_tokens": 400,
            }
        )
    finally:
        client.close()
    return reply["choices"][0]["message"]


def _tool_names(message: dict[str, Any]) -> list[str]:
    return [call["function"]["name"] for call in message.get("tool_calls") or []]


def test_the_real_model_moves_from_a_refused_calculator_call_to_run_python(
    workspace_tmp: Path,
) -> None:
    """Replays a real web-session step: after the calculator refused a generator expression,
    qwen3-4b-instruct-2507 resent it (3 of 3 runs with the old "unsupported syntax" text) and
    then guessed a number. With the error that names run_python, its next step is run_python."""
    call = {"name": "calculator", "arguments": json.dumps({"expression": PRIMES})}
    message = _next_step(
        workspace_tmp,
        [
            {
                "role": "user",
                "content": "How many prime numbers are there between 1000 and 4500 "
                "(both inclusive)? /no_think",
            },
            {
                "role": "assistant",
                "content": PREAMBLE,
                "tool_calls": [{"id": "tool_0", "type": "function", "function": call}],
            },
            {
                "role": "tool",
                "tool_call_id": "tool_0",
                "name": "calculator",
                "content": calculate(PRIMES),
            },
        ],
    )
    assert _tool_names(message) == ["run_python"], message


def test_the_real_model_computes_numbers_with_a_tool(workspace_tmp: Path) -> None:
    """A real web-session question: qwen3-4b-instruct-2507 worked the series out in text and
    answered 186,000,000 (the sum is 186,842,970). With the system prompt's arithmetic rule its
    first step is a tool call."""
    message = _next_step(workspace_tmp, [{"role": "user", "content": f"{MULTIPLES} /no_think"}])
    names = _tool_names(message)
    assert names and set(names) <= {"calculator", "run_python"}, message
