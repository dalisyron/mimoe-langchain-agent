"""The typer CLI against :class:`conftest.FakeMimoe` through ``CliRunner``: banner, scripted REPL
sessions (plain answers, approvals, slash commands), the ``models`` group, ``serve`` and the exit
codes. ``MimoeClient`` and the chat model are pointed at the fake transport, so nothing here opens
a socket, forks or sends a signal (Ctrl-C is simulated with ``KeyboardInterrupt``). The terminal
path (the prompt_toolkit line reader and approval menu) runs on a :class:`Session` with keys from
prompt_toolkit's pipe input."""

from __future__ import annotations

import _thread
import io
import json
import os
import re
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import Result
from conftest import FakeMimoe
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import PipeInput, create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax
from typer.testing import CliRunner

from mimoe_agent import cli, llm, terminal
from mimoe_agent.agent import APPROVAL_WARNING
from mimoe_agent.cli import CANCELLED_RESULT, NO_ANSWER, REJECT_MESSAGE, Session, app
from mimoe_agent.config import load_settings
from mimoe_agent.mimoe import MimoeClient, preflight
from mimoe_agent.models import DEFAULT_PRESET

Run = Callable[..., Result]

PING = {"tool_calls": [{"name": "ping", "args": {}}]}
"""The start-up probe consumes the first scripted turn; ``script()`` queues this one first."""
LIST_FILES = {"tool_calls": [{"name": "list_files", "args": {"path": "."}}]}
RUN_PYTHON = {"tool_calls": [{"name": "run_python", "args": {"code": "print(6*7)"}}]}
DEMO_PROMPT = "What files are in this workspace?"


def script(fake: FakeMimoe, *turns: dict[str, Any]) -> None:
    """Queue the probe reply and then ``turns`` (the REPL runs the probe before the first turn)."""
    fake.script(PING, *turns)


@pytest.fixture
def run(fake_mimoe: FakeMimoe, workspace_tmp: Path, monkeypatch: pytest.MonkeyPatch) -> Run:
    """Invoke the CLI with the engine client and the chat model bound to the fake transport."""
    real_make_model = llm.make_model

    def fake_client(base_url: str | None, api_key: str, **kwargs: Any) -> MimoeClient:
        return MimoeClient(base_url, api_key, client=fake_mimoe.client())

    def fake_make_model(settings: Any, pre: Any, **kwargs: Any) -> Any:
        return real_make_model(
            settings,
            pre,
            http_client=fake_mimoe.client(),
            http_async_client=fake_mimoe.async_client(),
        )

    monkeypatch.setattr(cli, "MimoeClient", fake_client)
    monkeypatch.setattr(llm, "make_model", fake_make_model)
    monkeypatch.setenv("COLUMNS", "200")  # rich wraps at COLUMNS when stdout is not a terminal
    monkeypatch.chdir(workspace_tmp.parent)  # ./workspace is the default, and there is no .env
    runner = CliRunner()

    def invoke(*args: str, input: str | None = None) -> Result:
        return runner.invoke(
            app, ["--base-url", fake_mimoe.base_url, *args], input=input, catch_exceptions=False
        )

    return invoke


def _tool_messages(fake: FakeMimoe) -> list[dict[str, Any]]:
    """The ``role: tool`` messages of the last request the fake received."""
    return [m for m in fake.calls[-1]["messages"] if m["role"] == "tool"]


# -- start-up ------------------------------------------------------------------------------------


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Help text without ANSI styling (typer forces a styled terminal on GitHub Actions)."""
    return ANSI_RE.sub("", text)


def test_help_and_version(run: Run) -> None:
    result = run("--help")
    assert result.exit_code == 0
    output = _plain(result.output)
    for flag in ("--workspace", "--model", "--think", "--auto-approve", "--allow-network"):
        assert flag in output
    assert "serve" in output and "models" in output

    models_help = run("models", "--help")
    assert models_help.exit_code == 0
    for command in ("list", "pull", "use", "unload"):
        assert command in _plain(models_help.output)


def test_help_survives_a_forced_styled_terminal(run: Run, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reproduces CI: typer.rich_utils.FORCE_TERMINAL is True when GITHUB_ACTIONS is set."""
    import typer.rich_utils

    monkeypatch.setattr(typer.rich_utils, "FORCE_TERMINAL", True)
    result = run("--help")
    assert result.exit_code == 0
    assert "--workspace" in _plain(result.output)

    assert run("--version").output.strip() == "mimoe-agent 0.1.0"


def test_banner_shows_settings_with_sources(
    run: Run, fake_mimoe: FakeMimoe, workspace_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MIMOE_ALLOW_NETWORK", "1")
    result = run("--think", "--auto-approve", input="/quit\n")
    assert result.exit_code == 0
    out = result.output
    assert "mimoe-agent 0.1.0" in out
    assert "qwen3-4b  (default: first loaded)" in out
    assert "35.4 tokens/s" in out and "12,000 tokens" in out
    assert "v3.22.8 (developer edition), 0.6-generation API" in out
    assert "fake-node" in out
    assert f"{fake_mimoe.base_url}  (flag)" in out
    assert f"{workspace_tmp}  (default)" in out
    assert "auto: model-written Python runs without asking  (flag)" in out
    assert "allowed for run_python  (MIMOE_ALLOW_NETWORK env)" in out
    assert "thinking   on  (flag)" in out
    assert "tools      on" in out
    assert "probe: qwen3-4b answered with a structured ping call" in out
    for prompt in cli.demo_prompts(workspace_tmp):
        assert prompt in out
    assert DEMO_PROMPT in out and "Summarize notes.md" in out


def test_banner_chat_only_warning(run: Run, fake_mimoe: FakeMimoe) -> None:
    fake_mimoe.tools_ok = False
    result = run(input="/quit\n")
    assert result.exit_code == 0
    assert "tools      off: chat-only mode" in result.output
    assert "! tool probe failed" in result.output and "chat-only mode" in result.output


def test_demo_prompts_come_from_the_workspace_readme(workspace_tmp: Path, tmp_path: Path) -> None:
    prompts = cli.demo_prompts(workspace_tmp)
    assert len(prompts) == 5 and prompts[0] == DEMO_PROMPT
    assert cli.demo_prompts(tmp_path) == []  # no README, no prompts, no crash


def test_config_error_exits_1_with_hint(run: Run, tmp_path: Path) -> None:
    result = run("--workspace", str(tmp_path / "missing"))
    assert result.exit_code == 1
    assert "workspace is not a directory" in result.output
    assert "pass --workspace PATH" in result.output


def test_preflight_error_exits_1_with_hint(run: Run, fake_mimoe: FakeMimoe) -> None:
    fake_mimoe.down = True
    result = run()
    assert result.exit_code == 1
    assert "error: mimOE Studio is not reachable" in result.output
    assert "Open mimOE Studio" in result.output


def test_bad_api_key_exits_1(run: Run) -> None:
    result = run("--api-key", "wrong")
    assert result.exit_code == 1
    assert "rejected the API key" in result.output and "--api-key KEY" in result.output


def test_tracing_env_is_forced_off(run: Run, fake_mimoe: FakeMimoe) -> None:
    script(fake_mimoe)
    assert run(input="/quit\n").exit_code == 0
    assert os.environ.get("LANGSMITH_TRACING") == "false"


# -- turns ---------------------------------------------------------------------------------------


def test_piped_session_with_a_plain_answer(run: Run, fake_mimoe: FakeMimoe) -> None:
    script(fake_mimoe, {"content": "Hello there."})
    result = run(input="hi\n/quit\n")
    assert result.exit_code == 0
    assert "you> hi" in result.output  # the line is echoed when stdin is not a terminal
    assert "Hello there." in result.output
    assert re.search(r"elapsed \d+\.\d s, 1 model call, \d+ tokens", result.output)
    assert fake_mimoe.calls[-1]["messages"][-1]["content"] == "hi /no_think"


def test_eof_without_quit_exits_0(run: Run, fake_mimoe: FakeMimoe) -> None:
    script(fake_mimoe, {"content": "Bye."})
    result = run(input="hi\n")
    assert result.exit_code == 0 and "Bye." in result.output


def test_tool_round_trip_is_rendered_and_truncated(run: Run, fake_mimoe: FakeMimoe) -> None:
    script(fake_mimoe, LIST_FILES, {"content": "Listed."}, LIST_FILES, {"content": "Again."})
    result = run(input="files\n/verbose\nfiles\n/quit\n")
    assert result.exit_code == 0
    first, second = result.output.split("verbose on")
    assert "-> list_files(path='.')" in first
    assert "README.md" in first and "notes.md" in first
    assert "src/utils.py" not in first  # the fourth line onwards is hidden
    assert "(+3 more lines; /verbose shows all)" in first
    assert "Listed." in first
    assert "src/utils.py" in second and "/verbose shows all" not in second
    assert "2 model calls" in second


def test_a_failed_tool_call_is_one_line_until_verbose(run: Run, fake_mimoe: FakeMimoe) -> None:
    """The error is for the model, which reads it and adjusts; the reader gets one plain line."""
    # input() is no calculation: the calculator refuses it, however much else it accepts
    code = {"tool_calls": [{"name": "calculator", "args": {"expression": "input()"}}]}
    script(fake_mimoe, code, {"content": "Counted."}, code, {"content": "Again."})
    result = run(input="count\n/verbose\ncount\n/quit\n")
    assert result.exit_code == 0
    first, second = result.output.split("verbose on")
    assert "-> calculator(expression='input()')" in first
    assert f"   {cli.FAILED_LINE}" in first and "Counted." in first
    error = _tool_messages(fake_mimoe)[-1]["content"]
    assert error.startswith("ERROR: ") and "call run_python" in error  # the model still reads why
    assert error[:30] not in first
    assert error[:30] in second and cli.FAILED_LINE not in second


def test_empty_answer_message(run: Run, fake_mimoe: FakeMimoe) -> None:
    script(fake_mimoe, {"content": ""})
    result = run(input="hi\n/quit\n")
    assert result.exit_code == 0 and NO_ANSWER in result.output


def test_error_event_prints_message_and_hint(run: Run, fake_mimoe: FakeMimoe) -> None:
    # --force-tools skips the probe, so the queued failure hits the turn's model call
    fake_mimoe.fail_next(500, {"message": "llama_decode() failed", "statusCode": 500})
    result = run("--force-tools", input="hi\n/quit\n")
    assert result.exit_code == 0  # the REPL survives a failed turn
    assert "error: the conversation exceeded the model's context window" in result.output
    assert "/new" in result.output
    assert "elapsed" not in result.output  # no done line after an error


def test_think_flag_shows_thinking_collapsed_then_verbose(run: Run, fake_mimoe: FakeMimoe) -> None:
    script(
        fake_mimoe,
        {"content": "Hi!", "reasoning": "the user greets me\nso I greet back"},
        {"content": "Hi again!", "reasoning": "still greeting"},
    )
    result = run("--think", input="hi\n/verbose\nhi\n/quit\n")
    assert result.exit_code == 0
    first, second = result.output.split("verbose on")
    assert "thought (34 chars): the user greets me" in first
    assert "so I greet back" not in first  # collapsed to the first line
    assert "thinking:" in second and "still greeting" in second
    assert "Hi!" in first and "Hi again!" in second
    assert fake_mimoe.calls[-1]["messages"][0]["content"].startswith("You are")  # no /no_think


def test_thinking_is_hidden_without_the_flag(run: Run, fake_mimoe: FakeMimoe) -> None:
    script(fake_mimoe, {"content": "Hi!", "reasoning": "hidden reasoning"})
    result = run(input="hi\n/quit\n")
    assert "hidden reasoning" not in result.output and "Hi!" in result.output


# -- approvals -----------------------------------------------------------------------------------


def test_approval_yes_executes_the_code(run: Run, fake_mimoe: FakeMimoe) -> None:
    script(fake_mimoe, RUN_PYTHON, {"content": "6*7 is 42."})
    result = run(input="Use run_python to print 6*7\ny\n/quit\n")
    assert result.exit_code == 0
    out = result.output
    assert "-> run_python(code='print(6*7)')" in out
    assert "approval request 1/1: run_python" in out
    assert re.search(r"1 +print\(6\*7\)", out)  # rich Syntax with line numbers
    assert APPROVAL_WARNING in out
    assert "Run this code? [y/N] y" in out and "approved" in out
    assert "warning: this code touches" not in out
    assert "exit_code: 0" in out and "6*7 is 42." in out
    tool = _tool_messages(fake_mimoe)[0]
    assert "stdout:\n42" in tool["content"]
    assert "2 model calls" in out


def test_approval_no_denies_and_tells_the_model(run: Run, fake_mimoe: FakeMimoe) -> None:
    script(fake_mimoe, RUN_PYTHON, {"content": "I did not run the code."})
    result = run(input="Use run_python to print 6*7\nn\n/quit\n")
    assert result.exit_code == 0
    assert "not executed; the model is told the code did not run" in result.output
    assert "I did not run the code." in result.output
    assert cli.FAILED_LINE not in result.output  # a denial is the user's decision, not a failure
    tool = _tool_messages(fake_mimoe)[0]
    assert "not executed" in tool["content"] and REJECT_MESSAGE in tool["content"]
    assert "stdout" not in tool["content"]


def test_approval_eof_denies(run: Run, fake_mimoe: FakeMimoe) -> None:
    script(fake_mimoe, RUN_PYTHON, {"content": "Understood."})
    result = run(input="Use run_python to print 6*7\n")  # stdin ends at the question
    assert result.exit_code == 0
    assert "not executed" in result.output and "Understood." in result.output
    assert REJECT_MESSAGE in _tool_messages(fake_mimoe)[0]["content"]


def test_approval_ctrl_c_denies_and_keeps_the_repl(
    run: Run, fake_mimoe: FakeMimoe, monkeypatch: pytest.MonkeyPatch
) -> None:
    script(fake_mimoe, RUN_PYTHON, {"content": "Skipped."})
    answers = iter(["Use run_python to print 6*7", KeyboardInterrupt(), "/quit"])

    def fake_input(prompt: str = "") -> str:
        item = next(answers)
        if isinstance(item, BaseException):
            raise item
        return item

    monkeypatch.setattr("builtins.input", fake_input)
    result = run()
    assert result.exit_code == 0
    assert "not executed" in result.output and "Skipped." in result.output
    assert REJECT_MESSAGE in _tool_messages(fake_mimoe)[0]["content"]


def test_red_flag_warning(run: Run, fake_mimoe: FakeMimoe) -> None:
    code = "import requests\nimport subprocess\nopen('x.txt', 'w').write('hi')\n"
    script(
        fake_mimoe,
        {"tool_calls": [{"name": "run_python", "args": {"code": code}}]},
        {"content": "Not run."},
    )
    result = run(input="fetch it\nn\n/quit\n")
    assert result.exit_code == 0
    assert "warning: this code touches the network, other processes or files" in result.output
    assert "requests" in result.output and "subprocess" in result.output
    assert "open('x.txt', 'w" in result.output  # the file-write pattern is reported too


def test_two_requests_in_one_approval(run: Run, fake_mimoe: FakeMimoe) -> None:
    script(
        fake_mimoe,
        {
            "tool_calls": [
                {"name": "run_python", "args": {"code": "print(1)"}},
                {"name": "run_python", "args": {"code": "print(2)"}},
            ]
        },
        {"content": "One ran, two did not."},
    )
    result = run(input="both\ny\nn\n/quit\n")
    assert result.exit_code == 0
    assert "approval request 1/2" in result.output and "approval request 2/2" in result.output
    tools = {t["tool_call_id"]: t["content"] for t in _tool_messages(fake_mimoe)}
    assert set(tools) == {"tool_0", "tool_1"}
    assert "stdout:\n1" in tools["tool_0"]  # approved and executed
    assert "not executed" in tools["tool_1"]  # rejected (the middleware lists it first)


def test_second_interrupt_in_the_same_turn_is_attempt_2(run: Run, fake_mimoe: FakeMimoe) -> None:
    script(fake_mimoe, RUN_PYTHON, RUN_PYTHON, {"content": "Done twice."})
    result = run(input="go\ny\ny\n/quit\n")
    assert result.exit_code == 0
    assert "approval request 1/1 (attempt 2): run_python" in result.output
    assert "Done twice." in result.output and "3 model calls" in result.output
    assert len(fake_mimoe.calls) == 4  # probe + three model calls


def test_auto_approve_skips_the_prompt(run: Run, fake_mimoe: FakeMimoe) -> None:
    script(fake_mimoe, RUN_PYTHON, {"content": "42"})
    result = run("--auto-approve", input="Use run_python to print 6*7\n/quit\n")
    assert result.exit_code == 0
    assert "Run this code?" not in result.output
    assert (
        "exit_code: 0" in result.output
        and "stdout:\n42" in _tool_messages(fake_mimoe)[0]["content"]
    )


# -- slash commands ------------------------------------------------------------------------------


def test_help_new_verbose_status_and_unknown_commands(run: Run, fake_mimoe: FakeMimoe) -> None:
    script(fake_mimoe, {"content": "one"}, {"content": "two"}, {"content": "three"})
    result = run(input="a\nb\n/help\n/verbose\n/verbose\n/status\n/bogus\n/new\nc\n/quit\n")
    assert result.exit_code == 0
    out = result.output
    assert "/model ID [--keep-loaded]" in out and "/quit" in out
    assert "verbose on" in out and "verbose off" in out
    assert "qwen3-4b on fake-node (v3.22.8 (developer edition), 0.6-generation API)" in out
    assert "tokens/s" in out and "tools on" in out
    assert "unknown command /bogus" in out
    assert "new conversation" in out
    before_new = fake_mimoe.calls[-2]["messages"]
    assert [m["role"] for m in before_new] == ["system", "user", "assistant", "user"]
    after_new = fake_mimoe.calls[-1]["messages"]
    assert [m["role"] for m in after_new] == ["system", "user"]
    assert after_new[-1]["content"] == "c /no_think"


def test_models_command_in_the_repl(run: Run, fake_mimoe: FakeMimoe) -> None:
    script(fake_mimoe)
    result = run(input="/models\n/quit\n")
    assert result.exit_code == 0
    assert "models on http://fake/mimik-ai/openai/v1 (0.6-generation engine)" in result.output
    assert re.search(r"qwen3-4b\s+│\s+2\.5 GB\s+│\s+loaded", result.output)
    assert f"{DEFAULT_PRESET} *" in result.output


def test_model_switch_in_the_repl(run: Run, fake_mimoe: FakeMimoe) -> None:
    # probe, then the probe of the new model, then the answer of the new model
    script(fake_mimoe, PING, {"content": "Hello from the small model."})
    result = run(input="/model smollm2-360m\nhi\n/status\n/quit\n")
    assert result.exit_code == 0
    assert "unloading qwen3-4b" in result.output and "loading smollm2-360m" in result.output
    assert "now using smollm2-360m: 35.4 tokens/s" in result.output
    assert "probe: smollm2-360m answered with a structured ping call" in result.output
    assert "Hello from the small model." in result.output
    assert fake_mimoe.loaded == ["smollm2-360m"]
    assert fake_mimoe.calls[-1]["model"] == "smollm2-360m"
    assert "smollm2-360m on fake-node" in result.output


def test_model_switch_keep_loaded_and_usage(run: Run, fake_mimoe: FakeMimoe) -> None:
    script(fake_mimoe, PING)
    result = run(
        input="/model\n/model smollm2-360m --bogus\n/model smollm2-360m --keep-loaded\n/quit\n"
    )
    assert result.exit_code == 0
    assert result.output.count("usage: /model ID [--keep-loaded]") == 2
    assert "unloading" not in result.output
    assert fake_mimoe.loaded == ["qwen3-4b", "smollm2-360m"]


def test_model_switch_unknown_id_keeps_the_session(run: Run, fake_mimoe: FakeMimoe) -> None:
    script(fake_mimoe, {"content": "Still here."})
    result = run(input="/model nope\nhi\n/quit\n")
    assert result.exit_code == 0
    assert "error: 'nope' is not in the model registry" in result.output
    assert "models pull" in result.output
    assert "Still here." in result.output and fake_mimoe.loaded == ["qwen3-4b"]


# -- Ctrl-C --------------------------------------------------------------------------------------


def test_approval_escapes_control_sequences_that_could_hide_code(
    run: Run, fake_mimoe: FakeMimoe
) -> None:
    """Code that moves the cursor up and erases a line could make the prompt show a harmless
    line while another one runs: the escapes are shown literally and flagged in red."""
    code = "print('looks harmless')\x1b[1A\x1b[2Kimport os  # the hidden line"
    script(
        fake_mimoe,
        {"tool_calls": [{"name": "run_python", "args": {"code": code}}]},
        {"content": "ok"},
    )
    result = run(input="go\nn\n/quit\n")
    assert result.exit_code == 0, result.output
    assert "\x1b" not in result.output, "a raw ESC reached the terminal"
    assert "\\x1b[1A\\x1b[2Kimport os" in result.output
    assert "invisible or terminal-control character(s)" in result.output


def test_answers_and_tool_results_cannot_send_terminal_sequences(
    run: Run, fake_mimoe: FakeMimoe
) -> None:
    """OSC 52 writes the clipboard, CSI sequences repaint the screen: text from the model or a
    file is shown with them escaped."""
    answer = "Done \x1b]52;c;ZXZpbA==\x07 and \u202ereversed"
    script(fake_mimoe, {"content": answer})
    result = run(input="hi\n/quit\n")
    assert result.exit_code == 0, result.output
    assert "\x1b" not in result.output and "\u202e" not in result.output
    assert "\\x1b]52;c;ZXZpbA==\\x07" in result.output
    assert "\\u202ereversed" in result.output


def test_real_ctrl_c_returns_at_once_while_the_engine_is_still_reading_the_prompt(
    run: Run, fake_mimoe: FakeMimoe, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CPU-only engine can take seconds before its first chunk; Ctrl-C must not wait for it.
    The prompt comes back within the grace period, the next message waits for the cancelled
    call, and langchain-core's warning about the deliberate cancel is not printed."""
    real_handler = fake_mimoe.handler
    stalled = _slow_answer(words=3, delay=4.0)  # the first chunk arrives after 4 s
    seen: dict[str, float] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat/completions"):
            body = json.loads(request.content)
            if body.get("stream") and body["messages"][-1]["content"].startswith("slow"):
                threading.Timer(
                    1.0,
                    lambda: (seen.setdefault("ctrl_c", time.monotonic()), _thread.interrupt_main()),
                ).start()
                return httpx.Response(
                    200, headers={"content-type": "text/event-stream"}, stream=stalled
                )
        return real_handler(request)

    real_cancel_turn = Session.cancel_turn

    def timed_cancel_turn(self: Session) -> None:
        seen["cancelled"] = time.monotonic()
        real_cancel_turn(self)

    monkeypatch.setattr(fake_mimoe, "handler", handler)
    monkeypatch.setattr(Session, "cancel_turn", timed_cancel_turn)
    script(fake_mimoe, {"content": "Still here."})
    result = run(input="slow question" + "\n" + "hi" + "\n" + "/quit" + "\n")
    assert result.exit_code == 0, result.output
    assert seen["cancelled"] - seen["ctrl_c"] < 2.5, "the prompt waited for the stalled engine"
    assert "turn cancelled" in result.output and "Still here." in result.output
    assert "Error in" not in result.output and "TurnCancelled" not in result.output


def test_models_table_fits_an_80_column_console(fake_mimoe: FakeMimoe) -> None:
    """80 columns is rich's width when stdout is not a terminal (a pipe, CI, some Windows
    consoles): model ids and the default's star must not be cut."""
    client = MimoeClient(fake_mimoe.base_url, "1234", client=fake_mimoe.client())
    client.discover()
    console = Console(file=io.StringIO(), width=80)
    console.print(
        cli.models_table(client.engine, client.loaded_models(), client.registry_models(), width=80)
    )
    out = console.file.getvalue()  # type: ignore[attr-defined]
    assert f"{DEFAULT_PRESET} *" in out
    assert "…" not in out
    assert max(len(line) for line in out.splitlines()) <= 80


def test_ctrl_c_at_the_prompt_exits_130(
    run: Run, fake_mimoe: FakeMimoe, monkeypatch: pytest.MonkeyPatch
) -> None:
    script(fake_mimoe)

    def interrupted(prompt: str = "") -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupted)
    assert run().exit_code == 130


def test_ctrl_c_mid_turn_cancels_and_keeps_the_repl(
    run: Run, fake_mimoe: FakeMimoe, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mimoe_agent import stream

    real_iter_events = stream.iter_events
    payloads: list[Any] = []

    def interrupted_once(agent: Any, payload: Any, config: Any) -> Any:
        payloads.append(payload)
        if len(payloads) == 1:
            yield {"event": "token", "text": "partial answer"}
            raise KeyboardInterrupt  # what Ctrl-C raises inside the graph run
        yield from real_iter_events(agent, payload, config)

    monkeypatch.setattr(stream, "iter_events", interrupted_once)
    script(fake_mimoe, {"content": "Still here."})
    result = run(input="slow question\nhi\n/quit\n")
    assert result.exit_code == 0
    assert "partial answer" in result.output
    assert "turn cancelled" in result.output
    assert "Still here." in result.output and len(payloads) == 2


def test_cancel_repairs_a_dangling_tool_call(
    fake_mimoe: FakeMimoe, workspace_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    real_make_model = llm.make_model
    monkeypatch.setattr(
        llm,
        "make_model",
        lambda settings, pre, **kw: real_make_model(
            settings,
            pre,
            http_client=fake_mimoe.client(),
            http_async_client=fake_mimoe.async_client(),
        ),
    )
    settings = load_settings(
        {"workspace": workspace_tmp, "base_url": fake_mimoe.base_url}, cwd=workspace_tmp.parent
    )
    client = MimoeClient(settings.base_url, settings.api_key, client=fake_mimoe.client())
    session = Session(
        settings, client, preflight(settings, client=client), Console(file=io.StringIO())
    )
    session.build_agent()
    # the state a cancel inside the tools node leaves behind: a tool call without its result
    call = {"name": "list_files", "args": {"path": "."}, "id": "tool_0", "type": "tool_call"}
    session.agent.update_state(
        session.run_config,
        {"messages": [HumanMessage("files?"), AIMessage(content="", tool_calls=[call])]},
        as_node="model",
    )
    session.cancel_turn()
    messages = session.agent.get_state(session.run_config).values["messages"]
    assert isinstance(messages[-1], ToolMessage)
    assert messages[-1].status == "error" and messages[-1].content == CANCELLED_RESULT
    assert messages[-1].tool_call_id == "tool_0" and messages[-1].name == "list_files"

    fake_mimoe.script({"content": "OK, that was cancelled."})
    session.run_turn("continue")
    sent = fake_mimoe.calls[-1]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "tool", "user"]
    assert CANCELLED_RESULT in sent[3]["content"]
    assert "OK, that was cancelled." in session.console.file.getvalue()  # type: ignore[union-attr]

    session.cancel_turn()  # nothing dangling now: a no-op
    assert session.agent.get_state(session.run_config).values["messages"][-1].content == (
        "OK, that was cancelled."
    )


# -- models subcommands --------------------------------------------------------------------------


def test_models_list(run: Run, fake_mimoe: FakeMimoe) -> None:
    fake_mimoe.registry_ready["qwen3-4b-instruct-2507"] = False
    result = run("models", "list")
    assert result.exit_code == 0
    out = result.output
    assert "models on http://fake/mimik-ai/openai/v1 (0.6-generation engine)" in out
    assert re.search(r"qwen3-4b\s+│\s+2\.5 GB\s+│\s+loaded\s+│\s+12,000\s+│\s+35\.4", out)
    # not downloaded: the store omits totalSize, so the preset's known size is shown
    assert re.search(rf"{re.escape(DEFAULT_PRESET)} \*\s+│\s+2\.5 GB\s+│\s+not downloaded", out)
    assert re.search(r"smollm2-360m\s+│\s+2\.5 GB\s+│\s+ready", out)
    assert re.search(r"qwen3-8b\s+│\s+5\.0 GB\s+│\s+not registered", out)
    assert "does not load on this engine" in out  # qwen3.5 presets on a 0.6 engine
    assert "* recommended default" in out


def test_models_list_works_without_a_workspace(
    run: Run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # no ./workspace here; the models commands do not need one
    result = run("models", "list")
    assert result.exit_code == 0 and "qwen3-4b" in result.output


def test_models_use_switches_via_the_fake(run: Run, fake_mimoe: FakeMimoe) -> None:
    result = run("models", "use", "smollm2-360m")
    assert result.exit_code == 0
    assert "unloading qwen3-4b" in result.output and "loading smollm2-360m" in result.output
    assert "loaded smollm2-360m: 35.4 tokens/s (avg 32.3), context 12,000" in result.output
    assert fake_mimoe.loaded == ["smollm2-360m"]
    assert "loading model 50%" not in result.output  # progress lines stay on the spinner

    kept = run("models", "use", "qwen3-4b", "--keep-loaded")
    assert kept.exit_code == 0 and fake_mimoe.loaded == ["smollm2-360m", "qwen3-4b"]
    assert "note:" not in kept.output


def test_models_use_keep_loaded_notes_an_eviction(
    run: Run, fake_mimoe: FakeMimoe, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Studio 0.6.5 keeps one model in memory: loading another silently unloads the first."""
    real_switch = cli.switch_model

    def evicting_switch(client: Any, model_id: str, **kwargs: Any) -> Any:
        loaded = real_switch(client, model_id, **kwargs)
        fake_mimoe.loaded.remove("qwen3-4b")  # what the real engine did in the live check
        return loaded

    monkeypatch.setattr(cli, "switch_model", evicting_switch)
    result = run("models", "use", "smollm2-360m", "--keep-loaded")
    assert result.exit_code == 0
    assert (
        "note: the engine unloaded qwen3-4b while loading smollm2-360m (it keeps one model in "
        "memory), so --keep-loaded had no effect"
    ) in result.output
    assert fake_mimoe.loaded == ["smollm2-360m"]


def test_models_use_unknown_id_exits_1(run: Run) -> None:
    result = run("models", "use", "nope")
    assert result.exit_code == 1
    assert "'nope' is not in the model registry" in result.output and "models pull" in result.output


def test_models_unload(run: Run, fake_mimoe: FakeMimoe) -> None:
    result = run("models", "unload", "qwen3-4b")
    assert result.exit_code == 0 and "unloaded qwen3-4b" in result.output
    assert fake_mimoe.loaded == []
    again = run("models", "unload", "qwen3-4b")
    assert again.exit_code == 1 and "is not loaded" in again.output


def test_models_pull_preset(run: Run, fake_mimoe: FakeMimoe) -> None:
    result = run("models", "pull", "smollm3-3b")
    assert result.exit_code == 0
    assert "registered smollm3-3b" in result.output
    assert (
        "smollm3-3b is ready (1.9 GB); load it with: mimoe-agent models use smollm3-3b"
        in result.output
    )
    assert fake_mimoe.registry_ready["smollm3-3b"] is True


def test_models_pull_warns_about_v10_only_presets_on_a_v06_engine(
    run: Run, fake_mimoe: FakeMimoe
) -> None:
    result = run("models", "pull", "qwen3.5-4b")
    assert result.exit_code == 0
    assert "warning: qwen3.5-4b does not load on 0.6-generation engines" in result.output


def test_models_pull_bad_spec_exits_1(run: Run) -> None:
    result = run("models", "pull", "not a spec")
    assert result.exit_code == 1
    assert "neither a preset nor an owner/repo[:QUANT] reference" in result.output


def test_models_commands_fail_with_hint_when_studio_is_down(
    run: Run, fake_mimoe: FakeMimoe
) -> None:
    fake_mimoe.down = True
    result = run("models", "list")
    assert result.exit_code == 1 and "not reachable" in result.output


# -- serve ---------------------------------------------------------------------------------------


def test_serve_delegates_to_server_run(
    run: Run, workspace_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mimoe_agent import server

    recorded: dict[str, Any] = {}
    monkeypatch.setattr(
        server,
        "run",
        lambda settings, port: recorded.update(settings=settings, port=port),
        raising=False,
    )
    result = run("--auto-approve", "serve", "--port", "8123")
    assert result.exit_code == 0
    assert recorded["port"] == 8123
    assert recorded["settings"].workspace == workspace_tmp
    assert recorded["settings"].auto_approve is True
    assert os.environ.get("LANGSMITH_TRACING") == "false"


def test_serve_history_option(run: Run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mimoe_agent import server

    recorded: dict[str, Any] = {}
    monkeypatch.setattr(server, "run", lambda settings, port: recorded.update(settings=settings))
    assert run("serve", "--history", "off").exit_code == 0
    assert recorded["settings"].history == "off"
    assert run("serve", "--history", str(tmp_path / "c.db")).exit_code == 0
    assert recorded["settings"].history == str(tmp_path / "c.db")
    assert run("serve").exit_code == 0
    assert recorded["settings"].history is None  # serve then uses the per-user file
    assert "--history" in _plain(run("serve", "--help").output)


def test_serve_config_error_exits_1(run: Run, tmp_path: Path) -> None:
    result = run("--workspace", str(tmp_path / "missing"), "serve")
    assert result.exit_code == 1 and "pass --workspace PATH" in result.output


# -- portability ---------------------------------------------------------------------------------


def test_cli_uses_no_posix_only_process_apis() -> None:
    source = Path(cli.__file__).read_text(encoding="utf-8")
    assert "import signal" not in source and "os.fork" not in source and "SIGKILL" not in source
    assert "os.kill" not in source


def test_format_args_and_cut() -> None:
    assert cli._format_args({"path": ".", "max_depth": 4}) == "path='.', max_depth=4"
    long = cli._format_args({"code": "x" * 100})
    assert long.startswith("code='xxx") and long.endswith("...") and len(long) <= 70
    assert cli._cut("abc", 3) == "abc" and cli._cut("abcdef", 5) == "ab..."


# -- adversarial probes (review) -----------------------------------------------------------------


def _session(
    fake_mimoe: FakeMimoe, workspace_tmp: Path, monkeypatch: pytest.MonkeyPatch, console: Console
) -> Session:
    """A built :class:`Session` on the fake engine writing to ``console`` (no CliRunner in
    between, so a ``force_terminal`` console exercises the rich Live path)."""
    real_make_model = llm.make_model
    monkeypatch.setattr(
        llm,
        "make_model",
        lambda settings, pre, **kw: real_make_model(
            settings,
            pre,
            http_client=fake_mimoe.client(),
            http_async_client=fake_mimoe.async_client(),
        ),
    )
    settings = load_settings(
        {"workspace": workspace_tmp, "base_url": fake_mimoe.base_url}, cwd=workspace_tmp.parent
    )
    client = MimoeClient(settings.base_url, settings.api_key, client=fake_mimoe.client())
    session = Session(settings, client, preflight(settings, client=client), console)
    session.build_agent()
    return session


def test_ctrl_c_inside_a_tool_goes_through_the_real_graph(
    run: Run, fake_mimoe: FakeMimoe, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ctrl-C while ``run_python`` runs (langgraph runs a lone task on the main thread): the
    interrupt crosses the middleware, langgraph and ``iter_events``, the REPL survives and the
    dangling tool call gets its error result before the next request."""
    from mimoe_agent.tools import run_python as run_python_module

    def interrupted(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(run_python_module, "run_python_code", interrupted)
    script(fake_mimoe, RUN_PYTHON, {"content": "Still here."})
    result = run("--auto-approve", input="go\nhi\n/quit\n")
    assert result.exit_code == 0, result.output
    assert "turn cancelled" in result.output and "Still here." in result.output
    sent = fake_mimoe.calls[-1]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "tool", "user"], sent
    assert CANCELLED_RESULT in sent[3]["content"]


def test_ctrl_c_during_the_model_call_goes_through_the_real_graph(
    run: Run, fake_mimoe: FakeMimoe, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ctrl-C while the completion request is in flight (raised inside the transport, like a
    socket read would): nothing of the interrupted call reaches the thread."""
    real_handler = fake_mimoe.handler
    completions = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat/completions"):
            completions["n"] += 1
            if completions["n"] == 2:  # the probe is the first, the turn's model call the second
                raise KeyboardInterrupt
        return real_handler(request)

    monkeypatch.setattr(fake_mimoe, "handler", handler)
    script(fake_mimoe, {"content": "Still here."})
    result = run(input="slow question\nhi\n/quit\n")
    assert result.exit_code == 0, result.output
    assert "turn cancelled" in result.output and "Still here." in result.output
    assert [m["role"] for m in fake_mimoe.calls[-1]["messages"]] == ["system", "user", "user"]


class _SlowStream(httpx.SyncByteStream):
    """An SSE body that yields one chunk every ``delay`` seconds and notices an early close."""

    def __init__(self, chunks: list[bytes], delay: float) -> None:
        self.chunks, self.delay = chunks, delay
        self.sent = 0
        self.closed = False

    def __iter__(self) -> Any:
        for chunk in self.chunks:
            if self.closed:
                return
            time.sleep(self.delay)
            self.sent += 1
            yield chunk

    def close(self) -> None:
        self.closed = True


def _slow_answer(words: int, delay: float) -> _SlowStream:
    def sse(delta: dict[str, Any], finish: str | None = None) -> bytes:
        chunk = {
            "id": "chatcmpl-slow",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "qwen3-4b",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(chunk)}\n\n".encode()

    chunks = [sse({"role": "assistant"})]
    chunks += [sse({"content": f"word{i} "}) for i in range(words)]
    chunks += [sse({}, "stop"), b"data: [DONE]\n\n"]
    return _SlowStream(chunks, delay)


def test_real_ctrl_c_ends_a_streaming_answer_at_once(
    run: Run, fake_mimoe: FakeMimoe, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real interrupt of the REPL thread while the model streams (the graph runs the model on
    a pool thread): the turn ends within a second and the HTTP stream is closed early, instead
    of the REPL waiting for the whole answer."""
    real_handler = fake_mimoe.handler
    slow = _slow_answer(words=80, delay=0.1)  # 8 s if read to the end
    seen: dict[str, float] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat/completions"):
            body = json.loads(request.content)
            if body.get("stream") and body["messages"][-1]["content"].startswith("slow"):
                seen["started"] = time.monotonic()
                threading.Timer(1.0, _thread.interrupt_main).start()
                return httpx.Response(
                    200, headers={"content-type": "text/event-stream"}, stream=slow
                )
            if "started" in seen and "next" not in seen:
                seen["next"] = time.monotonic()
        return real_handler(request)

    monkeypatch.setattr(fake_mimoe, "handler", handler)
    script(fake_mimoe, {"content": "Still here."})
    result = run(input="slow question\nhi\n/quit\n")
    assert result.exit_code == 0, result.output
    assert "turn cancelled" in result.output and "Still here." in result.output
    assert slow.sent < 40, f"the stream was read for {slow.sent} of 82 chunks"
    assert seen["next"] - seen["started"] < 3.0


def test_real_ctrl_c_kills_a_running_snippet(
    run: Run, fake_mimoe: FakeMimoe, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real interrupt while an approved run_python sleeps: the snippet's tree is killed, the
    model is told it was cancelled, and the REPL is back within seconds, not after 30 s."""
    real_handler = fake_mimoe.handler
    armed: dict[str, bool] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        response = real_handler(request)
        if (
            request.url.path.endswith("/chat/completions")
            and not armed
            and len(fake_mimoe.calls) == 2
        ):  # the probe, then the call that answers with run_python
            armed["yes"] = True
            threading.Timer(2.0, _thread.interrupt_main).start()
        return response

    monkeypatch.setattr(fake_mimoe, "handler", handler)
    sleeper = {
        "tool_calls": [{"name": "run_python", "args": {"code": "import time\ntime.sleep(30)"}}]
    }
    script(fake_mimoe, sleeper, {"content": "Still here."})
    started = time.monotonic()
    result = run("--auto-approve", input="go\nhi\n/quit\n")
    assert result.exit_code == 0, result.output
    assert time.monotonic() - started < 15.0
    assert "turn cancelled" in result.output and "Still here." in result.output
    sent = fake_mimoe.calls[-1]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "tool", "user"], sent
    assert "cancelled by the user" in sent[3]["content"]


def test_ctrl_c_during_preflight_exits_130(
    run: Run, fake_mimoe: FakeMimoe, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "preflight", interrupted)
    assert run().exit_code == 130


def test_ctrl_c_during_a_model_switch_is_not_a_cancelled_turn(
    run: Run, fake_mimoe: FakeMimoe, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "switch_model", interrupted)
    script(fake_mimoe, {"content": "Still here."})
    result = run(input="/model smollm2-360m\nhi\n/quit\n")
    assert result.exit_code == 0, result.output
    assert "switch interrupted" in result.output and "/status" in result.output
    assert "turn cancelled" not in result.output
    assert "Still here." in result.output and fake_mimoe.calls[-1]["model"] == "qwen3-4b"


def test_cp1252_stdout_is_reconfigured_to_utf8(
    fake_mimoe: FakeMimoe, workspace_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Windows pipe defaults to the code page: emoji and CJK in the answer must not crash."""
    real_make_model = llm.make_model
    monkeypatch.setattr(
        cli,
        "MimoeClient",
        lambda base_url, api_key, **kw: MimoeClient(base_url, api_key, client=fake_mimoe.client()),
    )
    monkeypatch.setattr(
        llm,
        "make_model",
        lambda settings, pre, **kw: real_make_model(
            settings,
            pre,
            http_client=fake_mimoe.client(),
            http_async_client=fake_mimoe.async_client(),
        ),
    )
    monkeypatch.setenv("COLUMNS", "120")
    monkeypatch.chdir(workspace_tmp.parent)
    raw = io.BytesIO()
    stdout = io.TextIOWrapper(raw, encoding="cp1252", errors="strict", write_through=True)
    stdin = io.TextIOWrapper(io.BytesIO(b"hi\n/quit\n"), encoding="cp1252")
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr("sys.stdin", stdin)
    script(fake_mimoe, {"content": "Done ✅ — café 🚀 表"})
    assert cli.run_repl({"base_url": fake_mimoe.base_url}) == 0
    stdout.flush()
    assert "Done ✅ — café 🚀 表" in raw.getvalue().decode("utf-8")


def test_long_answer_streams_throttled_and_prints_in_full_on_a_terminal(
    fake_mimoe: FakeMimoe, workspace_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a terminal the answer streams through rich Live (cropped to the screen while it grows)
    and is printed whole at the end; the Markdown re-parse is throttled to the refresh rate."""
    parses = {"n": 0}
    real_markdown = cli._Markdown

    class CountingMarkdown(real_markdown):  # type: ignore[misc,valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            parses["n"] += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(cli, "_Markdown", CountingMarkdown)
    buf = io.StringIO()
    console = Console(
        file=buf, force_terminal=True, force_interactive=True, width=100, height=24, emoji=False
    )
    session = _session(fake_mimoe, workspace_tmp, monkeypatch, console)
    lines = [f"Line {i} of a very long answer." for i in range(400)]
    fake_mimoe.script({"content": "\n\n".join(lines)})
    session.run_turn("tell me everything")
    out = buf.getvalue()
    assert "Line 0 of a very long answer." in out and "Line 399 of a very long answer." in out
    assert "elapsed" in out
    assert 1 <= parses["n"] <= 20, parses  # not once per streamed chunk


def test_blank_input_lines_are_ignored(run: Run, fake_mimoe: FakeMimoe) -> None:
    script(fake_mimoe, {"content": "Hello."})
    result = run(input="\n   \n\t\nhi\n/quit\n")
    assert result.exit_code == 0 and "Hello." in result.output
    assert len(fake_mimoe.calls) == 2  # the probe and one model call


def test_empty_stdin_exits_0_after_the_banner(run: Run, fake_mimoe: FakeMimoe) -> None:
    script(fake_mimoe)
    result = run(input="")
    assert result.exit_code == 0 and "mimoe-agent 0.1.0" in result.output
    assert len(fake_mimoe.calls) == 1  # only the probe


def test_backticks_in_approved_code(run: Run, fake_mimoe: FakeMimoe) -> None:
    code = 'print("```fence``` and `tick`")'
    script(
        fake_mimoe,
        {"tool_calls": [{"name": "run_python", "args": {"code": code}}]},
        {"content": "Printed."},
    )
    result = run(input="go\ny\n/quit\n")
    assert result.exit_code == 0, result.output
    assert "```fence``` and `tick`" in result.output  # shown verbatim, not as a fence
    assert "```fence``` and `tick`" in _tool_messages(fake_mimoe)[0]["content"]


def test_piped_approval_answers_from_the_next_stdin_line(run: Run, fake_mimoe: FakeMimoe) -> None:
    """Without a terminal the question is still asked: the next stdin line answers it, and
    anything but y/yes denies (it is never sent to the model as a message)."""
    script(fake_mimoe, RUN_PYTHON, {"content": "Not run."})
    result = run(input="go\nwhat is 2+2\n/quit\n")
    assert result.exit_code == 0, result.output
    assert "Run this code? [y/N] what is 2+2" in result.output
    assert "not executed" in result.output and "Not run." in result.output
    users = [m["content"] for m in fake_mimoe.calls[-1]["messages"] if m["role"] == "user"]
    assert users == ["go /no_think"]


def test_model_switch_shows_the_memory_note(run: Run, fake_mimoe: FakeMimoe) -> None:
    fake_mimoe.registry.append("qwen3-8b")
    fake_mimoe.registry_sizes["qwen3-8b"] = 5_000_000_000
    script(fake_mimoe, PING)
    result = run(input="/model qwen3-8b\n/quit\n")
    assert result.exit_code == 0, result.output
    assert re.search(r"warning: qwen3-8b is about \d+ GB at 12k context", result.output)
    assert "now using qwen3-8b" in result.output


def test_auto_approve_with_allow_network_warns(run: Run, fake_mimoe: FakeMimoe) -> None:
    script(fake_mimoe)
    result = run("--auto-approve", "--allow-network", input="/quit\n")
    assert result.exit_code == 0
    assert f"! {cli.UNATTENDED_NETWORK_WARNING}" in result.output
    alone = run("--auto-approve", input="/quit\n")
    assert cli.UNATTENDED_NETWORK_WARNING not in alone.output


# -- the terminal: line reader, approval menu, code colours --------------------------------------


@pytest.fixture
def menu_input(monkeypatch: pytest.MonkeyPatch) -> Iterator[PipeInput]:
    """The terminal as prompt_toolkit sees it: keys from a pipe, nothing drawn. The line reader
    and the approval menu run for real on it; the menu takes keys sent ahead of it at once (its
    timing rules are tested in test_terminal.py)."""
    monkeypatch.setattr(terminal, "ARM_DELAY_S", 0)
    monkeypatch.setattr(terminal, "SETTLE_S", 0)
    with create_pipe_input() as pipe, create_app_session(pipe, DummyOutput()):
        yield pipe


def _terminal_session(
    fake_mimoe: FakeMimoe, workspace_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Session, io.StringIO]:
    """A session that reads lines and approvals the terminal way (``menu_input`` must be on)."""
    buf = io.StringIO()
    session = _session(fake_mimoe, workspace_tmp, monkeypatch, Console(file=buf, width=120))
    session.reader = terminal.LineReader(cli.PROMPT)
    return session, buf


@pytest.mark.parametrize(("keys", "approved"), [("\r", True), ("\x1b[B\r", False), ("2", False)])
def test_the_approval_menu_answers_for_the_session(
    fake_mimoe: FakeMimoe,
    workspace_tmp: Path,
    monkeypatch: pytest.MonkeyPatch,
    menu_input: PipeInput,
    keys: str,
    approved: bool,
) -> None:
    session, buf = _terminal_session(fake_mimoe, workspace_tmp, monkeypatch)
    fake_mimoe.script(RUN_PYTHON, {"content": "Done."})
    menu_input.send_text(keys)
    session.run_turn("Use run_python to print 6*7")
    out = buf.getvalue()
    assert "approval request 1/1: run_python" in out and APPROVAL_WARNING in out
    assert "Run this code? [y/N]" not in out  # the menu asked, not the line
    tool = _tool_messages(fake_mimoe)[0]["content"]
    if approved:
        assert "approved" in out and "stdout:\n42" in tool
    else:
        assert "not executed; the model is told the code did not run" in out
        assert REJECT_MESSAGE in tool and "stdout" not in tool
    assert "Done." in out and "2 model calls" in out


def test_esc_in_the_approval_menu_cancels_the_turn(
    fake_mimoe: FakeMimoe,
    workspace_tmp: Path,
    monkeypatch: pytest.MonkeyPatch,
    menu_input: PipeInput,
) -> None:
    """Esc ends the turn like Ctrl-C: no decision is sent, the model is not called again, and
    the call gets a result saying the code did not run, so the next message goes out whole."""
    from langchain_core.messages import ToolMessage

    session, buf = _terminal_session(fake_mimoe, workspace_tmp, monkeypatch)
    fake_mimoe.script(RUN_PYTHON)
    menu_input.send_text("\x1b")
    before = len(fake_mimoe.calls)
    session.run_turn("Use run_python to print 6*7")
    out = buf.getvalue()
    assert cli.APPROVAL_CANCELLED in out
    assert "approved" not in out and "not executed" not in out and "elapsed" not in out
    assert len(fake_mimoe.calls) == before + 1  # only the call that asked for run_python
    assert not session.pending_interrupt()
    last = session.agent.get_state(session.run_config).values["messages"][-1]
    assert isinstance(last, ToolMessage) and last.status == "error"
    assert last.content == cli.APPROVAL_CANCELLED_RESULT and last.tool_call_id == "tool_0"

    fake_mimoe.script({"content": "OK, it did not run."})
    session.run_turn("hi")
    sent = fake_mimoe.calls[-1]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "tool", "user"]
    assert cli.APPROVAL_CANCELLED_RESULT in sent[3]["content"]
    assert "OK, it did not run." in buf.getvalue()


def test_a_new_message_at_an_approval_interrupt_would_leave_the_call_dangling(
    fake_mimoe: FakeMimoe, workspace_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Why a cancel mends the thread: LangGraph takes a new message on a thread that waits at the
    approval interrupt as a fresh run, drops the interrupt without a word and sends the model
    the tool call with no result after it."""
    from langchain_core.messages import HumanMessage

    from mimoe_agent import stream

    session = _session(fake_mimoe, workspace_tmp, monkeypatch, Console(file=io.StringIO()))
    fake_mimoe.script(RUN_PYTHON, {"content": "fresh"})
    go = {"messages": [HumanMessage("go")]}
    assert list(stream.iter_events(session.agent, go, session.run_config))[-1]["status"] == (
        "awaiting_approval"
    )
    assert session.pending_interrupt()
    hello = {"messages": [HumanMessage("hello")]}
    list(stream.iter_events(session.agent, hello, session.run_config))
    assert not session.pending_interrupt()
    sent = fake_mimoe.calls[-1]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
    assert sent[2]["tool_calls"][0]["function"]["name"] == "run_python"


def test_a_cancel_at_the_second_request_cancels_the_whole_step(
    fake_mimoe: FakeMimoe,
    workspace_tmp: Path,
    monkeypatch: pytest.MonkeyPatch,
    menu_input: PipeInput,
) -> None:
    """Yes to the first request, Esc at the second: nothing of the step runs, not even the
    approved snippet or the tool that needed no approval, and every call gets a result."""
    from langchain_core.messages import ToolMessage

    session, buf = _terminal_session(fake_mimoe, workspace_tmp, monkeypatch)
    answers = iter([terminal.YES, terminal.CANCEL])
    monkeypatch.setattr(terminal, "ask_approval", lambda question, pointer: next(answers))
    fake_mimoe.script(
        {
            "tool_calls": [
                {"name": "list_files", "args": {"path": "."}},
                {"name": "run_python", "args": {"code": "print(1)"}},
                {"name": "run_python", "args": {"code": "print(2)"}},
            ]
        }
    )
    before = len(fake_mimoe.calls)
    session.run_turn("both")
    out = buf.getvalue()
    assert "approval request 2/2" in out and "approved" in out
    # the "approved" above it did not run either, and the last line says so
    assert out.rstrip().endswith("cancelled: none of the 2 requests ran and the turn ended")
    assert cli.APPROVAL_CANCELLED not in out
    assert len(fake_mimoe.calls) == before + 1
    assert not session.pending_interrupt()
    messages = session.agent.get_state(session.run_config).values["messages"]
    results = {m.tool_call_id: m.content for m in messages if isinstance(m, ToolMessage)}
    assert results == {
        "tool_0": CANCELLED_RESULT,
        "tool_1": cli.APPROVAL_CANCELLED_RESULT,
        "tool_2": cli.APPROVAL_CANCELLED_RESULT,
    }


def test_the_repl_reads_with_the_line_reader_on_a_terminal(
    fake_mimoe: FakeMimoe,
    workspace_tmp: Path,
    monkeypatch: pytest.MonkeyPatch,
    menu_input: PipeInput,
) -> None:
    monkeypatch.setattr(terminal, "interactive", lambda: True)
    buf = io.StringIO()
    session = _session(fake_mimoe, workspace_tmp, monkeypatch, Console(file=buf, width=120))
    fake_mimoe.script({"content": "Hello."})
    menu_input.send_text("hi\r\x04")  # a line, then Ctrl-D on the empty prompt
    assert session.loop() == cli.EXIT_OK
    assert isinstance(session.reader, terminal.LineReader)
    assert "Hello." in buf.getvalue() and cli.PROMPT not in buf.getvalue()  # prompt_toolkit's
    assert fake_mimoe.calls[-1]["messages"][-1]["content"] == "hi /no_think"
    menu_input.send_text("typed\x03")  # Ctrl-C at the prompt
    assert session.loop() == cli.EXIT_INTERRUPT
    assert buf.getvalue().endswith("tokens\n")  # prompt_toolkit ended the line already


def test_plain_lines_when_prompt_toolkit_cannot_drive_the_terminal(
    fake_mimoe: FakeMimoe, workspace_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No line reader (mintty without winpty, say): the you> line and the y/N question are read
    with ``console.input`` as on a pipe."""
    monkeypatch.setattr(terminal, "interactive", lambda: True)
    monkeypatch.setattr(terminal.LineReader, "open", classmethod(lambda cls, prompt: None))
    lines = iter(["Use run_python to print 6*7", "y", "/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))
    buf = io.StringIO()
    session = _session(fake_mimoe, workspace_tmp, monkeypatch, Console(file=buf, width=120))
    fake_mimoe.script(RUN_PYTHON, {"content": "42."})
    assert session.loop() == cli.EXIT_OK
    assert session.reader is None
    assert "Run this code? [y/N] y" in buf.getvalue() and "approved" in buf.getvalue()


def test_a_line_reader_or_menu_that_fails_later_falls_back_to_plain_lines(
    fake_mimoe: FakeMimoe, workspace_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class GoneConsole:
        def read(self) -> str:
            raise OSError("the console went away")

    def broken_menu(question: str, pointer: str) -> str:
        raise RuntimeError("no console screen buffer")

    buf = io.StringIO()
    session = _session(fake_mimoe, workspace_tmp, monkeypatch, Console(file=buf, width=120))
    lines = iter(["/quit", "y"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))
    session.reader = GoneConsole()  # type: ignore[assignment]
    assert session.loop() == cli.EXIT_OK and session.reader is None

    session.reader = GoneConsole()  # type: ignore[assignment]
    monkeypatch.setattr(terminal, "ask_approval", broken_menu)
    fake_mimoe.script(RUN_PYTHON, {"content": "42."})
    session.run_turn("Use run_python to print 6*7")
    assert "Run this code? [y/N] y" in buf.getvalue() and "approved" in buf.getvalue()
    assert session.reader is None


SGR_RE = re.compile(r"\x1b\[([0-9;]*)m")


def _backgrounds(ansi: str) -> set[str]:
    """Background colours the SGR sequences in ``ansi`` set (49, the default, left out)."""
    found: set[str] = set()
    for params in SGR_RE.findall(ansi):
        codes = params.split(";")
        index = 0
        while index < len(codes):
            code = codes[index]
            if code in ("38", "48"):  # extended colour: 5;n or 2;r;g;b
                width = 3 if codes[index + 1 : index + 2] == ["5"] else 5
                if code == "48":
                    found.add(";".join(codes[index : index + width]))
                index += width
                continue
            if code.isdigit() and (40 <= int(code) <= 47 or 100 <= int(code) <= 107):
                found.add(code)
            index += 1
    return found


@pytest.mark.parametrize("light", [True, False])
def test_code_is_drawn_on_the_terminals_own_background(
    fake_mimoe: FakeMimoe, workspace_tmp: Path, monkeypatch: pytest.MonkeyPatch, light: bool
) -> None:
    """The approval code, a code block and inline code in an answer: palette colours on the
    terminal's background, where rich's defaults paint monokai's #272822 and black."""
    monkeypatch.delenv("NO_COLOR", raising=False)  # rich would draw no colour at all
    code = "def double(x):\n    return x * 2  # twice\n\nprint(double(21))\n"

    def console(buf: io.StringIO, **kwargs: Any) -> Console:
        return Console(file=buf, force_terminal=True, color_system="truecolor", width=100, **kwargs)

    buf = io.StringIO()
    themed = console(buf, theme=terminal.CONSOLE_THEME)  # what cli._console() adds
    session = _session(fake_mimoe, workspace_tmp, monkeypatch, themed)
    session.code_theme = terminal.code_theme(light)
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    request = {"name": "run_python", "args": {"code": code}}
    assert session.ask_decisions({"action_requests": [request]}, 1) == [
        {"type": "reject", "message": REJECT_MESSAGE}
    ]
    renderer = cli.Renderer(
        session.console,
        cli.TurnStats(),
        verbose=False,
        show_thinking=False,
        code_theme=session.code_theme,
    )
    renderer.handle({"event": "token", "text": f"Run `double(21)`:\n\n```python\n{code}```\n"})
    renderer.close()
    out = buf.getvalue()
    assert "double" in out and "\x1b[" in out
    assert _backgrounds(out) == set()
    assert "38;2;" not in out and "38;5;" not in out  # the 16 palette colours only

    reference = io.StringIO()
    rich_defaults = console(reference)
    rich_defaults.print(Syntax(code, "python", line_numbers=True))
    rich_defaults.print(Markdown("Run `double(21)`"))
    assert _backgrounds(reference.getvalue()) >= {"48;2;39;40;34", "40"}


def test_code_blocks_in_answers_sit_like_paragraphs(
    fake_mimoe: FakeMimoe, workspace_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a background band, rich's padding around a code block only left two blank rows
    above and below it and indented each line, and a copy of the code kept the indent."""
    buf = io.StringIO()
    session = _session(fake_mimoe, workspace_tmp, monkeypatch, Console(file=buf, width=60))
    fake_mimoe.script(
        {"content": "Here:\n\n```python\nfor n in range(3):\n    print(n)\n```\n\nDone."}
    )
    session.run_turn("show me")
    lines = [line.rstrip() for line in buf.getvalue().splitlines()]
    start = lines.index("Here:")
    assert lines[start : start + 6] == [
        "Here:",
        "",
        "for n in range(3):",
        "    print(n)",
        "",
        "Done.",
    ]


def test_the_code_theme_is_picked_once_at_start_up(
    run: Run, fake_mimoe: FakeMimoe, monkeypatch: pytest.MonkeyPatch
) -> None:
    guesses: list[bool] = []
    monkeypatch.setattr(terminal, "light_background", lambda: guesses.append(True) or True)
    themes: list[str | None] = []
    real_markdown = cli._Markdown

    class RecordingMarkdown(real_markdown):  # type: ignore[misc,valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            themes.append(kwargs.get("code_theme"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(cli, "_Markdown", RecordingMarkdown)
    script(fake_mimoe, {"content": "one"}, {"content": "two"})
    result = run(input="a\nb\n/quit\n")
    assert result.exit_code == 0, result.output
    assert guesses == [True]
    assert themes and set(themes) == {"ansi_light"}


def test_no_terminal_query_and_no_menu_without_a_terminal(
    run: Run, fake_mimoe: FakeMimoe, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("COLORFGBG", raising=False)
    script(fake_mimoe, RUN_PYTHON, {"content": "Not run."})
    result = run(input="go\nn\n/quit\n")
    assert result.exit_code == 0, result.output
    assert "\x1b]11;?" not in result.output and "\x1b[c" not in result.output
    assert "Run this code? [y/N] n" in result.output and "Esc to cancel" not in result.output
    assert cli._console().get_style("markdown.code").bgcolor is None
