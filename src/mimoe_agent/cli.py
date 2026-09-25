"""Command-line entry points: the REPL (default command), ``serve`` and the ``models`` group.

The REPL drives :func:`mimoe_agent.stream.iter_events` and renders the event stream with rich: a
spinner while the model works, tool calls as dim ``-> name(args)`` lines, tool results dim and
shortened (``/verbose`` shows everything), the answer as live-rendered Markdown, notices in
yellow and a closing ``elapsed 6.3 s, 2 model calls, 91 tokens`` line. A ``run_python`` request
ends the stream (``done{status: "awaiting_approval"}``): the code is shown with line numbers,
flagged in red when it touches the network, processes or files, and the answer to
``Run this code? [y/N]`` becomes exactly one decision per action request in the
``Command(resume=...)`` that continues the graph.

Ctrl-C at the prompt exits with 130; Ctrl-C during a turn cancels the turn (``run_python`` kills
its child on the way out) and keeps the REPL; EOF exits with 0, so scripted sessions work:
``printf "What files are here?\\n/quit\\n" | mimoe-agent --auto-approve``. Configuration and
preflight failures print the message and its hint and exit with 1.

LangChain is imported lazily, after :func:`mimoe_agent.config.apply_tracing_env` ran, so a
reviewer's ``LANGSMITH_TRACING=true`` never uploads a prompt unless ``--trace`` is given. The
only signal handling is Python's own ``KeyboardInterrupt``; nothing here forks or sends signals,
so the module behaves the same on Windows.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import queue
import re
import sys
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import typer
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.status import Status
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from mimoe_agent import __version__
from mimoe_agent.agent import APPROVAL_TOOL, APPROVAL_WARNING
from mimoe_agent.config import ConfigError, Settings, apply_tracing_env, env_name, load_settings
from mimoe_agent.mimoe import (
    EngineGeneration,
    EngineInfo,
    LoadedModel,
    MimoeClient,
    MimoeError,
    Preflight,
    RegistryModel,
    preflight,
)
from mimoe_agent.models import (
    DEFAULT_PRESET,
    PRESETS,
    V10_ONLY_PRESETS,
    preset_by_id,
    pull_model,
    switch_model,
)

PROMPT = "you> "
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_INTERRUPT = 130
RESULT_PREVIEW_LINES = 3
"""Lines of a tool result shown unless ``/verbose`` is on."""
ARG_PREVIEW_CHARS = 60
THINKING_PREVIEW_CHARS = 100
REFRESH_PER_SECOND = 8
SPINNER = "line"
"""ASCII spinner: renders on every terminal, including legacy Windows consoles."""
NO_ANSWER = "The model returned no answer; try rephrasing"
REJECT_MESSAGE = (
    "The user declined to run this code. Tell the user it was not executed and stop; do not retry."
)
CANCELLED_RESULT = "ERROR: cancelled by the user before the tool finished"
# Characters a terminal would act on instead of showing (C0/C1 controls such as ESC, which
# starts cursor-movement, colour, title and clipboard sequences), plus bidirectional overrides
# and line separators that reorder or hide text. Everything the model or a tool produced is
# shown with these escaped; the approval prompt also escapes zero-width characters.
_UNSAFE_RE = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f\u2028\u2029\u202a-\u202e\u2066-\u2069]")
_UNSAFE_OR_INVISIBLE_RE = re.compile(
    "[\x00-\x08\x0b-\x1f\x7f-\x9f\u200b-\u200f\u2028\u2029\u202a-\u202e\u2060-\u2064"
    "\u2066-\u2069\ufeff]"
)


def _escape_char(match: re.Match[str]) -> str:
    code = ord(match.group(0))
    return f"\\x{code:02x}" if code < 0x100 else f"\\u{code:04x}"


def terminal_safe(text: str, *, strict: bool = False) -> tuple[str, int]:
    """``text`` with terminal control, bidi and (``strict``) zero-width characters escaped.

    Returns the display text and how many characters were escaped. CRLF becomes LF first.
    """
    pattern = _UNSAFE_OR_INVISIBLE_RE if strict else _UNSAFE_RE
    return pattern.subn(_escape_char, text.replace("\r\n", "\n"))


def _safe_value(value: Any) -> Any:
    if isinstance(value, str):
        return terminal_safe(value)[0]
    if isinstance(value, Mapping):
        return {key: _safe_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_safe_value(item) for item in value]
    return value


NARROW_TABLE_WIDTH = 110
"""Console width below which ``models list`` moves its notes under the table."""
TURN_POLL_S = 0.1
"""How often the REPL thread wakes while a turn runs; keeps Ctrl-C prompt on every OS."""
CANCEL_GRACE_S = 1.0
"""How long Ctrl-C waits for the cancelled turn before the prompt comes back. A model call ends
at its next streamed chunk; on a CPU-only machine reading a long prompt that can take seconds,
so the rest happens in the background and the next message waits for it."""
CANCEL_WAIT_S = 15.0
"""How long the next message waits for a cancelled turn before moving to a new thread."""


class _HideCancelWarnings(logging.Filter):
    """langchain-core logs a warning before it re-raises a callback's exception; for the
    deliberate ``TurnCancelled`` that is noise in the terminal."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "TurnCancelled" not in record.getMessage()


logging.getLogger("langchain_core.callbacks.manager").addFilter(_HideCancelWarnings())


class TurnCancelled(Exception):
    """Raised inside the model call once the user has cancelled the turn."""


class _Failed:
    """Carries an exception from the turn's worker thread to the REPL thread."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


_DONE = object()


def _cancel_handler(cancel: threading.Event) -> Any:
    """A callback handler that aborts the model call at its next streamed chunk after Ctrl-C.

    mimOE streams a chunk every few tens of milliseconds (progress chunks while it reads the
    prompt), so raising from ``on_llm_new_token`` ends the HTTP stream almost at once; the
    start hooks stop a new model call from beginning after the cancel.
    """
    from langchain_core.callbacks import BaseCallbackHandler

    class CancelOnModelOutput(BaseCallbackHandler):
        raise_error = True

        def _check(self) -> None:
            if cancel.is_set():
                raise TurnCancelled("turn cancelled by the user")

        def on_chat_model_start(self, *args: Any, **kwargs: Any) -> None:
            self._check()

        def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
            self._check()

        def on_llm_new_token(self, *args: Any, **kwargs: Any) -> None:
            self._check()

    return CancelOnModelOutput()


RED_FLAG_RE = re.compile(
    r"""socket|urllib|requests|httpx|subprocess|os\.system|shutil\.rmtree|os\.remove|open\(.*["']w"""
)
"""Snippets that touch the network, other processes or files get a red warning in the prompt."""
DEMO_PROMPT_RE = re.compile(r"^\s*\d+\.\s+`(.+?)`\s*$")
"""The numbered, backticked prompts of ``workspace/README.md``."""
PERCENT_RE = re.compile(r"\d+\s*%")
"""Load-progress lines (1.0 engines send about a hundred of them) go to the spinner, not stdout."""
MODEL_SOURCE_REPL = "/model"
"""``Settings.sources["model"]`` after a switch from the REPL."""

HELP_TEXT = """\
/new                       start a new conversation (fresh thread, same model)
/status                    engine, model, speed and context in one line
/verbose                   toggle full tool results and full thinking text
/model ID [--keep-loaded]  switch the loaded model (the previous one is unloaded unless
                           --keep-loaded), then re-run the tool probe
/models                    list registered and loaded models
/help                      this list
/quit                      exit (also Ctrl-D, Ctrl-Z Enter on Windows, or Ctrl-C at the prompt)
Ctrl-C while the model is answering cancels that turn and keeps the session."""
UNATTENDED_NETWORK_WARNING = (
    "--auto-approve with --allow-network: model-written code runs unattended and may open "
    "network connections; file contents are untrusted input"
)

app = typer.Typer(
    help="A private local assistant running on the mimOE Studio endpoint.",
    add_completion=False,
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)
models_app = typer.Typer(help="Model registry: list, pull, load (use) and unload models.")
app.add_typer(models_app, name="models")


# -- typer commands ------------------------------------------------------------------------------


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    workspace: str | None = typer.Option(
        None,
        "--workspace",
        metavar="PATH",
        help="Folder the tools may read and run in (default: ./workspace).",
    ),
    model: str | None = typer.Option(
        None, "--model", help="Loaded model id (default: the first loaded chat model)."
    ),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        help="mimOE OpenAI-compatible base URL (default: auto-discover on localhost:8083).",
    ),
    api_key: str | None = typer.Option(None, "--api-key", help="mimOE API key (default 1234)."),
    think: bool | None = typer.Option(
        None, "--think", help="Turn the model's thinking mode on (slower; shown dim)."
    ),
    auto_approve: bool | None = typer.Option(
        None, "--auto-approve", help="Run model-written Python without asking (trust the model)."
    ),
    allow_network: bool | None = typer.Option(
        None, "--allow-network", help="Let approved Python code open network connections."
    ),
    force_tools: bool | None = typer.Option(
        None, "--force-tools", help="Skip the start-up tool probe and offer the tools anyway."
    ),
    trace: bool | None = typer.Option(
        None, "--trace", help="Keep LangSmith tracing variables (default: forced off)."
    ),
    version: bool = typer.Option(False, "--version", help="Print the version and exit."),
) -> None:
    """Chat with the agent in the terminal (default), or run a subcommand.

    Flags override MIMOE_* environment variables, which override ./.env, which overrides the
    defaults; the banner shows where every setting came from.
    """
    if version:
        typer.echo(f"mimoe-agent {__version__}")
        raise typer.Exit()
    ctx.obj = {
        "workspace": workspace,
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "think": think,
        "auto_approve": auto_approve,
        "allow_network": allow_network,
        "force_tools": force_tools,
        "trace": trace,
    }
    if ctx.invoked_subcommand is None:
        raise typer.Exit(run_repl(ctx.obj))


@app.command()
def serve(
    ctx: typer.Context,
    port: int = typer.Option(8000, "--port", help="Port on 127.0.0.1 for the web UI."),
) -> None:
    """Run the FastAPI server and the web chat UI on http://127.0.0.1:PORT (loopback only)."""
    _reconfigure_streams()
    settings = _settings_or_exit(_overrides(ctx))
    apply_tracing_env(settings)
    from mimoe_agent import server  # after apply_tracing_env: the server imports LangChain

    server.run(settings, port)


@models_app.command("list")
def models_list(ctx: typer.Context) -> None:
    """List registered and loaded models with sizes, capabilities and the recommended default."""
    console = _console()
    _reconfigure_streams()
    client = _connect(_settings_or_exit(_overrides(ctx), need_workspace=False))
    try:
        table = models_table(
            client.engine, client.loaded_models(), client.registry_models(), width=console.width
        )
    except MimoeError as exc:
        _fail(exc.message, exc.hint)
    console.print(table)


@models_app.command("pull")
def models_pull(
    ctx: typer.Context,
    spec: str = typer.Argument(
        ..., help="Preset id (see `models list`) or owner/repo:QUANT on HuggingFace."
    ),
) -> None:
    """Download a model into the engine's registry (a preset id or owner/repo:QUANT)."""
    console = _console()
    _reconfigure_streams()
    client = _connect(_settings_or_exit(_overrides(ctx), need_workspace=False))
    preset = preset_by_id(spec)
    if (
        preset is not None
        and preset.id in V10_ONLY_PRESETS
        and client.engine.generation is not EngineGeneration.V10
    ):
        console.print(
            Text(
                f"warning: {preset.id} does not load on {client.engine.generation}-generation "
                "engines like this one; the download will succeed but `models use` will fail",
                style="yellow",
            )
        )
    progress = Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )
    with progress:
        task = progress.add_task(f"pull {spec}", total=1.0)

        def on_progress(text: str, fraction: float | None) -> None:
            if fraction is None:
                style = "yellow" if text.lower().startswith("warning") else "dim"
                progress.console.print(Text(text, style=style))
            else:
                progress.update(task, completed=fraction, description=text)

        try:
            entry = pull_model(client, spec, on_progress=on_progress)
        except MimoeError as exc:
            progress.stop()
            _fail(exc.message, exc.hint)
        except KeyboardInterrupt:
            progress.stop()
            console.print(
                Text(
                    "pull interrupted; run the same command again to download it again",
                    style="yellow",
                )
            )
            raise typer.Exit(EXIT_INTERRUPT) from None
    size = f" ({entry.size_bytes / 1e9:.1f} GB)" if entry.size_bytes else ""
    console.print(
        Text(f"{entry.id} is ready{size}; load it with: mimoe-agent models use {entry.id}")
    )


@models_app.command("use")
def models_use(
    ctx: typer.Context,
    model_id: str = typer.Argument(..., metavar="ID", help="A registered model id."),
    keep_loaded: bool = typer.Option(
        False, "--keep-loaded", help="Keep the previously loaded model(s) in memory."
    ),
) -> None:
    """Load a registered model (and unload the previous one unless --keep-loaded)."""
    console = _console()
    _reconfigure_streams()
    client = _connect(_settings_or_exit(_overrides(ctx), need_workspace=False))
    try:
        with console.status(Text(f"loading {model_id}", style="dim"), spinner=SPINNER) as status:
            loaded = _switch(
                console,
                client,
                model_id,
                keep_loaded=keep_loaded,
                on_status=_status_printer(console, status),
            )
    except MimoeError as exc:
        _fail(exc.message, exc.hint)
    console.print(Text(f"loaded {loaded.id}: {_model_facts(loaded)}"))


@models_app.command("unload")
def models_unload(
    ctx: typer.Context,
    model_id: str = typer.Argument(..., metavar="ID", help="A loaded model id."),
) -> None:
    """Unload a model from the engine's memory."""
    console = _console()
    _reconfigure_streams()
    client = _connect(_settings_or_exit(_overrides(ctx), need_workspace=False))
    try:
        client.unload_model(model_id)
    except MimoeError as exc:
        _fail(exc.message, exc.hint)
    console.print(Text(f"unloaded {model_id}"))


# -- the REPL ------------------------------------------------------------------------------------


def run_repl(overrides: Mapping[str, object]) -> int:
    """Start the interactive session and return the process exit code.

    Args:
        overrides: CLI flag values keyed by settings field; ``None`` means "not given".

    Returns:
        0 after ``/quit`` or EOF, 1 for a configuration or preflight failure, 130 for Ctrl-C at
        the prompt (or during start-up).
    """
    console = _console()
    _reconfigure_streams()
    try:
        settings = load_settings(overrides)
    except ConfigError as exc:
        print_failure(_stderr_console(), exc.message, exc.hint)
        return EXIT_ERROR
    apply_tracing_env(settings)
    client = MimoeClient(settings.base_url, settings.api_key)
    try:
        with console.status(
            Text("connecting to mimOE Studio", style="dim"), spinner=SPINNER
        ) as status:
            pre = preflight(settings, client=client, on_status=status.update)
            status.update(Text("building the agent", style="dim"))
            session = Session(settings, client, pre, console)
            session.build_agent()
    except MimoeError as exc:
        print_failure(_stderr_console(), exc.message, exc.hint)
        return EXIT_ERROR
    except KeyboardInterrupt:
        console.print()
        return EXIT_INTERRUPT
    console.print(banner(settings, pre, prompts=demo_prompts(settings.workspace)))
    return session.loop()


@dataclass
class TurnStats:
    """What one user turn produced across all of its approval segments."""

    elapsed_s: float = 0.0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    visible_chars: int = 0
    notices: int = 0
    errors: int = 0

    def add_done(self, event: Mapping[str, Any]) -> None:
        """Fold a ``done`` event into the totals."""
        self.elapsed_s += _as_float(event.get("elapsed_s"))
        usage = event.get("usage_total")
        if isinstance(usage, Mapping):
            self.llm_calls += _as_int(usage.get("llm_calls"))
            self.input_tokens += _as_int(usage.get("input_tokens"))
            self.output_tokens += _as_int(usage.get("output_tokens"))

    def summary(self, *, verbose: bool) -> str:
        """``elapsed 6.3 s, 2 model calls, 91 tokens`` (plus prompt tokens when verbose)."""
        calls = f"{self.llm_calls} model call{'' if self.llm_calls == 1 else 's'}"
        line = f"elapsed {self.elapsed_s:.1f} s, {calls}, {self.output_tokens} tokens"
        if verbose:
            line += f" ({self.input_tokens} prompt tokens)"
        return line


class Renderer:
    """Render one ``iter_events`` segment (a turn, or the part after a resume) to the console.

    Exactly one rich live display is active at a time: the spinner while waiting, or the
    Markdown answer while tokens stream. :meth:`close` ends whatever is open, so a
    ``KeyboardInterrupt`` never leaves the console in live mode.
    """

    def __init__(
        self, console: Console, stats: TurnStats, *, verbose: bool, show_thinking: bool
    ) -> None:
        self.console = console
        self.stats = stats
        self.verbose = verbose
        self.show_thinking = show_thinking
        self.status = "completed"
        """``completed``, ``awaiting_approval`` or ``error`` once the segment ended."""
        self.approval: dict[str, Any] | None = None
        """The ``approval_required`` event, when the segment ended on one."""
        self._spinner: Status | None = None
        self._live: Live | None = None
        self._answer = ""
        self._thinking = ""
        self._rendered_at = 0.0
        """``time.monotonic()`` of the last live Markdown re-parse (throttled, see ``_token``)."""

    # -- events ------------------------------------------------------------------------------

    def handle(self, event: Mapping[str, Any]) -> None:
        """Render one event from :func:`mimoe_agent.stream.iter_events`.

        Every string in it came from the model, a tool or the engine, so control sequences are
        escaped before anything reaches the terminal (``approval_required`` keeps the raw code,
        which ``ask_decisions`` escapes and flags itself).
        """
        if event.get("event") != "approval_required":
            event = _safe_value(event)
        kind = event.get("event")
        if kind == "token":
            self._token(str(event.get("text") or ""))
        elif kind == "thinking":
            self._thought(str(event.get("text") or ""))
        elif kind == "tool_call":
            self._flush()
            name = str(event.get("name") or "tool")
            args = event.get("args")
            rendered = _format_args(args) if isinstance(args, Mapping) else ""
            self.console.print(Text(f"-> {name}({rendered})", style="dim"))
            self.spin(f"running {name}")
        elif kind == "tool_result":
            self._flush()
            self._result(event)
            self.spin("thinking")
        elif kind == "approval_required":
            self._flush()
            self.approval = dict(event)
        elif kind == "notice":
            self._flush()
            self.console.print(Text(str(event.get("text") or ""), style="yellow"))
            self.stats.notices += 1
        elif kind == "done":
            self._flush()
            self.status = str(event.get("status") or "completed")
            self.stats.add_done(event)
        elif kind == "error":
            self._flush()
            self.status = "error"
            self.stats.errors += 1
            print_failure(
                self.console,
                str(event.get("message") or "unknown error"),
                str(event.get("hint") or ""),
            )

    def spin(self, text: str) -> None:
        """Show (or retitle) the waiting spinner."""
        label = Text(text, style="dim")
        if self._spinner is None:
            self._spinner = self.console.status(label, spinner=SPINNER)
            self._spinner.start()
        else:
            self._spinner.update(label)

    def close(self) -> None:
        """End every live region (called in ``finally``, also after Ctrl-C)."""
        self._flush()

    # -- internals ---------------------------------------------------------------------------

    def _flush(self) -> None:
        self._stop_spinner()
        self._end_thinking()
        self._end_answer()

    def _stop_spinner(self) -> None:
        if self._spinner is not None:
            spinner, self._spinner = self._spinner, None
            spinner.stop()

    def _token(self, text: str) -> None:
        if not text:
            return
        self._stop_spinner()
        self._end_thinking()
        self._answer += text
        if not self.console.is_interactive:
            return  # a pipe or a dumb terminal gets the answer rendered once, at the end
        if self._live is None:
            self._live = Live(
                Markdown(""),
                console=self.console,
                refresh_per_second=REFRESH_PER_SECOND,
                redirect_stdout=False,
                redirect_stderr=False,
            )
            self._live.start()
        # Markdown() parses the whole answer on construction; doing that for every token is
        # quadratic in the answer length, so re-parse at most as often as the display refreshes
        # (the final, complete render happens in _end_answer).
        now = time.monotonic()
        if now - self._rendered_at >= 1 / REFRESH_PER_SECOND:
            self._rendered_at = now
            self._live.update(Markdown(self._answer))

    def _end_answer(self) -> None:
        if not self._answer:
            return
        answer, self._answer = self._answer, ""
        self.stats.visible_chars += len(answer)
        if self._live is None:
            self.console.print(Markdown(answer))
            return
        live, self._live = self._live, None
        live.update(Markdown(answer))
        live.stop()

    def _thought(self, text: str) -> None:
        if not text or not self.show_thinking:
            return  # thinking stays behind the spinner unless --think was given
        if self.verbose:
            self._stop_spinner()
            if not self._thinking:
                self.console.print(Text("thinking:", style="dim italic"))
            self.console.print(Text(text, style="dim"), end="", soft_wrap=True)
        else:
            self.spin(f"thinking ({len(self._thinking) + len(text)} chars)")
        self._thinking += text

    def _end_thinking(self) -> None:
        if not self._thinking:
            return
        text, self._thinking = self._thinking, ""
        if self.verbose:
            self.console.print()
            return
        self._stop_spinner()
        first = next((line.strip() for line in text.splitlines() if line.strip()), "")
        preview = _cut(first, THINKING_PREVIEW_CHARS)
        self.console.print(Text(f"thought ({len(text)} chars): {preview}", style="dim"))

    def _result(self, event: Mapping[str, Any]) -> None:
        content = str(event.get("content") or "")
        style = "dim red" if event.get("is_error") else "dim"
        lines = content.splitlines() or [""]
        shown = lines if self.verbose else lines[:RESULT_PREVIEW_LINES]
        width = max(self.console.width - 6, 20)
        for line in shown:
            shown_line = line if self.verbose else _cut(line, width)
            self.console.print(Text("   " + shown_line, style=style))
        hidden = len(lines) - len(shown)
        if hidden > 0:
            plural = "" if hidden == 1 else "s"
            self.console.print(
                Text(f"   ... (+{hidden} more line{plural}; /verbose shows all)", style=style)
            )


@dataclass
class Session:
    """One REPL session: settings, engine client, preflight result and the compiled agent."""

    settings: Settings
    client: MimoeClient
    pre: Preflight
    console: Console
    verbose: bool = False
    thread_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    agent: Any = None
    checkpointer: Any = None
    _pending: threading.Thread | None = None
    """A cancelled turn's worker that was still winding down when the prompt came back."""
    _repair_after_pending: bool = False

    @property
    def run_config(self) -> dict[str, Any]:
        """Graph config: the thread plus the model name for notice-only turns."""
        return {
            "configurable": {"thread_id": self.thread_id},
            "metadata": {"model": self.pre.model.id},
        }

    def turn_config(self, cancel: threading.Event) -> dict[str, Any]:
        """``run_config`` plus this turn's cancel signal: ``run_python`` kills its snippet when
        the event is set, and the callback handler aborts the streaming model call."""
        from mimoe_agent.tools.run_python import CANCEL_KEY

        config = self.run_config
        config["configurable"][CANCEL_KEY] = cancel
        config["callbacks"] = [_cancel_handler(cancel)]
        return config

    def build_agent(self) -> None:
        """(Re)build the agent for the current settings and preflight, keeping the threads."""
        from langgraph.checkpoint.memory import InMemorySaver

        from mimoe_agent import llm
        from mimoe_agent.agent import build_agent
        from mimoe_agent.tools import build_tools

        if self.checkpointer is None:
            self.checkpointer = InMemorySaver()
        self.agent = build_agent(
            self.settings,
            self.pre,
            llm=llm.make_model(self.settings, self.pre),
            tools=build_tools(self.settings, self.client),
            checkpointer=self.checkpointer,
        )

    # -- main loop ---------------------------------------------------------------------------

    def loop(self) -> int:
        """Read prompts until ``/quit``, EOF (0) or Ctrl-C at the prompt (130)."""
        while True:
            try:
                line = read_line(self.console)
            except KeyboardInterrupt:
                self.console.print()
                return EXIT_INTERRUPT
            if line is None:
                return EXIT_OK
            text = line.strip()
            if not text:
                continue
            try:
                if text.startswith("/"):
                    if self.handle_command(text) is False:
                        return EXIT_OK
                else:
                    self.run_turn(text)
            except KeyboardInterrupt:
                if text.startswith("/"):
                    # a slash command never changes the thread, so nothing needs repairing
                    self.console.print()
                    self.console.print(Text("command interrupted", style="yellow"))
                else:
                    self.cancel_turn()

    def handle_command(self, line: str) -> bool:
        """Run a slash command; ``False`` means quit."""
        command, _, argument = line.partition(" ")
        command = command.lower()
        argument = argument.strip()
        if command in ("/quit", "/exit", "/q"):
            return False
        if command in ("/help", "/?"):
            self.console.print(Text(HELP_TEXT))
        elif command == "/new":
            self.thread_id = uuid.uuid4().hex
            self.console.print(Text(f"new conversation (thread {self.thread_id[:8]})", style="dim"))
        elif command == "/verbose":
            self.verbose = not self.verbose
            self.console.print(Text(f"verbose {'on' if self.verbose else 'off'}", style="dim"))
        elif command == "/status":
            self.console.print(Text(self.status_line()))
        elif command == "/models":
            self.show_models()
        elif command == "/model":
            words = argument.split()
            flags = words[1:]
            if not words or any(flag != "--keep-loaded" for flag in flags):
                self.console.print(Text("usage: /model ID [--keep-loaded]", style="yellow"))
            else:
                self.switch(words[0], keep_loaded="--keep-loaded" in flags)
        else:
            self.console.print(Text(f"unknown command {command}; /help lists them", style="yellow"))
        return True

    # -- turns -------------------------------------------------------------------------------

    def run_turn(self, text: str) -> None:
        """Send one user message and render the whole turn, approvals included."""
        from langchain_core.messages import HumanMessage
        from langgraph.types import Command

        stats = TurnStats()
        payload: Any = {"messages": [HumanMessage(text)]}
        attempt = 0
        while True:
            renderer = self._stream(payload, stats)
            if renderer.status != "awaiting_approval" or renderer.approval is None:
                break
            attempt += 1
            decisions = self.ask_decisions(renderer.approval, attempt)
            if not self.pending_interrupt():
                self.console.print(
                    Text("nothing is waiting for a decision; the turn ended", style="yellow")
                )
                break
            payload = Command(resume={"decisions": decisions})
        if renderer.status == "completed":
            if not stats.visible_chars and not stats.notices:
                self.console.print(Text(NO_ANSWER, style="yellow"))
            self.console.print(Text(stats.summary(verbose=self.verbose), style="dim"))

    def _stream(self, payload: Any, stats: TurnStats) -> Renderer:
        """Run one graph pass on a worker thread and render its events on this thread.

        LangGraph runs every node on a pool thread when events are streamed, and Python raises
        ``KeyboardInterrupt`` only on the main thread, so the main thread never blocks inside
        the graph: it waits on a queue with a short timeout. Ctrl-C then reaches it at once
        (also on Windows, where untimed waits ignore Ctrl-C), and ``_abort_turn`` sets the
        turn's cancel signal: the snippet's process tree is killed and the model stream ends.
        """
        from mimoe_agent import stream as stream_module

        self._settle_pending()
        renderer = Renderer(
            self.console, stats, verbose=self.verbose, show_thinking=self.settings.think
        )
        cancel = threading.Event()
        config = self.turn_config(cancel)
        inbox: queue.SimpleQueue[Any] = queue.SimpleQueue()

        def work() -> None:
            events = stream_module.iter_events(self.agent, payload, config)
            try:
                for event in events:
                    inbox.put(event)
            except BaseException as exc:  # iter_events turns Exceptions into error events
                inbox.put(_Failed(exc))
            finally:
                events.close()
                inbox.put(_DONE)

        worker = threading.Thread(target=work, name="mimoe-turn", daemon=True)
        interrupted = False
        try:
            renderer.spin("thinking")
            worker.start()
            while True:
                try:
                    item = inbox.get(timeout=TURN_POLL_S)
                except queue.Empty:
                    continue
                if item is _DONE:
                    break
                if isinstance(item, _Failed):
                    raise item.exc
                renderer.handle(item)
        except KeyboardInterrupt:
            interrupted = True
        finally:
            renderer.close()
        if interrupted:
            self._abort_turn(worker, cancel)
            raise KeyboardInterrupt
        return renderer

    def _abort_turn(self, worker: threading.Thread, cancel: threading.Event) -> None:
        """Signal the cancel and give the worker CANCEL_GRACE_S to wind down.

        The snippet is killed and the model stream ends at its next chunk; a worker still busy
        after the grace period (a CPU-only machine reading a long prompt sends no chunk for a
        while) is kept in ``_pending`` and settled before the next graph run.
        """
        cancel.set()
        deadline = time.monotonic() + CANCEL_GRACE_S
        with contextlib.suppress(KeyboardInterrupt):
            while worker.is_alive() and time.monotonic() < deadline:
                worker.join(TURN_POLL_S)
        if worker.is_alive():
            self._pending = worker

    def _settle_pending(self) -> None:
        """Wait for a cancelled turn that was still winding down, then mend its thread.

        Two runs must never share a thread. If the old one is not done after CANCEL_WAIT_S (or on
        Ctrl-C), the session moves to a new conversation and leaves it to finish on its own.
        """
        worker, self._pending = self._pending, None
        if worker is None:
            return
        if worker.is_alive():
            deadline = time.monotonic() + CANCEL_WAIT_S
            with (
                contextlib.suppress(KeyboardInterrupt),
                self.console.status(
                    Text("finishing the cancelled turn", style="dim"), spinner=SPINNER
                ),
            ):
                while worker.is_alive() and time.monotonic() < deadline:
                    worker.join(TURN_POLL_S)
        repair, self._repair_after_pending = self._repair_after_pending, False
        if worker.is_alive():
            self.thread_id = uuid.uuid4().hex
            self.console.print(
                Text(
                    "the cancelled turn is still running in the background; "
                    "continuing in a new conversation",
                    style="yellow",
                )
            )
        elif repair:
            self._repair_thread()

    def ask_decisions(self, approval: Mapping[str, Any], attempt: int) -> list[dict[str, str]]:
        """Show every action request and collect one approve/reject decision per request."""
        requests = [r for r in approval.get("action_requests") or [] if isinstance(r, Mapping)]
        configs = [c for c in approval.get("review_configs") or [] if isinstance(c, Mapping)]
        decisions: list[dict[str, str]] = []
        for index, request in enumerate(requests):
            name = str(request.get("name") or APPROVAL_TOOL)
            args = request.get("args") if isinstance(request.get("args"), Mapping) else {}
            title = f"approval request {index + 1}/{len(requests)}"
            if attempt > 1:
                title += f" (attempt {attempt})"
            self.console.print(Text(f"{title}: {name}", style="bold"))
            if name == APPROVAL_TOOL:
                code, lexer = str(args.get("code") or ""), "python"
            else:
                code, lexer = json.dumps(args, indent=2, ensure_ascii=False), "json"
            shown, hidden = terminal_safe(code, strict=True)
            self.console.print(Syntax(shown, lexer, line_numbers=True, word_wrap=True))
            if hidden:
                self.console.print(
                    Text(
                        f"warning: this code contains {hidden} invisible or terminal-control "
                        "character(s), shown escaped above; they can make code look different "
                        "from what runs",
                        style="bold red",
                    )
                )
            flags = sorted({match.group(0) for match in RED_FLAG_RE.finditer(code)})
            if flags:
                self.console.print(
                    Text(
                        "warning: this code touches the network, other processes or files "
                        f"({', '.join(flags)})",
                        style="bold red",
                    )
                )
            self.console.print(Text(APPROVAL_WARNING, style="yellow"))
            allowed = _allowed_decisions(configs, index, name)
            answer = self._ask("Run this code? [y/N] ")
            if answer in ("y", "yes") and "approve" in allowed:
                decisions.append({"type": "approve"})
                self.console.print(Text("approved", style="green"))
            else:
                decisions.append({"type": "reject", "message": REJECT_MESSAGE})
                self.console.print(
                    Text("not executed; the model is told the code did not run", style="yellow")
                )
        return decisions

    def _ask(self, prompt: str) -> str:
        """One line from stdin, lower-cased; EOF, Ctrl-C or a closed stdin count as a denial."""
        try:
            answer = self.console.input(Text(prompt, style="bold"))
        except (EOFError, KeyboardInterrupt):
            self.console.print()
            return ""
        if not _stdin_is_tty():
            self.console.print(Text(answer))
        return answer.strip().lower()

    def pending_interrupt(self) -> bool:
        """Whether the thread is stopped at an interrupt (only then may a resume be sent)."""
        try:
            snapshot = self.agent.get_state(self.run_config)
        except Exception:
            return False
        if getattr(snapshot, "interrupts", ()):
            return True
        return any(getattr(task, "interrupts", ()) for task in getattr(snapshot, "tasks", ()))

    def cancel_turn(self) -> None:
        """After Ctrl-C mid-turn: say so and mend a tool call the cancel left without a result."""
        self.console.print()
        self.console.print(Text("turn cancelled", style="yellow"))
        if self._pending is not None and self._pending.is_alive():
            self._repair_after_pending = True  # mended once the worker is done
            return
        self._repair_thread()

    def _repair_thread(self) -> None:
        """Append error results for tool calls whose run was cancelled.

        A cancelled ``tools`` step leaves the last ``AIMessage`` with tool calls that have no
        ``ToolMessage``; the next request would send the model a dangling call. Writing the
        results as the ``tools`` node keeps the thread consistent. Best effort: any failure here
        only leaves the thread as the cancel left it.
        """
        from langchain_core.messages import AIMessage, ToolMessage

        try:
            snapshot = self.agent.get_state(self.run_config)
            values = snapshot.values if isinstance(snapshot.values, Mapping) else {}
            messages = list(values.get("messages") or [])
            # The last message that asked for tools; only tool results may follow it (a later
            # human message would make it history that was already answered or repaired).
            for index in range(len(messages) - 1, -1, -1):
                message = messages[index]
                if isinstance(message, AIMessage) and message.tool_calls:
                    break
                if not isinstance(message, ToolMessage):
                    return
            else:
                return
            answered = {m.tool_call_id for m in messages[index + 1 :] if isinstance(m, ToolMessage)}
            results = [
                ToolMessage(
                    content=CANCELLED_RESULT,
                    tool_call_id=str(call.get("id") or ""),
                    name=str(call.get("name") or APPROVAL_TOOL),
                    status="error",
                )
                for call in messages[index].tool_calls
                if str(call.get("id") or "") not in answered
            ]
            if results:
                self.agent.update_state(self.run_config, {"messages": results}, as_node="tools")
        except Exception:
            return

    # -- model management --------------------------------------------------------------------

    def switch(self, model_id: str, *, keep_loaded: bool) -> None:
        """``/model``: load another model, re-run the preflight probe and rebuild the agent."""
        self._settle_pending()
        previous = self.pre.model.id
        switched = False
        try:
            with self.console.status(
                Text(f"switching to {model_id}", style="dim"), spinner=SPINNER
            ) as status:
                notify = _status_printer(self.console, status)
                _switch(
                    self.console, self.client, model_id, keep_loaded=keep_loaded, on_status=notify
                )
                switched = True
                settings = dataclasses.replace(
                    self.settings,
                    model=model_id,
                    sources={**self.settings.sources, "model": MODEL_SOURCE_REPL},
                )
                pre = preflight(settings, client=self.client, on_status=status.update)
                self.settings, self.pre = settings, pre
                self.build_agent()
        except (MimoeError, KeyboardInterrupt) as exc:
            if isinstance(exc, MimoeError):
                print_failure(self.console, exc.message, exc.hint)
            else:
                self.console.print()
                self.console.print(
                    Text(
                        f"switch interrupted; the engine may still be loading {model_id} "
                        "(/status shows what is loaded)",
                        style="yellow",
                    )
                )
            if switched:
                self.console.print(
                    Text(
                        f"the engine now has {model_id} loaded but this session still targets "
                        f"{previous}; `/model {previous}` switches back",
                        style="yellow",
                    )
                )
            return
        self.console.print(Text(f"now using {self.pre.model.id}: {_model_facts(self.pre.model)}"))
        self.console.print(preflight_notes(self.pre))

    def status_line(self) -> str:
        """``/status``: the current model with fresh engine numbers."""
        engine = self.pre.engine
        try:
            models = self.client.loaded_models()
        except MimoeError as exc:
            return f"mimOE: {exc.message}"
        current = next((m for m in models if m.id == self.pre.model.id), self.pre.model)
        others = [m.id for m in models if m.id != current.id]
        tools = "on" if self.pre.tools_enabled else "off (chat-only)"
        line = (
            f"{current.id} on {engine.node_name or 'unknown node'} "
            f"({engine.version or 'unknown version'}, {engine.generation}-generation API): "
            f"{_model_facts(current)}, tools {tools}, "
            f"thinking {'on' if self.settings.think else 'off'}, "
            f"conversation {self.thread_id[:8]}"
        )
        if others:
            line += f"; also loaded: {', '.join(others)}"
        return line

    def show_models(self) -> None:
        """``/models``: the registry table, with the probe's verdict on the current model."""
        try:
            table = models_table(
                self.pre.engine,
                self.client.loaded_models(),
                self.client.registry_models(),
                tools_known={self.pre.model.id: self.pre.tools_enabled},
                width=self.console.width,
            )
        except MimoeError as exc:
            print_failure(self.console, exc.message, exc.hint)
            return
        self.console.print(table)


# -- rendering helpers ---------------------------------------------------------------------------


def banner(settings: Settings, pre: Preflight, *, prompts: Sequence[str]) -> Panel:
    """The start-up panel: model, engine, node, workspace and modes, each with its source."""
    sources = settings.sources
    model, engine = pre.model, pre.engine
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold", no_wrap=True)
    grid.add_column(overflow="fold")

    def row(label: str, value: str, source: str | None = None) -> None:
        text = Text(value)
        if source:
            text.append(f"  ({source})", style="dim")
        grid.add_row(label, text)

    row("model", model.id, _source_label(sources, "model", default="default: first loaded"))
    row("speed", _speed(model))
    row("context", f"{model.max_context:,} tokens" if model.max_context else "unknown")
    row("engine", f"{engine.version or 'unknown version'}, {engine.generation}-generation API")
    row("node", engine.node_name or "unknown")
    row("endpoint", engine.base_url, _source_label(sources, "base_url", default="discovered"))
    row("workspace", str(settings.workspace), _source_label(sources, "workspace"))
    row(
        "approval",
        "auto: model-written Python runs without asking"
        if settings.auto_approve
        else "ask before running model-written Python",
        _source_label(sources, "auto_approve"),
    )
    row(
        "network",
        "allowed for run_python" if settings.allow_network else "blocked for run_python",
        _source_label(sources, "allow_network"),
    )
    row("thinking", "on" if settings.think else "off", _source_label(sources, "think"))
    row("tools", "on" if pre.tools_enabled else "off: chat-only mode")
    parts: list[Any] = [grid]
    notes = preflight_notes(pre)
    if settings.auto_approve and settings.allow_network:
        if notes.plain:
            notes.append("\n")
        notes.append(f"! {UNATTENDED_NETWORK_WARNING}", style="yellow")
    if notes.plain:
        parts.append(notes)
    if prompts:
        tries = Text("Try, in order:\n", style="bold")
        for number, prompt in enumerate(prompts, 1):
            tries.append(f"  {number}. {prompt}\n")
        parts.append(tries)
    parts.append(Text("Commands: /new /status /verbose /model ID /models /help /quit", style="dim"))
    return Panel(
        Group(*parts), title=f"mimoe-agent {__version__}", title_align="left", border_style="cyan"
    )


def preflight_notes(pre: Preflight) -> Text:
    """The probe verdict (dim) and the preflight warnings (yellow) as lines."""
    notes = Text()
    if pre.probe is not None:
        notes.append(f"probe: {pre.probe.detail}\n", style="dim")
    for warning in pre.warnings:
        notes.append(f"! {warning}\n", style="yellow")
    notes.rstrip()
    return notes


def demo_prompts(workspace: Path) -> list[str]:
    """The numbered, backticked prompts of ``README.md`` in the workspace (at most five)."""
    try:
        text = (workspace / "README.md").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    found = [match.group(1) for line in text.splitlines() if (match := DEMO_PROMPT_RE.match(line))]
    return found[:5]


def models_table(
    engine: EngineInfo,
    loaded: Sequence[LoadedModel],
    registry: Sequence[RegistryModel],
    *,
    tools_known: Mapping[str, bool] | None = None,
    width: int | None = None,
) -> Table:
    """Registry, loaded models and presets in one table; ``*`` marks the recommended default.

    Below NARROW_TABLE_WIDTH columns (80 is rich's width when stdout is not a terminal) the
    thinking and notes columns move out and the notes are listed under the table, so the model
    ids are never cut.
    """
    narrow = width is not None and width < NARROW_TABLE_WIDTH
    loaded_by_id = {m.id: m for m in loaded}
    registry_by_id = {m.id: m for m in registry}
    ids = [m.id for m in registry]
    ids += [m.id for m in loaded if m.id not in registry_by_id]
    ids += [p.id for p in PRESETS if p.id not in registry_by_id and p.id not in loaded_by_id]
    table = Table(
        title=f"models on {engine.base_url} ({engine.generation}-generation engine)",
        title_justify="left",
    )
    label_width = max((len(i) + 2 for i in ids), default=5)
    table.add_column("model", no_wrap=True, min_width=label_width)
    for column in ("size", "state", "context", "tok/s", "tools") + (
        () if narrow else ("thinking",)
    ):
        table.add_column(column, no_wrap=True)
    if not narrow:
        table.add_column("notes", overflow="fold")
    notes_below: list[str] = []
    for model_id in ids:
        entry, live = registry_by_id.get(model_id), loaded_by_id.get(model_id)
        preset = preset_by_id(model_id)
        size_bytes = entry.size_bytes if entry is not None and entry.size_bytes else None
        if size_bytes is None and preset is not None:
            size_bytes = int(preset.size_gb * 1e9)
        if live is not None:
            state = "loaded"
        elif entry is None:
            state = "not registered"
        elif entry.ready:
            state = "ready"
        else:
            state = "not downloaded"
        context = _registry_context(entry)
        if live is not None and live.max_context:
            context = f"{live.max_context:,}"
        tools: bool | None = live.supports_tools if live is not None else None
        if tools is None and tools_known and model_id in tools_known:
            tools = tools_known[model_id]
        notes: list[str] = []
        if preset is not None:
            notes.append(preset.note)
            if preset.id in V10_ONLY_PRESETS and engine.generation is not EngineGeneration.V10:
                notes.append("does not load on this engine")
        if entry is not None and entry.raw.get("statusMessage"):
            notes.append(str(entry.raw["statusMessage"]))
        cells = [
            f"{model_id} *" if model_id == DEFAULT_PRESET else model_id,
            f"{size_bytes / 1e9:.1f} GB" if size_bytes else "?",
            state,
            context,
            f"{live.tokens_per_second:.1f}" if live and live.tokens_per_second else "-",
            _yes_no(tools),
        ]
        if narrow:
            if notes:
                notes_below.append(f"{model_id}: {'; '.join(notes)}")
        else:
            cells += [
                _yes_no(live.thinking_supported if live is not None else None),
                "; ".join(notes),
            ]
        table.add_row(*cells)
    table.caption = (
        "* recommended default. tools/thinking: from the engine's capability list on 1.0 "
        "engines, from the start-up probe on 0.6 engines, ? when unknown. "
        "`mimoe-agent models pull ID` downloads a preset, `models use ID` loads it."
    )
    if notes_below:
        table.caption += "\n\n" + "\n".join(notes_below)
    table.caption_justify = "left"
    return table


def print_failure(console: Console, message: str, hint: str) -> None:
    """``error: <message>`` in red, then the hint in yellow."""
    console.print(Text(f"error: {message}", style="bold red"))
    if hint:
        console.print(Text(hint, style="yellow"))


def read_line(console: Console) -> str | None:
    """Read one REPL line; ``None`` on EOF. The line is echoed when stdin is not a terminal, so a
    piped session leaves a readable transcript."""
    try:
        line = console.input(Text(PROMPT, style="bold cyan"))
    except EOFError:
        console.print()
        return None
    if not _stdin_is_tty():
        console.print(Text(line))
    return line


# -- small helpers -------------------------------------------------------------------------------


def _console() -> Console:
    """A console bound to the current ``sys.stdout`` (tests swap it); no markup surprises from
    model text because every dynamic string is printed as ``Text``."""
    with contextlib.suppress(ImportError):
        if _stdin_is_tty():
            import readline  # noqa: F401  (line editing and history on POSIX terminals)
    return Console(highlight=False, emoji=False)


def _stderr_console() -> Console:
    return Console(stderr=True, highlight=False, emoji=False)


def _reconfigure_streams() -> None:
    """UTF-8 with replacement on stdin/stdout/stderr (Windows pipes default to the code page)."""
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(Exception):
                reconfigure(encoding="utf-8", errors="replace")


def _stdin_is_tty() -> bool:
    try:
        return bool(sys.stdin is not None and sys.stdin.isatty())
    except (AttributeError, ValueError, OSError):
        return False


def _overrides(ctx: typer.Context) -> Mapping[str, object]:
    return ctx.obj if isinstance(ctx.obj, Mapping) else {}


def _settings_or_exit(overrides: Mapping[str, object], *, need_workspace: bool = True) -> Settings:
    """``load_settings`` for a subcommand; a config problem prints its hint and exits with 1.

    The ``models`` commands never touch the workspace, so a missing default workspace is not
    an error for them: the current directory stands in.
    """
    try:
        try:
            return load_settings(overrides)
        except ConfigError as exc:
            if need_workspace or exc.message != "no workspace":
                raise
            return load_settings({**overrides, "workspace": Path.cwd()})
    except ConfigError as exc:
        _fail(exc.message, exc.hint)


def _connect(settings: Settings) -> MimoeClient:
    """One client with the engine discovered, or exit 1 with the hint."""
    client = MimoeClient(settings.base_url, settings.api_key)
    try:
        client.discover()
    except MimoeError as exc:
        _fail(exc.message, exc.hint)
    return client


def _fail(message: str, hint: str) -> Any:
    """Print a failure to stderr and exit with 1 (typed ``Any`` so callers can ``return`` it)."""
    print_failure(_stderr_console(), message, hint)
    raise typer.Exit(EXIT_ERROR)


def _switch(
    console: Console,
    client: MimoeClient,
    model_id: str,
    *,
    keep_loaded: bool,
    on_status: Callable[[str], None],
) -> LoadedModel:
    """``switch_model`` plus a note when the engine evicted a model ``--keep-loaded`` meant to keep.

    Studio 0.6.5 keeps one chat model in memory: loading a second one silently unloads the
    first (verified live), so ``--keep-loaded`` is a request the engine may not honour.
    """
    before = {m.id for m in client.loaded_models()} if keep_loaded else set()
    loaded = switch_model(client, model_id, unload_previous=not keep_loaded, on_status=on_status)
    if keep_loaded:
        evicted = sorted(before - {m.id for m in client.loaded_models()} - {loaded.id})
        if evicted:
            console.print(
                Text(
                    f"note: the engine unloaded {', '.join(evicted)} while loading {loaded.id} "
                    "(it keeps one model in memory), so --keep-loaded had no effect",
                    style="yellow",
                )
            )
    return loaded


def _status_printer(console: Console, status: Status) -> Callable[[str], None]:
    """Progress callback: percentage lines overwrite the spinner, everything else is printed."""

    def notify(text: str) -> None:
        if PERCENT_RE.search(text):
            status.update(Text(text, style="dim"))
        else:
            style = "yellow" if text.lower().startswith("warning") else "dim"
            console.print(Text(text, style=style))

    return notify


def _allowed_decisions(configs: Sequence[Mapping[str, Any]], index: int, name: str) -> set[str]:
    """Decisions the interrupt allows for action request ``index`` (by position, else by name)."""
    config: Mapping[str, Any] | None = configs[index] if index < len(configs) else None
    if config is None or config.get("action_name") != name:
        config = next((c for c in configs if c.get("action_name") == name), config)
    allowed = config.get("allowed_decisions") if config is not None else None
    if isinstance(allowed, list) and allowed:
        return {str(decision) for decision in allowed}
    return {"approve", "reject"}


def _format_args(args: Mapping[str, Any]) -> str:
    """``path='.', max_depth=4`` with long values shortened (a snippet stays on one line)."""
    parts = []
    for key, value in args.items():
        shown = repr(value) if isinstance(value, str) else json.dumps(value, default=str)
        parts.append(f"{key}={_cut(shown, ARG_PREVIEW_CHARS)}")
    return ", ".join(parts)


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: max(limit - 3, 0)] + "..."


def _source_label(sources: Mapping[str, str], name: str, *, default: str = "default") -> str:
    source = sources.get(name, "default")
    if source == "env":
        return f"{env_name(name)} env"
    if source == "default":
        return default
    return source


def _speed(model: LoadedModel) -> str:
    if model.tokens_per_second:
        text = f"{model.tokens_per_second:.1f} tokens/s"
        if model.avg_tokens_per_second:
            text += f" (avg {model.avg_tokens_per_second:.1f})"
        return text
    return "not measured yet"


def _model_facts(model: LoadedModel) -> str:
    """``35.4 tokens/s, context 12,000`` for status lines."""
    context = f"context {model.max_context:,}" if model.max_context else "context unknown"
    return f"{_speed(model)}, {context}"


def _registry_context(entry: RegistryModel | None) -> str:
    gguf = entry.raw.get("gguf") if entry is not None else None
    context = _as_int(gguf.get("initContextSize")) if isinstance(gguf, Mapping) else 0
    return f"{context:,}" if context > 0 else "-"


def _yes_no(value: bool | None) -> str:
    if value is None:
        return "?"
    return "yes" if value else "no"


def _as_int(value: object) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return 0


def _as_float(value: object) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


__all__ = [
    "Renderer",
    "Session",
    "TurnStats",
    "app",
    "banner",
    "demo_prompts",
    "models_table",
    "run_repl",
]

if __name__ == "__main__":  # pragma: no cover - `python -m mimoe_agent.cli`
    app()
