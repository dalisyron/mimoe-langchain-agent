"""System tools: ``now``, ``git`` and ``mimoe_status`` (``calculator`` has a module of its own,
:mod:`mimoe_agent.tools.calculator`).

Every tool here returns a string, never raises and never returns ``""``: a failure becomes a
readable ``ERROR: ...`` line, because the model reads the result verbatim and small models
fabricate numbers when a tool result is empty.

* ``now`` uses ``zoneinfo``. The OS local zone needs no database; named zones come from the OS
  on macOS/Linux and from the ``tzdata`` package on Windows (declared in ``pyproject.toml``).
* ``git`` is one read-only tool (status / log / diff). It runs ``git`` with list arguments and
  ``shell=False``, disables pagers, prompts, the fsmonitor hook and external diff/textconv
  drivers, times out after ``GIT_TIMEOUT_S`` and caps its output. It reports on the repository
  that *contains* the workspace, which can be larger than the workspace itself (the sample
  workspace lives inside this project's repository).
* ``mimoe_status`` talks to the engine only through the ``discover``, ``loaded_models`` and
  ``node_info`` methods of the ``MimoeClient`` contract, typed as the ``StatusClient`` protocol,
  so this module never imports ``mimoe.py`` and any stub with those three methods works.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone, tzinfo
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones

from langchain_core.tools import BaseTool, StructuredTool, tool

__all__ = [
    "GIT_TIMEOUT_S",
    "StatusClient",
    "WorkspaceLike",
    "format_now",
    "make_git",
    "make_mimoe_status",
    "mimoe_status_report",
    "now",
    "run_git",
]


class _ToolError(Exception):
    """Internal control flow: carries a finished ``ERROR: ...`` line to the tool boundary."""


# --- now ----------------------------------------------------------------------------------------

_LOCAL_ALIASES = frozenset({"", "local", "system", "here", "none", "null", "default"})
_OFFSET_RE = re.compile(r"^(?:UTC|GMT)?\s*([+-])(\d{1,2})(?::?(\d{2}))?$", re.IGNORECASE)


def _current_time(zone: tzinfo | None) -> datetime:
    """The clock the ``now`` tool reads; tests replace it with a fixed instant."""
    return datetime.now(zone) if zone is not None else datetime.now().astimezone()


@lru_cache(maxsize=1)
def _zone_index() -> Mapping[str, str]:
    """Lower-cased IANA key -> canonical key, so 'utc' or 'europe/berlin' resolve everywhere."""
    try:
        return {name.lower(): name for name in available_timezones()}
    except Exception:  # no database at all (Windows without tzdata): exact keys still work
        return {}


def _resolve_zone(key: str) -> tuple[tzinfo, str] | None:
    """Find a zone by IANA key (case-insensitive, spaces tolerated) or a fixed 'UTC+5:30' offset."""
    offset = _OFFSET_RE.match(key)
    if offset:
        sign, hours, minutes = offset.group(1), int(offset.group(2)), int(offset.group(3) or 0)
        if hours > 23 or minutes > 59:
            return None
        delta = timedelta(hours=hours, minutes=minutes)
        sign = "+" if not delta else sign  # "UTC-0" is UTC+00:00, like _stamp() prints it
        label = f"UTC{sign}{hours:02d}:{minutes:02d}"
        return timezone(-delta if sign == "-" else delta, label), label
    candidate = key.replace(" ", "_")
    canonical = _zone_index().get(candidate.lower(), candidate)
    for name in dict.fromkeys((canonical, candidate)):
        try:
            return ZoneInfo(name), name
        except (ZoneInfoNotFoundError, ValueError, OSError):
            continue
    return None


def _stamp(moment: datetime) -> str:
    """'Friday 2026-09-25 10:15:42 PDT (UTC-07:00)'."""
    seconds = int((moment.utcoffset() or timedelta(0)).total_seconds())
    sign = "-" if seconds < 0 else "+"
    hours, minutes = divmod(abs(seconds) // 60, 60)
    offset = f"UTC{sign}{hours:02d}:{minutes:02d}"
    abbreviation = moment.tzname() or ""
    zone = f"{abbreviation} " if abbreviation and abbreviation != offset else ""
    return f"{moment:%A %Y-%m-%d %H:%M:%S} {zone}({offset})"


def format_now(timezone_name: str | None = None) -> str:
    """Current date and time as one readable line.

    Args:
        timezone_name: ``None``/``"local"`` for the OS zone, an IANA key such as ``"Asia/Tokyo"``
            (case-insensitive) or a fixed offset such as ``"UTC+2"``.

    Returns:
        A line like ``"Friday 2026-09-25 10:15:42 PDT (UTC-07:00), local time"`` or an
        ``ERROR: ...`` line for an unknown zone.
    """
    key = str(timezone_name or "").strip().strip("'\"")
    try:
        if key.lower() in _LOCAL_ALIASES:
            return f"{_stamp(_current_time(None))}, local time"
        found = _resolve_zone(key)
        if found is None:
            return (
                f"ERROR: unknown time zone {key!r}; use an IANA name such as 'UTC', "
                "'Europe/Berlin' or 'America/Toronto'."
            )
        zone, name = found
        return f"{_stamp(_current_time(zone))}, in {name}"
    except Exception as exc:  # a broken OS clock/zone must not take the agent down
        return f"ERROR: could not read the clock ({exc})."


@tool
def now(timezone: str | None = None) -> str:
    """Return the current date and time, by default in this computer's local time zone. Pass
    timezone as an IANA name such as 'UTC' or 'Europe/Berlin' to get the time elsewhere."""
    return format_now(timezone)


# --- git ----------------------------------------------------------------------------------------

GIT_TIMEOUT_S = 20.0
GIT_LOG_COUNT = 20
GIT_RESULT_CAP = 7_000  # chars; leaves room for the tail under GuardrailMiddleware's 8,000 cap
_GIT_COMMANDS = ("status", "log", "diff")
_GIT_STUB = "/usr/bin/git"  # macOS shim that launches the Command Line Tools installer
_FILTER_KEYS_RE = r"^filter\..*\.(clean|smudge|process)$"
"""Repository config keys naming filter commands (``git config --get-regexp`` pattern)."""
_STATUS_LEGEND = "(XY = index/worktree: M modified, A added, D deleted, R renamed, ?? untracked)"
_UNBORN_HEAD_MARKERS = ("ambiguous argument 'head'", "bad revision 'head'")
_CLT_HINT = (
    "ERROR: git cannot run yet: macOS needs the Xcode Command Line Tools. Run "
    "`xcode-select --install` in a terminal (or install git from https://git-scm.com), "
    "then ask again."
)


class WorkspaceLike(Protocol):
    """The part of ``tools.workspace.Workspace`` that the ``git`` tool relies on."""

    @property
    def root(self) -> Path:
        """Resolved absolute workspace directory."""
        ...

    def resolve(self, rel: str) -> Path:
        """Resolve a workspace-relative path; raises when it escapes the workspace."""
        ...


def _is_clt_stub(git_exe: str) -> bool:
    """True when ``git`` is Apple's ``/usr/bin/git`` shim (directly or through a symlink)."""
    return git_exe == _GIT_STUB or os.path.realpath(git_exe) == _GIT_STUB


def _clt_present() -> bool:
    """True when ``xcode-select -p`` names a developer directory that actually contains git.

    Querying ``xcode-select`` never opens the installer dialog; running the shim without the
    tools does, so this is checked first on macOS.
    """
    xcode_select = shutil.which("xcode-select")
    if xcode_select is None:
        return True  # cannot check; let git speak for itself
    try:
        proc = subprocess.run(
            [xcode_select, "-p"],
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return True
    if proc.returncode != 0:
        return False
    developer_dir = proc.stdout.decode("utf-8", errors="replace").strip()
    return bool(developer_dir) and Path(developer_dir, "usr", "bin", "git").is_file()


def _pathspec(ws: WorkspaceLike, path: str | None) -> str | None:
    """Turn the optional ``path`` argument into a workspace-relative git pathspec."""
    text = str(path or "").strip()
    if text in {"", ".", "./"}:
        return None
    try:
        rel = ws.resolve(text).relative_to(ws.root).as_posix()
    except Exception as exc:  # WorkspaceError, ValueError or OSError: all mean "not usable"
        raise _ToolError(f"ERROR: path {text!r} is not inside the workspace ({exc}).") from exc
    if rel in {"", "."}:
        return None
    return f"./{rel}" if rel[0] in ":-" else rel  # never let git read it as pathspec magic


def _git_args(command: str, pathspec: str | None, *, against_head: bool = True) -> list[str]:
    if command == "status":
        args = ["status", "--short", "--branch", "--ignore-submodules=all"]
    elif command == "log":
        args = ["log", "--oneline", "-n", str(GIT_LOG_COUNT), "--no-decorate"]
    else:
        # Against HEAD, so changes already staged with `git add` count as uncommitted too; a
        # plain `git diff` (index vs worktree) would report "(no uncommitted changes)" for them.
        args = ["diff", "--no-ext-diff", "--no-textconv", "--ignore-submodules=all", "--stat", "-p"]
        if against_head:
            args.append("HEAD")
    if pathspec is not None:
        args += ["--", pathspec]
    return args


def _git_env() -> dict[str, str]:
    """The environment for git: no inherited ``GIT_*`` variable, no prompt, no pager.

    A git hook, ``git rebase -x`` or a linked worktree exports ``GIT_DIR``, ``GIT_INDEX_FILE``
    and friends; inherited, they would point every command at that other repository instead of
    the workspace. None of them is needed for a local read-only command, so all are dropped.
    """
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
    env.update(
        {
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_PAGER": "cat",
            "PAGER": "cat",
            "LC_ALL": "C",
        }
    )
    return env


def _run(cmd: Sequence[str], env: Mapping[str, str]) -> subprocess.CompletedProcess[bytes]:
    extra: dict[str, Any] = {}
    if sys.platform == "win32":
        extra["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.run(
        list(cmd),
        capture_output=True,
        stdin=subprocess.DEVNULL,
        timeout=GIT_TIMEOUT_S,
        env=dict(env),
        check=False,
        **extra,
    )


def _filter_overrides(git_exe: str, root: Path, env: Mapping[str, str]) -> list[str]:
    """``-c`` options that disable every filter driver the repository's own config defines.

    ``filter.<name>.clean/smudge/process`` are shell commands git runs while it compares files,
    so an untrusted workspace repository could run code through a "read-only" status or diff.
    An empty command disables a driver. Only the repository's config is neutralised (drivers
    from your global config, such as Git LFS, stay as you set them up).
    """
    overrides: list[str] = []
    for scope in ("--local", "--worktree"):
        try:
            proc = _run(
                [git_exe, "-C", str(root), "config", scope, "--get-regexp", _FILTER_KEYS_RE], env
            )
        except (OSError, subprocess.SubprocessError):
            continue
        for line in proc.stdout.decode("utf-8", errors="replace").splitlines():
            key = line.split(" ", 1)[0]  # filter.<name>.<field>; <name> may contain dots
            name = key[len("filter.") : key.rfind(".")]
            if name and f"filter.{name}.clean=" not in overrides:
                for field in ("clean", "smudge", "process"):
                    overrides += ["-c", f"filter.{name}.{field}="]
                overrides += ["-c", f"filter.{name}.required=false"]
    return overrides


def _execute_git(git_exe: str, root: Path, args: Sequence[str]) -> tuple[int, str, str]:
    """Run git without a shell, pager, prompt, hook, filter or external driver; UTF-8 in and out.

    ``core.hooksPath`` points below the null device, where no hook can exist; this matters
    because ``git status`` and ``git diff`` rewrite the index, which fires ``post-index-change``.
    """
    env = _git_env()
    cmd = [
        git_exe,
        "--no-pager",
        "--no-optional-locks",
        "-C",
        str(root),
        "-c",
        f"core.hooksPath={os.devnull}",
        "-c",
        "core.quotepath=off",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "color.ui=never",
        "-c",
        "log.showSignature=false",
        *_filter_overrides(git_exe, root, env),
        *args,
    ]
    try:
        proc = _run(cmd, env)
    except subprocess.TimeoutExpired as exc:
        raise _ToolError(
            f"ERROR: git timed out after {GIT_TIMEOUT_S:.0f} s; the repository may be very large "
            "or on a slow disk."
        ) from exc
    except OSError as exc:
        raise _ToolError(f"ERROR: could not run git ({exc}).") from exc
    out = proc.stdout.decode("utf-8", errors="replace").replace("\r\n", "\n")
    err = proc.stderr.decode("utf-8", errors="replace").replace("\r\n", "\n").strip()
    return proc.returncode, out, err


def _explain_failure(command: str, code: int, err: str, root: Path) -> str:
    lowered = err.lower()
    if "not a git repository" in lowered:
        return f"ERROR: not a git repository: {root} has no .git folder (and neither has a parent)."
    if "does not have any commits yet" in lowered or "bad default revision" in lowered:
        return "(no commits yet in this repository)"
    lines = err.splitlines() or ["no error message"]
    fatal = (line for line in reversed(lines) if line.startswith(("fatal:", "error:")))
    detail = next(fatal, lines[0])
    return f"ERROR: git {command} failed with exit code {code}: {detail[:500]}"


def _cap(text: str, limit: int, hint: str) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n[truncated: {len(text):,} chars total; {hint}]"


def _shape_output(command: str, out: str, pathspec: str | None) -> str:
    text = out.strip("\n")
    where = f" under {pathspec}" if pathspec else ""
    if command == "status":
        lines = text.splitlines()
        if len(lines) <= 1:  # only the '## branch' line
            branch = lines[0] if lines else "## (unknown branch)"
            note = f"(no changes{where})" if pathspec else "(working tree clean)"
            return f"{branch}\n{note}"
        return _cap(f"{text}\n{_STATUS_LEGEND}", GIT_RESULT_CAP, "pass path=<folder> to narrow it")
    if command == "log":
        if not text:
            return f"(no commits{where})"
        return _cap(text, GIT_RESULT_CAP, "pass path=<file> to narrow it")
    if not text:
        return f"(no uncommitted changes{where})"
    return _cap(text, GIT_RESULT_CAP, "pass path=<file> to see one file")


def _find_git() -> str | None:
    """``git`` from PATH. On Windows the current directory is searched first by default (by
    ``shutil.which`` and by ``CreateProcess``), so a stray ``git.cmd`` there would run instead;
    ``NoDefaultCurrentDirectoryInExePath`` turns that search off for this process."""
    if sys.platform == "win32":
        os.environ.setdefault("NoDefaultCurrentDirectoryInExePath", "1")
    return shutil.which("git")


def run_git(ws: WorkspaceLike, command: str, path: str | None = None) -> str:
    """Run one read-only git command for the workspace and return its output as text.

    Args:
        ws: Workspace whose ``root`` git runs in; ``ws.resolve`` validates ``path``.
        command: ``"status"``, ``"log"`` or ``"diff"``.
        path: Optional file or folder, relative to the workspace, that narrows the output.

    Returns:
        git's output, a ``(...)`` note when there is nothing to show, or an ``ERROR: ...`` line.
        Never raises.
    """
    if command not in _GIT_COMMANDS:
        return f"ERROR: unknown git command {command!r}; use status, log or diff."
    try:
        git_exe = _find_git()
        if git_exe is None:
            return (
                "ERROR: git is not installed (no `git` on PATH); install it from "
                "https://git-scm.com and restart the assistant."
            )
        if sys.platform == "darwin" and _is_clt_stub(git_exe) and not _clt_present():
            return _CLT_HINT
        pathspec = _pathspec(ws, path)
        code, out, err = _execute_git(git_exe, ws.root, _git_args(command, pathspec))
        if command == "diff" and code != 0 and any(m in err.lower() for m in _UNBORN_HEAD_MARKERS):
            # No commit yet, so there is no HEAD to compare with: index vs working tree instead.
            args = _git_args(command, pathspec, against_head=False)
            code, out, err = _execute_git(git_exe, ws.root, args)
    except _ToolError as exc:
        return str(exc)
    except Exception as exc:  # the tool boundary: report, never raise
        return f"ERROR: git tool failed unexpectedly ({type(exc).__name__}: {exc})."
    if code != 0:
        return _explain_failure(command, code, err, ws.root)
    return _shape_output(command, out, pathspec)


def make_git(ws: WorkspaceLike) -> BaseTool:
    """Build the read-only ``git`` tool bound to one workspace."""

    def git(command: Literal["status", "log", "diff"], path: str | None = None) -> str:
        """Show the git status, the last 20 commits (log) or the uncommitted diff of the repository
        that contains the workspace. Read-only; path optionally narrows the result to one file or
        folder inside the workspace."""
        return run_git(ws, command, path)

    return StructuredTool.from_function(func=git, name="git")


# --- mimoe_status -------------------------------------------------------------------------------


class StatusClient(Protocol):
    """The three ``MimoeClient`` methods the status tool uses (see docs/CONTRACTS.md)."""

    def discover(self) -> Any:
        """Return an ``EngineInfo``-like object; raise when the engine is unreachable."""
        ...

    def loaded_models(self) -> Sequence[Any]:
        """Return ``LoadedModel``-like objects for what is loaded right now."""
        ...

    def node_info(self) -> Mapping[str, Any]:
        """Return the JSON-RPC ``getMe`` result (``name``, ``version``, ...) or ``{}``."""
        ...


def _describe_error(exc: BaseException) -> str:
    message = str(getattr(exc, "message", "") or "").strip() or str(exc).strip()
    message = message or type(exc).__name__
    hint = str(getattr(exc, "hint", "") or "").strip()
    return f"{message} (hint: {hint})" if hint else message


def _generation_label(generation: Any) -> str:
    value = str(getattr(generation, "value", generation) or "")
    labels = {"0.6": "0.6-era engine", "1.0": "1.0-era engine"}
    return labels.get(value, "engine generation unknown")


def _describe_model(model: Any) -> str:
    identifier = str(getattr(model, "id", None) or "unnamed model").rsplit("/", 1)[-1]
    kind = str(getattr(model, "kind", None) or "model")
    family = getattr(model, "family", None)
    head = f"{identifier} ({kind}{f', {family} family' if family else ''})"
    details: list[str] = []
    n_params = getattr(model, "n_params", None)
    if n_params:
        details.append(f"{n_params / 1e9:.1f}B params")
    raw = getattr(model, "raw", None)
    size = (raw.get("info") or {}).get("model_size") if isinstance(raw, Mapping) else None
    if isinstance(size, int | float) and size > 0:
        details.append(f"{size / 1e9:.1f} GB")
    max_context = getattr(model, "max_context", None)
    if max_context:
        details.append(f"max context {max_context} tokens")
    last = getattr(model, "tokens_per_second", None)
    average = getattr(model, "avg_tokens_per_second", None)
    if last is not None:
        speed = f"{last:.1f} tokens/s"
        details.append(f"{speed} (avg {average:.1f})" if average is not None else speed)
    elif average is not None:
        details.append(f"{average:.1f} tokens/s on average")
    else:
        details.append("tokens/s not measured yet")
    supports_tools = getattr(model, "supports_tools", None)
    if supports_tools is not None:
        details.append(f"tool calling: {'yes' if supports_tools else 'no'}")
    thinking = getattr(model, "thinking_supported", None)
    if thinking is False:
        details.append("thinking: not supported")
    elif thinking:
        details.append("thinking: supported (the assistant keeps it off unless --think)")
    return f"{head}: {', '.join(details)}"


def mimoe_status_report(client: StatusClient) -> str:
    """Describe the engine, node and loaded models; every failure becomes a readable line.

    Args:
        client: Anything with ``discover()``, ``loaded_models()`` and ``node_info()``.

    Returns:
        A few lines of text. Never raises and never returns an empty string.
    """
    try:
        engine = client.discover()
    except Exception as exc:
        return f"mimOE engine: not reachable: {_describe_error(exc)}. Is mimOE Studio running?"
    lines = [
        f"engine: {getattr(engine, 'base_url', None) or 'unknown endpoint'} "
        f"({_generation_label(getattr(engine, 'generation', None))})"
    ]
    name = getattr(engine, "node_name", None)
    version = getattr(engine, "version", None)
    try:
        info = client.node_info() or {}
        name = info.get("name") or name
        version = info.get("version") or version
    except Exception as exc:
        lines.append(f"node: details unavailable ({_describe_error(exc)})")
    lines.append(f"node: {name or 'unnamed'}, engine version {version or 'unknown'}")
    try:
        models = list(client.loaded_models())
    except Exception as exc:
        lines.append(f"loaded models: unavailable ({_describe_error(exc)})")
        return "\n".join(lines)
    if not models:
        lines.append("loaded models: none (in mimOE Studio open Models and click Load)")
        return "\n".join(lines)
    lines.append(f"loaded models: {len(models)}")
    for model in models:
        try:
            lines.append(f"- {_describe_model(model)}")
        except Exception as exc:
            lines.append(f"- {getattr(model, 'id', 'model')}: details unavailable ({exc})")
    return "\n".join(lines)


def make_mimoe_status(client: StatusClient) -> BaseTool:
    """Build the ``mimoe_status`` tool bound to an engine client."""

    def mimoe_status() -> str:
        """Report the local mimOE engine: endpoint, node name and version, and each loaded model
        with its context size and tokens per second. Takes no arguments and changes nothing."""
        return mimoe_status_report(client)

    return StructuredTool.from_function(func=mimoe_status, name="mimoe_status")
