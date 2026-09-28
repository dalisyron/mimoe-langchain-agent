"""Offline tests for tools/system.py: now, git and mimoe_status (the calculator's tests are in
test_tools_calculator.py).

The git tests build a throwaway repository with an explicit identity and an isolated HOME, so
the developer's global git configuration cannot change the output; they are skipped when git is
not installed. Nothing here needs a running mimOE engine.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from enum import StrEnum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from mimoe_agent.tools import system
from mimoe_agent.tools.calculator import calculator
from mimoe_agent.tools.system import (
    GIT_RESULT_CAP,
    format_now,
    make_git,
    make_mimoe_status,
    mimoe_status_report,
    now,
    run_git,
)

GIT = shutil.which("git")
needs_git = pytest.mark.skipif(GIT is None, reason="git is not installed")


@dataclass
class _Workspace:
    """Stand-in for tools.workspace.Workspace: a resolved root plus a jailed resolve()."""

    root: Path

    def resolve(self, rel: str) -> Path:
        candidate = (self.root / rel).resolve(strict=False)
        if not candidate.is_relative_to(self.root):
            raise ValueError(f"{rel!r} escapes the workspace")
        return candidate


# --- now ----------------------------------------------------------------------------------------

_FIXED = datetime(2026, 9, 24, 12, 0, 0)
_PDT = timezone(timedelta(hours=-7), "PDT")


def _fixed_clock(zone: tzinfo | None) -> datetime:
    return _FIXED.replace(tzinfo=zone or _PDT)


@pytest.fixture
def frozen_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(system, "_current_time", _fixed_clock)


def test_now_fixed_zone(frozen_clock: None) -> None:
    assert format_now("Asia/Tokyo") == "Thursday 2026-09-24 12:00:00 JST (UTC+09:00), in Asia/Tokyo"
    assert format_now("UTC") == "Thursday 2026-09-24 12:00:00 UTC (UTC+00:00), in UTC"


def test_now_local_default(frozen_clock: None) -> None:
    assert format_now(None) == "Thursday 2026-09-24 12:00:00 PDT (UTC-07:00), local time"
    assert format_now("") == format_now("local") == format_now(None)


@pytest.mark.parametrize(
    ("key", "fragment"),
    [
        ("utc", "in UTC"),  # case-insensitive on every platform, not only case-insensitive disks
        ("europe/berlin", "CEST (UTC+02:00), in Europe/Berlin"),
        ("America/New York", "in America/New_York"),
        ("UTC+2", "(UTC+02:00), in UTC+02:00"),
        ("+05:30", "(UTC+05:30), in UTC+05:30"),
        ("GMT-3", "(UTC-03:00), in UTC-03:00"),
        ("UTC-0", "12:00:00 (UTC+00:00), in UTC+00:00"),  # a zero offset has no sign of its own
        ("null", ", local time"),  # small models sometimes send the string "null" for "no zone"
    ],
)
def test_now_tolerant_zone_spelling(frozen_clock: None, key: str, fragment: str) -> None:
    out = format_now(key)
    assert fragment in out, out


@pytest.mark.parametrize("key", ["Mars/Olympus", "../../etc/passwd", "/etc/localtime", "UTC+25"])
def test_now_invalid_zone_message(key: str) -> None:
    out = format_now(key)
    assert out.startswith("ERROR: unknown time zone") and "IANA" in out


def test_now_real_clock_format() -> None:
    out = format_now("UTC")
    assert re.fullmatch(r"\w+ \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC \(UTC\+00:00\), in UTC", out)
    assert str(datetime.now(UTC).year) in out
    assert format_now(None).endswith(", local time")


def test_now_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(zone: tzinfo | None) -> datetime:
        raise OSError("no clock")

    monkeypatch.setattr(system, "_current_time", broken)
    assert format_now("Asia/Tokyo") == "ERROR: could not read the clock (no clock)."


def test_now_tool_contract() -> None:
    assert now.name == "now"
    assert list(now.args) == ["timezone"]
    assert "UTC" in now.invoke({"timezone": "UTC"})
    assert now.invoke({}).endswith(", local time")


# --- git ----------------------------------------------------------------------------------------


def _git(cwd: Path, *args: str) -> str:
    assert GIT is not None
    cmd = [
        GIT,
        "-c",
        "user.name=Test User",
        "-c",
        "user.email=test@example.com",
        "-c",
        "commit.gpgsign=false",
        "-c",
        "core.autocrlf=false",
        "-c",
        "init.defaultBranch=main",
        *args,
    ]
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        check=True,
        timeout=30,
        stdin=subprocess.DEVNULL,
        env=_git_test_env(),
    )
    return proc.stdout.decode("utf-8", errors="replace")


# Variables that name another repository. A git hook (for example a pre-commit hook that runs
# this suite) or a linked worktree exports them; inherited, they send the helper's `git commit`
# to that repository, whose hook then runs the suite again: a runaway recursion.
_GIT_LOCATION_VARS = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_NAMESPACE",
    "GIT_PREFIX",
    "GIT_CONFIG",
    "GIT_CONFIG_PARAMETERS",
    "GIT_CONFIG_COUNT",
)


def _git_test_env() -> dict[str, str]:
    """os.environ for the helper's git: no repository-location variable, no hooks' GIT_* state."""
    return {
        key: value
        for key, value in os.environ.items()
        if key.upper() not in _GIT_LOCATION_VARS
        and not key.upper().startswith("GIT_CONFIG_KEY_")
        and not key.upper().startswith("GIT_CONFIG_VALUE_")
    }


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


@pytest.fixture
def git_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Hide the developer's global git config and stop git from walking above tmp_path."""
    home = tmp_path / "home"
    home.mkdir()
    for key in ("HOME", "USERPROFILE", "XDG_CONFIG_HOME"):
        monkeypatch.setenv(key, str(home))
    for key in ("GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))


@pytest.fixture
def repo(tmp_path: Path, git_env: None) -> _Workspace:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "core.autocrlf", "false")
    _write(root / "notes.md", "todo: one\n")
    _write(root / "src" / "app.py", "print('hi')\n")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "first commit")
    _write(root / "src" / "utils.py", "def helper():\n    return 1\n")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "second: add utils")
    return _Workspace(root=root.resolve())


@needs_git
def test_git_status_clean(repo: _Workspace) -> None:
    out = run_git(repo, "status")
    assert out == "## main\n(working tree clean)"


@needs_git
def test_git_status_reports_changes(repo: _Workspace) -> None:
    _write(repo.root / "notes.md", "todo: one\ntodo: two\n")
    _write(repo.root / "new.txt", "hello\n")
    out = run_git(repo, "status")
    assert out.startswith("## main\n")
    assert " M notes.md" in out and "?? new.txt" in out
    assert "untracked" in out  # legend for the two-letter codes
    assert run_git(repo, "status", "src") == "## main\n(no changes under src)"


@needs_git
def test_git_log(repo: _Workspace) -> None:
    lines = run_git(repo, "log").splitlines()
    assert len(lines) == 2
    assert re.fullmatch(r"[0-9a-f]{7,} second: add utils", lines[0])
    assert re.fullmatch(r"[0-9a-f]{7,} first commit", lines[1])
    assert "HEAD" not in lines[0]  # --no-decorate


@needs_git
def test_git_log_narrowed_by_path(repo: _Workspace) -> None:
    out = run_git(repo, "log", "src/utils.py")
    assert out.count("\n") == 0 and "second: add utils" in out
    assert run_git(repo, "log", "nope.txt") == "(no commits under nope.txt)"


@needs_git
def test_git_diff(repo: _Workspace) -> None:
    assert run_git(repo, "diff") == "(no uncommitted changes)"
    _write(repo.root / "notes.md", "todo: one\nadded line\n")
    out = run_git(repo, "diff")
    assert "notes.md |" in out  # --stat header first
    assert "--- a/notes.md" in out and "+++ b/notes.md" in out and "\n+added line" in out
    assert run_git(repo, "diff", "src") == "(no uncommitted changes under src)"


@needs_git
def test_git_diff_includes_staged_changes(repo: _Workspace) -> None:
    """Staged edits are uncommitted too: status shows them, so diff must not say "no changes"."""
    _write(repo.root / "notes.md", "todo: one\nstaged line\n")
    _write(repo.root / "added.txt", "brand new\n")
    _git(repo.root, "add", "notes.md", "added.txt")
    _git(repo.root, "rm", "-q", "--cached", "src/utils.py")
    out = run_git(repo, "diff")
    assert "\n+staged line" in out and "--- a/notes.md" in out
    assert "new file mode" in out and "+brand new" in out
    assert "deleted file mode" in out and "-def helper():" in out
    assert run_git(repo, "status").splitlines()[1:4] == [
        "A  added.txt",
        "M  notes.md",
        "D  src/utils.py",
    ]
    _write(repo.root / "notes.md", "todo: one\nstaged line\nunstaged line\n")
    assert "+unstaged line" in run_git(repo, "diff", "notes.md")


@needs_git
def test_git_diff_is_capped_with_a_hint(repo: _Workspace) -> None:
    _write(repo.root / "big.txt", "".join(f"line {i}\n" for i in range(3000)))
    _git(repo.root, "add", ".")
    _git(repo.root, "commit", "-q", "-m", "big file")
    _write(repo.root / "big.txt", "".join(f"changed {i}\n" for i in range(3000)))
    out = run_git(repo, "diff")
    assert len(out) <= GIT_RESULT_CAP + 120
    assert "big.txt |" in out and "[truncated:" in out and "path=" in out


@needs_git
def test_git_empty_repository(tmp_path: Path, git_env: None) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    _git(root, "init", "-q")
    ws = _Workspace(root=root.resolve())
    assert run_git(ws, "log") == "(no commits yet in this repository)"
    assert run_git(ws, "status").startswith("## No commits yet on main")
    _write(root / "first.txt", "hello\n")
    _git(root, "add", "first.txt")
    # No HEAD to diff against yet: the tool falls back to index vs working tree, never an ERROR.
    assert run_git(ws, "diff") == "(no uncommitted changes)"
    _write(root / "first.txt", "hello\nworld\n")
    assert "+world" in run_git(ws, "diff")


@needs_git
def test_git_tool_contract(repo: _Workspace) -> None:
    tool = make_git(repo)
    assert tool.name == "git"
    assert tool.args["command"]["enum"] == ["status", "log", "diff"]
    assert tool.args["path"]["default"] is None
    assert "first commit" in tool.invoke({"command": "log"})
    assert tool.invoke({"command": "status", "path": "src"}).startswith("## main")
    with pytest.raises(ValidationError):  # the schema, not the tool body, rejects other commands
        tool.invoke({"command": "push"})
    assert run_git(repo, "push").startswith("ERROR: unknown git command")


@needs_git
def test_git_path_outside_workspace(repo: _Workspace) -> None:
    out = run_git(repo, "status", "../outside")
    assert out.startswith("ERROR: path '../outside' is not inside the workspace")


def test_git_not_a_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if GIT is None:
        pytest.skip("git is not installed")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    plain = tmp_path / "plain"
    plain.mkdir()
    out = run_git(_Workspace(root=plain.resolve()), "status")
    assert out.startswith("ERROR: not a git repository")


def test_git_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(system.shutil, "which", lambda *args, **kwargs: None)
    out = run_git(_Workspace(root=tmp_path), "status")
    assert out.startswith("ERROR: git is not installed")


@needs_git
def test_inherited_git_location_env_touches_no_other_repository(
    tmp_path: Path, git_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run from a git hook (GIT_DIR/GIT_INDEX_FILE exported), neither the helpers nor the tool
    may write to, or report on, that outer repository. No hook is installed here."""
    decoy = tmp_path / "decoy"
    decoy.mkdir()
    _git(decoy, "init", "-q")
    _write(decoy / "a.txt", "decoy\n")
    _git(decoy, "add", ".")
    _git(decoy, "commit", "-q", "-m", "decoy commit")
    decoy_head = _git(decoy, "rev-parse", "HEAD")

    monkeypatch.setenv("GIT_DIR", str(decoy / ".git"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(decoy / ".git" / "index"))
    monkeypatch.setenv("GIT_WORK_TREE", str(decoy))
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    _git(fresh, "init", "-q")
    _write(fresh / "x.txt", "x\n")
    _git(fresh, "add", ".")
    _git(fresh, "commit", "-q", "-m", "fresh commit")
    out = run_git(_Workspace(root=fresh.resolve()), "log")
    assert "fresh commit" in out and "decoy commit" not in out

    for name in ("GIT_DIR", "GIT_INDEX_FILE", "GIT_WORK_TREE"):
        monkeypatch.delenv(name)
    assert _git(decoy, "rev-parse", "HEAD") == decoy_head
    assert "fresh commit" not in _git(decoy, "log", "--oneline")


@needs_git
@pytest.mark.skipif(sys.platform == "win32", reason="the spy hook and filter are POSIX shell")
def test_git_tool_runs_no_repository_hook_or_filter(repo: _Workspace, tmp_path: Path) -> None:
    """status and diff rewrite the index (firing post-index-change) and run clean filters; a
    workspace repository must not get code executed that way. The spies only touch a file."""
    hook_ran, filter_ran = tmp_path / "hook-ran", tmp_path / "filter-ran"
    hook = repo.root / ".git" / "hooks" / "post-index-change"
    _write(hook, f"#!/bin/sh\ntouch '{hook_ran}'\n")
    hook.chmod(0o755)
    _git(repo.root, "config", "filter.spy.clean", f"sh -c 'touch {filter_ran}; cat'")
    _write(repo.root / ".gitattributes", "*.md filter=spy\n")
    notes = repo.root / "notes.md"
    for _ in range(2):
        stamp = notes.stat().st_mtime + 5
        os.utime(notes, (stamp, stamp))  # stat-dirty, same content: the case that runs both
        assert run_git(repo, "status").startswith("## ")
        run_git(repo, "diff")
    _write(notes, "todo: two\n")  # a real edit
    assert "todo: two" in run_git(repo, "diff")
    assert not hook_ran.exists(), "a repository hook ran"
    assert not filter_ran.exists(), "a repository filter ran"


def test_git_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(system.shutil, "which", lambda *args, **kwargs: str(tmp_path / "git"))

    def slow_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 20))

    monkeypatch.setattr(system.subprocess, "run", slow_run)
    out = run_git(_Workspace(root=tmp_path), "log")
    assert out.startswith("ERROR: git timed out after 20 s")


def test_git_cannot_start(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(system.shutil, "which", lambda *args, **kwargs: str(tmp_path / "git"))

    def denied(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        raise PermissionError("git: permission denied")

    monkeypatch.setattr(system.subprocess, "run", denied)
    assert run_git(_Workspace(root=tmp_path), "diff") == (
        "ERROR: could not run git (git: permission denied)."
    )


def test_git_failure_messages(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(system.shutil, "which", lambda *args, **kwargs: str(tmp_path / "git"))
    replies = iter(
        [
            (128, b"", b"warning: something\nfatal: detected dubious ownership in repository"),
            (128, b"", b"fatal: your current branch 'main' does not have any commits yet"),
            (0, "## main\n M caf\xc3\xa9.txt\r\n".encode("latin-1"), b""),
        ]
    )

    def scripted(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        if "config" in cmd and "--get-regexp" in cmd:  # the filter-driver lookup: none defined
            return subprocess.CompletedProcess(cmd, 1, b"", b"")
        assert kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0" and "--no-pager" in cmd
        assert cmd[cmd.index("-C") + 1] == str(tmp_path)
        code, out, err = next(replies)
        return subprocess.CompletedProcess(cmd, code, out, err)

    monkeypatch.setattr(system.subprocess, "run", scripted)
    ws = _Workspace(root=tmp_path)
    assert run_git(ws, "status") == (
        "ERROR: git status failed with exit code 128: "
        "fatal: detected dubious ownership in repository"
    )
    assert run_git(ws, "log") == "(no commits yet in this repository)"
    assert run_git(ws, "status").startswith("## main\n M café.txt\n")  # UTF-8, CRLF normalised


def _fake_which(mapping: dict[str, str | None]) -> Any:
    return lambda name, *args, **kwargs: mapping.get(name)


def test_git_macos_stub_without_command_line_tools(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(system.sys, "platform", "darwin")
    monkeypatch.setattr(
        system.shutil,
        "which",
        _fake_which({"git": "/usr/bin/git", "xcode-select": "/usr/bin/xcode-select"}),
    )
    executed: list[list[str]] = []

    def no_tools(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        executed.append(cmd)
        return subprocess.CompletedProcess(
            cmd, 2, b"", b"xcode-select: error: unable to get active developer directory"
        )

    monkeypatch.setattr(system.subprocess, "run", no_tools)
    out = run_git(_Workspace(root=tmp_path), "status")
    assert "xcode-select --install" in out and out.startswith("ERROR:")
    assert executed == [["/usr/bin/xcode-select", "-p"]]  # git itself was never launched


def test_git_macos_stub_with_command_line_tools(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    developer_dir = tmp_path / "CommandLineTools"
    (developer_dir / "usr" / "bin").mkdir(parents=True)
    (developer_dir / "usr" / "bin" / "git").write_bytes(b"")
    monkeypatch.setattr(system.sys, "platform", "darwin")
    monkeypatch.setattr(
        system.shutil,
        "which",
        _fake_which({"git": "/usr/bin/git", "xcode-select": "/usr/bin/xcode-select"}),
    )

    def scripted(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        if cmd[0].endswith("xcode-select"):
            return subprocess.CompletedProcess(cmd, 0, f"{developer_dir}\n".encode(), b"")
        return subprocess.CompletedProcess(cmd, 0, b"## main\n", b"")

    monkeypatch.setattr(system.subprocess, "run", scripted)
    assert run_git(_Workspace(root=tmp_path), "status") == "## main\n(working tree clean)"


# --- mimoe_status -------------------------------------------------------------------------------


class _EngineError(Exception):
    """Shaped like mimoe.MimoeError: a message plus a hint."""

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


class _Generation(StrEnum):
    V06 = "0.6"
    V10 = "1.0"


def _engine(**overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "base_url": "http://localhost:8083/mimik-ai/openai/v1",
        "store_url": "http://localhost:8083/mimik-ai/store/v1",
        "rpc_url": "http://localhost:8083/jsonrpc/v1",
        "node_name": "fallback-node",
        "version": "v0.0.0 (fallback)",
        "generation": _Generation.V06,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _model(**overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "id": "qwen3-4b",
        "kind": "llm",
        "family": "qwen3",
        "max_context": 12000,
        "n_params": 4022468096,
        "tokens_per_second": 35.37,
        "avg_tokens_per_second": 32.27,
        "supports_tools": None,
        "thinking_supported": None,
        "thinking_can_disable": None,
        "raw": {"info": {"model_size": 2491323904}},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


_NODE = {"name": "MacBookPro.lan", "nodeId": "976350e2", "version": "v3.22.8 (developer edition)"}


@dataclass
class _Client:
    """Duck-typed MimoeClient; a BaseException stored in a field is raised by that method."""

    engine: Any = field(default_factory=_engine)
    models: Any = field(default_factory=lambda: [_model()])
    node: Any = field(default_factory=lambda: dict(_NODE))

    @staticmethod
    def _give(value: Any) -> Any:
        if isinstance(value, BaseException):
            raise value
        return value

    def discover(self) -> Any:
        return self._give(self.engine)

    def loaded_models(self) -> Any:
        return self._give(self.models)

    def node_info(self) -> Any:
        return self._give(self.node)


def test_mimoe_status_happy_path() -> None:
    out = mimoe_status_report(_Client())
    assert out.splitlines() == [
        "engine: http://localhost:8083/mimik-ai/openai/v1 (0.6-era engine)",
        "node: MacBookPro.lan, engine version v3.22.8 (developer edition)",
        "loaded models: 1",
        "- qwen3-4b (llm, qwen3 family): 4.0B params, 2.5 GB, max context 12000 tokens, "
        "35.4 tokens/s (avg 32.3)",
    ]


def test_mimoe_status_v10_capabilities_and_prefixed_id() -> None:
    model = _model(
        id="976350e2/qwen3.5-4b",
        family="qwen35",
        supports_tools=True,
        thinking_supported=True,
        thinking_can_disable=False,
        avg_tokens_per_second=None,
        raw={},
    )
    client = _Client(engine=_engine(generation=_Generation.V10), models=[model])
    out = mimoe_status_report(client)
    assert "(1.0-era engine)" in out
    assert "- qwen3.5-4b (llm, qwen35 family): 4.0B params, max context 12000 tokens, " in out
    assert (
        "35.4 tokens/s, tool calling: yes, "
        "thinking: supported (the assistant keeps it off unless --think)" in out
    )
    assert " GB" not in out  # no size in raw -> no size claim


def test_mimoe_status_unreachable_engine() -> None:
    client = _Client(engine=_EngineError("mimOE Studio is not reachable", hint="open mimOE Studio"))
    assert mimoe_status_report(client) == (
        "mimOE engine: not reachable: mimOE Studio is not reachable (hint: open mimOE Studio). "
        "Is mimOE Studio running?"
    )
    plain = _Client(engine=ConnectionError("[Errno 61] Connection refused"))
    out = mimoe_status_report(plain)
    assert out.startswith("mimOE engine: not reachable: [Errno 61] Connection refused")


def test_mimoe_status_partial_failures() -> None:
    client = _Client(node=_EngineError("getMe failed"))
    lines = mimoe_status_report(client).splitlines()
    assert lines[1] == "node: details unavailable (getMe failed)"
    assert lines[2] == "node: fallback-node, engine version v0.0.0 (fallback)"
    assert lines[3] == "loaded models: 1" and lines[4].startswith("- qwen3-4b")

    client = _Client(models=RuntimeError("models endpoint returned 503"))
    lines = mimoe_status_report(client).splitlines()
    assert lines[-1] == "loaded models: unavailable (models endpoint returned 503)"

    client = _Client(models=[])
    assert mimoe_status_report(client).endswith(
        "loaded models: none (in mimOE Studio open Models and click Load)"
    )


def test_mimoe_status_tolerates_odd_models() -> None:
    class _Broken:
        id = "broken"

        @property
        def kind(self) -> str:
            raise RuntimeError("no kind")

    bare = _model(tokens_per_second=None, avg_tokens_per_second=None, n_params=None, raw=None)
    out = mimoe_status_report(_Client(models=[bare, _Broken(), object()]))
    assert "- qwen3-4b (llm, qwen3 family): max context 12000 tokens, tokens/s not measured" in out
    assert "- broken: details unavailable (no kind)" in out
    assert "- unnamed model (model): tokens/s not measured yet" in out


def test_mimoe_status_never_raises_or_returns_empty() -> None:
    for client in (
        _Client(engine=_EngineError("", hint="")),
        _Client(engine=None, models=None, node=None),
        _Client(node=None, models=[None]),
        _Client(models=OSError()),
    ):
        out = mimoe_status_report(client)
        assert isinstance(out, str) and out.strip()


def test_mimoe_status_tool_contract() -> None:
    client = _Client()
    tool = make_mimoe_status(client)
    assert tool.name == "mimoe_status"
    assert tool.args == {}
    assert "changes nothing" in tool.description
    assert tool.invoke({}) == mimoe_status_report(client)


# --- cross-tool contract ------------------------------------------------------------------------


def test_every_tool_result_is_a_non_empty_string(tmp_path: Path) -> None:
    tools = [calculator, now, make_git(_Workspace(root=tmp_path)), make_mimoe_status(_Client())]
    calls = [
        {"expression": ""},
        {"timezone": "Nowhere/Nope"},
        {"command": "diff", "path": "../.."},
        {},
    ]
    for tool, call in zip(tools, calls, strict=True):
        out = tool.invoke(call)
        assert isinstance(out, str) and out.strip(), tool.name
