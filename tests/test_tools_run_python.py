"""Tests for tools.run_python and its child runner.

Offline: every test spawns real child interpreters but nothing touches the network beyond a
loopback HTTP server owned by the test. Windows-only branches (taskkill, creation flags) are
unit-tested by flipping the module's ``_WINDOWS`` switch; the process tests themselves also run
on windows-latest in CI.
"""

from __future__ import annotations

import http.server
import importlib
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from langchain_core.tools import BaseTool

from mimoe_agent.tools import run_python as rp
from mimoe_agent.tools.run_python import (
    NO_OUTPUT,
    RESULT_CAP,
    STREAM_CAP,
    RunResult,
    format_result,
    make_run_python,
    run_python_code,
)

POSIX = os.name != "nt"
NETWORK_DISABLED = "network disabled for run_python; ask to run with --allow-network"
SLEEPER = "import time; time.sleep(30)"
SALES_CSV = (
    "date,region,product,units,revenue\n"
    "2026-01-02,north,widget,10,100.0\n"
    "2026-01-03,south,widget,20,200.0\n"
    "2026-01-04,north,gadget,30,300.0\n"
    "2026-01-05,east,gadget,40,400.0\n"
    "2026-01-06,west,widget,13,136.6\n"
    "2026-01-07,south,gizmo,25,250.0\n"
    "2026-01-08,east,gizmo,25,250.0\n"
    "2026-01-09,west,gadget,20,200.0\n"
)


class FakeWorkspace:
    """The slice of tools.workspace.Workspace that run_python uses."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def resolve(self, rel: str) -> Path:
        return (self.root / rel).resolve()


@pytest.fixture
def ws(tmp_path: Path) -> FakeWorkspace:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "sales.csv").write_text(SALES_CSV, encoding="utf-8")
    return FakeWorkspace(root)


@pytest.fixture
def http_server() -> Iterator[tuple[int, list[str]]]:
    """Loopback HTTP server; yields (port, list of request paths seen)."""
    hits: list[str] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            hits.append(self.path)
            body = b"hello from the test server"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1], hits
    finally:
        server.shutdown()
        server.server_close()


def _run(
    code: str, ws: FakeWorkspace, *, allow_network: bool = False, timeout_s: float = 20.0
) -> RunResult:
    return run_python_code(code, ws, allow_network=allow_network, timeout_s=timeout_s)


def _pid_alive(pid: int) -> bool:
    if not POSIX:
        listing = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
            capture_output=True,
            text=True,
            check=False,
        )
        return f'"{pid}"' in listing.stdout
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    if sys.platform.startswith("linux"):  # a zombie answers kill(0); check its state
        try:
            status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
        except OSError:
            return False
        return "State:\tZ" not in status
    return True


def _wait_dead(pid: int, seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.05)
    return not _pid_alive(pid)


# --- notebook semantics ------------------------------------------------------------------------


def test_trailing_expression_is_echoed(ws: FakeWorkspace) -> None:
    res = _run("x = 20\ny = 22\nx + y", ws)
    assert res.exit_code == 0
    assert res.stdout == "42\n"
    assert res.stderr == ""
    assert not res.timed_out and not res.killed_for_size


def test_model_style_tuple_is_echoed(ws: FakeWorkspace) -> None:
    code = "rows = 8\naverage = 229.575\nrows, average"
    assert _run(code, ws).stdout == "(8, 229.575)\n"


def test_final_print_is_not_echoed_twice(ws: FakeWorkspace) -> None:
    assert _run('print("a")', ws).stdout == "a\n"


def test_none_expression_and_empty_code_give_no_output_error(ws: FakeWorkspace) -> None:
    tool = make_run_python(ws, allow_network=False, timeout_s=20.0)
    assert tool.invoke({"code": "x = None\nx"}) == NO_OUTPUT
    assert tool.invoke({"code": ""}) == NO_OUTPUT
    assert tool.invoke({"code": "# only a comment\n"}) == NO_OUTPUT
    assert tool.invoke({"code": "print()"}) == NO_OUTPUT


# --- errors ------------------------------------------------------------------------------------


def test_traceback_is_captured_without_runner_frames(ws: FakeWorkspace) -> None:
    res = _run("total = 1\nprint('before')\ntotal / 0\n", ws)
    assert res.exit_code == 1
    assert res.stdout == "before\n"
    assert "ZeroDivisionError: division by zero" in res.stderr
    assert 'File "snippet.py", line 3, in <module>' in res.stderr
    assert "total / 0" in res.stderr  # source line shown thanks to the linecache entry
    assert "_runner" not in res.stderr
    text = format_result(res, timeout_s=20.0)
    assert text.startswith("exit_code: 1\nstdout:\nbefore\nstderr:\n")
    assert "ZeroDivisionError" in text


def test_syntax_error_is_reported(ws: FakeWorkspace) -> None:
    res = _run("x = = 1\n", ws)
    assert res.exit_code == 1
    assert "SyntaxError" in res.stderr
    assert 'File "snippet.py", line 1' in res.stderr
    assert "_runner" not in res.stderr


def test_sys_exit_code_propagates(ws: FakeWorkspace) -> None:
    res = _run("import sys\nprint('bye')\nsys.exit(3)\n", ws)
    assert res.exit_code == 3
    assert res.stdout == "bye\n"
    assert res.stderr == ""


def test_stdin_is_closed(ws: FakeWorkspace) -> None:
    code = "try:\n    input()\nexcept EOFError:\n    print('eof')\n"
    assert _run(code, ws).stdout == "eof\n"


def test_traceback_survives_replaced_streams(ws: FakeWorkspace) -> None:
    code = (
        "import io, sys\n"
        "print('before')\n"
        "sys.stdout = None\n"
        "sys.stderr = io.StringIO()\n"
        "raise ValueError('boom')\n"
    )
    res = _run(code, ws)
    assert res.exit_code == 1
    assert res.stdout == "before\n"
    assert "ValueError: boom" in res.stderr
    assert 'File "snippet.py", line 5, in <module>' in res.stderr
    assert "_runner" not in res.stderr


@pytest.mark.skipif(not POSIX, reason="an open file cannot be unlinked on Windows")
def test_result_survives_snippet_unlinking_its_capture_files(ws: FakeWorkspace) -> None:
    code = (
        "import os, sys\n"
        "def path_of(fd):\n"
        "    if sys.platform == 'darwin':\n"
        "        import fcntl\n"
        "        raw = fcntl.fcntl(fd, fcntl.F_GETPATH, bytes(1024))\n"
        "        return raw.split(b'\\0', 1)[0].decode()\n"
        "    return os.readlink(f'/proc/self/fd/{fd}')\n"
        "out, err = path_of(1), path_of(2)\n"
        "print('before')\n"
        "os.unlink(out)\n"
        "os.unlink(err)\n"
        "print('after')\n"
        "raise RuntimeError('still captured')\n"
    )
    res = _run(code, ws)
    assert res.exit_code == 1
    assert res.stdout == "before\nafter\n"
    assert "RuntimeError: still captured" in res.stderr


# --- limits ------------------------------------------------------------------------------------


def test_timeout_kills_grandchild(ws: FakeWorkspace) -> None:
    code = (
        "import subprocess, sys, time\n"
        f"child = subprocess.Popen([sys.executable, '-c', {SLEEPER!r}])\n"
        "print(child.pid, flush=True)\n"
        "time.sleep(30)\n"
    )
    started = time.monotonic()
    res = _run(code, ws, timeout_s=2.0)  # room for a slow CI runner to spawn the grandchild
    elapsed = time.monotonic() - started
    assert res.timed_out and not res.killed_for_size
    assert 2.0 <= elapsed < 6.0
    assert res.stdout, "the child was killed before it printed the grandchild pid"
    grandchild = int(res.stdout.split()[0])
    assert _wait_dead(grandchild), f"grandchild {grandchild} survived the timeout"
    text = format_result(res, timeout_s=2.0)
    assert text.startswith(f"exit_code: {res.exit_code} (killed: timed out after 2 s)")


# 300 MB written byte by byte, so it is resident (a bare bytearray(n) may stay unmapped).
HOG = (
    "blob = b'x' * (300 * 1024 * 1024)\n"
    "print('allocated', flush=True)\n"
    "import time\n"
    "time.sleep(30)\n"
)


def test_memory_cap_kills_the_child(ws: FakeWorkspace) -> None:
    started = time.monotonic()
    res = run_python_code(HOG, ws, allow_network=False, timeout_s=30.0, memory_limit_mb=120)
    elapsed = time.monotonic() - started
    assert res.killed_for_memory and not res.timed_out and not res.cancelled
    assert elapsed < 10.0
    text = format_result(res, timeout_s=30.0, memory_limit_mb=120)
    assert "(killed: memory exceeded 120 MB; process the data in smaller pieces)" in text


def test_memory_cap_counts_grandchildren(ws: FakeWorkspace) -> None:
    code = (
        "import subprocess, sys, time\n"
        f"child = subprocess.Popen([sys.executable, '-c', {HOG!r}])\n"
        "print(child.pid, flush=True)\n"
        "time.sleep(30)\n"
    )
    res = run_python_code(code, ws, allow_network=False, timeout_s=30.0, memory_limit_mb=120)
    assert res.killed_for_memory
    grandchild = int(res.stdout.split()[0])
    assert _wait_dead(grandchild), f"grandchild {grandchild} survived the memory kill"


def test_normal_snippets_stay_well_under_the_default_cap(ws: FakeWorkspace) -> None:
    code = "import pandas as pd\nprint(len(pd.read_csv('sales.csv')))"
    res = run_python_code(code, ws, allow_network=False, timeout_s=30.0)
    assert res.exit_code == 0 and res.stdout.strip() == "8" and not res.killed_for_memory


def test_cancel_event_kills_the_tree(ws: FakeWorkspace) -> None:
    cancel = threading.Event()
    threading.Timer(1.0, cancel.set).start()
    started = time.monotonic()
    res = run_python_code(
        "print('started', flush=True)\n" + SLEEPER,
        ws,
        allow_network=False,
        timeout_s=30.0,
        cancel=cancel,
    )
    elapsed = time.monotonic() - started
    assert res.cancelled and not res.timed_out
    assert 0.9 <= elapsed < 5.0
    assert res.stdout.strip() == "started"
    assert "(killed: cancelled by the user" in format_result(res, timeout_s=30.0)


def test_tool_reads_the_cancel_event_from_the_run_config(ws: FakeWorkspace) -> None:
    tool = make_run_python(ws, allow_network=False)  # type: ignore[arg-type]
    assert list(tool.args) == ["code"], "the config parameter must not reach the model's schema"
    cancel = threading.Event()
    cancel.set()
    started = time.monotonic()
    out = tool.invoke({"code": SLEEPER}, config={"configurable": {rp.CANCEL_KEY: cancel}})
    assert time.monotonic() - started < 5.0
    assert "cancelled by the user" in out


def test_output_flood_is_killed_by_size_poll(ws: FakeWorkspace) -> None:
    code = "import time\nwhile True:\n    print('y' * 65536)\n    time.sleep(0.001)\n"
    started = time.monotonic()
    res = _run(code, ws, timeout_s=30.0)
    elapsed = time.monotonic() - started
    assert res.killed_for_size and not res.timed_out
    assert elapsed < 6.0
    assert len(res.stdout) <= STREAM_CAP
    text = format_result(res, timeout_s=30.0)
    assert "(killed: output exceeded 32 MB)" in text
    assert len(text) <= RESULT_CAP


def test_large_output_is_capped_head_and_tail(ws: FakeWorkspace) -> None:
    code = (
        "import sys\n"
        "print('x' * 200_000)\n"
        "for i in range(3000):\n"
        "    print(f'warn {i}', file=sys.stderr)\n"
    )
    res = _run(code, ws)
    assert res.exit_code == 0
    assert len(res.stdout) <= STREAM_CAP
    assert res.stderr.rstrip().endswith("warn 2999")
    text = format_result(res, timeout_s=20.0)
    assert len(text) <= RESULT_CAP
    assert "[stdout truncated; print less or aggregate in code]" in text
    assert "[... earlier stderr omitted]" in text
    assert text.rstrip().endswith("warn 2999")


@pytest.mark.skipif(not POSIX, reason="RLIMIT_FSIZE is POSIX only")
def test_file_size_limit_applies_to_snippet_writes(ws: FakeWorkspace) -> None:
    code = (
        "try:\n"
        "    with open('big.bin', 'wb') as fh:\n"
        "        for _ in range(70):\n"
        "            fh.write(bytes(1024 * 1024))\n"
        "    print('wrote everything')\n"
        "except OSError as exc:\n"
        "    print('blocked', exc.errno)\n"
    )
    res = _run(code, ws)
    assert res.stdout.startswith("blocked ")
    assert (ws.root / "big.bin").stat().st_size <= 64 * 1024 * 1024


@pytest.mark.skipif(not POSIX, reason="the post-exit sweep uses killpg")
def test_daemon_grandchild_is_swept_after_normal_exit(ws: FakeWorkspace) -> None:
    code = (
        "import subprocess, sys\n"
        f"child = subprocess.Popen([sys.executable, '-c', {SLEEPER!r}], "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "print(child.pid)\n"
    )
    res = _run(code, ws)
    assert res.exit_code == 0
    grandchild = int(res.stdout.split()[0])
    assert _wait_dead(grandchild), f"daemon grandchild {grandchild} survived"


def test_interrupt_while_waiting_kills_tree(
    ws: FakeWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid_file = ws.root / "pid.txt"

    def interrupted_pause(proc: subprocess.Popen[bytes], seconds: float) -> None:
        deadline = time.monotonic() + 15.0  # let the child spawn its grandchild and record the pid
        while time.monotonic() < deadline and not pid_file.exists():
            time.sleep(0.05)
        raise KeyboardInterrupt

    monkeypatch.setattr(rp, "_pause", interrupted_pause)
    code = (
        "import subprocess, sys, time\n"
        f"child = subprocess.Popen([sys.executable, '-c', {SLEEPER!r}])\n"
        "with open('pid.tmp', 'w') as fh:\n"
        "    fh.write(str(child.pid))\n"
        "import os; os.replace('pid.tmp', 'pid.txt')\n"
        "time.sleep(30)\n"
    )
    with pytest.raises(KeyboardInterrupt):
        _run(code, ws)
    grandchild = int(pid_file.read_text())
    assert _wait_dead(grandchild), f"grandchild {grandchild} survived Ctrl-C"


# --- isolation ---------------------------------------------------------------------------------


def test_secrets_do_not_reach_child(ws: FakeWorkspace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOO_SECRET", "hunter2")
    monkeypatch.setenv("PYTHONPATH", str(ws.root / "nope"))
    code = (
        "import os, sys\n"
        "print(os.environ.get('FOO_SECRET', 'absent'))\n"
        "print('PYTHONPATH' in os.environ)\n"
        "print(os.environ.get('PYTHONUTF8'), os.environ.get('MPLBACKEND'), sys.flags.utf8_mode)\n"
        "print(sorted(k for k in os.environ if 'SECRET' in k.upper()))\n"
    )
    lines = _run(code, ws).stdout.splitlines()
    assert lines == ["absent", "False", "1 Agg 1", "[]"]


def test_child_env_allow_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        os,
        "environ",
        {
            "SystemRoot": r"C:\Windows",
            "Path": "/usr/bin",
            "TEMP": "/t",
            "HOME": "/home/u",
            "LANG": "C.UTF-8",
            "OPENAI_API_KEY": "sk-secret",
            "PYTHONPATH": "/elsewhere",
            "AWS_SECRET_ACCESS_KEY": "x",
        },
    )
    env = rp._child_env(allow_network=False)
    assert env == {
        "SystemRoot": r"C:\Windows",
        "Path": "/usr/bin",
        "TEMP": "/t",
        "HOME": "/home/u",
        "LANG": "C.UTF-8",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "MPLBACKEND": "Agg",
    }
    assert rp._child_env(allow_network=True)["MIMOE_AGENT_ALLOW_NETWORK"] == "1"


def test_workspace_sitecustomize_is_not_executed(ws: FakeWorkspace) -> None:
    hook = "import pathlib\npathlib.Path('pwned.txt').write_text('x')\nprint('PWNED')\n"
    (ws.root / "sitecustomize.py").write_text(hook, encoding="utf-8")
    (ws.root / "usercustomize.py").write_text(hook, encoding="utf-8")
    res = _run("print('ok')", ws)
    assert res.stdout == "ok\n"
    assert not (ws.root / "pwned.txt").exists()


def test_agent_package_is_not_importable(ws: FakeWorkspace) -> None:
    code = (
        "import importlib\n"
        "for name in ('mimoe_agent', 'mimoe_agent.tools.run_python', 'run_python', '_runner'):\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "        print('IMPORTED', name)\n"
        "    except ImportError as exc:\n"
        "        print('blocked', name, type(exc).__name__)\n"
    )
    res = _run(code, ws)
    assert res.exit_code == 0
    assert "IMPORTED" not in res.stdout
    assert res.stdout.count("blocked") == 4


def test_socket_guard_blocks_local_http(
    ws: FakeWorkspace, http_server: tuple[int, list[str]]
) -> None:
    port, hits = http_server
    code = (
        "import ssl, urllib.request\n"
        f"print(urllib.request.urlopen('http://127.0.0.1:{port}/').read().decode())\n"
    )
    res = _run(code, ws)
    assert res.exit_code == 1
    assert NETWORK_DISABLED in res.stderr
    assert hits == []
    direct = _run(
        "import socket\ntry:\n    socket.socket()\nexcept RuntimeError as e:\n    print(e)", ws
    )
    assert direct.stdout.strip() == NETWORK_DISABLED


def test_allow_network_lifts_socket_guard(
    ws: FakeWorkspace, http_server: tuple[int, list[str]]
) -> None:
    port, hits = http_server
    code = (
        "import urllib.request\n"
        f"print(urllib.request.urlopen('http://127.0.0.1:{port}/').read().decode())\n"
    )
    res = _run(code, ws, allow_network=True)
    assert res.exit_code == 0, res.stderr
    assert res.stdout == "hello from the test server\n"
    assert hits == ["/"]


# --- environment of the child ------------------------------------------------------------------


def test_pandas_is_available(ws: FakeWorkspace) -> None:
    code = (
        "import pandas as pd\n"
        "df = pd.read_csv('sales.csv')\n"
        "print(len(df), round(float(df['revenue'].sum()), 1))\n"
    )
    res = _run(code, ws)
    assert res.exit_code == 0, res.stderr
    assert res.stdout == "8 1836.6\n"


def test_cwd_is_workspace_and_workspace_is_importable(ws: FakeWorkspace) -> None:
    (ws.root / "helper.py").write_text("VALUE = 7\n", encoding="utf-8")
    code = (
        "import os, sys\n"
        "import helper\n"
        "print(os.getcwd())\n"
        "print(helper.VALUE)\n"
        "print(sys.path[0])\n"
    )
    res = _run(code, ws)
    assert res.exit_code == 0, res.stderr
    cwd, value, first_path = res.stdout.splitlines()
    assert Path(cwd).resolve() == ws.root
    assert value == "7"
    assert Path(first_path).resolve() == ws.root
    assert not (ws.root / "__pycache__").exists()


def test_unicode_output_survives(ws: FakeWorkspace) -> None:
    text = "héllo wörld — ✓ 日本語 🐍"
    res = _run(f"s = {text!r}\nprint(s)\ns\n", ws)
    assert res.exit_code == 0, res.stderr
    assert res.stdout == f"{text}\n{text!r}\n"


def test_works_with_real_workspace_class(tmp_path: Path) -> None:
    workspace_module = importlib.import_module("mimoe_agent.tools.workspace")
    workspace_cls = getattr(workspace_module, "Workspace", None)
    if workspace_cls is None:
        pytest.skip("tools.workspace.Workspace is not implemented yet (step B1)")
    root = tmp_path / "ws"
    root.mkdir()
    res = run_python_code("print('hi')", workspace_cls(root), allow_network=False, timeout_s=20.0)
    assert res.stdout == "hi\n"


# --- formatting and the tool object ------------------------------------------------------------


def test_format_result_layout() -> None:
    ok = RunResult(stdout="42\n", stderr="", exit_code=0, timed_out=False, killed_for_size=False)
    assert format_result(ok, timeout_s=30.0) == "exit_code: 0\nstdout:\n42\nstderr: (empty)"
    failed = RunResult(
        stdout="", stderr="Traceback\nBoom\n", exit_code=1, timed_out=False, killed_for_size=False
    )
    assert (
        format_result(failed, timeout_s=30.0)
        == "exit_code: 1\nstdout: (empty)\nstderr:\nTraceback\nBoom"
    )
    silent = RunResult(stdout=" \n", stderr="", exit_code=0, timed_out=False, killed_for_size=False)
    assert format_result(silent, timeout_s=30.0) == NO_OUTPUT
    silent_failure = RunResult(
        stdout="", stderr="", exit_code=2, timed_out=False, killed_for_size=False
    )
    assert (
        format_result(silent_failure, timeout_s=30.0)
        == "exit_code: 2\nstdout: (empty)\nstderr: (empty)"
    )
    killed = RunResult(stdout="", stderr="", exit_code=-9, timed_out=True, killed_for_size=False)
    assert format_result(killed, timeout_s=30.0).startswith(
        "exit_code: -9 (killed: timed out after 30 s)"
    )
    flooded = RunResult(
        stdout="y" * 100, stderr="", exit_code=-9, timed_out=False, killed_for_size=True
    )
    assert "(killed: output exceeded 32 MB)" in format_result(flooded, timeout_s=30.0)


def test_format_result_never_exceeds_cap() -> None:
    res = RunResult(
        stdout="o" * STREAM_CAP,
        stderr="e" * STREAM_CAP,
        exit_code=1,
        timed_out=True,
        killed_for_size=True,
    )
    text = format_result(res, timeout_s=30.0)
    assert len(text) <= RESULT_CAP
    assert "[stdout truncated; print less or aggregate in code]" in text
    assert text.endswith("e" * 100)


def test_tool_metadata_and_invoke(ws: FakeWorkspace) -> None:
    tool = make_run_python(ws, allow_network=False)
    assert isinstance(tool, BaseTool)
    assert tool.name == "run_python"
    assert tool.args == {"code": {"title": "Code", "type": "string"}}
    assert "workspace as the current directory" in tool.description
    assert "pandas" in tool.description
    assert "killed after 30 s" in tool.description
    assert "network access is disabled" in tool.description
    assert "network access is allowed" in make_run_python(ws, allow_network=True).description
    assert tool.invoke({"code": "print(1)"}) == "exit_code: 0\nstdout:\n1\nstderr: (empty)"


def test_tool_never_raises(ws: FakeWorkspace, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> RunResult:
        raise OSError("no interpreter")

    monkeypatch.setattr(rp, "run_python_code", boom)
    tool = make_run_python(ws, allow_network=False)
    assert (
        tool.invoke({"code": "print(1)"})
        == "ERROR: run_python could not run the code: no interpreter"
    )


# --- platform branches (unit-tested on every OS by flipping the switch) ------------------------


def test_popen_kwargs_per_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rp, "_WINDOWS", True)
    assert rp._popen_kwargs() == {"creationflags": 0x00000200 | 0x08000000}
    monkeypatch.setattr(rp, "_WINDOWS", False)
    assert rp._popen_kwargs() == {"start_new_session": True}


class FakeProc:
    """Enough of Popen for _kill_tree: pid, poll, kill, wait."""

    def __init__(self, *, dies_on_taskkill: bool) -> None:
        self.pid = 4242
        self.alive = True
        self.dies_on_taskkill = dies_on_taskkill
        self.killed = False

    def poll(self) -> int | None:
        return None if self.alive else 0

    def kill(self) -> None:
        self.killed = True
        self.alive = False

    def wait(self, timeout: float | None = None) -> int:
        if self.alive:
            raise subprocess.TimeoutExpired("python", timeout or 0)
        return 0


def test_windows_kill_tree_uses_taskkill(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rp, "_WINDOWS", True)
    monkeypatch.setenv("SYSTEMROOT", r"C:\Windows")
    calls: list[tuple[list[str], dict[str, object]]] = []
    proc = FakeProc(dies_on_taskkill=True)

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append((argv, kwargs))
        if proc.dies_on_taskkill:
            proc.alive = False
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    rp._kill_tree(proc)  # type: ignore[arg-type]
    argv, kwargs = calls[0]
    assert argv[0].endswith("taskkill.exe") and argv[0].startswith(r"C:\Windows")
    assert argv[1:] == ["/F", "/T", "/PID", "4242"]
    assert kwargs["creationflags"] == 0x08000000 and kwargs["timeout"] == 15
    assert not proc.killed  # taskkill did the job; no TerminateProcess fallback needed

    survivor = FakeProc(dies_on_taskkill=False)
    proc = survivor
    rp._kill_tree(survivor)  # type: ignore[arg-type]
    assert survivor.killed  # fallback to Popen.kill when taskkill leaves it running


def test_windows_kill_tree_survives_missing_taskkill(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rp, "_WINDOWS", True)

    def missing(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise FileNotFoundError(argv[0])

    monkeypatch.setattr(subprocess, "run", missing)
    proc = FakeProc(dies_on_taskkill=False)
    rp._kill_tree(proc)  # type: ignore[arg-type]
    assert proc.killed and not proc.alive
