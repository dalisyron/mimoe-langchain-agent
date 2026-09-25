"""run_python: execute model-written Python in a child interpreter. Not a sandbox.

The snippet runs as the current user with the workspace as its working directory; the
approval interrupt in the agent is the safety boundary. This module bounds accidents instead:
a wall-clock timeout and an output cap that kill the whole process tree, capture to temp files
(never pipes, so a grandchild cannot hang the parent), an allow-listed environment so the
parent's secrets never reach the child, and an 8 KB result for the model. The child side lives
in ``_runner.py``.

The tree kill is a process-group kill: on POSIX a grandchild that calls ``setsid`` itself
escapes it, and on Windows ``taskkill /T`` cannot reach grandchildren whose parent already
exited. Background processes the snippet leaves behind are swept on POSIX after a normal exit.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from langchain_core.tools import BaseTool, StructuredTool

if TYPE_CHECKING:
    from mimoe_agent.tools.workspace import Workspace

RESULT_CAP = 8_000
"""Characters of tool result handed to the model (GuardrailMiddleware caps at the same size)."""
OUTPUT_LIMIT = 32 * 1024 * 1024
"""Combined stdout+stderr bytes at which the process tree is killed."""
STREAM_CAP = 64 * 1024
"""Bytes of each stream kept in RunResult: the head of stdout, the tail of stderr."""
STDERR_TAIL = 3_000
"""Characters of stderr the formatted result keeps before stdout fills the rest."""
POLL_S = 0.25
"""Seconds between checks of the capture files while the child runs."""
NO_OUTPUT = "ERROR: the code produced no output. Print the values you need."

_KEEP_ENV = frozenset({"SYSTEMROOT", "PATH", "TEMP", "TMP", "HOME", "USERPROFILE", "LANG"})
_RUNNER = Path(__file__).with_name("_runner.py")
_WINDOWS = os.name == "nt"
_CREATE_NEW_PROCESS_GROUP = 0x00000200  # subprocess.CREATE_NEW_PROCESS_GROUP (Windows only)
_CREATE_NO_WINDOW = 0x08000000  # subprocess.CREATE_NO_WINDOW (Windows only)


@dataclass
class RunResult:
    """Outcome of one child run.

    ``stdout`` holds the first STREAM_CAP bytes of the child's stdout and ``stderr`` the last
    STREAM_CAP bytes of its stderr, both decoded as UTF-8 with replacement and CRLF-normalised.
    ``exit_code`` is negative on POSIX when the child died from a signal.
    """

    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool
    killed_for_size: bool


def _child_env(*, allow_network: bool) -> dict[str, str]:
    """Allow-listed environment for the child: no secrets, UTF-8 I/O, headless matplotlib."""
    env = {key: value for key, value in os.environ.items() if key.upper() in _KEEP_ENV}
    env.update(
        {
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",  # no __pycache__ litter when the workspace is imported
            "MPLBACKEND": "Agg",
        }
    )
    if allow_network:
        env["MIMOE_AGENT_ALLOW_NETWORK"] = "1"
    return env


def _popen_kwargs() -> dict[str, Any]:
    """Put the child in its own process group so the whole tree can be killed at once."""
    if _WINDOWS:
        return {"creationflags": _CREATE_NEW_PROCESS_GROUP | _CREATE_NO_WINDOW}
    return {"start_new_session": True}


def _killpg(pid: int) -> None:
    """SIGKILL the process group led by ``pid`` (start_new_session made pgid == pid)."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGKILL)


def _kill_tree(proc: subprocess.Popen[bytes]) -> None:
    """Kill the child and everything it spawned (best effort), then reap it."""
    if _WINDOWS:
        system_root = os.environ.get("SYSTEMROOT", r"C:\Windows")
        taskkill = Path(system_root, "System32", "taskkill.exe")
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run(
                [str(taskkill), "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=15,
                creationflags=_CREATE_NO_WINDOW,
            )
        if proc.poll() is None:
            proc.kill()
    else:
        _killpg(proc.pid)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _pause(proc: subprocess.Popen[bytes], seconds: float) -> None:
    """Wait up to ``seconds`` for the child to exit; returns early when it does."""
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=seconds)


def _supervise(
    proc: subprocess.Popen[bytes], out_file: IO[bytes], err_file: IO[bytes], timeout_s: float
) -> tuple[bool, bool]:
    """Poll every POLL_S until exit, timeout or output flood; returns (timed_out, killed)."""
    deadline = time.monotonic() + timeout_s
    while proc.poll() is None:
        captured = os.fstat(out_file.fileno()).st_size + os.fstat(err_file.fileno()).st_size
        if captured > OUTPUT_LIMIT:
            _kill_tree(proc)
            return False, True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _kill_tree(proc)
            return True, False
        _pause(proc, min(POLL_S, remaining))
    return False, False


def _decode(data: bytes) -> str:
    return data.decode("utf-8", errors="replace").replace("\r\n", "\n")


def _read_head(capture: IO[bytes], cap: int) -> str:
    """First ``cap`` bytes of a capture file; never loads the whole capture into memory.

    Reads go through the parent's own descriptor, so the result survives a snippet that
    unlinks its capture files; the shared offset may be moved because the child is dead.
    """
    capture.seek(0)
    return _decode(capture.read(cap))


def _read_tail(capture: IO[bytes], cap: int) -> str:
    """Last ``cap`` bytes of a capture file (the end of a traceback is the informative part)."""
    size = os.fstat(capture.fileno()).st_size
    capture.seek(max(size - cap, 0))
    return _decode(capture.read(cap))


def run_python_code(
    code: str, ws: Workspace, *, allow_network: bool, timeout_s: float
) -> RunResult:
    """Run ``code`` in a fresh interpreter with the workspace as its working directory.

    The child is ``sys.executable -X utf8 -u -P _runner.py snippet.py`` in its own process
    group, with stdin closed, stdout/stderr captured to temp files and an allow-listed
    environment. The capture files are polled every POLL_S seconds; the whole process tree is
    killed after ``timeout_s`` seconds or once the captures exceed OUTPUT_LIMIT bytes.

    Args:
        code: Python source written by the model.
        ws: The workspace; its root becomes the child's current directory and ``sys.path[0]``.
        allow_network: Lift the child's socket guard (see ``_runner.py``).
        timeout_s: Wall-clock limit for the run.

    Returns:
        A RunResult with the stdout head, stderr tail, exit code and kill flags.

    Raises:
        OSError: If the child could not be started (the tool wrapper turns this into text).
        BaseException: Whatever interrupted the wait (Ctrl-C, cancellation) is re-raised after
            the child tree has been killed.
    """
    root = ws.resolve(".")
    with tempfile.TemporaryDirectory(prefix="mimoe-run-", ignore_cleanup_errors=True) as tmp:
        snippet = Path(tmp, "snippet.py")
        snippet.write_text(code, encoding="utf-8", newline="\n")
        out_path, err_path = Path(tmp, "stdout.bin"), Path(tmp, "stderr.bin")
        with out_path.open("w+b") as out_file, err_path.open("w+b") as err_file:
            proc = subprocess.Popen(
                [sys.executable, "-X", "utf8", "-u", "-P", str(_RUNNER), str(snippet)],
                cwd=str(root),
                env=_child_env(allow_network=allow_network),
                stdin=subprocess.DEVNULL,
                stdout=out_file,
                stderr=err_file,
                **_popen_kwargs(),
            )
            try:
                timed_out, killed_for_size = _supervise(proc, out_file, err_file, timeout_s)
            except BaseException:  # Ctrl-C in the CLI, cancellation in the server
                _kill_tree(proc)
                raise
            if not _WINDOWS:
                _killpg(proc.pid)  # sweep daemons the snippet left behind in its process group
            return RunResult(
                stdout=_read_head(out_file, STREAM_CAP),
                stderr=_read_tail(err_file, STREAM_CAP),
                exit_code=proc.returncode,
                timed_out=timed_out,
                killed_for_size=killed_for_size,
            )


def format_result(result: RunResult, *, timeout_s: float) -> str:
    """Render a RunResult for the model in at most RESULT_CAP characters.

    Layout: an ``exit_code`` line with the kill flags, the head of stdout and the tail of
    stderr. A clean run with nothing on either stream returns NO_OUTPUT instead of an empty
    string, because an empty result made the model invent numbers.

    Args:
        result: The run to render.
        timeout_s: The limit the run had, so a timeout message can state it.

    Returns:
        Text for the ToolMessage, never empty.
    """
    stdout = result.stdout.rstrip()
    stderr = result.stderr.strip()
    if result.exit_code == 0 and not stdout and not stderr:
        return NO_OUTPUT
    header = f"exit_code: {result.exit_code}"
    if result.timed_out:
        header += f" (killed: timed out after {timeout_s:g} s)"
    if result.killed_for_size:
        header += f" (killed: output exceeded {OUTPUT_LIMIT // (1024 * 1024)} MB)"
    if len(stderr) > STDERR_TAIL:
        stderr = "[... earlier stderr omitted]\n" + stderr[-STDERR_TAIL:]
    stderr_block = f"stderr:\n{stderr}" if stderr else "stderr: (empty)"
    if stdout:
        marker = "\n[stdout truncated; print less or aggregate in code]"
        budget = RESULT_CAP - len(header) - len(stderr_block) - len("\nstdout:\n\n")
        if len(stdout) > budget:
            stdout = stdout[: max(budget - len(marker), 0)] + marker
        stdout_block = f"stdout:\n{stdout}"
    else:
        stdout_block = "stdout: (empty)"
    return f"{header}\n{stdout_block}\n{stderr_block}"


def make_run_python(ws: Workspace, *, allow_network: bool, timeout_s: float = 30.0) -> BaseTool:
    """Build the ``run_python(code)`` tool bound to a workspace.

    Args:
        ws: Workspace whose root is the child's working directory.
        allow_network: Whether the child's socket guard is lifted (``--allow-network``).
        timeout_s: Wall-clock limit per run.

    Returns:
        A StructuredTool named ``run_python`` that returns text and never raises.
    """
    network = "allowed" if allow_network else "disabled"
    description = (
        "Run a Python snippet with the workspace as the current directory and return what it "
        "prints; a trailing bare expression is echoed like in a notebook. Print the values you "
        "need; pandas and the standard library are available, output is capped at 8 KB, the run "
        f"is killed after {timeout_s:g} s, and network access is {network}."
    )

    def run_python(code: str) -> str:
        """Run ``code`` in the child interpreter and return the formatted result."""
        try:
            result = run_python_code(code, ws, allow_network=allow_network, timeout_s=timeout_s)
        except Exception as exc:
            return f"ERROR: run_python could not run the code: {exc}"
        return format_result(result, timeout_s=timeout_s)

    return StructuredTool.from_function(run_python, name="run_python", description=description)
