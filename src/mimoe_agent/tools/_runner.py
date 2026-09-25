"""Child-side runner for run_python. Standalone script: must not import mimoe_agent.

run_python.py launches it as ``python -X utf8 -u -P _runner.py SNIPPET`` with the workspace as
the current directory and an allow-listed environment. Everything here happens after interpreter
start-up, so nothing in the workspace runs before the approved snippet: the workspace goes on
``sys.path`` only now, the agent's own package is made unimportable, accidental network use is
blocked unless ``MIMOE_AGENT_ALLOW_NETWORK=1``, files the snippet writes are capped at 64 MB on
POSIX, and a trailing bare expression is echoed the way a notebook cell does. None of this is a
security boundary; the approval prompt in the agent is.
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import linecache
import os
import socket
import sys
import traceback
from pathlib import Path

SNIPPET_NAME = "snippet.py"
FILE_SIZE_LIMIT = 64 * 1024 * 1024
NETWORK_DISABLED = "network disabled for run_python; ask to run with --allow-network"
ALLOW_NETWORK_ENV = "MIMOE_AGENT_ALLOW_NETWORK"


def _block_network() -> None:
    """Make every socket constructor raise so accidental network use fails loudly.

    ``socket.socket`` must stay a class (``ssl`` subclasses it at import time, so a plain
    function there breaks ``import ssl`` and everything above it); only instantiating raises.
    Name resolution (``getaddrinfo``) is not blocked, and anything that needs a local socket
    pair (``asyncio`` event loops) fails with the same message. A snippet can still reach
    ``_socket`` directly: this is a guard, not a boundary.
    """

    class BlockedSocket(socket.socket):
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError(NETWORK_DISABLED)

    def blocked(*args: object, **kwargs: object) -> object:
        raise RuntimeError(NETWORK_DISABLED)

    socket.socket = BlockedSocket  # type: ignore[misc]
    socket.SocketType = BlockedSocket  # type: ignore[misc]
    socket.create_connection = blocked  # type: ignore[assignment]


def _limit_file_size() -> None:
    """Cap every file the snippet writes at FILE_SIZE_LIMIT (POSIX only; no-op on Windows).

    CPython ignores SIGXFSZ, so an oversized write raises ``OSError(EFBIG)`` instead of
    killing the process; the snippet sees a normal traceback.
    """
    try:
        import resource
    except ImportError:  # Windows has no resource module
        return
    try:
        _soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
        limit = FILE_SIZE_LIMIT if hard == resource.RLIM_INFINITY else min(FILE_SIZE_LIMIT, hard)
        resource.setrlimit(resource.RLIMIT_FSIZE, (limit, hard))
    except (ValueError, OSError):
        pass


def _execute(source: str) -> None:
    """Run ``source`` as ``__main__`` and echo a trailing bare expression notebook-style.

    If the last statement is an expression, it is evaluated after the rest and its ``repr`` is
    printed unless the value is ``None`` (so a final ``print(...)`` is not echoed twice).
    Line numbers are preserved because the original AST nodes are compiled.
    """
    tree = ast.parse(source, SNIPPET_NAME)
    body = tree.body
    trailing = body[-1] if body and isinstance(body[-1], ast.Expr) else None
    if trailing is not None:
        body = body[:-1]
    module_globals: dict[str, object] = {"__name__": "__main__", "__builtins__": builtins}
    if body:
        module = ast.Module(body=body, type_ignores=[])
        exec(compile(module, SNIPPET_NAME, "exec"), module_globals)
    if trailing is not None:
        expression = ast.Expression(body=trailing.value)
        value = eval(compile(expression, SNIPPET_NAME, "eval"), module_globals)
        if value is not None:
            print(repr(value))


def _print_snippet_traceback(exc: BaseException) -> None:
    """Print ``exc`` the way the interpreter would, minus this runner's own frames."""
    tb = exc.__traceback__
    while tb is not None and tb.tb_frame.f_code.co_filename == __file__:
        tb = tb.tb_next
    with contextlib.suppress(Exception):  # the snippet may have replaced or closed sys.stdout
        sys.stdout.flush()
    # The original stderr is the capture file even if the snippet rebound ``sys.stderr``.
    traceback.print_exception(type(exc), exc, tb, file=sys.__stderr__ or sys.stderr)


def main(argv: list[str]) -> int:
    """Prepare the interpreter, run the snippet file named in ``argv[1]``, return an exit code."""
    if len(argv) != 2:
        print(f"usage: {Path(argv[0]).name} SNIPPET", file=sys.stderr)
        return 2
    source = Path(argv[1]).read_text(encoding="utf-8")
    sys.argv = [SNIPPET_NAME]
    # The workspace joins sys.path only now, after site initialisation, so a sitecustomize.py
    # in it cannot run before the approved code; ``-P`` already kept this directory off the path.
    sys.path.insert(0, os.getcwd())
    # The editable install (or site-packages) still exposes the agent's package: a None entry
    # in sys.modules makes ``import mimoe_agent`` raise ImportError.
    sys.modules["mimoe_agent"] = None  # type: ignore[assignment]
    if os.environ.get(ALLOW_NETWORK_ENV) != "1":
        _block_network()
    _limit_file_size()
    # Tracebacks then show the snippet's lines under a short, stable file name.
    lines = source.splitlines(keepends=True)
    linecache.cache[SNIPPET_NAME] = (len(source), None, lines, SNIPPET_NAME)
    try:
        _execute(source)
    except SystemExit:
        raise
    except BaseException as exc:
        _print_snippet_traceback(exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
