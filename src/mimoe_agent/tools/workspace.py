"""Workspace jail and the read-only file tools: ``list_files``, ``read_file``, ``search_files``.

Every path a tool receives is resolved against the workspace root and refused when it escapes
(``..``, absolute paths, drive letters or UNC shares outside the root, symlinks or junctions that
point outside).
The tools return strings, never raise and never return ``""``; error strings start with
``error:`` so the model can react to them. Outputs stay within 8,000 characters so the
``[truncated ...]`` tails and ``offset=N`` hints survive the middleware result cap.
"""

from __future__ import annotations

import fnmatch
import ntpath
import os
import stat
import sys
import time
import unicodedata
from collections.abc import Callable, Iterator
from difflib import SequenceMatcher
from pathlib import Path, PurePosixPath, PureWindowsPath

from langchain_core.tools import BaseTool, StructuredTool

# Directories that are never listed or searched (caches, dependencies, build output).
SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".idea",
        "dist",
        "build",
    }
)
# Credential files that read_file refuses and search_files skips, matched after Unicode NFKC
# normalisation and case folding. Deliberately specific: broad patterns such as *token* or
# *secret* also hid ordinary source files (tokenizer.py, secrets_manager.py).
SECRET_GLOBS: tuple[str, ...] = (
    ".env",
    ".env.*",
    "*.env",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.keystore",
    "*.jks",
    "*.ppk",
    "*.kdbx",
    "id_rsa*",
    "id_dsa*",
    "id_ecdsa*",
    "id_ed25519*",
    "credentials",
    "credentials.*",
    ".netrc",
    "_netrc",
    ".npmrc",
    ".pypirc",
    ".pgpass",
    ".htpasswd",
    ".git-credentials",
    "secrets.json",
    "secrets.yaml",
    "secrets.yml",
    "secrets.toml",
    "*.secret",
    "*.secrets",
    "token.json",
    "*.token",
    "client_secret*.json",
)

OUTPUT_CAP = 8_000  # characters; equal to the GuardrailMiddleware result cap
MAX_LIST_ENTRIES = 500
MAX_LIST_DEPTH = 10
MAX_READ_BYTES = 2_000_000  # "2 MB" file cap for read_file
MAX_READ_LINES = 200  # lines per read_file call; page with offset
MAX_LINE_CHARS = 2_000  # longer lines are cut in read_file output
MAX_SEARCH_RESULTS = 200
MAX_SEARCH_FILE_BYTES = 1_000_000  # larger files are skipped by search_files
MAX_SEARCH_DEPTH = 12
MAX_HIT_CHARS = 200  # characters of a matching line shown by search_files
BINARY_SNIFF_BYTES = 8_192
SEARCH_DEADLINE_S: float = 10.0  # cooperative search time budget; tests monkeypatch this

_WINDOWS = sys.platform == "win32"
_FOOTER_RESERVE = 200  # characters kept free for the truncation tail / paging hint
_TEXT_BYTES = bytes({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)))
# O_NONBLOCK keeps open() from blocking on a FIFO (POSIX only); O_BINARY stops the Windows
# CRT from translating line endings and treating ^Z as EOF on the raw descriptor.
_OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)


class WorkspaceError(ValueError):
    """A path is outside the workspace or otherwise not allowed."""


class Workspace:
    """A directory the tools may read; every path is resolved and checked against ``root``.

    The root itself is resolved (``/tmp`` is ``/private/tmp`` on macOS), otherwise every
    resolved candidate would fail the containment check.
    """

    def __init__(self, root: Path | str) -> None:
        """Bind to an existing directory.

        Raises:
            WorkspaceError: ``root`` does not exist or is not a directory.
        """
        try:
            self.root: Path = Path(root).resolve(strict=True)
        except OSError as exc:
            raise WorkspaceError(f"workspace root {str(root)!r} is not accessible: {exc}") from exc
        if not self.root.is_dir():
            raise WorkspaceError(f"workspace root {str(root)!r} is not a directory")

    def resolve(self, rel: str) -> Path:
        """Return the absolute path for a workspace-relative string.

        The path does not have to exist. Symlinks and junctions are followed before the
        containment check, so a link that points outside the workspace is refused.

        An absolute path under the root (the model copies the root from the system prompt) counts
        as the relative path it names and goes through the same checks.

        Raises:
            WorkspaceError: the path contains a NUL, is absolute or uses a drive letter or a UNC
                share outside the root, resolves outside the root, or (on Windows) names a
                reserved device.
        """
        text = str(rel).strip() or "."
        if "\x00" in text:
            raise WorkspaceError("path contains a NUL character")
        relative = text
        # Path.__truediv__ silently replaces the root with an absolute, rooted, drive-relative
        # or UNC operand, so those never reach a join (or the filesystem: resolving a UNC path
        # connects to its server); only the lexical part below the root is kept.
        if PurePosixPath(text).anchor or PureWindowsPath(text).anchor:
            inside = self._below_root(text)
            if inside is None:
                raise WorkspaceError(
                    "absolute paths outside the workspace, drive letters and UNC shares are not "
                    f"allowed: {text!r}; use a path relative to the workspace, such as '.' for "
                    "the workspace itself"
                )
            relative = inside
        try:
            candidate = (self.root / relative).resolve(strict=False)
        except (OSError, RuntimeError, ValueError) as exc:  # symlink loop, too long, odd bytes
            raise WorkspaceError(f"invalid path {text!r}: {exc}") from exc
        if not candidate.is_relative_to(self.root):
            raise WorkspaceError(f"path escapes the workspace: {text!r}")
        if _WINDOWS and ntpath.isreserved(str(candidate)):  # CON, NUL, COM1, "name.", "name "
            raise WorkspaceError(f"reserved Windows device name: {text!r}")
        return candidate

    def _below_root(self, text: str) -> str | None:
        """The part of an absolute path below the root, or ``None`` when it is not under it.

        Lexical only: ``..`` stays in the result for :meth:`resolve` to judge. Windows paths
        compare case-insensitively with either separator; a rooted path without a drive, a
        drive-relative path and a UNC share outside the root are never under it.
        """
        flavour = PureWindowsPath if _WINDOWS else PurePosixPath
        path, root = flavour(text), flavour(self.root)
        if not path.is_absolute() or not path.is_relative_to(root):
            return None
        return path.relative_to(root).as_posix()

    def rel(self, path: Path) -> str:
        """Workspace-relative POSIX form of an absolute path inside the root, for display."""
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return path.as_posix()


# --- helpers ---------------------------------------------------------------------------------


def _is_secret(name: str) -> bool:
    folded = unicodedata.normalize("NFKC", name).casefold()
    return any(fnmatch.fnmatchcase(folded, glob) for glob in SECRET_GLOBS)


def _utf16(data: bytes) -> bool:
    """A UTF-16 byte-order mark: text that Windows tools (PowerShell 5.1 ``>``, Notepad's
    "Unicode") write; its NUL bytes would otherwise make it look binary."""
    return data.startswith((b"\xff\xfe", b"\xfe\xff"))


def _looks_binary(sample: bytes) -> bool:
    """The git/`file` heuristic: a NUL byte, or more than 30% non-text bytes."""
    if not sample:
        return False
    if b"\x00" in sample:
        return True
    return len(sample.translate(None, _TEXT_BYTES)) / len(sample) > 0.30


def _decode(data: bytes) -> tuple[str, str]:
    """Decode UTF-16 (by its BOM) or UTF-8 (BOM stripped), else Windows-1252 with replacement."""
    if _utf16(data):
        return data.decode("utf-16", errors="replace"), "utf-16"
    try:
        return data.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace"), "cp1252"


def _read_capped(path: Path, cap: int) -> bytes | None:
    """Read at most ``cap + 1`` bytes of a regular file; ``None`` if it is not a regular file.

    The type check runs on the open descriptor (``fstat``), so a FIFO, socket or device that
    replaces the file between a ``stat()`` and the ``open()`` can neither block nor be read,
    and a file that grows while it is read never exceeds the cap in memory. Callers treat a
    result longer than ``cap`` as "over the limit".

    Raises:
        OSError: the file cannot be opened or read.
    """
    fd = os.open(path, _OPEN_FLAGS)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None
        with open(fd, "rb", closefd=False) as fh:
            return fh.read(cap + 1)
    finally:
        os.close(fd)


def _split_lines(text: str) -> list[str]:
    """Split on LF, CRLF and CR only (``str.splitlines`` also splits on form feeds etc.)."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _to_int(value: object, default: int) -> int:
    try:
        return int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return default


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def _walk(start: Path, max_depth: int) -> Iterator[tuple[Path, list[tuple[str, str]], list[str]]]:
    """Walk top-down without following symlinks or junctions.

    Yields ``(directory, subdirectories, file names)`` with names sorted; each subdirectory is
    ``(name, kind)`` where kind is ``"dir"`` (descended), ``"symlink"``, ``"junction"`` or
    ``"deep"`` (not descended). Directories in ``SKIP_DIRS`` are omitted entirely.
    """
    base = len(start.parts)
    for dirpath, dirnames, filenames in os.walk(start, topdown=True, followlinks=False):
        here = Path(dirpath)
        depth = len(here.parts) - base  # 0 for ``start``
        subdirs: list[tuple[str, str]] = []
        descend: list[str] = []
        for name in sorted(dirnames):
            if name in SKIP_DIRS:
                continue
            sub = here / name
            if sub.is_symlink():
                subdirs.append((name, "symlink"))
            elif sub.is_junction():  # os.walk(followlinks=False) would still descend these
                subdirs.append((name, "junction"))
            elif depth + 1 >= max_depth:
                subdirs.append((name, "deep"))
            else:
                subdirs.append((name, "dir"))
                descend.append(name)
        dirnames[:] = descend
        yield here, subdirs, sorted(filenames)


MAX_SUGGEST_NAMES = 5_000  # workspace names compared for a "did you mean" hint
MAX_SUGGEST_DEPTH = 6
SUGGEST_CUTOFF = 0.7  # how alike two names without extensions must be to count as "close"


def _closest(wanted: str, candidates: list[str], limit: int = 3) -> list[str]:
    """The workspace paths most like ``wanted``, best first, ignoring case: the same path in
    another case, the same name in another folder, the same name with another extension, then
    the closest spelling of the name without its extension (a shared ``.py`` or folder is not
    a likeness)."""
    want = PurePosixPath(wanted.lower())
    scored: list[tuple[float, str]] = []
    for candidate in candidates:
        path = PurePosixPath(candidate.lower())
        if path == want:
            score = 1.0
        elif path.name == want.name:
            score = 0.95
        elif path.stem == want.stem:
            score = 0.9
        else:
            score = SequenceMatcher(None, path.stem, want.stem).ratio()
        if score >= SUGGEST_CUTOFF:
            scored.append((score, candidate))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [candidate for _, candidate in scored[:limit]]


_HINT_ACTIONS = {"file": ("file", "read"), "dir": ("folder", "list"), "any": ("path", "search")}


def _did_you_mean(ws: Workspace, missing: Path, want: str) -> str:
    """``"; did you mean 'notes.md'? If that is the file the user means, read it."`` for a path
    that does not exist, or ``""`` when nothing is close.

    Told only that ``notes.txt`` does not exist, qwen3-4b-instruct-2507 gave up on "read the
    notes file"; with "did you mean 'notes.md'?" it asked the user whether to read that, and only
    the added "If that is the file the user means, read it." made it read the file (one replayed
    step each, temperature 0). Candidates come from the walk ``list_files`` uses (no symlinks,
    no ``.git`` or ``node_modules``): files for ``want="file"`` (never credential files), folders
    for ``"dir"``, both for ``"any"``.
    """
    names: list[str] = []
    for here, subdirs, files in _walk(ws.root, MAX_SUGGEST_DEPTH):
        found = [] if want == "dir" else [n for n in files if not _is_secret(n)]
        if want != "file":
            found += [n for n, kind in subdirs if kind in ("dir", "deep")]
        names += [ws.rel(here / name) for name in found]
        if len(names) >= MAX_SUGGEST_NAMES:
            break
    picks = [repr(p) for p in _closest(ws.rel(missing), names[:MAX_SUGGEST_NAMES])]
    if not picks:
        return ""
    noun, verb = _HINT_ACTIONS[want]
    if len(picks) == 1:
        return f"; did you mean {picks[0]}? If that is the {noun} the user means, {verb} it."
    listed = f"{', '.join(picks[:-1])} or {picks[-1]}"
    return f"; did you mean {listed}? If one of them is the {noun} the user means, {verb} it."


class _Output:
    """Collects result lines under a character budget so a short footer always fits."""

    def __init__(self, header_chars: int = 0) -> None:
        self.lines: list[str] = []
        self.chars = 0
        self.budget = OUTPUT_CAP - header_chars - _FOOTER_RESERVE
        self.full = False

    def add(self, line: str) -> bool:
        """Append ``line`` unless it would exceed the budget; returns False when full."""
        cost = len(line) + 1
        if self.chars + cost > self.budget:
            self.full = True
            return False
        self.lines.append(line)
        self.chars += cost
        return True


# --- tool bodies (pure functions of a Workspace; they may raise, the wrappers never do) -------


def _list_files(ws: Workspace, path: str, max_depth: int) -> str:
    try:
        start = ws.resolve(path)
    except WorkspaceError as exc:
        return f"error: {exc}"
    if not start.exists():
        return f"error: {path!r} does not exist in the workspace{_did_you_mean(ws, start, 'dir')}"
    if not start.is_dir():
        return f"error: {path!r} is a file, not a directory; use read_file"
    depth = _clamp(_to_int(max_depth, 4), 1, MAX_LIST_DEPTH)

    out = _Output()
    entries = 0
    unexpanded = 0
    truncated = False

    def add(line: str) -> bool:
        nonlocal entries, truncated
        if entries >= MAX_LIST_ENTRIES or not out.add(line):
            truncated = True
            return False
        entries += 1
        return True

    for here, subdirs, files in _walk(start, depth):
        for name in files:
            file = here / name
            try:
                size = f"{file.stat().st_size:,} B"
            except OSError:
                size = "size unknown"
            tag = "  -> symlink" if file.is_symlink() else ""
            if not add(f"{ws.rel(file)}  ({size}){tag}"):
                break
        else:
            for name, kind in subdirs:
                label = ws.rel(here / name) + "/"
                if kind == "symlink":
                    label += "  -> symlink (not expanded)"
                elif kind == "junction":
                    label += "  (junction, not expanded)"
                elif kind == "deep":
                    label += "  (not expanded: max_depth reached)"
                    unexpanded += 1
                if not add(label):
                    break
            else:
                continue
        break

    if not out.lines:
        return f"{ws.rel(start)!r} is an empty directory"
    footer: list[str] = []
    if truncated:
        footer.append(
            f"[truncated at {entries} entries; narrow the path or lower max_depth]",
        )
    elif unexpanded:
        footer.append(
            f"[{unexpanded} director{'y' if unexpanded == 1 else 'ies'} not expanded; "
            "call list_files with path=<directory> to see inside]"
        )
    return "\n".join([*out.lines, *footer])


def _read_file(ws: Workspace, path: str, offset: int, limit: int) -> str:
    try:
        target = ws.resolve(path)
    except WorkspaceError as exc:
        return f"error: {exc}"
    if _is_secret(target.name):
        return f"error: {path!r} looks like a secrets file (its name matches a secret pattern)"
    if not target.exists():
        return f"error: {path!r} does not exist in the workspace{_did_you_mean(ws, target, 'file')}"
    if target.is_dir():
        return f"error: {path!r} is a directory; use list_files"
    if not target.is_file():
        return f"error: {path!r} is not a regular file (device, socket or pipe); not readable"
    too_big = f"error: {path!r} is over {MAX_READ_BYTES:,} bytes; use run_python to process it"
    try:
        size = target.stat().st_size
        if size > MAX_READ_BYTES:
            return too_big
        data = _read_capped(target, MAX_READ_BYTES)
    except OSError as exc:
        return f"error: cannot read {path!r}: {exc}"
    if data is None:  # replaced by a FIFO/device between the checks and the open
        return f"error: {path!r} is not a regular file (device, socket or pipe); not readable"
    if len(data) > MAX_READ_BYTES:  # grew past the cap while being read
        return too_big
    if not _utf16(data) and _looks_binary(data[:BINARY_SNIFF_BYTES]):
        return f"error: {path!r} looks like a binary file ({size:,} bytes); use run_python instead"

    text, encoding = _decode(data)
    lines = _split_lines(text)
    total = len(lines)
    rel = ws.rel(target)
    if total == 0:
        return f"{rel} is empty (0 lines)"
    start = max(1, _to_int(offset, 1))
    count = _clamp(_to_int(limit, MAX_READ_LINES), 1, MAX_READ_LINES)
    if start > total:
        return f"{rel} has {total} lines; offset {start} is past the end"
    last_wanted = min(total, start + count - 1)
    width = len(str(last_wanted))

    out = _Output(header_chars=len(rel) + 60)
    for number in range(start, last_wanted + 1):
        line = lines[number - 1]
        if len(line) > MAX_LINE_CHARS:
            line = f"{line[:MAX_LINE_CHARS]} [line truncated: {len(line):,} chars]"
        if not out.add(f"{number:>{width}}| {line}"):
            break
    end = start + len(out.lines) - 1
    note = "" if encoding == "utf-8" else f" [decoded as {encoding}]"
    header = f"{rel} lines {start}-{end} of {total}{note}"
    footer: list[str] = []
    if end < total:
        footer.append(f"... {total - end} more lines (call again with offset={end + 1})")
    return "\n".join([header, *out.lines, *footer])


def _search_candidates(start: Path) -> Iterator[Path]:
    if start.is_file():
        yield start
        return
    for here, _subdirs, files in _walk(start, MAX_SEARCH_DEPTH):
        for name in files:
            yield here / name


def _search_files(ws: Workspace, pattern: str, path: str, glob: str, max_results: int) -> str:
    if not isinstance(pattern, str) or pattern == "":
        return "error: pattern must be a non-empty string"
    try:
        start = ws.resolve(path)
    except WorkspaceError as exc:
        return f"error: {exc}"
    if not start.exists():
        return f"error: {path!r} does not exist in the workspace{_did_you_mean(ws, start, 'any')}"
    needle = pattern.lower()
    name_glob = (str(glob).strip() or "*").lower()
    name_glob = name_glob.removeprefix("**/") or "*"  # "**/*.py" must also match top-level files
    limit = _clamp(_to_int(max_results, 100), 1, MAX_SEARCH_RESULTS)
    deadline = time.monotonic() + SEARCH_DEADLINE_S  # read at call time: tests patch the constant

    out = _Output()
    hits = 0
    scanned = 0
    too_big = 0
    credential_files = 0
    stopped = ""
    for file in _search_candidates(start):
        if time.monotonic() >= deadline:
            stopped = (
                f"stopped at the {SEARCH_DEADLINE_S:g} s time limit after {scanned} files; "
                "narrow the path or glob"
            )
            break
        name = file.name
        rel = ws.rel(file)
        if not (
            fnmatch.fnmatchcase(name.lower(), name_glob)
            or fnmatch.fnmatchcase(rel.lower(), name_glob)
        ):
            continue
        if _is_secret(name):
            credential_files += 1
            continue
        try:
            if file.is_symlink():
                continue
            if file.stat().st_size > MAX_SEARCH_FILE_BYTES:
                too_big += 1
                continue
            data = _read_capped(file, MAX_SEARCH_FILE_BYTES)  # never blocks on a FIFO or device
        except OSError:
            continue
        if data is None or (not _utf16(data) and _looks_binary(data[:BINARY_SNIFF_BYTES])):
            continue
        if len(data) > MAX_SEARCH_FILE_BYTES:  # grew past the cap while being read
            too_big += 1
            continue
        scanned += 1
        text, _encoding = _decode(data)
        if needle not in text.lower():
            continue
        for number, line in enumerate(_split_lines(text), 1):
            if needle not in line.lower():
                continue
            snippet = line.strip()
            if len(snippet) > MAX_HIT_CHARS:
                snippet = snippet[:MAX_HIT_CHARS] + "..."
            if not out.add(f"{rel}:{number}: {snippet}"):
                stopped = f"output truncated at {OUTPUT_CAP:,} characters; refine the pattern"
                break
            hits += 1
            if hits >= limit:
                stopped = f"stopped at {limit} matches; refine the pattern or narrow the path"
                break
        if stopped:
            break

    notes = [stopped] if stopped else []
    if credential_files:
        notes.append(
            f"{credential_files} credential file{'s' if credential_files > 1 else ''} "
            "(.env, keys and similar) not searched"
        )
    if too_big:
        notes.append(
            f"{too_big} file{'s' if too_big > 1 else ''} over {MAX_SEARCH_FILE_BYTES:,} bytes "
            "skipped; use run_python for those"
        )
    if hits == 0:
        where = ws.rel(start)
        tail = f" ({'; '.join(notes)})" if notes else ""
        return f"no matches for {pattern!r} in {scanned} text files under {where!r}{tail}"
    footer = [f"[{'; '.join(notes)}]"] if notes else []
    return "\n".join([*out.lines, *footer])


def _never_raise(body: Callable[[], str]) -> str:
    """Run a tool body; any exception becomes an error string and "" is never returned.

    Lone surrogates (undecodable bytes in POSIX file names) are replaced so the result can
    always be JSON-encoded for the model and printed by the CLI.
    """
    try:
        result = body()
    except Exception as exc:  # a tool must never raise into the agent loop
        result = f"error: {type(exc).__name__}: {exc}"
    result = result or "error: the tool produced no output"
    return result.encode("utf-8", errors="replace").decode("utf-8")


def _invalid_arguments(exc: Exception) -> str:
    """Turn a pydantic validation error into the same ``error:`` string shape as the bodies."""
    problems: list[str] = []
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            for item in errors():
                where = ".".join(str(part) for part in item.get("loc", ())) or "input"
                problems.append(f"{where}: {item.get('msg', 'invalid value')}")
        except Exception:  # defensive: a foreign exception type with an odd errors()
            problems = []
    detail = "; ".join(problems) or str(exc).splitlines()[0]
    return f"error: invalid arguments ({detail}); check the types and call again"


# --- public factory --------------------------------------------------------------------------


def make_workspace_tools(ws: Workspace) -> list[BaseTool]:
    """Build the ``list_files``, ``read_file`` and ``search_files`` tools bound to ``ws``.

    The docstrings below are the descriptions the model reads, so they stay short.
    """

    def list_files(path: str = ".", max_depth: int = 4) -> str:
        """List files and folders under a workspace-relative path, with sizes in bytes.
        Skips .git, node_modules, .venv and similar folders; at most 500 entries.
        """
        return _never_raise(lambda: _list_files(ws, path, max_depth))

    def read_file(path: str, offset: int = 1, limit: int = 200) -> str:
        """Read a text file from the workspace with line numbers, starting at line offset.
        Refuses binary files, files over 2 MB and credential files (.env, private keys, .netrc
        and similar); at most 200 lines per call, so page through long files with offset.
        """
        return _never_raise(lambda: _read_file(ws, path, offset, limit))

    def search_files(pattern: str, path: str = ".", glob: str = "*", max_results: int = 100) -> str:
        """Find lines containing a case-insensitive substring in workspace text files, returned
        as path:line: text. Plain text only (no regex); glob such as *.py filters file names;
        skips binary and credential files; at most 200 results.
        """
        return _never_raise(lambda: _search_files(ws, pattern, path, glob, max_results))

    return [
        StructuredTool.from_function(
            func=func, name=func.__name__, handle_validation_error=_invalid_arguments
        )
        for func in (list_files, read_file, search_files)
    ]
