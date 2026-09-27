"""Offline tests for the workspace jail and the list_files / read_file / search_files tools."""

from __future__ import annotations

import contextlib
import os
import re
import sys
import threading
from collections.abc import Callable
from pathlib import Path

import pytest
from langchain_core.tools import BaseTool

from mimoe_agent.tools import workspace as wsmod
from mimoe_agent.tools.workspace import (
    MAX_LIST_ENTRIES,
    MAX_READ_BYTES,
    MAX_SEARCH_FILE_BYTES,
    MAX_SEARCH_RESULTS,
    OUTPUT_CAP,
    Workspace,
    WorkspaceError,
    make_workspace_tools,
)

Tools = dict[str, BaseTool]
WINDOWS = sys.platform == "win32"


def call_with_timeout(fn: Callable[[], str], seconds: float = 5.0) -> str:
    """Run ``fn`` in a daemon thread; fail (instead of hanging the suite) if it does not return."""
    box: dict[str, object] = {}

    def target() -> None:
        try:
            box["result"] = fn()
        except BaseException as exc:  # surfaced below
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        pytest.fail(f"tool call did not return within {seconds} s (blocked on a special file?)")
    if "error" in box:
        raise box["error"]  # type: ignore[misc]
    return box["result"]  # type: ignore[return-value]


def write(root: Path, rel: str, content: str | bytes) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8", newline="")
    return path


def symlink_or_skip(target: Path, link: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not permitted on this system")


def list_(tools: Tools, **kwargs: object) -> str:
    return tools["list_files"].invoke(kwargs)


def read(tools: Tools, path: str, **kwargs: object) -> str:
    return tools["read_file"].invoke({"path": path, **kwargs})


def search(tools: Tools, pattern: str, **kwargs: object) -> str:
    return tools["search_files"].invoke({"pattern": pattern, **kwargs})


@pytest.fixture
def root(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    write(ws, "notes.md", "# Notes\n- TODO: one\n- done\n- todo: two\n")
    write(ws, "src/app.py", "import csv\nprint('hi')\n")
    write(ws, "src/utils.py", "def f():\n    return 1\n")
    write(ws, "data/sales.csv", "a,b\n1,2\n")
    return ws


@pytest.fixture
def ws(root: Path) -> Workspace:
    return Workspace(root)


@pytest.fixture
def tools(ws: Workspace) -> Tools:
    return {tool.name: tool for tool in make_workspace_tools(ws)}


# --- jail ------------------------------------------------------------------------------------


class TestWorkspaceJail:
    def test_root_is_resolved_through_symlink(self, root: Path, tmp_path: Path) -> None:
        link = tmp_path / "link"
        symlink_or_skip(root, link)
        ws = Workspace(link)
        assert ws.root == root.resolve()
        assert ws.resolve("notes.md") == root.resolve() / "notes.md"

    def test_missing_or_file_root_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(WorkspaceError):
            Workspace(tmp_path / "nope")
        plain = tmp_path / "plain.txt"
        plain.write_text("x")
        with pytest.raises(WorkspaceError):
            Workspace(plain)

    @pytest.mark.parametrize(
        "rel",
        [".", "", "  ", "./notes.md", "src/../notes.md", "src/app.py", "does/not/exist.txt"],
    )
    def test_inside_paths_accepted(self, ws: Workspace, rel: str) -> None:
        assert ws.resolve(rel).is_relative_to(ws.root)

    def test_dot_and_empty_are_the_root(self, ws: Workspace) -> None:
        assert ws.resolve(".") == ws.root
        assert ws.resolve("") == ws.root
        assert ws.resolve("src/..") == ws.root

    @pytest.mark.parametrize(
        "rel",
        [
            "..",
            "../x",
            "src/../../x",
            "a/../../..",
            "/etc/passwd",
            "/",
            "\\etc\\passwd",
            "C:/Windows/System32",
            "C:\\Windows",
            "c:notes.md",
            "\\\\server\\share\\x",
            "//server/share/x",
            "a\x00b",
            "\x00",
        ],
    )
    def test_escapes_rejected(self, ws: Workspace, rel: str) -> None:
        with pytest.raises(WorkspaceError):
            ws.resolve(rel)

    def test_absolute_paths_under_the_root_are_accepted(self, ws: Workspace) -> None:
        """The model copies the root from the system prompt: qwen3-4b-instruct-2507 answered
        "Show me the list of files in my workspace" with list_files(path=<the root>)."""
        root = ws.root
        assert ws.resolve(str(root)) == root
        assert ws.resolve(root.as_posix()) == root  # the system prompt's spelling, also on Windows
        assert ws.resolve(root.as_posix() + "/") == root
        assert ws.resolve(str(root / "notes.md")) == root / "notes.md"
        assert ws.resolve(root.as_posix() + "/src/app.py") == root / "src" / "app.py"
        assert ws.resolve(str(root / "src" / ".." / "notes.md")) == root / "notes.md"

    def test_absolute_paths_are_still_jailed(
        self, ws: Workspace, root: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(WorkspaceError, match="escapes"):
            ws.resolve(str(ws.root / ".." / "x"))  # under the root lexically, outside once resolved
        sibling = Path(str(ws.root) + "2")  # shares the root's spelling as a prefix
        for outside in (ws.root.parent, sibling, sibling / "notes.md"):
            with pytest.raises(WorkspaceError, match="outside the workspace.*'\\.' for"):
                ws.resolve(str(outside))
        (tmp_path / "outside").mkdir()
        symlink_or_skip(tmp_path / "outside", root / "link_dir")
        with pytest.raises(WorkspaceError, match="escapes"):
            ws.resolve(str(ws.root / "link_dir"))

    @pytest.mark.skipif(not WINDOWS, reason="Windows path rules")
    def test_windows_absolute_paths_under_the_root(self, ws: Workspace) -> None:
        root, drive = str(ws.root), ws.root.drive
        assert ws.resolve(root.upper() + "\\notes.md") == ws.root / "notes.md"  # any case
        assert ws.resolve(root.replace("\\", "/") + "/src/app.py") == ws.root / "src" / "app.py"
        for bad in (
            drive + "notes.md",
            root[len(drive) :] + "\\notes.md",
        ):  # drive-relative, rooted
            with pytest.raises(WorkspaceError):
                ws.resolve(bad)

    def test_symlink_escapes_rejected(self, ws: Workspace, root: Path, tmp_path: Path) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("s")
        symlink_or_skip(outside, root / "link_dir")
        symlink_or_skip(outside / "secret.txt", root / "link_file.txt")
        for rel in ["link_dir", "link_dir/secret.txt", "link_file.txt"]:
            with pytest.raises(WorkspaceError, match="escapes"):
                ws.resolve(rel)
        # ".." after a link: POSIX realpath follows the link first (-> outside, refused);
        # ntpath.realpath collapses ".." lexically first (-> notes.md, inside). Either way
        # the result must never be outside the root.
        with contextlib.suppress(WorkspaceError):
            assert ws.resolve("link_dir/../notes.md").is_relative_to(ws.root)
        if not WINDOWS:
            with pytest.raises(WorkspaceError, match="escapes"):
                ws.resolve("link_dir/../notes.md")
        symlink_or_skip(root / "notes.md", root / "alias.md")  # a link that stays inside is fine
        assert ws.resolve("alias.md") == root.resolve() / "notes.md"

    def test_symlink_chain_and_loop(self, ws: Workspace, root: Path, tmp_path: Path) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "leak.txt").write_text("leak")
        symlink_or_skip(outside, root / "hop3")
        symlink_or_skip(root / "hop3", root / "hop2")
        symlink_or_skip(root / "hop2", root / "hop1")  # hop1 -> hop2 -> hop3 -> outside
        with pytest.raises(WorkspaceError, match="escapes"):
            ws.resolve("hop1/leak.txt")
        symlink_or_skip(root / "notes.md", root / "in3")
        symlink_or_skip(root / "in3", root / "in2")
        symlink_or_skip(root / "in2", root / "in1")  # stays inside
        assert ws.resolve("in1") == root.resolve() / "notes.md"
        symlink_or_skip(root / "loop_b", root / "loop_a")
        symlink_or_skip(root / "loop_a", root / "loop_b")  # a -> b -> a
        with contextlib.suppress(WorkspaceError):  # refused, or resolved to an inside path
            assert ws.resolve("loop_a/x").is_relative_to(ws.root)
        tools = {tool.name: tool for tool in make_workspace_tools(ws)}
        assert read(tools, "loop_a").startswith("error:")
        assert read(tools, "hop1/leak.txt").startswith("error:")
        assert "leak.txt" not in call_with_timeout(lambda: search(tools, "leak"))

    def test_windows_reserved_names(self, ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(wsmod, "_WINDOWS", True)
        for rel in ["CON", "nul.txt", "src/COM1", "foo.", "notes.md:stream", "a<b"]:
            # On real Windows resolve() itself may already refuse some of these; the property
            # under test is that the path is refused, whichever check fires first.
            with pytest.raises(WorkspaceError, match="reserved|escapes|invalid path"):
                ws.resolve(rel)
        assert ws.resolve("notes.md").name == "notes.md"
        if not WINDOWS:  # the gate is off on POSIX, where such names are legal
            monkeypatch.setattr(wsmod, "_WINDOWS", False)
            assert ws.resolve("CON").name == "CON"

    def test_rel_is_posix_and_relative(self, ws: Workspace) -> None:
        assert ws.rel(ws.root / "src" / "app.py") == "src/app.py"
        assert ws.rel(ws.root) == "."


# --- did you mean ------------------------------------------------------------------------------


class TestDidYouMean:
    """A missing path gets the closest workspace names: the model guesses names ("the notes file"
    -> notes.txt) and, told only that the file does not exist, gave up instead of looking."""

    def test_read_file_names_the_closest_file_and_says_to_read_it(self, tools: Tools) -> None:
        assert read(tools, "notes.txt") == (
            "error: 'notes.txt' does not exist in the workspace; did you mean 'notes.md'? "
            "If that is the file the user means, read it."
        )
        assert "did you mean 'src/app.py'?" in read(tools, "app.py")  # same name, other folder
        assert "did you mean 'src/utils.py'?" in read(tools, "src/util.py")  # a typo
        assert "did you mean 'data/sales.csv'?" in read(tools, "data/sale.csv")

    def test_list_and_search_suggest_folders_and_paths(self, tools: Tools) -> None:
        assert list_(tools, path="dat").endswith(
            "did you mean 'data'? If that is the folder the user means, list it."
        )
        assert search(tools, "x", path="src/util.py").endswith(
            "did you mean 'src/utils.py'? If that is the path the user means, search it."
        )

    def test_nothing_close_means_no_hint(self, tools: Tools) -> None:
        assert read(tools, "todo.txt") == "error: 'todo.txt' does not exist in the workspace"
        assert "did you mean" not in read(tools, "src/main.py")  # a shared folder is no likeness

    def test_several_close_names(self, root: Path, tools: Tools) -> None:
        for name in ("report.csv", "data/reports.csv", "data/report.xlsx"):
            write(root, name, "x")
        assert read(tools, "data/report.csv").endswith(
            "did you mean 'report.csv', 'data/reports.csv' or 'data/report.xlsx'? "
            "If one of them is the file the user means, read it."
        )

    def test_never_suggests_credentials_skipped_or_linked_folders(
        self, root: Path, tmp_path: Path, tools: Tools
    ) -> None:
        write(root, ".env", "TOKEN=1")
        write(root, "node_modules/notes.txt", "x")
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "plan.md").write_text("private")
        symlink_or_skip(outside, root / "linked")
        assert "did you mean" not in read(tools, "env")  # .env is close, but a credential file
        assert "node_modules" not in read(tools, "notes.txt")
        assert "did you mean" not in read(tools, "plan.md")  # linked/plan.md is not walked

    def test_closest_ranks_case_folder_extension_then_spelling(self) -> None:
        names = ["readme.md", "notez.md", "notes.txt", "docs/notes.md", "Notes.MD"]
        assert wsmod._closest("notes.md", names) == ["Notes.MD", "docs/notes.md", "notes.txt"]
        assert wsmod._closest("notes.md", names, limit=5) == [
            "Notes.MD",
            "docs/notes.md",
            "notes.txt",
            "notez.md",
        ]


# --- tool wiring -----------------------------------------------------------------------------


class TestToolWiring:
    def test_names_schemas_and_descriptions(self, tools: Tools) -> None:
        assert list(tools) == ["list_files", "read_file", "search_files"]
        assert list(tools["list_files"].args) == ["path", "max_depth"]
        assert list(tools["read_file"].args) == ["path", "offset", "limit"]
        assert list(tools["search_files"].args) == ["pattern", "path", "glob", "max_results"]
        assert tools["list_files"].args["path"]["default"] == "."
        assert tools["read_file"].args["offset"]["default"] == 1
        assert tools["read_file"].args["limit"]["default"] == 200
        assert tools["search_files"].args["max_results"]["default"] == 100
        for tool in tools.values():
            assert tool.description.strip()
            assert not tool.description.startswith(" ")

    @pytest.mark.parametrize(
        "bad", ["..", "/etc/passwd", "C:\\Windows", "\\\\srv\\share", "a\x00b"]
    )
    def test_tools_return_error_strings_and_never_raise(self, tools: Tools, bad: str) -> None:
        calls = [
            ("list_files", {"path": bad}),
            ("read_file", {"path": bad}),
            ("search_files", {"pattern": "x", "path": bad}),
        ]
        for name, args in calls:
            out = tools[name].invoke(args)
            assert isinstance(out, str)
            assert out.startswith("error:")

    def test_tools_take_the_absolute_workspace_path(self, ws: Workspace, tools: Tools) -> None:
        """What the model sends when a question says "my workspace" (see the jail tests)."""
        listing = list_(tools, path=ws.root.as_posix())
        assert not listing.startswith("error:") and "src/app.py" in listing
        assert "TODO: one" in read(tools, str(ws.root / "notes.md"))
        assert "notes.md" in search(tools, "todo", path=str(ws.root))
        refused = list_(tools, path="/")  # the filesystem root is not the workspace
        assert refused.startswith("error:") and "'.' for the workspace itself" in refused

    def test_numeric_strings_are_coerced(self, root: Path, tools: Tools) -> None:
        write(root, "long.txt", "".join(f"line {i}\n" for i in range(1, 21)))
        out = tools["read_file"].invoke({"path": "long.txt", "offset": "5", "limit": "2"})
        assert "lines 5-6 of 20" in out

    def test_body_exception_becomes_error_string(
        self, tools: Tools, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*_args: object, **_kwargs: object) -> str:
            raise RuntimeError("kaboom")

        monkeypatch.setattr(wsmod, "_list_files", boom)
        out = tools["list_files"].invoke({"path": "."})
        assert out == "error: RuntimeError: kaboom"

    def test_empty_body_result_is_replaced(
        self, tools: Tools, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(wsmod, "_search_files", lambda *_a, **_k: "")
        assert tools["search_files"].invoke({"pattern": "x"}) != ""

    @pytest.mark.parametrize(
        ("name", "args"),
        [
            ("read_file", {"path": "notes.md", "offset": "abc"}),
            ("read_file", {"path": "notes.md", "offset": 2.5}),
            ("read_file", {"path": 123}),
            ("list_files", {"path": ".", "max_depth": None}),
            ("search_files", {"pattern": 5}),
            ("search_files", {}),
        ],
    )
    def test_invalid_arguments_become_error_strings(
        self, tools: Tools, name: str, args: dict[str, object]
    ) -> None:
        out = tools[name].invoke(args)  # pydantic rejects these; the tool must not raise
        assert isinstance(out, str)
        assert out.startswith("error: invalid arguments")

    def test_lone_surrogates_are_sanitized(
        self, tools: Tools, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A POSIX file name with undecodable bytes reaches Python as a surrogate-escaped str;
        # the tool result must still be UTF-8 encodable for the model request and the CLI.
        monkeypatch.setattr(wsmod, "_list_files", lambda *_a, **_k: "bad\udcff.txt  (1 B)")
        out = tools["list_files"].invoke({})
        out.encode("utf-8")  # must not raise
        assert "bad" in out
        assert "\udcff" not in out


# --- read_file -------------------------------------------------------------------------------


class TestReadFile:
    def test_bom_and_crlf(self, root: Path, tools: Tools) -> None:
        write(root, "bom.txt", b"\xef\xbb\xbfhello\r\nworld\r\n")
        out = read(tools, "bom.txt")
        assert "\ufeff" not in out
        assert "\r" not in out
        assert "bom.txt lines 1-2 of 2" in out
        assert "1| hello" in out
        assert "2| world" in out
        assert "decoded as" not in out

    def test_cr_only_line_endings(self, root: Path, tools: Tools) -> None:
        write(root, "mac.txt", b"one\rtwo\rthree")
        out = read(tools, "mac.txt")
        assert "lines 1-3 of 3" in out
        assert "3| three" in out

    def test_cp1252_fallback(self, root: Path, tools: Tools) -> None:
        write(root, "legacy.txt", b"caf\xe9 \x93quoted\x94\n")
        out = read(tools, "legacy.txt")
        assert "café \u201cquoted\u201d" in out
        assert "[decoded as cp1252]" in out

    def test_utf8_needs_no_note(self, root: Path, tools: Tools) -> None:
        write(root, "utf8.txt", "héllo wörld\n")
        out = read(tools, "utf8.txt")
        assert "héllo wörld" in out
        assert "decoded as" not in out

    @pytest.mark.parametrize(
        ("name", "content"),
        [
            ("blob.bin", bytes(range(256)) * 4),
            ("img.png", b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + bytes(range(128, 256)) * 8),
            (
                "nulbyte.dat",
                b"text with a \x00 byte\n",
            ),  # not "nul.dat": a reserved name on Windows
        ],
        ids=["control-bytes", "png", "nul"],
    )
    def test_binary_refused(self, root: Path, tools: Tools, name: str, content: bytes) -> None:
        write(root, name, content)
        out = read(tools, name)
        assert out.startswith("error:")
        assert "binary" in out

    @pytest.mark.parametrize(
        "name",
        [
            ".env",
            ".ENV",
            ".env.local",
            "ID_RSA",
            "id_ed25519.pub",
            "server.PEM",
            "private.key",
            "credentials.json",
            "Credentials",
            ".netrc",
            ".npmrc",
            ".pypirc",
            ".git-credentials",
            "putty.ppk",
            "keystore.jks",
            "client.p12",
            "cert.PFX",
            "release.keystore",
            ".pgpass",
            ".htpasswd",
            "api.token",
            "Secrets.YAML",
            "prod.env",
            "token.json",
            "client_secret_123.apps.json",
            "id_ed25519.pub",
            "\uff0eenv",  # fullwidth full stop: NFKC turns it into ".env"
        ],
    )
    def test_secrets_refused(self, root: Path, tools: Tools, name: str) -> None:
        write(root, name, "hush")
        out = read(tools, name)
        assert out.startswith("error:")
        assert "secret" in out.lower()

    def test_ordinary_files_are_not_secrets(self, root: Path, tools: Tools) -> None:
        assert not read(tools, "notes.md").startswith("error:")
        assert not read(tools, "src/app.py").startswith("error:")
        for name in ("tokenizer.py", "secrets_manager.py", "id_utils.py", "csrf_token.py"):
            write(root, name, "x = 1\n")
            assert read(tools, name).startswith("1\t") or "x = 1" in read(tools, name), name

    def test_utf16_text_is_read_not_refused_as_binary(self, root: Path, tools: Tools) -> None:
        write(root, "windows.txt", "caf\u00e9 notes\r\n".encode("utf-16"))
        out = read(tools, "windows.txt")
        assert "café notes" in out and not out.startswith("error:")
        assert "windows.txt:1: café notes" in search(tools, "CAFÉ")

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs are POSIX-only")
    def test_fifo_refused(self, root: Path, tools: Tools) -> None:
        os.mkfifo(root / "pipe")  # a blocking open() would hang the agent
        out = call_with_timeout(lambda: read(tools, "pipe"))
        assert out.startswith("error:")
        assert "regular file" in out

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs are POSIX-only")
    def test_read_capped_never_opens_special_files(self, root: Path) -> None:
        os.mkfifo(root / "pipe")
        assert call_with_timeout(lambda: str(wsmod._read_capped(root / "pipe", 10))) == "None"
        write(root, "twelve.txt", b"0123456789ab")
        assert wsmod._read_capped(root / "twelve.txt", 5) == b"012345"  # cap + 1 bytes, no more
        assert wsmod._read_capped(root / "twelve.txt", 12) == b"0123456789ab"

    def test_file_that_grows_past_the_cap_is_refused(
        self, root: Path, tools: Tools, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # stat() saw a small file, but by the time it was read it had grown past the limit
        write(root, "growing.log", "small\n")
        monkeypatch.setattr(wsmod, "_read_capped", lambda *_a: b"x" * (MAX_READ_BYTES + 1))
        out = read(tools, "growing.log")
        assert out.startswith("error:")
        assert "run_python" in out

    def test_directory_and_missing(self, tools: Tools) -> None:
        assert read(tools, "src").startswith("error:")
        assert "list_files" in read(tools, "src")
        assert read(tools, "missing.txt").startswith("error:")

    def test_empty_file(self, root: Path, tools: Tools) -> None:
        write(root, "empty.txt", "")
        out = read(tools, "empty.txt")
        assert out
        assert "empty" in out

    def test_size_cap(self, root: Path, tools: Tools) -> None:
        write(root, "big.txt", b"x" * (MAX_READ_BYTES + 1))
        out = read(tools, "big.txt")
        assert out.startswith("error:")
        assert "run_python" in out

    def test_default_page_and_offset_hint(self, root: Path, tools: Tools) -> None:
        write(root, "long.txt", "".join(f"line {i}\n" for i in range(1, 1001)))
        out = read(tools, "long.txt")
        assert "long.txt lines 1-200 of 1000" in out
        assert "call again with offset=201" in out
        assert "200| line 200" in out
        assert "line 201" not in out

        out = read(tools, "long.txt", offset=201, limit=50)
        assert "lines 201-250 of 1000" in out
        assert "offset=251" in out

        out = read(tools, "long.txt", offset=999)
        assert "lines 999-1000 of 1000" in out
        assert "call again" not in out

        out = read(tools, "long.txt", offset=5000)
        assert out
        assert "past the end" in out

        assert "lines 1-200 of 1000" in read(tools, "long.txt", limit=5000)  # limit clamped
        assert "lines 1-200 of 1000" in read(tools, "long.txt", offset=0)  # offset clamped
        assert "lines 1-1 of 1000" in read(tools, "long.txt", limit=-3)

    def test_output_char_cap_keeps_hint(self, root: Path, tools: Tools) -> None:
        write(root, "wide.txt", "".join("x" * 100 + "\n" for _ in range(200)))
        out = read(tools, "wide.txt")
        assert len(out) <= OUTPUT_CAP
        match = re.search(r"lines 1-(\d+) of 200", out)
        assert match is not None
        end = int(match.group(1))
        assert end < 200
        assert f"call again with offset={end + 1}" in out

    def test_long_single_line_truncated(self, root: Path, tools: Tools) -> None:
        write(root, "min.js", "y" * 20_000 + "\n")
        out = read(tools, "min.js")
        assert len(out) <= OUTPUT_CAP
        assert "[line truncated" in out
        assert "lines 1-1 of 1" in out


# --- list_files ------------------------------------------------------------------------------


class TestListFiles:
    def test_sizes_and_relative_paths(self, tools: Tools) -> None:
        out = list_(tools)
        assert re.search(r"^notes\.md  \(\d+ B\)$", out, re.MULTILINE)
        assert "src/" in out
        assert "src/app.py" in out
        assert "data/sales.csv" in out

    def test_skip_dirs(self, root: Path, tools: Tools) -> None:
        skipped = [".git", "node_modules", ".venv", "__pycache__", "dist"]
        for name in skipped:
            write(root, f"{name}/inside.txt", "x")
        out = list_(tools)
        for name in skipped:
            assert name not in out
        assert "inside.txt" not in out

    def test_depth(self, root: Path, tools: Tools) -> None:
        write(root, "a/b/c/d/deep.txt", "x")
        out = list_(tools, max_depth=2)
        assert "a/" in out
        assert "a/b/" in out
        assert "a/b/c/" not in out
        assert "deep.txt" not in out
        assert "not expanded" in out

        out = list_(tools, max_depth=4)
        assert "a/b/c/d/" in out
        assert "deep.txt" not in out

        assert "a/b/c/d/deep.txt" in list_(tools, max_depth=10)
        assert "a/b/c/d/deep.txt" in list_(tools, path="a/b/c", max_depth=2)
        assert "a/b/" not in list_(tools, max_depth=1)
        assert "a/b/" in list_(tools, max_depth=999)  # clamped, not an error

    def test_entry_cap(self, tmp_path: Path) -> None:
        flat = tmp_path / "flat"  # short names keep 500 entries under the character cap
        flat.mkdir()
        for i in range(MAX_LIST_ENTRIES + 100):
            (flat / f"{i:03d}").write_text("")
        tools = {tool.name: tool for tool in make_workspace_tools(Workspace(flat))}
        out = list_(tools)
        lines = out.splitlines()
        assert len([line for line in lines if not line.startswith("[")]) == MAX_LIST_ENTRIES
        assert "truncated" in lines[-1]
        assert len(out) <= OUTPUT_CAP

    def test_char_cap(self, root: Path, tools: Tools) -> None:
        wide = root / "wide"
        wide.mkdir()
        for i in range(400):
            (wide / f"{'n' * 60}{i:03d}.txt").write_text("")
        out = list_(tools, path="wide")
        assert len(out) <= OUTPUT_CAP
        assert "truncated" in out

    def test_empty_dir(self, root: Path, tools: Tools) -> None:
        (root / "empty").mkdir()
        out = list_(tools, path="empty")
        assert out
        assert "empty" in out

    def test_symlinked_dir_is_not_expanded(self, root: Path, tools: Tools, tmp_path: Path) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "leak.txt").write_text("x")
        symlink_or_skip(outside, root / "linked")
        out = list_(tools)
        assert "leak.txt" not in out
        assert "linked/" in out
        assert "symlink" in out

    def test_file_path_and_missing(self, tools: Tools) -> None:
        out = list_(tools, path="notes.md")
        assert out.startswith("error:")
        assert "read_file" in out
        assert list_(tools, path="nope").startswith("error:")


# --- search_files ----------------------------------------------------------------------------


class TestSearchFiles:
    def test_case_insensitive_substring_and_format(self, tools: Tools) -> None:
        out = search(tools, "TODO")
        assert "notes.md:2: - TODO: one" in out
        assert "notes.md:4: - todo: two" in out
        assert "notes.md:3" not in out

    def test_substring_not_regex(self, root: Path, tools: Tools) -> None:
        write(root, "re.txt", "a.c\nabc\n")
        out = search(tools, "a.c")
        assert "re.txt:1: a.c" in out
        assert "re.txt:2" not in out
        assert not search(tools, "(a+)+$").startswith("error:")

    def test_glob_filters_names(self, tools: Tools) -> None:
        out = search(tools, "import", glob="*.py")
        assert "src/app.py:1: import csv" in out
        assert "no matches" in search(tools, "todo", glob="*.py")
        assert "data/sales.csv:2: 1,2" in search(tools, "1", glob="*.CSV")
        assert "src/app.py:1" in search(tools, "import", glob="src/*")
        # "**/" is what models write for "anywhere"; fnmatch alone would miss top-level files
        assert "notes.md:1" in search(tools, "notes", glob="**/*.md")
        assert "src/app.py:1" in search(tools, "import", glob="**/*.py")
        assert "notes.md:1" in search(tools, "notes", glob="**/")

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs are POSIX-only")
    def test_fifo_in_tree_does_not_block_search(self, root: Path, tools: Tools) -> None:
        os.mkfifo(root / "src" / "pipe")  # open() for reading blocks until a writer appears
        out = call_with_timeout(lambda: search(tools, "import"))
        assert "src/app.py:1: import csv" in out
        assert "pipe" not in out
        out = call_with_timeout(lambda: search(tools, "x", path="src/pipe"))
        assert out.startswith("no matches")
        assert "0 text files" in out

    def test_oversized_files_are_reported_not_silently_skipped(
        self, root: Path, tools: Tools
    ) -> None:
        write(root, "huge.log", b"needle\n" * (MAX_SEARCH_FILE_BYTES // 7 + 1))
        out = search(tools, "needle")
        assert "no matches" in out
        assert "huge.log" not in out
        assert "1 file over" in out
        assert "run_python" in out
        write(root, "small.log", "needle\n")
        out = search(tools, "needle")
        assert "small.log:1: needle" in out
        assert out.splitlines()[-1].startswith("[1 file over")

    def test_path_may_be_a_single_file(self, root: Path, tools: Tools) -> None:
        write(root, "other.md", "todo elsewhere\n")
        out = search(tools, "todo", path="notes.md")
        assert "notes.md:2" in out
        assert "other.md" not in out

    def test_skips_binaries_secrets_skip_dirs_and_symlinks(
        self, root: Path, tools: Tools, tmp_path: Path
    ) -> None:
        write(root, "blob.bin", b"needle\x00\x01\x02")
        write(root, ".env", "NEEDLE=1\n")
        write(root, "node_modules/pkg/index.js", "needle\n")
        write(root, "plain.txt", "needle here\n")
        outside = tmp_path / "o.txt"
        outside.write_text("needle outside\n")
        with contextlib.suppress(OSError, NotImplementedError):  # no symlink: still a valid test
            (root / "link.txt").symlink_to(outside)
        out = search(tools, "needle")
        assert "plain.txt:1: needle here" in out
        assert "blob.bin" not in out
        assert ".env:1" not in out and "NEEDLE=1" not in out
        assert "1 credential file (.env, keys and similar) not searched" in out
        assert "node_modules" not in out
        assert "link.txt" not in out

    def test_cp1252_files_are_searchable(self, root: Path, tools: Tools) -> None:
        write(root, "legacy.txt", b"un caf\xe9 noir\n")
        assert "legacy.txt:1: un café noir" in search(tools, "CAFÉ")

    def test_max_results_clamped(self, root: Path, tools: Tools) -> None:
        write(root, "many.txt", "hit\n" * 300)
        out = search(tools, "hit", max_results=1000)
        hits = [line for line in out.splitlines() if line.startswith("many.txt:")]
        assert len(hits) == MAX_SEARCH_RESULTS
        assert f"stopped at {MAX_SEARCH_RESULTS} matches" in out
        assert len([line for line in search(tools, "hit").splitlines() if ":" in line]) == 100
        assert search(tools, "hit", max_results=0).count("many.txt:") == 1

    def test_deadline_is_cooperative_and_injectable(
        self, tools: Tools, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(wsmod, "SEARCH_DEADLINE_S", 0.0)
        out = search(tools, "todo")
        assert out
        assert "no matches" in out
        assert "time limit" in out
        monkeypatch.setattr(wsmod, "SEARCH_DEADLINE_S", 30.0)
        assert "notes.md:2" in search(tools, "todo")

    def test_empty_results_are_a_message(self, tools: Tools) -> None:
        out = search(tools, "zzz-nothing-here")
        assert out.startswith("no matches for 'zzz-nothing-here'")
        assert "4 text files" in out

    def test_empty_pattern(self, tools: Tools) -> None:
        assert search(tools, "").startswith("error:")

    def test_output_cap(self, root: Path, tools: Tools) -> None:
        write(root, "wide.txt", ("hit " + "x" * 300 + "\n") * 100)
        out = search(tools, "hit", max_results=200)
        assert len(out) <= OUTPUT_CAP
        assert "truncated" in out
        assert "..." in out  # long matching lines are cut


# --- helpers ---------------------------------------------------------------------------------


def test_split_lines() -> None:
    assert wsmod._split_lines("a\r\nb\rc\n") == ["a", "b", "c"]
    assert wsmod._split_lines("") == []
    assert wsmod._split_lines("a\n\n") == ["a", ""]
    assert wsmod._split_lines("no newline") == ["no newline"]


def test_looks_binary() -> None:
    assert not wsmod._looks_binary(b"")
    assert not wsmod._looks_binary(b"plain text\n\twith tabs")
    assert wsmod._looks_binary(b"abc\x00def")
    assert wsmod._looks_binary(bytes(range(1, 32)) * 10)
