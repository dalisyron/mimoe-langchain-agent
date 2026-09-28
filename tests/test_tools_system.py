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
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from enum import StrEnum
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

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

_INSTANT = datetime(2026, 9, 24, 19, 0, 0, tzinfo=UTC)  # noon in Vancouver, Friday 04:00 in Tokyo
_VANCOUVER = ZoneInfo("America/Vancouver")
_WINDOWS = ("Pacific Standard Time", "Pacific Daylight Time")  # what Windows calls that zone
_LOCAL = "User's local time (America/Vancouver): Thursday 2026-09-24 12:00:00 PDT (UTC-07:00)"


def _freeze(
    monkeypatch: pytest.MonkeyPatch,
    instant: datetime,
    rules: tzinfo = _VANCOUVER,
    name: str | None = "America/Vancouver",
    names: tuple[str, str] = ("PST", "PDT"),
) -> None:
    """Stop the clock at ``instant`` on an OS whose zone follows ``rules``, calls its standard and
    daylight time ``names`` (``time.tzname``) and has ``name`` as the detected IANA zone. The OS
    shows a moment as ``datetime.astimezone()`` does: a fixed offset named by the abbreviation."""

    def os_clock(moment: datetime) -> datetime:
        at = moment.astimezone(rules)
        return at.astimezone(timezone(at.utcoffset() or timedelta(0), names[bool(at.dst())]))

    monkeypatch.setattr(system, "_os_clock", os_clock)
    monkeypatch.setattr(system, "_current_time", lambda: os_clock(instant))
    monkeypatch.setattr(system, "_local_zone_name", lambda: name)
    monkeypatch.setattr(time, "tzname", names)


@pytest.fixture
def frozen_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Mac in Vancouver at noon on Thursday 2026-09-24."""
    _freeze(monkeypatch, _INSTANT)


def test_now_local_time_comes_first(frozen_clock: None) -> None:
    assert format_now("Asia/Tokyo") == (
        f"{_LOCAL}\nIn Asia/Tokyo: Friday 2026-09-25 04:00:00 JST (UTC+09:00)"
    )
    assert format_now("UTC") == f"{_LOCAL}\nIn UTC: Thursday 2026-09-24 19:00:00 UTC (UTC+00:00)"


def test_now_what_day_is_it_whatever_zone_the_model_guesses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reported case: qwen3-4b called now(timezone='America/New_York') for "What day is it?"
    on a Mac in Vancouver. At 23:30 there it is Monday in New York already; the user's Sunday
    comes first."""
    _freeze(monkeypatch, datetime(2026, 9, 28, 6, 30, tzinfo=UTC))
    assert format_now("America/New_York").splitlines() == [
        "User's local time (America/Vancouver): Sunday 2026-09-27 23:30:00 PDT (UTC-07:00)",
        "In America/New_York: Monday 2026-09-28 02:30:00 EDT (UTC-04:00)",
    ]


@pytest.mark.parametrize(
    "key",
    [
        None,
        "",
        "local",
        "Local Time",
        "null",  # small models send "null" too
        "'none'",
        "America/Vancouver",
        "america/vancouver",
        "Vancouver",
        "vancouver",
        "user's local time",  # the description's own words
        "the user’s local time zone",
        "Local timezone",
        "my time zone",
        "pdt",  # the abbreviation the clock shows now
        "-",  # no word at all names no place
        "…",
    ],
)
def test_now_the_local_zone_is_one_line(frozen_clock: None, key: str | None) -> None:
    assert format_now(key) == _LOCAL


@pytest.mark.parametrize("key", ["America/Los_Angeles", "PST", "Pacific Time"])
def test_now_a_zone_showing_the_local_time_is_one_line(frozen_clock: None, key: str) -> None:
    """The same UTC offset and abbreviation as the user's zone right now: one line that says so."""
    assert format_now(key) == f"{_LOCAL}, also the time in America/Los_Angeles"


def test_now_a_zone_keeping_the_local_time_all_year_is_one_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Other abbreviations, but the user's offset now, in January and in July: one line. The same
    offset only now (Phoenix keeps UTC-7 through the winter) is two."""
    _freeze(monkeypatch, _INSTANT, ZoneInfo("Asia/Kolkata"), "Asia/Kolkata", ("IST", "IST"))
    local = "User's local time (Asia/Kolkata): Friday 2026-09-25 00:30:00 IST (UTC+05:30)"
    assert format_now("Asia/Colombo") == f"{local}, also the time in Asia/Colombo"
    _freeze(monkeypatch, _INSTANT)
    assert format_now("America/Phoenix") == (
        f"{_LOCAL}\nIn America/Phoenix: Thursday 2026-09-24 12:00:00 MST (UTC-07:00)"
    )


@pytest.mark.parametrize(
    ("key", "line"),
    [
        ("utc", "In UTC: Thursday 2026-09-24 19:00:00 UTC (UTC+00:00)"),  # any case, any disk
        ("europe/berlin", "In Europe/Berlin: Thursday 2026-09-24 21:00:00 CEST (UTC+02:00)"),
        ("America/New York", "In America/New_York: Thursday 2026-09-24 15:00:00 EDT (UTC-04:00)"),
        ("new york", "In America/New_York: Thursday 2026-09-24 15:00:00 EDT (UTC-04:00)"),
        ("New York City", "In America/New_York: Thursday 2026-09-24 15:00:00 EDT (UTC-04:00)"),
        ("Tokyo", "In Asia/Tokyo: Friday 2026-09-25 04:00:00 JST (UTC+09:00)"),
        ("Apia", "In Pacific/Apia: Friday 2026-09-25 08:00:00 +13 (UTC+13:00)"),  # Samoa
        ("São Paulo", "In America/Sao_Paulo: Thursday 2026-09-24 16:00:00 -03 (UTC-03:00)"),
        # America/Buenos_Aires is a link that shows the same time: the current, deeper key wins
        (
            "buenos aires",
            "In America/Argentina/Buenos_Aires: Thursday 2026-09-24 16:00:00 -03 (UTC-03:00)",
        ),
        # people mean Eastern time; the IANA key "EST" would be a fixed UTC-5 all summer
        ("EST", "In America/New_York: Thursday 2026-09-24 15:00:00 EDT (UTC-04:00)"),
        (
            "Central Standard Time",
            "In America/Chicago: Thursday 2026-09-24 14:00:00 CDT (UTC-05:00)",
        ),
        ("UTC+2", "In UTC+02:00: Thursday 2026-09-24 21:00:00 (UTC+02:00)"),
        ("+05:30", "In UTC+05:30: Friday 2026-09-25 00:30:00 (UTC+05:30)"),
        ("GMT-3", "In UTC-03:00: Thursday 2026-09-24 16:00:00 (UTC-03:00)"),
        ("UTC-0", "In UTC+00:00: Thursday 2026-09-24 19:00:00 (UTC+00:00)"),  # zero has no sign
    ],
)
def test_now_tolerant_zone_spelling(frozen_clock: None, key: str, line: str) -> None:
    assert format_now(key) == f"{_LOCAL}\n{line}"


def test_now_city_in_several_zones(frozen_clock: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keys ending in one city that show different times are an error naming them, never a
    guess; keys that show the same time now count as one."""
    cities = {
        "springfield": ("America/Chicago", "America/New_York"),
        "twin": ("Europe/Paris", "Europe/Berlin"),
    }
    monkeypatch.setattr(system, "_city_index", lambda: cities)
    assert format_now("Springfield") == (
        "ERROR: 'Springfield' matches several time zones (America/Chicago, America/New_York); "
        "pass one of them."
    )
    assert format_now("twin").splitlines()[1] == (
        "In Europe/Berlin: Thursday 2026-09-24 21:00:00 CEST (UTC+02:00)"
    )


@pytest.mark.parametrize(
    "key",
    [
        "Mars/Olympus",
        "../../etc/passwd",
        "/etc/localtime",
        "UTC+25",
        "IST",  # India, Israel or Ireland: not guessed
        "Canada/Vancouver",  # not a key; only a bare city is looked up by name
        "Victoria",  # Australia/Victoria is a state (Melbourne), not a city
        "Samoa",  # Pacific/Samoa is American Samoa; the country keeps Pacific/Apia
        "East",  # Brazil/East
        # a place in another script is a place, never "no zone": the model retries in English
        "東京",
        "Москва",
        "Αθήνα",
        "서울",
        "القاهرة",
        "🗼",
        "\x00",
        "x" * 5_000,
    ],
)
def test_now_invalid_zone_message(frozen_clock: None, key: str) -> None:
    out = format_now(key)
    assert out.startswith("ERROR: unknown time zone") and "IANA" in out, out
    assert "leave timezone out for the user's local time" in out and len(out) < 250


def test_now_city_lookup_skips_region_links(monkeypatch: pytest.MonkeyPatch) -> None:
    """Old links named after a state or region are no city, on disks that still have them."""
    keys = ["Australia/Melbourne", "Australia/Victoria", "Pacific/Apia", "Pacific/Samoa"]
    keys += ["US/Hawaii", "Brazil/East", "Australia/Canberra", "America/North_Dakota/Center"]
    monkeypatch.setattr(system, "_zone_index", lambda: {system._fold(key): key for key in keys})
    assert system._city_index.__wrapped__() == {
        "melbourne": ("Australia/Melbourne",),
        "canberra": ("Australia/Canberra",),  # a city, though an old link too
        "apia": ("Pacific/Apia",),
        "center": ("America/North_Dakota/Center",),  # a town
    }


def test_now_without_a_zone_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows gives no IANA name for its zone: the user's time still comes first, labelled with
    what the OS calls the zone. The user's own zone asked by name is one line, although Windows
    never calls it 'PDT'."""
    _freeze(monkeypatch, _INSTANT, name=None, names=_WINDOWS)
    local = "User's local time: Thursday 2026-09-24 12:00:00 Pacific Daylight Time (UTC-07:00)"
    assert format_now(None) == local
    assert format_now("Asia/Tokyo") == (
        f"{local}\nIn Asia/Tokyo: Friday 2026-09-25 04:00:00 JST (UTC+09:00)"
    )
    for key in ("America/Vancouver", "Vancouver"):
        assert format_now(key) == f"{local}, also the time in America/Vancouver", key
    assert format_now("PST") == f"{local}, also the time in America/Los_Angeles"
    assert format_now("America/Phoenix").splitlines()[1].startswith("In America/Phoenix: ")


@pytest.mark.parametrize("names", [("PST", "PDT"), _WINDOWS])  # a container, Windows
def test_now_the_zone_the_description_names_is_local(
    monkeypatch: pytest.MonkeyPatch, names: tuple[str, str]
) -> None:
    """Where the OS gives no IANA name, a model that copies the zone the description names, or the
    abbreviation the clock shows, gets the user's time instead of an error."""
    _freeze(monkeypatch, _INSTANT, name=None, names=names)
    label = "/".join(names)
    assert f"the user's local time ({label})." in system._make_now().description
    local = f"User's local time: Thursday 2026-09-24 12:00:00 {names[1]} (UTC-07:00)"
    for key in (label, names[1]):
        assert format_now(key) == local, key


def test_now_the_other_half_of_the_zone_name_is_elsewhere(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the abbreviation the clock shows now is the user's time: 'GMT' in a London summer is
    an hour behind it."""
    _freeze(monkeypatch, _INSTANT, ZoneInfo("Europe/London"), None, ("GMT", "BST"))
    assert format_now("GMT") == (
        "User's local time: Thursday 2026-09-24 20:00:00 BST (UTC+01:00)\n"
        "In GMT: Thursday 2026-09-24 19:00:00 GMT (UTC+00:00)"
    )


def test_now_a_zone_name_the_clock_contradicts_is_left_out(monkeypatch: pytest.MonkeyPatch) -> None:
    _freeze(monkeypatch, _INSTANT, name="Asia/Tokyo")  # say, the zone changed after the start
    assert format_now(None) == "User's local time: Thursday 2026-09-24 12:00:00 PDT (UTC-07:00)"


def test_now_description_names_the_local_zone(frozen_clock: None) -> None:
    description = system._make_now().description
    assert "Without timezone it is the user's local time (America/Vancouver)." in description
    assert "only when the user asks about another place" in description


@pytest.mark.parametrize(("names", "label"), [(("UTC", "UTC"), "UTC"), (("", ""), "UTC-07:00")])
def test_now_description_without_a_zone_name(
    monkeypatch: pytest.MonkeyPatch, names: tuple[str, str], label: str
) -> None:
    """One name for both halves, or none (the other cases: see the test above)."""
    _freeze(monkeypatch, _INSTANT, name=None, names=names)
    description = system._make_now().description
    assert f"Without timezone it is the user's local time ({label})." in description


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("America/Vancouver", "America/Vancouver"),  # TZ or /etc/timezone
        (":Asia/Tokyo", "Asia/Tokyo"),  # TZ with POSIX's optional colon
        ("/var/db/timezone/zoneinfo/America/Vancouver", "America/Vancouver"),  # macOS link
        ("../usr/share/zoneinfo/Europe/Berlin", "Europe/Berlin"),  # a relative link
        ("/usr/share/zoneinfo/posix/America/New_York", "America/New_York"),
        ("/usr/share/zoneinfo.default/Asia/Tokyo", "Asia/Tokyo"),
        ("PST8PDT,M3.2.0,M11.1.0", None),  # a POSIX rule names no zone
        ("/etc/localtime", None),
        ("../../etc/passwd", None),
        ("", None),
    ],
)
def test_now_zone_key_in(text: str, key: str | None) -> None:
    assert system._zone_key_in(text) == key


def test_now_detects_the_zone_from_tz(monkeypatch: pytest.MonkeyPatch) -> None:
    """A set TZ wins over /etc/localtime, as in the C library, even when it names no zone."""
    monkeypatch.setenv("TZ", ":America/Vancouver")
    assert system._detect_local_zone() == "America/Vancouver"
    monkeypatch.setenv("TZ", "EST5EDT,M3.2.0,M11.1.0")
    assert system._detect_local_zone() is None


def test_now_real_clock_format() -> None:
    local = format_now(None)
    assert local.startswith("User's local time") and "\n" not in local
    utc = format_now("UTC")  # its own line, or the local line on a machine that keeps UTC
    assert "(UTC+00:00)" in utc and str(datetime.now(UTC).year) in utc, utc
    far = format_now("Pacific/Chatham")  # UTC+12:45/+13:45, which no test machine keeps
    assert far.startswith("User's local time")
    assert re.fullmatch(
        r"In Pacific/Chatham: \w+ \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \+1[23]45 \(UTC\+1[23]:45\)",
        far.splitlines()[-1],
    ), far


def test_now_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*_: object) -> datetime:
        raise OSError("no clock")

    _freeze(monkeypatch, _INSTANT)
    monkeypatch.setattr(system, "_local_zone_name", broken)  # detection fails: no name, same time
    assert format_now(None) == "User's local time: Thursday 2026-09-24 12:00:00 PDT (UTC-07:00)"
    _freeze(monkeypatch, _INSTANT, name=None, names=_WINDOWS)
    monkeypatch.setattr(system, "_os_clock", broken)  # January and July cannot be read: two lines
    assert format_now("America/Vancouver").splitlines()[1].startswith("In America/Vancouver: ")
    monkeypatch.setattr(system, "_current_time", broken)
    assert format_now("Asia/Tokyo") == "ERROR: could not read the clock (no clock)."
    assert "the user's local time. Pass" in system._make_now().description


def test_now_tool_contract(frozen_clock: None) -> None:
    assert now.name == "now"
    assert list(now.args) == ["timezone"]
    assert now.invoke({}) == _LOCAL
    assert now.invoke({"timezone": "Asia/Tokyo"}).startswith(f"{_LOCAL}\nIn Asia/Tokyo: ")
    assert "Without timezone it is the user's local time" in now.description


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
