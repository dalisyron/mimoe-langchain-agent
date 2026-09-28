"""The REPL's terminal layer (:mod:`mimoe_agent.terminal`): the approval menu's keys, timing and
look, the ``you>`` line reader, the type-ahead flush and the light/dark guess with the keys it
gives back. prompt_toolkit reads from its pipe input and draws on ``DummyOutput`` (or a VT100
output into a string); the OSC 11 query, the late-answer drop and the flush of the terminal's
queue run against a pseudo-terminal pair on POSIX. The menu's timing rules see each key at its
time in the test's script (:class:`_KeyClock`), not when a busy machine delivered it; keys that
must arrive while a prompt is up are sent from a thread (:func:`typist`)."""

from __future__ import annotations

import asyncio
import io
import os
import re
import select
import sys
import threading
import time
from collections.abc import Callable, Iterator
from types import SimpleNamespace

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import PipeInput, create_pipe_input
from prompt_toolkit.input.typeahead import store_typeahead
from prompt_toolkit.key_binding import KeyPress
from prompt_toolkit.keys import Keys
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.output.vt100 import Vt100_Output
from rich.console import Console

from mimoe_agent import terminal
from mimoe_agent.terminal import (
    CANCEL,
    NO,
    YES,
    LineReader,
    Reply,
    approval_menu,
    ask_approval,
    discard_typeahead,
    drop_late_answer,
    parse_background_reply,
    parse_colorfgbg,
    parse_reply,
    query_background,
)

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX terminals (termios)")
QUESTION = "Run this code?"
SGR_RE = re.compile(r"\x1b\[([0-9;]*)m")


@pytest.fixture
def pipe() -> Iterator[PipeInput]:
    """prompt_toolkit reads keys from a pipe and draws nothing."""
    with create_pipe_input() as pipe_input, create_app_session(pipe_input, DummyOutput()):
        yield pipe_input


@pytest.fixture
def instant_menu(monkeypatch: pytest.MonkeyPatch) -> None:
    """The menu takes the first key at once and closes at once: for the tests of what each key
    does. Its timing rules have tests of their own."""
    monkeypatch.setattr(terminal, "ARM_DELAY_S", 0)
    monkeypatch.setattr(terminal, "SETTLE_S", 0)


Script = list[tuple[float, str]]
"""Keys to type: ``(seconds, text)`` pairs."""


@pytest.fixture
def typist(pipe: PipeInput) -> Iterator[Callable[[Script], None]]:
    """Types into the pipe from a thread, each ``(seconds, text)`` that many seconds after the
    call, as a person would. The thread is stopped and joined when the test ends, pass or fail:
    left running, it typed into a later test's pipe (a closed pipe's descriptors are reused)."""
    stop = threading.Event()
    threads: list[threading.Thread] = []

    def type_keys(keys: Script) -> None:
        def run() -> None:
            started = time.monotonic()
            for at, text in keys:
                if stop.wait(max(0.0, at - (time.monotonic() - started))):
                    return
                pipe.send_text(text)

        thread = threading.Thread(target=run, daemon=True)
        threads.append(thread)
        thread.start()

    yield type_keys
    stop.set()
    for thread in threads:
        thread.join(5.0)


class _KeyClock:
    """The menu's clock in a test, moved by the keys: each key is seen at its time in the script
    (seconds after the menu was drawn) however late a busy machine delivers it, and after the
    last key time flows as usual. So a test checks the timing rules, not the scheduler: with
    real sleeps a key that woke 0.1 s late armed the menu or closed it early."""

    def __init__(self, times: list[float]) -> None:
        self._times = iter(times)
        self._at = 0.0
        self._flowing_since: float | None = None

    def monotonic(self) -> float:
        if self._flowing_since is None:
            return self._at
        return self._at + time.monotonic() - self._flowing_since

    def next_key(self, *_: object) -> None:
        at = next(self._times, None)
        if at is None:
            self._flowing_since = time.monotonic()
        else:
            self._at = at


def _answer(pipe: PipeInput, monkeypatch: pytest.MonkeyPatch, keys: Script) -> str:
    """The menu's answer to a script of keys, one key per entry, all in the pipe before it
    starts (see :class:`_KeyClock`)."""
    clock = _KeyClock([at for at, _ in keys])
    monkeypatch.setattr(terminal, "time", clock)
    app = approval_menu(QUESTION)
    app.on_reset += clock.next_key  # after the menu's own start (at 0): the first key's time
    app.key_processor.after_key_press += clock.next_key
    pipe.send_text("".join(text for _, text in keys))

    async def watchdog() -> None:  # a menu that never answers fails the test, not hangs it
        await asyncio.sleep(10)
        app.exit(result="(no answer after 10 s)")

    return app.run(pre_run=lambda: app.create_background_task(watchdog()))


def _vt100(buffer: io.StringIO) -> Vt100_Output:
    return Vt100_Output(
        buffer, lambda: Size(rows=24, columns=80), term="xterm-256color", enable_cpr=False
    )


class _TtyBuffer(io.StringIO):
    """Drawn output that says it is a terminal, as prompt_toolkit's CPR check wants."""

    def isatty(self) -> bool:
        return True


def _sgr_before(drawn: str, text: str) -> set[str]:
    """The parameters of the last SGR sequence before the last ``text`` in ``drawn``."""
    head = drawn[: drawn.rindex(text)]
    codes = SGR_RE.findall(head)
    return set(codes[-1].split(";")) if codes else set()


# -- the approval menu ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("keys", "expected"),
    [
        ("\r", YES),  # Enter on the highlighted default
        ("\x1b[B\r", NO),  # Down, Enter
        ("\x1b[B\x1b[A\r", YES),  # Down, Up, Enter
        ("\x1b[A\r", YES),  # Up on the first row stays there
        ("\x1b[B\x1b[B\r", NO),  # Down on the last row stays there
        ("j\r", NO),
        ("jk\r", YES),
        ("\x0e\r", NO),  # Ctrl-N
        ("\x0e\x10\r", YES),  # Ctrl-N, Ctrl-P
        ("1", YES),
        ("2", NO),
        ("y", YES),
        ("n", NO),
        ("Y", YES),
        ("N", NO),
        ("x\r", YES),  # other keys do nothing
        ("\x1b", CANCEL),  # Esc
        ("\x03", CANCEL),  # Ctrl-C
    ],
)
@pytest.mark.usefixtures("instant_menu")
def test_menu_keys(pipe: PipeInput, keys: str, expected: str) -> None:
    pipe.send_text(keys)
    assert ask_approval(QUESTION) == expected


def test_menu_esc_answers_at_once(pipe: PipeInput) -> None:
    """A lone Esc is told from an escape sequence after ttimeoutlen (0.5 s by default)."""
    app = approval_menu(QUESTION)
    assert app.ttimeoutlen <= 0.1
    pipe.send_text("\x1b")
    started = time.monotonic()
    assert app.run() == CANCEL
    assert time.monotonic() - started < 0.45


@pytest.mark.usefixtures("instant_menu")
def test_menu_draws_like_claude_codes_permission_prompt() -> None:
    drawn = io.StringIO()
    with create_pipe_input() as pipe_input, create_app_session(pipe_input, _vt100(drawn)):
        pipe_input.send_text("\r")
        assert ask_approval(QUESTION) == YES
    out = drawn.getvalue()
    lines = [SGR_RE.sub("", line) for line in out.split("\r\n")]
    assert lines[-5].endswith(QUESTION) and lines[-4:-1] == ["❯ 1. Yes", "  2. No", ""]
    assert lines[-1].startswith("Esc to cancel")
    assert {"1", "36"} <= _sgr_before(out, "❯ 1. Yes")  # bold, ANSI cyan
    assert _sgr_before(out, "  2. No") == {"0"}
    assert "2" in _sgr_before(out, "Esc to cancel")  # dim
    assert "38;" not in out and "48;" not in out  # palette colours only, no RGB or 256-colour
    # erased when done: back up over the menu, then erase down
    assert re.search(r"Esc to cancel.*\x1b\[4A.*\x1b\[J", out, re.DOTALL)


@pytest.mark.usefixtures("instant_menu")
def test_menu_ascii_pointer() -> None:
    drawn = io.StringIO()
    with create_pipe_input() as pipe_input, create_app_session(pipe_input, _vt100(drawn)):
        pipe_input.send_text("2")
        assert ask_approval(QUESTION, pointer=">") == NO
    plain = SGR_RE.sub("", drawn.getvalue())
    assert "> 1. Yes\r\n  2. No" in plain and "❯" not in plain


def test_menu_pointer_falls_back_to_ascii() -> None:
    assert terminal.menu_pointer(Console(file=io.StringIO(), legacy_windows=False)) == "❯"
    cp1252 = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    assert terminal.menu_pointer(Console(file=cp1252, legacy_windows=False)) == ">"
    assert terminal.menu_pointer(Console(file=io.StringIO(), legacy_windows=True)) == ">"


@pytest.mark.usefixtures("instant_menu")
def test_menu_ignores_keys_a_previous_prompt_read_ahead(pipe: PipeInput) -> None:
    """An Enter typed right after the question was sent is read by the ``you>`` prompt and kept
    for the next prompt_toolkit application: the menu must not take it as a Yes."""
    store_typeahead(pipe, [KeyPress(Keys.ControlM, "\r")])
    pipe.send_text("2")
    assert ask_approval(QUESTION) == NO


def test_menu_drops_keys_until_the_keyboard_was_quiet(
    pipe: PipeInput, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An impatient user presses Enter again and again while the model works, and one of the
    presses lands just after the menu appeared: it must not approve code nobody read. Each key
    that comes sooner than ARM_DELAY_S after the last one (or the first draw) is dropped."""
    monkeypatch.setattr(terminal, "ARM_DELAY_S", 0.3)
    monkeypatch.setattr(terminal, "SETTLE_S", 0)
    enters = [(0.0, "\r"), (0.2, "\r"), (0.4, "\r"), (0.6, "\r")]
    assert _answer(pipe, monkeypatch, [*enters, (1.1, "2")]) == NO  # the 2 after a pause counts
    assert _answer(pipe, monkeypatch, [*enters, (0.9, "\r")]) == YES  # so does an Enter
    assert _answer(pipe, monkeypatch, [(0.2, "2"), (0.5, "\r")]) == YES  # 0.2 s after the draw


def test_menu_drops_a_sentence_typed_across_its_appearance(
    pipe: PipeInput, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The user types the next question while the model works; the menu appears mid-sentence.
    The y of "my" (or the Enter at the end) must not answer it."""
    monkeypatch.setattr(terminal, "ARM_DELAY_S", 0.3)
    monkeypatch.setattr(terminal, "SETTLE_S", 0)
    sentence = "summarize my notes\r"
    keys = [(0.05 * index, char) for index, char in enumerate(sentence)]
    assert _answer(pipe, monkeypatch, [*keys, (0.05 * len(sentence) + 0.5, "n")]) == NO


@pytest.mark.parametrize("key", ["\x1b", "\x03"])
def test_esc_and_ctrl_c_cancel_before_the_menu_takes_answers(
    pipe: PipeInput, monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    monkeypatch.setattr(terminal, "ARM_DELAY_S", 5.0)
    pipe.send_text(key)
    started = time.monotonic()
    assert ask_approval(QUESTION) == CANCEL
    assert time.monotonic() - started < 2.0


def test_menu_with_its_own_timing(pipe: PipeInput, monkeypatch: pytest.MonkeyPatch) -> None:
    """The defaults: a key at once is dropped, a key after a reader's pause answers."""
    assert 0.3 <= terminal.ARM_DELAY_S <= 1.0 and terminal.SETTLE_S < terminal.ARM_DELAY_S
    keys = [(0.05, "\r"), (0.05 + terminal.ARM_DELAY_S + 0.3, "2")]
    assert _answer(pipe, monkeypatch, keys) == NO


@pytest.mark.parametrize("keys", [[(0.0, "y"), (0.1, "\r")], [(0.0, "y"), (0.0, "\r")]])
def test_menu_swallows_the_enter_after_a_letter(
    pipe: PipeInput, monkeypatch: pytest.MonkeyPatch, keys: Script
) -> None:
    """y chooses at once, and an Enter right after it (the old ``[y/N]`` habit) is swallowed:
    left over, it would send an empty line from the next prompt."""
    monkeypatch.setattr(terminal, "ARM_DELAY_S", 0)
    monkeypatch.setattr(terminal, "SETTLE_S", 0.3)
    assert _answer(pipe, monkeypatch, keys) == YES
    pipe.send_text("next\r")
    assert LineReader("you> ").read() == "next"


def test_menu_closes_after_a_letter_once_the_keyboard_is_quiet(
    pipe: PipeInput, monkeypatch: pytest.MonkeyPatch, typist: Callable[[Script], None]
) -> None:
    """In real time: the settling runs on the clock, not on keys. Keys come every 0.1 s, so
    only one that is 0.3 s late would close the menu early."""
    monkeypatch.setattr(terminal, "ARM_DELAY_S", 0)
    monkeypatch.setattr(terminal, "SETTLE_S", 0.4)
    monkeypatch.setattr(terminal, "SETTLE_MAX_S", 0.8)
    pipe.send_text("2")
    started = time.monotonic()
    assert ask_approval(QUESTION) == NO
    assert 0.4 <= time.monotonic() - started < 1.5
    # keys that keep coming ("no" and more) keep it up, but only for SETTLE_MAX_S
    typist([(0.1 * step, "no thanks"[step % 9]) for step in range(17)])
    started = time.monotonic()
    assert ask_approval(QUESTION) == NO
    assert 0.8 <= time.monotonic() - started < 1.6


def test_esc_after_a_letter_still_cancels(pipe: PipeInput, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(terminal, "ARM_DELAY_S", 0)
    monkeypatch.setattr(terminal, "SETTLE_S", 0.5)
    assert _answer(pipe, monkeypatch, [(0.0, "y"), (0.1, "\x1b")]) == CANCEL


def test_menu_and_line_reader_send_no_cursor_position_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without an answer prompt_toolkit printed a warning and held every Enter for a second."""
    monkeypatch.delenv("PROMPT_TOOLKIT_NO_CPR", raising=False)  # it turns CPR off beforehand
    for use in ("line", "menu"):
        drawn = _TtyBuffer()
        output = Vt100_Output(drawn, lambda: Size(rows=24, columns=80), term="xterm-256color")
        assert output.responds_to_cpr  # prompt_toolkit's default on a terminal
        with create_pipe_input() as pipe_input, create_app_session(pipe_input, output):
            if use == "line":
                pipe_input.send_text("hi\r")
                assert LineReader("you> ").read() == "hi"
            else:
                pipe_input.send_text("\x1b")
                assert ask_approval(QUESTION) == CANCEL
        assert not output.responds_to_cpr
        assert "\x1b[6n" not in drawn.getvalue()


class _Fd:
    """The two methods ``discard_typeahead`` needs from a prompt_toolkit input."""

    def __init__(self, fd: int) -> None:
        self.fd = fd

    def fileno(self) -> int:
        return self.fd

    def typeahead_hash(self) -> str:
        return f"test-fd-{self.fd}"


@posix_only
def test_discard_typeahead_flushes_the_terminals_queue() -> None:
    master, slave = os.openpty()
    try:
        os.write(master, b"\r")  # an Enter typed while the code was still being printed
        assert select.select([slave], [], [], 2.0)[0]
        discard_typeahead(_Fd(slave))  # type: ignore[arg-type]
        assert not select.select([slave], [], [], 0.1)[0]
    finally:
        os.close(master)
        os.close(slave)


@pytest.mark.usefixtures("instant_menu")
def test_discard_typeahead_survives_an_input_without_a_queue(pipe: PipeInput) -> None:
    pipe.send_text("n")
    discard_typeahead(pipe)  # a pipe has no terminal queue: nothing is flushed, nothing raised
    assert ask_approval(QUESTION) == NO


# -- the you> line -------------------------------------------------------------------------------


LATER = 0.3
"""Seconds to wait before typing at a prompt: prompt_toolkit loads the history in a task started
by the first render, and keys already waiting (a Windows pipe hands them over at once) would be
handled before it ran, which no one typing can do."""


def test_line_reader_edits_and_recalls_history(
    pipe: PipeInput, typist: Callable[[Script], None]
) -> None:
    reader = LineReader("you> ")
    pipe.send_text("ab\x7fc\r")  # Backspace
    assert reader.read() == "ac"
    pipe.send_text("hello\x15world\r")  # Ctrl-U erases what was typed
    assert reader.read() == "world"
    typist([(LATER, "\x1b[A\x1b[A\r")])  # Up, Up: the line before the last one
    assert reader.read() == "ac"
    typist([(LATER, "\x1b[A\x1b[B\x1b[B\r")])  # Up, then Down past the newest: an empty line
    assert reader.read() == ""


def test_line_reader_ends_like_input(pipe: PipeInput) -> None:
    reader = LineReader("you> ")
    pipe.send_text("half a line\x03")
    with pytest.raises(KeyboardInterrupt):
        reader.read()
    pipe.send_text("\x04")  # Ctrl-D on an empty line
    with pytest.raises(EOFError):
        reader.read()
    pipe.send_text("ab\x04\r")  # Ctrl-D after text is not the end
    assert reader.read() == "ab"
    # Ctrl-Z Enter, the Windows end of input: prompt_toolkit types ^Z literally there. A bare
    # Ctrl-Z would suspend the whole process group on POSIX (the test runner included), so the
    # ^Z goes in through quoted insert (Ctrl-Q) here.
    pipe.send_text("\x11\x1a\r")
    with pytest.raises(EOFError):
        reader.read()


def test_line_reader_draws_the_prompt_in_ansi_cyan_and_typed_text_plain() -> None:
    drawn = io.StringIO()
    with create_pipe_input() as pipe_input, create_app_session(pipe_input, _vt100(drawn)):
        pipe_input.send_text("a\x7fhi\r")
        assert LineReader("you> ").read() == "hi"
    out = drawn.getvalue()
    assert {"1", "36"} <= _sgr_before(out, "you> ")
    assert _sgr_before(out, "hi") == {"0"}
    assert "38;" not in out and "48;" not in out
    # every redraw after a key starts with the prompt: Backspace never leaves the row blank
    frames = out.split("\x1b[J")[1:]
    assert frames and all(SGR_RE.sub("", frame).startswith("you> ") for frame in frames[:-1])


@pytest.mark.parametrize(("keys", "error"), [("typed\x03", KeyboardInterrupt), ("\x04", EOFError)])
def test_line_reader_keeps_the_palette_when_ctrl_c_or_ctrl_d_ends_it(
    keys: str, error: type[BaseException]
) -> None:
    """PromptSession's own keys repaint the ended line in a fixed grey (38;5;102)."""
    drawn = io.StringIO()
    with create_pipe_input() as pipe_input, create_app_session(pipe_input, _vt100(drawn)):
        pipe_input.send_text(keys)
        with pytest.raises(error):
            LineReader("you> ").read()
    out = drawn.getvalue()
    assert "38;" not in out and "48;" not in out
    assert {"1", "36"} <= _sgr_before(out, "you> ")  # the last frame, drawn as done


def test_ctrl_c_while_searching_the_history_only_ends_the_search(
    pipe: PipeInput, typist: Callable[[Script], None]
) -> None:
    reader = LineReader("you> ")
    pipe.send_text("first\r")
    assert reader.read() == "first"
    typist([(LATER, "\x12fir\x03ok\r")])  # Ctrl-R, "fir", Ctrl-C, then a new line
    assert reader.read() == "ok"


def test_line_reader_open_gives_up_when_prompt_toolkit_cannot_drive_the_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import prompt_toolkit

    class NoConsoleScreenBufferError(Exception):
        pass

    def no_console(*args: object, **kwargs: object) -> None:
        raise NoConsoleScreenBufferError("No Windows console found. Are you running cmd.exe?")

    monkeypatch.setattr(prompt_toolkit, "PromptSession", no_console)
    assert LineReader.open("you> ") is None


# -- light or dark -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "light"),
    [
        ("\x1b]11;rgb:ffff/ffff/ffff\x1b\\", True),
        ("\x1b]11;rgb:fdfd/f6f6/e3e3\x07", True),  # Solarized light, BEL-terminated
        ("\x1b]11;rgb:fd/f6/e3\x1b\\", True),  # two hex digits per channel
        ("\x1b]11;rgb:f/f/f\x1b\\", True),  # one
        ("\x1b]11;rgb:fff/fff/fff\x1b\\", True),  # three
        ("\x1b]11;rgba:ffff/ffff/ffff/ffff\x1b\\", True),
        ("\x1b]11;rgb:0000/0000/0000\x1b\\", False),
        ("\x1b]11;rgb:1e1e/1e1e/1e1e\x1b\\\x1b[?62;22c", False),  # a VS Code dark theme
        ("\x1b]11;rgb:2828/2c2c/3434\x07", False),
        ("\x1b]11;rgb:9999/9999/9999\x1b\\", True),  # mid-grey: dark text reads better
        ("\x1b]11;rgb:4040/4040/4040\x1b\\", False),
        ("\x1b[?62;22c", None),  # DA1 only: no OSC 11 support
        ("", None),
        ("\x1b]11;?\x1b\\", None),
    ],
)
def test_parse_background_reply(reply: str, light: bool | None) -> None:
    assert parse_background_reply(reply) is light


@pytest.mark.parametrize(
    ("value", "light"),
    [
        ("11;15", True),  # iTerm2 with a light profile
        ("0;15", True),
        ("0;7", True),
        ("0;default;15", True),  # rxvt's fg;xpm;bg form
        ("15;0", False),
        ("7;8", False),
        ("15;default;0", False),
        ("7;4", False),
        ("12;9", None),  # a bright colour as the background tells nothing
        ("default;default", None),
        ("15", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_colorfgbg(value: str | None, light: bool | None) -> None:
    assert parse_colorfgbg(value) is light


def _mode(fd: int) -> list[object]:
    """The terminal's mode without PENDIN, a state bit BSD kernels set on the switch back to
    line buffering ("retype pending input"), not a setting."""
    import termios

    mode = termios.tcgetattr(fd)
    mode[3] &= ~getattr(termios, "PENDIN", 0)
    return mode


def _terminal(master: int, answer: bytes, seen: list[bytes]) -> None:
    """Play the terminal: read the queries, answer once the DA1 query arrived."""
    received = b""
    end = time.monotonic() + 3.0
    while b"\x1b[c" not in received and time.monotonic() < end:
        if select.select([master], [], [], 0.05)[0]:
            received += os.read(master, 1024)
    seen.append(received[received.find(b"\x1b]11;?") :])  # without the echo of early typing
    if answer:
        os.write(master, answer)


@posix_only
@pytest.mark.parametrize(
    ("early", "answer", "light", "typed"),
    [
        (b"", b"\x1b]11;rgb:fdfd/f6f6/e3e3\x1b\\\x1b[?62;22c", True, b""),
        (b"", b"\x1b]11;rgb:1e1e/1e1e/1e1e\x07\x1b[?1;2c", False, b""),
        (b"", b"\x1b[?6c", None, b""),  # no OSC 11 support: the DA1 answer ends the wait
        # keys typed while the program started, and between the two answers: kept, in order
        (b"hel", b"\x1b]11;rgb:ffff/ffff/ffff\x1b\\lo\x1b[?6c", True, b"hello"),
    ],
)
def test_query_background_asks_the_terminal(
    early: bytes, answer: bytes, light: bool | None, typed: bytes
) -> None:
    master, slave = os.openpty()
    try:
        before = _mode(slave)
        if early:
            os.write(master, early)  # a line still being typed: held by the line discipline
        seen: list[bytes] = []
        player = threading.Thread(target=_terminal, args=(master, answer, seen))
        player.start()
        started = time.monotonic()
        assert query_background(slave, slave, timeout=3.0) == Reply(light, typed, complete=True)
        assert time.monotonic() - started < 2.0  # stopped at the DA1 answer, not the timeout
        player.join(3.0)
        assert seen == [b"\x1b]11;?\x1b\\\x1b[c"]
        assert _mode(slave) == before
    finally:
        os.close(master)
        os.close(slave)


@posix_only
def test_query_background_gives_up_after_the_timeout() -> None:
    master, slave = os.openpty()
    try:
        before = _mode(slave)
        started = time.monotonic()
        # the answer may still come in: the first prompt drops it (drop_late_answer)
        assert query_background(slave, slave, timeout=0.2) == Reply(None, b"", complete=False)
        assert 0.15 < time.monotonic() - started < 1.5
        assert _mode(slave) == before
    finally:
        os.close(master)
        os.close(slave)


def test_the_query_waits_long_enough_for_a_slow_link() -> None:
    """A far ssh host answers after a round trip; 0.3 s let the answer arrive late and become
    typing at the first prompt. The DA1 answer ends the wait, so a long limit costs nothing."""
    assert terminal.QUERY_TIMEOUT_S >= 1.0


@pytest.mark.parametrize(
    ("data", "light", "typed"),
    [
        (b"\x1b]11;rgb:ffff/ffff/ffff\x07\x1b[?6c", True, b""),
        (b"ab\x1b]11;rgb:0000/0000/0000\x1b\\cd\x1b[?62;22cef", False, b"abcdef"),
        (b"\x1b]11;rgb:ffff/ffff/ffff\x9c", True, b""),  # 8-bit ST
        (b"\x1b[?6chi\r", None, b"hi\r"),
        ("hé".encode(), None, "hé".encode()),  # UTF-8 comes back unchanged
        (b"\x1b[A", None, b"\x1b[A"),  # an arrow key is typing, not an answer
    ],
)
def test_parse_reply_splits_answers_from_typing(
    data: bytes, light: bool | None, typed: bytes
) -> None:
    assert parse_reply(data) == Reply(light, typed)


@posix_only
def test_drop_late_answer_keeps_only_what_the_user_typed() -> None:
    """The answer came in after the query gave up, with the user typing around it; the terminal
    is back in line mode, so the half-typed line is held by the line discipline."""
    master, slave = os.openpty()
    try:
        before = _mode(slave)
        os.write(master, b"he\x1b]11;rgb:ffff/ffff/ffff\x1b\\llo\x1b[?62;22c wor")
        time.sleep(0.1)
        assert drop_late_answer(slave) == b"hello wor"
        assert _mode(slave) == before
        assert drop_late_answer(slave) == b""  # nothing queued: no wait
    finally:
        os.close(master)
        os.close(slave)


@posix_only
def test_drop_late_answer_waits_for_an_answer_still_on_its_way() -> None:
    """Over a slow link the answer comes after the query gave up and after the first prompt is
    due: the prompt waits for it (up to its timeout) rather than let it be typed."""
    master, slave = os.openpty()
    answer = b"\x1b]11;rgb:ffff/ffff/ffff\x1b\\\x1b[?62;22c"
    late = threading.Timer(0.3, os.write, [master, b"hi" + answer])
    try:
        before = _mode(slave)
        late.start()
        started = time.monotonic()
        assert drop_late_answer(slave, timeout=3.0) == b"hi"
        assert 0.25 < time.monotonic() - started < 2.0  # stopped at the answer
        started = time.monotonic()
        assert drop_late_answer(slave, timeout=0.2) == b""  # none comes: the timeout ends it
        assert 0.15 < time.monotonic() - started < 1.5
        assert _mode(slave) == before
    finally:
        late.cancel()
        late.join()
        os.close(master)
        os.close(slave)


@posix_only
def test_drop_late_answer_without_a_terminal() -> None:
    read_end, write_end = os.pipe()
    try:
        os.write(write_end, b"x")
        assert drop_late_answer(read_end) == b""  # not a terminal: left alone
        assert os.read(read_end, 10) == b"x"
    finally:
        os.close(read_end)
        os.close(write_end)


@pytest.fixture
def early_input(monkeypatch: pytest.MonkeyPatch) -> terminal._EarlyInput:
    """A fresh store for the keys the start-up query gives back."""
    store = terminal._EarlyInput()
    monkeypatch.setattr(terminal, "_early_input", store)
    return store


def test_the_first_prompt_gets_the_keys_typed_during_start_up(
    pipe: PipeInput, early_input: terminal._EarlyInput
) -> None:
    early_input.keep(Reply(True, b"hello "))
    reader = LineReader("you> ")
    pipe.send_text("world\r")
    assert reader.read() == "hello world"
    pipe.send_text("again\r")
    assert reader.read() == "again"  # given back once


def test_the_first_prompt_drops_a_late_answer_first(
    pipe: PipeInput, early_input: terminal._EarlyInput, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It waits for the answer until LATE_ANSWER_S after the query gave up, whatever time the
    start-up took in between."""
    early_input.keep(Reply(None, b"ab", complete=False))
    waits: list[float] = []
    monkeypatch.setattr(
        terminal, "drop_late_answer", lambda *, timeout: waits.append(timeout) or b"c"
    )
    reader = LineReader("you> ")
    pipe.send_text("\r")
    assert reader.read() == "abc"
    assert len(waits) == 1 and terminal.LATE_ANSWER_S - 1.0 < waits[0] <= terminal.LATE_ANSWER_S
    pipe.send_text("d\r")
    assert reader.read() == "d" and len(waits) == 1
    early_input.keep(Reply(None, b"", complete=False))
    early_input.due_by = time.monotonic() - 1.0  # the start-up took longer than that
    assert early_input.take() == b"c" and waits[1] == 0.0


@posix_only
def test_query_background_restores_the_terminal_after_ctrl_c(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupted(*args: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(terminal, "select", SimpleNamespace(select=interrupted))
    master, slave = os.openpty()
    try:
        before = _mode(slave)
        with pytest.raises(KeyboardInterrupt):
            query_background(slave, slave)
        assert _mode(slave) == before
    finally:
        os.close(master)
        os.close(slave)


@posix_only
def test_query_background_writes_nothing_to_a_pipe() -> None:
    read_end, write_end = os.pipe()
    master, slave = os.openpty()
    try:
        assert query_background(read_end, write_end) == Reply(None)
        assert query_background(slave, write_end) == Reply(None)  # a terminal in, a pipe out
        assert not select.select([read_end], [], [], 0.05)[0]
    finally:
        for fd in (read_end, write_end, master, slave):
            os.close(fd)


def _never(*args: object) -> None:
    raise AssertionError("the terminal must not be queried here")


@pytest.fixture
def posix_terminal(
    monkeypatch: pytest.MonkeyPatch, early_input: terminal._EarlyInput
) -> pytest.MonkeyPatch:
    """A POSIX terminal session as ``light_background`` sees it."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(terminal, "interactive", lambda: True)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("COLORFGBG", raising=False)
    return monkeypatch


def test_light_background_takes_the_terminals_answer_first(
    posix_terminal: pytest.MonkeyPatch,
) -> None:
    posix_terminal.setenv("COLORFGBG", "15;0")  # says dark
    posix_terminal.setattr(terminal, "query_background", lambda: Reply(True))
    assert terminal.light_background() is True
    posix_terminal.setattr(terminal, "query_background", lambda: Reply(False))
    posix_terminal.setenv("COLORFGBG", "0;15")
    assert terminal.light_background() is False


def test_light_background_falls_back_to_colorfgbg_then_dark(
    posix_terminal: pytest.MonkeyPatch,
) -> None:
    posix_terminal.setattr(terminal, "query_background", lambda: Reply(None))  # no OSC 11
    posix_terminal.setenv("COLORFGBG", "11;15")
    assert terminal.light_background() is True
    posix_terminal.setenv("COLORFGBG", "15;0")
    assert terminal.light_background() is False
    posix_terminal.delenv("COLORFGBG")
    assert terminal.light_background() is False


def test_light_background_keeps_the_keys_typed_meanwhile_for_the_first_prompt(
    posix_terminal: pytest.MonkeyPatch, early_input: terminal._EarlyInput
) -> None:
    posix_terminal.setattr(terminal, "query_background", lambda: Reply(None, b"hi", False))
    posix_terminal.setenv("COLORFGBG", "0;15")
    assert terminal.light_background() is True
    assert early_input.typed == b"hi" and early_input.due_by is not None


@posix_only
def test_light_background_survives_streams_without_a_descriptor(
    posix_terminal: pytest.MonkeyPatch,
) -> None:
    """IDLE's shell, for one, says isatty() but has no fileno(): no query, no traceback."""

    class NoDescriptor(io.StringIO):
        def isatty(self) -> bool:
            return True

        def fileno(self) -> int:
            raise io.UnsupportedOperation("fileno")

    posix_terminal.setattr(sys, "stdin", NoDescriptor())
    posix_terminal.setattr(sys, "stdout", NoDescriptor())
    assert query_background() == Reply(None)
    assert terminal.light_background() is False
    assert drop_late_answer() == b""


@pytest.mark.parametrize("where", ["pipe", "windows", "linux console"])
def test_light_background_never_queries_a_pipe_windows_or_the_linux_console(
    posix_terminal: pytest.MonkeyPatch, where: str
) -> None:
    posix_terminal.setattr(terminal, "query_background", _never)
    if where == "pipe":
        posix_terminal.setattr(terminal, "interactive", lambda: False)
    elif where == "windows":
        posix_terminal.setattr(sys, "platform", "win32")
    else:
        posix_terminal.setenv("TERM", "linux")
    posix_terminal.setenv("COLORFGBG", "0;15")
    assert terminal.light_background() is True
    posix_terminal.delenv("COLORFGBG")
    assert terminal.light_background() is False


def test_interactive_needs_two_terminals_that_move_the_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tty = SimpleNamespace(isatty=lambda: True)
    pipe_end = SimpleNamespace(isatty=lambda: False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(sys, "stdin", tty)
    monkeypatch.setattr(sys, "stdout", tty)
    assert terminal.interactive() is True
    monkeypatch.setattr(sys, "stdout", pipe_end)
    assert terminal.interactive() is False
    monkeypatch.setattr(sys, "stdout", tty)
    monkeypatch.setattr(sys, "stdin", pipe_end)
    assert terminal.interactive() is False
    monkeypatch.setattr(sys, "stdin", tty)
    monkeypatch.setenv("TERM", "dumb")  # Emacs' shell mode: no cursor movement
    assert terminal.interactive() is False


def test_code_themes_and_the_console_theme() -> None:
    assert terminal.code_theme(True) == "ansi_light"
    assert terminal.code_theme(False) == "ansi_dark"
    console = Console(file=io.StringIO(), theme=terminal.CONSOLE_THEME)
    for name in ("markdown.code", "markdown.code_block"):
        assert console.get_style(name).bgcolor is None
    assert Console(file=io.StringIO()).get_style("markdown.code").bgcolor is not None
