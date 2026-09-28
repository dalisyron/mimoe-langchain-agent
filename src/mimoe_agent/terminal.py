"""Terminal input and colours for the REPL: the ``you>`` line editor, the approval menu and the
light/dark guess that picks the code colours.

**Line editing.** On a terminal (stdin and stdout both TTYs, see :func:`interactive`)
:class:`LineReader` reads the ``you>`` lines with prompt_toolkit, which draws the prompt itself,
so Backspace and Ctrl-U erase only what was typed and Up/Down recall the earlier lines of the
session. Python's ``input()`` could not do that here: rich printed the prompt and called
``input("")``, and libedit (the readline of the macOS Pythons uv installs) then took the row for
an empty prompt and blanked it on Backspace, prompt included. Pipes keep the plain
``console.input`` path in the CLI.

**Approval menu.** :func:`ask_approval` draws a two-item menu like Claude Code's permission
prompt (``❯ 1. Yes`` / ``2. No``, a dim ``Esc to cancel`` under it) as a small non-full-screen
prompt_toolkit application that erases itself when done, and returns :data:`YES`, :data:`NO` or
:data:`CANCEL`. Only a key pressed on purpose answers it: keys typed before it appeared are
discarded, and it answers once the keyboard has been quiet for :data:`ARM_DELAY_S`, so neither an
Enter pressed while the code was on its way nor a sentence typed across its appearance approves
code nobody read.

**Colours.** Everything is named with the 16 ANSI colours, so the terminal's own palette
decides the shades. :func:`light_background` guesses once at start-up whether the background is
light (it asks the terminal with OSC 11 on POSIX, else reads ``COLORFGBG``, else assumes dark)
and :func:`code_theme` names the matching rich syntax theme; :data:`CONSOLE_THEME` drops the
black background rich gives inline code. Keys the user typed while the program started, which
the query reads along with the terminal's answer, go to the first ``you>`` prompt. When the query
gave up before the answer came, the first prompt waits for it a little longer
(:data:`LATE_ANSWER_S`) and drops it instead of taking it for typing.

prompt_toolkit shows the cursor with ``ESC[?12l ESC[?25h``, xterm's terminfo ``cnorm``, the
sequence vim and ncurses programs such as htop send too: a terminal that honours ``?12l`` stops
blinking the cursor after this program as it does after those.
"""

from __future__ import annotations

import contextlib
import os
import re
import select
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rich.theme import Theme

if TYPE_CHECKING:
    from collections.abc import Iterator

    from prompt_toolkit.application import Application
    from prompt_toolkit.input import Input
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.output import Output
    from rich.console import Console

YES = "yes"
NO = "no"
CANCEL = "cancel"
"""The three answers of :func:`ask_approval`."""

PROMPT_STYLE = "bold ansicyan"
SELECTED_STYLE = "bold ansicyan"
HINT_STYLE = "dim"
POINTER = "❯"
ASCII_POINTER = ">"
"""The pointer where ``❯`` may not render (a legacy Windows console, a non-UTF-8 stdout)."""
CANCEL_HINT = "Esc to cancel"
ESC_TIMEOUT_S = 0.05
"""How long a lone Esc waits for the rest of an escape sequence (arrow keys arrive as one); the
prompt_toolkit default of 0.5 s makes Esc feel stuck."""
ARM_DELAY_S = 0.5
"""The approval menu takes an answer only after the keyboard was quiet this long, counted from
its first draw; a key that comes sooner is dropped and starts the count again. Nobody reads code
that fast, so such a key was meant for something else: another press of an impatient Enter, or
the next question typed ahead while the model worked. Esc and Ctrl-C cancel at any time."""
SETTLE_S = 0.25
"""After 1, 2, y or n the menu stays up, the choice highlighted, until the keyboard has been
quiet this long (at most :data:`SETTLE_MAX_S`), and swallows what comes: the Enter of the old
``[y/N]`` habit would otherwise land in the next prompt. An Enter closes it at once."""
SETTLE_MAX_S = 1.0

LIGHT_CODE_THEME = "ansi_light"
DARK_CODE_THEME = "ansi_dark"
CONSOLE_THEME = Theme({"markdown.code": "bold cyan", "markdown.code_block": "cyan"})
"""rich's inline-code styles without their ``on black``, so code sits on the terminal's own
background (the syntax themes above set no background either)."""

BACKGROUND_QUERY = b"\x1b]11;?\x1b\\"
"""OSC 11: "what is your background colour?"."""
ATTRIBUTES_QUERY = b"\x1b[c"
"""DA1, sent after the OSC 11 query: nearly every terminal answers it, so when its answer
arrives without an OSC 11 answer before it, the terminal does not support the query."""
QUERY_TIMEOUT_S = 1.0
"""How long the start-up query waits for the DA1 answer. A local terminal answers within
milliseconds and the answer ends the wait, so this only runs out on a slow link (ssh to a far
host) or with a terminal that answers nothing."""
LATE_ANSWER_S = 2.0
"""How much longer an answer may take and still be dropped rather than typed: after the query
gave up, the first ``you>`` prompt waits for the DA1 answer until this long after that (the
preflight in between counts) before it draws. A terminal that answers nothing costs at most this
once; an answer slower still would be typed at the first prompt."""
DRAIN_LIMIT = 65536
"""Most bytes read when a late answer is dropped before the first prompt."""
_BACKGROUND_REPLY_RE = re.compile(
    r"\x1b\]11;rgba?:([0-9a-fA-F]{1,4})/([0-9a-fA-F]{1,4})/([0-9a-fA-F]{1,4})"
)
_OSC_REPLY_RE = re.compile("\x1b\\]11;[^\x07\x1b\x9c]*(?:\x07|\x1b\\\\|\x9c)")
"""Any answer to the OSC 11 query, ending in BEL or ST (7- or 8-bit)."""
_ATTRIBUTES_REPLY_RE = re.compile(r"\x1b\[\?[0-9;]*c")
DUMB_TERMS = {"dumb", "unknown"}
"""TERM values of terminals that do not interpret cursor movement (Emacs' shell mode, for
instance); rich treats them as non-interactive too."""
NO_QUERY_TERMS = {"linux"}
"""TERM values whose terminals print an OSC query instead of answering it (the Linux console)."""
LIGHT_BACKGROUNDS = {"7", "15"}
DARK_BACKGROUNDS = {"0", "1", "2", "3", "4", "5", "6", "8"}
"""``COLORFGBG`` background numbers (the 16-colour palette): white and bright white are light,
black, the dark colours and bright black are dark; the other bright colours tell nothing."""


def interactive() -> bool:
    """Whether stdin and stdout are both terminals that interpret cursor movement: the only
    case where the line reader and the menu are drawn with prompt_toolkit."""
    if os.environ.get("TERM", "").lower() in DUMB_TERMS:
        return False
    return isatty(sys.stdin) and isatty(sys.stdout)


def isatty(stream: Any) -> bool:
    """``stream.isatty()``; ``False`` for ``None`` or a stream that cannot tell."""
    try:
        return bool(stream is not None and stream.isatty())
    except (AttributeError, ValueError, OSError):
        return False


# -- the you> line -------------------------------------------------------------------------------


class LineReader:
    """Read REPL lines with prompt_toolkit; one instance per REPL session keeps the history.

    The prompt is drawn in bold ANSI cyan, typed text unstyled, and it keeps these colours when
    Ctrl-C or Ctrl-D end it. Ctrl-C raises ``KeyboardInterrupt`` and Ctrl-D on an empty line
    ``EOFError``, like ``input()``; so does a line holding only Ctrl-Z (prompt_toolkit types ^Z
    literally on Windows, where Ctrl-Z Enter means end of input). On POSIX Ctrl-Z suspends the
    process, as it does while a turn runs. The first line starts with the keys typed while the
    program started (see :func:`light_background`).
    """

    def __init__(self, prompt: str) -> None:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import InMemoryHistory

        self._session: PromptSession[str] = PromptSession(
            [(PROMPT_STYLE, prompt)],
            history=InMemoryHistory(),
            enable_suspend=True,
            key_bindings=_line_keys(),
            output=_output(),
        )
        self._first = True

    @classmethod
    def open(cls, prompt: str) -> LineReader | None:
        """A reader for this terminal, or ``None`` when prompt_toolkit cannot drive it (such as
        ``NoConsoleScreenBufferError`` in mintty without winpty on Windows)."""
        try:
            return cls(prompt)
        except Exception:
            return None

    def read(self) -> str:
        """One line, without its newline."""
        if self._first:
            self._first = False
            _give_back(self._session.app.input, _early_input.take())
        line = self._session.prompt()
        if line.strip() == "\x1a":
            raise EOFError
        return line


def _line_keys() -> KeyBindings:
    """Ctrl-C and Ctrl-D (on an empty line) end the prompt like PromptSession's own keys, minus
    their exit style, which repaints the line in a fixed grey (#888888) outside the terminal's
    palette. Ctrl-C while searching the history (Ctrl-R) still only ends the search."""
    from prompt_toolkit.application.current import get_app
    from prompt_toolkit.enums import DEFAULT_BUFFER
    from prompt_toolkit.filters import Condition, has_focus
    from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent

    keys = KeyBindings()
    typing = has_focus(DEFAULT_BUFFER)

    @keys.add("c-c", filter=typing)
    @keys.add("<sigint>")
    def _interrupt(event: KeyPressEvent) -> None:
        event.app.exit(exception=KeyboardInterrupt)

    @keys.add("c-d", filter=typing & Condition(lambda: not get_app().current_buffer.text))
    def _end(event: KeyPressEvent) -> None:
        event.app.exit(exception=EOFError)

    return keys


def _output() -> Output:
    """prompt_toolkit's output for this terminal, without cursor position requests (CPR).

    prompt_toolkit asks where the cursor is to learn how many rows are free below it, which the
    one-line prompt and the five-row menu do not need; a terminal that never answers made it
    print a warning into the transcript and hold every Enter for a second, and an answer that
    comes late would be read as typing.
    """
    from prompt_toolkit.application.current import get_app_session
    from prompt_toolkit.output.vt100 import Vt100_Output

    output = get_app_session().output
    if isinstance(output, Vt100_Output):
        output.enable_cpr = False
    return output


def _give_back(input_obj: Input, data: bytes) -> None:
    """Hand ``data``, keys read from the terminal, to the next prompt_toolkit application on
    ``input_obj`` as if they had just been typed."""
    if not data:
        return
    from prompt_toolkit.input.typeahead import store_typeahead
    from prompt_toolkit.input.vt100_parser import Vt100Parser
    from prompt_toolkit.key_binding import KeyPress

    keys: list[KeyPress] = []
    Vt100Parser(keys.append).feed_and_flush(data.decode("utf-8", "replace"))
    store_typeahead(input_obj, keys)


# -- the approval menu ---------------------------------------------------------------------------


def menu_pointer(console: Console) -> str:
    """``❯``, or ``>`` on a legacy Windows console or a stdout that is not UTF-8."""
    encoding = (console.encoding or "").lower().replace("-", "").replace("_", "")
    if console.legacy_windows or not encoding.startswith("utf"):
        return ASCII_POINTER
    return POINTER


class _MenuKeys:
    """When the approval menu's keys came, for its two timing rules: it takes an answer only
    after :data:`ARM_DELAY_S` without a key, and after 1, 2, y or n it closes once the keyboard
    has been quiet for :data:`SETTLE_S`."""

    def __init__(self) -> None:
        self.last = time.monotonic()
        self.armed = False
        self.chosen: str | None = None
        self.chosen_at = 0.0

    def start(self) -> None:
        """The menu is being drawn: the quiet time counts from now."""
        self.last = time.monotonic()
        self.armed = ARM_DELAY_S <= 0

    def pressed(self) -> None:
        """A key came in (called before its binding runs); after a quiet time it may answer."""
        now = time.monotonic()
        self.armed = self.armed or now - self.last >= ARM_DELAY_S
        self.last = now

    def may_answer(self) -> bool:
        """Whether the key being handled may move the pointer or answer."""
        return self.armed and self.chosen is None

    def choose(self, answer: str) -> None:
        self.chosen = answer
        self.chosen_at = time.monotonic()

    def settles_in(self) -> float:
        """Seconds until the menu closes on the chosen answer (none left: now)."""
        return min(self.last + SETTLE_S, self.chosen_at + SETTLE_MAX_S) - time.monotonic()


def approval_menu(question: str, *, pointer: str = POINTER) -> Application[str]:
    """The approval menu as a prompt_toolkit application that returns YES, NO or CANCEL.

    Up/Down (also k/j and Ctrl-P/Ctrl-N) move, Enter confirms the highlighted row, 1 or y
    choose Yes and 2 or n choose No at once; Esc and Ctrl-C cancel. "Yes" starts highlighted.
    Every key but Esc and Ctrl-C is dropped until the keyboard has been quiet for
    :data:`ARM_DELAY_S`; after 1, 2, y or n the menu lingers for :data:`SETTLE_S` to swallow a
    trailing Enter.
    """
    import asyncio

    from prompt_toolkit.application import Application
    from prompt_toolkit.formatted_text import StyleAndTextTuples
    from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
    from prompt_toolkit.layout import FormattedTextControl, Layout, Window

    choices = ((YES, "Yes"), (NO, "No"))
    selected = [0]
    timing = _MenuKeys()

    def text() -> StyleAndTextTuples:
        lines: StyleAndTextTuples = [("bold", f"{question}\n")]
        for index, (_, label) in enumerate(choices):
            if index == selected[0]:
                lines.append((SELECTED_STYLE, f"{pointer} {index + 1}. {label}\n"))
            else:
                lines.append(("", f"{' ' * len(pointer)} {index + 1}. {label}\n"))
        lines.append(("", "\n"))
        lines.append((HINT_STYLE, CANCEL_HINT))
        return lines

    keys = KeyBindings()

    @keys.add("up")
    @keys.add("k")
    @keys.add("c-p")
    def _up(event: KeyPressEvent) -> None:
        if timing.may_answer():
            selected[0] = max(selected[0] - 1, 0)

    @keys.add("down")
    @keys.add("j")
    @keys.add("c-n")
    def _down(event: KeyPressEvent) -> None:
        if timing.may_answer():
            selected[0] = min(selected[0] + 1, len(choices) - 1)

    @keys.add("enter")
    @keys.add("c-j")
    def _confirm(event: KeyPressEvent) -> None:
        if timing.chosen is not None:  # the Enter after 1, 2, y or n
            event.app.exit(result=timing.chosen)
        elif timing.may_answer():
            event.app.exit(result=choices[selected[0]][0])

    def choose(event: KeyPressEvent, answer: str) -> None:
        if not timing.may_answer():
            return
        timing.choose(answer)
        selected[0] = [value for value, _ in choices].index(answer)
        event.app.create_background_task(settle(event.app, answer))

    async def settle(app: Application[str], answer: str) -> None:
        while (wait := timing.settles_in()) > 0:
            await asyncio.sleep(wait)
        if not app.is_done:
            app.exit(result=answer)

    @keys.add("1")
    @keys.add("y")
    @keys.add("Y")
    def _yes(event: KeyPressEvent) -> None:
        choose(event, YES)

    @keys.add("2")
    @keys.add("n")
    @keys.add("N")
    def _no(event: KeyPressEvent) -> None:
        choose(event, NO)

    # eager: in case a binding that starts with Esc is ever active here (prompt_toolkit's emacs
    # bindings are, only while a buffer has focus); ESC_TIMEOUT_S is what makes a lone Esc quick
    @keys.add("escape", eager=True)
    @keys.add("c-c")
    @keys.add("<sigint>")
    def _cancel(event: KeyPressEvent) -> None:
        event.app.exit(result=CANCEL)

    window = Window(
        FormattedTextControl(text, show_cursor=False),
        dont_extend_height=True,
        always_hide_cursor=True,
    )
    app: Application[str] = Application(
        layout=Layout(window),
        key_bindings=keys,
        full_screen=False,
        erase_when_done=True,
        output=_output(),
    )
    app.ttimeoutlen = ESC_TIMEOUT_S
    app.on_reset += lambda _app: timing.start()
    # every key counts, also those only prompt_toolkit's own bindings handle (Backspace, a paste)
    app.key_processor.before_key_press += lambda _processor: timing.pressed()
    return app


def ask_approval(question: str, *, pointer: str = POINTER) -> str:
    """Show the approval menu under ``question`` and return YES, NO or CANCEL.

    Keys typed before the menu appeared are dropped first, and the menu drops the keys that
    follow too quickly (:data:`ARM_DELAY_S`). A closed terminal (``EOFError``) or a
    ``KeyboardInterrupt`` that reaches this far count as a cancel.
    """
    app = approval_menu(question, pointer=pointer)
    discard_typeahead(app.input)
    try:
        return app.run()
    except (EOFError, KeyboardInterrupt):
        return CANCEL


def discard_typeahead(input_obj: Input) -> None:
    """Drop keys typed ahead: those prompt_toolkit read but did not use (kept for the next
    application) and those still queued in the terminal."""
    from prompt_toolkit.input.typeahead import clear_typeahead

    clear_typeahead(input_obj)
    with contextlib.suppress(Exception):  # best effort: a pipe or a missing console has no queue
        if sys.platform == "win32":
            _flush_console_input()
        else:
            import termios

            termios.tcflush(input_obj.fileno(), termios.TCIFLUSH)


def _flush_console_input() -> None:
    """``FlushConsoleInputBuffer`` on the process's console input (Windows only)."""
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    kernel32.GetStdHandle.restype = ctypes.c_void_p
    kernel32.FlushConsoleInputBuffer.argtypes = [ctypes.c_void_p]
    kernel32.FlushConsoleInputBuffer(kernel32.GetStdHandle(-10))  # STD_INPUT_HANDLE


# -- light or dark -------------------------------------------------------------------------------


def code_theme(light: bool) -> str:
    """The rich syntax theme for code: ANSI colours on the terminal's own background."""
    return LIGHT_CODE_THEME if light else DARK_CODE_THEME


def light_background() -> bool:
    """Guess, once at start-up, whether the terminal's background is light.

    1. POSIX with stdin and stdout on a terminal (:func:`interactive`), except the Linux
       console: ask the terminal (:func:`query_background`). Keys the user typed meanwhile, and
       a note that the answer is still due, are kept for the first ``you>`` prompt.
    2. Otherwise, or when it did not answer: ``COLORFGBG`` (:func:`parse_colorfgbg`), which
       rxvt, Konsole and iTerm2 set.
    3. Otherwise dark.

    Nothing is written to a pipe or a Windows console; ``KeyboardInterrupt`` propagates.
    """
    term = os.environ.get("TERM", "").lower()
    if sys.platform != "win32" and interactive() and term not in NO_QUERY_TERMS:
        reply = query_background()
        _early_input.keep(reply)
        if reply.light is not None:
            return reply.light
    return parse_colorfgbg(os.environ.get("COLORFGBG")) is True


@dataclass(frozen=True)
class Reply:
    """What the terminal sent back for the start-up query."""

    light: bool | None
    """Whether it named a light background; ``None`` when it named none."""
    typed: bytes = b""
    """The other input read with the answer: keys the user typed meanwhile."""
    complete: bool = True
    """``False`` when the wait ended before the DA1 answer, which may still come in."""


def query_background(
    fd_in: int | None = None, fd_out: int | None = None, *, timeout: float = QUERY_TIMEOUT_S
) -> Reply:
    """Ask the terminal for its background colour (OSC 11) and say whether it is light.

    The OSC 11 query is followed by a DA1 query that nearly every terminal answers, and the
    replies are read with echo and line buffering off (:func:`_unbuffered`) until the DA1 reply
    or ``timeout``. Anything else read on the way, such as keys typed while the program started,
    comes back in :attr:`Reply.typed`. Nothing is written unless both ``fd_in`` and ``fd_out``
    are terminals. POSIX only (``termios``).
    """
    import termios

    received = b""
    due = False  # the queries went out and the DA1 answer has not come back
    try:
        fd_in = sys.stdin.fileno() if fd_in is None else fd_in
        fd_out = sys.stdout.fileno() if fd_out is None else fd_out
        if not os.isatty(fd_out):
            return Reply(None)
        with _unbuffered(fd_in):
            with contextlib.suppress(Exception):
                sys.stdout.flush()
            os.write(fd_out, BACKGROUND_QUERY + ATTRIBUTES_QUERY)
            due = True
            deadline = time.monotonic() + timeout
            while due:
                if _ATTRIBUTES_REPLY_RE.search(received.decode("latin-1")):
                    due = False
                    break
                left = deadline - time.monotonic()
                if left <= 0 or not select.select([fd_in], [], [], left)[0]:
                    break
                chunk = os.read(fd_in, 1024)
                if not chunk:  # the terminal went away: nothing more will come
                    due = False
                received += chunk
    except (termios.error, OSError, ValueError):  # no terminal, or a stream without a descriptor
        due = False
    return parse_reply(received, complete=not due)


@contextlib.contextmanager
def _unbuffered(fd_in: int) -> Iterator[None]:
    """Echo and line buffering off on the terminal ``fd_in`` inside the block, so answers and keys
    are read as they come; the mode is restored after it, also after Ctrl-C (ISIG stays on, so
    Ctrl-C still works). Raises ``termios.error`` when ``fd_in`` is no terminal."""
    import termios

    saved = termios.tcgetattr(fd_in)
    mode = termios.tcgetattr(fd_in)
    mode[3] &= ~(termios.ECHO | termios.ICANON)  # lflag
    mode[6][termios.VMIN] = 1
    mode[6][termios.VTIME] = 0
    termios.tcsetattr(fd_in, termios.TCSANOW, mode)
    try:
        yield
    finally:
        with contextlib.suppress(termios.error, OSError):
            termios.tcsetattr(fd_in, termios.TCSANOW, saved)


def parse_reply(data: bytes, *, complete: bool = True) -> Reply:
    """Split terminal input into the answers to the start-up query and the rest (typed keys)."""
    text = data.decode("latin-1")  # one character per byte, so the rest encodes back unchanged
    rest = _ATTRIBUTES_REPLY_RE.sub("", _OSC_REPLY_RE.sub("", text))
    return Reply(parse_background_reply(text), rest.encode("latin-1"), complete)


def drop_late_answer(fd_in: int | None = None, *, timeout: float = 0.0) -> bytes:
    """Read what the terminal has queued, waiting up to ``timeout`` for the DA1 answer to the
    start-up query that gave up before it came; drop the answers and return the rest: keys the
    user typed. POSIX only; best effort (``b""`` when there is no terminal to read)."""
    if sys.platform == "win32":
        return b""
    import termios

    received = b""
    try:
        fd_in = sys.stdin.fileno() if fd_in is None else fd_in
        with _unbuffered(fd_in):  # a line still being typed is read too
            deadline = time.monotonic() + timeout
            while len(received) < DRAIN_LIMIT:
                answered = _ATTRIBUTES_REPLY_RE.search(received.decode("latin-1"))
                wait = 0.0 if answered else max(0.0, deadline - time.monotonic())
                if not select.select([fd_in], [], [], wait)[0]:
                    break
                chunk = os.read(fd_in, 1024)
                if not chunk:
                    break
                received += chunk
    except (termios.error, OSError, ValueError):
        pass
    return parse_reply(received).typed


class _EarlyInput:
    """Terminal input the start-up query read that was not its answer: keys typed while the
    program started, which the first ``you>`` prompt gets back (:meth:`LineReader.read`).

    ``due_by``: the query gave up before the terminal answered. The answer can still come in,
    and prompt_toolkit would take it for typing, so the first prompt waits for it until then and
    drops it (:func:`drop_late_answer`) before it draws.
    """

    def __init__(self) -> None:
        self.typed = b""
        self.due_by: float | None = None

    def keep(self, reply: Reply) -> None:
        self.typed += reply.typed
        if not reply.complete:
            self.due_by = time.monotonic() + LATE_ANSWER_S

    def take(self) -> bytes:
        """The kept keys, plus those that come with a late answer; empties the store."""
        typed, due_by = self.typed, self.due_by
        self.typed, self.due_by = b"", None
        if due_by is None:
            return typed
        return typed + drop_late_answer(timeout=max(0.0, due_by - time.monotonic()))


_early_input = _EarlyInput()


def parse_background_reply(reply: str) -> bool | None:
    """Whether an OSC 11 reply (``ESC ] 11 ; rgb:RRRR/GGGG/BBBB`` ending in ST or BEL) names a
    light colour; ``None`` when ``reply`` holds no such reply.

    Channels have one to four hex digits. Light means a luma above one half: the relative
    luminance weights (0.2126, 0.7152, 0.0722) applied to the channels as reported, which
    splits the backgrounds where dark text reads better from those where light text does.
    """
    match = _BACKGROUND_REPLY_RE.search(reply)
    if match is None:
        return None
    red, green, blue = (int(part, 16) / (16 ** len(part) - 1) for part in match.groups())
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue > 0.5


def parse_colorfgbg(value: str | None) -> bool | None:
    """Whether ``COLORFGBG`` (``fg;bg`` or ``fg;default;bg``) names a light background: 7 or 15
    is light, 0-6 and 8 are dark; anything else is ``None`` (unknown)."""
    parts = (value or "").split(";")
    if len(parts) < 2:
        return None
    background = parts[-1].strip()
    if background in LIGHT_BACKGROUNDS:
        return True
    if background in DARK_BACKGROUNDS:
        return False
    return None


__all__ = [
    "CANCEL",
    "CONSOLE_THEME",
    "NO",
    "YES",
    "LineReader",
    "Reply",
    "approval_menu",
    "ask_approval",
    "code_theme",
    "discard_typeahead",
    "drop_late_answer",
    "interactive",
    "isatty",
    "light_background",
    "menu_pointer",
    "parse_background_reply",
    "parse_colorfgbg",
    "parse_reply",
    "query_background",
]
