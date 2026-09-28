"""Offline tests for tools/calculator.py: what it evaluates, what it refuses, and its limits.

Functionality is checked against Python's own ``eval`` of the same (trusted, fast) expression,
rendered by the calculator's formatter, so only the evaluation is compared. The security cases
run under an audit hook that records every event but ``compile`` (``ast.parse`` raises that one)
while the calculator works. Every resource case was first run in a separate process with a hard
timeout and a memory watchdog; here each one asserts an upper bound on its elapsed time, and the
slow ones run with lowered limits so the suite stays fast.
"""

from __future__ import annotations

import json
import math
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from mimoe_agent.middleware import reports_failure
from mimoe_agent.tools import calculator as calc
from mimoe_agent.tools.calculator import (
    MAX_RESULT_DIGITS,
    MAX_SECONDS,
    RUN_PYTHON_HINT,
    calculate,
    calculator,
)

_STATISTICS = (
    "correlation covariance fmean geometric_mean harmonic_mean linear_regression mean median "
    "median_grouped median_high median_low mode multimode pstdev pvariance quantiles stdev variance"
)
_NAMESPACE: dict[str, Any] = {
    **{name: getattr(math, name) for name in dir(math) if not name.startswith("_")},
    **{name: getattr(statistics, name) for name in _STATISTICS.split()},
    "pow": pow,  # the builtin, as in the calculator (math.pow stays math.pow)
    "math": math,
    "statistics": statistics,
}


def _python(expression: str) -> Any:
    """Python's own value for a trusted expression, shaped the way the calculator shows it."""
    value = eval(expression, dict(_NAMESPACE))  # noqa: S307 - trusted test input
    if isinstance(value, Iterator):
        value = list(value)
    if isinstance(value, tuple) and type(value) is not tuple:  # LinearRegression
        value = tuple(value)
    return value


# --- what it evaluates --------------------------------------------------------------------------

PYTHON_EXPRESSIONS = [
    # the user's question and the counting questions that used to need run_python
    "sum(range(10, 94))",
    "sum(x**2 for x in range(1, 11))",
    "sum(i for i in range(432, 43230) if i % 5 == 0)",
    "len([n for n in range(1000, 4501) if all(n % d for d in range(2, isqrt(n) + 1))])",
    "sum(1 for n in range(1000, 4501) if all(n % i != 0 for i in range(2, int(n**0.5) + 1)))",
    "sum(int(d) for d in str(2**100))",
    "sum(map(int, str(2**100)))",
    "max(x % 7 for x in range(100))",
    "sorted([5, 3, 9, 1])[:3]",
    "len('hello') + len([1, 2]) + len((1,)) + len({1, 2, 2}) + len(range(0, 100, 7))",
    # comparisons, logic, conditionals
    "1 < 2 < 3",
    "1 < 3 < 2",
    "3 > 2 == 2",
    "1 != 1.0",
    "not 0 and 5 or 7",
    "0 or 3",
    "1 if 2 > 1 else 0",
    "'b' in 'abc'",
    "2 not in [1, 2, 3]",
    # displays, comprehensions, indexing and slicing
    "[x for x in range(10) if x % 3 == 0]",
    "{x % 3 for x in range(10)}",
    "(1, 2) + (3,)",
    "[1, 2, 3][1:]",
    "(1, 2, 3)[-1]",
    "'hello'[::-1]",
    "range(10)[2:8:2]",
    "range(10**20)[-1]",
    "[(i, j) for i in range(3) for j in range(i)]",
    "[[j for j in range(i)] for i in range(4)]",
    "sum(a * b for a, b in zip([1, 2], [3, 4]))",
    "[i * c for i, c in enumerate('abc', 1)]",
    "[e for e in range(3)]",
    # math, by bare name and as math.<name>
    "math.sqrt(2)",
    "sqrt(16) + cbrt(27) + exp2(3)",
    "comb(10, 3) + perm(5, 2) + perm(4)",
    "gcd(12, 18, 30) + lcm(4, 6, 10)",
    "isqrt(99) + factorial(20)",
    "prod(range(1, 11))",
    "fsum([0.1] * 10)",
    "log(8, 2) + log10(1000) + log2(8) + log(e)",
    "hypot(3, 4) + dist((0, 0), (3, 4))",
    "sumprod([1, 2], [3, 4])",
    "fmod(7, 3) + copysign(1, -0.0) + trunc(-2.5) + degrees(pi)",
    "isclose(0.1 + 0.2, 0.3)",
    "frexp(8.0)",
    "modf(2.5)",
    "math.pow(2, 10)",
    "floor(2.7) + ceil(2.1)",
    "sin(pi / 2) + cos(0) + tan(0) + atan2(1, 1)",
    # statistics, by bare name and as statistics.<name>
    "mean([1, 2, 3, 4])",
    "median([3, 1, 4]) + median_low([1, 2, 3, 4]) + median_high([1, 2, 3, 4])",
    "stdev([2, 4, 4, 4, 5, 5, 7, 9])",
    "statistics.mode([1, 1, 2])",
    "multimode('aabbc')",
    "variance([1, 2, 3, 4]) + pstdev([1, 2, 3]) + pvariance([1, 2, 3])",
    "harmonic_mean([1, 2, 4]) + geometric_mean([1, 2, 4])",
    "quantiles(range(1, 101), n=4)",
    "fmean([1, 2, 3], weights=[3, 2, 1])",
    "correlation([1, 2, 3], [1, 2, 3.5]) + covariance([1, 2, 3], [1, 2, 3.5])",
    "statistics.linear_regression([1, 2, 3], [2, 4, 6])",
    "median_grouped([1, 2, 2, 3, 4])",
    # constants and conversions
    "pi + e + tau",
    "-inf",
    "bin(10) + hex(255) + oct(8)",
    "int('1010', 2) + int('ff', 16) + int(3.9) + int('  42 ')",
    "float('1.5') + abs(-3) + round(3.14159, 2) + round(2.5)",
    "divmod(17, 5)",
    "divmod(-7.5, 2)",
    "pow(2, 10) + pow(3, 4, 5) + pow(3, -1, 7)",
    "bool(0) or bool([0])",
    "str(42) + str(1.5) + str(True)",
    "list(range(5))",
    "tuple('ab')",
    "set([1, 1, 2]) | frozenset([3])",
    # iteration helpers
    "list(enumerate('ab'))",
    "list(zip([1, 2], 'ab', (True, False)))",
    "list(reversed([1, 2, 3]))",
    "list(reversed(range(4)))",
    "any([0, 1]) + all([1, 1, 0]) + any(range(1)) + all(range(1, 5))",
    "list(filter(bool, [0, 1, 2, '', 'a']))",
    "list(map(pow, [2, 3], [3, 2]))",
    "sorted(['bb', 'a', 'ccc'], key=len)",
    "sorted([3, -5, 1], key=abs, reverse=True)",
    "max([-5, 3], key=abs) + min([4, -1], key=abs)",
    "min([], default=0) + max((), default=7)",
    "max(3, -5, key=abs)",
    # ranges: the O(1) paths agree with Python
    "sum(range(1, 101)) + sum(range(10, 0, -3)) + sum(range(5, 5))",
    "len(range(10, 0, -3)) + len(range(3, 0))",
    "max(range(5, 0, -1)) + min(range(5, 0, -1)) + max(range(0, 100, 7))",
    "[7 in range(0, 100, 7), 7.0 in range(0, 100, 7), 7.5 in range(100), True in range(2)]",
    "sum(range(435, 43230, 5))",
    # operators on every value type
    "'ab' + 'cd'",
    "'ab' * 3 + 2 * 'c'",
    "[1, 2] + [3]",
    "[0] * 3",
    "{1, 2} & {2}",
    "{1, 2} - {2}",
    "{1, 2} <= {1, 2, 3}",
    "[1, 2] < [1, 3]",
    "'abc' < 'abd'",
    "5 << 2",
    "~5 + (20 >> 2) + (3 & 1) + (3 | 4)",
    "-7 // 2 + -7 % 3 + 2 ** -1",
    "True + True",
    "10 / 4",
    "2**100",
    "len(str(factorial(100)))",
    # big and repeated values are weighed before C hashes or compares them, and accepted
    "len(set([10**99999] * 1000))",
    "[10**99999] * 1000 == [10**99999] * 1000",
    "(10**99999 - 1) in [10**99999] * 1000",
    "len({b + k * (2**61 - 1) for b in [10**99999] for k in range(100)})",
    "len({(i, j) for i in range(100) for j in range(100)})",
    "sorted([(2, 'b'), (1, 'a'), (1, 'b')])",
    "max([(1, 2), (1, 3)]) + min((1, 2), (0, 5))",
]


@pytest.mark.parametrize("expression", PYTHON_EXPRESSIONS)
def test_the_calculator_agrees_with_python(expression: str) -> None:
    started = time.perf_counter()
    out = calculate(expression)
    elapsed = time.perf_counter() - started
    assert out == calc._format_result(_python(expression)), out
    assert elapsed < 1.0, f"{expression}: {elapsed:.3f} s"


def test_ranges_have_exact_constant_time_answers() -> None:
    """len, sum, min, max and membership of a range never walk it: Python's own sum would take
    hours here, and 1.5 in range(10**15) makes CPython scan the whole range."""
    big = range(10**15, 0, -7)
    started = time.perf_counter()
    assert calculate("sum(range(1, 10**12))") == str((10**12 - 1) * 10**12 // 2)
    assert calculate("sum(range(10**15, 0, -7))") == str(len(big) * (big[0] + big[-1]) // 2)
    assert calculate("len(range(10**20))") == str(10**20)
    assert calculate("max(range(10**12))") == str(10**12 - 1)
    assert calculate("min(range(10**15, 0, -7))") == str(big[-1])
    assert calculate("10**15 in range(0, 10**16, 5)") == "True"
    assert calculate("1.5 in range(10**15)") == "False"
    assert calculate("all(range(1, 10**12))") == "True"
    assert calculate("any(range(1))") == "False"
    assert calculate("any(range(10**18)) and any(range(0, -10**18, -1))") == "True"
    assert time.perf_counter() - started < 0.5


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("print(sum(range(10, 94)))", "4326"),
        ("```python\nprint(2 ** 10)\n```", "1024"),
        ("```\nsum(range(5))\n```", "10"),
        ("```2 + 2```", "4"),  # on one line: no language name to drop (2 before)
        ("```sum(range(5))```", "10"),  # (the range itself before)
        ("```2+2\n```", "4"),  # (an empty expression before)
        ("`sum(range(5))`", "10"),
        ("import math\nmath.comb(5, 2)", "10"),
        ("from statistics import mean\nmean([1, 2, 3])", "2"),
        ("sum(range(4));", "6"),
        ("= 2 + 2 =", "4"),
        ("3 ≤ 4 ≠ 5", "True"),
        ("2 × π", "6.28318530718"),
        ("−3 + 10 ÷ 4", "-0.5"),
        ("sum(range(10, 94))  # the sum of 10 to 93", "4326"),
    ],
)
def test_what_models_wrap_around_an_expression_is_tolerated(expression: str, expected: str) -> None:
    assert calculate(expression) == expected


@pytest.mark.parametrize(
    ("expression", "start"),
    [
        ("```" + " " * 3000 + "x", "ERROR: unknown name 'x'"),  # 7.9 s in a regular expression
        ("```" + " " * 5000 + "x", "ERROR: expression longer than 1,000 characters"),
        ("```\n" + " " * 10**6 + "1\n```", "ERROR: expression longer than 1,000 characters"),
    ],
    ids=["3000-spaces", "5000-spaces", "a-million-spaces"],  # an id goes into an environment
)
def test_unwrapping_takes_linear_time(expression: str, start: str) -> None:
    """The fence came off with a regular expression that backtracked cubically on spaces after
    the opening backticks, and the length limit applied only to what it left."""
    started = time.perf_counter()
    out = calculate(expression)
    assert time.perf_counter() - started < 0.2, out
    assert out.startswith(start), out


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("(x for x in range(3))", "[0, 1, 2]"),  # a bare generator shows its items
        ("map(abs, [-1, 2])", "[1, 2]"),
        ("(1,)", "(1,)"),
        ("[0.1 + 0.2, 1/3, 2.0]", "[0.3, 0.333333333333, 2]"),
        ("['a', 1.5, True, (2,), {3}]", "['a', 1.5, True, (2,), {3}]"),
        ("[nan, inf, -inf]", "[nan, inf, -inf]"),
        ("nan", "nan (undefined result)"),
        ("set()", "set()"),
        ("frozenset([1])", "frozenset({1})"),
        ("range(0)", "[]"),
        ("''", "''"),
        ("' '", "' '"),
        ("'ERROR: not really'", "'ERROR: not really'"),  # a string never reads as a failure
        ("[10**5000]", "[1e+5000 (about 5,001 digits)]"),
    ],
)
def test_results_are_shown_plainly(expression: str, expected: str) -> None:
    assert calculate(expression) == expected


@pytest.mark.parametrize("text", ["ERROR: x", "error: x", "exit_code: 1", " exit_code: 2 (killed)"])
def test_a_text_result_never_reads_as_a_failed_call(text: str) -> None:
    """GuardrailMiddleware marks a result that starts like a failed tool call as an error (the
    CLI then says it failed): a string value that starts so is shown quoted."""
    out = calculate(repr(text))
    assert out == repr(text) and not reports_failure(out)


def test_long_results_are_cut_with_a_count() -> None:
    listing = calculate("list(range(10**6))")
    assert listing.startswith("[0, 1, 2, 3, ") and len(listing) < 2_200
    assert listing.endswith(" shown)") and "(1,000,000 items, the first " in listing
    text = calculate("'ab' * 5000")
    assert text.startswith("abab") and text.endswith("... (10,000 characters)")
    assert calculate("bin(10**99999)").endswith("... (332,192 characters)")


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 3", "5"),
        ("10 - 4 * 2", "2"),
        ("(10 - 4) * 2", "12"),
        ("2**3**2", "512"),
        ("-2**2", "-4"),
        ("7 / 2", "3.5"),
        ("7 // 2", "3"),
        ("7 % 3", "1"),
        ("1836.6 * 0.15", "275.49"),
        ("0.1 + 0.2", "0.3"),
        ("1/3", "0.333333333333"),
        ("2**100", "1267650600228229401496703205376"),
        ("2×3", "6"),
        ("10 ÷ 4", "2.5"),
        ("2+2=", "4"),
        ("1e3", "1000"),
        ("1_000 + 1", "1001"),
    ],
)
def test_calculator_basic_and_precedence(expression: str, expected: str) -> None:
    assert calculate(expression) == expected


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("sqrt(16)", "4"),
        ("round(3.14159, 2)", "3.14"),
        ("round(2.5)", "2"),
        ("max(1, 5, 3)", "5"),
        ("min([4, 2, 8])", "2"),
        ("sum([1, 2, 3.5])", "6.5"),
        ("abs(-7)", "7"),
        ("floor(2.7)", "2"),
        ("ceil(2.1)", "3"),
        ("log10(1000)", "3"),
        ("log2(8)", "3"),
        ("log(e)", "1"),
        ("sin(pi/2)", "1"),
        ("cos(0)", "1"),
        ("tan(0)", "0"),
        ("2 * pi", "6.28318530718"),
        ("factorial(20)", "2432902008176640000"),
        ("gcd(12, 18)", "6"),
        ("pow(2, 10)", "1024"),
        ("math.sqrt(2)", "1.41421356237"),
    ],
)
def test_calculator_functions_and_constants(expression: str, expected: str) -> None:
    assert calculate(expression) == expected


def test_calculator_large_but_allowed_results_are_summarised() -> None:
    assert calculate("10**5000") == "1e+5000 (about 5,001 digits)"
    out = calculate("2**100000")
    assert out.startswith("9.99002e+30102") and "30,103 digits" in out
    assert calculate("2**332000").endswith("(about 99,942 digits)")  # just under the bound
    refused = calculate("10**100001")
    assert refused.startswith("ERROR:") and "100,000 digits" in refused


def test_calculator_caret_hint() -> None:
    out = calculate("2^3")
    assert out.startswith("ERROR:") and "**" in out and "^" in out
    assert calculate("10^10**10").startswith("ERROR: '^'")  # the hint wins over the bomb check


@pytest.mark.parametrize("expression", ["1/0", "5 % 0", "7 // 0", "1 / (2 - 2)"])
def test_calculator_division_by_zero(expression: str) -> None:
    assert calculate(expression) == "ERROR: division by zero."


@pytest.mark.parametrize(
    ("expression", "fragment"),
    [
        ("hello", "unknown name 'hello'"),
        ("what is two plus two", "not a valid Python expression"),
        ("None", "only numbers, strings and True/False are allowed"),
        ("12,000 + 5", "commas are not allowed"),
        ("12,500 + 5", "commas are not allowed"),
        ("", "empty expression"),
        ("   ", "empty expression"),
        ("x" * 1001, "longer than 1,000 characters"),
    ],
)
def test_calculator_non_numeric_input(expression: str, fragment: str) -> None:
    out = calculate(expression)
    assert out.startswith("ERROR:") and fragment in out


@pytest.mark.parametrize(
    ("expression", "fragment"),
    [
        ("sqrt(-1)", "math domain error"),
        ("exp(1000)", "too large"),
        ("float(10**400)", "too large"),
        ("(-8) ** (1/3)", "not a real number"),
        ("max()", "max expected at least 1 argument"),
        ("sum()", "sum() missing 1 required positional argument"),
        ("1e400", "inf"),
    ],
)
def test_calculator_domain_and_overflow_messages(expression: str, fragment: str) -> None:
    assert fragment in calculate(expression)


# --- what it refuses ----------------------------------------------------------------------------

_RECORDING: dict[str, Any] = {"thread": None, "events": []}
_HOOKED: list[bool] = []


def _audit(event: str, args: tuple[Any, ...]) -> None:
    if _RECORDING["thread"] == threading.get_ident() and event != "compile":
        _RECORDING["events"].append((event, args))


@contextmanager
def _audited() -> Iterator[list[tuple[str, tuple[Any, ...]]]]:
    """The audit events (open, import, exec, os.system, subprocess.Popen, ...) raised on this
    thread inside the block, with their arguments; ``compile`` is left out because
    ``ast.parse`` raises it. An audit hook cannot be removed, so one hook is installed once and
    switched on per block."""
    if not _HOOKED:
        sys.addaudithook(_audit)
        _HOOKED.append(True)
    events: list[tuple[str, tuple[Any, ...]]] = []
    _RECORDING.update(thread=threading.get_ident(), events=events)
    try:
        yield events
    finally:
        _RECORDING.update(thread=None, events=[])


ATTACKS = [
    "().__class__",
    "[].__class__.__base__.__subclasses__()",
    "(1).__class__",
    "math.__dict__",
    "math.__loader__",
    "math.__spec__",
    "math.sqrt.__self__",
    "sum.__self__",
    "statistics.sys",
    "statistics.random",
    "statistics.NormalDist",
    "statistics.NormalDist(0, 1).pdf(0)",
    "statistics.kde([1, 2], h=1)",
    "(lambda: 1)()",
    "lambda: 1",
    "__import__('os')",
    "__import__('os').system('echo hacked')",
    "__builtins__",
    "open('x')",
    "open('/etc/passwd').read()",
    "globals()",
    "vars()",
    "locals()",
    "dir()",
    "getattr(1, 'real')",
    "setattr(1, 'x', 2)",
    "type(1)",
    "object()",
    "'{0.__class__}'.format(1)",
    "f'{1:999999999}'",
    "f'{().__class__}'",
    "'%0999999999d' % 1",
    "format(1, '999999999')",
    "eval('1')",
    "exec('1')",
    "compile('1', '', 'eval')",
    "breakpoint()",
    "input()",
    "help()",
    "exit()",
    "quit()",
    "[x.__class__ for x in [1]]",
    "(x for x in [1]).gi_frame",
    "(x for x in [1]).gi_frame.f_back.f_globals",
    "[f for f in [1]][0].__self__",
    "map(eval, ['1'])",
    "sorted([1], key=eval)",
    "max([1], key=lambda x: x)",
    "[__class__ for __class__ in [1]]",
    "[_x for _x in [1]]",
    "'a'.join(['b'])",
    "str(1).zfill(10**9)",
    "x = 1",
    "(y := 1)",
    "import os",
    "import os\nos.system('echo hacked')",
    "from os import system\nsystem('echo hacked')",
    "import math as os\nos.sqrt(4)",
    "print(*[1, 2])",
    "min(**{'key': 1})",
    "{'a': 1}",
    "{k: 1 for k in 'ab'}",
    "b'abc'",
    "bytes(10**9)",
    "memoryview(b'a')",
    "iter([1])",
    "id(1)",
    "hash(1)",
    "repr(1)",
    "None",
    "...",
    "1j",
    "range(3).__iter__()",
    "[].append(1)",
    "min(x=1)",
]


@pytest.mark.parametrize("expression", ATTACKS)
def test_escapes_are_refused_without_side_effects(expression: str) -> None:
    with _audited() as events:
        out = calculate(expression)
    assert out.startswith("ERROR: "), out
    assert events == []


@pytest.mark.parametrize(
    ("expression", "what"),
    [
        ("().__class__", "attribute access and method calls such as .__class__ are not supported"),
        ("math.__dict__", "names that start with an underscore (math.__dict__)"),
        ("__import__('os')", "names that start with an underscore (__import__)"),
        ("statistics.NormalDist", "statistics.NormalDist is not available"),
        ("open('x')", "unknown function 'open'"),
        ("x = 1", "not statements such as assignments"),
        ("'%d' % 1", "string formatting with % is not supported"),
        ("f'{1}'", "does no string formatting"),
        ("(y := 1)", "has no variables"),
        ("print(*[1, 2])", "cannot unpack with *"),
        ("{'a': 1}", "has no dictionaries"),
        ("import os\nos.getcwd()", "not statements such as assignments, imports"),
        ("n * 2", "unknown name 'n'; the calculator has no variables"),
        ("x.real", "attribute access and method calls such as .real are not supported"),
        ("isprime(7)", "unknown function 'isprime'"),
        ("(lambda n: n)(2)", "cannot define functions"),
    ],
)
def test_refusals_say_what_and_point_at_run_python(expression: str, what: str) -> None:
    """With the bare "unsupported syntax: GeneratorExp" qwen3-4b resent the same expression and
    then guessed a number; naming run_python sends it there."""
    out = calculate(expression)
    assert out.startswith("ERROR: ") and what in out, out
    assert out.endswith(f"{RUN_PYTHON_HINT}.")


def test_the_tables_hold_only_plain_names_and_numbers() -> None:
    """Nothing reachable by name is private, a class or a function factory."""
    names = set(calc._FUNCTIONS) | set(calc._CONSTANTS)
    assert not [name for name in names if name.split(".")[-1].startswith("_")]
    assert {"eval", "exec", "compile", "open", "getattr", "type", "format", "vars"}.isdisjoint(
        names
    )
    for missing in ("NormalDist", "kde", "kde_random", "sys", "random", "Fraction", "Decimal"):
        assert f"statistics.{missing}" not in names
    assert all(type(value) is float for value in calc._CONSTANTS.values())
    assert all(key.split(".")[0] in ("math", "statistics") for key in names if "." in key)


def test_a_syntax_error_reads_no_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CPython reads the line of a syntax error back from the parse's file name. The default,
    '<unknown>', named a file in the current directory: it was read, and a FIFO or a link to
    /dev/zero there hung the call."""
    if sys.platform != "win32":  # Windows file names cannot contain < or >
        (tmp_path / "<unknown>").write_text("SECRET-1\nSECRET-2\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    for expression in ("import math\n1 +", "sum(range(10)", "1 +"):
        with _audited() as events:
            out = calculate(expression)
        assert out.startswith("ERROR: not a valid Python expression"), out
        assert [args for event, args in events if event == "open" and args[0]] == []


def test_parser_warnings_stay_off_stderr() -> None:
    """The parser warns about text it accepts (an invalid escape, a number glued to a keyword),
    and on stderr that lands in the middle of the terminal UI. It runs in a fresh interpreter
    that imports the calculator as the agent does: pytest captures warnings itself."""
    expressions = [r"len('\d')", r"'\400'", "1if 1 else 2", "import math\nlen('\\q')"]
    script = (
        "import json, sys\n"
        "from mimoe_agent.tools.calculator import calculate\n"
        "print(json.dumps([calculate(text) for text in json.loads(sys.stdin.read())]))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script],
        input=json.dumps(expressions),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.stderr == ""
    assert json.loads(done.stdout) == ["2", "Ā", "1", "2"]


@pytest.mark.parametrize(
    "expression",
    [
        "statistics.quantiles([1, 2, 3], method=(x for x in [1]))",
        "statistics.quantiles([1, 2, 3], method=map(abs, [1]))",
        "statistics.correlation([1, 2, 3], [1, 2, 4], method=(1, 2))",
        "statistics.quantiles([1, 2, 3], method=tuple([tuple(range(1000))] * 1000))",
    ],
)
def test_a_statistics_method_must_be_a_string(expression: str) -> None:
    """statistics shows an unknown method with repr(): a generator's internal name and address,
    or the whole text of a nested tuple (three levels of a thousand took gigabytes)."""
    out = calculate(expression)
    assert out.startswith("ERROR: method must be a string such as 'inclusive' or 'ranked'"), out


def test_long_error_messages_are_cut() -> None:
    """Python's own message can repeat a whole argument; the calculator's refusals stay whole,
    so they still end with the hint."""
    for expression, start in [
        ("float('x' * 10**6)", "ERROR: could not convert string to float: 'xxx"),
        ("statistics.quantiles([1, 2, 3], method='a' * 10**6)", "ERROR: Unknown method: 'aaa"),
    ]:
        out = calculate(expression)
        assert out.startswith(start) and out.endswith(" characters).") and len(out) < 600, out
    assert calculate("x" * 900).endswith(f"{RUN_PYTHON_HINT}.")


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("sum(1, 2, 3)", "sum() takes from 1 to 2 positional arguments but 3 were given"),
        ("len(1, 2)", "len() takes 1 positional argument but 2 were given"),
        ("bin(1, 2)", "bin() takes 1 positional argument but 2 were given"),
        ("sorted([1], 2, key=abs)", "sorted() takes 1 positional argument but 2 were given"),
        ("pow(2, 3, base=5)", "pow() got multiple values for argument 'base'"),
        # a function passed as a value
        (
            "list(map(sum, [1], [2], [3]))",
            "sum() takes from 1 to 2 positional arguments but 3 were given",
        ),
        ("list(map(len, [1], [2]))", "len() takes 1 positional argument but 2 were given"),
        ("min([1], key=pow)", "pow() missing 1 required positional argument: 'exp'"),
        (
            "list(filter(filter, [1]))",
            "filter() missing 1 required positional argument: 'iterable'",
        ),
        ("list(map(math.comb, [5]))", "math.comb() missing 1 required positional argument: 'k'"),
    ],
)
def test_argument_errors_name_the_function_that_was_called(expression: str, message: str) -> None:
    """Not the implementation (``_sum``), and counting only the arguments the expression passed
    (every implementation takes the runtime first)."""
    assert calculate(expression) == f"ERROR: {message}."


# --- limits -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "limit"),
    [
        ("9**9**9", "100,000 digits"),
        ("2**10**6", "100,000 digits"),
        ("2**10**7", "100,000 digits"),
        ("(10**500)**10000", "100,000 digits"),
        ("pow(2, 10**8)", "100,000 digits"),
        ("1 << 10**7", "100,000 digits"),
        ("10**50000 * 10**50001", "100,000 digits"),
        ("factorial(10**6)", "100,000 digits"),
        ("comb(10**7, 5*10**6)", "100,000 digits"),
        ("perm(10**7, 10**6)", "100,000 digits"),
        ("pow(3, 10**99999, 10**99999 + 1)", "would take too long"),
        ("pow(3, 10**100000, 10**100000 + 1)", "would take too long"),
        # the inverse a negative exponent takes first: 8.9 s in one C call here before
        ("pow(3**209000, -1, 2**332003 - 1)", "negative 1-bit exponent and a 332,003-bit modulus"),
        ("pow(3**209000 + 2, -1, 2**67000 - 1)", "would take too long"),  # 0.4 s just below
        ("prod(range(1, 10**6))", "100,000 digits"),
        ("[0]*10**9", "MB of memory"),
        ("[1]*10**8", "MB of memory"),
        ("'a'*10**9", "1,000,000 characters"),
        ("'ab' + 'c' * 999999", "1,000,000 characters"),
        ("list(range(10**9))", "MB of memory"),
        ("sorted(range(10**8))", "MB of memory"),
        ("[[0]*10**4 for i in range(10**4)]", "MB of memory"),
        ("sum([[1]]*10**5, [])", "sum() adds numbers"),
        ("str(10**100000)", "4,300 digits"),
        ("int('1' * 5000)", "4,300 digits"),
        ("round(1, -10**9)", "100 digits of precision"),
        ("mean(range(10**6))", "at most 100,000 values"),
        ("quantiles([1, 2], n=10**9)", "n up to 10,000"),
        ("len(set(i * (2**61 - 1) for i in range(20000)))", "share one hash"),
        ("len({(i * (2**61 - 1),) for i in range(20000)})", "share one hash"),
        ("len(frozenset(i * (2**61 - 1) for i in range(20000)))", "share one hash"),
        ("mode([i * (2**61 - 1) for i in range(20000)])", "share one hash"),
        (
            "len({k * (2**61 - 1) for k in range(1200)}"
            " | {k * (2**61 - 1) for k in range(1200, 2400)})",
            "share one hash",
        ),
        # repeats are no new collisions, but each is compared with the values of its hash
        ("len(set([k * (2**61 - 1) for k in range(999)] * 100))", "3,000,000 steps"),
        ("variance([1, 2, 3] * 30000, 10**99999)", "numbers up to about 1e308"),
        ("-" * 900 + "1", "nested more than 100 levels deep"),
        ("(" * 300 + "1" + ")" * 300, "too many nested parentheses"),
    ],
)
def test_bombs_are_refused_before_they_run(expression: str, limit: str) -> None:
    started = time.perf_counter()
    out = calculate(expression)
    assert time.perf_counter() - started < 1.5, out  # 0.25 s at most on the development machine
    assert out.startswith("ERROR: ") and limit in out, out


@pytest.mark.parametrize(
    "expression",
    [
        "gcd(2**332100 - 1, 3**209500 - 1)",  # Lehmer gcd is quadratic in the operand size
        "10**99999 % 10**50000",  # schoolbook division, quadratic in quotient x divisor
        "10**99999 // 10**50000",
        "10**99999 * 10**99999",
    ],
)
def test_calculator_quadratic_big_int_paths_stay_fast(expression: str) -> None:
    """With a 10**6-digit bound these took 10-20 s each; the bound is what keeps them cheap."""
    started = time.perf_counter()
    out = calculate(expression)
    assert time.perf_counter() - started < 2.0, out
    assert not out.startswith("ERROR:") or "digits" in out


_DEEP = "tuple([tuple([tuple([7] * 1000)] * 1000)] * 1000)"
"""A billion sevens for C to hash or compare, built in a few thousand steps and 25 KB."""


@pytest.mark.parametrize(
    "expression",
    [
        # one 100,000-digit integer, repeated: C hashed, compared or scanned it in one call
        "len(set([10**99999] * 100000))",  # 4.2 s before
        "len(frozenset([10**99999] * 200000))",
        "len({0} | set([10**99999] * 200000))",
        "([10**99999] * 700000) == ([10**99999] * 700000)",  # True after 4.2 s
        "([10**99999] * 400000) < ([10**99999] * 400000)",
        "(10**99999 - 1) in ([10**99999] * 800000)",  # False after 3.4 s
        "len(sorted([10**99999, 10**99999] * 200000))",  # 2.4 s
        # a set too small for the collision count, of big values sharing one hash
        "len({b + k * (2**61 - 1) for b in [10**99999] for k in range(1000)})",  # 3.8 s
    ],
)
def test_repeated_values_are_paid_for_by_weight(expression: str) -> None:
    """``*`` repeats a reference, cheap to build, but C hashes and compares every copy; the
    calculator weighs the value first and refuses before C starts."""
    started = time.perf_counter()
    out = calculate(expression)
    assert time.perf_counter() - started < 1.5, out  # 0.2 s at most on the development machine
    assert out.startswith("ERROR: the calculation needs more than 3,000,000 steps"), out


@pytest.mark.parametrize(
    "expression",
    [
        f"len({{{_DEEP}}})",  # 2.1 s in C before; one more level never returned
        f"{_DEEP} == {_DEEP}",  # True after 0.6 s
        f"{_DEEP} in [{_DEEP}]",
        f"{_DEEP} in {{1}}",
        f"len(max({_DEEP}, {_DEEP}))",
        f"len(min([{_DEEP}, {_DEEP}], key=tuple))",
        f"len(sorted([{_DEEP}, {_DEEP}], key=tuple))",
    ],
)
def test_nested_repetition_is_refused_while_it_is_weighed(
    expression: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each level of ``tuple([...] * 1000)`` multiplies C's work by a thousand; weighing walks
    the value and runs out of steps (0.5 s with the real budget, lowered here)."""
    monkeypatch.setattr(calc, "MAX_STEPS", 300_000)
    started = time.perf_counter()
    out = calculate(expression)
    assert time.perf_counter() - started < 0.5, out
    assert out.startswith("ERROR: the calculation needs more than 300,000 steps"), out


@pytest.mark.parametrize(
    "expression",
    [
        "sum(1 for i in range(10**9))",
        "max(x for x in range(10**12))",
        "sum(1 for i in range(10**6) for j in range(10**6))",
        "sum(x * y for x in range(10**6) for y in range(10**6))",
        "gcd(range(10**9))",
        "sorted(range(10**6))",
        "len([n for n in range(2, 10**6) if all(n % d for d in range(2, isqrt(n) + 1))])",
    ],
)
def test_long_loops_stop_at_the_step_budget(
    expression: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(calc, "MAX_STEPS", 200_000)
    started = time.perf_counter()
    out = calculate(expression)
    assert time.perf_counter() - started < 1.5, out
    assert out.startswith("ERROR: the calculation needs more than 200,000 steps"), out
    assert "sum(range(1, 10**9))" in out and "run_python" in out


@pytest.mark.parametrize(
    "expression",
    [
        "sum(i**i for i in range(10**4))",
        "lcm(range(1, 10**6))",
        "[10**99999 + i for i in range(10**6)]",
        "sum(10**50000 * 10**49999 % 7 for i in range(10**6))",
        "sum(len(str(10**4000)) for i in range(10**6))",
        "sum(gcd(2**332100 - 1, 3**209500 - 1) for i in range(100))",
        "sum([10**99990] * 3000000)",
    ],
)
def test_slow_work_stops_at_the_deadline(expression: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(calc, "MAX_SECONDS", 0.3)
    started = time.perf_counter()
    out = calculate(expression)
    assert time.perf_counter() - started < 2.0, out  # the deadline plus one operation
    assert out.startswith("ERROR: the calculation took longer than 0.3 s"), out


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("sum(1 for i in range(10**9))", "ERROR: the calculation needs more than 3,000,000 steps"),
        ("len([i for i in range(1400000)])", "1400000"),
        ("len(set([10**99999] * 8000))", "1"),  # 700 MB of hashing, just in budget
        ("comb(332000, 166000)", "1.25873e+99939 (about 99,940 digits)"),  # the slowest operation
    ],
)
def test_the_real_budgets_hold(
    expression: str, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The configured step, byte and digit budgets, not lowered ones: the worst accepted and
    refused cases, 0.35-0.9 s each on the development machine and up to 1.4 s on the Windows VM.
    The deadline is out of the way, so a slow or busy machine cannot turn one into a timeout."""
    monkeypatch.setattr(calc, "MAX_SECONDS", 60.0)
    started = time.perf_counter()
    out = calculate(expression)
    assert time.perf_counter() - started < 10.0, out
    assert out.startswith(expected), out


def test_the_real_deadline_holds(monkeypatch: pytest.MonkeyPatch) -> None:
    """The configured deadline, with the step budget out of the way: an endless loop stops just
    after it."""
    monkeypatch.setattr(calc, "MAX_STEPS", 10**15)
    started = time.perf_counter()
    out = calculate("sum(1 for i in range(10**12))")
    elapsed = time.perf_counter() - started
    assert out.startswith(f"ERROR: the calculation took longer than {MAX_SECONDS:g} s"), out
    assert MAX_SECONDS - 0.1 < elapsed < MAX_SECONDS + 1.5


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("stdev([2.0**k for k in range(-1000, 1000)] * 50)", "1.38228070197e+299"),
        ("stdev(x * 1e-300 for x in range(10**5))", "2.88676577967e-296"),
        ("variance([1e308, 5e-324] * 50000)", "ERROR: the result is too large"),
        ("median(x * 1e-300 for x in range(10**5))", "4.99995e-296"),
    ],
)
def test_statistics_on_adversarial_floats_stay_fast(expression: str, expected: str) -> None:
    """They compute with exact fractions; 100,000 floats with exponents from -1074 to 1023 take
    0.15 s at most on the development machine."""
    started = time.perf_counter()
    out = calculate(expression)
    assert time.perf_counter() - started < 1.5, out
    assert out.startswith(expected), out


def test_sets_with_ordinary_values_are_unaffected_by_the_hash_check() -> None:
    """Python hashes an int to its value modulo 2**61 - 1; only many different values that share
    a hash are refused (20,000 of them took 2 s in set() and 4.6 s in mode())."""
    started = time.perf_counter()
    assert calculate("len(set(range(10**5)))") == "100000"
    assert calculate("len(set([0] * 10**5))") == "1"
    assert calculate("len({i * (2**61 - 1) for i in range(2000)})") == "2000"
    assert calculate("mode([1, 1, 2] * 30000)") == "1"
    assert time.perf_counter() - started < 1.5


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("len({n % 5 - 2 for n in range(10000)})", "5"),
        ("statistics.mode(n % 5 - 2 for n in range(10000))", "-2"),
        ("statistics.multimode([-1, -2, -2] * 1000)", "[-2]"),
        ("len(set([-1, -2] * 1500))", "2"),
        ("len({(-1)**n * (n % 3) for n in range(10000)})", "5"),
        ("len(set([1.0, 2.0**61] * 1500))", "2"),
        ("len(set(range(-3, 3)) | set([-1, -2] * 1500))", "6"),
    ],
)
def test_repeats_of_values_that_share_a_hash_are_no_collisions(
    expression: str, expected: str
) -> None:
    """hash(-1) == hash(-2) in CPython (and hash(1.0) == hash(2.0**61)): two values, repeated,
    were counted as thousands of different values sharing one hash, and refused."""
    started = time.perf_counter()
    assert calculate(expression) == expected
    assert time.perf_counter() - started < 0.5


def test_refused_work_leaves_memory_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(calc, "MAX_SECONDS", 60.0)  # the byte budget stops them, not the clock
    process = psutil.Process()
    before = process.memory_info().rss
    for expression in [
        "[[0]*10**4 for i in range(10**4)]",  # 800 MB without the byte budget
        "[x + i for x in [10**99999] for i in range(3000)]",  # 125 MB
        # new numbers C put in a tuple or range item: charged as the item's own 60 bytes, each
        # of these held 590 MB when it returned and more before the deadline
        "len(list(enumerate(range(12000), 10**99999)))",
        "len([range(x + i) for x in [10**99999] for i in range(6000)])",
        "len([divmod(x + i, 1) for x in [10**99999] for i in range(12000)])",
        "len([print(x + i, 0) for x in [10**99999] for i in range(6000)])",
        "len(list(enumerate(divmod(x + i, 1) for x in [10**99999] for i in range(12000))))",
        "len([quantiles([1, 2], n=10000) for i in range(3000)])",  # new floats: 520 MB before
    ]:
        assert "MB of memory" in calculate(expression), expression
    assert process.memory_info().rss - before < 400 * 2**20


@pytest.mark.parametrize(
    "expression",
    [
        "[map(abs, [1])]",
        "len([(y for y in range(x + i)) for x in [10**99999] for i in range(10**5)])",
        "len([enumerate([1], 10**99999 + i) for i in range(10**5)])",
        "len(list(map(reversed, [[1], [2]])))",
        "sorted([[1], [2]], key=reversed)",
        "{zip([1], [2])}",
        "statistics.correlation([1, 2, 3], [1, 2, 4], method=(1, (x for x in [1])))",
    ],
)
def test_a_list_tuple_or_set_holds_no_generator(expression: str) -> None:
    """What a generator holds (a range with a huge bound, enumerate's counter, a source per
    argument of zip) is out of sight, so a list of them could fill the memory unseen; nothing a
    calculation needs keeps one."""
    out = calculate(expression)
    assert out.startswith("ERROR: a list, tuple or set cannot hold a generator"), out
    assert "list(...)" in out


def test_calls_do_not_share_a_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(calc, "MAX_SECONDS", 60.0)  # the steps run out, however slow the machine
    results: dict[str, str] = {}

    def run(expression: str) -> None:
        results[expression] = calculate(expression)

    expressions = ["sum(1 for i in range(10**9))", "sum(range(10, 94))", "sqrt(2) * sqrt(2)"]
    threads = [threading.Thread(target=run, args=(expression,)) for expression in expressions]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert results["sum(1 for i in range(10**9))"].startswith("ERROR: the calculation needs")
    assert results["sum(range(10, 94))"] == "4326"
    assert results["sqrt(2) * sqrt(2)"] == "2"


# --- contract -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expression",
    [None, 42, 3.5, "", "\n", "#", "()", "[]", "print()", "sum()", "(", "1 +", "\x00", "𝟙 + 1"],
)
def test_never_raises_or_returns_empty(expression: Any) -> None:
    out = calculate(expression)
    assert isinstance(out, str) and out.strip()


def test_the_tool_contract() -> None:
    """A 4B model reads the description on every call; longer ones measurably confused it."""
    assert calculator.name == "calculator" and list(calculator.args) == ["expression"]
    description = " ".join(calculator.description.split())  # the docstring wraps lines
    assert len(description) < 400
    assert "sum(range(10, 94))" in description and " for n in range(" in description
    assert "use ** for powers" in description
    assert "cannot use variables, statements, imports or files: use run_python" in description
    assert calculator.invoke({"expression": "sum(range(10, 94))"}) == "4326"
    assert MAX_RESULT_DIGITS == 100_000
