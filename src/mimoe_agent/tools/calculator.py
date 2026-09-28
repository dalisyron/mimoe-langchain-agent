"""The ``calculator`` tool: Python math expressions run by a small interpreter with hard limits.

The model writes what it would write in Python (``sum(range(10, 94))``,
``len([n for n in range(2, 100) if all(n % d for d in range(2, isqrt(n) + 1))])``,
``statistics.median([3, 1, 4])``) and gets the value back without an approval step, because
nothing the calculator can express has a side effect.

How it stays safe:

* The input is only parsed (``ast.parse``); it is never compiled to bytecode or passed to
  ``eval`` or ``exec``, and no function that could do either is reachable. ``_Compiler`` checks
  every node of the tree against an explicit whitelist *before anything runs* and turns the tree
  into closures, so a refused part refuses the whole expression.
* There is no attribute access. ``math.<name>`` and ``statistics.<name>`` are looked up in fixed
  tables of pure functions and constants, and a name that starts with ``_`` is refused
  everywhere (a bare ``_`` may name a loop variable), so nothing can climb from a value to its
  class, module, frame or builtins (the escape route of other "safe eval" libraries). A name is
  a comprehension loop variable, a constant (``pi``, ``e``, ``tau``, ``inf``, ``nan``) or a
  function from the table; a callee is always a table function, never a computed value.
* No ``lambda``, ``:=``, ``*``/``**`` unpacking, dictionaries, statements or string formatting
  (``%`` on a string, ``format``, f-strings: format specs are a memory bomb and format strings
  reach attributes). The only values are ``bool``, ``int``, ``float``, ``str``, ``list``,
  ``tuple``, ``range``, ``set``, ``frozenset`` and the interpreter's own generators; a table
  function can be named as a value only where one is expected (``map(int, ...)``, ``key=len``).
* Every expensive step is bounded before it runs: integers above ``MAX_RESULT_DIGITS`` digits are
  refused by estimate for ``**``, ``<<``, ``*``, ``pow``, ``factorial``, ``comb`` and ``perm``
  and step by step for ``lcm`` and ``prod``; modular ``pow`` is bounded by its cost, a negative
  exponent's inverse included; sequences and strings by their size; hashing, comparing and
  sorting by the weight of the values, the bytes C reads, which counts an item repeated by ``*``
  or by nesting once per reference and so can be far more than the memory they take; sets also
  by how many values share a hash; ``statistics`` by the number and size of its data points; and
  each call has a budget of ``MAX_STEPS`` steps (operations, loop iterations and weight),
  ``MAX_BYTES`` of materialised values (an item of a list, tuple or set counts with the numbers
  a tuple or range item holds, and no generator is kept in one) and ``MAX_SECONDS`` of wall
  time.
  Ranges get O(1) ``len``, ``sum``, ``min``, ``max`` and membership, so
  ``sum(range(1, 10**12))`` is exact and instant.

What remains: a whitelisted C function could have a bug of its own, and one C-level operation
cannot be interrupted, so the bounds keep every single one under about a second (the slowest,
``comb(332000, 166000)``, takes 0.9 s; the resource tests measure the worst cases).
"""

from __future__ import annotations

import ast
import itertools
import math
import operator
import re
import statistics
import sys
import time
import warnings
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from types import GeneratorType
from typing import Any, NoReturn

from langchain_core.tools import tool

__all__ = [
    "MAX_BYTES",
    "MAX_RESULT_DIGITS",
    "MAX_SECONDS",
    "MAX_STEPS",
    "RUN_PYTHON_HINT",
    "calculate",
    "calculator",
]

MAX_EXPRESSION_CHARS = 1_000
MAX_DEPTH = 100
"""Deepest nesting of the syntax tree; keeps the compiler and the closures far from the
recursion limit."""
MAX_RESULT_DIGITS = 100_000
"""Largest integer, in decimal digits. At 10**6 digits ``gcd`` and ``%`` on maximal operands
stalled for 10-20 s; at this bound the slowest big-integer operation takes a fraction of a
second."""
MAX_ROUND_DIGITS = 100
MAX_STEPS = 3_000_000
"""Steps per call: loop iterations, calls, arithmetic and the items a function walks (about half
a second of simple work)."""
MAX_BYTES = 100 * 2**20
"""Bytes of lists, tuples, sets and strings the interpreter may build in one call."""
MAX_SECONDS = 2.0
"""Wall-clock limit per call, checked between operations."""
MAX_STRING_CHARS = 1_000_000
MAX_STATS_ITEMS = 100_000
"""Data points per ``statistics`` call; they compute with exact fractions internally, which
stays under 0.2 s at this size even for adversarial floats."""
MAX_STATS_INT = 2**1024
"""Largest integer (in absolute value) a ``statistics`` function accepts: the float range."""
MAX_POW_MOD_COST = 2**37
"""``exponent bits * modulus bits**2`` bound for ``pow(b, e, m)``: 0.4-0.6 s at the bound."""
POW_INVERSE_BITS = 30
"""What the modular inverse ``pow(b, -e, m)`` computes first costs, in exponent bits of the bound
above: CPython's extended Euclid is quadratic in the modulus with a large constant (0.8 s for a
100,000-bit modulus, 9 s at the digit bound, in one C call no deadline reaches)."""
MAX_STR_INT_DIGITS = 4_300
"""Longest integer ``str()`` and ``int()`` convert; CPython's own default limit for that
quadratic conversion."""
MAX_HASH_COLLISIONS = 1_000
"""How many different values may share one hash among the items of a set, or of ``mode``, once
there are more than twice that many items (each comparison with a value of the same hash is paid
for as a step, heavy values by their weight). CPython hashes an int to its value modulo
2**61 - 1, so all multiples of 2**61 - 1 collide, and a set of 20,000 of them takes seconds
inside one C call, where no deadline reaches."""
BYTES_PER_STEP = 256
"""Bytes of a value that hashing or comparing it in C may read per step. Hashing a big integer,
the slowest, reads about 2 GB/s, so a step of it takes about 120 ns, no more than a step of the
interpreter."""
RESULT_CHARS = 2_000
"""Longest rendering of a list, tuple, set or string result."""
DETAIL_CHARS = 500
"""Longest part of an error message taken from Python (``float()`` repeats its whole
argument)."""

_MAX_RESULT_BITS = int(MAX_RESULT_DIGITS * math.log2(10)) + 1
_MAX_STR_INT_BITS = int(MAX_STR_INT_DIGITS * math.log2(10)) + 1
_TOO_LARGE = f"the result would have more than {MAX_RESULT_DIGITS:,} digits"
_TOO_LONG_TEXT = f"the text would be longer than {MAX_STRING_CHARS:,} characters"
_STATS_RANGE = "statistics functions take numbers up to about 1e308"
_UNICODE_OPERATORS = str.maketrans(
    {"×": "*", "÷": "/", "−": "-", "≤": "<=", "≥": ">=", "≠": "!=", "π": "pi"}
)
_CLOSED_FORMS = (
    "sum, len, min and max of a range() are instant, e.g. sum(range(1, 10**9)); otherwise use a "
    "closed form such as n * (n + 1) // 2, or run_python"
)

RUN_PYTHON_HINT = (
    "For anything beyond one expression (variables, statements, imports, files), call run_python "
    "with code that prints the result"
)
"""Appended to every rejection that means "this is a program, not an expression". A small model
that only reads "unsupported syntax" tends to resend the same expression and then guess a
number; naming the tool that can do it sends it to run_python instead (measured on
qwen3-4b-instruct-2507)."""


class _OutOfScope(ValueError):
    """Syntax or a name the calculator does not evaluate; the message ends with the hint."""

    def __init__(self, what: str) -> None:
        super().__init__(f"{what}. {RUN_PYTHON_HINT}")


class _OverBudget(ValueError):
    """A resource limit was hit; the message names the limit."""


def _out_of_scope(what: str) -> NoReturn:
    raise _OutOfScope(what)


class _Budget:
    """Steps, bytes and wall time spent by one ``calculate`` call.

    ``tick`` runs at every loop iteration, call, arithmetic operator and subscript, and with a
    cost for work done in C (a scan, a sort, a copy, hashing or comparing a container);
    ``charge`` runs before anything that materialises a list, tuple, set or string. Comparing
    numbers and unary operators take no step of their own: they are cheap, and a loop around them
    ticks anyway (comparing strings or containers pays by their size). The limits are read when
    the call starts, so tests can lower them with ``monkeypatch``.
    """

    __slots__ = ("bytes", "deadline", "max_bytes", "max_steps", "seconds", "steps")

    def __init__(self) -> None:
        self.steps = 0
        self.bytes = 0
        self.max_steps = MAX_STEPS
        self.max_bytes = MAX_BYTES
        self.seconds = MAX_SECONDS
        self.deadline = time.monotonic() + MAX_SECONDS

    def tick(self, cost: int = 1) -> None:
        """Count ``cost`` steps and check the step budget and the deadline."""
        self.steps += cost
        if self.steps > self.max_steps:
            raise _OverBudget(
                f"the calculation needs more than {self.max_steps:,} steps (the calculator's "
                f"limit); {_CLOSED_FORMS}"
            )
        if time.monotonic() > self.deadline:
            raise _OverBudget(
                f"the calculation took longer than {self.seconds:g} s (the calculator's limit); "
                f"{_CLOSED_FORMS}"
            )

    def charge(self, size: int) -> None:
        """Count ``size`` bytes about to be materialised; refuse above the byte budget."""
        self.bytes += size
        if self.bytes > self.max_bytes:
            raise _OverBudget(
                f"the calculation needs more than {self.max_bytes // 2**20} MB of memory (the "
                f"calculator's limit); {_CLOSED_FORMS}"
            )


# --- values -------------------------------------------------------------------------------------

_NUMBERS = frozenset({int, float, bool})
_INTEGERS = frozenset({int, bool})
_SEQUENCES = frozenset({str, list, tuple})
_SETS = frozenset({set, frozenset})
_ITERABLES = frozenset({str, list, tuple, range, set, frozenset, GeneratorType})
_SIZED = frozenset({str, list, tuple, set, frozenset})
_CONTAINERS = frozenset({list, tuple, set, frozenset})
_WEIGHED = _CONTAINERS | {range}
"""Values whose weight (``_Runtime.weight``) is more than their own size."""
_HOLDERS = frozenset({tuple, range, GeneratorType})
"""Values that hold more memory than their own size (``_Runtime.held``)."""
_PASS_THROUGH = frozenset({float, bool, list, tuple, range, set, frozenset, GeneratorType})


def _type_name(value: Any) -> str:
    return "generator" if isinstance(value, GeneratorType) else type(value).__name__


def _admit(value: Any) -> Any:
    """Let only the interpreter's own value types through, within their size bounds.

    Every operation and function result passes here, so a C function can never hand the
    expression an object of another type (``statistics.linear_regression`` returns a named
    tuple, which becomes a plain tuple).
    """
    kind = type(value)
    if kind is int:
        if value.bit_length() > _MAX_RESULT_BITS:
            raise _OverBudget(_TOO_LARGE)
        return value
    if kind in _PASS_THROUGH:
        return value
    if kind is str:
        if len(value) > MAX_STRING_CHARS:
            raise _OverBudget(_TOO_LONG_TEXT)
        return value
    if kind is complex:
        raise ValueError("the result is not a real number")
    if isinstance(value, tuple):
        return tuple(value)
    raise TypeError(f"the calculator does not handle {kind.__name__} values")


def _iterable(value: Any) -> Iterable[Any]:
    """``value`` itself when the interpreter may iterate it, else Python's TypeError."""
    if type(value) not in _ITERABLES:
        raise TypeError(f"'{_type_name(value)}' object is not iterable")
    return value


def _range_len(r: range) -> int:
    """``len(r)``, also for ranges longer than ``sys.maxsize`` (where ``len`` overflows)."""
    step = r.step
    return max(0, (r.stop - r.start + step - (1 if step > 0 else -1)) // step)


def _range_last(r: range, count: int) -> int:
    return r.start + (count - 1) * r.step


def _range_sum(r: range) -> int:
    """The closed form of ``sum(r)``: exact and O(1)."""
    count = _range_len(r)
    return count * (r.start + _range_last(r, count)) // 2 if count else 0


def _range_contains(r: range, item: Any) -> bool:
    """O(1) membership. CPython scans the whole range for a float, and 1.5 in range(10**15)
    would never return."""
    kind = type(item)
    if kind in _INTEGERS:
        return int(item) in r
    if kind is float:
        return item.is_integer() and int(item) in r
    return False  # a string or a container never equals an int


def _repeat_size(sequence: Any, times: int) -> tuple[int, int]:
    """(items, bytes) of ``sequence * times``, computed without building it."""
    count = len(sequence) * max(times, 0)
    if type(sequence) is str:
        if count > MAX_STRING_CHARS:
            raise _OverBudget(_TOO_LONG_TEXT)
        return count, count * (1 if sequence.isascii() else 4) + 64
    return count, 8 * count + 64


def _checked_pow(base: Any, exponent: Any) -> Any:
    """``base ** exponent`` with the digit bound estimated before anything big is computed."""
    if type(base) in _INTEGERS and type(exponent) in _INTEGERS and exponent > 1 and abs(base) > 1:
        # bit_length() first: it guards the float multiplication against an OverflowError.
        too_big = exponent.bit_length() > 40
        if too_big or exponent * math.log10(abs(base)) > MAX_RESULT_DIGITS:
            raise _OverBudget(_TOO_LARGE)
    return operator.pow(base, exponent)


def _checked_mul(left: Any, right: Any) -> Any:
    """``left * right`` for numbers, refused up front when the product is too long."""
    if (
        type(left) in _INTEGERS
        and type(right) in _INTEGERS
        and left.bit_length() + right.bit_length() > _MAX_RESULT_BITS
    ):
        raise _OverBudget(_TOO_LARGE)
    return operator.mul(left, right)


def _log10_factorial(n: int) -> float:
    return math.lgamma(n + 1) / math.log(10)


def _check_digits(log10_estimate: float) -> None:
    """Refuse a result whose estimated number of digits is above the bound (with 1% slack for
    the estimate's rounding)."""
    if log10_estimate > MAX_RESULT_DIGITS * 1.01:
        raise _OverBudget(_TOO_LARGE)


class _Runtime(_Budget):
    """The budget plus the helpers every operation uses to walk or build values."""

    __slots__ = ()

    def iterate(self, value: Any) -> Iterator[Any]:
        """The items of ``value``, one step each. The interpreter's own generators already take a
        step for every item they produce, so they pass through as they are."""
        if type(value) is GeneratorType:
            return value
        return self.stepped(_iterable(value))

    def stepped(self, items: Iterable[Any]) -> Iterator[Any]:
        """``items``, one step each."""
        tick = self.tick
        for item in items:
            tick()
            yield item

    def numbers(self, value: Any, what: str) -> Iterator[Any]:
        """The items of ``value``, one step each; anything but a number is a TypeError, so
        ``sum`` never concatenates lists or strings (quadratic in CPython)."""
        for item in self.iterate(value):
            if type(item) not in _NUMBERS:
                raise TypeError(f"{what} takes numbers, not {_type_name(item)}")
            yield item

    def collect(self, value: Any) -> list[Any]:
        """A new list of the items of ``value``, every step and byte paid for up front when the
        size is known."""
        kind = type(value)
        if kind is range:
            count = _range_len(value)
            if count:
                largest = max(abs(value.start), abs(_range_last(value, count)))
                self.charge(count * (8 + sys.getsizeof(largest)) + 64)
                self.tick(count)
            return list(value)
        if kind in _SIZED:
            per_item = 8 if kind is not str or value.isascii() else 88
            # a tuple's items may be numbers C just made (divmod's quotient), kept only by it
            made = self.held(value) if kind is tuple else 0
            self.charge(per_item * len(value) + 64 + made)
            self.tick(len(value))
            return list(value)
        items: list[Any] = []
        charge, held, sizeof, append = self.charge, self.held, sys.getsizeof, items.append
        for item in _iterable(value):  # a generator: it took the steps, the items take bytes
            charge((held(item) if type(item) in _HOLDERS else sizeof(item)) + 8)
            append(item)
        return items

    def held(self, value: Any) -> int:
        """The bytes ``value`` keeps alive as an item of a list, tuple or set: its own size and,
        for a tuple or a range, the values in it, which C may have just made (the quotient of
        ``divmod``, the index of ``enumerate``, the bounds and length of ``range(x + 1)``). A list
        or set paid for its items when it was built (``collect`` counts a tuple's this way too).
        A generator is refused: what it holds (a range, a counter, other generators) is out of
        sight, and nothing needs to keep one."""
        kind = type(value)
        if kind is tuple:
            self.tick(len(value) // 2 + 1)
            held, sizeof = self.held, sys.getsizeof
            total = sizeof(value)
            for item in value:
                total += held(item) if type(item) in _HOLDERS else sizeof(item)
            return total
        if kind is range:  # the bounds, the step and the length it computed
            parts = (value.start, value.stop, value.step)
            return sys.getsizeof(value) + 2 * sum(map(sys.getsizeof, parts))
        if kind is GeneratorType:
            raise TypeError(
                "a list, tuple or set cannot hold a generator (map, zip, filter, enumerate, "
                "reversed or ( ... for ... )); keep its items with list(...) instead"
            )
        return sys.getsizeof(value)

    def weight(self, value: Any) -> int:
        """About how many bytes hashing or comparing ``value`` reads in C: its own size and, for
        a list, tuple or set, the weight of each item, once per reference. That can be far more
        than the memory it takes: ``[t] * 1000`` holds ``t`` a thousand times, and nesting
        multiplies. Walking a container takes steps, so a value built to be vast in C is refused
        while it is weighed."""
        kind = type(value)
        if kind in _CONTAINERS:
            self.tick(len(value) // 2 + 2)
            weight, size = self.weight, sys.getsizeof
            total = size(value)
            for item in value:
                total += weight(item) if type(item) in _WEIGHED else size(item)
            return total
        if kind is range:  # hashing or comparing a range reads its start, step and length
            parts = (value.start, value.stop, value.step)
            return sys.getsizeof(value) + 2 * sum(map(sys.getsizeof, parts))
        return sys.getsizeof(value)

    def comparable(self, value: Any) -> Any:
        """``value``, paid for by its weight first when it is a list, tuple, set or range, before
        C compares it with another (numbers and strings compare within microseconds)."""
        if type(value) in _WEIGHED:
            self.tick(self.weight(value) // BYTES_PER_STEP)
        return value

    def sorting(self, items: list[Any]) -> None:
        """Pay for sorting ``items`` in C: at most about n log n comparisons, each a quarter of a
        step plus the weight of the heaviest item."""
        weight, size = self.weight, sys.getsizeof
        heaviest = max(
            (weight(item) if type(item) in _WEIGHED else size(item) for item in items), default=0
        )
        comparisons = len(items) * max(1, len(items).bit_length())
        self.tick(comparisons // 4 + comparisons * (heaviest // BYTES_PER_STEP))

    def check_hashes(self, items: list[Any] | set[Any] | frozenset[Any]) -> None:
        """Pay for hashing ``items`` before a set is built from them: the weight of each item
        (hashed here and again by the set) and, for an item whose hash an earlier, different
        value has, a comparison with each value of that hash (the set compares it with all of
        them): a step each, plus the weight of a heavy item. More than ``MAX_HASH_COLLISIONS``
        different values that share one hash, among more than twice as many items, are refused;
        repeats of the same values are not (``-1`` and ``-2`` share a hash). Its dictionaries are
        paid for as they grow."""
        first: dict[int, Any] = {}
        chains: dict[int, list[Any]] = {}  # a shared hash -> the other values that have it
        paid = 0
        weight, tick = self.weight, self.tick
        large = len(items) > 2 * MAX_HASH_COLLISIONS
        for count, item in enumerate(items):
            cost = weight(item) // BYTES_PER_STEP
            if cost:
                tick(2 * cost)
            key = hash(item)
            seen = first.setdefault(key, item)
            if seen is not item:
                others: Sequence[Any] = chains.get(key, ())
                if cost or others:
                    tick((len(others) + 1) * cost + len(others))
                if seen != item and all(other is not item and other != item for other in others):
                    chain = chains.setdefault(key, [])
                    chain.append(item)
                    self.charge(64)
                    if large and len(chain) >= MAX_HASH_COLLISIONS:
                        raise _OverBudget(
                            f"more than {MAX_HASH_COLLISIONS:,} different values share one "
                            "hash, which would make a set of them quadratic (the calculator's "
                            "limit); use run_python"
                        )
            if not count & 0xFFF:  # about 64 bytes per entry, paid every 4,096 items
                self.charge(64 * (len(first) - paid))
                paid = len(first)
        self.charge(64 * (len(first) - paid))

    def build_set(self, items: list[Any], kind: type = set) -> Any:
        """A set or frozenset of ``items`` (hashes checked first), charged at its real size: its
        hash table outgrows the list it came from."""
        self.check_hashes(items)
        result = kind(items)
        self.charge(sys.getsizeof(result))
        return result

    def subscript(self, container: Any, key: Any) -> Any:
        """``container[key]`` for lists, tuples, strings and ranges; a slice is paid for by its
        length (a range's slice is another range, O(1))."""
        kind = type(container)
        if kind not in _SEQUENCES and kind is not range:
            raise TypeError(f"'{_type_name(container)}' object is not subscriptable")
        if type(key) is slice:
            parts = (key.start, key.stop, key.step)
            if any(part is not None and type(part) not in _INTEGERS for part in parts):
                raise TypeError("slice indices must be integers or None")
            if kind is not range:
                count = len(range(*key.indices(len(container))))
                wide = kind is str and not container.isascii()
                self.charge((4 * count if wide else count) if kind is str else 8 * count + 64)
                self.tick(count // 64 + 1)
            return container[key]
        if type(key) not in _INTEGERS:
            raise TypeError(f"{kind.__name__} indices must be integers, not {_type_name(key)}")
        return container[key]

    def contains(self, item: Any, container: Any) -> bool:
        """``item in container`` with its cost paid first: O(1) for a range, hashing the item
        for a set, the scan for a string, and for anything else a comparison with every element,
        each reading at most the item's weight."""
        kind = type(container)
        if kind is range:
            return _range_contains(container, item)
        if kind is str:
            if type(item) is not str:
                raise TypeError(
                    f"'in <string>' requires string as left operand, not {_type_name(item)}"
                )
            self.tick(len(container) // 64 + 1)
            return item in container
        per_element = 1 + self.weight(item) // BYTES_PER_STEP
        if kind in _SETS:
            self.tick(per_element)
            return item in container
        if kind in (list, tuple):
            self.tick(len(container) * per_element + 1)
            return item in container
        extra = per_element - 1
        for element in self.iterate(container):  # a generator takes a step per element itself
            if extra:
                self.tick(extra)
            if element is item or element == item:
                return True
        return False

    def compare(self, compare: Callable[[Any, Any], Any], left: Any, right: Any) -> Any:
        """A comparison with its cost paid first: two strings by length; two lists, tuples or
        sets by the weight of the one with fewer items, since C compares them item by item, as
        deep as they go."""
        if type(left) is str and type(right) is str:
            self.tick(min(len(left), len(right)) // 64 + 1)
        elif type(left) in _CONTAINERS and type(right) in _CONTAINERS:
            shorter = left if len(left) <= len(right) else right
            self.tick(self.weight(shorter) // BYTES_PER_STEP + 1)
        return compare(left, right)

    # -- operators with size bounds ------------------------------------------------------------

    def add(self, left: Any, right: Any) -> Any:
        kind = type(left)
        if kind in _SEQUENCES and type(right) is kind:
            count = len(left) + len(right)
            if kind is str:
                if count > MAX_STRING_CHARS:
                    raise _OverBudget(_TOO_LONG_TEXT)
                self.charge(count * (1 if left.isascii() and right.isascii() else 4) + 64)
            else:
                self.charge(8 * count + 64)
            self.tick(count // 64 + 1)
        return operator.add(left, right)

    def mul(self, left: Any, right: Any) -> Any:
        if type(left) in _SEQUENCES and type(right) in _INTEGERS:
            return self.repeat(left, right)
        if type(right) in _SEQUENCES and type(left) in _INTEGERS:
            return self.repeat(right, left)
        return _checked_mul(left, right)

    def repeat(self, sequence: Any, times: int) -> Any:
        count, size = _repeat_size(sequence, times)
        self.charge(size)
        self.tick(count // 64 + 1)
        return sequence * times

    def set_operation(self, apply: Callable[[Any, Any], Any], left: Any, right: Any) -> Any:
        if type(left) not in _SETS or type(right) not in _SETS:
            return apply(left, right)
        count = len(left) + len(right)
        self.charge(64 * count + 216)
        self.tick(count + 1)
        result = apply(left, right)
        if apply is operator.or_:  # a union can gather more same-hash values than either side
            self.check_hashes(result)
        return result

    def sub(self, left: Any, right: Any) -> Any:
        return self.set_operation(operator.sub, left, right)

    def bit_and(self, left: Any, right: Any) -> Any:
        return self.set_operation(operator.and_, left, right)

    def bit_or(self, left: Any, right: Any) -> Any:
        return self.set_operation(operator.or_, left, right)

    def mod(self, left: Any, right: Any) -> Any:
        if type(left) is str:
            _out_of_scope("string formatting with % is not supported")
        return operator.mod(left, right)

    def lshift(self, left: Any, right: Any) -> Any:
        integers = type(left) in _INTEGERS and type(right) in _INTEGERS
        if integers and left and right > 0 and left.bit_length() + right > _MAX_RESULT_BITS:
            raise _OverBudget(_TOO_LARGE)
        return operator.lshift(left, right)

    def power(self, left: Any, right: Any) -> Any:
        return _checked_pow(left, right)

    def truediv(self, left: Any, right: Any) -> Any:
        return operator.truediv(left, right)

    def floordiv(self, left: Any, right: Any) -> Any:
        return operator.floordiv(left, right)

    def rshift(self, left: Any, right: Any) -> Any:
        return operator.rshift(left, right)


# --- functions ----------------------------------------------------------------------------------

_MISSING: Any = object()


@dataclass(frozen=True, slots=True)
class _Spec:
    """A table function: ``impl(runtime, *args, **kwargs)`` plus the keyword arguments it takes
    and the argument positions (or keywords) that take a function, as in ``map(int, ...)``."""

    impl: Callable[..., Any]
    keywords: frozenset[str] = frozenset()
    function_args: frozenset[int | str] = frozenset()


def _called(
    rt: _Runtime, name: str, spec: _Spec, args: Sequence[Any], kwargs: dict[str, Any]
) -> Any:
    """Call ``spec`` as ``name``. When Python refuses the arguments, the message names the
    function the expression called and counts only what it passed, as for ``sum(1, 2, 3)``:
    'sum() takes from 1 to 2 positional arguments but 3 were given', not '_sum() takes from 2
    to 3 ...' (every implementation takes the runtime first)."""
    impl = spec.impl
    try:
        return _admit(impl(rt, *args, **kwargs))
    except TypeError as exc:
        internal = f"{impl.__qualname__}()"
        detail = str(exc)
        if not detail.startswith(internal):
            raise
        detail = detail.removeprefix(internal)
        code = getattr(impl, "__code__", None)
        if code is not None and detail.startswith(" takes ") and detail.endswith(" given"):
            most = code.co_argcount - 1
            least = most - len(getattr(impl, "__defaults__", None) or ())
            takes = str(most) if least == most else f"from {least} to {most}"
            noun = "argument" if takes == "1" else "arguments"
            verb = "was" if len(args) == 1 else "were"
            detail = f" takes {takes} positional {noun} but {len(args)} {verb} given"
        raise TypeError(f"{name}(){detail}") from None


@dataclass(frozen=True, slots=True)
class _Function:
    """A table function passed as a value (``map(int, ...)``, ``key=abs``). Only the
    implementations of map, filter, sorted, min and max receive one, and they can only call it."""

    name: str
    spec: _Spec

    def call(self, rt: _Runtime, *args: Any) -> Any:
        rt.tick()
        return _called(rt, self.name, self.spec, args, {})


def _function_arg(value: Any, owner: str) -> _Function:
    if not isinstance(value, _Function):
        raise TypeError(
            f"{owner} takes a function such as abs, int or len, not {_type_name(value)}"
        )
    return value


def _pure(function: Callable[..., Any]) -> Callable[..., Any]:
    """A C function of numbers: nothing to bound beyond the result check every call gets."""

    def call(rt: _Runtime, *args: Any, **kwargs: Any) -> Any:
        return function(*args, **kwargs)

    return call


def _text(function: Callable[[Any], str]) -> Callable[..., str]:
    """``bin``, ``hex`` and ``oct``: the string they build is paid for."""

    def call(rt: _Runtime, value: Any, /) -> str:
        result = function(value)
        rt.charge(len(result) + 64)
        return result

    return call


def _min_max(pick: Callable[..., Any], name: str) -> Callable[..., Any]:
    """``min`` or ``max``. C compares one item (or key) at a time with the best so far, so each
    one is paid for by its weight as it arrives."""

    def call(rt: _Runtime, *args: Any, key: Any = None, default: Any = _MISSING) -> Any:
        if not args:
            raise TypeError(f"{name} expected at least 1 argument, got 0")
        if len(args) > 1:
            if default is not _MISSING:
                raise TypeError(
                    f"Cannot specify a default for {name}() with multiple positional arguments"
                )
            source: Iterator[Any] = iter(args)
        elif type(args[0]) is range and key is None and (count := _range_len(args[0])):
            return pick(args[0].start, _range_last(args[0], count))  # O(1): an end of the range
        else:
            source = rt.iterate(args[0])
        # an empty input: the default, else Python's own ValueError
        if key is None:
            source = map(rt.comparable, source)
            return pick(source) if default is _MISSING else pick(source, default=default)
        function = _function_arg(key, f"{name}(key=...)")
        pairs = ((rt.comparable(function.call(rt, item)), item) for item in source)
        if default is _MISSING:
            return pick(pairs, key=operator.itemgetter(0))[1]
        return pick(pairs, key=operator.itemgetter(0), default=(None, default))[1]

    return call


def _sum(rt: _Runtime, iterable: Any, /, start: Any = 0) -> Any:
    if type(start) not in _NUMBERS:
        raise TypeError("sum() adds numbers; its start value must be a number")
    if type(iterable) is range:  # closed form: exact and O(1)
        return start + _range_sum(iterable)
    return sum(rt.numbers(iterable, "sum()"), start)


def _len(rt: _Runtime, value: Any, /) -> int:
    if type(value) is range:
        return _range_len(value)
    if type(value) in _SIZED:
        return len(value)
    raise TypeError(f"object of type '{_type_name(value)}' has no len()")


def _sorted(rt: _Runtime, iterable: Any, /, *, key: Any = None, reverse: Any = False) -> list[Any]:
    items = rt.collect(iterable)
    if key is None:
        rt.sorting(items)
        items.sort(reverse=reverse)
        return items
    function = _function_arg(key, "sorted(key=...)")
    keys = []
    for item in items:
        value = function.call(rt, item)
        rt.charge(rt.held(value) + 8)
        keys.append(value)
    rt.sorting(keys)
    order = sorted(range(len(items)), key=keys.__getitem__, reverse=reverse)
    return [items[index] for index in order]


def _reversed(rt: _Runtime, value: Any, /) -> Any:
    if type(value) is range:
        return value[::-1]
    if type(value) in _SEQUENCES:
        return rt.stepped(reversed(value))
    raise TypeError(f"'{_type_name(value)}' object is not reversible")


def _enumerate(rt: _Runtime, iterable: Any, /, start: Any = 0) -> Iterator[tuple[int, Any]]:
    if type(start) not in _INTEGERS:
        raise TypeError(f"enumerate() start must be an integer, not {_type_name(start)}")
    items = rt.iterate(iterable)
    return ((index, item) for index, item in zip(itertools.count(start), items))


def _zip(rt: _Runtime, *iterables: Any) -> Iterator[tuple[Any, ...]]:
    sources = [rt.iterate(iterable) for iterable in iterables]
    return (items for items in zip(*sources, strict=False))


def _map(rt: _Runtime, function: Any, /, *iterables: Any) -> Iterator[Any]:
    apply = _function_arg(function, "map()")
    if not iterables:
        raise TypeError("map() must have at least two arguments.")
    sources = [rt.iterate(iterable) for iterable in iterables]
    return (apply.call(rt, *items) for items in zip(*sources, strict=False))


def _filter(rt: _Runtime, function: Any, iterable: Any, /) -> Iterator[Any]:
    keep = _function_arg(function, "filter()")
    return (item for item in rt.iterate(iterable) if keep.call(rt, item))


def _any(rt: _Runtime, iterable: Any, /) -> bool:
    return any(rt.iterate(iterable))  # a range has a true value among its first two


def _all(rt: _Runtime, iterable: Any, /) -> bool:
    if type(iterable) is range:
        return not _range_contains(iterable, 0)
    return all(rt.iterate(iterable))


def _list(rt: _Runtime, iterable: Any = (), /) -> list[Any]:
    return rt.collect(iterable)


def _tuple(rt: _Runtime, iterable: Any = (), /) -> tuple[Any, ...]:
    items = rt.collect(iterable)
    rt.charge(8 * len(items) + 64)
    return tuple(items)


def _set(rt: _Runtime, iterable: Any = (), /) -> Any:
    return rt.build_set(rt.collect(iterable))


def _frozenset(rt: _Runtime, iterable: Any = (), /) -> Any:
    return rt.build_set(rt.collect(iterable), frozenset)


def _str(rt: _Runtime, value: Any = "", /) -> str:
    kind = type(value)
    if kind is int and value.bit_length() > _MAX_STR_INT_BITS:
        raise ValueError(f"str() converts integers of up to {MAX_STR_INT_DIGITS:,} digits")
    if kind in _NUMBERS or kind is str:
        return str(value)
    raise TypeError(f"str() takes a number or a string here, not {_type_name(value)}")


_POWER_OF_TWO_BASES = frozenset({2, 4, 8, 16, 32})


def _int(rt: _Runtime, *args: Any, **kwargs: Any) -> int:
    """``int()``; strings above ``MAX_STR_INT_DIGITS`` digits are refused (except in the bases
    that are powers of two) whatever the process's own ``sys.set_int_max_str_digits`` says: that
    conversion is quadratic."""
    base = kwargs.get("base", args[1] if len(args) > 1 else 10)
    if args and type(args[0]) is str and base not in _POWER_OF_TWO_BASES:
        text = args[0].strip().lower()
        prefixed = base == 0 and text.lstrip("+-").startswith(("0x", "0o", "0b"))
        if not prefixed and sum(char.isalnum() for char in text) > MAX_STR_INT_DIGITS:
            raise ValueError(f"int() converts strings of up to {MAX_STR_INT_DIGITS:,} digits")
    return int(*args, **kwargs)


def _print(rt: _Runtime, *values: Any) -> Any:
    """``print(x)`` is ``x``: models often wrap the expression in print()."""
    if not values:
        raise TypeError("print() needs the value to show")
    return values[0] if len(values) == 1 else values


def _round(rt: _Runtime, number: Any, ndigits: Any = None) -> Any:
    """``round`` with bounded precision (``round(5, -10**9)`` would otherwise build 10**10**9)."""
    if ndigits is None:
        return round(number)
    if type(ndigits) not in _INTEGERS or abs(ndigits) > MAX_ROUND_DIGITS:
        raise ValueError(f"round() accepts at most {MAX_ROUND_DIGITS} digits of precision")
    return round(number, ndigits)


def _pow(rt: _Runtime, base: Any, exp: Any, mod: Any = None) -> Any:
    """``pow``; with a modulus the cost is exponent bits times modulus bits squared, and a
    negative exponent adds the inverse taken first (:data:`POW_INVERSE_BITS`)."""
    if mod is None:
        return _checked_pow(base, exp)
    if type(base) in _INTEGERS and type(exp) in _INTEGERS and type(mod) in _INTEGERS:
        exp_bits, mod_bits = abs(exp).bit_length(), abs(mod).bit_length()
        inverse = POW_INVERSE_BITS if exp < 0 else 0
        if (max(exp_bits, 1) + inverse) * max(mod_bits, 64) ** 2 > MAX_POW_MOD_COST:
            sign = "negative " if exp < 0 else ""
            raise _OverBudget(
                f"pow() with a {sign}{exp_bits:,}-bit exponent and a {mod_bits:,}-bit modulus "
                "would take too long (the calculator's limit); use run_python"
            )
    return pow(base, exp, mod)


def _factorial(rt: _Runtime, n: Any, /) -> Any:
    if type(n) in _INTEGERS and n > 1:
        if n > 10**7:
            raise _OverBudget(_TOO_LARGE)
        _check_digits(_log10_factorial(n))
    return math.factorial(n)


def _log10_perm(n: int, k: int) -> float:
    """log10 of perm(n, k) for 0 < k <= n, without cancellation when k is tiny next to n (then
    the first form is a slight overestimate)."""
    if k * 1000 <= n:
        return k * math.log10(n)
    return (math.lgamma(n + 1) - math.lgamma(n - k + 1)) / math.log(10)


def _comb(rt: _Runtime, n: Any, k: Any, /) -> Any:
    if type(n) in _INTEGERS and type(k) in _INTEGERS and 0 < k < n:
        small = min(k, n - k)
        _check_digits(_log10_perm(n, small) - _log10_factorial(small))
    return math.comb(n, k)


def _perm(rt: _Runtime, n: Any, k: Any = None, /) -> Any:
    if k is None:
        return _factorial(rt, n)
    if type(n) in _INTEGERS and type(k) in _INTEGERS and 0 < k <= n:
        _check_digits(_log10_perm(n, k))
    return math.perm(n, k)


def _operands(rt: _Runtime, args: tuple[Any, ...]) -> Iterator[Any]:
    """gcd and lcm take numbers, or one list or range of them (``lcm(range(1, 21))``)."""
    if len(args) == 1 and type(args[0]) in _ITERABLES and type(args[0]) is not str:
        return rt.iterate(args[0])
    return rt.stepped(args)


def _gcd(rt: _Runtime, *args: Any) -> int:
    result = 0
    for value in _operands(rt, args):
        result = math.gcd(result, value)
    return result


def _lcm(rt: _Runtime, *args: Any) -> int:
    result = 1
    for value in _operands(rt, args):
        result = _admit(math.lcm(result, value))
    return result


def _prod(rt: _Runtime, iterable: Any, /, *, start: Any = 1) -> Any:
    if type(start) not in _NUMBERS:
        raise TypeError("prod() multiplies numbers; its start value must be a number")
    result = start
    for value in rt.numbers(iterable, "prod()"):
        result = _admit(_checked_mul(result, value))
    return result


def _fsum(rt: _Runtime, iterable: Any, /) -> float:
    return math.fsum(rt.numbers(iterable, "fsum()"))


def _sumprod(rt: _Runtime, p: Any, q: Any, /) -> Any:
    return math.sumprod(rt.numbers(p, "sumprod()"), rt.numbers(q, "sumprod()"))


def _dist(rt: _Runtime, p: Any, q: Any, /) -> float:
    return math.dist(tuple(rt.collect(p)), tuple(rt.collect(q)))


def _stats_data(rt: _Runtime, data: Any, text: bool) -> list[Any]:
    """The data points of a statistics call: at most ``MAX_STATS_ITEMS`` numbers within the
    float range (``mode`` and ``multimode`` also count strings, and hash what they count)."""
    too_many = f"statistics functions take at most {MAX_STATS_ITEMS:,} values"
    if (type(data) is range and _range_len(data) > MAX_STATS_ITEMS) or (
        type(data) in _SIZED and len(data) > MAX_STATS_ITEMS
    ):
        raise _OverBudget(too_many)
    items = rt.collect(data)
    if len(items) > MAX_STATS_ITEMS:
        raise _OverBudget(too_many)
    for item in items:
        kind = type(item)
        if kind is int and abs(item) > MAX_STATS_INT:
            raise ValueError(_STATS_RANGE)
        if kind not in _NUMBERS and not (text and kind is str):
            raise TypeError(f"statistics functions take numbers, not {_type_name(item)}")
    if text:
        rt.check_hashes(items)  # mode and multimode count the values in a dict
    return items


def _statistics(
    function: Callable[..., Any], data_args: int = 1, text: bool = False
) -> Callable[..., Any]:
    def call(rt: _Runtime, *args: Any, **kwargs: Any) -> Any:
        values = [
            _stats_data(rt, arg, text) if index < data_args else arg
            for index, arg in enumerate(args)
        ]
        if kwargs.get("weights") is not None:
            kwargs["weights"] = _stats_data(rt, kwargs["weights"], False)
        for value in (*args[data_args:], *kwargs.values()):  # xbar, mu, interval, n
            if type(value) is int and abs(value) > MAX_STATS_INT:
                raise ValueError(_STATS_RANGE)
        pieces = kwargs.get("n", 4)
        if type(pieces) in _INTEGERS and pieces > 10_000:
            raise _OverBudget("quantiles() takes n up to 10,000")
        method = kwargs.get("method", "")
        if type(method) is not str:  # an unknown method is shown with repr(), deep and slow
            raise TypeError(
                f"method must be a string such as 'inclusive' or 'ranked', not {_type_name(method)}"
            )
        result = function(*values, **kwargs)
        if type(result) is list:  # the cut points of quantiles: new floats a list may keep
            rt.charge(sum(map(sys.getsizeof, result)))
        return result

    return call


def _spec(
    impl: Callable[..., Any], keywords: str = "", functions: tuple[int | str, ...] = ()
) -> _Spec:
    return _Spec(impl, frozenset(keywords.split()), frozenset(functions))


_BUILTINS: dict[str, _Spec] = {
    "abs": _spec(_pure(abs)),
    "all": _spec(_all),
    "any": _spec(_any),
    "bin": _spec(_text(bin)),
    "bool": _spec(_pure(bool)),
    "divmod": _spec(_pure(divmod)),
    "enumerate": _spec(_enumerate, "start"),
    "filter": _spec(_filter, functions=(0,)),
    "float": _spec(_pure(float)),
    "frozenset": _spec(_frozenset),
    "hex": _spec(_text(hex)),
    "int": _spec(_int, "base"),
    "len": _spec(_len),
    "list": _spec(_list),
    "map": _spec(_map, functions=(0,)),
    "max": _spec(_min_max(max, "max"), "key default", ("key",)),
    "min": _spec(_min_max(min, "min"), "key default", ("key",)),
    "oct": _spec(_text(oct)),
    "pow": _spec(_pow, "base exp mod"),
    "print": _spec(_print),
    "range": _spec(_pure(range)),
    "reversed": _spec(_reversed),
    "round": _spec(_round, "ndigits"),
    "set": _spec(_set),
    "sorted": _spec(_sorted, "key reverse", ("key",)),
    "str": _spec(_str),
    "sum": _spec(_sum, "start"),
    "tuple": _spec(_tuple),
    "zip": _spec(_zip),
}
_MATH_PURE = (
    "acos acosh asin asinh atan atan2 atanh cbrt ceil copysign cos cosh degrees erf erfc exp exp2 "
    "expm1 fabs floor fma fmod frexp gamma hypot isfinite isinf isnan isqrt ldexp lgamma log "
    "log10 log1p log2 modf pow radians remainder sin sinh sqrt tan tanh trunc ulp"
)
_MATH: dict[str, _Spec] = {name: _spec(_pure(getattr(math, name))) for name in _MATH_PURE.split()}
_MATH |= {
    "comb": _spec(_comb),
    "dist": _spec(_dist),
    "factorial": _spec(_factorial),
    "fsum": _spec(_fsum),
    "gcd": _spec(_gcd),
    "isclose": _spec(_pure(math.isclose), "rel_tol abs_tol"),
    "lcm": _spec(_lcm),
    "nextafter": _spec(_pure(math.nextafter), "steps"),
    "perm": _spec(_perm),
    "prod": _spec(_prod, "start"),
    "sumprod": _spec(_sumprod),
}
_STATISTICS: dict[str, _Spec] = {
    "correlation": _spec(_statistics(statistics.correlation, 2), "method"),
    "covariance": _spec(_statistics(statistics.covariance, 2)),
    "fmean": _spec(_statistics(statistics.fmean), "weights"),
    "geometric_mean": _spec(_statistics(statistics.geometric_mean)),
    "harmonic_mean": _spec(_statistics(statistics.harmonic_mean), "weights"),
    "linear_regression": _spec(_statistics(statistics.linear_regression, 2), "proportional"),
    "mean": _spec(_statistics(statistics.mean)),
    "median": _spec(_statistics(statistics.median)),
    "median_grouped": _spec(_statistics(statistics.median_grouped), "interval"),
    "median_high": _spec(_statistics(statistics.median_high)),
    "median_low": _spec(_statistics(statistics.median_low)),
    "mode": _spec(_statistics(statistics.mode, text=True)),
    "multimode": _spec(_statistics(statistics.multimode, text=True)),
    "pstdev": _spec(_statistics(statistics.pstdev), "mu"),
    "pvariance": _spec(_statistics(statistics.pvariance), "mu"),
    "quantiles": _spec(_statistics(statistics.quantiles), "n method"),
    "stdev": _spec(_statistics(statistics.stdev), "xbar"),
    "variance": _spec(_statistics(statistics.variance), "xbar"),
}
_FUNCTIONS: dict[str, _Spec] = (
    {f"math.{name}": spec for name, spec in _MATH.items()}
    | {f"statistics.{name}": spec for name, spec in _STATISTICS.items()}
    | {name: spec for name, spec in _MATH.items() if name != "pow"}  # bare pow is the builtin
    | _STATISTICS
    | _BUILTINS
)
"""Every callable name: the builtins, math and statistics functions by their bare names and as
``math.<name>`` / ``statistics.<name>``. Nothing else can be called."""

_CONSTANT_VALUES = {"pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf, "nan": math.nan}
_CONSTANTS: dict[str, float] = _CONSTANT_VALUES | {
    f"math.{name}": value for name, value in _CONSTANT_VALUES.items()
}


# --- compiler -----------------------------------------------------------------------------------

Env = tuple[dict[str, Any], ...]
"""One frame per enclosing comprehension, outermost first; a frame maps its loop variables."""
Code = Callable[[Env], Any]
Scope = tuple[frozenset[str], ...]
"""The loop variables of each enclosing comprehension, known before anything runs."""

_BINARY: dict[type[ast.operator], str] = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
    ast.Div: "truediv",
    ast.FloorDiv: "floordiv",
    ast.Mod: "mod",
    ast.Pow: "power",
    ast.LShift: "lshift",
    ast.RShift: "rshift",
    ast.BitAnd: "bit_and",
    ast.BitOr: "bit_or",
}
"""Operator -> ``_Runtime`` method (each one checks sizes before it computes)."""
_UNARY: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Invert: operator.invert,
    ast.Not: operator.not_,
}
_COMPARE: dict[type[ast.cmpop], Callable[[Any, Any], Any]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_REFUSED: dict[type[ast.AST], str] = {
    ast.Lambda: (
        "cannot define functions (lambda); a comprehension such as sum(x**2 for x in xs) usually "
        "does the same job"
    ),
    ast.NamedExpr: "has no variables (:=)",
    ast.JoinedStr: "does no string formatting (f-strings)",
    ast.FormattedValue: "does no string formatting (f-strings)",
    ast.Starred: "cannot unpack with *; pass the list or range itself, e.g. lcm(range(1, 21))",
    ast.Dict: "has no dictionaries",
    ast.DictComp: "has no dictionaries",
}
_UNDERSCORE = "names that start with an underscore ({}) are not available"
_UNKNOWN_FUNCTION = (
    "unknown function {!r}; the calculator has Python's built-in functions such as sum, len, "
    "range, sorted, min, max, round and abs, and the math and statistics functions such as "
    "sqrt, comb, isqrt, mean and median"
)


def _refuse(node: ast.AST) -> NoReturn:
    what = _REFUSED.get(type(node), f"cannot evaluate {type(node).__name__} syntax")
    _out_of_scope(f"the calculator {what}")


def _bound(name: str, scope: Scope) -> bool:
    return any(name in names for names in scope)


def _unpack(value: Any, count: int) -> tuple[Any, ...] | list[Any]:
    """``a, b = value`` for a comprehension target; reads at most ``count + 1`` items."""
    if type(value) in (tuple, list):
        items = value
    elif type(value) in _ITERABLES:
        items = tuple(itertools.islice(value, count + 1))
    else:
        raise TypeError(f"cannot unpack non-iterable {_type_name(value)} object")
    if len(items) > count:
        raise ValueError(f"too many values to unpack (expected {count})")
    if len(items) < count:
        raise ValueError(f"not enough values to unpack (expected {count}, got {len(items)})")
    return items


class _Compiler:
    """Checks a syntax tree against the whitelist and turns it into closures over a runtime.

    Every node is checked while compiling, so a refused part refuses the whole expression before
    any of it runs. The closures take the comprehension frames (``Env``) and call only the
    runtime's bounded operations and the table functions.
    """

    def __init__(self, rt: _Runtime) -> None:
        self.rt = rt

    def compile(self, node: ast.AST, scope: Scope) -> Code:
        """The closure for ``node``; anything not handled here is refused."""
        if isinstance(node, ast.Constant):
            return self.constant(node)
        if isinstance(node, ast.Name):
            return self.name(node, scope)
        if isinstance(node, ast.BinOp):
            return self.binary(node, scope)
        if isinstance(node, ast.UnaryOp):
            return self.unary(node, scope)
        if isinstance(node, ast.Call):
            return self.call(node, scope)
        if isinstance(node, ast.Compare):
            return self.compare(node, scope)
        if isinstance(node, ast.BoolOp):
            return self.bool_op(node, scope)
        if isinstance(node, ast.IfExp):
            return self.if_exp(node, scope)
        if isinstance(node, ast.Subscript):
            return self.subscript(node, scope)
        if isinstance(node, ast.Attribute):
            return self.attribute(node, scope)
        if isinstance(node, ast.List):
            return self.display(node.elts, scope, list)
        if isinstance(node, ast.Tuple):
            return self.display(node.elts, scope, tuple)
        if isinstance(node, ast.Set):
            return self.display(node.elts, scope, self.rt.build_set)
        if isinstance(node, ast.ListComp):
            return self.comprehension(node, scope, list)
        if isinstance(node, ast.SetComp):
            return self.comprehension(node, scope, set)
        if isinstance(node, ast.GeneratorExp):
            return self.comprehension(node, scope, None)
        _refuse(node)

    # -- leaves --------------------------------------------------------------------------------

    def constant(self, node: ast.Constant) -> Code:
        value = node.value
        if type(value) is complex:
            raise ValueError("complex numbers are not supported")
        if type(value) not in _NUMBERS and type(value) is not str:
            raise ValueError(f"only numbers, strings and True/False are allowed, not {value!r}")
        value = _admit(value)
        return lambda env: value

    def name(self, node: ast.Name, scope: Scope) -> Code:
        name = node.id
        if name.startswith("_") and name != "_":
            _out_of_scope(_UNDERSCORE.format(name))
        for depth in range(len(scope) - 1, -1, -1):
            if name in scope[depth]:
                return lambda env: env[depth][name]
        if name in _CONSTANTS:
            value = _CONSTANTS[name]
            return lambda env: value
        if name in _FUNCTIONS:
            raise ValueError(f"{name} is a function; call it, as in {name}(...)")
        _out_of_scope(
            f"unknown name {name!r}; the calculator has no variables, only numbers, constants "
            "such as pi, the loop variables of a comprehension and functions such as sqrt() or "
            "sum()"
        )

    def qualified(self, node: ast.Attribute, scope: Scope) -> str:
        """``math.<name>`` or ``statistics.<name>`` from the tables; any other attribute is
        refused, so no value's attributes are ever reachable."""
        owner = node.value
        module = owner.id if isinstance(owner, ast.Name) and not _bound(owner.id, scope) else ""
        if module not in ("math", "statistics"):
            _out_of_scope(
                f"attribute access and method calls such as .{node.attr} are not supported, only "
                "math.<name> and statistics.<name> functions"
            )
        if node.attr.startswith("_"):
            _out_of_scope(_UNDERSCORE.format(f"{module}.{node.attr}"))
        key = f"{module}.{node.attr}"
        if key not in _FUNCTIONS and key not in _CONSTANTS:
            _out_of_scope(
                f"{key} is not available; the calculator has functions such as math.comb, "
                "math.isqrt, statistics.mean and statistics.median"
            )
        return key

    def attribute(self, node: ast.Attribute, scope: Scope) -> Code:
        key = self.qualified(node, scope)
        if key in _FUNCTIONS:
            raise ValueError(f"{key} is a function; call it, as in {key}(...)")
        value = _CONSTANTS[key]
        return lambda env: value

    # -- operators -----------------------------------------------------------------------------

    def binary(self, node: ast.BinOp, scope: Scope) -> Code:
        kind = type(node.op)
        if kind is ast.BitXor:
            raise ValueError("'^' is bitwise XOR in Python; use ** for powers, e.g. 2**10")
        if kind not in _BINARY:
            raise ValueError(f"the {kind.__name__} operator is not supported")
        apply = getattr(self.rt, _BINARY[kind])
        left, right = self.compile(node.left, scope), self.compile(node.right, scope)
        tick = self.rt.tick

        def run(env: Env) -> Any:
            a, b = left(env), right(env)
            tick()
            value = apply(a, b)
            kind = type(value)
            if kind is float or (kind is int and value.bit_length() <= _MAX_RESULT_BITS):
                return value
            return _admit(value)

        return run

    def unary(self, node: ast.UnaryOp, scope: Scope) -> Code:
        apply = _UNARY[type(node.op)]
        operand = self.compile(node.operand, scope)
        return lambda env: _admit(apply(operand(env)))  # O(size): no step of its own

    def compare(self, node: ast.Compare, scope: Scope) -> Code:
        kinds = [type(op) for op in node.ops]
        if ast.Is in kinds or ast.IsNot in kinds:
            raise ValueError("'is' compares object identity; use == or != instead")
        first = self.compile(node.left, scope)
        rest = [self.compile(comparator, scope) for comparator in node.comparators]
        rt = self.rt

        def run(env: Env) -> Any:
            left = first(env)
            result: Any = True
            for kind, code in zip(kinds, rest, strict=True):
                right = code(env)  # numbers compare in O(size); containers pay in rt
                if kind is ast.In:
                    result = rt.contains(left, right)
                elif kind is ast.NotIn:
                    result = not rt.contains(left, right)
                else:
                    result = rt.compare(_COMPARE[kind], left, right)
                if not result:
                    return result
                left = right
            return result

        return run

    def bool_op(self, node: ast.BoolOp, scope: Scope) -> Code:
        codes = [self.compile(value, scope) for value in node.values]
        stop_on_true = isinstance(node.op, ast.Or)

        def run(env: Env) -> Any:
            value: Any = None
            for code in codes:
                value = code(env)
                if bool(value) is stop_on_true:
                    return value
            return value

        return run

    def if_exp(self, node: ast.IfExp, scope: Scope) -> Code:
        test = self.compile(node.test, scope)
        body, orelse = self.compile(node.body, scope), self.compile(node.orelse, scope)
        return lambda env: body(env) if test(env) else orelse(env)

    def subscript(self, node: ast.Subscript, scope: Scope) -> Code:
        container = self.compile(node.value, scope)
        piece = node.slice
        index: Code
        if isinstance(piece, ast.Slice):
            parts = [
                None if part is None else self.compile(part, scope)
                for part in (piece.lower, piece.upper, piece.step)
            ]

            def index(env: Env) -> Any:
                return slice(*(None if part is None else part(env) for part in parts))
        else:
            index = self.compile(piece, scope)
        rt = self.rt

        def run(env: Env) -> Any:
            value, key = container(env), index(env)
            rt.tick()
            return _admit(rt.subscript(value, key))

        return run

    # -- containers ----------------------------------------------------------------------------

    def display(self, elements: list[ast.expr], scope: Scope, build: Callable[..., Any]) -> Code:
        codes = [self.compile(element, scope) for element in elements]
        rt = self.rt

        def run(env: Env) -> Any:
            items = [code(env) for code in codes]
            rt.charge(sum(map(rt.held, items)) + 8 * len(items) + 64)
            return build(items)

        return run

    def target(self, node: ast.expr, bound: set[str]) -> Callable[[dict[str, Any], Any], None]:
        """The assignment of a comprehension's ``for`` target: a name or a tuple of names."""
        if isinstance(node, ast.Name):
            name = node.id
            if name.startswith("_") and name != "_":
                _out_of_scope(_UNDERSCORE.format(name))
            bound.add(name)

            def bind(frame: dict[str, Any], value: Any) -> None:
                frame[name] = value

            return bind
        if isinstance(node, ast.Tuple | ast.List):
            parts = [self.target(element, bound) for element in node.elts]
            count = len(parts)

            def bind_all(frame: dict[str, Any], value: Any) -> None:
                for part, item in zip(parts, _unpack(value, count), strict=True):
                    part(frame, item)

            return bind_all
        if isinstance(node, ast.Starred):
            _refuse(node)
        _out_of_scope("a comprehension can only assign to names, as in 'for x in' or 'for i, x in'")

    def comprehension(
        self,
        node: ast.ListComp | ast.SetComp | ast.GeneratorExp,
        scope: Scope,
        build: Callable[..., Any] | None,
    ) -> Code:
        """A list, set or generator comprehension. As in Python, the first iterable is evaluated
        outside the comprehension and everything else inside it, where the loop variables live
        in a frame of their own; a generator runs lazily, one step per item."""
        bound: set[str] = set()
        clauses: list[tuple[Code, Callable[[dict[str, Any], Any], None], list[Code]]] = []
        for position, clause in enumerate(node.generators):
            if clause.is_async:
                _out_of_scope("the calculator cannot evaluate async comprehensions")
            outer = scope if position == 0 else (*scope, frozenset(bound))
            iterable = self.compile(clause.iter, outer)
            bind = self.target(clause.target, bound)
            inner = (*scope, frozenset(bound))
            conditions = [self.compile(test, inner) for test in clause.ifs]
            clauses.append((iterable, bind, conditions))
        element = self.compile(node.elt, (*scope, frozenset(bound)))
        rt, last = self.rt, len(clauses) - 1
        tick = rt.tick

        def loop(env: Env, frame: dict[str, Any], position: int, items: Any) -> Iterator[Any]:
            _, bind, conditions = clauses[position]
            for item in items:
                tick()
                bind(frame, item)
                for condition in conditions:
                    if not condition(env):
                        break
                else:
                    if position == last:
                        yield element(env)
                    else:
                        following = _iterable(clauses[position + 1][0](env))
                        yield from loop(env, frame, position + 1, following)

        def run(env: Env) -> Any:
            first = _iterable(clauses[0][0](env))
            frame: dict[str, Any] = {}
            items = loop((*env, frame), frame, 0, first)
            if build is None:
                return items
            collected = rt.collect(items)
            return collected if build is list else rt.build_set(collected)

        return run

    # -- calls ---------------------------------------------------------------------------------

    def callee(self, node: ast.expr, scope: Scope) -> str:
        """The table key of the function being called; never a computed value."""
        if isinstance(node, ast.Name):
            name = node.id
            if name.startswith("_"):
                _out_of_scope(_UNDERSCORE.format(name))
            if _bound(name, scope):
                raise ValueError(f"{name} is a loop variable, not a function")
            if name not in _FUNCTIONS:
                _out_of_scope(_UNKNOWN_FUNCTION.format(name))
            return name
        if isinstance(node, ast.Attribute):
            key = self.qualified(node, scope)
            if key not in _FUNCTIONS:
                raise ValueError(f"{key} is a number, not a function")
            return key
        if isinstance(node, ast.Lambda):
            _refuse(node)
        _out_of_scope("only named functions such as sqrt(2) or math.comb(5, 2) can be called")

    def argument(self, node: ast.expr, scope: Scope, takes_function: bool) -> Code:
        """An argument; where the function takes a function (``map(int, ...)``, ``key=len``)
        a table function's name becomes a ``_Function``."""
        if takes_function:
            key = ""
            if isinstance(node, ast.Name) and node.id in _FUNCTIONS and not _bound(node.id, scope):
                key = node.id
            elif isinstance(node, ast.Attribute):
                key = self.qualified(node, scope)
            if key in _FUNCTIONS:
                function = _Function(key, _FUNCTIONS[key])
                return lambda env: function
        return self.compile(node, scope)

    def call(self, node: ast.Call, scope: Scope) -> Code:
        key = self.callee(node.func, scope)
        spec = _FUNCTIONS[key]
        codes = [  # an ``*`` argument is an ast.Starred, which compile() refuses
            self.argument(argument, scope, position in spec.function_args)
            for position, argument in enumerate(node.args)
        ]
        keywords: dict[str, Code] = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                _out_of_scope("the calculator cannot unpack with **")
            if keyword.arg not in spec.keywords:
                takes = f"; it takes {', '.join(sorted(spec.keywords))}" if spec.keywords else ""
                raise ValueError(f"{key}() has no keyword argument {keyword.arg!r}{takes}")
            takes_function = keyword.arg in spec.function_args
            keywords[keyword.arg] = self.argument(keyword.value, scope, takes_function)
        rt = self.rt

        def run(env: Env) -> Any:
            args = [code(env) for code in codes]
            kwargs = {name: code(env) for name, code in keywords.items()}
            rt.tick()
            return _called(rt, key, spec, args, kwargs)

        return run


# --- parsing and results ------------------------------------------------------------------------

_LANGUAGE_TAG = re.compile(r"(?:[A-Za-z][\w+-]*)?")
"""The language name on a code fence's first line (``python``, ``py3``, ``c++``), or none."""
_THOUSANDS = re.compile(r"\d,\d{3}(?!\d)")
_COMMAS = "commas are not allowed in numbers; write 12000 instead of 12,000"
_BRACKETS: dict[type, tuple[str, str]] = {
    list: ("[", "]"),
    range: ("[", "]"),
    tuple: ("(", ")"),
    set: ("{", "}"),
    frozenset: ("frozenset({", "})"),
}
_FAILURE_MARKS = ("error:", "exit_code:")
"""How a tool result that reports a failure starts (``middleware.reports_failure``): a string
value that starts so is shown quoted."""
_EMPTY: dict[type, str] = {
    list: "[]",
    range: "[]",
    tuple: "()",
    set: "set()",
    frozenset: "frozenset()",
}


def _clean(text: str) -> str:
    """Undo what models wrap around an expression: a code fence, backticks, a trailing ``;`` or
    ``=``, and the Unicode operators ``×``, ``÷``, ``−``, ``≤``, ``≥``, ``≠`` and ``π``. Linear
    in the text (a regular expression for the fence backtracked cubically on spaces)."""
    text = text.translate(_UNICODE_OPERATORS).strip()
    if len(text) >= 6 and text.startswith("```") and text.endswith("```"):
        text = text[3:-3]
        tag, newline, code = text.partition("\n")
        if newline and _LANGUAGE_TAG.fullmatch(tag.rstrip(" \t")):
            text = code
    return text.strip().strip("`").strip().rstrip(";").strip().strip("= ")


def _harmless_import(statement: ast.stmt) -> bool:
    """``import math`` or ``from statistics import mean``: names the calculator already has."""
    if isinstance(statement, ast.Import):
        return all(
            alias.name in ("math", "statistics") and alias.asname is None
            for alias in statement.names
        )
    if isinstance(statement, ast.ImportFrom):
        return (
            statement.module in ("math", "statistics")
            and statement.level == 0
            and all(alias.asname is None for alias in statement.names)
        )
    return False


# The parser reports some text it accepts, such as the invalid escape in '\d' or a number glued
# to a keyword ('1if'), with a SyntaxWarning that is printed to stderr on every call, into the
# middle of the terminal UI. A parse with an empty file name reports as module '<unknown>', so
# this silences only anonymous parses. It is set once: catch_warnings() in every call would swap
# the process-wide filters under the agent's other threads.
warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"<unknown>\Z")


def _parse(text: str) -> ast.expr:
    """The expression to evaluate. Statements are refused, pointing at run_python, except
    ``import math``-style lines in front of one expression (the import changes nothing).

    The file name is empty because for a syntax error in "exec" mode CPython reads the line back
    from the named file, and the default, '<unknown>', is a relative path: a file of that name in
    the current directory would be read, and a FIFO there would hang the call. An empty name
    opens nothing."""
    try:
        body = ast.parse(text, filename="", mode="eval").body
    except SyntaxError as error:
        try:
            statements = ast.parse(text, filename="", mode="exec").body
        except SyntaxError:
            raise error from None
        last = statements[-1] if statements else None
        if isinstance(last, ast.Expr) and all(map(_harmless_import, statements[:-1])):
            return last.value
        _out_of_scope(
            "the calculator evaluates one expression, not statements such as assignments, "
            "imports, loops or several lines"
        )
    if isinstance(body, ast.Tuple) and _THOUSANDS.search(text):
        raise ValueError(_COMMAS)  # "12,000 + 5" parses as the tuple (12, 5)
    return body


def _check_depth(tree: ast.AST) -> None:
    """Refuse trees nested deeper than ``MAX_DEPTH`` before the recursive compiler sees them."""
    stack = [(tree, 1)]
    while stack:
        node, depth = stack.pop()
        if depth > MAX_DEPTH:
            raise ValueError(f"the expression is nested more than {MAX_DEPTH} levels deep")
        stack.extend((child, depth + 1) for child in ast.iter_child_nodes(node))


def _scientific(value: int) -> str:
    """Approximate a huge integer as ``m.mmmmmme+N`` without a full decimal conversion."""
    magnitude = math.log10(abs(value))
    exponent = int(magnitude)
    mantissa = f"{10 ** (magnitude - exponent):.6g}"
    if mantissa.startswith("10"):
        mantissa, exponent = "1", exponent + 1
    sign = "-" if value < 0 else ""
    return f"{sign}{mantissa}e+{exponent} (about {exponent + 1:,} digits)"


def _number(value: Any) -> str:
    """Integers exactly (huge ones as a summary), floats with 12 significant digits."""
    if type(value) is bool:
        return str(value)
    if type(value) is float:
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.12g}"
    if value.bit_length() > _MAX_STR_INT_BITS:
        return _scientific(value)
    try:
        return str(value)
    except ValueError:  # beyond sys.get_int_max_str_digits()
        return _scientific(value)


def _item(value: Any, room: int) -> str:
    kind = type(value)
    if kind in _NUMBERS:
        return _number(value)
    if kind is str:
        return repr(value) if len(value) <= room else f"{value[:room]!r}..."
    if kind in _BRACKETS:
        return _items(value, room)
    return f"<{_type_name(value)}>"


def _items(value: Any, room: int) -> str:
    """A list, tuple, set or range in Python's notation, cut after about ``room`` characters
    with the item count."""
    kind = type(value)
    count = _range_len(value) if kind is range else len(value)
    if not count:
        return _EMPTY[kind]
    opening, closing = _BRACKETS[kind]
    parts: list[str] = []
    used = 0
    for item in value:
        text = _item(item, max(room - used, 20))
        if parts and used + len(text) > room:
            break
        parts.append(text)
        used += len(text) + 2
        if used >= room:
            break
    body = ", ".join(parts) + ("," if kind is tuple and count == 1 else "")
    if len(parts) < count:
        return f"{opening}{body}, ...{closing} ({count:,} items, the first {len(parts):,} shown)"
    return f"{opening}{body}{closing}"


def _format_result(value: Any) -> str:
    if type(value) is float and math.isnan(value):
        return "nan (undefined result)"
    if type(value) in _NUMBERS:
        return _number(value)
    if type(value) is str:
        if not value.strip() or value.lstrip()[:10].lower().startswith(_FAILURE_MARKS):
            return repr(value)  # never "" and never a text that reads as a failure
        if len(value) > RESULT_CHARS:
            return f"{value[:RESULT_CHARS]}... ({len(value):,} characters)"
        return value
    return _items(value, RESULT_CHARS)


def _cut(text: str) -> str:
    """``text`` up to ``DETAIL_CHARS`` characters, with its length when cut."""
    if len(text) <= DETAIL_CHARS:
        return text
    return f"{text[:DETAIL_CHARS]}... ({len(text):,} characters)"


def _detail(exc: BaseException) -> str:
    """The message for the ERROR line: Python's own cut to ``DETAIL_CHARS``, the calculator's
    refusals whole (they are bounded by the expression and end with the run_python hint)."""
    detail = str(exc).strip().rstrip(".") or type(exc).__name__
    if "math domain error" in detail:
        return "math domain error (square root or logarithm of a negative number?)"
    return detail if isinstance(exc, _OutOfScope) else _cut(detail)


def _evaluate(text: str) -> str:
    tree = _parse(text)
    _check_depth(tree)
    rt = _Runtime()
    value = _Compiler(rt).compile(tree, ())(())
    if type(value) is GeneratorType:
        value = rt.collect(value)  # a bare generator shows its items
    return _format_result(value)


def calculate(expression: str) -> str:
    """Evaluate one Python expression safely and return its value as text.

    Args:
        expression: A Python expression such as ``"1836.6 * 0.15"``, ``"sum(range(10, 94))"``
            or ``"sum(n for n in range(20, 81) if n % 3 == 0)"``. Unicode ``×``, ``÷`` and
            ``−``, a stray ``=``, backticks, a code fence, ``print(...)`` and ``import math``
            lines in front are tolerated.

    Returns:
        The value as plain text, or an ``ERROR: ...`` line that says what was refused or which
        limit was hit. Never raises and never returns ``""``.
    """
    too_long = f"ERROR: expression longer than {MAX_EXPRESSION_CHARS:,} characters; split it up."
    raw = str(expression)
    if len(raw) > 4 * MAX_EXPRESSION_CHARS:  # before any cleaning, whatever it would strip
        return too_long
    text = _clean(raw)
    if not text:
        return "ERROR: empty expression; send arithmetic such as 2 + 2."
    if len(text) > MAX_EXPRESSION_CHARS:
        return too_long
    try:
        return _evaluate(text)
    except SyntaxError as exc:
        hint = (
            _COMMAS
            if _THOUSANDS.search(text)
            else "use Python syntax such as 1836.6 * 0.15 or sum(range(10, 94))"
        )
        return f"ERROR: not a valid Python expression ({exc.msg}); {hint}."
    except ZeroDivisionError:
        return "ERROR: division by zero."
    except OverflowError:
        return "ERROR: the result is too large to represent as a number."
    except RecursionError:
        return "ERROR: the expression is too deeply nested."
    except MemoryError:
        return "ERROR: the calculation ran out of memory."
    except (ValueError, TypeError, IndexError, KeyError) as exc:
        return f"ERROR: {_detail(exc)}."
    except Exception as exc:  # the tool boundary: report, never raise
        return f"ERROR: the calculation failed ({type(exc).__name__}: {_cut(str(exc))})."


@tool
def calculator(expression: str) -> str:
    """Evaluate one Python expression and return its value, e.g. '1836.6 * 0.15',
    'sum(range(10, 94))' or 'sum(n for n in range(20, 81) if n % 3 == 0)'. Built-in functions,
    comprehensions, comparisons and math/statistics functions work; use ** for powers, not ^. It
    cannot use variables, statements, imports or files: use run_python for those."""
    # The last example keeps both bounds and filters: with 'len([n for n in range(1, 100) if
    # n % 7 == 0])' instead, qwen3-4b wrote "multiples of k between a and b" as
    # sum(range(a, b, k)), which is wrong unless a is a multiple of k (7 of 12 such questions
    # right, against 11 of 12 with this one; first-step replays at temperature 0).
    return calculate(expression)
