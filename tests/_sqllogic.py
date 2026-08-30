"""A small sqllogictest runner that puts the engine's own tests through the bridge.

DuckDB's `test/sql/**/*.test` files are the engine's statement of what it
accepts. Running them here checks, at the engine's own standard, that a
statement the engine accepts passes through `sql()` and `run()` unchanged.

The format, as far as this runner reads it:

    statement ok            the SQL that follows must succeed
    statement error         it must fail; a message after ---- is a substring
    query <types> [mode]    the SQL must succeed and match the rows after ----
    require ...             a precondition; only built-in extensions are met
    loop / foreach / mode / load / restart / ...   not read: the file is skipped

Rows are compared after the mode's sort, value by value, with every value
turned into text the way the engine's runner prints it.
"""

from __future__ import annotations

import datetime
import decimal
import math
import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

import duckdb
from duckdb import exceptions, sql

if TYPE_CHECKING:
    from collections.abc import Iterator

#: Extensions the engine bundle carries, so a `require` of one is met.
BUILT_IN = {"parquet", "json", "icu", "tpch", "core_functions"}
#: Directives this runner does not read. A file using one is skipped whole.
UNSUPPORTED = {
    "loop",
    "foreach",
    "concurrentloop",
    "mode",
    "load",
    "restart",
    "reconnect",
    "halt",
    "sleep",
    "set",
    "unzip",
    "endloop",
    "concurrentforeach",
}
#: Test-harness pragmas: verification modes of the engine's own runner. Not
#: statements this engine build accepts, and not statements a user would run.
IGNORED = re.compile(
    r"^\s*pragma\s+((enable|disable)_verification|verify_[a-z_]+|disable_verify_[a-z_]+)\s*;?\s*$", re.IGNORECASE
)
HASHED = re.compile(r"^\d+ values hashing to [0-9a-f]+$")
#: The engine's names for the types whose values compare as numbers.
NUMERIC_TYPES = {
    "TINYINT",
    "SMALLINT",
    "INTEGER",
    "BIGINT",
    "HUGEINT",
    "UTINYINT",
    "USMALLINT",
    "UINTEGER",
    "UBIGINT",
    "UHUGEINT",
    "FLOAT",
    "DOUBLE",
    "DECIMAL",
    "BIGNUM",
}
#: The spellings of a boolean the engine's runner accepts.
BOOLEAN_TEXT = {"true": True, "1": True, "false": False, "0": False}
#: The type a `query` line's letter stands for, when the binder gave none.
_LETTER_TYPES = {"I": "BIGINT", "R": "DOUBLE"}


@dataclass
class Outcome:
    """What happened to one file."""

    path: Path
    skipped: str | None = None
    statements: int = 0
    queries: int = 0
    compared: int = 0
    carried: int = 0
    matched: int = 0
    not_carried: list[str] = field(default_factory=list)
    mismatched: list[str] = field(default_factory=list)


@dataclass
class Record:
    kind: str
    header: list[str]
    sql: str
    expected: list[str]


def records(text: str) -> Iterator[Record]:
    """The records of a test file, in order. Raises on an unread directive."""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        words = line.split()
        kind = words[0]
        if kind in UNSUPPORTED:
            raise LookupError(kind)
        if kind == "hash-threshold":
            i += 1
            continue
        if kind == "require":
            if words[1] not in BUILT_IN or len(words) > 2:
                unmet = f"require {' '.join(words[1:])}"
                raise LookupError(unmet)
            i += 1
            continue
        if kind not in {"statement", "query"}:
            raise LookupError(kind)
        i += 1
        body: list[str] = []
        while i < len(lines) and lines[i].strip() and lines[i].strip() != "----":
            body.append(lines[i])
            i += 1
        expected: list[str] = []
        if i < len(lines) and lines[i].strip() == "----":
            i += 1
            while i < len(lines) and lines[i].strip():
                expected.append(lines[i])
                i += 1
        yield Record(kind, words[1:], "\n".join(body), expected)


def as_text(value: object, type_text: str = "") -> str:
    """One value as the engine's runner prints it, given the column's SQL type.

    The type decides what a dict is: a STRUCT prints as `{'k': v}`, a MAP as
    `{k=v}`. Strings inside a nested value print unquoted, as the engine's
    own VARCHAR cast prints them.
    """
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return repr(value) if value != int(value) or abs(value) >= 1e16 else f"{value:.1f}"
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return "".join(chr(b) if 32 <= b < 127 and b != 92 else f"\\x{b:02X}" for b in value)
    if isinstance(value, datetime.timedelta):
        return _interval_text(value)
    if isinstance(value, datetime.datetime):
        return str(value).rstrip("0").rstrip(".") if value.microsecond else str(value)
    if isinstance(value, datetime.time):
        return str(value).rstrip("0").rstrip(".") if value.microsecond else str(value)
    # The type is read outside-in: `MAP(INTEGER, VARCHAR)[]` is a list of
    # maps, and only the element type says what each of its dicts is.
    upper = type_text.strip().upper()
    if isinstance(value, list) and _list_element(upper) is not None:
        return "[" + ", ".join(_nested_text(v, _list_element(upper) or "") for v in value) + "]"
    if upper.startswith("MAP("):
        # A MAP arrives as a dict, or as (key, value) pairs when its keys are
        # lists or structs, which Python cannot hash.
        key_type, value_type = _map_types(upper)
        entries = value.items() if isinstance(value, dict) else cast("list[tuple[object, object]]", value)
        return "{" + ", ".join(f"{_nested_text(k, key_type)}={_nested_text(v, value_type)}" for k, v in entries) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(_nested_text(v, "") for v in value) + "]"
    if isinstance(value, dict):
        fields = _struct_fields(upper)
        return (
            "{"
            + ", ".join(f"{_quoted(k)}: {_nested_text(v, fields.get(k.upper(), ''))}" for k, v in value.items())
            + "}"
        )
    return str(value)


#: Characters that make the engine quote a value inside a nested one.
_QUOTED_ON = set("\"'(),:=[]{}")


def _nested_text(value: object, type_text: str) -> str:
    """A value inside a list, struct or map, quoted the way the engine's own cast quotes it.

    The engine quotes the text of any leaf that is empty, starts or ends
    with a space, reads as `null`, or holds one of `" ' ( ) , : = [ ] { }`,
    whatever its type: a TIME is quoted for its colons, a DATE is not. A
    quote or a backslash inside is escaped with a backslash.
    """
    if value is None or isinstance(value, (list, dict)):
        return as_text(value, type_text)
    text = as_text(value, type_text)
    needs_quotes = (
        not text
        or text[0].isspace()
        or (len(text) >= 2 and text[-1].isspace())
        or text.lower() == "null"
        or any(c in _QUOTED_ON for c in text)
    )
    return _quoted(text) if needs_quotes else text


def _quoted(text: str) -> str:
    """Text in single quotes, escaped as the engine escapes it."""
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _split_top(text: str) -> list[str]:
    """Split on the commas outside parentheses and quotes: the members of a STRUCT(...) or MAP(...)."""
    parts: list[str] = []
    depth = 0
    quote = ""
    start = 0
    for i, c in enumerate(text):
        if quote:
            if c == quote:
                quote = ""
        elif c in "\"'":
            quote = c
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:].strip())
    return [p for p in parts if p]


def _list_element(upper: str) -> str | None:
    """The element type of `T[]` or `T[n]`, or None for a type that is not a list."""
    if not upper.endswith("]"):
        return None
    return upper[: upper.rindex("[")]


def _map_types(upper: str) -> tuple[str, str]:
    """The key and value types of `MAP(K, V)`."""
    inner = _split_top(upper[len("MAP(") : -1])
    return (inner[0], inner[1]) if len(inner) == 2 else ("", "")


def _struct_fields(upper: str) -> dict[str, str]:
    """The fields of `STRUCT(name T, ...)` by upper-cased name; empty for anything else."""
    if not upper.startswith("STRUCT("):
        return {}
    fields: dict[str, str] = {}
    for part in _split_top(upper[len("STRUCT(") : -1]):
        if part.startswith('"'):
            end = part.index('"', 1)
            fields[part[1:end]] = part[end + 1 :].strip()
        else:
            name, _, type_text = part.partition(" ")
            fields[name] = type_text.strip()
    return fields


def _interval_text(value: datetime.timedelta) -> str:
    """An interval the way the engine prints one: `1 day 00:00:01.5`, `-1 day -01:00:00`.

    A timedelta is normalised to a non-negative clock (`-1 hour` arrives as
    `-1 day, 23:00:00`), so it is undone here and the sign put on both parts,
    which is how the engine writes a negative interval. What a timedelta
    cannot carry is a mixed sign, `1 day -01:00:00`, or months: the engine
    prints both and this runner cannot, and such rows compare as mismatches.
    """
    total = (value.days * 86400 + value.seconds) * 1_000_000 + value.microseconds
    sign = "-" if total < 0 else ""
    days, rest = divmod(abs(total), 86400 * 1_000_000)
    seconds, micro = divmod(rest, 1_000_000)
    parts: list[str] = []
    if days:
        parts.append(f"{sign}{days} day{'s' if days != 1 else ''}")
    if rest or not parts:
        clock = f"{sign}{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
        if micro:
            clock += f".{micro:06d}".rstrip("0")
        parts.append(clock)
    return " ".join(parts)


def close_enough(expected: str, actual: str, column_type: str) -> bool:
    """Whether one expected value matches, the way the engine's runner decides it.

    Equal text matches. A `<REGEX>:` or `<!REGEX>:` expectation is a full
    match. Otherwise the column's type decides: a numeric column compares as
    numbers, so `10` matches `10.0` and a double matches to six places; a
    boolean column takes `1` and `0` for `true` and `false`; any other
    column, text included, matches only as text. The letter in the file's
    `query` line plays no part, and neither does it in the engine's runner.
    """
    if expected == actual:
        return True
    if expected.startswith(("<REGEX>:", "<!REGEX>:")):
        return matches_regex(actual, expected)
    kind = _type_head(column_type)
    if kind == "BOOLEAN":
        # A value outside the table, a list of booleans or garbage, matches
        # nothing; the text rule already said they differ.
        known = BOOLEAN_TEXT.get(expected.lower())
        return known is not None and known == BOOLEAN_TEXT.get(actual.lower())
    if kind in NUMERIC_TYPES:
        if expected.upper() == "NULL" or actual.upper() == "NULL":
            return expected.upper() == actual.upper()
        if expected.lower() == actual.lower() and expected.lower() in {"nan", "-nan", "inf", "-inf"}:
            return True
        try:
            if kind in {"FLOAT", "DOUBLE"}:
                return math.isclose(float(expected), float(actual), rel_tol=1e-6, abs_tol=1e-9)
            return decimal.Decimal(expected) == decimal.Decimal(actual)
        except (ValueError, decimal.InvalidOperation):
            return False
    return False


def _type_head(type_text: str) -> str:
    """`DECIMAL` for `DECIMAL(18,3)`, `INTEGER` for `INTEGER[]`: the engine's type name without its arguments."""
    return re.split(r"[(\[]", type_text.strip().upper(), maxsplit=1)[0]


def matches_regex(text: str, expectation: str) -> bool:
    """Whether text matches a `<REGEX>:pattern` whole, or fails to match a `<!REGEX>:pattern`."""
    want = expectation.startswith("<REGEX>:")
    pattern = expectation.split(":", 1)[1]
    try:
        matched = re.fullmatch(pattern, text, re.DOTALL) is not None
    except re.error:
        return False
    return matched == want


def compare(record: Record, rows: list[tuple[object, ...]], column_types: list[str], expected: list[str]) -> bool:
    """Whether the rows match an expected block."""
    if len(expected) == 1 and HASHED.match(expected[0].strip()):
        return True  # a hashed result block; not checked, not held against the bridge
    types = record.header[0] if record.header else ""
    mode = record.header[1] if len(record.header) > 1 else ""
    width = len(types)
    actual = [
        [as_text(v, column_types[i] if i < len(column_types) else "") or "(empty)" for i, v in enumerate(row)]
        for row in rows
    ]
    expected_rows = [line.split("\t") for line in expected]
    if (
        width > 1
        and expected_rows
        and all(len(r) == 1 for r in expected_rows)
        and len(expected_rows) == len(actual) * width
    ):
        # The classic form: one value per line, rows read off in order.
        flat = [r[0] for r in expected_rows]
        expected_rows = [flat[i : i + width] for i in range(0, len(flat), width)]
    if mode in {"rowsort", "sort"}:
        actual.sort()
        expected_rows.sort()
    elif mode == "valuesort":
        actual = sorted([[v] for row in actual for v in row])
        expected_rows = sorted([[v] for row in expected_rows for v in row])
    if len(actual) != len(expected_rows):
        return False
    for got, want in zip(actual, expected_rows, strict=True):
        if len(got) != len(want):
            # One value per line is the classic form for one-column results.
            return False
        for i, (g, w) in enumerate(zip(got, want, strict=True)):
            # The engine's types where the binder gave them; the file's
            # letters stand in when it did not (I and R are numeric).
            column_type = column_types[i] if i < len(column_types) else _LETTER_TYPES.get(types[i : i + 1], "")
            if not close_enough(w, g, column_type):
                return False
    return True


def placeholders(text: str, tmp: Path, cwd: Path, test_name: str) -> str:
    """The file's placeholders resolved as the engine's runner resolves them.

    `{TEST_DIR}` and the older `__TEST_DIR__` are a scratch directory,
    `__WORKING_DIRECTORY__` is the checkout the file's data lives under,
    `{UUID}` is fresh each time, and `{TEST_NAME}` / `{BASE_TEST_NAME}` name
    the file.
    """
    return (
        text.replace("{TEST_DIR}", str(tmp))
        .replace("__TEST_DIR__", str(tmp))
        .replace("__WORKING_DIRECTORY__", str(cwd))
        .replace("{UUID}", str(uuid.uuid4()))
        .replace("{TEST_NAME}", test_name)
        .replace("{BASE_TEST_NAME}", test_name.replace("/", "_"))
    )


def error_matches(message: str, expected: list[str]) -> bool:
    """Whether a failed statement's message is the one its file expects.

    As the engine's runner reads it: the expected text is a substring of the
    message, or a `<REGEX>:` / `<!REGEX>:` line is matched against the whole
    message. An empty expectation is met by any error.
    """
    wanted = "\n".join(line for line in expected if line.strip()).strip()
    if not wanted:
        return True
    if wanted.startswith(("<REGEX>:", "<!REGEX>:")):
        return matches_regex(message, wanted)
    return wanted in message


def run_file(path: Path, tmp: Path, cwd: Path | None = None) -> Outcome:
    """Run one file on a fresh in-memory database.

    `cwd` is the engine checkout: the files name their data relative to it.
    """
    outcome = Outcome(path)
    try:
        parsed = list(records(path.read_text()))
    except LookupError as directive:
        outcome.skipped = str(directive)
        return outcome
    con = duckdb.connect()
    previous = Path.cwd()
    if cwd is not None:
        os.chdir(cwd)
    labelled: dict[str, list[str]] = {}
    test_name = str(path.relative_to(cwd)) if cwd is not None and path.is_relative_to(cwd) else path.name
    try:
        for record in parsed:
            text = placeholders(record.sql, tmp, cwd or previous, test_name)
            if "${" in text or "{DATA_DIR}" in text:
                outcome.skipped = "a ${variable} the runner does not read"
                return outcome
            if IGNORED.match(text):
                continue
            if record.kind == "statement":
                outcome.statements += 1
                expect_error = record.header[:1] == ["error"]
                try:
                    # A record may hold several statements; run() takes one.
                    for statement in text.split(";\n") if not expect_error else [text]:
                        if statement.strip():
                            con.run(statement)
                    failed = False
                except exceptions.Error as error:
                    failed = True
                    message = str(error)
                except Exception as error:  # a crash in the client is a finding, not a skip
                    outcome.not_carried.append(f"statement CRASHED: {text[:80]!r} -> {type(error).__name__}: {error}")
                    if expect_error:
                        continue
                    return outcome
                if failed == expect_error:
                    outcome.carried += 1
                    if expect_error and not error_matches(message, record.expected):
                        outcome.mismatched.append(f"error text: wanted {' '.join(record.expected)[:60]!r}")
                else:
                    outcome.not_carried.append(f"{record.kind} {' '.join(record.header)}: {text[:80]!r}")
                    if expect_error:
                        continue
                    return outcome  # the file's later records depend on this one
            else:
                outcome.queries += 1
                plan = sql(text)
                try:
                    # Types first, from the binder: they decide how a nested
                    # value prints. Binding has no side effects; running twice
                    # would, and some queries here advance sequences.
                    try:
                        column_types = plan.types(con)
                    except exceptions.Error:
                        column_types = []
                    rows = plan.rows(con)
                except exceptions.Error as error:
                    outcome.not_carried.append(f"query: {text[:80]!r} -> {str(error).splitlines()[0][:80]}")
                    continue
                except Exception as error:  # a crash in the client is a finding, not a skip
                    outcome.not_carried.append(f"query CRASHED: {text[:80]!r} -> {type(error).__name__}: {error}")
                    continue
                outcome.carried += 1
                if text.lstrip().upper().startswith("EXPLAIN"):
                    continue  # plan text; this client renders plans differently from the engine's runner
                label = record.header[2] if len(record.header) > 2 else None
                expected = record.expected
                try:
                    if label is not None and not expected:
                        # A labelled result: the first query stores it, later
                        # ones must reproduce it.
                        stored = labelled.get(label)
                        if stored is None:
                            labelled[label] = [
                                "\t".join(
                                    as_text(v, column_types[i] if i < len(column_types) else "")
                                    for i, v in enumerate(row)
                                )
                                for row in rows
                            ]
                            continue
                        expected = stored
                    outcome.compared += 1
                    matched = compare(record, rows, column_types, expected)
                except Exception as error:  # the runner's own formatting failed: a runner bug, recorded as one
                    outcome.mismatched.append(f"query RUNNER CRASHED: {text[:80]!r} -> {type(error).__name__}: {error}")
                    continue
                if matched:
                    outcome.matched += 1
                else:
                    outcome.mismatched.append(f"query: {text[:80]!r}")
    finally:
        con.close()
        os.chdir(previous)
    return outcome
