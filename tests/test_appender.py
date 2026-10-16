"""The appender: rows buffered on the client and written to the database by one statement at each flush."""

from __future__ import annotations

import collections.abc
import contextlib
import gc
import re
import subprocess
import sys
import textwrap
import threading
import warnings
from typing import TYPE_CHECKING, Any, Literal

import pytest

import duckdb
from duckdb import dbapi, exceptions, frame

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from duckdb.frame.typing import PythonValue

NEEDS_TABLE_LOOKUP_FIX = pytest.mark.xfail(
    strict=True,
    reason="waits for DuckDB to refuse a view or an ambiguous table name without aborting the open transaction "
    "(core branch fix-tableinfo-view-entry)",
)
NEEDS_BUFFER_FIRST = pytest.mark.xfail(
    strict=True, reason="waits for DuckDB to place the buffer before the statement's own common table expressions"
)

REFUSED = ValueError
WRONG_ARITY = ValueError
CONCURRENT_USE = exceptions.OperationalError
BUSY = exceptions.OperationalError
CLOSED = exceptions.InterfaceError
BROKEN = exceptions.InterfaceError

Action = Literal["insert", "upsert", "update", "delete"]

LIVE_RESULT = "live result; finish reading it or close it"
QUERY_RUNS = "cannot be used while its query runs"
KV = {"k": "INTEGER", "v": "VARCHAR"}
VECTOR = 2048
# The engine's own appenders flush by themselves once they hold this many rows.
ENGINE_FLUSH_COUNT = 100 * VECTOR
# A read of fewer rows than one fetch returns finishes its result, which then no longer holds the connection.
OPEN_ROWS = 10_000
REST_OF_OPEN_RESULT = [(i,) for i in range(1, OPEN_ROWS)]


@pytest.fixture
def con() -> Iterator[frame.Connection]:
    connection = frame.connect()
    connection.run("CREATE TABLE t (k INTEGER PRIMARY KEY, v VARCHAR)")
    yield connection
    connection.close()


@pytest.fixture
def produced(con: frame.Connection) -> list[int]:
    count = [0]

    def tally(i: int) -> int:
        count[0] += 1
        return i

    con.create_function("tally", tally, ["BIGINT"], "BIGINT", stability="volatile")
    return count


@pytest.fixture
def held(con: frame.Connection) -> tuple[threading.Event, threading.Event]:
    running, release = threading.Event(), threading.Event()

    def hold(x: int) -> int:
        running.set()
        release.wait(10)
        return x

    con.create_function("hold", hold, ["BIGINT"], "BIGINT")
    return running, release


@contextlib.contextmanager
def refused(error: type[Exception], match: str | None = None) -> Iterator[None]:
    """Expect exactly `error`: neo's error classes nest, so `pytest.raises` alone would also accept a subclass."""
    with pytest.raises(error, match=match) as caught:
        yield
    assert type(caught.value) is error, f"{type(caught.value).__name__}: {caught.value}"
    # Its traceback holds this frame and the test's, a cycle that would keep the test's appender, and the warning
    # for its rows, for whichever later test collects garbage.
    del caught


def discard(app: frame.Appender) -> None:
    # Collecting an appender that still holds rows warns, and this suite turns every warning into an error.
    app.clear()


def run_all(con: frame.Connection, *statements: str) -> None:
    for statement in statements:
        con.run(statement)


def table_rows(con: frame.Connection, table: str = "t") -> list[tuple[Any, ...]]:
    return frame.sql(f"SELECT * FROM {table} ORDER BY ALL").rows(con)


def open_result(con: frame.Connection) -> Iterator[tuple[Any, ...]]:
    rows = frame.sql(f"SELECT * FROM range({OPEN_ROWS})").iter_rows(con)
    next(rows)
    return rows


def contend(
    running: threading.Event,
    release: threading.Event,
    call: Callable[[], object],
    outcome: list[object],
    refusal: type[BaseException],
) -> None:
    running.wait(10)
    try:
        call()
        outcome.append("ran")
    except refusal as error:
        outcome.append(error)
    finally:
        release.set()


def append_then_raise(app: frame.Appender, exception: type[BaseException]) -> None:
    app.append_row(1, "a")
    raise exception


def append_close_then_raise(app: frame.Appender, con: frame.Connection, exception: BaseException) -> None:
    app.append_row(1, "a")
    con.close()
    raise exception


def append_until_full(app: frame.Appender, row: tuple[PythonValue, ...], limit: int) -> None:
    for _ in range(limit):
        app.append_row(*row)


def run_isolated(script: str, timeout: float = 15) -> tuple[int, str]:
    # A child process keeps a deadlock from stalling the suite; the marker tells a hang from a slow start.
    try:
        proc = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as hang:
        started = hang.stdout or ""
        if "started" not in (started.decode() if isinstance(started, bytes) else started):
            msg = "the child never reached the statement under test"
            raise AssertionError(msg) from hang
        raise
    return proc.returncode, proc.stdout + proc.stderr


def attach_read_only(con: frame.Connection, path: Path) -> None:
    file = (path / "ro.db").as_posix()
    with frame.connect(file) as writer:
        writer.run("CREATE TABLE t (k INTEGER)")
    con.run(f"ATTACH '{file}' AS ro (READ_ONLY)")


def empty_transaction(con: frame.Connection) -> None:
    with con.transaction():
        pass


def appender_reading_prices(con: frame.Connection) -> frame.Appender:
    return con.query_appender(
        "INSERT INTO priced SELECT b.k, p.price FROM appender_data b JOIN prices p USING (k)", {"k": "INTEGER"}
    )


def held_flush_appender(con: frame.Connection) -> frame.Appender:
    return con.query_appender("INSERT INTO t SELECT hold(k), v FROM appender_data", KV)


#: A child that interrupts a flush held inside a Python function, running `before` and `after` around it.
INTERRUPTED_FLUSH = """
import threading
from duckdb import exceptions, frame
con = frame.connect()
con.run("CREATE TABLE t (k INTEGER PRIMARY KEY, v VARCHAR)")
running, release = threading.Event(), threading.Event()
def hold(x):
    running.set()
    release.wait(10)
    return x
con.create_function("hold", hold, ["BIGINT"], "BIGINT")
app = con.query_appender("INSERT INTO t SELECT hold(k), v FROM appender_data", {kv!r})
def interrupt_while_held():
    running.wait(10)
    con.interrupt()
    release.set()
{before}
app.append_row(1, "a")
interrupter = threading.Thread(target=interrupt_while_held)
interrupter.start()
print("started", flush=True)
try:
    app.flush()
    print("flush: returned")
except exceptions.Error as error:
    print("flush:", type(error).__name__)
interrupter.join()
{after}
print("rows:", frame.sql("SELECT * FROM t ORDER BY ALL").rows({reader}))
"""


def interrupted_flush(before: tuple[str, ...] = (), after: tuple[str, ...] = (), reader: str = "con") -> str:
    # In a child, because a flush the interrupt fails to end would otherwise hang the whole suite.
    code, output = run_isolated(
        INTERRUPTED_FLUSH.format(kv=KV, before="\n".join(before), after="\n".join(after), reader=reader), timeout=60
    )
    assert code == 0, output
    assert "flush: InterruptError\n" in output
    return output


class ShrinksAfterRead(collections.abc.Mapping[str, str]):
    """Columns x and y to the first iteration, and only x to every later one."""

    def __init__(self) -> None:
        self.shown = {"x": "INTEGER", "y": "BIGINT"}

    def __iter__(self) -> Iterator[str]:
        names = list(self.shown)
        self.shown = {"x": "INTEGER"}
        return iter(names)

    def __getitem__(self, name: str) -> str:
        return {"x": "INTEGER", "y": "BIGINT"}[name]

    def __len__(self) -> int:
        return len(self.shown)


class GivesANameTwice(collections.abc.Mapping[str, str]):
    """Column k, given twice by its iteration."""

    def __iter__(self) -> Iterator[str]:
        return iter(["k", "k"])

    def __getitem__(self, name: str) -> str:
        return "INTEGER"

    def __len__(self) -> int:
        return 2


REFUSED_TABLE_CASES: list[
    tuple[Callable[[frame.Connection, Path], object], str | tuple[str, ...], type[Exception], str]
] = [
    (lambda con, path: None, "missing", exceptions.CatalogError, "missing"),
    (lambda con, path: con.run("CREATE VIEW v AS SELECT 1 AS k"), "v", exceptions.CatalogError, "view"),
    (attach_read_only, ("ro", "t"), exceptions.InvalidInputError, "read_only"),
    (
        lambda con, path: run_all(
            con, "ATTACH ':memory:' AS s", "CREATE SCHEMA s", "CREATE TABLE memory.s.t (k INTEGER)"
        ),
        ("s", "t"),
        exceptions.ProgrammingError,
        "ambiguous",
    ),
]


def refused_tables(*, waiting: tuple[str, ...] = ()) -> pytest.MarkDecorator:
    return pytest.mark.parametrize(
        ("setup", "name", "error"),
        [
            pytest.param(setup, name, error, id=case, marks=NEEDS_TABLE_LOOKUP_FIX if case in waiting else ())
            for setup, name, error, case in REFUSED_TABLE_CASES
        ],
    )


class TestDefaultInsert:
    def test_rows_go_into_the_table(self, con: frame.Connection) -> None:
        app = con.appender("t")
        app.append_row(1, "a")
        app.append_row(2, None)
        app.flush()
        assert table_rows(con) == [(1, "a"), (2, None)]

    @pytest.mark.parametrize(
        ("setup", "name", "target"),
        [
            pytest.param(
                ("CREATE SCHEMA s", "CREATE TABLE s.t (k INTEGER, v VARCHAR)", "SET search_path = 's'"),
                "t",
                "s.t",
                id="search_path",
            ),
            pytest.param(
                ("CREATE SCHEMA s", "CREATE TABLE s.t (k INTEGER, v VARCHAR)"), ("s", "t"), "s.t", id="schema"
            ),
            pytest.param(
                ("ATTACH ':memory:' AS other", "CREATE TABLE other.t (k INTEGER, v VARCHAR)"),
                ("other", "t"),
                "other.t",
                id="catalog",
            ),
            pytest.param(
                (
                    "ATTACH ':memory:' AS other",
                    "CREATE SCHEMA other.s",
                    "CREATE TABLE other.s.t (k INTEGER, v VARCHAR)",
                ),
                ("other", "s", "t"),
                "other.s.t",
                id="catalog_and_schema",
            ),
            pytest.param(("CREATE TEMP TABLE t (k INTEGER, v VARCHAR)",), "t", "temp.t", id="temp_shadows_main"),
            pytest.param(("CREATE TABLE Mixed (k INTEGER, v VARCHAR)",), "MIXED", "mixed", id="case_insensitive"),
            pytest.param(('CREATE TABLE "a.b" (k INTEGER, v VARCHAR)',), "a.b", '"a.b"', id="dotted_string"),
            pytest.param(
                ("CREATE SCHEMA s", 'CREATE TABLE s."a.b" (k INTEGER, v VARCHAR)'),
                ("s", "a.b"),
                's."a.b"',
                id="dotted_part",
            ),
        ],
    )
    def test_the_name_resolves_as_sql_resolves_it(
        self, con: frame.Connection, setup: tuple[str, ...], name: str | tuple[str, ...], target: str
    ) -> None:
        run_all(con, *setup)
        app = con.appender(name)
        app.append_row(1, "a")
        app.flush()
        assert table_rows(con, target) == [(1, "a")]

    def test_the_name_resolves_once_at_creation(self, con: frame.Connection) -> None:
        run_all(con, "CREATE SCHEMA s", "CREATE TABLE s.t (k INTEGER, v VARCHAR)")
        app = con.appender("t")
        con.run("SET search_path = 's'")
        app.append_row(1, "a")
        app.flush()
        assert table_rows(con, "main.t") == [(1, "a")]

    def test_a_table_created_in_the_open_transaction_resolves(self, con: frame.Connection) -> None:
        con.run("BEGIN")
        con.run("CREATE TABLE u (k INTEGER)")
        app = con.appender("u")
        app.append_row(1)
        app.flush()
        con.run("COMMIT")
        assert table_rows(con, "u") == [(1,)]

    @refused_tables()
    def test_a_name_that_is_not_a_writable_table_is_refused(
        self,
        con: frame.Connection,
        tmp_path: Path,
        setup: Callable[[frame.Connection, Path], object],
        name: str | tuple[str, ...],
        error: type[Exception],
    ) -> None:
        setup(con, tmp_path)
        with refused(error):
            con.appender(name)

    @refused_tables(waiting=("view", "ambiguous"))
    def test_a_refused_name_leaves_the_open_transaction_usable(
        self,
        con: frame.Connection,
        tmp_path: Path,
        setup: Callable[[frame.Connection, Path], object],
        name: str | tuple[str, ...],
        error: type[Exception],
    ) -> None:
        setup(con, tmp_path)
        con.run("BEGIN")
        con.run("INSERT INTO t VALUES (1, 'a')")
        with refused(error):
            con.appender(name)
        con.run("INSERT INTO t VALUES (2, 'b')")
        con.run("COMMIT")
        assert table_rows(con) == [(1, "a"), (2, "b")]

    def test_a_dotted_string_is_one_name_and_never_parsed(self, con: frame.Connection) -> None:
        with refused(exceptions.CatalogError):
            con.appender("main.t")

    @pytest.mark.parametrize(
        ("name", "error"),
        [
            pytest.param("", ValueError, id="empty"),
            pytest.param(("main", ""), ValueError, id="empty_part"),
            pytest.param(["main", "t"], TypeError, id="a_list"),
            pytest.param(("memory", "main", "t", "u"), ValueError, id="four_parts"),
        ],
    )
    def test_a_malformed_name_is_refused(self, con: frame.Connection, name: object, error: type[Exception]) -> None:
        with refused(error):
            con.appender(name)  # type: ignore[arg-type]

    def test_generated_columns_are_not_in_the_buffer(self, con: frame.Connection) -> None:
        con.run("CREATE TABLE g (a INTEGER, b INTEGER GENERATED ALWAYS AS (a * 2), c VARCHAR)")
        app = con.appender("g")
        app.append_row(1, "x")
        app.flush()
        assert table_rows(con, "g") == [(1, 2, "x")]

    def test_a_value_for_a_generated_column_is_one_too_many(self, con: frame.Connection) -> None:
        con.run("CREATE TABLE g (a INTEGER, b INTEGER GENERATED ALWAYS AS (a * 2), c VARCHAR)")
        app = con.appender("g")
        with refused(WRONG_ARITY):
            app.append_row(1, 2, "x")

    def test_a_null_for_a_column_with_a_default_stays_null(self, con: frame.Connection) -> None:
        run_all(
            con,
            "CREATE SEQUENCE seq",
            "CREATE TABLE d (id INTEGER DEFAULT nextval('seq'), v VARCHAR DEFAULT 'none')",
        )
        app = con.appender("d")
        app.append_row(None, None)
        app.flush()
        assert table_rows(con, "d") == [(None, None)]

    def test_a_column_added_after_creation_takes_its_default(self, con: frame.Connection) -> None:
        con.run("CREATE TABLE p (k INTEGER)")
        app = con.appender("p")
        con.run("ALTER TABLE p ADD COLUMN w INTEGER DEFAULT 7")
        app.append_row(1)
        app.flush()
        assert table_rows(con, "p") == [(1, 7)]

    def test_a_column_subset_keeps_its_order_and_the_rest_take_their_defaults(self, con: frame.Connection) -> None:
        run_all(
            con,
            "CREATE SEQUENCE seq",
            "CREATE TABLE d (id INTEGER DEFAULT nextval('seq'), a VARCHAR, b VARCHAR, w INTEGER DEFAULT 7, n INTEGER)",
        )
        app = con.appender("d", columns=["b", "a"])
        app.append_rows([("b1", "a1"), ("b2", "a2")])
        app.flush()
        either_order = (
            [(1, "a1", "b1", 7, None), (2, "a2", "b2", 7, None)],
            [(1, "a2", "b2", 7, None), (2, "a1", "b1", 7, None)],
        )
        assert table_rows(con, "d") in either_order

    @pytest.mark.parametrize(
        "columns",
        [[], ["nope"], ["b"], ["a", "a"], ["a", "A"]],
        ids=["empty", "unknown", "generated", "repeated", "repeated_ignoring_case"],
    )
    def test_a_bad_column_subset_is_refused(self, con: frame.Connection, columns: list[str]) -> None:
        con.run("CREATE TABLE g (a INTEGER, b INTEGER GENERATED ALWAYS AS (a * 2), c VARCHAR)")
        with refused(REFUSED):
            con.appender("g", columns=columns)

    def test_subset_names_resolve_ignoring_case(self, con: frame.Connection) -> None:
        app = con.appender("t", columns=["V", "K"])
        app.append_row("a", 1)
        app.flush()
        assert table_rows(con) == [(1, "a")]

    def test_a_bare_string_is_one_column(self, con: frame.Connection) -> None:
        app = con.appender("t", columns="k")
        app.append_row(1)
        app.flush()
        assert table_rows(con) == [(1, None)]

    def test_the_columns_are_not_a_dict(self, con: frame.Connection) -> None:
        with pytest.raises(TypeError):
            con.appender("t", columns={"k": "INTEGER"})

    def test_the_columns_are_read_once_at_creation(self, con: frame.Connection) -> None:
        con.run("CREATE TABLE t2 (a INTEGER, b INTEGER)")
        # A generator can be read once, so a second read would find no columns.
        app = con.appender("t2", columns=(name for name in ["a", "b"]))
        app.append_row(1, 2)
        app.flush()
        assert table_rows(con, "t2") == [(1, 2)]


class TestTableActions:
    def test_upsert_replaces_rows_on_the_unique_key(self, con: frame.Connection) -> None:
        con.run("INSERT INTO t VALUES (1, 'old'), (2, 'kept')")
        app = con.appender("t", action="upsert")
        app.append_rows([(1, "new"), (3, "added")])
        app.flush()
        assert table_rows(con) == [(1, "new"), (2, "kept"), (3, "added")]

    @pytest.mark.parametrize(
        "ddl",
        [
            pytest.param("CREATE TABLE plain (k INTEGER, v VARCHAR)", id="none"),
            pytest.param("CREATE TABLE plain (k INTEGER PRIMARY KEY, v VARCHAR UNIQUE)", id="several"),
        ],
    )
    def test_upsert_without_exactly_one_unique_constraint_is_refused_at_creation(
        self, con: frame.Connection, ddl: str
    ) -> None:
        con.run(ddl)
        with refused(exceptions.ProgrammingError):
            con.appender("plain", action="upsert")

    def test_upsert_of_a_column_subset_writes_only_those_columns(self, con: frame.Connection) -> None:
        con.run("CREATE TABLE d (k INTEGER PRIMARY KEY, a VARCHAR, b VARCHAR DEFAULT 'default')")
        con.run("INSERT INTO d VALUES (1, 'old', 'kept')")
        app = con.appender("d", action="upsert", columns=["k", "a"])
        app.append_rows([(1, "new"), (2, "added")])
        app.flush()
        assert table_rows(con, "d") == [(1, "new", "kept"), (2, "added", "default")]

    def test_an_upsert_that_leaves_out_the_key_is_accepted_and_its_flush_fails(self, con: frame.Connection) -> None:
        app = con.appender("t", action="upsert", columns=["v"])
        app.append_row("no key")
        with refused(exceptions.ConstraintError):
            app.flush()

    def test_an_upsert_of_only_the_key_inserts_new_keys_and_leaves_existing_rows_alone(
        self, con: frame.Connection
    ) -> None:
        con.run("INSERT INTO t VALUES (1, 'old')")
        app = con.appender("t", action="upsert", columns=["k"])
        app.append_rows([(1,), (2,)])
        app.flush()
        assert table_rows(con) == [(1, "old"), (2, None)]

    def test_an_upsert_matches_on_a_composite_primary_key(self, con: frame.Connection) -> None:
        con.run("CREATE TABLE cp (a INTEGER, b INTEGER, v VARCHAR, PRIMARY KEY (a, b))")
        con.run("INSERT INTO cp VALUES (1, 1, 'x'), (1, 2, 'y')")
        app = con.appender("cp", action="upsert")
        app.append_rows([(1, 2, "Y"), (2, 1, "new")])
        app.flush()
        assert table_rows(con, "cp") == [(1, 1, "x"), (1, 2, "Y"), (2, 1, "new")]

    def test_update_sets_every_other_buffered_column_on_rows_matching_the_key(self, con: frame.Connection) -> None:
        con.run("CREATE TABLE u (k INTEGER PRIMARY KEY, v VARCHAR, w INTEGER)")
        con.run("INSERT INTO u VALUES (1, 'a', 10), (2, 'b', 20)")
        app = con.appender("u", action="update", key=["k"])
        app.append_rows([(1, "A", 11), (3, "C", 33), (None, "N", 0)])
        app.flush()
        assert table_rows(con, "u") == [(1, "A", 11), (2, "b", 20)]

    def test_update_of_a_column_subset_leaves_the_other_columns_alone(self, con: frame.Connection) -> None:
        con.run("CREATE TABLE u (k INTEGER PRIMARY KEY, v VARCHAR, w INTEGER)")
        con.run("INSERT INTO u VALUES (1, 'a', 10), (2, 'b', 20)")
        app = con.appender("u", action="update", key=["k"], columns=["v", "k"])
        app.append_row("B", 2)
        app.flush()
        assert table_rows(con, "u") == [(1, "a", 10), (2, "B", 20)]

    def test_delete_removes_the_rows_matching_the_key(self, con: frame.Connection) -> None:
        con.run("CREATE TABLE pairs (a INTEGER, b INTEGER, note VARCHAR)")
        con.run("INSERT INTO pairs VALUES (1, 1, 'x'), (1, 2, 'y'), (2, 1, 'z')")
        app = con.appender("pairs", action="delete", key=["a", "b"])
        app.append_rows([(1, 2), (2, 2), (None, 1)])
        app.flush()
        assert table_rows(con, "pairs") == [(1, 1, "x"), (2, 1, "z")]

    def test_a_key_name_resolves_ignoring_case(self, con: frame.Connection) -> None:
        con.run("CREATE TABLE u (k INTEGER PRIMARY KEY, v VARCHAR)")
        con.run("INSERT INTO u VALUES (1, 'a')")
        app = con.appender("u", action="update", key=["K"])
        app.append_row(1, "A")
        app.flush()
        assert table_rows(con, "u") == [(1, "A")]

    def test_delete_buffers_only_the_key(self, con: frame.Connection) -> None:
        app = con.appender("t", action="delete", key=["k"])
        with refused(WRONG_ARITY):
            app.append_row(1, "a")

    def test_delete_buffers_its_key_in_the_order_given(self, con: frame.Connection) -> None:
        con.run("CREATE TABLE pairs (a INTEGER, b INTEGER, note VARCHAR)")
        con.run("INSERT INTO pairs VALUES (1, 1, 'x'), (1, 2, 'y'), (2, 1, 'z')")
        app = con.appender("pairs", action="delete", key=["b", "a"])
        app.append_row(2, 1)
        app.flush()
        assert table_rows(con, "pairs") == [(1, 1, "x"), (2, 1, "z")]

    @pytest.mark.parametrize(
        ("options", "reason"),
        [
            pytest.param({"key": ["k"]}, "key applies to the update and delete", id="key_with_insert"),
            pytest.param(
                {"action": "insert", "key": ["k"]},
                "key applies to the update and delete",
                id="key_with_explicit_insert",
            ),
            pytest.param(
                {"action": "upsert", "key": ["k"]}, "key applies to the update and delete", id="key_with_upsert"
            ),
            pytest.param({"action": "update"}, "update action needs key", id="update_without_key"),
            pytest.param({"action": "delete"}, "delete action needs key", id="delete_without_key"),
            pytest.param(
                {"action": "update", "key": ["k"], "columns": ["v"]},
                "not one of the appender's columns",
                id="key_outside_the_buffer",
            ),
            pytest.param({"action": "update", "key": ["nope"]}, "nope is not a column", id="key_not_a_column"),
            pytest.param(
                {"action": "update", "key": ["k"], "columns": ["k"]},
                "columns other than the key",
                id="update_of_nothing_but_the_key",
            ),
            pytest.param(
                {"action": "delete", "key": ["k"], "columns": ["k"]}, "takes no columns", id="delete_with_columns"
            ),
            pytest.param({"action": "merge"}, '"insert", "upsert", "update" or "delete"', id="unknown_action"),
        ],
    )
    def test_a_bad_action_is_refused_at_creation(
        self, con: frame.Connection, options: dict[str, Any], reason: str
    ) -> None:
        with refused(REFUSED, match=re.escape(reason)):
            con.appender("t", **options)

    @pytest.mark.parametrize("action", ["update", "delete"])
    @pytest.mark.parametrize(
        ("key", "reason"),
        [
            pytest.param([], "action needs key", id="empty"),
            pytest.param(["k", "k"], "key names k twice", id="repeated"),
            pytest.param(["k", "K"], "key names K twice", id="repeated_ignoring_case"),
        ],
    )
    def test_an_empty_or_repeated_key_is_refused(
        self, con: frame.Connection, action: Action, key: list[str], reason: str
    ) -> None:
        with refused(REFUSED, match=reason):
            con.appender("t", action=action, key=key)

    @pytest.mark.parametrize("action", ["update", "delete"])
    def test_a_generated_key_column_is_refused(self, con: frame.Connection, action: Action) -> None:
        con.run("CREATE TABLE g (a INTEGER, b INTEGER GENERATED ALWAYS AS (a * 2), c VARCHAR)")
        with refused(REFUSED, match="b is a generated column"):
            con.appender("g", action=action, key=["b"])

    def test_a_bare_string_key_is_one_column(self, con: frame.Connection) -> None:
        con.run("INSERT INTO t VALUES (1, 'a'), (2, 'b')")
        app = con.appender("t", action="delete", key="k")
        app.append_row(1)
        app.flush()
        assert table_rows(con) == [(2, "b")]

    def test_update_matches_on_every_key_column(self, con: frame.Connection) -> None:
        con.run("CREATE TABLE pairs (a INTEGER, b INTEGER, note VARCHAR)")
        con.run("INSERT INTO pairs VALUES (1, 1, 'x'), (1, 2, 'y'), (2, 1, 'z')")
        app = con.appender("pairs", action="update", key=["a", "b"])
        app.append_rows([(1, 2, "Y"), (2, 2, "nothing matches")])
        app.flush()
        assert table_rows(con, "pairs") == [(1, 1, "x"), (1, 2, "Y"), (2, 1, "z")]

    @pytest.mark.parametrize("name", ["appender_data", "appender_buffer", "appender_target", "excluded"])
    @pytest.mark.parametrize(
        ("action", "key", "rows", "expected"),
        [
            pytest.param(
                "upsert", None, [(1, "new"), (3, "added")], [(1, "new"), (2, "old"), (3, "added")], id="upsert"
            ),
            pytest.param("update", ["k"], [(1, "new"), (3, "missing")], [(1, "new"), (2, "old")], id="update"),
            pytest.param("delete", ["k"], [(1,)], [(2, "old")], id="delete"),
        ],
    )
    def test_each_action_works_on_a_table_named_like_a_name_the_statement_uses(
        self,
        con: frame.Connection,
        name: str,
        action: Action,
        key: list[str] | None,
        rows: list[tuple[PythonValue, ...]],
        expected: list[tuple[object, ...]],
    ) -> None:
        con.run(f"CREATE TABLE {name} (k INTEGER PRIMARY KEY, v VARCHAR)")
        con.run(f"INSERT INTO {name} VALUES (1, 'old'), (2, 'old')")
        app = con.appender(name, action=action, key=key)
        app.append_rows(rows)
        app.flush()
        assert table_rows(con, name) == expected

    @pytest.mark.parametrize(
        ("action", "key", "rows", "expected"),
        [
            pytest.param("upsert", None, [(1, "A"), (3, "c")], [(1, "A"), (2, "b"), (3, "c")], id="upsert"),
            pytest.param("update", ["select"], [(1, "A")], [(1, "A"), (2, "b")], id="update"),
            pytest.param("delete", ["select"], [(1,)], [(2, "b")], id="delete"),
        ],
    )
    def test_each_action_quotes_a_qualified_table_and_odd_column_names(
        self,
        con: frame.Connection,
        action: Action,
        key: list[str] | None,
        rows: list[tuple[PythonValue, ...]],
        expected: list[tuple[object, ...]],
    ) -> None:
        con.run('CREATE SCHEMA "odd schema"')
        con.run('CREATE TABLE "odd schema"."my table" ("select" INTEGER PRIMARY KEY, "two words" VARCHAR)')
        con.run("""INSERT INTO "odd schema"."my table" VALUES (1, 'a'), (2, 'b')""")
        app = con.appender(("odd schema", "my table"), action=action, key=key)
        app.append_rows(rows)
        app.flush()
        assert table_rows(con, '"odd schema"."my table"') == expected

    @pytest.mark.parametrize(
        ("action", "rows", "expected"),
        [
            pytest.param("update", [(1, "z")], [(1, "z"), (1, "z"), (2, "c")], id="update"),
            pytest.param("delete", [(1,)], [(2, "c")], id="delete"),
        ],
    )
    def test_a_key_that_matches_several_rows_acts_on_all_of_them(
        self,
        con: frame.Connection,
        action: Action,
        rows: list[tuple[PythonValue, ...]],
        expected: list[tuple[object, ...]],
    ) -> None:
        con.run("CREATE TABLE groups (g INTEGER, v VARCHAR)")
        con.run("INSERT INTO groups VALUES (1, 'a'), (1, 'b'), (2, 'c')")
        app = con.appender("groups", action=action, key=["g"])
        app.append_rows(rows)
        app.flush()
        assert table_rows(con, "groups") == expected

    @pytest.mark.parametrize(
        ("action", "key"), [pytest.param("upsert", None, id="upsert"), pytest.param("update", ["k"], id="update")]
    )
    def test_a_generated_column_follows_the_columns_an_action_writes(
        self, con: frame.Connection, action: Action, key: list[str] | None
    ) -> None:
        con.run("CREATE TABLE gen (k INTEGER PRIMARY KEY, a INTEGER, d INTEGER GENERATED ALWAYS AS (a * 2))")
        con.run("INSERT INTO gen (k, a) VALUES (1, 10)")
        app = con.appender("gen", action=action, key=key)
        app.append_row(1, 11)
        app.flush()
        assert table_rows(con, "gen") == [(1, 11, 22)]

    def test_an_upsert_refused_by_the_bind_leaves_the_open_transaction_usable(self, con: frame.Connection) -> None:
        con.run("CREATE TABLE plain (k INTEGER, v VARCHAR)")
        con.run("BEGIN")
        con.run("INSERT INTO t VALUES (1, 'a')")
        with refused(exceptions.ProgrammingError):
            con.appender("plain", action="upsert")
        con.run("INSERT INTO t VALUES (2, 'b')")
        con.run("COMMIT")
        assert table_rows(con) == [(1, "a"), (2, "b")]

    def test_the_key_is_read_once_at_creation(self, con: frame.Connection) -> None:
        con.run("CREATE TABLE u (k INTEGER PRIMARY KEY, v VARCHAR)")
        con.run("INSERT INTO u VALUES (1, 'a')")
        # A generator can be read once, so a second read would find no key.
        app = con.appender("u", action="update", key=(name for name in ["k"]))
        app.append_row(1, "A")
        app.flush()
        assert table_rows(con, "u") == [(1, "A")]

    @pytest.mark.parametrize("action", ["update", "delete"])
    def test_the_key_is_not_a_dict(self, con: frame.Connection, action: Action) -> None:
        with refused(TypeError):
            con.appender("t", action=action, key={"k": 1})

    @pytest.mark.parametrize("action", [1, None, ["insert"]], ids=["int", "none", "list"])
    def test_an_action_that_is_not_a_string_is_refused(self, con: frame.Connection, action: object) -> None:
        with refused(TypeError, match=re.escape('"insert", "upsert", "update" or "delete"')):
            con.appender("t", action=action)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("action", "key", "rows", "expected"),
        [
            pytest.param("INSERT", None, [(3, "c")], [(1, "a"), (2, "b"), (3, "c")], id="INSERT"),
            pytest.param("Upsert", None, [(1, "A"), (3, "c")], [(1, "A"), (2, "b"), (3, "c")], id="Upsert"),
            pytest.param("UPDATE", ["k"], [(2, "B")], [(1, "a"), (2, "B")], id="UPDATE"),
            pytest.param("dElEtE", ["k"], [(1,)], [(2, "b")], id="dElEtE"),
        ],
    )
    def test_an_action_word_ignores_the_case_of_its_letters(
        self,
        con: frame.Connection,
        action: str,
        key: list[str] | None,
        rows: list[tuple[PythonValue, ...]],
        expected: list[tuple[object, ...]],
    ) -> None:
        con.run("INSERT INTO t VALUES (1, 'a'), (2, 'b')")
        app = con.appender("t", action=action, key=key)  # type: ignore[arg-type]
        app.append_rows(rows)
        app.flush()
        assert table_rows(con) == expected

    @pytest.mark.parametrize("action", ["update", "delete"])
    def test_a_null_in_either_column_of_a_two_column_key_matches_nothing(
        self, con: frame.Connection, action: Action
    ) -> None:
        con.run("CREATE TABLE pairs (a INTEGER, b INTEGER, note VARCHAR)")
        con.run("INSERT INTO pairs VALUES (1, NULL, 'x'), (NULL, 1, 'y'), (NULL, NULL, 'z'), (1, 1, 'w')")
        app = con.appender("pairs", action=action, key=["a", "b"])
        keys: list[tuple[PythonValue, ...]] = [(1, None), (None, 1), (None, None), (1, 1)]
        app.append_rows([(*k, "new") for k in keys] if action == "update" else keys)
        app.flush()
        untouched = [(1, None, "x"), (None, 1, "y"), (None, None, "z")]
        assert table_rows(con, "pairs") == ([(1, 1, "new"), *untouched] if action == "update" else untouched)

    @pytest.mark.parametrize("action", ["update", "delete"])
    def test_a_null_key_matches_no_row_whose_key_is_null(self, con: frame.Connection, action: Action) -> None:
        con.run("CREATE TABLE groups (g INTEGER, v VARCHAR)")
        con.run("INSERT INTO groups VALUES (NULL, 'a'), (1, 'b')")
        app = con.appender("groups", action=action, key=["g"])
        app.append_rows([(None, "new"), (1, "new")] if action == "update" else [(None,), (1,)])
        app.flush()
        assert table_rows(con, "groups") == ([(1, "new"), (None, "a")] if action == "update" else [(None, "a")])


class TestCustomColumns:
    def test_columns_declare_the_buffer_in_order(self, con: frame.Connection) -> None:
        con.run("CREATE TABLE products (id INTEGER, sku VARCHAR)")
        con.run("INSERT INTO products VALUES (10, 'widget'), (20, 'gadget')")
        con.run("CREATE TABLE orders (product_id INTEGER, qty INTEGER)")
        app = con.query_appender(
            "INSERT INTO orders (product_id, qty) SELECT p.id, b.qty FROM appender_data b JOIN products p USING (sku)",
            {"sku": "VARCHAR", "qty": "INTEGER"},
        )
        app.append_row("widget", 3)
        app.append_row("gadget", None)
        app.flush()
        assert table_rows(con, "orders") == [(10, 3), (20, None)]

    @pytest.mark.parametrize(
        ("declared", "expected"),
        [
            pytest.param("INTEGER", "INTEGER", id="sql_text"),
            pytest.param("INT[]", "INTEGER[]", id="sql_text_list"),
            pytest.param("STRUCT(a INT, b VARCHAR)", "STRUCT(a INTEGER, b VARCHAR)", id="sql_text_struct"),
        ],
    )
    def test_a_type_is_sql_text(self, con: frame.Connection, declared: str, expected: str) -> None:
        con.run("CREATE TABLE types (name VARCHAR)")
        app = con.query_appender("INSERT INTO types SELECT typeof(x) FROM appender_data", {"x": declared})
        app.append_row(None)
        app.flush()
        assert table_rows(con, "types") == [(expected,)]

    @pytest.mark.parametrize(
        "declared", [int, list[str], {"a": int}], ids=["python_type", "generic_alias", "python_dict"]
    )
    def test_a_type_that_is_not_sql_text_is_refused(self, con: frame.Connection, declared: object) -> None:
        with pytest.raises(TypeError):
            con.query_appender("INSERT INTO t (k) FROM appender_data", {"x": declared})  # type: ignore[dict-item]

    def test_a_type_name_resolves_against_the_connection(self, con: frame.Connection) -> None:
        con.run("CREATE TYPE mood AS ENUM ('happy', 'sad')")
        con.run("CREATE TABLE ranges (r VARCHAR)")
        app = con.query_appender("INSERT INTO ranges SELECT enum_range(m)::VARCHAR FROM appender_data", {"m": "mood"})
        app.append_row("happy")
        app.flush()
        assert table_rows(con, "ranges") == [("[happy, sad]",)]

    @pytest.mark.parametrize(
        ("declared", "error"),
        [
            pytest.param("NULL", ValueError, id="null"),
            pytest.param("no_such_type", exceptions.CatalogError, id="unknown"),
            pytest.param("DECIMAL(40, 2)", exceptions.ProgrammingError, id="out_of_range"),
        ],
    )
    def test_a_type_that_cannot_hold_values_is_refused_at_creation(
        self, con: frame.Connection, declared: str, error: type[Exception]
    ) -> None:
        with refused(error):
            con.query_appender("INSERT INTO t (k) FROM appender_data", {"x": declared})

    @pytest.mark.parametrize(
        ("dml", "columns"),
        [
            pytest.param("INSERT INTO t FROM appender_data", {"k": "INTEGER", "K": "VARCHAR"}, id="read"),
            pytest.param("INSERT INTO t VALUES (7, 'fixed')", {"x": "BIGINT", "X": "BIGINT"}, id="never_read"),
        ],
    )
    def test_names_differing_only_in_case_are_refused(
        self, con: frame.Connection, dml: str, columns: dict[str, str]
    ) -> None:
        with refused(ValueError):
            con.query_appender(dml, columns)

    def test_names_are_identifiers_not_sql(self, con: frame.Connection) -> None:
        con.run('CREATE TABLE odd ("my col" INTEGER, "select" VARCHAR)')
        app = con.query_appender(
            'INSERT INTO odd SELECT "my col", "select" FROM appender_data', {"my col": "INTEGER", "select": "VARCHAR"}
        )
        app.append_row(1, "a")
        app.flush()
        assert table_rows(con, "odd") == [(1, "a")]

    def test_an_empty_columns_dict_is_refused(self, con: frame.Connection) -> None:
        with refused(REFUSED):
            con.query_appender("INSERT INTO t FROM appender_data", {})

    def test_the_columns_are_not_a_list_of_names(self, con: frame.Connection) -> None:
        with pytest.raises(TypeError):
            con.query_appender("INSERT INTO t FROM appender_data", ["k", "v"])  # type: ignore[arg-type]

    def test_an_empty_column_name_is_refused(self, con: frame.Connection) -> None:
        with refused(ValueError):
            con.query_appender("INSERT INTO t (k) FROM appender_data", {"": "BIGINT"})

    def test_the_columns_are_read_once_at_creation(self, con: frame.Connection) -> None:
        con.run("CREATE TABLE pair (x INTEGER, y BIGINT)")
        app = con.query_appender("INSERT INTO pair FROM appender_data", ShrinksAfterRead())
        app.append_row(1, 2)
        app.flush()
        assert table_rows(con, "pair") == [(1, 2)]

    def test_a_name_that_is_not_a_string_is_refused(self, con: frame.Connection) -> None:
        with refused(TypeError):
            con.query_appender("INSERT INTO t (k) FROM appender_data", {1: "INTEGER"})  # type: ignore[dict-item]

    def test_a_mapping_that_gives_a_name_twice_is_refused(self, con: frame.Connection) -> None:
        with refused(ValueError, match="twice"):
            con.query_appender("INSERT INTO t (k) FROM appender_data", GivesANameTwice())

    @pytest.mark.parametrize(
        ("declared", "error"),
        [
            pytest.param("no_such_type", exceptions.CatalogError, id="unknown"),
            pytest.param("DECIMAL(40, 2)", exceptions.ProgrammingError, id="out_of_range"),
            pytest.param("INTEGER(", exceptions.InvalidInputError, id="not_a_type"),
        ],
    )
    def test_a_refused_type_name_aborts_the_open_transaction(
        self, con: frame.Connection, declared: str, error: type[Exception]
    ) -> None:
        con.run("BEGIN")
        con.run("INSERT INTO t VALUES (1, 'a')")
        with refused(error):
            con.query_appender("INSERT INTO t (k) FROM appender_data", {"x": declared})
        with refused(exceptions.TransactionError):
            con.run("INSERT INTO t VALUES (2, 'b')")
        con.run("ROLLBACK")
        assert table_rows(con) == []


class TestStatementKinds:
    @pytest.mark.parametrize(
        ("dml", "expected"),
        [
            pytest.param(
                "INSERT INTO t SELECT k + 10, v FROM appender_data",
                [(1, "old1"), (2, "old2"), (12, "new2"), (13, "new3")],
                id="insert",
            ),
            pytest.param(
                "INSERT OR IGNORE INTO t FROM appender_data", [(1, "old1"), (2, "old2"), (3, "new3")], id="or_ignore"
            ),
            pytest.param(
                "INSERT OR REPLACE INTO t FROM appender_data", [(1, "old1"), (2, "new2"), (3, "new3")], id="or_replace"
            ),
            pytest.param(
                "INSERT INTO t SELECT * FROM appender_data ON CONFLICT (k) DO UPDATE SET v = excluded.v",
                [(1, "old1"), (2, "new2"), (3, "new3")],
                id="on_conflict_do_update",
            ),
            pytest.param(
                "INSERT INTO t SELECT * FROM appender_data ON CONFLICT DO NOTHING",
                [(1, "old1"), (2, "old2"), (3, "new3")],
                id="on_conflict_do_nothing",
            ),
            pytest.param(
                "UPDATE t SET v = d.v FROM appender_data d WHERE t.k = d.k",
                [(1, "old1"), (2, "new2")],
                id="update_from",
            ),
            pytest.param("DELETE FROM t USING appender_data d WHERE t.k = d.k", [(1, "old1")], id="delete_using"),
            pytest.param(
                "MERGE INTO t USING appender_data d ON t.k = d.k "
                "WHEN MATCHED THEN UPDATE SET v = d.v WHEN NOT MATCHED THEN INSERT VALUES (d.k, d.v)",
                [(1, "old1"), (2, "new2"), (3, "new3")],
                id="merge",
            ),
        ],
    )
    def test_each_statement_kind_applies_the_buffered_rows(
        self, con: frame.Connection, dml: str, expected: list[tuple[object, ...]]
    ) -> None:
        con.run("INSERT INTO t VALUES (1, 'old1'), (2, 'old2')")
        app = con.query_appender(dml, KV)
        app.append_rows([(2, "new2"), (3, "new3")])
        app.flush()
        assert table_rows(con) == expected

    @pytest.mark.parametrize(
        ("dml", "expected"),
        [
            pytest.param("INSERT INTO t FROM appender_buffer", [(1, "a")], id="buffer"),
            pytest.param(
                "INSERT INTO t SELECT * FROM appender_data UNION ALL SELECT k + 10, v FROM appender_buffer",
                [(1, "a"), (11, "a")],
                id="both_names",
            ),
        ],
    )
    def test_the_dml_reads_the_buffer_as_appender_buffer_too(
        self, con: frame.Connection, dml: str, expected: list[tuple[object, ...]]
    ) -> None:
        app = con.query_appender(dml, KV)
        app.append_row(1, "a")
        app.flush()
        assert table_rows(con) == expected

    @NEEDS_BUFFER_FIRST
    def test_a_cte_of_the_dml_can_read_the_buffer(self, con: frame.Connection) -> None:
        app = con.query_appender("WITH d AS (SELECT k + 10 AS k, v FROM appender_data) INSERT INTO t FROM d", KV)
        app.append_row(1, "a")
        app.flush()
        assert table_rows(con) == [(11, "a")]

    @pytest.mark.parametrize(
        "dml",
        [
            pytest.param("SELECT * FROM appender_data", id="select"),
            pytest.param("CREATE TABLE u AS FROM appender_data", id="create_table_as"),
            pytest.param("COPY (FROM appender_data) TO '{out}'", id="copy"),
            pytest.param("EXPLAIN INSERT INTO t FROM appender_data", id="explain"),
            pytest.param("INSERT INTO t FROM appender_data; INSERT INTO t FROM appender_data", id="two_statements"),
            pytest.param("", id="empty"),
        ],
    )
    def test_any_other_statement_is_refused_at_creation(self, con: frame.Connection, tmp_path: Path, dml: str) -> None:
        dml = dml.format(out=(tmp_path / "out.csv").as_posix())
        with refused(REFUSED):
            con.query_appender(dml, KV)

    @pytest.mark.parametrize(
        ("dml", "row", "expected"),
        [
            pytest.param("INSERT INTO t FROM appender_data RETURNING k", (2, "b"), [(1, "old"), (2, "b")], id="insert"),
            pytest.param(
                "INSERT OR REPLACE INTO t FROM appender_data RETURNING *", (1, "b"), [(1, "b")], id="or_replace"
            ),
            pytest.param(
                "UPDATE t SET v = d.v FROM appender_data d WHERE t.k = d.k RETURNING t.k",
                (1, "b"),
                [(1, "b")],
                id="update",
            ),
            pytest.param(
                "DELETE FROM t USING appender_data d WHERE t.k = d.k RETURNING t.k", (1, "b"), [], id="delete"
            ),
            pytest.param(
                "MERGE INTO t USING appender_data d ON t.k = d.k WHEN MATCHED THEN DELETE RETURNING merge_action",
                (1, "b"),
                [],
                id="merge",
            ),
        ],
    )
    def test_returning_is_allowed_and_what_it_returns_is_thrown_away(
        self, con: frame.Connection, dml: str, row: tuple[PythonValue, ...], expected: list[tuple[object, ...]]
    ) -> None:
        con.run("INSERT INTO t VALUES (1, 'old')")
        app = con.query_appender(dml, KV)
        app.append_row(*row)
        app.flush()
        assert table_rows(con) == expected

    @pytest.mark.parametrize(
        ("dml", "expected"),
        [
            pytest.param("INSERT INTO t SELECT k AS returning, v FROM appender_data", [(1, "a")], id="alias"),
            pytest.param(
                'INSERT INTO t SELECT d.k, d.v AS "returning" FROM appender_data AS d', [(1, "a")], id="quoted_alias"
            ),
            pytest.param(
                "INSERT INTO t SELECT {'returning': d.k}.returning, d.v FROM appender_data d",
                [(1, "a")],
                id="struct_field",
            ),
            pytest.param(
                "INSERT INTO t SELECT d.k, d.v FROM (SELECT k, v, k AS returning FROM appender_data) d "
                "WHERE d.returning > 0",
                [(1, "a")],
                id="column_after_a_dot",
            ),
            pytest.param("INSERT INTO t SELECT k, 'returning' FROM appender_data", [(1, "returning")], id="string"),
            pytest.param("INSERT INTO t SELECT k, v FROM appender_data /* RETURNING k */", [(1, "a")], id="comment"),
            pytest.param(
                "WITH logged AS (INSERT INTO log VALUES (5) RETURNING n) "
                "INSERT INTO t SELECT d.k, d.v FROM appender_data d, logged",
                [(1, "a")],
                id="table_expression",
            ),
        ],
    )
    def test_the_word_returning_outside_a_returning_clause_is_allowed(
        self, con: frame.Connection, dml: str, expected: list[tuple[object, ...]]
    ) -> None:
        con.run("CREATE TABLE log (n INTEGER)")
        app = con.query_appender(dml, KV)
        app.append_row(1, "a")
        app.flush()
        assert table_rows(con) == expected

    @pytest.mark.parametrize(
        ("each", "body", "logged"),
        [
            pytest.param("STATEMENT", "INSERT INTO log SELECT count(*) FROM t", [(2,), (3,)], id="statement"),
            pytest.param("ROW", "INSERT INTO log SELECT NEW.k", [(1,), (2,), (3,)], id="row"),
        ],
    )
    def test_a_target_with_a_trigger_is_allowed(
        self, con: frame.Connection, each: str, body: str, logged: list[tuple[object, ...]]
    ) -> None:
        con.run("CREATE TABLE log (n INTEGER)")
        con.run(f"CREATE TRIGGER logged AFTER INSERT ON t FOR EACH {each} {body}")
        app = con.query_appender("INSERT INTO t FROM appender_data", KV)
        app.append_rows([(1, "a"), (2, "b")])
        app.flush()
        app.append_row(3, "c")
        app.flush()
        assert table_rows(con) == [(1, "a"), (2, "b"), (3, "c")]
        assert table_rows(con, "log") == logged

    @pytest.mark.parametrize(
        "dml",
        [
            pytest.param("INSERT INTO t SELECT k, ? FROM appender_data", id="positional"),
            pytest.param("INSERT INTO t SELECT k, $1 FROM appender_data", id="numbered"),
            pytest.param("INSERT INTO t SELECT k, $v FROM appender_data", id="named"),
        ],
    )
    def test_a_parameter_is_refused_at_creation(self, con: frame.Connection, dml: str) -> None:
        with refused(REFUSED):
            con.query_appender(dml, KV)

    @pytest.mark.parametrize(
        "name",
        ["appender_buffer", "appender_data", "Appender_Data"],
        ids=["buffer", "data", "data_ignoring_case"],
    )
    def test_a_cte_of_the_dml_named_like_ours_is_refused_at_creation(self, con: frame.Connection, name: str) -> None:
        dml = f"WITH {name} AS (SELECT 1 AS k, 'x' AS v) INSERT INTO t FROM appender_data"
        with refused(REFUSED):
            con.query_appender(dml, KV)

    @pytest.mark.parametrize(
        "dml",
        [
            pytest.param(
                "WITH appender_data AS (SELECT 1 AS k) UPDATE t SET v = 'x' WHERE k IN (FROM appender_data)",
                id="update",
            ),
            pytest.param(
                "WITH appender_data AS (SELECT 1 AS k) DELETE FROM t WHERE k IN (FROM appender_data)", id="delete"
            ),
            pytest.param(
                "WITH appender_data AS (SELECT 1 AS k) MERGE INTO t USING appender_data d ON t.k = d.k "
                "WHEN MATCHED THEN DELETE",
                id="merge",
            ),
        ],
    )
    def test_a_cte_named_like_ours_is_refused_in_every_statement_kind(self, con: frame.Connection, dml: str) -> None:
        with refused(REFUSED, match="defines appender_data "):
            con.query_appender(dml, KV)

    @pytest.mark.parametrize(
        ("name", "reserved"),
        [
            pytest.param("appender_buffer", "appender_buffer", id="buffer"),
            pytest.param("appender_data", "appender_data", id="data"),
            pytest.param("APPENDER_BUFFER", "appender_buffer", id="buffer_ignoring_case"),
            pytest.param("Appender_Data", "appender_data", id="data_ignoring_case"),
        ],
    )
    def test_the_refusal_of_a_cte_named_like_ours_names_it(
        self, con: frame.Connection, name: str, reserved: str
    ) -> None:
        dml = f"WITH {name} AS (SELECT 1 AS k, 'x' AS v) INSERT INTO t FROM appender_data"
        with refused(REFUSED, match=f"defines {reserved} "):
            con.query_appender(dml, KV)

    def test_a_dml_that_never_reads_the_buffer_is_allowed(self, con: frame.Connection) -> None:
        app = con.query_appender("INSERT INTO t VALUES (7, 'fixed')", KV)
        app.append_row(1, "a")
        app.flush()
        assert table_rows(con) == [(7, "fixed")]

    def test_buffer_columns_nothing_reads_are_allowed(self, con: frame.Connection) -> None:
        app = con.query_appender(
            "INSERT INTO t SELECT k, v FROM appender_data", {"k": "INTEGER", "v": "VARCHAR", "note": "VARCHAR"}
        )
        app.append_row(1, "a", "unread")
        app.flush()
        assert table_rows(con) == [(1, "a")]

    @pytest.mark.parametrize(
        "source",
        [
            pytest.param("SELECT * FROM appender_buffer", id="the_default_spelled_out"),
            pytest.param("VALUES (5, 'v')", id="values"),
            pytest.param("FROM appender_buffer", id="from_first"),
            pytest.param(
                "SELECT k, v FROM appender_buffer UNION ALL SELECT k + 10, v FROM appender_buffer", id="union_all"
            ),
            pytest.param("WITH b AS (FROM appender_buffer) SELECT k + 20 AS k, v FROM b", id="own_with"),
            pytest.param("SELECT b.k + r.range AS k, b.v FROM appender_buffer b, range(2) r", id="join_with_range"),
            pytest.param("INSERT INTO t VALUES (9, 'x') RETURNING k, v", id="insert_returning"),
            pytest.param("CALL range(3)", id="call"),
            pytest.param("EXPLAIN SELECT * FROM appender_buffer", id="explain"),
            pytest.param("FROM appender_buffer; FROM appender_buffer", id="two_statements"),
        ],
    )
    def test_a_source_is_refused_until_duckdb_places_the_buffer_first(self, con: frame.Connection, source: str) -> None:
        with refused(exceptions.NotSupportedError, match="DuckDB"):
            con.query_appender("INSERT INTO t FROM appender_data", KV, source=source)


BIND_ERRORS = pytest.mark.parametrize(
    ("dml", "error"),
    [
        pytest.param("INSERT INTO missing FROM appender_data", exceptions.CatalogError, id="table"),
        pytest.param("INSERT INTO t SELECT nte, v FROM appender_data", exceptions.ProgrammingError, id="column"),
        pytest.param("INSERT INTO t SELECT k, v, k FROM appender_data", exceptions.ProgrammingError, id="arity"),
        pytest.param(
            "INSERT INTO t SELECT no_such_function(k), v FROM appender_data", exceptions.CatalogError, id="function"
        ),
        pytest.param("INSERT INTO t SELEC k FROM appender_data", exceptions.ParserError, id="syntax"),
    ],
)


BAD_QUERY_ARGUMENTS = pytest.mark.parametrize(
    ("arguments", "error"),
    [
        pytest.param({"dml": 5, "columns": KV}, TypeError, id="dml_not_text"),
        pytest.param(
            {"dml": "INSERT INTO t FROM appender_data", "columns": ["k"]}, TypeError, id="columns_not_a_mapping"
        ),
        pytest.param({"dml": "INSERT INTO t FROM appender_data", "columns": {}}, REFUSED, id="no_columns"),
        pytest.param(
            {"dml": "INSERT INTO t FROM appender_data", "columns": KV, "source": "FROM appender_buffer"},
            exceptions.NotSupportedError,
            id="source",
        ),
        pytest.param({"dml": "SELECT 1", "columns": KV}, REFUSED, id="another_kind"),
    ],
)


class TestCheckedAtCreation:
    @BIND_ERRORS
    def test_a_statement_that_cannot_bind_is_refused_at_creation(
        self, con: frame.Connection, dml: str, error: type[Exception]
    ) -> None:
        with refused(error):
            con.query_appender(dml, KV)

    @BIND_ERRORS
    def test_a_refusal_at_creation_leaves_the_open_transaction_usable(
        self, con: frame.Connection, dml: str, error: type[Exception]
    ) -> None:
        con.run("BEGIN")
        con.run("INSERT INTO t VALUES (1, 'a')")
        with refused(error):
            con.query_appender(dml, KV)
        con.run("INSERT INTO t VALUES (2, 'b')")
        con.run("COMMIT")
        assert table_rows(con) == [(1, "a"), (2, "b")]

    def test_creation_runs_nothing(self, con: frame.Connection) -> None:
        con.run("INSERT INTO t VALUES (1, 'a')")
        con.query_appender("MERGE INTO t USING appender_data d ON t.k = d.k WHEN NOT MATCHED BY SOURCE THEN DELETE", KV)
        assert table_rows(con) == [(1, "a")]

    @BAD_QUERY_ARGUMENTS
    def test_an_aborted_transaction_is_reported_before_bad_query_arguments(
        self, con: frame.Connection, arguments: dict[str, Any], error: type[Exception]
    ) -> None:
        con.run("BEGIN")
        with refused(exceptions.InvalidInputError):
            con.run("SELECT error('boom')")
        with refused(exceptions.TransactionError):
            con.query_appender(**arguments)
        con.run("ROLLBACK")
        with refused(error):
            con.query_appender(**arguments)

    @pytest.mark.parametrize(
        ("dml", "columns"),
        [
            pytest.param("SELECT * FROM appender_data", KV, id="another_kind"),
            pytest.param("INSERT INTO t SELECT k, ? FROM appender_data", KV, id="parameter"),
            pytest.param("INSERT INTO t (k) FROM appender_data", {"k": "NULL"}, id="null_type"),
            pytest.param(
                "WITH appender_data AS (SELECT 1 AS k, 'x' AS v) INSERT INTO t FROM appender_data",
                KV,
                id="reserved_name",
            ),
        ],
    )
    def test_the_appenders_own_refusals_leave_the_open_transaction_usable(
        self, con: frame.Connection, dml: str, columns: dict[str, str]
    ) -> None:
        con.run("BEGIN")
        con.run("INSERT INTO t VALUES (1, 'a')")
        with refused(REFUSED):
            con.query_appender(dml, columns)
        con.run("INSERT INTO t VALUES (2, 'b')")
        con.run("COMMIT")
        assert table_rows(con) == [(1, "a"), (2, "b")]

    @pytest.mark.parametrize(
        "create",
        [
            pytest.param('con.appender("t")', id="table"),
            pytest.param(f'con.query_appender("INSERT INTO t FROM appender_data", {KV!r})', id="query"),
        ],
    )
    def test_creation_from_a_python_function_inside_a_query_on_its_connection_is_refused(self, create: str) -> None:
        code, output = run_isolated(
            f"""
            from duckdb import exceptions, frame
            con = frame.connect()
            con.run("CREATE TABLE t (k INTEGER PRIMARY KEY, v VARCHAR)")
            def make_appender(x):
                try:
                    {create}
                    print("created", flush=True)
                except exceptions.Error as error:
                    print("refused:", type(error).__name__, flush=True)
                return x
            con.create_function("make_appender", make_appender, ["BIGINT"], "BIGINT", stability="volatile")
            print("started", flush=True)
            print("query:", frame.sql("SELECT make_appender(1)").rows(con))
            """
        )
        assert code == 0, output
        assert f"refused: {BUSY.__name__}\n" in output
        assert "query: [(1,)]" in output

    def test_an_invalid_input_error_of_the_bind_keeps_its_class(self, con: frame.Connection) -> None:
        with refused(exceptions.InvalidInputError):
            con.query_appender("INSERT INTO t SELECT * FROM query_table([])", KV)

    def test_creation_beside_an_open_result_resolves_a_catalog_type(self, con: frame.Connection) -> None:
        con.run("CREATE TYPE mood AS ENUM ('happy', 'sad')")
        rows = open_result(con)
        app = con.query_appender("INSERT INTO t SELECT k, m::VARCHAR FROM appender_data", {"k": "INTEGER", "m": "mood"})
        assert list(rows) == REST_OF_OPEN_RESULT
        app.append_row(1, "happy")
        app.flush()
        assert table_rows(con) == [(1, "happy")]

    @pytest.mark.parametrize(
        ("dml", "row", "error"),
        [
            pytest.param(
                "INSERT INTO t SELECT v::INTEGER, v FROM appender_data",
                (1, "not a number"),
                exceptions.ConversionError,
                id="cast",
            ),
            pytest.param(
                "INSERT INTO t FROM appender_data", (1, "again"), exceptions.ConstraintError, id="primary_key"
            ),
        ],
    )
    def test_casts_and_constraints_fail_only_at_flush(
        self, con: frame.Connection, dml: str, row: tuple[PythonValue, ...], error: type[Exception]
    ) -> None:
        con.run("INSERT INTO t VALUES (1, 'a')")
        app = con.query_appender(dml, KV)
        app.append_row(*row)
        with refused(error):
            app.flush()

    def test_a_custom_dml_into_a_read_only_catalog_fails_only_at_flush(
        self, con: frame.Connection, tmp_path: Path
    ) -> None:
        attach_read_only(con, tmp_path)
        app = con.query_appender("INSERT INTO ro.t FROM appender_data", {"k": "INTEGER"})
        app.append_row(1)
        with refused(exceptions.InvalidInputError):
            app.flush()

    @pytest.mark.parametrize(
        "create",
        [
            pytest.param(lambda c: c.appender("t"), id="table"),
            pytest.param(
                lambda c: c.query_appender("INSERT INTO t FROM appender_data", KV),
                id="custom",
            ),
        ],
    )
    def test_creation_works_while_a_result_is_open(
        self, con: frame.Connection, create: Callable[[frame.Connection], object]
    ) -> None:
        rows = open_result(con)
        create(con)
        assert list(rows) == REST_OF_OPEN_RESULT

    @pytest.mark.parametrize(
        "create",
        [
            pytest.param(lambda c: c.appender("t"), id="table"),
            pytest.param(
                lambda c: c.query_appender("INSERT INTO t FROM appender_data", KV),
                id="query_with_type_names",
            ),
            pytest.param(
                lambda c: c.query_appender("INSERT INTO t FROM appender_data", {"k": "INTEGER", "v": "mood"}),
                id="query_with_a_catalog_type_name",
            ),
        ],
    )
    def test_creation_inside_an_aborted_transaction_is_refused(
        self, con: frame.Connection, create: Callable[[frame.Connection], object]
    ) -> None:
        con.run("CREATE TYPE mood AS ENUM ('happy', 'sad')")
        con.run("BEGIN")
        with refused(exceptions.InvalidInputError):
            con.run("SELECT error('boom')")
        with refused(exceptions.TransactionError):
            create(con)
        con.run("ROLLBACK")
        create(con)

    @pytest.mark.parametrize(
        "options", [{"action": "merge"}, {"action": "update"}], ids=["unknown_action", "update_without_key"]
    )
    def test_an_aborted_transaction_is_reported_before_a_bad_action(
        self, con: frame.Connection, options: dict[str, Any]
    ) -> None:
        con.run("BEGIN")
        with refused(exceptions.InvalidInputError):
            con.run("SELECT error('boom')")
        with refused(exceptions.TransactionError):
            con.appender("t", **options)
        con.run("ROLLBACK")
        with refused(REFUSED):
            con.appender("t", **options)

    @pytest.mark.parametrize("kind", ["frame", "dbapi"])
    def test_an_aborted_transaction_is_reported_before_a_malformed_name(self, kind: str) -> None:
        four_parts = ("memory", "main", "t", "u")
        connection: frame.Connection | dbapi.Connection
        if kind == "frame":
            connection = frame.connect()
            connection.run("BEGIN")
            with refused(exceptions.InvalidInputError):
                connection.run("SELECT error('boom')")
        else:
            connection = dbapi.connect()
            with refused(exceptions.InvalidInputError):
                connection.cursor().execute("SELECT error('boom')")
        try:
            with refused(exceptions.TransactionError):
                connection.appender(four_parts)
            if isinstance(connection, dbapi.Connection):
                connection.rollback()
            else:
                connection.run("ROLLBACK")
            with refused(REFUSED):
                connection.appender(four_parts)
        finally:
            connection.close()


class TestRows:
    @pytest.mark.parametrize("values", [(2,), (2, "b", "extra"), ()], ids=["too_few", "too_many", "none"])
    def test_wrong_arity_raises_and_buffers_nothing(
        self, con: frame.Connection, values: tuple[PythonValue, ...]
    ) -> None:
        app = con.appender("t")
        app.append_row(1, "a")
        with refused(WRONG_ARITY):
            app.append_row(*values)
        app.flush()
        assert table_rows(con) == [(1, "a")]

    def test_a_failed_conversion_buffers_nothing_of_its_row(self, con: frame.Connection) -> None:
        con.run("CREATE TABLE w (a INTEGER, b SMALLINT)")
        app = con.appender("w")
        app.append_row(1, 1)
        with refused(exceptions.ConversionError):
            app.append_row(2, 100_000)
        app.flush()
        assert table_rows(con, "w") == [(1, 1)]

    def test_a_failed_row_leaves_no_null_behind(self, con: frame.Connection) -> None:
        con.run("CREATE TABLE w (a INTEGER, b SMALLINT)")
        app = con.appender("w")
        app.append_row(1, 1)
        with refused(exceptions.ConversionError):
            app.append_row(None, 100_000)
        app.append_row(3, 3)
        app.flush()
        assert table_rows(con, "w") == [(1, 1), (3, 3)]

    @pytest.mark.parametrize(
        ("bad", "error"),
        [
            pytest.param((3,), WRONG_ARITY, id="arity"),
            pytest.param(("x", "c"), exceptions.ConversionError, id="value"),
            pytest.param("12", TypeError, id="str"),
            pytest.param(b"12", TypeError, id="bytes"),
            pytest.param({3: "x", 4: "y"}, TypeError, id="dict"),
        ],
    )
    def test_append_rows_stops_at_the_first_bad_row_and_keeps_the_rows_before_it(
        self, con: frame.Connection, bad: object, error: type[Exception]
    ) -> None:
        consumed = []

        def rows() -> Iterator[object]:
            for row in [(1, "a"), (2, "b"), bad, (4, "d")]:
                consumed.append(row)
                yield row

        app = con.appender("t")
        with refused(error):
            app.append_rows(rows())  # type: ignore[arg-type]
        app.flush()
        assert table_rows(con) == [(1, "a"), (2, "b")]
        assert len(consumed) == 3

    def test_an_exception_from_the_iterable_propagates_unchanged_and_keeps_the_rows_before_it(
        self, con: frame.Connection
    ) -> None:
        failure = ValueError("the source of rows broke")

        def rows() -> Iterator[tuple[PythonValue, ...]]:
            yield (1, "a")
            raise failure

        app = con.appender("t")
        with pytest.raises(ValueError, match="the source of rows broke") as raised:
            app.append_rows(rows())
        app.flush()
        assert raised.value is failure
        assert table_rows(con) == [(1, "a")]

    def test_append_rows_takes_any_iterable_of_sequences(self, con: frame.Connection) -> None:
        app = con.appender("t")
        app.append_rows((k, str(k)) for k in range(2))
        app.append_rows([[2, None]])
        app.flush()
        assert table_rows(con) == [(0, "0"), (1, "1"), (2, None)]


class TestAppendingRunsNothing:
    def test_appending_works_while_a_result_is_open(self, con: frame.Connection) -> None:
        app = con.appender("t")
        rows = open_result(con)
        app.append_row(1, "a")
        app.append_rows([(2, "b")])
        assert list(rows) == REST_OF_OPEN_RESULT
        discard(app)

    def test_clear_works_while_a_result_is_open(self, con: frame.Connection) -> None:
        app = con.appender("t")
        app.append_row(1, "a")
        rows = open_result(con)
        app.clear()
        assert list(rows) == REST_OF_OPEN_RESULT
        app.flush()
        assert table_rows(con) == []

    def test_buffered_rows_are_invisible_until_flushed(self, con: frame.Connection) -> None:
        other = con.duplicate()
        app = con.appender("t")
        app.append_row(1, "a")
        assert table_rows(con) == []
        assert table_rows(other) == []
        discard(app)

    def test_appending_runs_no_part_of_the_flush_statement(self, con: frame.Connection, produced: list[int]) -> None:
        app = con.query_appender("INSERT INTO t SELECT tally(k), v FROM appender_data", KV)
        app.append_rows([(1, "a"), (2, "b")])
        assert produced[0] == 0
        discard(app)

    def test_appending_works_while_another_thread_runs_a_statement(
        self, con: frame.Connection, held: tuple[threading.Event, threading.Event]
    ) -> None:
        running, release = held
        app = con.appender("t")
        outcome: list[object] = []
        thread = threading.Thread(
            target=contend, args=(running, release, lambda: app.append_row(1, "a"), outcome, exceptions.Error)
        )
        thread.start()
        frame.sql("SELECT hold(i) FROM range(1) t(i)").rows(con)
        thread.join(10)
        assert outcome == ["ran"]
        discard(app)

    def test_a_callback_inside_a_query_can_append(self, con: frame.Connection) -> None:
        app = con.appender("t")

        def collect(x: int) -> int:
            app.append_row(x, "from a callback")
            return x

        con.create_function("collect", collect, ["BIGINT"], "BIGINT", stability="volatile")
        frame.sql("SELECT collect(i) FROM range(2) t(i)").rows(con)
        app.flush()
        assert table_rows(con) == [(0, "from a callback"), (1, "from a callback")]


class TestFlush:
    def test_each_flush_applies_only_the_rows_buffered_since_the_last(self, con: frame.Connection) -> None:
        con.run("CREATE TABLE log (k INTEGER)")
        app = con.appender("log")
        app.append_row(1)
        app.flush()
        app.append_rows([(2,), (3,)])
        app.flush()
        assert table_rows(con, "log") == [(1,), (2,), (3,)]

    def test_a_flush_applies_every_row_across_chunk_boundaries(self, con: frame.Connection) -> None:
        con.run("CREATE TABLE log (k INTEGER)")
        app = con.appender("log")
        rows = 2 * VECTOR + 1
        app.append_rows((i,) for i in range(rows))
        app.flush()
        assert frame.sql("SELECT count(*), sum(k) FROM log").rows(con) == [(rows, rows * (rows - 1) // 2)]

    def test_long_strings_survive_across_chunk_boundaries(self, con: frame.Connection) -> None:
        con.run("CREATE TABLE words (s VARCHAR)")
        app = con.appender("words")
        strings = [f"longer than the inline length {i:05d}" for i in range(2 * VECTOR + 1)]
        app.append_rows((s,) for s in strings)
        app.flush()
        assert sorted(s for (s,) in frame.sql("SELECT s FROM words").rows(con)) == sorted(strings)

    def test_nested_values_survive_across_chunk_boundaries(self, con: frame.Connection) -> None:
        con.run("CREATE TABLE nested (l INTEGER[], s STRUCT(a INTEGER, b VARCHAR))")
        app = con.appender("nested")
        rows = [
            (
                None if i % 7 == 0 else [i, None, i + 1][: i % 4],
                None if i % 11 == 0 else {"a": i, "b": f"longer than the inline length {i:05d}"},
            )
            for i in range(2 * VECTOR + 1)
        ]
        app.append_rows(rows)
        app.flush()
        stored = frame.sql("SELECT l, s FROM nested").rows(con)
        assert sorted(stored, key=repr) == sorted(rows, key=repr)

    def test_a_plain_insert_keeps_the_append_order(self, con: frame.Connection) -> None:
        con.run("SET threads = 4")
        con.run("CREATE TABLE log (k INTEGER)")
        app = con.appender("log")
        appended = list(range(3 * VECTOR, -1, -1))
        app.append_rows((k,) for k in appended)
        app.flush()
        assert [k for (k,) in frame.sql("SELECT k FROM log").rows(con)] == appended

    def test_nulls_that_move_between_chunks_arrive(self, con: frame.Connection) -> None:
        con.run("CREATE TABLE log (k INTEGER)")
        app = con.appender("log")
        values = [None if (i % 2 == 0) == (i < VECTOR) else i for i in range(2 * VECTOR + 1)]
        app.append_rows((v,) for v in values)
        app.flush()
        present = [v for v in values if v is not None]
        expected = [(len(values), len(present), sum(present))]
        assert frame.sql("SELECT count(*), count(k), sum(k) FROM log").rows(con) == expected

    def test_two_appenders_on_one_connection_flush_only_their_own_rows(self, con: frame.Connection) -> None:
        con.run("CREATE TABLE log (k INTEGER)")
        first, second = con.appender("log"), con.appender("log")
        first.append_row(1)
        second.append_row(2)
        first.flush()
        first.append_row(3)
        second.flush()
        first.flush()
        assert table_rows(con, "log") == [(1,), (2,), (3,)]

    def test_the_target_resolves_at_each_flush(self, con: frame.Connection) -> None:
        app = con.appender("t")
        run_all(con, "DROP TABLE t", "CREATE TABLE t (k INTEGER PRIMARY KEY, v VARCHAR, w INTEGER DEFAULT 5)")
        app.append_row(1, "a")
        app.flush()
        assert table_rows(con) == [(1, "a", 5)]

    @pytest.mark.requires("pandas")
    def test_a_registered_frame_named_in_the_dml_is_read_at_the_flush(self, con: frame.Connection) -> None:
        import pandas as pd

        con.run("CREATE TABLE priced (k INTEGER, price BIGINT)")
        con.register("prices", pd.DataFrame({"k": [1, 2], "price": [10, 20]}))
        app = appender_reading_prices(con)
        app.append_rows([(1,), (2,)])
        app.flush()
        assert table_rows(con, "priced") == [(1, 10), (2, 20)]

    @pytest.mark.requires("pandas")
    def test_an_object_registered_again_before_a_flush_is_the_one_read(self, con: frame.Connection) -> None:
        import pandas as pd

        con.run("CREATE TABLE priced (k INTEGER, price BIGINT)")
        con.register("prices", pd.DataFrame({"k": [1, 2], "price": [10, 20]}))
        app = appender_reading_prices(con)
        con.register("prices", pd.DataFrame({"k": [1, 2], "price": [-1, -2]}))
        app.append_row(1)
        app.flush()
        assert table_rows(con, "priced") == [(1, -1)]

    @pytest.mark.requires("pandas")
    def test_a_registered_frame_serves_every_flush(self, con: frame.Connection) -> None:
        import pandas as pd

        con.run("CREATE TABLE priced (k INTEGER, price BIGINT)")
        con.register("prices", pd.DataFrame({"k": [1, 2], "price": [10, 20]}))
        app = appender_reading_prices(con)
        app.append_row(1)
        app.flush()
        app.append_row(2)
        app.flush()
        assert table_rows(con, "priced") == [(1, 10), (2, 20)]

    @pytest.mark.requires("pyarrow")
    def test_a_registered_arrow_table_serves_every_flush(self, con: frame.Connection) -> None:
        import pyarrow as pa

        con.run("CREATE TABLE priced (k INTEGER, price BIGINT)")
        con.register("prices", pa.table({"k": [1, 2], "price": [10, 20]}))
        app = appender_reading_prices(con)
        app.append_row(1)
        app.flush()
        app.append_row(2)
        app.flush()
        assert table_rows(con, "priced") == [(1, 10), (2, 20)]

    @pytest.mark.requires("pandas")
    def test_a_dropped_table_is_not_replaced_by_a_python_variable_at_flush(self, con: frame.Connection) -> None:
        import pandas as pd

        run_all(
            con,
            "CREATE TABLE prices (k INTEGER, price BIGINT)",
            "INSERT INTO prices VALUES (1, 10)",
            "CREATE TABLE priced (k INTEGER, price BIGINT)",
        )
        app = appender_reading_prices(con)
        con.run("DROP TABLE prices")
        prices = pd.DataFrame({"k": [1], "price": [-1]})  # noqa: F841
        app.append_row(1)
        with refused(exceptions.CatalogError):
            app.flush()
        assert table_rows(con, "priced") == []

    @pytest.mark.requires("pandas")
    def test_a_dropped_table_is_replaced_by_an_object_registered_under_its_name(self, con: frame.Connection) -> None:
        import pandas as pd

        run_all(
            con,
            "CREATE TABLE prices (k INTEGER, price BIGINT)",
            "INSERT INTO prices VALUES (1, 10)",
            "CREATE TABLE priced (k INTEGER, price BIGINT)",
        )
        app = appender_reading_prices(con)
        con.run("DROP TABLE prices")
        con.register("prices", pd.DataFrame({"k": [1], "price": [-1]}))
        app.append_row(1)
        app.flush()
        assert table_rows(con, "priced") == [(1, -1)]

    def test_an_interrupt_while_no_flush_runs_does_not_affect_the_next(self, con: frame.Connection) -> None:
        app = con.appender("t")
        app.append_row(1, "a")
        con.interrupt()
        app.flush()
        assert table_rows(con) == [(1, "a")]

    def test_the_dml_can_read_the_buffer_twice(self, con: frame.Connection) -> None:
        con.run("CREATE TABLE log (k INTEGER, v VARCHAR)")
        app = con.query_appender(
            "INSERT INTO log SELECT * FROM appender_data UNION ALL SELECT * FROM appender_data", KV
        )
        app.append_rows([(1, "a"), (2, "b")])
        app.flush()
        assert table_rows(con, "log") == [(1, "a"), (1, "a"), (2, "b"), (2, "b")]

    def test_a_flush_outside_a_transaction_commits_by_itself(self, con: frame.Connection) -> None:
        other = con.duplicate()
        app = con.appender("t")
        app.append_row(1, "a")
        app.flush()
        assert table_rows(other) == [(1, "a")]

    def test_a_flush_inside_begin_is_visible_elsewhere_only_after_commit(self, con: frame.Connection) -> None:
        other = con.duplicate()
        app = con.appender("t")
        con.run("BEGIN")
        app.append_row(1, "a")
        app.flush()
        assert table_rows(other) == []
        con.run("COMMIT")
        assert table_rows(other) == [(1, "a")]

    def test_a_flush_undone_by_rollback_is_not_applied_again(self, con: frame.Connection) -> None:
        app = con.appender("t")
        con.run("BEGIN")
        app.append_row(1, "a")
        app.flush()
        con.run("ROLLBACK")
        app.flush()
        assert table_rows(con) == []

    def test_a_flush_runs_on_the_appenders_connection(self, con: frame.Connection) -> None:
        other = con.duplicate()
        other.run("CREATE TEMP TABLE scratch (k INTEGER)")
        app = other.appender("scratch")
        app.append_row(1)
        app.flush()
        assert table_rows(other, "scratch") == [(1,)]

    def test_an_empty_flush_runs_nothing(self, con: frame.Connection) -> None:
        con.run("INSERT INTO t VALUES (1, 'a')")
        app = con.query_appender(
            "MERGE INTO t USING appender_data d ON t.k = d.k WHEN NOT MATCHED BY SOURCE THEN DELETE", KV
        )
        app.flush()
        assert table_rows(con) == [(1, "a")]

    def test_an_empty_flush_does_not_need_the_connection(self, con: frame.Connection) -> None:
        app = con.appender("t")
        rows = open_result(con)
        app.flush()
        assert list(rows) == REST_OF_OPEN_RESULT

    @pytest.mark.parametrize(
        "dml",
        [
            pytest.param("INSERT OR REPLACE INTO t FROM appender_data", id="or_replace"),
            pytest.param(
                "INSERT INTO t SELECT * FROM appender_data ON CONFLICT (k) DO UPDATE SET v = excluded.v",
                id="on_conflict_do_update",
            ),
        ],
    )
    def test_a_key_repeated_within_one_upsert_keeps_one_of_its_rows(self, con: frame.Connection, dml: str) -> None:
        con.run("INSERT INTO t VALUES (1, 'old')")
        app = con.query_appender(dml, KV)
        app.append_rows([(1, "first"), (1, "second"), (2, "first"), (2, "second")])
        app.flush()
        one_per_key = [[(1, a), (2, b)] for a in ("first", "second") for b in ("first", "second")]
        assert table_rows(con) in one_per_key

    def test_a_key_repeated_within_one_update_from_applies_one_of_its_rows(self, con: frame.Connection) -> None:
        con.run("INSERT INTO t VALUES (1, 'old')")
        app = con.query_appender("UPDATE t SET v = d.v FROM appender_data d WHERE t.k = d.k", KV)
        app.append_rows([(1, "first"), (1, "second")])
        app.flush()
        assert table_rows(con) in ([(1, "first")], [(1, "second")])

    def test_a_key_repeated_within_one_merge_raises(self, con: frame.Connection) -> None:
        con.run("INSERT INTO t VALUES (1, 'old')")
        app = con.query_appender(
            "MERGE INTO t USING appender_data d ON t.k = d.k "
            "WHEN MATCHED THEN UPDATE SET v = d.v WHEN NOT MATCHED THEN INSERT VALUES (d.k, d.v)",
            KV,
        )
        app.append_rows([(1, "first"), (1, "second")])
        with refused(exceptions.InvalidInputError, match="more than once"):
            app.flush()

    def test_a_key_repeated_within_one_merge_whose_condition_picks_one_row_applies_it(
        self, con: frame.Connection
    ) -> None:
        con.run("INSERT INTO t VALUES (1, 'old')")
        app = con.query_appender(
            "MERGE INTO t USING appender_data d ON t.k = d.k WHEN MATCHED AND d.v = 'first' THEN UPDATE SET v = d.v", KV
        )
        app.append_rows([(1, "first"), (1, "second")])
        app.flush()
        assert table_rows(con) == [(1, "first")]

    @pytest.mark.parametrize("action", ["insert", "upsert", "update", "delete"])
    def test_one_appender_applies_each_of_three_flushes_larger_than_a_staging_chunk(
        self, con: frame.Connection, action: Action
    ) -> None:
        per_flush = VECTOR + 100
        keyed = action in ("update", "delete")
        expected: dict[int, str] = {}
        if keyed:
            con.run(f"INSERT INTO t SELECT range, 'old' FROM range({3 * per_flush})")
            expected = dict.fromkeys(range(3 * per_flush), "old")
        app = con.appender("t", action=action, key=["k"] if keyed else None)
        for flush in range(3):
            # An upsert's flushes overlap by half, so each one both replaces rows and adds them.
            start = flush * (per_flush // 2 if action == "upsert" else per_flush)
            keys = range(start, start + per_flush)
            if action == "delete":
                app.append_rows((k,) for k in keys)
                for k in keys:
                    del expected[k]
            else:
                app.append_rows((k, f"flush {flush}") for k in keys)
                expected.update(dict.fromkeys(keys, f"flush {flush}"))
            app.flush()
            assert dict(table_rows(con)) == expected


#: A child that presses Ctrl-C at a range of moments during a flush of one large row, and retries the flush after it.
CTRL_C_TRIALS = """
import _thread
import os
import signal
import threading
import traceback
from duckdb import frame
# A background job starts with SIGINT ignored, which would turn the simulated Ctrl-C into nothing
signal.signal(signal.SIGINT, signal.default_int_handler)
root = {root!r}
print("started", flush=True)
trial = 0
for threads in (1, 4):
    # A large row makes the step that writes it, and commits it, long enough for a Ctrl-C to land around it
    for size in (8, 16, 32):
        row = os.urandom(size << 20)
        for delay in (0.001, 0.003, 0.006, 0.012, 0.025, 0.05):
            trial += 1
            path = f"{{root}}/{{trial}}.db"
            con = frame.connect(path, threads=threads)
            con.run("CREATE TABLE t (v BLOB)")
            app = con.appender("t")
            app.append_row(row)
            done = threading.Event()
            def interrupt():
                if not done.wait(delay):
                    _thread.interrupt_main()
            interrupter = threading.Thread(target=interrupt)
            # "inside" when the flush raised it, "after" when it landed once the flush had returned.
            raised = "none"
            try:
                interrupter.start()
                try:
                    app.flush()
                finally:
                    done.set()
                    interrupter.join()
            except KeyboardInterrupt as error:
                frames = [f.f_code.co_name for f, _ in traceback.walk_tb(error.__traceback__)]
                raised = "inside" if "_flush_appender" in frames else "after"
            stored = frame.sql("SELECT count(*) FROM t").rows(con)[0][0]
            app.flush()
            final = frame.sql("SELECT count(*) FROM t").rows(con)[0][0]
            print("trial", threads, size, delay, raised, stored, final, flush=True)
            con.close()
            for leftover in (path, path + ".wal"):
                if os.path.exists(leftover):
                    os.remove(leftover)
"""


@pytest.fixture(scope="module")
def ctrl_c_trials(tmp_path_factory: pytest.TempPathFactory) -> list[tuple[str, ...]]:
    """Per trial: threads, row size in MiB, delay, where Ctrl-C raised, rows stored before and after the retry."""
    root = tmp_path_factory.mktemp("ctrl_c")
    code, output = run_isolated(CTRL_C_TRIALS.format(root=root.as_posix()), timeout=120)
    assert code == 0, output
    trials = [tuple(line.split()[1:]) for line in output.splitlines() if line.startswith("trial ")]
    assert len(trials) == 36, output
    return trials


#: A child that presses Ctrl-C during a flush inside BEGIN, and tells by the next statement whether it wrote the rows.
CTRL_C_INSIDE_BEGIN = """
import _thread
import signal
import threading
import time
from duckdb import exceptions, frame
# A background job starts with SIGINT ignored, which would turn the simulated Ctrl-C into nothing
signal.signal(signal.SIGINT, signal.default_int_handler)
con = frame.connect(threads=4)
{setup}
con.run("BEGIN")
print("started", flush=True)
try:
    app.flush()
    print("flush: returned")
except KeyboardInterrupt:
    print("flush: KeyboardInterrupt")
written = "SELECT count(*) FROM t"
try:
    print("next statement:", frame.sql(written).rows(con))
except exceptions.Error as error:
    print("next statement:", type(error).__name__)
    con.run("ROLLBACK")
    app.flush()
else:
    app.flush()
    con.run("COMMIT")
print("kept:", frame.sql(written).rows(con.duplicate()))
"""

CTRL_C_BEFORE_THE_WRITE = """
calls = []
con.create_function("record", calls.append, ["BIGINT"], "BIGINT", null_handling="special", stability="volatile")
con.run("CREATE TABLE t (k INTEGER CHECK (record(k) IS NULL), v VARCHAR)")
app = con.appender("t")
app.append_rows((i, "a") for i in range(1000000))
# The CHECK runs on every row before the write takes effect, so the Ctrl-C lands while it runs
def press_once_the_check_runs():
    while len(calls) < 10000:
        time.sleep(0.001)
    _thread.interrupt_main()
threading.Thread(target=press_once_the_check_runs, daemon=True).start()
"""

CTRL_C_AFTER_THE_WRITE = """
# A builtin runs no bytecode, and RETURNING runs it once the insert has written its row
con.create_function("press", _thread.interrupt_main, [], "BIGINT", null_handling="special", stability="volatile")
con.run("CREATE TABLE t (k INTEGER, v VARCHAR)")
app = con.query_appender("INSERT INTO t FROM appender_data RETURNING press()", {"k": "INTEGER", "v": "VARCHAR"})
app.append_row(1, "a")
"""


#: A child that cancels a flush updating the table's rows while its CHECK runs on them, then retries the flush.
CANCELLED_TABLE_ACTION = """
import _thread
import signal
import threading
import time
from duckdb import exceptions, frame
# A background job starts with SIGINT ignored, which would turn the simulated Ctrl-C into nothing
signal.signal(signal.SIGINT, signal.default_int_handler)
con = frame.connect(threads=4)
# A builtin runs no bytecode, so the Ctrl-C reaches the flush, not a Python function it runs
calls = []
con.create_function("record", calls.append, ["BIGINT"], "BIGINT", null_handling="special", stability="volatile")
con.run("CREATE TABLE t (k INTEGER PRIMARY KEY, v VARCHAR CHECK (record(length(v)) IS NULL))")
con.run("INSERT INTO t SELECT range, 'old' FROM range(1000000)")
app = {create}
app.append_rows((i, "new") for i in range({first}, {first} + 1000000))
calls.clear()
print("started", flush=True)

# The CHECK runs on every written row before the write takes effect, so the cancel lands while it runs
def cancel_once_the_check_runs():
    while len(calls) < 10000:
        time.sleep(0.001)
    {cancel}

threading.Thread(target=cancel_once_the_check_runs, daemon=True).start()
try:
    app.flush()
    print("flush: returned")
except (KeyboardInterrupt, exceptions.Error) as error:
    print("flush:", type(error).__name__, flush=True)
written = "SELECT count(*), count(*) FILTER (v = 'new') FROM t"
print("stored:", frame.sql(written).rows(con))
app.flush()
print("after the retry:", frame.sql(written).rows(con))
"""

#: Per action: its key, the first key its rows write, and the table's rows and rows holding the new value once applied.
CANCELLED_TABLE_ACTION_CASES = [
    pytest.param("upsert", None, 500_000, (1_500_000, 1_000_000), id="upsert"),
    pytest.param("update", ["k"], 0, (1_000_000, 1_000_000), id="update"),
]


def cancelled_flush(create: str, first: int, cancel: str) -> str:
    # In a child, because a flush the cancel fails to end would otherwise hang the whole suite.
    code, output = run_isolated(CANCELLED_TABLE_ACTION.format(create=create, first=first, cancel=cancel), timeout=60)
    assert code == 0, output
    assert "stored: [(1000000, 0)]" in output
    return output


def cancelled_table_action(action: str, key: list[str] | None, first: int, cancel: str) -> str:
    return cancelled_flush(f'con.appender("t", action={action!r}, key={key!r})', first, cancel)


class TestFlushFailures:
    def test_a_flush_refused_by_an_open_result_keeps_the_rows_and_the_transaction(self, con: frame.Connection) -> None:
        app = con.appender("t")
        con.run("BEGIN")
        con.run("INSERT INTO t VALUES (1, 'a')")
        app.append_row(2, "b")
        rows = open_result(con)
        with refused(BUSY, match=LIVE_RESULT):
            app.flush()
        assert list(rows) == REST_OF_OPEN_RESULT
        app.flush()
        con.run("COMMIT")
        assert table_rows(con) == [(1, "a"), (2, "b")]

    def test_a_flush_refused_while_another_thread_runs_a_statement_keeps_the_rows(
        self, con: frame.Connection, held: tuple[threading.Event, threading.Event]
    ) -> None:
        running, release = held
        app = con.appender("t")
        app.append_row(1, "a")
        outcome: list[object] = []
        thread = threading.Thread(target=contend, args=(running, release, app.flush, outcome, exceptions.Error))
        thread.start()
        frame.sql("SELECT hold(i) FROM range(1) t(i)").rows(con)
        thread.join(10)
        assert len(outcome) == 1
        assert type(outcome[0]) is BUSY
        assert QUERY_RUNS in str(outcome[0])
        app.flush()
        assert table_rows(con) == [(1, "a")]

    @pytest.mark.parametrize(
        ("settle", "expected"),
        [
            pytest.param(("app.flush()",), "rows: [(1, 'a')]", id="retried"),
            pytest.param(("app.clear()", "app.flush()"), "rows: []", id="cleared"),
        ],
    )
    def test_an_interrupted_flush_keeps_the_rows(self, settle: tuple[str, ...], expected: str) -> None:
        assert expected in interrupted_flush(after=settle)

    def test_an_interrupted_flush_inside_begin_keeps_the_rows_for_after_rollback(self) -> None:
        output = interrupted_flush(
            before=('con.run("BEGIN")', "con.run(\"INSERT INTO t VALUES (5, 'in the transaction')\")"),
            after=('con.run("ROLLBACK")', "app.flush()"),
            reader="con.duplicate()",
        )
        assert "rows: [(1, 'a')]" in output

    def test_a_retry_before_rollback_drops_the_rows_of_an_interrupted_flush(self) -> None:
        output = interrupted_flush(
            before=('con.run("BEGIN")',),
            after=(
                "try:",
                "    app.flush()",
                "except exceptions.Error as error:",
                '    print("retry:", type(error).__name__)',
                'con.run("ROLLBACK")',
                "app.flush()",
            ),
        )
        assert "retry: TransactionError\n" in output
        assert "rows: []" in output

    @pytest.mark.parametrize(
        ("setup", "expected"),
        [
            pytest.param(
                CTRL_C_BEFORE_THE_WRITE,
                ("next statement: TransactionError", "kept: [(1000000,)]"),
                id="before_the_write",
            ),
            pytest.param(CTRL_C_AFTER_THE_WRITE, ("next statement: [(1,)]", "kept: [(1,)]"), id="after_the_write"),
        ],
    )
    def test_after_a_ctrl_c_inside_begin_the_next_statement_tells_whether_the_rows_were_written(
        self, setup: str, expected: tuple[str, ...]
    ) -> None:
        code, output = run_isolated(CTRL_C_INSIDE_BEGIN.format(setup=setup), timeout=60)
        assert code == 0, output
        assert "flush: KeyboardInterrupt\n" in output
        for line in expected:
            assert f"{line}\n" in output

    def test_a_ctrl_c_during_a_flush_writes_each_row_exactly_once(self, ctrl_c_trials: list[tuple[str, ...]]) -> None:
        assert [trial for trial in ctrl_c_trials if trial[-1] != "1"] == []

    def test_a_ctrl_c_after_the_write_took_effect_drops_the_rows_and_still_raises(
        self, ctrl_c_trials: list[tuple[str, ...]]
    ) -> None:
        # Raised by the flush, the row stored by it, and the retry adding nothing.
        assert any(trial[3:] == ("inside", "1", "1") for trial in ctrl_c_trials)

    @pytest.mark.parametrize("kind", ["frame", "dbapi"])
    def test_a_ctrl_c_before_the_write_took_effect_keeps_the_rows(self, kind: str) -> None:
        code, output = run_isolated(
            f"""
            import _thread
            import signal
            import threading
            import time
            from duckdb import _duckdb, dbapi, frame
            # A background job starts with SIGINT ignored, which would turn the simulated Ctrl-C into nothing
            signal.signal(signal.SIGINT, signal.default_int_handler)
            # One database for both kinds: the DB-API takes no Python function, so a frame connection registers it.
            database = _duckdb.Database(":memory:", [("threads", "4")])
            con = frame.Connection(database)
            # A builtin runs no bytecode, so the Ctrl-C reaches the flush, not a Python function it runs
            calls = []
            con.create_function(
                "record", calls.append, ["BIGINT"], "BIGINT", null_handling="special", stability="volatile"
            )
            con.run("CREATE TABLE t (k INTEGER CHECK (record(k) IS NULL), v VARCHAR)")
            target = con if {kind!r} == "frame" else dbapi.Connection(database.connect())
            app = target.appender("t")
            app.append_rows((i, "a") for i in range(1_000_000))
            print("started", flush=True)

            # The CHECK runs on every row before the insert takes effect, so the Ctrl-C lands while it runs
            def interrupt_once_the_check_runs():
                while len(calls) < 10000:
                    time.sleep(0.001)
                _thread.interrupt_main()

            threading.Thread(target=interrupt_once_the_check_runs, daemon=True).start()
            try:
                app.flush()
            except KeyboardInterrupt as error:
                print("interrupted:", type(error).__name__, flush=True)
            print("stored:", frame.sql("SELECT count(*) FROM t").rows(con))
            if isinstance(target, dbapi.Connection):
                # The cancelled flush wrote nothing; the implicit transaction it opened ends here.
                target.rollback()
            app.flush()
            if isinstance(target, dbapi.Connection):
                target.commit()
            print("after the retry:", frame.sql("SELECT count(*) FROM t").rows(con))
            """,
            timeout=60,
        )
        assert code == 0, output
        assert "interrupted: KeyboardInterrupt" in output
        assert "stored: [(0,)]" in output
        assert "after the retry: [(1000000,)]" in output

    @pytest.mark.parametrize(("action", "key", "first", "applied"), CANCELLED_TABLE_ACTION_CASES)
    def test_a_ctrl_c_during_an_upsert_or_update_flush_keeps_the_rows_for_the_retry(
        self, action: Action, key: list[str] | None, first: int, applied: tuple[int, int]
    ) -> None:
        output = cancelled_table_action(action, key, first, "_thread.interrupt_main()")
        assert "flush: KeyboardInterrupt\n" in output
        assert f"after the retry: [{applied}]" in output

    @pytest.mark.parametrize(("action", "key", "first", "applied"), CANCELLED_TABLE_ACTION_CASES)
    def test_an_interrupt_during_an_upsert_or_update_flush_keeps_the_rows_for_the_retry(
        self, action: Action, key: list[str] | None, first: int, applied: tuple[int, int]
    ) -> None:
        output = cancelled_table_action(action, key, first, "con.interrupt()")
        assert "flush: InterruptError\n" in output
        assert f"after the retry: [{applied}]" in output

    @pytest.mark.parametrize(
        ("cancel", "raised"),
        [
            pytest.param("_thread.interrupt_main()", "KeyboardInterrupt", id="ctrl_c"),
            pytest.param("con.interrupt()", "InterruptError", id="interrupt"),
        ],
    )
    def test_a_cancelled_flush_of_a_statement_with_returning_keeps_the_rows_for_the_retry(
        self, cancel: str, raised: str
    ) -> None:
        create = (
            'con.query_appender("UPDATE t SET v = d.v FROM appender_data d WHERE t.k = d.k RETURNING t.k", '
            '{"k": "INTEGER", "v": "VARCHAR"})'
        )
        output = cancelled_flush(create, 0, cancel)
        assert f"flush: {raised}\n" in output
        assert "after the retry: [(1000000, 1000000)]" in output

    def test_a_flush_interrupted_by_ctrl_c_keeps_the_rows(self) -> None:
        code, output = run_isolated(
            f"""
            import _thread
            import signal
            import threading
            import time
            from duckdb import frame
            # A background job starts with SIGINT ignored, which would turn the simulated Ctrl-C into nothing
            signal.signal(signal.SIGINT, signal.default_int_handler)
            con = frame.connect()
            con.run("SET threads = 4")
            con.run("CREATE TABLE t (k INTEGER, v VARCHAR)")
            con.run("CREATE TABLE big AS SELECT range AS i FROM range(2000000)")
            # A builtin runs no bytecode, so the Ctrl-C reaches the client's signal check while the scan's
            # tasks are still in the UDF, instead of raising inside a UDF call and failing the flush
            calls = []
            con.create_function(
                "record", calls.append, ["BIGINT"], "BIGINT", null_handling="special", stability="volatile"
            )
            app = con.query_appender(
                "INSERT INTO t SELECT k, v FROM appender_data, (SELECT count(record(i)) AS n FROM big) WHERE n >= 0",
                {KV!r},
            )
            app.append_row(1, "a")
            print("started", flush=True)

            # Waiting for the UDF to run, not for a fixed time, keeps a fast machine from finishing the scan first
            def interrupt_once_the_scan_runs():
                while len(calls) < 10000:
                    time.sleep(0.001)
                _thread.interrupt_main()

            threading.Thread(target=interrupt_once_the_scan_runs, daemon=True).start()
            try:
                app.flush()
            except KeyboardInterrupt as error:
                print("interrupted:", type(error).__name__, flush=True)
            con.run("DELETE FROM big")
            app.flush()
            print("kept:", frame.sql("SELECT * FROM t").rows(con))
            """,
            # Ending the flush waits for its running scan tasks, so the child lasts about as long as the whole scan
            timeout=60,
        )
        assert code == 0, output
        assert "interrupted: KeyboardInterrupt" in output
        assert "kept: [(1, 'a')]" in output

    @pytest.mark.parametrize(
        "row",
        [pytest.param((1, "again"), id="primary_key"), pytest.param((None, "no key"), id="not_null")],
    )
    def test_a_failed_flush_drops_the_rows_and_raises(
        self, con: frame.Connection, row: tuple[PythonValue, ...]
    ) -> None:
        con.run("INSERT INTO t VALUES (1, 'a')")
        app = con.appender("t")
        app.append_row(2, "b")
        app.append_row(*row)
        with refused(exceptions.ConstraintError):
            app.flush()
        app.append_row(3, "c")
        app.flush()
        assert table_rows(con) == [(1, "a"), (3, "c")]

    def test_a_failed_flush_inside_begin_aborts_the_transaction(self, con: frame.Connection) -> None:
        con.run("INSERT INTO t VALUES (1, 'a')")
        app = con.appender("t")
        con.run("BEGIN")
        con.run("INSERT INTO t VALUES (2, 'b')")
        app.append_row(1, "again")
        with refused(exceptions.ConstraintError):
            app.flush()
        with refused(exceptions.TransactionError):
            con.run("SELECT 1")
        con.run("ROLLBACK")
        assert table_rows(con) == [(1, "a")]

    def test_a_flush_inside_an_aborted_transaction_drops_the_rows(self, con: frame.Connection) -> None:
        app = con.appender("t")
        con.run("BEGIN")
        with refused(exceptions.InvalidInputError):
            con.run("SELECT error('boom')")
        app.append_row(1, "a")
        with refused(exceptions.TransactionError):
            app.flush()
        con.run("ROLLBACK")
        app.flush()
        assert table_rows(con) == []


class TestBuffer:
    @pytest.fixture
    def tight(self) -> Iterator[frame.Connection]:
        connection = frame.connect()
        run_all(
            connection,
            "SET memory_limit = '16MB'",
            "SET temp_directory = ''",
            "CREATE TABLE s (k INTEGER, v VARCHAR)",
        )
        yield connection
        connection.close()

    def test_a_failed_write_into_the_buffer_breaks_the_appender_until_clear(self, tight: frame.Connection) -> None:
        app = tight.appender("s")
        # Rows reach the limited memory a whole chunk at a time, so one chunk must outgrow the limit.
        with refused(exceptions.OutOfMemoryError):
            append_until_full(app, (0, "x" * (16 << 10)), 4 * VECTOR)
        with refused(BROKEN, match="clear"):
            app.append_row(1, "a")
        with refused(BROKEN, match="clear"):
            app.append_rows([(1, "a")])
        with refused(BROKEN, match="clear"):
            app.flush()
        app.clear()
        app.append_row(1, "a")
        app.flush()
        assert table_rows(tight, "s") == [(1, "a")]

    def test_a_buffer_larger_than_the_memory_limit_spills_and_arrives_whole(self, tmp_path: Path) -> None:
        connection = frame.connect((tmp_path / "spill.db").as_posix())
        try:
            run_all(
                connection,
                "SET memory_limit = '16MB'",
                f"SET temp_directory = '{(tmp_path / 'spill').as_posix()}'",
                # The flush's parallel scan can pin two 8 MiB chunks at once, which the 16 MB limit cannot hold.
                "SET threads = 1",
                "CREATE TABLE s (k INTEGER, v VARCHAR)",
            )
            app = connection.appender("s")
            # A staged chunk of 8 MiB fits the limit and the 64 MiB buffer does not, so only spilling lets it arrive.
            rows, value = 8 * VECTOR, "x" * (4 << 10)
            app.append_rows((i, value) for i in range(rows))
            app.flush()
            expected = [(rows, rows * (rows - 1) // 2, rows * len(value))]
            assert frame.sql("SELECT count(*), sum(k), sum(length(v)) FROM s").rows(connection) == expected
        finally:
            connection.close()


class TestNothingFlushesOnItsOwn:
    @pytest.mark.parametrize(
        "statement",
        [
            pytest.param(lambda c: c.run("SELECT 1"), id="run"),
            pytest.param(lambda c: run_all(c, "BEGIN", "COMMIT"), id="begin_and_commit"),
            pytest.param(lambda c: empty_transaction(c), id="transaction_block"),
            pytest.param(lambda c: c.run("CHECKPOINT"), id="checkpoint"),
        ],
    )
    def test_other_statements_do_not_flush(
        self, con: frame.Connection, statement: Callable[[frame.Connection], object]
    ) -> None:
        app = con.appender("t")
        app.append_row(1, "a")
        statement(con)
        assert table_rows(con) == []
        discard(app)

    def test_many_rows_stay_buffered(self, con: frame.Connection) -> None:
        con.run("CREATE TABLE log (k INTEGER)")
        app = con.appender("log")
        app.append_rows((i,) for i in range(ENGINE_FLUSH_COUNT + 1))
        assert frame.sql("SELECT count(*) FROM log").rows(con) == [(0,)]
        discard(app)

    def test_garbage_collecting_buffered_rows_warns_and_drops_them(self, con: frame.Connection) -> None:
        app = con.appender("t")
        app.append_row(1, "a")
        with pytest.warns(ResourceWarning):
            del app
        assert table_rows(con) == []

    def test_a_resource_warning_turned_into_an_error_goes_to_the_unraisable_hook(
        self, con: frame.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = con.appender("t")
        app.append_row(1, "a")
        unraisable = []
        # Keeping the hook's argument would keep its traceback, and with it this frame and the connection, alive.
        monkeypatch.setattr(sys, "unraisablehook", lambda hook: unraisable.append(hook.exc_type))
        with warnings.catch_warnings():
            warnings.simplefilter("error", ResourceWarning)
            del app
        assert unraisable == [ResourceWarning]
        assert table_rows(con) == []

    def test_garbage_collecting_buffered_rows_warns_after_the_connection_is_closed(self, con: frame.Connection) -> None:
        app = con.appender("t")
        app.append_row(1, "a")
        con.close()
        with pytest.warns(ResourceWarning):
            del app

    @pytest.mark.parametrize(
        "prepare",
        [
            pytest.param(lambda app: app.append_row(1, "a") or app.flush(), id="flushed"),
            pytest.param(lambda app: app.append_row(1, "a") or app.clear(), id="cleared"),
            pytest.param(lambda app: None, id="never_appended"),
        ],
    )
    def test_garbage_collecting_an_appender_without_rows_is_silent(
        self, con: frame.Connection, prepare: Callable[[frame.Appender], object]
    ) -> None:
        app = con.appender("t")
        prepare(app)
        gc.collect()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            del app
            gc.collect()
        assert [w for w in caught if issubclass(w.category, ResourceWarning)] == []


class TestWithBlock:
    def test_normal_exit_flushes(self, con: frame.Connection) -> None:
        with con.appender("t") as app:
            app.append_row(1, "a")
        assert table_rows(con) == [(1, "a")]

    def test_a_failed_flush_on_normal_exit_propagates(self, con: frame.Connection) -> None:
        con.run("INSERT INTO t VALUES (1, 'a')")
        with refused(exceptions.ConstraintError), con.appender("t") as app:
            app.append_row(1, "again")

    def test_a_refused_flush_on_normal_exit_propagates_and_keeps_the_rows(self, con: frame.Connection) -> None:
        app = con.appender("t")
        rows = open_result(con)
        with refused(BUSY, match=LIVE_RESULT), app:
            app.append_row(1, "a")
        assert list(rows) == REST_OF_OPEN_RESULT
        app.flush()
        assert table_rows(con) == [(1, "a")]

    @pytest.mark.parametrize("raised", [ValueError, KeyboardInterrupt])
    def test_exit_by_exception_clears_and_the_exception_propagates(
        self, con: frame.Connection, raised: type[BaseException]
    ) -> None:
        app = con.appender("t")
        with pytest.raises(raised), app:
            append_then_raise(app, raised)
        app.append_row(2, "b")
        app.flush()
        assert table_rows(con) == [(2, "b")]

    def test_exit_by_exception_after_the_connection_is_closed_propagates_the_exception(
        self, con: frame.Connection
    ) -> None:
        app = con.appender("t")
        with pytest.raises(ValueError, match="the block failed"), app:
            append_close_then_raise(app, con, ValueError("the block failed"))
        gc.collect()
        # The exit cleared the row, so letting go of the appender has nothing to warn about.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            del app
            gc.collect()
        assert [w for w in caught if issubclass(w.category, ResourceWarning)] == []

    def test_the_appender_is_usable_across_with_blocks(self, con: frame.Connection) -> None:
        app = con.appender("t")
        with app:
            app.append_row(1, "a")
        with app:
            app.append_row(2, "b")
        assert table_rows(con) == [(1, "a"), (2, "b")]


class TestOneCallerAtATime:
    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(lambda app: app.append_row(9, "z"), id="append_row"),
            pytest.param(lambda app: app.append_rows([(9, "z")]), id="append_rows"),
            pytest.param(lambda app: app.flush(), id="flush"),
            pytest.param(lambda app: app.clear(), id="clear"),
        ],
    )
    def test_another_thread_is_refused_during_a_flush_without_waiting(
        self,
        con: frame.Connection,
        held: tuple[threading.Event, threading.Event],
        call: Callable[[frame.Appender], object],
    ) -> None:
        running, release = held
        app = held_flush_appender(con)
        app.append_row(1, "a")
        outcome: list[object] = []
        thread = threading.Thread(target=contend, args=(running, release, lambda: call(app), outcome, exceptions.Error))
        thread.start()
        app.flush()
        thread.join(10)
        assert [type(o) for o in outcome] == [CONCURRENT_USE]
        app.flush()
        assert table_rows(con) == [(1, "a")]

    @pytest.mark.parametrize("call", ["app.append_row(9, 'z')", "app.flush()", "app.clear()"])
    def test_a_callback_during_its_flush_is_refused(self, call: str) -> None:
        code, output = run_isolated(
            f"""
            from duckdb import exceptions, frame
            con = frame.connect()
            con.run("CREATE TABLE t (k INTEGER PRIMARY KEY, v VARCHAR)")
            def reenter(x):
                try:
                    {call}
                    print("ran", flush=True)
                except exceptions.Error as error:
                    print("refused:", type(error).__name__, flush=True)
                return x
            con.create_function("reenter", reenter, ["BIGINT"], "BIGINT", stability="volatile")
            app = con.query_appender("INSERT INTO t SELECT reenter(k), v FROM appender_data", {KV!r})
            app.append_row(1, "a")
            print("started", flush=True)
            app.flush()
            print("after:", frame.sql("SELECT * FROM t").rows(con))
            """
        )
        assert code == 0, output
        assert f"refused: {CONCURRENT_USE.__name__}\n" in output
        assert "after: [(1, 'a')]" in output

    @pytest.mark.parametrize(
        ("action", "key", "row", "expected"),
        [
            pytest.param("insert", None, (2, "b"), [(1, "a"), (2, "b")], id="insert"),
            pytest.param("upsert", None, (1, "A"), [(1, "A")], id="upsert"),
            pytest.param("update", ["k"], (1, "A"), [(1, "A")], id="update"),
        ],
    )
    def test_another_thread_is_refused_during_a_table_flush_without_waiting(
        self,
        con: frame.Connection,
        action: Action,
        key: list[str] | None,
        row: tuple[PythonValue, ...],
        expected: list[tuple[object, ...]],
    ) -> None:
        running, release, armed = threading.Event(), threading.Event(), threading.Event()

        def hold(x: int) -> int:
            if armed.is_set():
                running.set()
                release.wait(10)
            return x

        con.create_function("hold_write", hold, ["BIGINT"], "BIGINT", stability="volatile")
        con.run("CREATE TABLE h (k INTEGER PRIMARY KEY, v VARCHAR CHECK (hold_write(length(v)) >= 0))")
        con.run("INSERT INTO h VALUES (1, 'a')")
        # Armed only now, so the CHECK holds the flush and not the row the table starts with.
        armed.set()
        app = con.appender("h", action=action, key=key)
        app.append_row(*row)
        outcome: list[object] = []
        thread = threading.Thread(
            target=contend, args=(running, release, lambda: app.append_row(9, "z"), outcome, exceptions.Error)
        )
        thread.start()
        app.flush()
        thread.join(10)
        assert [type(o) for o in outcome] == [CONCURRENT_USE]
        assert table_rows(con, "h") == expected

    @pytest.mark.parametrize(
        ("action", "key", "row", "expected"),
        [
            pytest.param("insert", None, (2, "b"), [(1, "a"), (2, "b")], id="insert"),
            pytest.param("upsert", None, (1, "A"), [(1, "A")], id="upsert"),
            pytest.param("update", ["k"], (1, "A"), [(1, "A")], id="update"),
        ],
    )
    def test_a_check_that_reenters_its_table_flush_is_refused(
        self,
        action: Action,
        key: list[str] | None,
        row: tuple[PythonValue, ...],
        expected: list[tuple[object, ...]],
    ) -> None:
        code, output = run_isolated(
            f"""
            from duckdb import exceptions, frame
            con = frame.connect()
            app = None
            def reenter(x):
                # Only the flush reenters: the appender does not exist yet when the table gets its first row.
                if app is not None:
                    try:
                        app.append_row(9, "z")
                        print("ran", flush=True)
                    except exceptions.Error as error:
                        print("refused:", type(error).__name__, flush=True)
                return x
            con.create_function("reenter", reenter, ["BIGINT"], "BIGINT", stability="volatile")
            con.run("CREATE TABLE t (k INTEGER PRIMARY KEY, v VARCHAR CHECK (reenter(length(v)) >= 0))")
            con.run("INSERT INTO t VALUES (1, 'a')")
            app = con.appender("t", action={action!r}, key={key!r})
            app.append_row(*{row!r})
            print("started", flush=True)
            app.flush()
            print("after:", frame.sql("SELECT * FROM t ORDER BY ALL").rows(con))
            """
        )
        assert code == 0, output
        assert f"refused: {CONCURRENT_USE.__name__}\n" in output
        assert f"after: {expected}" in output

    @pytest.mark.parametrize("caller", ["the_flush_itself", "another_thread"])
    def test_a_delete_flush_refuses_a_second_caller_without_waiting(self, caller: str) -> None:
        code, output = run_isolated(
            f"""
            import threading
            from duckdb import dbapi, exceptions
            connection = dbapi.connect()
            cursor = connection.cursor()
            cursor.execute("CREATE TABLE t (k INTEGER PRIMARY KEY, v VARCHAR)")
            cursor.execute("INSERT INTO t VALUES (1, 'a'), (2, 'b')")
            connection.commit()
            app = connection.appender("t", action="delete", key="k")
            app.append_row(1)
            begin = connection._begin_if_needed
            def second_caller():
                try:
                    app.append_row(2)
                    print("ran", flush=True)
                except exceptions.Error as error:
                    print("refused:", type(error).__name__, flush=True)
            # A delete runs no expression of the table's, so a second caller meets the flush in its transaction
            # hook, which runs while the flush holds the appender.
            def begin_beside_a_second_caller():
                if {caller!r} == "the_flush_itself":
                    second_caller()
                else:
                    thread = threading.Thread(target=second_caller)
                    thread.start()
                    thread.join(10)
                begin()
            connection._begin_if_needed = begin_beside_a_second_caller
            print("started", flush=True)
            app.flush()
            connection.commit()
            cursor.execute("SELECT * FROM t ORDER BY ALL")
            print("after:", cursor.fetchall())
            """
        )
        assert code == 0, output
        assert f"refused: {CONCURRENT_USE.__name__}\n" in output
        assert "after: [(2, 'b')]" in output


class TestClosedConnection:
    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(lambda app: app.append_row(2, "b"), id="append_row"),
            pytest.param(lambda app: app.append_rows([(2, "b")]), id="append_rows"),
            pytest.param(lambda app: app.flush(), id="flush"),
            pytest.param(lambda app: app.clear(), id="clear"),
            pytest.param(lambda app: app.__enter__(), id="enter"),
        ],
    )
    @pytest.mark.filterwarnings("ignore::ResourceWarning")
    def test_every_method_raises_once_the_connection_is_closed(
        self, con: frame.Connection, call: Callable[[frame.Appender], object]
    ) -> None:
        app = con.appender("t")
        app.append_row(1, "a")
        con.close()
        with refused(CLOSED):
            call(app)

    @BAD_QUERY_ARGUMENTS
    def test_a_closed_connection_is_reported_before_bad_query_arguments(
        self, con: frame.Connection, arguments: dict[str, Any], error: type[Exception]
    ) -> None:
        con.close()
        with refused(CLOSED):
            con.query_appender(**arguments)

    def test_an_empty_flush_raises_once_the_connection_is_closed(self, con: frame.Connection) -> None:
        app = con.appender("t")
        con.close()
        with refused(CLOSED):
            app.flush()

    @pytest.mark.parametrize(
        "options",
        [{}, {"action": "merge"}, {"action": "update"}],
        ids=["insert", "unknown_action", "update_without_key"],
    )
    def test_creating_an_appender_on_a_closed_connection_raises(
        self, con: frame.Connection, options: dict[str, Any]
    ) -> None:
        con.close()
        with refused(CLOSED):
            con.appender("t", **options)

    def test_reconnecting_to_the_file_works_while_an_appender_lives(self, tmp_path: Path) -> None:
        code, output = run_isolated(
            f"""
            from duckdb import frame
            path = {(tmp_path / "db.duckdb").as_posix()!r}
            con = frame.connect(path)
            con.run("CREATE TABLE t (k INTEGER PRIMARY KEY, v VARCHAR)")
            app = con.appender("t")
            app.append_row(1, "a")
            con.close()
            print("started", flush=True)
            again = frame.connect(path)
            print("rows:", frame.sql("SELECT * FROM t").rows(again))
            """
        )
        assert code == 0, output
        assert "rows: []" in output

    def test_a_transaction_open_at_close_is_released_while_an_appender_lives(self, tmp_path: Path) -> None:
        code, output = run_isolated(
            f"""
            from duckdb import frame
            path = {(tmp_path / "db.duckdb").as_posix()!r}
            con = frame.connect(path)
            other = con.duplicate()
            con.run("CREATE TABLE t (k INTEGER PRIMARY KEY, v VARCHAR)")
            app = con.appender("t")
            app.append_row(1, "a")
            con.run("BEGIN")
            con.run("INSERT INTO t VALUES (2, 'uncommitted')")
            con.close()
            print("started", flush=True)
            other.run("INSERT INTO t VALUES (2, 'b')")
            print("rows:", frame.sql("SELECT * FROM t ORDER BY ALL").rows(other))
            """
        )
        assert code == 0, output
        assert "rows: [(2, 'b')]" in output

    @pytest.mark.parametrize("call", ["app.append_row(1, 2)", "app.clear()"])
    def test_closing_the_connection_inside_a_row_raises_instead_of_crashing(self, call: str) -> None:
        code, output = run_isolated(
            f"""
            from duckdb import exceptions, frame
            con = frame.connect()
            con.run("CREATE TABLE t (a BIGINT, b BIGINT)")
            app = con.appender("t")
            class Row:
                def __len__(self):
                    return 2
                def __getitem__(self, i):
                    if i == 1:
                        con.close()
                    if i >= 2:
                        raise IndexError
                    return 7
            print("started", flush=True)
            try:
                app.append_rows([Row()])
                print("append_rows: returned")
            except exceptions.Error as error:
                print("append_rows:", type(error).__name__)
            try:
                {call}
                print("next call: returned")
            except exceptions.Error as error:
                print("next call:", type(error).__name__)
            """
        )
        assert code == 0, output
        assert re.search(rf"append_rows: (returned|{CLOSED.__name__})\n", output), output
        assert f"next call: {CLOSED.__name__}" in output

    def test_reconnecting_from_inside_the_rows_iterator_raises_at_the_next_row(self, tmp_path: Path) -> None:
        code, output = run_isolated(
            f"""
            from duckdb import exceptions, frame
            path = {(tmp_path / "db.duckdb").as_posix()!r}
            con = frame.connect(path)
            con.run("CREATE TABLE t (i INTEGER)")
            app = con.appender("t")
            def rows():
                yield (1,)
                con.close()
                frame.connect(path).close()
                yield (2,)
            print("started", flush=True)
            try:
                app.append_rows(rows())
                print("append_rows: returned")
            except exceptions.Error as error:
                print("append_rows:", type(error).__name__)
            print("rows:", frame.sql("SELECT * FROM t").rows(frame.connect(path)))
            """
        )
        assert code == 0, output
        assert f"append_rows: {CLOSED.__name__}" in output
        assert "rows: []" in output

    @pytest.mark.parametrize(
        "create",
        [
            pytest.param('con.appender("t", columns=reconnecting_names())', id="table"),
            pytest.param('con.query_appender("INSERT INTO t FROM appender_data", ReconnectingColumns())', id="query"),
        ],
    )
    def test_reconnecting_while_the_column_names_are_read_raises_at_creation(self, tmp_path: Path, create: str) -> None:
        code, output = run_isolated(
            f"""
            import collections.abc
            from duckdb import exceptions, frame
            path = {(tmp_path / "db.duckdb").as_posix()!r}
            con = frame.connect(path)
            con.run("CREATE TABLE t (i INTEGER)")
            def reconnect():
                con.close()
                frame.connect(path).close()
            def reconnecting_names():
                reconnect()
                yield "i"
            class ReconnectingColumns(collections.abc.Mapping):
                def __iter__(self):
                    reconnect()
                    return iter(["i"])
                def __getitem__(self, name):
                    return "INTEGER"
                def __len__(self):
                    return 1
            print("started", flush=True)
            try:
                {create}
                print("creation: returned")
            except exceptions.Error as error:
                print("creation:", type(error).__name__)
            print("rows:", frame.sql("SELECT * FROM t").rows(frame.connect(path)))
            """
        )
        assert code == 0, output
        assert f"creation: {CLOSED.__name__}" in output
        assert "rows: []" in output


class TestDbapiConnection:
    """The DB-API connection carries the same two methods; there are no module-level forms."""

    @pytest.fixture
    def dcon(self) -> Iterator[dbapi.Connection]:
        connection = dbapi.connect()
        connection.cursor().execute("CREATE TABLE t (k INTEGER PRIMARY KEY, v VARCHAR)")
        yield connection
        connection.close()

    @staticmethod
    def rows(connection: dbapi.Connection, table: str = "t") -> list[tuple[Any, ...]]:
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT * FROM {table} ORDER BY ALL")
            return cursor.fetchall()

    def test_the_appender_lives_on_the_connections_only(self) -> None:
        assert hasattr(frame.Connection, "appender")
        assert hasattr(frame.Connection, "query_appender")
        assert hasattr(dbapi.Connection, "appender")
        assert hasattr(dbapi.Connection, "query_appender")
        assert frame.Appender is dbapi.Appender
        assert not hasattr(duckdb, "appender")
        assert not hasattr(duckdb, "query_appender")

    def test_rows_go_into_the_table(self, dcon: dbapi.Connection) -> None:
        app = dcon.appender("t")
        app.append_row(1, "a")
        app.flush()
        assert self.rows(dcon) == [(1, "a")]

    def test_an_action_and_a_key(self, dcon: dbapi.Connection) -> None:
        dcon.cursor().execute("INSERT INTO t VALUES (1, 'a'), (2, 'b')")
        app = dcon.appender("t", action="delete", key=["k"])
        app.append_row(1)
        app.flush()
        assert self.rows(dcon) == [(2, "b")]

    @pytest.mark.parametrize(
        ("action", "key", "row", "inside"),
        [
            pytest.param("upsert", None, (1, "A"), [(1, "A"), (2, "b")], id="upsert"),
            pytest.param("update", ["k"], (2, "B"), [(1, "a"), (2, "B")], id="update"),
        ],
    )
    def test_an_upsert_or_update_joins_the_transaction_and_rolls_back_with_it(
        self,
        dcon: dbapi.Connection,
        action: Action,
        key: list[str] | None,
        row: tuple[PythonValue, ...],
        inside: list[tuple[object, ...]],
    ) -> None:
        dcon.cursor().execute("INSERT INTO t VALUES (1, 'a'), (2, 'b')")
        dcon.commit()
        app = dcon.appender("t", action=action, key=key)
        app.append_row(*row)
        app.flush()
        assert self.rows(dcon) == inside
        dcon.rollback()
        assert self.rows(dcon) == [(1, "a"), (2, "b")]

    def test_a_query_appender(self, dcon: dbapi.Connection) -> None:
        app = dcon.query_appender("INSERT INTO t FROM appender_data", KV)
        app.append_row(1, "a")
        app.flush()
        assert self.rows(dcon) == [(1, "a")]

    def test_a_refused_type_name_fails_the_transaction_the_connection_opened(self, dcon: dbapi.Connection) -> None:
        dcon.commit()
        dcon.cursor().execute("INSERT INTO t VALUES (1, 'a')")
        with refused(exceptions.CatalogError):
            dcon.query_appender("INSERT INTO t (k) FROM appender_data", {"k": "no_such_type"})
        with refused(exceptions.TransactionError):
            dcon.cursor().execute("INSERT INTO t VALUES (2, 'b')")
        dcon.rollback()
        assert self.rows(dcon) == []

    def test_commit_does_not_flush(self, dcon: dbapi.Connection) -> None:
        app = dcon.appender("t")
        app.append_row(1, "a")
        dcon.commit()
        assert self.rows(dcon) == []
        discard(app)

    def test_a_flush_joins_the_transaction_the_connection_holds_open(self, dcon: dbapi.Connection) -> None:
        dcon.commit()
        app = dcon.appender("t")
        app.append_row(1, "a")
        app.flush()
        dcon.rollback()
        assert self.rows(dcon) == []
        app.append_row(2, "b")
        app.flush()
        dcon.commit()
        dcon.rollback()
        assert self.rows(dcon) == [(2, "b")]

    def test_a_flush_under_autocommit_commits_by_itself(self, tmp_path: Path) -> None:
        path = tmp_path / "autocommit.db"
        connection = dbapi.connect(path, autocommit=True)
        try:
            connection.cursor().execute("CREATE TABLE t (k INTEGER PRIMARY KEY, v VARCHAR)")
            app = connection.appender("t")
            app.append_row(1, "a")
            app.flush()
            del app
        finally:
            connection.close()
        # A transaction the flush left open ends with its connection, so only a committed row survives the reopen.
        reopened = dbapi.connect(path)
        try:
            assert self.rows(reopened) == [(1, "a")]
        finally:
            reopened.close()

    def test_a_flush_refused_while_a_cursor_holds_an_unread_result_keeps_the_rows(self, dcon: dbapi.Connection) -> None:
        app = dcon.appender("t")
        app.append_row(1, "a")
        reading = dcon.cursor()
        reading.execute(f"SELECT * FROM range({OPEN_ROWS})")
        assert reading.fetchone() == (0,)
        with refused(BUSY, match=LIVE_RESULT):
            app.flush()
        assert reading.fetchall() == REST_OF_OPEN_RESULT
        app.flush()
        assert self.rows(dcon) == [(1, "a")]

    @pytest.mark.requires("pyarrow")
    def test_a_flush_refused_while_another_call_runs_keeps_the_rows(self, dcon: dbapi.Connection) -> None:
        import pyarrow as pa

        running, release = threading.Event(), threading.Event()

        # The DB-API takes no Python function, so a reader's generator holds the scan, and with it the call, open.
        def batches() -> Iterator[pa.RecordBatch]:
            yield pa.record_batch({"i": [0]})
            running.set()
            release.wait(10)
            yield pa.record_batch({"i": [1]})

        dcon.register("slow", pa.RecordBatchReader.from_batches(pa.schema([("i", pa.int64())]), batches()))
        app = dcon.appender("t")
        app.append_row(1, "a")
        outcome: list[object] = []
        thread = threading.Thread(target=contend, args=(running, release, app.flush, outcome, exceptions.Error))
        thread.start()
        reading = dcon.cursor()
        reading.execute("SELECT * FROM slow")
        assert reading.fetchall() == [(0,), (1,)]
        thread.join(10)
        assert [type(o) for o in outcome] == [BUSY]
        app.flush()
        assert self.rows(dcon) == [(1, "a")]

    def test_an_empty_flush_opens_no_transaction(self, dcon: dbapi.Connection) -> None:
        dcon.commit()
        app = dcon.appender("t")
        app.flush()
        assert not dcon._in_transaction

    def test_another_thread_cannot_append_while_a_flush_decides_on_its_transaction(
        self, dcon: dbapi.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dcon.commit()
        app = dcon.appender("t")
        app.append_row(1, "a")
        outcome: list[object] = []
        begin = dcon._begin_if_needed

        # The flush decides whether to open the implicit transaction while it holds the appender.
        def begin_while_another_thread_appends() -> None:
            def append() -> None:
                try:
                    app.append_row(42, "late")
                    outcome.append("ran")
                except exceptions.Error as error:
                    outcome.append(error)

            thread = threading.Thread(target=append)
            thread.start()
            thread.join(10)
            begin()

        monkeypatch.setattr(dcon, "_begin_if_needed", begin_while_another_thread_appends)
        app.flush()
        monkeypatch.undo()
        assert [type(o) for o in outcome] == [CONCURRENT_USE]
        assert self.rows(dcon) == [(1, "a")]
        dcon.rollback()
        assert self.rows(dcon) == []
