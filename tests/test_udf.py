"""Python scalar functions: registration, execution, and failure."""

from __future__ import annotations

import datetime
import decimal
import gc
import subprocess
import sys
import threading
import time
import weakref
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

import duckdb
from duckdb import _duckdb, exceptions
from duckdb._expressions.expr import render_literal
from duckdb.frame import col, fn

from ._support import Emptying, Failing

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


@pytest.fixture
def con() -> duckdb.frame.Connection:
    return duckdb.frame.connect()


def rows(con: duckdb.frame.Connection, sql: str) -> list[tuple[object, ...]]:
    with con._execute(sql) as result:
        return result.fetch_all()


# The extension module takes enum values; the string forms belong to duckdb.frame.Connection.
DEFAULT = _duckdb.FunctionNullHandling.DEFAULT
CONSISTENT = _duckdb.FunctionStability.CONSISTENT


class TestRegistration:
    def test_a_function_is_callable_from_sql(self, con: duckdb.frame.Connection) -> None:
        con.create_function("plus_one", lambda x: x + 1, ["BIGINT"], "BIGINT")
        assert rows(con, "SELECT plus_one(i) FROM range(5) t(i)") == [(1,), (2,), (3,), (4,), (5,)]

    def test_a_function_is_callable_from_a_plan(self, con: duckdb.frame.Connection) -> None:
        con.create_function("plus_one", lambda x: x + 1, ["BIGINT"], "BIGINT")
        frame = duckdb.frame.sql("SELECT * FROM range(3) t(i)").select(fn("plus_one", col("i")).alias("j"))
        assert frame.on(con).rows() == [(1,), (2,), (3,)]

    def test_multiple_arguments(self, con: duckdb.frame.Connection) -> None:
        con.create_function("weave", lambda a, b, c: f"{a}-{b}-{c}", ["VARCHAR", "BIGINT", "BOOLEAN"], "VARCHAR")
        assert rows(con, "SELECT weave('x', 7, true)") == [("x-7-True",)]

    def test_zero_arguments(self, con: duckdb.frame.Connection) -> None:
        con.create_function("answer", lambda: 42, [], "INTEGER", stability="volatile")
        assert rows(con, "SELECT answer()") == [(42,)]

    def test_a_closure_carries_its_state(self, con: duckdb.frame.Connection) -> None:
        prefix = "state"
        con.create_function("tag", lambda x: f"{prefix}:{x}", ["BIGINT"], "VARCHAR")
        assert rows(con, "SELECT tag(1)") == [("state:1",)]

    def test_registering_the_name_again_replaces_the_function(self, con: duckdb.frame.Connection) -> None:
        con.create_function("f", lambda x: x, ["BIGINT"], "BIGINT")
        con.create_function("f", lambda x: x * 10, ["BIGINT"], "BIGINT")
        assert rows(con, "SELECT f(2)") == [(20,)]

    def test_an_unknown_type_text_is_refused(self, con: duckdb.frame.Connection) -> None:
        with pytest.raises(exceptions.Error, match="NO_SUCH_TYPE"):
            con.create_function("f", lambda x: x, ["NO_SUCH_TYPE"], "BIGINT")

    def test_a_bad_null_handling_is_refused(self, con: duckdb.frame.Connection) -> None:
        with pytest.raises(exceptions.InvalidInputError, match="null_handling"):
            con.create_function("f", lambda x: x, ["BIGINT"], "BIGINT", null_handling="never")

    def test_a_bad_stability_is_refused(self, con: duckdb.frame.Connection) -> None:
        with pytest.raises(exceptions.InvalidInputError, match="stability"):
            con.create_function("f", lambda x: x, ["BIGINT"], "BIGINT", stability="jittery")

    def test_a_closed_connection_refuses(self, con: duckdb.frame.Connection) -> None:
        con.close()
        with pytest.raises(exceptions.InterfaceError):
            con.create_function("f", lambda x: x, ["BIGINT"], "BIGINT")

    @pytest.mark.parametrize("text", ["ANY", "any", " ANY "])
    def test_an_any_parameter_is_refused(self, con: duckdb.frame.Connection, text: str) -> None:
        with pytest.raises(exceptions.InvalidInputError) as info:
            con.create_function("f", lambda x, y: x, ["BIGINT", text], "BIGINT")
        assert str(info.value) == "Invalid Input Error: ANY parameters are not supported yet"

    @pytest.mark.parametrize("text", ["ANY", "any", " ANY "])
    def test_an_any_return_type_is_refused(self, con: duckdb.frame.Connection, text: str) -> None:
        with pytest.raises(exceptions.InvalidInputError) as info:
            con.create_function("f", lambda x: x, ["BIGINT"], text)
        assert str(info.value) == "Invalid Input Error: an ANY return type is not supported yet"

    def test_a_nested_any_gets_the_parsers_refusal(self, con: duckdb.frame.Connection) -> None:
        with pytest.raises(exceptions.InvalidInputError, match="can not be converted to a DuckDB Type"):
            con.create_function("f", lambda x: x, ["ANY[]"], "BIGINT")


class TestValues:
    def test_scalar_types_round_trip(self, con: duckdb.frame.Connection) -> None:
        con.create_function("echo", lambda x: x, ["VARCHAR"], "VARCHAR")
        con.create_function("echo_d", lambda x: x, ["DOUBLE"], "DOUBLE")
        con.create_function("echo_b", lambda x: x, ["BLOB"], "BLOB")
        con.create_function("echo_dt", lambda x: x, ["TIMESTAMP"], "TIMESTAMP")
        con.create_function("echo_dec", lambda x: x, ["DECIMAL(9,3)"], "DECIMAL(9,3)")
        got = rows(
            con,
            "SELECT echo('hi'), echo_d(1.5), echo_b('ab'::BLOB), "
            "echo_dt(TIMESTAMP '2020-06-01 12:30:00'), echo_dec(1.250::DECIMAL(9,3))",
        )
        assert got == [
            (
                "hi",
                1.5,
                b"ab",
                datetime.datetime(2020, 6, 1, 12, 30),
                decimal.Decimal("1.250"),
            )
        ]

    def test_nested_arguments_arrive_as_python_values(self, con: duckdb.frame.Connection) -> None:
        con.create_function("total", lambda xs: sum(xs), ["BIGINT[]"], "BIGINT")
        con.create_function("field", lambda s: s["a"], ["STRUCT(a INTEGER)"], "INTEGER")
        assert rows(con, "SELECT total([1, 2, 3]), field({'a': 7})") == [(6, 7)]

    def test_nested_returns_are_converted(self, con: duckdb.frame.Connection) -> None:
        con.create_function("pair", lambda x: [x, x + 1], ["BIGINT"], "BIGINT[]")
        con.create_function("wrap", lambda x: {"v": x}, ["BIGINT"], "STRUCT(v BIGINT)")
        assert rows(con, "SELECT pair(3), wrap(9)") == [([3, 4], {"v": 9})]

    def test_the_return_is_cast_to_the_declared_type(self, con: duckdb.frame.Connection) -> None:
        con.create_function("half", lambda x: x / 2, ["BIGINT"], "INTEGER")
        assert rows(con, "SELECT half(9)") == [(4,)]

    def test_an_uncastable_return_fails_loudly(self, con: duckdb.frame.Connection) -> None:
        con.create_function("wide", lambda x: 2**70, ["BIGINT"], "INTEGER")
        with pytest.raises(exceptions.Error):
            rows(con, "SELECT wide(1)")

    def test_an_unbindable_return_object_fails_loudly(self, con: duckdb.frame.Connection) -> None:
        con.create_function("obj", lambda x: object(), ["BIGINT"], "VARCHAR")
        with pytest.raises(exceptions.Error, match="returned a value of type object"):
            rows(con, "SELECT obj(1)")


class TestNulls:
    def test_default_null_handling_skips_the_function(self, con: duckdb.frame.Connection) -> None:
        calls: list[object] = []

        def observe(x: object) -> object:
            calls.append(x)
            return x

        con.create_function("observe", observe, ["BIGINT"], "BIGINT")
        assert rows(con, "SELECT observe(x) FROM (VALUES (1), (NULL), (3)) t(x)") == [(1,), (None,), (3,)]
        assert calls == [1, 3]

    def test_special_null_handling_passes_none_through(self, con: duckdb.frame.Connection) -> None:
        con.create_function("backfill", lambda x: -1 if x is None else x, ["BIGINT"], "BIGINT", null_handling="special")
        assert rows(con, "SELECT backfill(x) FROM (VALUES (1), (NULL)) t(x)") == [(1,), (-1,)]

    def test_returning_none_makes_the_result_null(self, con: duckdb.frame.Connection) -> None:
        con.create_function("odd_only", lambda x: x if x % 2 else None, ["BIGINT"], "BIGINT")
        assert rows(con, "SELECT odd_only(i) FROM range(4) t(i)") == [(None,), (1,), (None,), (3,)]


class TestUnbindableReturns:
    def test_an_unbindable_return_fails_the_query(self, con: duckdb.frame.Connection) -> None:
        con.create_function("obj2", lambda x: object(), ["BIGINT"], "VARCHAR")
        with pytest.raises(exceptions.Error, match="returned a value of type object"):
            rows(con, "SELECT obj2(1)")


class TestExecution:
    def test_every_row_of_a_large_input_is_converted(self, con: duckdb.frame.Connection) -> None:
        con.create_function("twice", lambda x: x * 2, ["BIGINT"], "BIGINT")
        got = rows(con, "SELECT sum(twice(i)), count(*) FROM range(10000) t(i)")
        assert got == [(2 * sum(range(10000)), 10000)]

    def test_a_volatile_function_runs_per_row(self, con: duckdb.frame.Connection) -> None:
        counter = iter(range(1000000))
        con.create_function("tick", lambda: next(counter), [], "BIGINT", stability="volatile")
        got = rows(con, "SELECT tick() FROM range(100) t(i)")
        assert {value for (value,) in got} == set(range(100))

    def test_engine_workers_run_the_function_during_a_fetch(self, con: duckdb.frame.Connection) -> None:
        con.run("SET threads = 4")
        seen: set[int] = set()

        def bump(x: int) -> int:
            seen.add(threading.get_ident())
            return x + 1

        con.create_function("bump", bump, ["BIGINT"], "BIGINT")
        con.run("CREATE TABLE big AS SELECT i FROM range(1000000) t(i)")
        assert rows(con, "SELECT sum(bump(i)) FROM big") == [(sum(range(1000000)) + 1000000,)]
        # More than one thread proves the fetch released the global interpreter lock; holding it would deadlock.
        assert len(seen) >= 2

    def test_the_function_runs_inside_other_threads_queries(self, con: duckdb.frame.Connection) -> None:
        con.create_function("slow_id", lambda x: x, ["BIGINT"], "BIGINT")
        results: list[list[tuple[object, ...]]] = []

        def work() -> None:
            other = con.duplicate()
            results.append(rows(other, "SELECT sum(slow_id(i)) FROM range(1000) t(i)"))

        threads = [threading.Thread(target=work) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert results == [[(499500,)]] * 3


class TestFailure:
    def test_a_python_error_names_the_function_and_the_cause(self, con: duckdb.frame.Connection) -> None:
        def boom(x: int) -> int:
            message = f"no thanks: {x}"
            raise ValueError(message)

        con.create_function("boom", boom, ["BIGINT"], "BIGINT")
        with pytest.raises(exceptions.InvalidInputError, match="boom") as info:
            rows(con, "SELECT boom(7)")
        message = str(info.value)
        assert "Python exception occurred while executing the UDF 'boom'" in message
        assert message.index("ValueError: no thanks: 7") < message.index("Traceback")

    def test_the_connection_survives_a_failing_function(self, con: duckdb.frame.Connection) -> None:
        con.create_function("bad", lambda x: 1 / 0, ["BIGINT"], "BIGINT")
        with pytest.raises(exceptions.Error):
            rows(con, "SELECT bad(1)")
        assert rows(con, "SELECT 42") == [(42,)]

    def test_a_return_the_declared_type_cannot_hold_carries_one_prefix(self, con: duckdb.frame.Connection) -> None:
        con.create_function("big", lambda x: 2**100, ["BIGINT"], "BIGINT")
        with pytest.raises(exceptions.InvalidInputError) as info:
            rows(con, "SELECT big(1)")
        message = str(info.value)
        assert message.startswith(
            "Invalid Input Error: the UDF 'big' returned a value that cannot be converted to its declared type BIGINT: "
            "Failed to cast value"
        )
        assert message.count("Invalid Input Error:") == 1

    @pytest.mark.parametrize("declared", ["TIMESTAMPTZ[]", "UNION(a TIMESTAMPTZ[], b DATE[])"], ids=["plain", "union"])
    def test_a_python_error_while_converting_the_return_keeps_its_own_wording(
        self, con: duckdb.frame.Connection, declared: str
    ) -> None:
        returned = datetime.datetime(2020, 1, 1, tzinfo=Failing(ValueError, "no offset today"))
        con.create_function("f", lambda _: [returned], ["BIGINT"], declared)
        with pytest.raises(exceptions.InvalidInputError) as info:
            rows(con, "SELECT f(1)")
        message = str(info.value)
        assert "Python exception occurred while executing the UDF 'f'" in message
        assert "ValueError: no offset today" in message

    @pytest.mark.parametrize(
        ("returned", "declared", "reason"),
        [
            (np.datetime64("1677-09-21T00:12:43.145224194"), "TIMESTAMP_NS", "beyond the range of its engine type"),
            (
                [datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC), datetime.datetime(2020, 1, 1)],
                "VARCHAR[]",
                "a list holds values with and without a time zone",
            ),
            ({"x": 1}, "STRUCT(a INTEGER)", "STRUCT to STRUCT cast must have at least one matching member"),
        ],
        ids=["out of range", "zones mixed", "no field matches"],
    )
    def test_every_conversion_failure_names_the_function(
        self, con: duckdb.frame.Connection, returned: object, declared: str, reason: str
    ) -> None:
        con.create_function("f", lambda _: returned, ["BIGINT"], declared)
        with pytest.raises(exceptions.InvalidInputError) as info:
            rows(con, "SELECT f(1)")
        message = str(info.value)
        assert message.startswith("Invalid Input Error: the UDF 'f' returned a value that cannot be converted")
        assert reason in message

    def test_an_unconvertible_return_names_the_function_and_the_type(self, con: duckdb.frame.Connection) -> None:
        con.create_function("o", lambda x: object(), ["BIGINT"], "BIGINT")
        with pytest.raises(exceptions.InvalidInputError) as info:
            rows(con, "SELECT o(1)")
        assert str(info.value) == (
            "Invalid Input Error: the UDF 'o' returned a value of type object, "
            "which cannot be converted to a DuckDB value"
        )

    def test_an_unconvertible_element_of_a_return_names_its_type(self, con: duckdb.frame.Connection) -> None:
        con.create_function("wrap", lambda x: [x, object()], ["BIGINT"], "BIGINT[]")
        with pytest.raises(exceptions.InvalidInputError, match="the UDF 'wrap' returned a value of type object"):
            rows(con, "SELECT wrap(1)")

    def test_a_conversion_refusal_inside_the_function_carries_one_prefix(self, con: duckdb.frame.Connection) -> None:
        con.create_function("nan", lambda x: decimal.Decimal("NaN"), ["BIGINT"], "DOUBLE")
        with pytest.raises(exceptions.InvalidInputError) as info:
            rows(con, "SELECT nan(1)")
        assert str(info.value) == (
            "Invalid Input Error: the UDF 'nan' returned a value that cannot be converted to its declared type DOUBLE: "
            "cannot bind a non-finite Decimal"
        )

    def test_parameter_binding_keeps_its_own_wording(self, con: duckdb.frame.Connection) -> None:
        with pytest.raises(exceptions.InvalidInputError) as info:
            con._execute("SELECT $1", [object()])
        assert str(info.value) == "Invalid Input Error: cannot bind a parameter of type object"


class TestCollection:
    """A callable that holds its own connection is a reference cycle Python's collector must see."""

    def test_a_connection_held_by_a_bound_method_is_collected(self) -> None:
        class Service:
            def __init__(self) -> None:
                self.con = duckdb.frame.connect()
                self.con.create_function("f", self.transform, ["BIGINT"], "BIGINT")

            def transform(self, x: int) -> int:
                return x + 1

        ref = weakref.ref(Service())
        gc.collect()
        assert ref() is None

    def test_duplicates_and_live_results_do_not_hide_the_cycle(self) -> None:
        # More handles share the database than the registry holds references, so a miscount shows up here.
        class Service:
            def __init__(self) -> None:
                self.con = duckdb.frame.connect()
                self.con.create_function("f", self.transform, ["BIGINT"], "BIGINT")
                self.twins = [self.con.duplicate() for _ in range(3)]
                self.result = self.con._execute("SELECT f(1)")

            def transform(self, x: int) -> int:
                return x + 1

        ref = weakref.ref(Service())
        gc.collect()
        assert ref() is None

    def test_a_closure_over_a_global_is_collected(self) -> None:
        namespace: dict[str, object] = {}
        exec(
            "import duckdb\n"
            "con = duckdb.frame.connect()\n"
            "con.create_function('f', lambda x: x + con.run('SELECT 0'), ['BIGINT'], 'BIGINT')\n",
            namespace,
        )
        ref = weakref.ref(namespace["con"])
        del namespace
        gc.collect()
        assert ref() is None

    def test_the_function_survives_a_collection(self, con: duckdb.frame.Connection) -> None:
        con.create_function("f", lambda x: x + 1, ["BIGINT"], "BIGINT")
        con.create_function("f", lambda x: x + 2, ["BIGINT"], "BIGINT")
        gc.collect()
        assert rows(con, "SELECT f(1)") == [(3,)]

    def test_a_module_global_connection_does_not_leak_at_exit(self) -> None:
        script = (
            "import duckdb\n"
            "con = duckdb.frame.connect()\n"
            "con.create_function('f', lambda x: x, ['BIGINT'], 'BIGINT')\n"
            "assert duckdb.frame.sql('SELECT f(1)').rows(con) == [(1,)]\n"
        )
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
        assert "nanobind: leaked" not in result.stderr, result.stderr


#: Registers 320 functions from eight threads on duplicates of `con`, which share its database.
REGISTER_FROM_THREADS = """
import gc, threading, weakref
import duckdb

def register_from_threads(con):
    twins = [con.duplicate() for _ in range(8)]
    barrier = threading.Barrier(8)
    failures = []

    def register(index, twin):
        try:
            barrier.wait()
            for j in range(40):
                offset = index * 40 + j
                twin.create_function(f"f{offset}", lambda x, offset=offset: x + offset, ["BIGINT"], "BIGINT")
        except BaseException as error:
            failures.append(error)

    threads = [threading.Thread(target=register, args=(i, twin)) for i, twin in enumerate(twins)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not failures, failures
"""


def run_isolated(script: str) -> None:
    """Run `script` in a child process, so a registration that deadlocks is killed rather than hanging the suite."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", REGISTER_FROM_THREADS + script],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("a concurrent registration hung")
    assert result.returncode == 0, result.stderr


class TestConcurrentRegistration:
    """Duplicate connections share one database, and so the store that keeps every registered callable alive."""

    def test_every_function_registered_concurrently_is_callable(self) -> None:
        run_isolated(
            "con = duckdb.frame.connect()\n"
            "register_from_threads(con)\n"
            "gc.collect()\n"
            "calls = ' + '.join(f'f{i}(0)' for i in range(320))\n"
            "assert duckdb.frame.sql(f'SELECT {calls}').rows(con) == [(sum(range(320)),)]\n"
        )

    def test_callables_registered_concurrently_are_collected_with_their_database(self) -> None:
        run_isolated(
            "class Service:\n"
            "    def __init__(self):\n"
            "        self.con = duckdb.frame.connect()\n"
            "        self.con.create_function('own', self.transform, ['BIGINT'], 'BIGINT')\n"
            "    def transform(self, x):\n"
            "        return x\n"
            "service = Service()\n"
            "register_from_threads(service.con)\n"
            "ref = weakref.ref(service)\n"
            "del service\n"
            "gc.collect()\n"
            "assert ref() is None\n"
        )


class TestClosedHandles:
    """Closing a handle is also what Python's collector does to it, and every method refuses afterwards."""

    def test_a_closed_result_refuses_every_call(self) -> None:
        connection = _duckdb.Database().connect()
        result = connection.execute("SELECT 1")
        result.close()
        result.close()
        calls: list[Callable[[], object]] = [
            lambda: result.schema,
            lambda: result.result_type,
            lambda: result.statement_type,
            lambda: result.schema_types,
            result.fetch_all,
            lambda: result.fetch_rows(1),
            result.drain,
            result.fetch_chunk_view,
        ]
        for call in calls:
            with pytest.raises(exceptions.InterfaceError, match="result is closed"):
                call()

    def test_a_closed_connection_refuses_every_call(self) -> None:
        connection = _duckdb.Database().connect()
        connection.close()
        connection.close()
        calls: list[Callable[[], object]] = [
            lambda: connection.execute("SELECT 1"),
            lambda: connection.bind("SELECT 1"),
            lambda: connection.create_scalar_function("f", abs, ["BIGINT"], "BIGINT", DEFAULT, CONSISTENT),
            connection.interrupt,
            lambda: connection.get_option("threads"),
            lambda: connection.set_option("threads", "1"),
        ]
        for call in calls:
            with pytest.raises(exceptions.InterfaceError, match="connection is closed"):
                call()

    def test_a_connection_keeps_its_database_alive(self) -> None:
        # A Database has no close: its connections hold it, and it goes with the last of them.
        assert not hasattr(_duckdb.Database, "close")
        database = _duckdb.Database()
        connection = database.connect()

        def probe(x: int) -> int:
            return x

        # The database owns its registered callables, so one outliving its last reference shows it is alive.
        connection.create_scalar_function("probe", probe, ["BIGINT"], "BIGINT", DEFAULT, CONSISTENT)
        ref = weakref.ref(probe)
        del database, probe
        gc.collect()
        assert ref() is not None, "the connection dropped its database"
        assert connection.execute("SELECT probe(1)").fetch_all() == [(1,)]
        connection.close()
        gc.collect()
        assert ref() is None, "a closed connection still pinned its database"

    def test_closing_from_another_thread_during_execute_is_safe(self) -> None:
        # A constant argument is worked out inside execute(), so the gate holds the worker there, with no timing.
        connection = _duckdb.Database().connect()
        started = threading.Event()
        release = threading.Event()

        def gate(x: int) -> int:
            started.set()
            release.wait(timeout=30)
            return x

        connection.create_scalar_function("gate", gate, ["BIGINT"], "BIGINT", DEFAULT, CONSISTENT)
        outcome: list[object] = []

        def run() -> None:
            try:
                outcome.append(connection.execute("SELECT gate(1)").fetch_all())
            except BaseException as error:
                outcome.append(error)

        worker = threading.Thread(target=run)
        worker.start()
        assert started.wait(timeout=30), "the function never ran"
        assert not outcome, "execute returned before the gate opened"
        connection.close()
        with pytest.raises(exceptions.InterfaceError, match="connection is closed"):
            connection.interrupt()
        release.set()
        worker.join(timeout=30)

        assert not worker.is_alive(), "the call outlived the close"
        (result,) = outcome
        assert isinstance(result, list | exceptions.InterruptError | exceptions.InterfaceError), repr(result)
        with pytest.raises(exceptions.InterfaceError, match="connection is closed"):
            connection.execute("SELECT 1")

    def test_closing_from_another_thread_during_a_fetch_is_safe(self) -> None:
        connection = _duckdb.Database().connect()
        outcome: list[object] = []

        def run() -> None:
            try:
                # Long enough to outlast the close, short enough to end on its own once nothing can interrupt it.
                outcome.append(connection.execute("SELECT count(*) FROM range(20_000_000_000)").fetch_all())
            except BaseException as error:
                outcome.append(error)

        worker = threading.Thread(target=run)
        worker.start()
        # Give the query time to actually start.
        time.sleep(0.2)
        connection.close()
        with pytest.raises(exceptions.InterfaceError, match="connection is closed"):
            connection.interrupt()
        worker.join(timeout=60)

        assert not worker.is_alive(), "the query outlived the close"
        (result,) = outcome
        assert isinstance(result, list | exceptions.InterruptError | exceptions.InterfaceError), repr(result)
        with pytest.raises(exceptions.InterfaceError, match="connection is closed"):
            connection.execute("SELECT 1")


class TestTemporalReturns:
    """A return converts to the declared type in its own unit, and never by assuming a time zone."""

    @pytest.mark.parametrize(
        ("returned", "declared"),
        [
            (datetime.datetime(2020, 1, 1, 12), "TIMESTAMPTZ"),
            (datetime.datetime(2020, 1, 1, 12, tzinfo=datetime.UTC), "TIMESTAMP"),
            (datetime.date(2020, 1, 1), "TIMESTAMPTZ"),
            (datetime.time(12, tzinfo=datetime.UTC), "TIME"),
        ],
        ids=["naive for zoned", "aware for naive", "date for zoned", "aware time for naive"],
    )
    def test_a_return_needing_a_time_zone_is_refused(
        self, con: duckdb.frame.Connection, returned: object, declared: str
    ) -> None:
        con.create_function("f", lambda _: returned, ["BIGINT"], declared)
        with pytest.raises(exceptions.InvalidInputError, match="converting between them would assume a time zone"):
            rows(con, "SELECT f(1)")

    @pytest.mark.parametrize(
        ("returned", "declared"),
        [
            (datetime.datetime(2020, 1, 1, 12), "TIMETZ"),
            (datetime.datetime(2020, 1, 1, 12, tzinfo=datetime.UTC), "TIME"),
            (datetime.datetime(2020, 1, 1, 12, tzinfo=datetime.UTC), "TIMETZ"),
            (datetime.date(2020, 1, 1), "TIMETZ"),
            (datetime.time(12, tzinfo=datetime.UTC), "TIMESTAMP"),
        ],
        ids=[
            "naive for a zoned time",
            "aware for a time",
            "aware for a zoned time",
            "date for a zoned time",
            "zoned time",
        ],
    )
    def test_a_time_of_day_is_never_taken_in_a_time_zone(
        self, con: duckdb.frame.Connection, returned: object, declared: str
    ) -> None:
        # A timestamp with a time zone has a time of day only in a zone chosen for it, which the value does not name.
        con.create_function("f", lambda _: returned, ["BIGINT"], declared)
        with pytest.raises(exceptions.InvalidInputError, match="converting between them would assume a time zone"):
            rows(con, "SELECT f(1)")

    def test_a_naive_return_drops_its_date_for_a_time(self, con: duckdb.frame.Connection) -> None:
        con.create_function("f", lambda _: datetime.datetime(2020, 1, 1, 12, 30), ["BIGINT"], "TIME")
        assert rows(con, "SELECT f(1)") == [(datetime.time(12, 30),)]

    def test_an_aware_return_keeps_its_instant(self, con: duckdb.frame.Connection) -> None:
        returned = datetime.datetime(2020, 1, 1, 12, tzinfo=datetime.timezone(datetime.timedelta(hours=2)))
        con.create_function("f", lambda _: returned, ["BIGINT"], "TIMESTAMPTZ")
        assert rows(con, "SELECT f(1)") == [(returned,)]

    def test_text_returned_for_timestamptz_reads_in_the_session_zone(self, con: duckdb.frame.Connection) -> None:
        # As SQL's CAST of the same text does; a reading fixed to UTC would silently disagree with it.
        con._execute("SET TimeZone = 'Asia/Kolkata'").drain()
        assert rows(con, "SELECT current_setting('TimeZone')") == [("Asia/Kolkata",)]
        con.create_function("f", lambda _: "2020-01-01 12:00:00", ["BIGINT"], "TIMESTAMPTZ")
        offset = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
        expected = datetime.datetime(2020, 1, 1, 12, tzinfo=offset)
        assert rows(con, "SELECT f(1) = TIMESTAMPTZ '2020-01-01 12:00:00+05:30', f(1)") == [(True, expected)]

    def test_a_pandas_timestamp_keeps_its_nanoseconds(self, con: duckdb.frame.Connection) -> None:
        stamp = pd.Timestamp("2020-01-02 03:04:05.123456789")
        con.create_function("f", lambda _: stamp, ["BIGINT"], "TIMESTAMP_NS")
        assert rows(con, "SELECT epoch_ns(f(1))") == [(stamp.value,)]

    def test_a_numpy_datetime_converts(self, con: duckdb.frame.Connection) -> None:
        con.create_function("f", lambda _: np.datetime64("2020-01-02T03:04:05", "s"), ["BIGINT"], "TIMESTAMP")
        assert rows(con, "SELECT f(1)") == [(datetime.datetime(2020, 1, 2, 3, 4, 5),)]

    def test_an_argument_before_1970_floors_to_its_microsecond(self, con: duckdb.frame.Connection) -> None:
        seen: list[object] = []

        def record(argument: object) -> int:
            seen.append(argument)
            return 0

        con.create_function("f", record, ["TIMESTAMP_NS"], "BIGINT")
        rows(con, "SELECT f(TIMESTAMP_NS '1969-12-31 23:59:59.999999999')")
        assert seen == [datetime.datetime(1969, 12, 31, 23, 59, 59, 999999)]


class TestTemporalReturnsExactly:
    """A returned date or timestamp is held exactly or refused, nested ones included, never rounded or rezoned."""

    @pytest.mark.parametrize(
        ("returned", "declared"),
        [
            ([datetime.datetime(2020, 1, 1, 12)], "TIMESTAMPTZ[]"),
            ({"a": datetime.datetime(2020, 1, 1, 12, tzinfo=datetime.UTC)}, "STRUCT(a TIMESTAMP)"),
        ],
        ids=["list", "struct"],
    )
    def test_a_nested_return_needing_a_time_zone_is_refused(
        self, con: duckdb.frame.Connection, returned: object, declared: str
    ) -> None:
        con.create_function("f", lambda _: returned, ["BIGINT"], declared)
        with pytest.raises(exceptions.InvalidInputError, match="converting between them would assume a time zone"):
            rows(con, "SELECT f(1)")

    @pytest.mark.parametrize(
        ("returned", "declared"),
        [
            (pd.Timestamp("2020-01-01 00:00:00.123456789"), "TIMESTAMP"),
            (pd.Timestamp("2020-01-01 00:00:00.123456789", tz="UTC"), "TIMESTAMPTZ"),
            (np.datetime64("2020-01-01T00:00:00.5", "ms"), "TIMESTAMP_S"),
            (datetime.datetime(2020, 1, 1, 12), "DATE"),
        ],
        ids=["ns into us", "zoned ns into us", "ms into s", "datetime into date"],
    )
    def test_an_instant_its_type_cannot_hold_is_refused(
        self, con: duckdb.frame.Connection, returned: object, declared: str
    ) -> None:
        con.create_function("f", lambda _: returned, ["BIGINT"], declared)
        with pytest.raises(exceptions.InvalidInputError, match="which cannot hold it exactly"):
            rows(con, "SELECT f(1)")

    @pytest.mark.parametrize(
        ("returned", "declared", "expected"),
        [
            (datetime.datetime(2020, 1, 1), "DATE", datetime.date(2020, 1, 1)),
            (pd.Timestamp("2020-01-01 00:00:00.123456"), "TIMESTAMP", datetime.datetime(2020, 1, 1, 0, 0, 0, 123456)),
            (np.datetime64("2020-01-01T00:00:01", "s"), "TIMESTAMP_NS", datetime.datetime(2020, 1, 1, 0, 0, 1)),
            (
                datetime.datetime(2020, 1, 1, 12, tzinfo=datetime.UTC),
                "TIMESTAMPTZ_NS",
                datetime.datetime(2020, 1, 1, 12, tzinfo=datetime.UTC),
            ),
        ],
        ids=["midnight into date", "whole us", "s into ns", "zoned us into ns"],
    )
    def test_an_instant_its_type_holds_is_kept(
        self, con: duckdb.frame.Connection, returned: object, declared: str, expected: object
    ) -> None:
        con.create_function("f", lambda _: returned, ["BIGINT"], declared)
        assert rows(con, "SELECT f(1)") == [(expected,)]


class TestNestedTemporalReturns:
    """Each date or timestamp anywhere in a return is held exactly or refused; the rest is cast as CAST would."""

    @pytest.mark.parametrize(
        ("returned", "declared", "expected"),
        [
            (
                pd.Timestamp(1000, unit="ns", tz="UTC"),
                "TIMESTAMPTZ",
                datetime.datetime(1970, 1, 1, 0, 0, 0, 1, datetime.UTC),
            ),
            (pd.Timestamp(-(10**9), unit="ns"), "TIMESTAMP_S", datetime.datetime(1969, 12, 31, 23, 59, 59)),
            ([datetime.datetime(2020, 1, 1)], "DATE[]", [datetime.date(2020, 1, 1)]),
            ([pd.Timestamp("2020-01-01 00:00:00.000001")], "TIMESTAMP[]", [datetime.datetime(2020, 1, 1, 0, 0, 0, 1)]),
            ({"a": [datetime.datetime(2020, 1, 1)]}, "STRUCT(a DATE[])", {"a": [datetime.date(2020, 1, 1)]}),
            ({1: datetime.datetime(2020, 1, 1)}, "MAP(INTEGER, DATE)", {1: datetime.date(2020, 1, 1)}),
            ([1.7, None], "INTEGER[]", [2, None]),
            (
                [
                    pd.Timestamp("2020-01-01", tz="UTC").as_unit("ns"),
                    datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC),
                ],
                "TIMESTAMPTZ[]",
                [datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC)] * 2,
            ),
            ([datetime.datetime(2020, 1, 1), None], "TIMESTAMP[]", [datetime.datetime(2020, 1, 1), None]),
            ([None, None], "DATE[]", [None, None]),
            ({"a": None}, "STRUCT(a TIMESTAMP)", {"a": None}),
            (
                {1: None, 2: datetime.datetime(2020, 1, 1)},
                "MAP(INTEGER, TIMESTAMP)",
                {1: None, 2: datetime.datetime(2020, 1, 1)},
            ),
            ([None, [datetime.datetime(2020, 1, 1)]], "TIMESTAMP[][]", [None, [datetime.datetime(2020, 1, 1)]]),
        ],
        ids=[
            "zoned ns into us",
            "ns into s",
            "list",
            "list of ns",
            "struct",
            "map",
            "numbers cast",
            "mixed units",
            "a null element",
            "only null elements",
            "a null field",
            "a null map value",
            "a null inner list",
        ],
    )
    def test_what_its_type_holds_is_kept(
        self, con: duckdb.frame.Connection, returned: object, declared: str, expected: object
    ) -> None:
        con.create_function("f", lambda _: returned, ["BIGINT"], declared)
        assert rows(con, "SELECT f(1)") == [(expected,)]

    @pytest.mark.parametrize(
        ("returned", "declared"),
        [
            ([pd.Timestamp("2020-01-01 00:00:00.000000001")], "TIMESTAMP[]"),
            ([datetime.datetime(2020, 1, 1, 13)], "DATE[]"),
            ({"a": datetime.datetime(2020, 1, 1, 13)}, "STRUCT(a DATE)"),
            ({1: datetime.datetime(2020, 1, 1, 13)}, "MAP(INTEGER, DATE)"),
            ([datetime.datetime(1500, 1, 1)], "TIMESTAMP_NS[]"),
        ],
        ids=["list of ns", "list", "struct", "map", "past nanoseconds"],
    )
    def test_what_its_type_cannot_hold_is_refused(
        self, con: duckdb.frame.Connection, returned: object, declared: str
    ) -> None:
        con.create_function("f", lambda _: returned, ["BIGINT"], declared)
        with pytest.raises(exceptions.InvalidInputError, match="which cannot hold it exactly"):
            rows(con, "SELECT f(1)")

    @pytest.mark.parametrize(
        ("returned", "declared"),
        [
            ([datetime.datetime(2020, 1, 1)], "TIMESTAMPTZ[1]"),
            ({"a": [datetime.date(2020, 1, 1)]}, "STRUCT(a TIMESTAMPTZ[1])"),
        ],
        ids=["array", "array in a struct"],
    )
    def test_a_list_for_an_array_needs_no_time_zone_either(
        self, con: duckdb.frame.Connection, returned: object, declared: str
    ) -> None:
        con.create_function("f", lambda _: returned, ["BIGINT"], declared)
        with pytest.raises(exceptions.InvalidInputError, match="converting between them would assume a time zone"):
            rows(con, "SELECT f(1)")

    @pytest.mark.parametrize(
        ("shape", "declared"),
        [
            ("dict", "STRUCT(a TIMESTAMPTZ, b TIMESTAMPTZ)"),
            ("dict", "MAP(VARCHAR, TIMESTAMPTZ)"),
            ("dict", "UNION(s STRUCT(a TIMESTAMPTZ, b TIMESTAMPTZ))"),
            ("list", "TIMESTAMPTZ[]"),
            ("list", "UNION(a TIMESTAMPTZ[], v VARCHAR)"),
        ],
        ids=["struct", "map", "struct in a union", "list", "list in a union"],
    )
    def test_a_return_its_own_tzinfo_empties_converts_as_returned(
        self, con: duckdb.frame.Connection, shape: str, declared: str
    ) -> None:
        # Only the list or dict holds its values, so emptying it while one converts would cut the rest or free them.
        stamps = [datetime.datetime(2020 + i, 1, 1, tzinfo=datetime.UTC) for i in range(2)]
        returned: object
        expected: object
        if shape == "list":
            elements: list[datetime.datetime] = []
            elements.extend(stamp.replace(tzinfo=Emptying(elements)) for stamp in stamps)
            returned, expected = elements, stamps
        else:
            fields: dict[str, datetime.datetime] = {}
            fields.update(
                {key: stamp.replace(tzinfo=Emptying(fields)) for key, stamp in zip("ab", stamps, strict=True)}
            )
            returned, expected = fields, dict(zip("ab", stamps, strict=True))
        con.create_function("f", lambda _: returned, ["BIGINT"], declared)
        assert rows(con, "SELECT f(1)") == [(expected,)]

    @pytest.mark.parametrize(
        ("declared", "wrap"),
        [
            ("TIMESTAMPTZ[]", lambda stamp: [stamp]),
            ("STRUCT(s TIMESTAMPTZ)", lambda stamp: {"s": stamp}),
            ("MAP(VARCHAR, TIMESTAMPTZ)", lambda stamp: {"s": stamp}),
            ("STRUCT(s TIMESTAMPTZ)[]", lambda stamp: [{"s": stamp}]),
            ("UNION(t STRUCT(s TIMESTAMPTZ))", lambda stamp: {"s": stamp}),
            ("UNION(t TIMESTAMPTZ[])", lambda stamp: [stamp]),
            ("UNION(a TIMESTAMPTZ[], b DATE[])", lambda stamp: [stamp]),
            ("UNION(a TIMESTAMPTZ[], i INT)", lambda stamp: [stamp]),
            ("UNION(u UNION(t TIMESTAMPTZ[]))", lambda stamp: [stamp]),
        ],
        ids=[
            "list",
            "struct",
            "map",
            "list of structs",
            "struct in a union",
            "list in a union",
            "a union of two list members",
            "a union with another kind",
            "a nested union",
        ],
    )
    def test_each_date_is_read_once(
        self, con: duckdb.frame.Connection, declared: str, wrap: Callable[[datetime.datetime], object]
    ) -> None:
        calls: list[object] = []

        class Counting(datetime.tzinfo):
            def utcoffset(self, moment: datetime.datetime | None) -> datetime.timedelta:
                calls.append(moment)
                return datetime.timedelta(0)

            def dst(self, moment: datetime.datetime | None) -> None:
                return None

            def tzname(self, moment: datetime.datetime | None) -> str:
                return "counting"

        stamp = datetime.datetime(2020, 1, 1, tzinfo=Counting())
        con.create_function("g", lambda _: stamp, ["BIGINT"], "TIMESTAMPTZ")
        rows(con, "SELECT g(1)")
        alone = len(calls)
        calls.clear()
        con.create_function("f", lambda _: wrap(stamp), ["BIGINT"], declared)
        rows(con, "SELECT f(1)")
        assert len(calls) == alone > 0

    @pytest.mark.parametrize("declared", ["MAP(VARCHAR, TIMESTAMP)", "STRUCT(a TIMESTAMP)"])
    def test_a_dict_subclass_is_read_by_its_entries(self, con: duckdb.frame.Connection, declared: str) -> None:
        class Uncounted(dict[str, datetime.datetime]):
            def __len__(self) -> int:
                return 0

        returned = Uncounted(a=datetime.datetime(2020, 1, 1))
        con.create_function("f", lambda _: returned, ["BIGINT"], declared)
        assert rows(con, "SELECT f(1)") == [({"a": datetime.datetime(2020, 1, 1)},)]


class TestReturnedFieldsPairByName:
    """A returned dict meets its declared STRUCT or MAP as the engine's cast pairs them: by field name."""

    @pytest.mark.parametrize(
        ("returned", "declared", "expected"),
        [
            (
                {"b": datetime.datetime(2020, 1, 1), "a": datetime.datetime(2021, 1, 1)},
                "STRUCT(a TIMESTAMP, b TIMESTAMP)",
                {"a": datetime.datetime(2021, 1, 1), "b": datetime.datetime(2020, 1, 1)},
            ),
            (
                {"b": datetime.datetime(2020, 1, 1), "a": 2},
                "STRUCT(a INTEGER, b TIMESTAMP)",
                {"a": 2, "b": datetime.datetime(2020, 1, 1)},
            ),
            ({"k": datetime.datetime(2020, 1, 1)}, "MAP(VARCHAR, DATE)", {"k": datetime.date(2020, 1, 1)}),
        ],
        ids=["reordered", "mixed types reordered", "text keys into a map"],
    )
    def test_fields_meet_their_namesakes(
        self, con: duckdb.frame.Connection, returned: object, declared: str, expected: object
    ) -> None:
        con.create_function("f", lambda _: returned, ["BIGINT"], declared)
        assert rows(con, "SELECT f(1)") == [(expected,)]

    @pytest.mark.parametrize(
        ("returned", "declared", "message"),
        [
            ({"k": np.datetime64(1, "ns")}, "MAP(VARCHAR, TIMESTAMP)", "which cannot hold it exactly"),
            ({"k": datetime.datetime(2020, 1, 1, 5)}, "MAP(VARCHAR, DATE)", "which cannot hold it exactly"),
            (
                {"k": datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC)},
                "MAP(VARCHAR, TIMESTAMP)",
                "converting between them would assume a time zone",
            ),
            (
                {"k": datetime.datetime(2020, 1, 1)},
                "MAP(VARCHAR, TIMESTAMPTZ)",
                "converting between them would assume a time zone",
            ),
            (
                {"b": datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC), "a": datetime.datetime(2020, 1, 1)},
                "STRUCT(a TIMESTAMPTZ, b TIMESTAMP)",
                "converting between them would assume a time zone",
            ),
        ],
        ids=["ns into a map", "time into a map of dates", "aware into a map", "naive into a map", "reordered zones"],
    )
    def test_what_a_namesake_cannot_hold_is_refused(
        self, con: duckdb.frame.Connection, returned: object, declared: str, message: str
    ) -> None:
        con.create_function("f", lambda _: returned, ["BIGINT"], declared)
        with pytest.raises(exceptions.InvalidInputError, match=message):
            rows(con, "SELECT f(1)")


class TestUnnamedAndCaseFoldedFields:
    """A field named by the empty string pairs by position, as the engine's cast pairs it; other names fold case."""

    @pytest.mark.parametrize(
        ("returned", "declared", "message"),
        [
            ({"": datetime.datetime(2020, 1, 1, 5)}, "STRUCT(a DATE)", "which cannot hold it exactly"),
            ({"": np.datetime64(1, "ns")}, "STRUCT(a TIMESTAMP)", "which cannot hold it exactly"),
            (
                {"": datetime.datetime(2020, 1, 1, 5, tzinfo=datetime.UTC), "b": 1},
                "STRUCT(a TIMESTAMP, b INTEGER)",
                "converting between them would assume a time zone",
            ),
            ({"": {"x": datetime.datetime(2020, 1, 1, 5)}}, "STRUCT(a STRUCT(x DATE))", "which cannot hold it exactly"),
            ({"A": datetime.datetime(2020, 1, 1, 5)}, "STRUCT(a DATE)", "which cannot hold it exactly"),
        ],
        ids=["unnamed", "unnamed ns", "unnamed zoned beside a named", "unnamed nested", "case folded"],
    )
    def test_a_field_is_checked_against_the_field_it_meets(
        self, con: duckdb.frame.Connection, returned: object, declared: str, message: str
    ) -> None:
        con.create_function("f", lambda _: returned, ["BIGINT"], declared)
        with pytest.raises(exceptions.InvalidInputError, match=message):
            rows(con, "SELECT f(1)")

    def test_a_later_case_twin_is_dropped_as_the_engine_drops_it(self, con: duckdb.frame.Connection) -> None:
        # The first field naming `a` takes it, and the cast drops the second unread.
        returned = {"a": datetime.datetime(2020, 1, 1), "A": datetime.datetime(2021, 1, 1, 5)}
        con.create_function("f", lambda _: returned, ["BIGINT"], "STRUCT(a DATE)")
        assert rows(con, "SELECT f(1)") == [({"a": datetime.date(2020, 1, 1)},)]

    def test_unnamed_fields_are_kept_in_order(self, con: duckdb.frame.Connection) -> None:
        returned = {"": datetime.datetime(2020, 1, 1), "y": datetime.date(2021, 1, 1)}
        con.create_function("f", lambda _: returned, ["BIGINT"], "STRUCT(a DATE, b DATE)")
        assert rows(con, "SELECT f(1)") == [({"a": datetime.date(2020, 1, 1), "b": datetime.date(2021, 1, 1)},)]

    def test_a_later_unnamed_field_is_dropped_as_the_engine_drops_it(self, con: duckdb.frame.Connection) -> None:
        # Pairing is by name once the first field has one, and no declared field is named by the empty string.
        returned = {"a": datetime.datetime(2020, 1, 1), "": datetime.datetime(2020, 1, 1, 12)}
        con.create_function("f", lambda _: returned, ["BIGINT"], "STRUCT(a DATE, b DATE)")
        assert rows(con, "SELECT f(1)") == [({"a": datetime.date(2020, 1, 1), "b": None},)]

    def test_matching_fields_are_kept_with_the_declared_names(self, con: duckdb.frame.Connection) -> None:
        returned = {"A": datetime.datetime(2020, 1, 1), "b": None}
        con.create_function("f", lambda _: returned, ["BIGINT"], "STRUCT(a DATE, b TIMESTAMP, c INTEGER)")
        assert rows(con, "SELECT f(1)") == [({"a": datetime.date(2020, 1, 1), "b": None, "c": None},)]


class NeverIterated(list[object]):
    """A list that fails when iterated, standing for one whose own `__iter__` runs code."""

    def __iter__(self) -> Iterator[object]:
        message = "a dropped field was iterated"
        raise AssertionError(message)


class TestDroppedFields:
    """A returned field the declared STRUCT has no place for is never read, as the engine's cast never reads it.

    The cast still checks the struct's shape.
    """

    @pytest.mark.parametrize(
        "ignored",
        [
            [datetime.datetime(2020, 1, 1), datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC)],
            np.datetime64(1, "ns"),
        ],
        ids=["a list of both kinds", "nanoseconds"],
    )
    def test_a_dropped_field_is_not_read(self, con: duckdb.frame.Connection, ignored: object) -> None:
        # Values the conversion would refuse, so a check that read them anyway would fail loudly.
        returned = {"a": datetime.datetime(2020, 1, 1), "ignored": ignored}
        con.create_function("f", lambda _: returned, ["BIGINT"], "STRUCT(a TIMESTAMP)")
        assert rows(con, "SELECT f(1)") == [({"a": datetime.datetime(2020, 1, 1)},)]

    @pytest.mark.parametrize(
        ("returned", "declared", "expected"),
        [
            ({"k": {"b": 1, "ignored": object()}}, "MAP(VARCHAR, STRUCT(b INTEGER))", {"k": {"b": 1}}),
            (
                [{"a": datetime.datetime(2020, 1, 1), "ignored": object()}],
                "STRUCT(a TIMESTAMP)[]",
                [{"a": datetime.datetime(2020, 1, 1)}],
            ),
            (
                {"s": {"a": datetime.datetime(2020, 1, 1), "ignored": {"c": [object()]}}},
                "STRUCT(s STRUCT(a TIMESTAMP))",
                {"s": {"a": datetime.datetime(2020, 1, 1)}},
            ),
        ],
        ids=["in a map", "in a list", "in a nested struct"],
    )
    def test_a_dropped_field_deeper_is_not_read(
        self, con: duckdb.frame.Connection, returned: object, declared: str, expected: object
    ) -> None:
        # The list and nested-struct shapes also pin that the declared description walks LIST and STRUCT parts.
        con.create_function("f", lambda _: returned, ["BIGINT"], declared)
        assert rows(con, "SELECT f(1)") == [(expected,)]

    @pytest.mark.parametrize(
        ("returned", "declared", "expected"),
        [
            (
                {"a": datetime.datetime(2020, 1, 1), "ignored": NeverIterated([1])},
                "STRUCT(a TIMESTAMP)",
                {"a": datetime.datetime(2020, 1, 1)},
            ),
            (
                {"s": {"a": datetime.datetime(2020, 1, 1), "ignored": NeverIterated([1])}},
                "STRUCT(s STRUCT(a TIMESTAMP))",
                {"s": {"a": datetime.datetime(2020, 1, 1)}},
            ),
            ({"b": 1, "ignored": NeverIterated([1])}, "STRUCT(b INTEGER)", {"b": 1}),
            (
                {"a": datetime.datetime(2020, 1, 1), "ignored": (NeverIterated([1]),)},
                "STRUCT(a TIMESTAMP)",
                {"a": datetime.datetime(2020, 1, 1)},
            ),
            (
                {"a": datetime.datetime(2020, 1, 1), "ignored": [[NeverIterated([1])]]},
                "STRUCT(a TIMESTAMP)",
                {"a": datetime.datetime(2020, 1, 1)},
            ),
            (
                [{"a": datetime.datetime(2020, 1, 1), "ignored": NeverIterated([1])}],
                "STRUCT(a TIMESTAMP)[]",
                [{"a": datetime.datetime(2020, 1, 1)}],
            ),
        ],
        ids=["beside a date", "in a nested struct", "with no date", "in a tuple", "lists deep", "in a list of structs"],
    )
    def test_a_dropped_list_is_never_iterated(
        self, con: duckdb.frame.Connection, returned: object, declared: str, expected: object
    ) -> None:
        con.create_function("f", lambda _: returned, ["BIGINT"], declared)
        assert rows(con, "SELECT f(1)") == [(expected,)]

    @pytest.mark.parametrize(
        ("returned", "reason"),
        [
            ({"": datetime.datetime(2020, 1, 1), "x": object()}, "Cannot cast STRUCTs of different size"),
            ({"x": object()}, "STRUCT to STRUCT cast must have at least one matching member"),
        ],
        ids=["unnamed with a field too many", "no field matches"],
    )
    def test_the_cast_still_checks_the_shape(self, con: duckdb.frame.Connection, returned: object, reason: str) -> None:
        con.create_function("f", lambda _: returned, ["BIGINT"], "STRUCT(a TIMESTAMP)")
        with pytest.raises(exceptions.InvalidInputError, match=reason):
            rows(con, "SELECT f(1)")

    def test_each_struct_of_a_list_meets_the_declared_one_on_its_own(self, con: duckdb.frame.Connection) -> None:
        # The elements need not combine to one type first, as they would read whole.
        con.create_function("f", lambda _: [{"b": 1}, {"b": "x"}], ["BIGINT"], "STRUCT(b VARCHAR)[]")
        assert rows(con, "SELECT f(1)") == [([{"b": "1"}, {"b": "x"}],)]


class TestUnionReturns:
    """A returned value is cast to the UNION as the engine's cast does, which chooses the member.

    A date or time inside is then held, in the member chosen, to the rules any declared type holds it to: no time zone
    assumed, and a date or timestamp exactly or not at all. A value that fails them is refused, and no other member is
    tried.
    """

    @pytest.mark.parametrize(
        ("returned", "declared"),
        [
            (datetime.datetime(2020, 1, 1, 12), "UNION(t TIMESTAMP, z TIMESTAMPTZ)"),
            (datetime.datetime(2020, 1, 1, 12, tzinfo=datetime.UTC), "UNION(t TIMESTAMP, z TIMESTAMPTZ)"),
            (np.datetime64("2020-01-01T00:00:00.5", "ms"), "UNION(t TIMESTAMP, i INTEGER)"),
            (np.datetime64("2020-01-01T00:00:00.5", "ms"), "UNION(a TIMESTAMP, b TIMESTAMP_S)"),
            (datetime.datetime(2020, 1, 1, 0, 0, 0, 1), "UNION(m TIMESTAMP_MS, n TIMESTAMP_NS)"),
            ([np.datetime64(1, "s")], "UNION(a TIMESTAMP[], b TIMESTAMP_MS[])"),
            (
                {"a": datetime.datetime(2020, 1, 1), "b": 1},
                "UNION(z STRUCT(a TIMESTAMPTZ, b HUGEINT), t STRUCT(a TIMESTAMP, b HUGEINT))",
            ),
            ([datetime.datetime(2020, 1, 1)] * 3, "UNION(a TIMESTAMP[3], s VARCHAR)"),
            (
                [
                    pd.Timestamp("2020-01-01", tz="UTC").as_unit("ns"),
                    datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC),
                ],
                "UNION(t TIMESTAMPTZ[])",
            ),
            ({"a": datetime.datetime(2020, 1, 1), "b": 1}, "UNION(t STRUCT(a TIMESTAMP, b HUGEINT))"),
            ([np.datetime64(1, "s")], "UNION(t TIMESTAMP[])"),
            ([datetime.datetime(2020, 1, 1)], "UNION(a TIMESTAMP[], b DATE[], s VARCHAR)"),
            ({"a": datetime.datetime(2020, 1, 1)}, "UNION(w STRUCT(a TIMESTAMP, b INTEGER), s STRUCT(a TIMESTAMP))"),
            ({1: datetime.datetime(2020, 1, 1)}, "UNION(m MAP(BIGINT, TIMESTAMP), s VARCHAR)"),
            (datetime.time(12), "UNION(t TIME, z TIMETZ)"),
            (datetime.time(12, tzinfo=datetime.UTC), "UNION(t TIME, z TIMETZ)"),
            (datetime.datetime(2020, 1, 1), "UNION(u UNION(t TIMESTAMP))"),
            ([datetime.datetime(2020, 1, 1)], "UNION(u UNION(t TIMESTAMP[]))"),
            (datetime.datetime(2020, 1, 1), "UNION(z TIMESTAMPTZ, u UNION(s VARCHAR, t TIMESTAMP))"),
            (datetime.datetime(2020, 1, 1), "UNION(u UNION(u TIMESTAMP, s VARCHAR))"),
        ],
        ids=[
            "naive",
            "aware",
            "the one member of its kind",
            "the unit it widens to exactly",
            "a finer unit",
            "a list in the unit it widens to",
            "a struct beside one needing a time zone",
            "an array of its length",
            "a zoned list of two units at whole microseconds",
            "a field widened",
            "a list in another unit",
            "its own type among several",
            "a dict of its own type among several",
            "a dict with other keys",
            "a naive time",
            "an aware time",
            "nested, a date",
            "nested, a list",
            "nested, beside a member of the other kind",
            "nested, a tag named as its member",
        ],
    )
    def test_the_member_matches_the_casts_own_choice(
        self, con: duckdb.frame.Connection, returned: object, declared: str
    ) -> None:
        # One oracle for every shape: the value lands in the member SQL's CAST of the same literal picks, and holds
        # the same value there, so an engine bump that moves the choice moves both sides together.
        con.create_function("f", lambda _: returned, ["BIGINT"], declared)
        literal = f"CAST({render_literal(returned)} AS {declared})"
        expected = rows(con, f"SELECT union_tag({literal}), {literal}")
        assert rows(con, "SELECT union_tag(f(1)), f(1)") == expected

    def test_an_unnamed_dict_takes_a_struct_member_by_position(self, con: duckdb.frame.Connection) -> None:
        returned = {"": datetime.datetime(2020, 1, 1), "x": 1}
        con.create_function("f", lambda _: returned, ["BIGINT"], "UNION(s STRUCT(a TIMESTAMP, b BIGINT), v VARCHAR)")
        assert rows(con, "SELECT union_tag(f(1)), f(1)") == [("s", {"a": datetime.datetime(2020, 1, 1), "b": 1})]

    @pytest.mark.parametrize(
        ("returned", "declared"),
        [
            (np.datetime64(1, "ns"), "UNION(t TIMESTAMP)"),
            ({"u": np.datetime64(1, "ns")}, "STRUCT(u UNION(t TIMESTAMP))"),
            ([np.datetime64(1, "ns")], "UNION(t TIMESTAMP)[]"),
            (np.datetime64(1, "ns"), "UNION(u UNION(t TIMESTAMP))"),
        ],
        ids=["nanoseconds", "in a struct", "in a list", "in a nested union"],
    )
    def test_a_member_that_cannot_hold_it_exactly_is_refused(
        self, con: duckdb.frame.Connection, returned: object, declared: str
    ) -> None:
        con.create_function("f", lambda _: returned, ["BIGINT"], declared)
        with pytest.raises(exceptions.InvalidInputError, match="which cannot hold it exactly"):
            rows(con, "SELECT f(1)")

    @pytest.mark.parametrize(
        ("returned", "declared"),
        [
            (datetime.datetime(2020, 1, 1, 12), "UNION(z TIMESTAMPTZ, s VARCHAR)"),
            ({"x": datetime.datetime(2020, 1, 1, 12)}, "UNION(s STRUCT(x TIMESTAMPTZ), v VARCHAR)"),
            (datetime.datetime(2020, 1, 1, 12), "UNION(m TIMESTAMP_MS, z TIMESTAMPTZ)"),
            ([datetime.datetime(2020, 1, 1, 12)], "UNION(t TIMESTAMP_MS[], z TIMESTAMPTZ[])"),
            (datetime.datetime(2020, 1, 1, 12), "UNION(u UNION(z TIMESTAMPTZ), s VARCHAR)"),
            (datetime.date(2020, 1, 1), "UNION(z TIMESTAMPTZ, s VARCHAR)"),
        ],
        ids=[
            "naive",
            "naive in a struct",
            "naive beside a member it cannot fill exactly",
            "a naive list",
            "naive in a nested union",
            "a date",
        ],
    )
    def test_a_member_needing_a_time_zone_is_refused(
        self, con: duckdb.frame.Connection, returned: object, declared: str
    ) -> None:
        # The engine's cast takes such a value into the zoned member, in the session's time zone.
        con.create_function("f", lambda _: returned, ["BIGINT"], declared)
        with pytest.raises(exceptions.InvalidInputError, match="converting between them would assume a time zone"):
            rows(con, "SELECT f(1)")

    @pytest.mark.parametrize(
        ("returned", "declared", "message"),
        [
            (
                {"a": datetime.datetime(2020, 1, 1)},
                "UNION(t STRUCT(a TIMESTAMP, b INTEGER), v VARCHAR)",
                "can't be implicitly cast",
            ),
            (
                [datetime.datetime(2020, 1, 1), datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC)],
                "UNION(a TIMESTAMP[], b TIMESTAMPTZ[])",
                "a list holds values with and without a time zone",
            ),
        ],
        ids=["a dict missing a field of a struct member", "both kinds in one list"],
    )
    def test_a_value_the_cast_cannot_take_is_refused(
        self, con: duckdb.frame.Connection, returned: object, declared: str, message: str
    ) -> None:
        con.create_function("f", lambda _: returned, ["BIGINT"], declared)
        with pytest.raises(exceptions.InvalidInputError, match=message):
            rows(con, "SELECT f(1)")

    @pytest.mark.parametrize(
        ("returned", "declared"),
        [
            (None, "UNION(t TIMESTAMP, v VARCHAR)"),
            ({"u": None}, "STRUCT(u UNION(t TIMESTAMP))"),
            ([None], "UNION(t TIMESTAMP)[]"),
        ],
        ids=["alone", "in a struct", "in a list"],
    )
    def test_a_none_is_a_null_of_the_declared_union(
        self, con: duckdb.frame.Connection, returned: object, declared: str
    ) -> None:
        con.create_function("f", lambda _: returned, ["BIGINT"], declared)
        expected = None if returned is None else {"u": None} if isinstance(returned, dict) else [None]
        assert rows(con, "SELECT f(1)") == [(expected,)]

    @pytest.mark.parametrize(
        ("returned", "declared"),
        [
            ([datetime.datetime(2020, 1, 1)] * 3, "UNION(a TIMESTAMP[2])"),
            ([datetime.datetime(2020, 1, 1)] * 2, "UNION(a TIMESTAMP[3], m TIMESTAMP_MS[])"),
            ({"s": [datetime.datetime(2020, 1, 1)] * 3}, "UNION(t STRUCT(s TIMESTAMP[2]))"),
            (["a", "b"], "UNION(a VARCHAR[3])"),
            ({"s": ["a"]}, "UNION(t STRUCT(s VARCHAR[2]))"),
        ],
        ids=[
            "alone",
            "beside a member it cannot fill exactly",
            "in a struct member",
            "text",
            "text in a struct member",
        ],
    )
    def test_a_list_of_another_length_than_an_array_member_is_refused(
        self, con: duckdb.frame.Connection, returned: object, declared: str
    ) -> None:
        # SQL's CAST refuses it; the cast of a value usually leaves the member NULL, which the client refuses, but
        # on some platforms the engine's cast trips its own bounds check first and refuses by itself.
        con.create_function("f", lambda _: returned, ["BIGINT"], declared)
        refusals = "which the engine's cast to it leaves NULL|cannot be converted to its declared type"
        with pytest.raises(exceptions.InvalidInputError, match=refusals):
            rows(con, "SELECT f(1)")

    @pytest.mark.parametrize(
        ("returned", "declared"),
        [
            (datetime.datetime(2020, 1, 1), "UNION(a TIMESTAMP, b TIMESTAMP)"),
            ([datetime.datetime(2020, 1, 1)] * 2, "UNION(t TIMESTAMP[2], u TIMESTAMP[])"),
            ({"a": datetime.datetime(2020, 1, 1)}, "UNION(x STRUCT(a UNION(p TIMESTAMP, q TIMESTAMP)))"),
        ],
        ids=["two alike", "an array and a list", "in a struct in a union"],
    )
    def test_two_members_the_cast_ties_between_are_refused_as_ambiguous(
        self, con: duckdb.frame.Connection, returned: object, declared: str
    ) -> None:
        con.create_function("f", lambda _: returned, ["BIGINT"], declared)
        with pytest.raises(exceptions.InvalidInputError, match="The cast is ambiguous"):
            rows(con, "SELECT f(1)")
