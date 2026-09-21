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

import pytest

import duckdb
from duckdb import _duckdb, exceptions
from duckdb.frame import col, fn

if TYPE_CHECKING:
    from collections.abc import Callable


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
        assert message.startswith("Invalid Input Error: Failed to cast value")
        assert message.count("Invalid Input Error:") == 1

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
        assert str(info.value) == "Invalid Input Error: cannot bind a non-finite Decimal"

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
