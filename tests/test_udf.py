"""Python scalar functions: registration, execution, and failure."""

from __future__ import annotations

import datetime
import decimal
import threading

import pytest

import duckdb
from duckdb import col, exceptions, fn


@pytest.fixture
def con() -> duckdb.Connection:
    return duckdb.connect()


def rows(con: duckdb.Connection, sql: str) -> list[tuple[object, ...]]:
    with con._execute(sql) as result:
        return result.fetch_all()


class TestRegistration:
    def test_a_function_is_callable_from_sql(self, con: duckdb.Connection) -> None:
        con.create_function("plus_one", lambda x: x + 1, ["BIGINT"], "BIGINT")
        assert rows(con, "SELECT plus_one(i) FROM range(5) t(i)") == [(1,), (2,), (3,), (4,), (5,)]

    def test_a_function_is_callable_from_a_plan(self, con: duckdb.Connection) -> None:
        con.create_function("plus_one", lambda x: x + 1, ["BIGINT"], "BIGINT")
        frame = duckdb.sql("SELECT * FROM range(3) t(i)").select(fn("plus_one", col("i")).alias("j"))
        assert frame.on(con).rows() == [(1,), (2,), (3,)]

    def test_multiple_arguments(self, con: duckdb.Connection) -> None:
        con.create_function("weave", lambda a, b, c: f"{a}-{b}-{c}", ["VARCHAR", "BIGINT", "BOOLEAN"], "VARCHAR")
        assert rows(con, "SELECT weave('x', 7, true)") == [("x-7-True",)]

    def test_zero_arguments(self, con: duckdb.Connection) -> None:
        con.create_function("answer", lambda: 42, [], "INTEGER", stability="volatile")
        assert rows(con, "SELECT answer()") == [(42,)]

    def test_a_closure_carries_its_state(self, con: duckdb.Connection) -> None:
        prefix = "state"
        con.create_function("tag", lambda x: f"{prefix}:{x}", ["BIGINT"], "VARCHAR")
        assert rows(con, "SELECT tag(1)") == [("state:1",)]

    def test_registering_the_name_again_replaces_the_function(self, con: duckdb.Connection) -> None:
        con.create_function("f", lambda x: x, ["BIGINT"], "BIGINT")
        con.create_function("f", lambda x: x * 10, ["BIGINT"], "BIGINT")
        assert rows(con, "SELECT f(2)") == [(20,)]

    def test_an_unknown_type_text_is_refused(self, con: duckdb.Connection) -> None:
        with pytest.raises(exceptions.Error, match="NO_SUCH_TYPE"):
            con.create_function("f", lambda x: x, ["NO_SUCH_TYPE"], "BIGINT")

    def test_a_bad_null_handling_is_refused(self, con: duckdb.Connection) -> None:
        with pytest.raises(exceptions.InvalidInputError, match="null_handling"):
            con.create_function("f", lambda x: x, ["BIGINT"], "BIGINT", null_handling="never")

    def test_a_bad_stability_is_refused(self, con: duckdb.Connection) -> None:
        with pytest.raises(exceptions.InvalidInputError, match="stability"):
            con.create_function("f", lambda x: x, ["BIGINT"], "BIGINT", stability="jittery")

    def test_a_closed_connection_refuses(self, con: duckdb.Connection) -> None:
        con.close()
        with pytest.raises(exceptions.InterfaceError):
            con.create_function("f", lambda x: x, ["BIGINT"], "BIGINT")


class TestValues:
    def test_scalar_types_round_trip(self, con: duckdb.Connection) -> None:
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

    def test_nested_arguments_arrive_as_python_values(self, con: duckdb.Connection) -> None:
        con.create_function("total", lambda xs: sum(xs), ["BIGINT[]"], "BIGINT")
        con.create_function("field", lambda s: s["a"], ["STRUCT(a INTEGER)"], "INTEGER")
        assert rows(con, "SELECT total([1, 2, 3]), field({'a': 7})") == [(6, 7)]

    def test_nested_returns_are_converted(self, con: duckdb.Connection) -> None:
        con.create_function("pair", lambda x: [x, x + 1], ["BIGINT"], "BIGINT[]")
        con.create_function("wrap", lambda x: {"v": x}, ["BIGINT"], "STRUCT(v BIGINT)")
        assert rows(con, "SELECT pair(3), wrap(9)") == [([3, 4], {"v": 9})]

    def test_the_return_is_cast_to_the_declared_type(self, con: duckdb.Connection) -> None:
        con.create_function("half", lambda x: x / 2, ["BIGINT"], "INTEGER")
        assert rows(con, "SELECT half(9)") == [(4,)]

    def test_an_uncastable_return_fails_loudly(self, con: duckdb.Connection) -> None:
        con.create_function("wide", lambda x: 2**70, ["BIGINT"], "INTEGER")
        with pytest.raises(exceptions.Error):
            rows(con, "SELECT wide(1)")

    def test_an_unbindable_return_object_fails_loudly(self, con: duckdb.Connection) -> None:
        con.create_function("obj", lambda x: object(), ["BIGINT"], "VARCHAR")
        with pytest.raises(exceptions.Error, match="cannot bind"):
            rows(con, "SELECT obj(1)")


class TestNulls:
    def test_default_null_handling_skips_the_function(self, con: duckdb.Connection) -> None:
        calls: list[object] = []

        def observe(x: object) -> object:
            calls.append(x)
            return x

        con.create_function("observe", observe, ["BIGINT"], "BIGINT")
        assert rows(con, "SELECT observe(x) FROM (VALUES (1), (NULL), (3)) t(x)") == [(1,), (None,), (3,)]
        assert calls == [1, 3]

    def test_special_null_handling_passes_none_through(self, con: duckdb.Connection) -> None:
        con.create_function("backfill", lambda x: -1 if x is None else x, ["BIGINT"], "BIGINT", null_handling="special")
        assert rows(con, "SELECT backfill(x) FROM (VALUES (1), (NULL)) t(x)") == [(1,), (-1,)]

    def test_returning_none_makes_the_result_null(self, con: duckdb.Connection) -> None:
        con.create_function("odd_only", lambda x: x if x % 2 else None, ["BIGINT"], "BIGINT")
        assert rows(con, "SELECT odd_only(i) FROM range(4) t(i)") == [(None,), (1,), (None,), (3,)]


class TestPandasNA:
    def test_a_pandas_na_return_is_null(self, con: duckdb.Connection) -> None:
        pd = pytest.importorskip("pandas")
        con.create_function("na_out", lambda x: pd.NA, ["BIGINT"], "BIGINT")
        assert rows(con, "SELECT na_out(1)") == [(None,)]

    def test_other_unbindable_returns_still_fail(self, con: duckdb.Connection) -> None:
        pytest.importorskip("pandas")
        con.create_function("obj2", lambda x: object(), ["BIGINT"], "VARCHAR")
        with pytest.raises(exceptions.Error, match="cannot bind"):
            rows(con, "SELECT obj2(1)")


class TestExecution:
    def test_every_row_of_a_large_input_is_converted(self, con: duckdb.Connection) -> None:
        con.create_function("twice", lambda x: x * 2, ["BIGINT"], "BIGINT")
        got = rows(con, "SELECT sum(twice(i)), count(*) FROM range(10000) t(i)")
        assert got == [(2 * sum(range(10000)), 10000)]

    def test_a_volatile_function_runs_per_row(self, con: duckdb.Connection) -> None:
        counter = iter(range(1000000))
        con.create_function("tick", lambda: next(counter), [], "BIGINT", stability="volatile")
        got = rows(con, "SELECT tick() FROM range(100) t(i)")
        assert {value for (value,) in got} == set(range(100))

    def test_engine_workers_run_the_function_during_a_fetch(self, con: duckdb.Connection) -> None:
        con.run("SET threads = 4")
        seen: set[int] = set()

        def bump(x: int) -> int:
            seen.add(threading.get_ident())
            return x + 1

        con.create_function("bump", bump, ["BIGINT"], "BIGINT")
        con.run("CREATE TABLE big AS SELECT i FROM range(1000000) t(i)")
        assert rows(con, "SELECT sum(bump(i)) FROM big") == [(sum(range(1000000)) + 1000000,)]
        # More than one thread proves the fetch loop released the GIL: the
        # engine's own workers acquired it to call back into Python while
        # this thread was driving the query. A held GIL would deadlock here,
        # not fail.
        assert len(seen) >= 2

    def test_the_function_runs_inside_other_threads_queries(self, con: duckdb.Connection) -> None:
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
    def test_a_python_error_names_the_function_and_the_cause(self, con: duckdb.Connection) -> None:
        def boom(x: int) -> int:
            message = f"no thanks: {x}"
            raise ValueError(message)

        con.create_function("boom", boom, ["BIGINT"], "BIGINT")
        with pytest.raises(exceptions.InvalidInputError, match="boom") as info:
            rows(con, "SELECT boom(7)")
        message = str(info.value)
        assert "Python exception occurred while executing the UDF 'boom'" in message
        assert message.index("ValueError: no thanks: 7") < message.index("Traceback")

    def test_the_connection_survives_a_failing_function(self, con: duckdb.Connection) -> None:
        con.create_function("bad", lambda x: 1 / 0, ["BIGINT"], "BIGINT")
        with pytest.raises(exceptions.Error):
            rows(con, "SELECT bad(1)")
        assert rows(con, "SELECT 42") == [(42,)]
