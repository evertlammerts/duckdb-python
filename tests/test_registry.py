"""Python objects registered as tables: resolution, scope, streams read once, and what reaches the engine."""

from __future__ import annotations

import datetime
import decimal
import gc
import sys
import threading
import time
import uuid
import weakref
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc
import pytest

import duckdb
from duckdb import _duckdb, dbapi, exceptions
from duckdb._sources import CapsuleSource, Source, StreamSource, TableSource, adapt
from duckdb.frame import col, sql, table

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path


@pytest.fixture
def con() -> duckdb.frame.Connection:
    return duckdb.frame.connect()


@pytest.fixture
def numbers() -> pa.Table:
    return pa.table({"n": list(range(10)), "s": [str(i) for i in range(10)]})


def rows(con: duckdb.frame.Connection, query: str) -> list[tuple[object, ...]]:
    with con._execute(query) as result:
        return result.fetch_all()


def reader_over(t: pa.Table) -> pa.RecordBatchReader:
    return pa.RecordBatchReader.from_batches(t.schema, t.to_batches())


class TestClassification:
    def test_tables_and_batches_are_read_repeatedly(self, numbers: pa.Table) -> None:
        for obj in (numbers, numbers.to_batches()[0]):
            source = adapt(obj)
            assert isinstance(source, TableSource)
            assert source.obj is obj
            assert not source.one_shot

    def test_readers_and_capsules_are_streams(self, numbers: pa.Table) -> None:
        for obj, kind in ((reader_over(numbers), StreamSource), (numbers.__arrow_c_stream__(), CapsuleSource)):
            source = adapt(obj)
            assert isinstance(source, kind)
            assert source.obj is obj
            assert source.one_shot

    def test_a_class_named_like_a_capsule_is_not_one(self, numbers: pa.Table) -> None:
        class PyCapsule:
            def __arrow_c_stream__(self, requested_schema: object = None) -> object:
                return numbers.__arrow_c_stream__()

        assert not adapt(PyCapsule()).one_shot

    def test_only_a_stream_capsule_is_a_stream(self, numbers: pa.Table) -> None:
        assert adapt(numbers.__arrow_c_stream__()).one_shot
        assert _duckdb.capsule_name(numbers.__arrow_c_stream__()) == "arrow_array_stream"
        assert _duckdb.capsule_name(numbers.schema.__arrow_c_schema__()) == "arrow_schema"
        assert _duckdb.capsule_name(numbers) is None

    def test_anything_else_is_refused(self) -> None:
        with pytest.raises(TypeError, match=r"__arrow_c_stream__.*list is none of these"):
            adapt([1, 2, 3])
        with pytest.raises(TypeError, match="object is none of these"):
            adapt(object())

    def test_a_schema_alone_is_not_a_dataset(self) -> None:
        class OnlySchema:
            schema = pa.schema([("a", pa.int64())])

        with pytest.raises(TypeError, match="OnlySchema is none of these"):
            adapt(OnlySchema())


class TestResolution:
    def test_a_registered_table_is_a_plan_source(self, con: duckdb.frame.Connection, numbers: pa.Table) -> None:
        con.register("numbers", numbers)
        plan = table("numbers").filter(col("n") >= 8).select(col("n"), col("s"))
        assert plan.on(con).rows() == [(8, "8"), (9, "9")]

    def test_a_registered_table_is_readable_from_sql(self, con: duckdb.frame.Connection, numbers: pa.Table) -> None:
        con.register("numbers", numbers)
        assert sql("SELECT sum(n) FROM numbers").on(con).rows() == [(45,)]

    def test_the_schema_is_the_arrow_schema(self, con: duckdb.frame.Connection, numbers: pa.Table) -> None:
        con.register("numbers", numbers)
        assert table("numbers").schema(con) == [("n", "BIGINT"), ("s", "VARCHAR")]

    def test_names_resolve_like_sql_identifiers(self, con: duckdb.frame.Connection, numbers: pa.Table) -> None:
        con.register("MyNumbers", numbers)
        assert rows(con, "SELECT count(*) FROM mynumbers") == [(10,)]
        assert rows(con, 'SELECT count(*) FROM "MyNumbers"') == [(10,)]
        assert table("MYNUMBERS").on(con).rows()[0] == (0, "0")

    def test_a_real_table_wins_over_a_registration(self, con: duckdb.frame.Connection, numbers: pa.Table) -> None:
        con.run("CREATE TABLE numbers AS SELECT 42 AS v")
        con.register("numbers", numbers)
        assert rows(con, "SELECT * FROM numbers") == [(42,)]

    def test_a_file_name_still_reaches_the_file_reader(
        self, con: duckdb.frame.Connection, numbers: pa.Table, tmp_path: Path
    ) -> None:
        path = tmp_path / "v.csv"
        path.write_text("v\n7\n")
        con.register(str(path), numbers)
        assert rows(con, f"SELECT * FROM '{path}'") == [(7,)]

    def test_a_qualified_name_is_not_a_registration(self, con: duckdb.frame.Connection, numbers: pa.Table) -> None:
        con.register("numbers", numbers)
        with pytest.raises(exceptions.CatalogError):
            rows(con, "SELECT * FROM main.numbers")

    def test_a_registration_replaces_an_earlier_one(self, con: duckdb.frame.Connection, numbers: pa.Table) -> None:
        con.register("t", numbers)
        con.register("t", pa.table({"other": [True]}))
        assert table("t").schema(con) == [("other", "BOOLEAN")]

    def test_unregister_forgets_the_name(self, con: duckdb.frame.Connection, numbers: pa.Table) -> None:
        con.register("numbers", numbers)
        con.unregister("numbers")
        with pytest.raises(exceptions.CatalogError, match="numbers"):
            rows(con, "SELECT * FROM numbers")

    def test_unregistering_an_unknown_name_is_an_error(self, con: duckdb.frame.Connection) -> None:
        with pytest.raises(exceptions.InvalidInputError, match="nothing is registered as 'ghost'"):
            con.unregister("ghost")

    def test_a_name_must_be_a_string(self, con: duckdb.frame.Connection, numbers: pa.Table) -> None:
        with pytest.raises(TypeError, match="a registered name is a string"):
            con.register(42, numbers)  # type: ignore[arg-type]

    def test_an_object_without_a_stream_is_refused(self, con: duckdb.frame.Connection) -> None:
        with pytest.raises(TypeError, match="__arrow_c_stream__"):
            con.register("t", {"a": [1]})

    def test_a_closed_connection_refuses(self, con: duckdb.frame.Connection, numbers: pa.Table) -> None:
        con.close()
        with pytest.raises(exceptions.InterfaceError):
            con.register("numbers", numbers)


class TestScope:
    def test_every_connection_to_the_database_sees_a_registration(
        self, con: duckdb.frame.Connection, numbers: pa.Table
    ) -> None:
        con.register("numbers", numbers)
        assert rows(con.duplicate(), "SELECT count(*) FROM numbers") == [(10,)]

    def test_a_separately_opened_database_does_not(self, con: duckdb.frame.Connection, numbers: pa.Table) -> None:
        con.register("numbers", numbers)
        with pytest.raises(exceptions.CatalogError):
            rows(duckdb.frame.connect(), "SELECT * FROM numbers")

    def test_the_stub_cache_forgets_on_register_and_unregister(
        self, con: duckdb.frame.Connection, numbers: pa.Table
    ) -> None:
        con.register("numbers", numbers)
        plan = table("numbers").select(col("n") + 1)
        assert plan.schema(con) == [("(n + 1)", "BIGINT")]
        con.register("numbers", pa.table({"n": pa.array([1.5], pa.float64())}))
        assert plan.schema(con) == [("(n + 1)", "DOUBLE")]
        con.unregister("numbers")
        with pytest.raises(exceptions.CatalogError):
            plan.schema(con)

    def test_the_object_is_released_on_unregister(self, con: duckdb.frame.Connection) -> None:
        source = pa.table({"a": [1]})
        con.register("t", source)
        before = sys.getrefcount(source)
        con.unregister("t")
        assert sys.getrefcount(source) == before - 1

    def test_the_object_is_released_when_the_database_closes(self) -> None:
        con = duckdb.frame.connect()
        reader = reader_over(pa.table({"a": [1]}))
        ref = weakref.ref(reader)
        con.register("r", reader)
        del reader
        assert ref() is not None
        con.close()
        gc.collect()
        assert ref() is None


class TestConcurrency:
    def test_unregistering_during_a_scan_does_not_stop_it(self, con: duckdb.frame.Connection) -> None:
        big = pa.Table.from_batches(pa.table({"i": list(range(50_000))}).to_batches(max_chunksize=1_000))
        con.register("big", big)
        with con._execute("SELECT i FROM big") as result:
            first = result.fetch_rows(5)
            con.unregister("big")
            rest = result.fetch_all()
        assert [r[0] for r in first + rest] == list(range(50_000))
        with pytest.raises(exceptions.CatalogError):
            rows(con, "SELECT * FROM big")

    def test_registering_and_querying_from_several_threads(
        self, con: duckdb.frame.Connection, numbers: pa.Table
    ) -> None:
        stop = threading.Event()
        failures: list[BaseException] = []
        con.register("shared", numbers)

        def churn() -> None:
            while not stop.is_set():
                con.register("shared", numbers)

        def query() -> None:
            mine = con.duplicate()
            while not stop.is_set():
                try:
                    assert rows(mine, "SELECT count(*) FROM shared") == [(10,)]
                except BaseException as error:
                    failures.append(error)
                    return

        workers = [threading.Thread(target=churn)] + [threading.Thread(target=query) for _ in range(3)]
        for worker in workers:
            worker.start()
        time.sleep(0.5)
        stop.set()
        for worker in workers:
            worker.join()
        assert not failures

    def test_a_failed_export_does_not_consume_a_stream(self, con: duckdb.frame.Connection, numbers: pa.Table) -> None:
        class Flaky:
            def __init__(self) -> None:
                self.calls = 0
                self.schema = numbers.schema

            def __arrow_c_stream__(self, requested_schema: object = None) -> object:
                self.calls += 1
                if self.calls == 1:
                    message = "not yet"
                    raise RuntimeError(message)
                return numbers.__arrow_c_stream__()

        con.register("flaky", Flaky())
        with pytest.raises(exceptions.InvalidInputError, match="not yet"):
            rows(con, "SELECT count(*) FROM flaky")
        assert rows(con, "SELECT count(*) FROM flaky") == [(10,)]


class TestDbapi:
    def test_register_and_unregister_on_a_connection(self, numbers: pa.Table) -> None:
        con = dbapi.connect()
        con.register("numbers", numbers)
        cur = con.cursor()
        cur.execute("SELECT sum(n) FROM numbers")
        assert cur.fetchall() == [(45,)]
        con.unregister("numbers")
        with pytest.raises(exceptions.CatalogError):
            cur.execute("SELECT * FROM numbers")
        with pytest.raises(exceptions.ProgrammingError, match="nothing is registered as 'numbers'"):
            con.unregister("numbers")

    def test_a_name_must_be_a_string(self, numbers: pa.Table) -> None:
        with pytest.raises(TypeError, match="a registered name is a string"):
            dbapi.connect().register(3, numbers)  # type: ignore[arg-type]

    def test_a_closed_connection_refuses(self, numbers: pa.Table) -> None:
        con = dbapi.connect()
        con.close()
        with pytest.raises(exceptions.InterfaceError):
            con.register("numbers", numbers)

    def test_an_object_without_a_stream_is_refused_at_register(self) -> None:
        con = dbapi.connect()
        with pytest.raises(TypeError, match="dict is none of these"):
            con.register("t", {"a": [1]})
        with pytest.raises(exceptions.CatalogError):
            con.cursor().execute("SELECT * FROM t")


class TestStreams:
    def test_a_reader_is_read_once(self, con: duckdb.frame.Connection, numbers: pa.Table) -> None:
        con.register("r", reader_over(numbers))
        assert rows(con, "SELECT count(*) FROM r") == [(10,)]
        with pytest.raises(exceptions.InvalidInputError, match="registered as 'r' has been read already"):
            rows(con, "SELECT count(*) FROM r")

    def test_a_capsule_is_read_once(self, con: duckdb.frame.Connection, numbers: pa.Table) -> None:
        con.register("s", numbers.__arrow_c_stream__())
        assert rows(con, "SELECT count(*) FROM s") == [(10,)]
        with pytest.raises(exceptions.InvalidInputError, match="registered as 's' has been read already"):
            rows(con, "SELECT count(*) FROM s")

    def test_binding_does_not_consume_a_stream(self, con: duckdb.frame.Connection, numbers: pa.Table) -> None:
        con.register("r", reader_over(numbers))
        assert table("r").schema(con) == [("n", "BIGINT"), ("s", "VARCHAR")]
        assert "PYTHON_OBJECT_SCAN" in table("r").on(con).explain()
        assert rows(con, "SELECT count(*) FROM r") == [(10,)]

    def test_registering_again_makes_it_readable_again(self, con: duckdb.frame.Connection, numbers: pa.Table) -> None:
        con.register("r", reader_over(numbers))
        rows(con, "SELECT count(*) FROM r")
        con.register("r", reader_over(numbers))
        assert rows(con, "SELECT count(*) FROM r") == [(10,)]

    def test_one_frame_used_twice_reads_the_stream_once(self, con: duckdb.frame.Connection, numbers: pa.Table) -> None:
        con.register("r", reader_over(numbers))
        source = table("r")
        plan = source.join(source, on="n", suffix="_r").select(col("n")).sort(col("n"))
        assert plan.on(con).rows() == [(i,) for i in range(10)]

    def test_two_steps_naming_a_stream_share_one_cte(self, con: duckdb.frame.Connection, numbers: pa.Table) -> None:
        con.register("r", reader_over(numbers))
        plan = table("r").join(table("R"), on="n", suffix="_r").select(col("n"))
        rendered = plan.render(con)
        assert rendered.count("python_object_scan") == 0
        assert rendered.count('FROM "r"') + rendered.count('FROM "R"') == 1
        assert plan.on(con).rows() == [(i,) for i in range(10)]

    def test_a_macro_body_shares_one_cte_per_stream(self, con: duckdb.frame.Connection, numbers: pa.Table) -> None:
        con.register("r", reader_over(numbers))
        body = table("r").select(col("n")).union(table("R").select(col("n")))
        con.create_macro("twice", [], body)
        assert rows(con, "SELECT count(*) FROM twice()") == [(20,)]

    def test_two_steps_naming_a_table_stay_separate(self, con: duckdb.frame.Connection, numbers: pa.Table) -> None:
        con.register("t", numbers)
        plan = table("t").join(table("t"), on="n", suffix="_r").select(col("n"))
        assert plan.render(con).count('FROM "t"') == 2
        assert len(plan.on(con).rows()) == 10

    def test_two_references_in_sql_fail_at_the_second_read(
        self, con: duckdb.frame.Connection, numbers: pa.Table
    ) -> None:
        con.register("r", reader_over(numbers))
        with pytest.raises(exceptions.InvalidInputError, match="another reference to it in this one"):
            rows(con, "SELECT count(*) FROM r a, r b")

    def test_a_cte_in_sql_reads_the_stream_once(self, con: duckdb.frame.Connection, numbers: pa.Table) -> None:
        con.register("r", reader_over(numbers))
        assert rows(con, "WITH s AS (SELECT * FROM r) SELECT count(*) FROM s a JOIN s b USING (n)") == [(10,)]

    def test_a_python_error_in_the_stream_fails_the_query(self, con: duckdb.frame.Connection) -> None:
        schema = pa.schema([("a", pa.int64())])

        def batches() -> Iterator[pa.RecordBatch]:
            yield pa.record_batch([pa.array([1, 2])], schema=schema)
            message = "no more rows for you"
            raise ValueError(message)

        con.register("bad", pa.RecordBatchReader.from_batches(schema, batches()))
        with pytest.raises(exceptions.InvalidInputError, match=r"registered as 'bad' failed: .*no more rows for you"):
            rows(con, "SELECT sum(a) FROM bad")

    def test_a_released_stream_is_refused(self, con: duckdb.frame.Connection, numbers: pa.Table) -> None:
        capsule = numbers.__arrow_c_stream__()
        pa.RecordBatchReader._import_from_c_capsule(capsule).read_all()
        con.register("s", capsule)
        with pytest.raises(exceptions.InvalidInputError, match="registered as 's' is released already"):
            rows(con, "SELECT * FROM s")


class Recording(Source):
    """A source over a pyarrow table that remembers which columns each scan asked for."""

    def __init__(self, table_: pa.Table, *, narrows: bool = True) -> None:
        super().__init__(table_)
        self.narrows = narrows
        self.asked: list[list[int] | None] = []

    def __arrow_c_schema__(self) -> object:
        return self.obj.schema.__arrow_c_schema__()

    def stream(self, columns: Sequence[int] | None) -> tuple[object, bool]:
        self.asked.append(None if columns is None else list(columns))
        if columns is None or not self.narrows:
            return self.obj.__arrow_c_stream__(), False
        return self.obj.select(list(columns)).__arrow_c_stream__(), True


class TestProjection:
    def test_only_the_columns_a_query_uses_are_asked_for(self, con: duckdb.frame.Connection, numbers: pa.Table) -> None:
        source = Recording(numbers)
        con.register("t", source)
        assert rows(con, "SELECT s FROM t WHERE n = 3") == [("3",)]
        assert rows(con, "SELECT * FROM t LIMIT 1") == [(0, "0")]
        assert rows(con, "SELECT n FROM t WHERE n = 9") == [(9,)]
        assert rows(con, "SELECT s FROM t LIMIT 1") == [("0",)]
        # Every declared column, in order, is asked for as None; a narrower need names the columns.
        assert source.asked == [None, None, [0], [1]]

    def test_the_requested_order_is_the_output_order(self, con: duckdb.frame.Connection, numbers: pa.Table) -> None:
        source = Recording(numbers)
        con.register("t", source)
        assert rows(con, "SELECT s, n FROM t WHERE n = 4") == [("4", 4)]
        assert rows(con, "SELECT s, n FROM t WHERE s = '5'") == [("5", 5)]
        # The engine may ask for the columns in any order; the source answers in that order and the rows are right.
        assert all(asked is None or sorted(asked) == [0, 1] for asked in source.asked)

    def test_a_count_needs_no_columns_but_still_scans(self, con: duckdb.frame.Connection, numbers: pa.Table) -> None:
        source = Recording(numbers)
        con.register("t", source)
        assert rows(con, "SELECT count(*) FROM t") == [(10,)]
        assert len(source.asked) == 1

    def test_a_source_that_cannot_narrow_is_picked_from(self, con: duckdb.frame.Connection, numbers: pa.Table) -> None:
        source = Recording(numbers, narrows=False)
        con.register("t", source)
        assert rows(con, "SELECT s FROM t WHERE n = 3") == [("3",)]
        assert rows(con, "SELECT s FROM t LIMIT 1") == [("0",)]
        assert rows(con, "SELECT n FROM t WHERE n = 9") == [(9,)]
        assert source.asked == [None, [1], [0]]

    def test_a_permuted_request_is_picked_from_a_full_stream(
        self, con: duckdb.frame.Connection, numbers: pa.Table
    ) -> None:
        source = Recording(numbers, narrows=False)
        con.register("t", source)
        assert rows(con, "SELECT s, n FROM t WHERE s = '5'") == [("5", 5)]
        assert rows(con, "SELECT n, s FROM t WHERE n = 6") == [(6, "6")]
        assert all(asked is None or sorted(asked) == [0, 1] for asked in source.asked)

    def test_the_plan_shows_the_projection(self, con: duckdb.frame.Connection, numbers: pa.Table) -> None:
        con.register("t", numbers)
        assert "Projections: n" in table("t").select(col("n")).on(con).explain()
        assert "Projections: s" in sql("SELECT s FROM t WHERE s = '1'").on(con).explain()

    def test_a_source_that_raises_while_exporting(self, con: duckdb.frame.Connection, numbers: pa.Table) -> None:
        class Broken(Recording):
            def stream(self, columns: Sequence[int] | None) -> tuple[object, bool]:
                message = "no stream today"
                raise ValueError(message)

        con.register("t", Broken(numbers))
        expected = "exporting a stream from the object registered as 't' failed: ValueError: no stream today"
        with pytest.raises(exceptions.InvalidInputError, match=expected):
            rows(con, "SELECT * FROM t")

    def test_a_source_must_answer_with_a_pair(self, con: duckdb.frame.Connection, numbers: pa.Table) -> None:
        class Odd(Recording):
            def stream(self, columns: Sequence[int] | None) -> object:  # type: ignore[override]
                return object.__getattribute__(self.obj, "__arrow_c_stream__")()

        con.register("t", Odd(numbers))
        with pytest.raises(
            exceptions.InvalidInputError, match=r"did not answer stream\(\) with a \(capsule, projected\)"
        ):
            rows(con, "SELECT * FROM t")

    def test_a_source_answering_the_wrong_width_is_refused(
        self, con: duckdb.frame.Connection, numbers: pa.Table
    ) -> None:
        class Lying(Recording):
            def stream(self, columns: Sequence[int] | None) -> tuple[object, bool]:
                return self.obj.select([0]).__arrow_c_stream__(), False

        con.register("t", Lying(numbers))
        with pytest.raises(exceptions.InvalidInputError, match="answered with 1 columns where it declared 2"):
            rows(con, "SELECT s FROM t")

        class Short(Recording):
            def stream(self, columns: Sequence[int] | None) -> tuple[object, bool]:
                return self.obj.__arrow_c_stream__(), True

        con.register("t", Short(numbers))
        with pytest.raises(exceptions.InvalidInputError, match="answered with 2 columns where 1 were requested"):
            rows(con, "SELECT s FROM t")

    def test_a_source_answering_in_another_order_is_refused(self, con: duckdb.frame.Connection) -> None:
        wide = pa.table({"n": [100], "s": ["zero"], "x": [0]})

        class Sorting(Recording):
            def stream(self, columns: Sequence[int] | None) -> tuple[object, bool]:
                if columns is None:
                    return self.obj.__arrow_c_stream__(), False
                return self.obj.select(sorted(columns)).__arrow_c_stream__(), True

        con.register("t", Sorting(wide))
        with pytest.raises(exceptions.InvalidInputError, match="answered with column 'n' at position 0 where 'x'"):
            rows(con, "SELECT x, s, n FROM t WHERE x = 0")

        class Swapping(Recording):
            def stream(self, columns: Sequence[int] | None) -> tuple[object, bool]:
                return self.obj.select([2, 1, 0]).__arrow_c_stream__(), False

        con.register("t", Swapping(wide))
        with pytest.raises(exceptions.InvalidInputError, match="answered with column 'x' at position 0 where 'n'"):
            rows(con, "SELECT * FROM t")

    def test_duplicate_names_project_by_position(self, con: duckdb.frame.Connection) -> None:
        columns = [pa.array([1]), pa.array([2]), pa.array([3]), pa.array([4])]
        source = Recording(pa.table(columns, names=["a_1", "a", "a", "a_2"]))
        con.register("dup", source)
        assert rows(con, "SELECT a_2 FROM dup") == [(3,)]
        assert source.asked == [[2]]

    def test_a_stream_source_yields_every_column(self, con: duckdb.frame.Connection, numbers: pa.Table) -> None:
        con.register("r", reader_over(numbers))
        assert rows(con, "SELECT s FROM r WHERE n = 2") == [("2",)]


class TestArrays:
    """An object exporting one array is one batch; a plain array or a stream of them is a single column."""

    def test_a_plain_array_is_one_column_named_value(self, con: duckdb.frame.Connection) -> None:
        con.register("a", pa.array([1, 2, None]))
        assert table("a").schema(con) == [("value", "BIGINT")]
        assert rows(con, "SELECT value FROM a") == [(1,), (2,), (None,)]
        assert rows(con, "SELECT sum(value) FROM a") == [(3,)]

    def test_a_struct_array_is_a_table_of_its_fields(self, con: duckdb.frame.Connection) -> None:
        array = pa.StructArray.from_arrays([pa.array([1, 2]), pa.array(["a", "b"])], names=["n", "s"])
        con.register("st", array)
        assert table("st").schema(con) == [("n", "BIGINT"), ("s", "VARCHAR")]
        assert rows(con, "SELECT s FROM st WHERE n = 2") == [("b",)]
        assert rows(con, "SELECT n, s FROM st ORDER BY n") == [(1, "a"), (2, "b")]

    def test_a_chunked_array_is_a_stream_of_plain_arrays(self, con: duckdb.frame.Connection) -> None:
        con.register("c", pa.chunked_array([list(range(3000)), list(range(3000, 5000))]))
        assert table("c").schema(con) == [("value", "BIGINT")]
        assert rows(con, "SELECT count(*), sum(value), max(value) FROM c") == [(5000, 12_497_500, 4999)]

    def test_an_array_with_nulls_and_offsets(self, con: duckdb.frame.Connection) -> None:
        sliced = pa.array(["x", None, "y", "z"]).slice(1, 2)
        con.register("s", sliced)
        assert rows(con, "SELECT value FROM s") == [(None,), ("y",)]

    def test_a_struct_array_projects_by_position(self, con: duckdb.frame.Connection) -> None:
        array = pa.StructArray.from_arrays([pa.array([1]), pa.array(["a"]), pa.array([2.5])], names=["n", "s", "f"])
        con.register("st", array)
        assert rows(con, "SELECT f, n FROM st") == [(2.5, 1)]
        assert "Projections: f, n" in sql("SELECT f, n FROM st").on(con).explain()


class TestRowsAndTypes:
    def test_many_batches_arrive_in_order(self, con: duckdb.frame.Connection) -> None:
        big = pa.table({"i": list(range(100_000))})
        batched = pa.Table.from_batches(big.to_batches(max_chunksize=3_000))
        assert batched.num_rows == 100_000
        con.register("big", batched)
        assert rows(con, "SELECT count(*), sum(i), min(i), max(i) FROM big") == [(100_000, 4_999_950_000, 0, 99_999)]
        assert rows(con, "SELECT i FROM big LIMIT 3 OFFSET 2_999") == [(2_999,), (3_000,), (3_001,)]
        assert rows(con, "SELECT i FROM big") == [(i,) for i in range(100_000)]

    def test_a_record_batch_is_a_source(self, con: duckdb.frame.Connection, numbers: pa.Table) -> None:
        con.register("b", numbers.to_batches()[0])
        assert rows(con, "SELECT count(*) FROM b") == [(10,)]
        assert rows(con, "SELECT count(*) FROM b") == [(10,)]

    def test_an_empty_table_has_its_schema_and_no_rows(self, con: duckdb.frame.Connection) -> None:
        con.register("empty", pa.table({"a": pa.array([], pa.int32())}))
        assert table("empty").schema(con) == [("a", "INTEGER")]
        assert rows(con, "SELECT * FROM empty") == []

    def test_a_table_without_columns_is_refused(self, con: duckdb.frame.Connection) -> None:
        con.register("none", pa.table({}))
        with pytest.raises(exceptions.InvalidInputError, match="did not declare any result columns"):
            rows(con, "SELECT * FROM none")

    def test_the_arrow_types_arrive_as_duckdb_types(self, con: duckdb.frame.Connection) -> None:
        typed = pa.table(
            {
                "i8": pa.array([1, None], pa.int8()),
                "u64": pa.array([2**63, None], pa.uint64()),
                "f": pa.array([1.5, None]),
                "s": pa.array(["a", None]),
                "big_s": pa.array(["b", None], pa.large_string()),
                "bin": pa.array([b"\x00\x01", None], pa.binary()),
                "flag": pa.array([True, None]),
                "d": pa.array([datetime.date(2020, 1, 2), None]),
                "ts": pa.array([datetime.datetime(2020, 1, 2, 3, 4, 5, 6), None]),
                "dec": pa.array([decimal.Decimal("1.23"), None], pa.decimal128(10, 2)),
                "lst": pa.array([[1, 2], None]),
                "st": pa.array([{"x": 1, "y": "z"}, None]),
                "dictionary": pa.array(["p", "q"]).dictionary_encode(),
                "m": pa.array([[("k", 1)], None], pa.map_(pa.string(), pa.int64())),
            }
        )
        con.register("typed", typed)
        assert table("typed").schema(con) == [
            ("i8", "TINYINT"),
            ("u64", "UBIGINT"),
            ("f", "DOUBLE"),
            ("s", "VARCHAR"),
            ("big_s", "VARCHAR"),
            ("bin", "BLOB"),
            ("flag", "BOOLEAN"),
            ("d", "DATE"),
            ("ts", "TIMESTAMP"),
            ("dec", "DECIMAL(10,2)"),
            ("lst", "BIGINT[]"),
            ("st", "STRUCT(x BIGINT, y VARCHAR)"),
            ("dictionary", "VARCHAR"),
            ("m", "MAP(VARCHAR, BIGINT)"),
        ]
        first, second = rows(con, "SELECT * FROM typed")
        assert first == (
            1,
            2**63,
            1.5,
            "a",
            "b",
            b"\x00\x01",
            True,
            datetime.date(2020, 1, 2),
            datetime.datetime(2020, 1, 2, 3, 4, 5, 6),
            decimal.Decimal("1.23"),
            [1, 2],
            {"x": 1, "y": "z"},
            "p",
            {"k": 1},
        )
        assert second == (None,) * 12 + ("q", None)

    def test_the_scan_shows_in_the_plan(self, con: duckdb.frame.Connection, numbers: pa.Table) -> None:
        con.register("numbers", numbers)
        assert "PYTHON_OBJECT_SCAN" in table("numbers").on(con).explain()

    def test_the_scan_function_takes_the_name(self, con: duckdb.frame.Connection, numbers: pa.Table) -> None:
        con.register("numbers", numbers)
        assert rows(con, "SELECT count(*) FROM python_object_scan('NUMBERS')") == [(10,)]
        with pytest.raises(exceptions.InvalidInputError, match="nothing is registered as 'ghost'"):
            rows(con, "SELECT * FROM python_object_scan('ghost')")


class StreamOnly:
    """A source that counts how often it is asked for a stream."""

    def __init__(self, numbers: pa.Table) -> None:
        self.numbers = numbers
        self.exports = 0

    def __arrow_c_stream__(self, requested_schema: object = None) -> object:
        self.exports += 1
        return self.numbers.__arrow_c_stream__()


class WithSchemaAttribute(StreamOnly):
    @property
    def schema(self) -> pa.Schema:
        return self.numbers.schema


class WithSchemaDunder(StreamOnly):
    def __arrow_c_schema__(self) -> object:
        return self.numbers.schema.__arrow_c_schema__()


class TestSchemaProbing:
    """Binding takes a schema, never a stream, so a source that exports once is not spent by planning."""

    def test_a_schema_attribute_answers_binding(self, con: duckdb.frame.Connection, numbers: pa.Table) -> None:
        source = WithSchemaAttribute(numbers)
        con.register("src", source)
        assert table("src").schema(con) == [("n", "BIGINT"), ("s", "VARCHAR")]
        table("src").on(con).explain()
        assert source.exports == 0
        assert rows(con, "SELECT count(*) FROM src") == [(10,)]
        assert source.exports == 1

    def test_the_schema_dunder_is_preferred(self, con: duckdb.frame.Connection, numbers: pa.Table) -> None:
        source = WithSchemaDunder(numbers)
        con.register("src", source)
        assert table("src").schema(con) == [("n", "BIGINT"), ("s", "VARCHAR")]
        assert source.exports == 0

    def test_without_either_the_schema_costs_an_export(self, con: duckdb.frame.Connection, numbers: pa.Table) -> None:
        source = StreamOnly(numbers)
        con.register("src", source)
        assert table("src").schema(con) == [("n", "BIGINT"), ("s", "VARCHAR")]
        assert source.exports == 1
        assert rows(con, "SELECT count(*) FROM src") == [(10,)]
        assert source.exports == 3

    def test_a_repeatable_object_is_read_twice_in_one_query(
        self, con: duckdb.frame.Connection, numbers: pa.Table
    ) -> None:
        source = WithSchemaAttribute(numbers)
        con.register("src", source)
        assert rows(con, "SELECT count(*) FROM (SELECT * FROM src UNION ALL SELECT * FROM src)") == [(20,)]
        assert source.exports == 2


class TestNamesAndShadowing:
    def test_a_view_wins_and_a_registration_waits_behind_it(
        self, con: duckdb.frame.Connection, numbers: pa.Table
    ) -> None:
        con.run("CREATE VIEW numbers AS SELECT 42 AS v")
        con.register("numbers", numbers)
        assert rows(con, "SELECT * FROM numbers") == [(42,)]
        con.run("DROP VIEW numbers")
        assert rows(con, "SELECT count(*) FROM numbers") == [(10,)]

    def test_a_non_ascii_name_folds_only_its_ascii_letters(
        self, con: duckdb.frame.Connection, numbers: pa.Table
    ) -> None:
        con.register("Ärger", numbers)
        assert rows(con, 'SELECT count(*) FROM "Ärger"') == [(10,)]
        assert rows(con, "SELECT count(*) FROM Ärger") == [(10,)]
        with pytest.raises(exceptions.CatalogError):
            rows(con, 'SELECT count(*) FROM "ärger"')

    def test_a_name_with_quotes_dots_and_spaces(self, con: duckdb.frame.Connection, numbers: pa.Table) -> None:
        name = 'test with .s and "s and  s'
        con.register(name, numbers)
        assert table(name).on(con).count() == 10
        assert rows(con, 'SELECT count(*) FROM "test with .s and ""s and  s"') == [(10,)]
        con.unregister(name)
        with pytest.raises(exceptions.CatalogError):
            table(name).on(con).count()

    def test_duplicate_column_names_are_renamed_minimally(self, con: duckdb.frame.Connection) -> None:
        columns = [pa.array([1]), pa.array([2]), pa.array([3]), pa.array([4])]
        con.register("dup", pa.table(columns, names=["a_1", "a", "a", "a_2"]))
        assert table("dup").columns(con) == ["a_1", "a", "a_2", "a_2_1"]
        assert rows(con, "SELECT * FROM dup") == [(1, 2, 3, 4)]

    def test_case_colliding_column_names_are_renamed(self, con: duckdb.frame.Connection) -> None:
        con.register("dup", pa.table([pa.array([1]), pa.array([2])], names=["A", "a"]))
        assert table("dup").columns(con) == ["A", "a_1"]

    def test_a_schema_capsule_is_not_a_stream(self, con: duckdb.frame.Connection, numbers: pa.Table) -> None:
        with pytest.raises(TypeError, match="a bare 'arrow_schema' capsule is not an Arrow stream"):
            con.register("wrong", numbers.schema.__arrow_c_schema__())

    def test_a_source_handing_back_the_wrong_capsule_is_refused(
        self, con: duckdb.frame.Connection, numbers: pa.Table
    ) -> None:
        class WrongKind(Recording):
            def stream(self, columns: Sequence[int] | None) -> tuple[object, bool]:
                return self.obj.schema.__arrow_c_schema__(), False

        con.register("wrong", WrongKind(numbers))
        with pytest.raises(exceptions.InvalidInputError, match="did not export an 'arrow_array_stream' capsule but"):
            rows(con, "SELECT * FROM wrong")

    def test_dropping_a_registered_name_as_a_table_fails_as_for_a_missing_table(
        self, con: duckdb.frame.Connection, numbers: pa.Table
    ) -> None:
        con.register("numbers", numbers)
        with pytest.raises(exceptions.CatalogError, match="numbers"):
            con.run("DROP TABLE numbers")
        assert rows(con, "SELECT count(*) FROM numbers") == [(10,)]


class TestLifetimes:
    def test_an_object_serves_a_second_database_after_the_first_closes(self, numbers: pa.Table) -> None:
        first = duckdb.frame.connect()
        first.register("numbers", numbers)
        assert rows(first, "SELECT count(*) FROM numbers") == [(10,)]
        first.close()
        gc.collect()
        second = duckdb.frame.connect()
        second.register("numbers", numbers)
        assert rows(second, "SELECT sum(n) FROM numbers") == [(45,)]

    def test_registrations_do_not_survive_reopening_a_file(self, numbers: pa.Table, tmp_path: Path) -> None:
        path = tmp_path / "db.duckdb"
        con = duckdb.frame.connect(path)
        con.register("numbers", numbers)
        assert rows(con, "SELECT count(*) FROM numbers") == [(10,)]
        assert rows(con, "SELECT count(*) FROM duckdb_tables()") == [(0,)]
        con.close()
        reopened = duckdb.frame.connect(path)
        assert rows(reopened, "SELECT count(*) FROM duckdb_tables()") == [(0,)]
        with pytest.raises(exceptions.CatalogError):
            rows(reopened, "SELECT * FROM numbers")

    def test_a_register_query_unregister_loop_leaks_nothing(self, con: duckdb.frame.Connection) -> None:
        resource = pytest.importorskip("resource")
        numbers = pa.table({"n": list(range(50_000)), "s": [str(i) for i in range(50_000)]})

        def cycle() -> None:
            con.register("numbers", numbers)
            assert rows(con, "SELECT count(*) FROM numbers") == [(50_000,)]
            con.unregister("numbers")

        for _ in range(5):
            cycle()
        gc.collect()
        start_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        start_objects = len(gc.get_objects())
        for _ in range(100):
            cycle()
        gc.collect()
        end_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        end_objects = len(gc.get_objects())
        assert end_rss < start_rss * 1.5
        assert end_objects - start_objects < 100


class TestExtensionTypes:
    def test_uuid_and_json(self, con: duckdb.frame.Connection) -> None:
        identity = uuid.uuid4()
        con.register(
            "ext", pa.table({"u": pa.array([identity.bytes], pa.uuid()), "j": pa.array(['{"a": 1}'], pa.json_())})
        )
        assert table("ext").schema(con) == [("u", "UUID"), ("j", "JSON")]
        assert rows(con, "SELECT u, j->>'a' FROM ext") == [(identity, "1")]

    def test_bool8_is_true_for_any_nonzero_byte(self, con: duckdb.frame.Connection) -> None:
        stored = pa.array([-1, 0, 1, 2, None], pa.int8())
        con.register("flags", pa.table({"b": pa.ExtensionArray.from_storage(pa.bool8(), stored)}))
        assert table("flags").schema(con) == [("b", "BOOLEAN")]
        assert rows(con, "SELECT b FROM flags") == [(True,), (False,), (True,), (True,), (None,)]

    def test_an_unknown_extension_falls_back_to_its_storage(self, con: duckdb.frame.Connection) -> None:
        storage = pa.array([7], pa.int32())
        field = pa.field("x", pa.int32(), metadata={"ARROW:extension:name": "mystery.ext"})
        con.register("ext", pa.table([storage], schema=pa.schema([field])))
        assert table("ext").schema(con) == [("x", "INTEGER")]
        assert rows(con, "SELECT * FROM ext") == [(7,)]

    def test_opaque_hugeint(self, con: duckdb.frame.Connection) -> None:
        big = 2**100
        storage = pa.array([big.to_bytes(16, "little", signed=True)], pa.binary(16))
        con.register(
            "ext",
            pa.table({"h": pa.ExtensionArray.from_storage(pa.opaque(pa.binary(16), "hugeint", "DuckDB"), storage)}),
        )
        assert table("ext").schema(con) == [("h", "HUGEINT")]
        assert rows(con, "SELECT h FROM ext") == [(big,)]

    def test_malformed_metadata_on_a_known_extension_is_an_error(self, con: duckdb.frame.Connection) -> None:
        field = pa.field(
            "h",
            pa.binary(16),
            metadata={"ARROW:extension:name": "arrow.opaque", "ARROW:extension:metadata": "not json"},
        )
        con.register("ext", pa.table([pa.array([b"\x00" * 16], pa.binary(16))], schema=pa.schema([field])))
        with pytest.raises(exceptions.Error, match="Failed to parse JSON string"):
            rows(con, "SELECT * FROM ext")
        assert rows(con, "SELECT 1") == [(1,)]


class TestTemporalAndBinaryTypes:
    def test_time_at_every_precision(self, con: duckdb.frame.Connection) -> None:
        con.register(
            "times",
            pa.table(
                {
                    "s": pa.array([datetime.time(1, 2, 3)], pa.time32("s")),
                    "ms": pa.array([datetime.time(1, 2, 3, 4000)], pa.time32("ms")),
                    "us": pa.array([datetime.time(1, 2, 3, 4)], pa.time64("us")),
                    "ns": pa.array([1_500], pa.time64("ns")),
                }
            ),
        )
        assert table("times").schema(con) == [("s", "TIME"), ("ms", "TIME"), ("us", "TIME"), ("ns", "TIME_NS")]
        assert rows(con, "SELECT * FROM times") == [
            (datetime.time(1, 2, 3), datetime.time(1, 2, 3, 4000), datetime.time(1, 2, 3, 4), datetime.time(0, 0, 0, 1))
        ]

    def test_timestamps_keep_their_unit_and_zone(self, con: duckdb.frame.Connection) -> None:
        utc = datetime.UTC
        con.register(
            "stamps",
            pa.table(
                {
                    "s": pa.array([1], pa.timestamp("s")),
                    "ms": pa.array([1], pa.timestamp("ms")),
                    "ns": pa.array([1_500], pa.timestamp("ns")),
                    "tz": pa.array([datetime.datetime(2020, 1, 1, tzinfo=utc)], pa.timestamp("us", "UTC")),
                    "ny": pa.array([1], pa.timestamp("ms", "America/New_York")),
                }
            ),
        )
        assert table("stamps").schema(con) == [
            ("s", "TIMESTAMP_S"),
            ("ms", "TIMESTAMP_MS"),
            ("ns", "TIMESTAMP_NS"),
            ("tz", "TIMESTAMP WITH TIME ZONE"),
            ("ny", "TIMESTAMP WITH TIME ZONE"),
        ]
        epoch = datetime.datetime(1970, 1, 1)
        assert rows(con, "SELECT * FROM stamps") == [
            (
                epoch + datetime.timedelta(seconds=1),
                epoch + datetime.timedelta(milliseconds=1),
                epoch + datetime.timedelta(microseconds=1),
                datetime.datetime(2020, 1, 1, tzinfo=utc),
                datetime.datetime(1970, 1, 1, 0, 0, 0, 1000, tzinfo=utc),
            )
        ]

    def test_a_coarse_timestamp_beyond_microseconds_is_unrepresentable_not_wrapped(
        self, con: duckdb.frame.Connection
    ) -> None:
        con.register(
            "far", pa.table({"s": pa.array([2**62], pa.timestamp("s")), "ms": pa.array([2**62], pa.timestamp("ms"))})
        )
        message = r"timestamp 4611686018427387904 seconds since the epoch is outside the range Python's datetime"
        with pytest.raises(exceptions.ConversionError, match=message):
            rows(con, "SELECT s FROM far")
        with pytest.raises(exceptions.ConversionError, match="4611686018427387904 milliseconds since the epoch"):
            rows(con, "SELECT ms FROM far")

    def test_durations_and_intervals(self, con: duckdb.frame.Connection) -> None:
        con.register(
            "spans",
            pa.table(
                {
                    "s": pa.array([1], pa.duration("s")),
                    "ms": pa.array([1], pa.duration("ms")),
                    "us": pa.array([1], pa.duration("us")),
                    "ns": pa.array([1_500], pa.duration("ns")),
                    "mdn": pa.array([pa.MonthDayNano([1, 2, 3_000])], pa.month_day_nano_interval()),
                }
            ),
        )
        assert table("spans").types(con) == ["INTERVAL"] * 5
        assert rows(con, "SELECT * FROM spans") == [
            (
                datetime.timedelta(seconds=1),
                datetime.timedelta(milliseconds=1),
                datetime.timedelta(microseconds=1),
                datetime.timedelta(microseconds=1),
                datetime.timedelta(days=32, microseconds=3),
            )
        ]

    def test_a_duration_beyond_microseconds_is_a_conversion_error(self, con: duckdb.frame.Connection) -> None:
        con.register("spans", pa.table({"s": pa.array([2**62], pa.duration("s"))}))
        with pytest.raises(exceptions.ConversionError, match="Could not convert Interval to Microsecond"):
            rows(con, "SELECT * FROM spans")

    def test_binary_and_date_variants(self, con: duckdb.frame.Connection) -> None:
        last = datetime.date(9999, 12, 31)
        con.register(
            "bytes_and_days",
            pa.table(
                {
                    "fixed": pa.array([b"ab"], pa.binary(2)),
                    "large": pa.array([b"x"], pa.large_binary()),
                    "d64": pa.array([last], pa.date64()),
                    "d32": pa.array([last], pa.date32()),
                }
            ),
        )
        assert table("bytes_and_days").types(con) == ["BLOB", "BLOB", "DATE", "DATE"]
        assert rows(con, "SELECT * FROM bytes_and_days") == [(b"ab", b"x", last, last)]


class TestUnsupportedAndUnusualSchemas:
    """Registration validates nothing; the first query over an object reports what the importer cannot take."""

    def test_decimal256_fails_at_the_first_query(self, con: duckdb.frame.Connection) -> None:
        con.register("wide", pa.table({"d": pa.array([decimal.Decimal("1.5")], pa.decimal256(12, 4))}))
        with pytest.raises(exceptions.NotSupportedError, match="Decimal"):
            rows(con, "SELECT * FROM wide")

    def test_a_dense_union_fails_at_the_first_query(self, con: duckdb.frame.Connection) -> None:
        union = pa.UnionArray.from_dense(
            pa.array([0, 1], pa.int8()), pa.array([0, 0], pa.int32()), [pa.array([1]), pa.array(["x"])]
        )
        con.register("dense", pa.table({"u": union}))
        with pytest.raises(exceptions.NotSupportedError, match="Union"):
            rows(con, "SELECT * FROM dense")

    def test_a_sparse_union_of_struct_arrives(self, con: duckdb.frame.Connection) -> None:
        union = pa.UnionArray.from_sparse(
            pa.array([0, 1], pa.int8()), [pa.array([{"a": 1}, {"a": 2}]), pa.array(["x", "y"])]
        )
        con.register("sparse", pa.table({"u": union}))
        assert table("sparse").types(con) == ['UNION("0" STRUCT(a BIGINT), "1" VARCHAR)']
        assert rows(con, "SELECT * FROM sparse") == [({"a": 1},), ("y",)]

    def test_null_typed_and_fieldless_struct_columns(self, con: duckdb.frame.Connection) -> None:
        con.register("odd", pa.table({"n": pa.array([None, None], pa.null()), "s": pa.array([{}, {}], pa.struct([]))}))
        assert table("odd").types(con) == ['"NULL"', "STRUCT"]
        assert rows(con, "SELECT * FROM odd") == [(None, {}), (None, {})]


class TestEncodedLayouts:
    """One smoke test per layout: the importer is the engine's, so only the wiring is checked."""

    def test_run_end_encoding_within_one_batch(self, con: duckdb.frame.Connection) -> None:
        con.register("ree", pa.table({"r": pc.run_end_encode(pa.array([1, 1, 2, None, None]))}))
        assert rows(con, "SELECT * FROM ree") == [(1,), (1,), (2,), (None,), (None,)]

    @pytest.mark.xfail(strict=True, reason="the engine's importer loses run ends for every chunk after the first")
    def test_run_end_encoding_across_the_batch_boundary(self, con: duckdb.frame.Connection) -> None:
        con.register("ree", pa.table({"r": pc.run_end_encode(pa.array([7] * 3000 + [8] * 3000))}))
        assert rows(con, "SELECT r, count(*) FROM ree GROUP BY r ORDER BY r") == [(7, 3000), (8, 3000)]

    def test_dictionaries_survive_the_batch_boundary(self, con: duckdb.frame.Connection) -> None:
        con.register("dict", pa.table({"d": pa.array(["x", "y"] * 3000).dictionary_encode()}))
        assert rows(con, "SELECT d, count(*) FROM dict GROUP BY d ORDER BY d") == [("x", 3000), ("y", 3000)]

    def test_views_and_nested_dictionaries(self, con: duckdb.frame.Connection) -> None:
        con.register(
            "views",
            pa.table(
                {
                    "l": pa.array([[1, 2], None, []], pa.list_view(pa.int64())),
                    "s": pa.array(["a", None, "b" * 20], pa.string_view()),
                    "b": pa.array([b"a", None, b"c"], pa.binary_view()),
                    "d": pa.array([["p"], ["q"], None]).cast(pa.list_(pa.dictionary(pa.int32(), pa.string()))),
                }
            ),
        )
        assert table("views").types(con) == ["BIGINT[]", "VARCHAR", "BLOB", "VARCHAR[]"]
        assert rows(con, "SELECT * FROM views") == [
            ([1, 2], "a", b"a", ["p"]),
            (None, None, None, ["q"]),
            ([], "b" * 20, b"c", None),
        ]

    def test_nested_offsets_past_the_list_boundary_and_a_sliced_table(self, con: duckdb.frame.Connection) -> None:
        n = 2**17 + 1
        con.register("deep", pa.table({"s": pa.array([{"l": [i]} for i in range(n)])}).slice(n - 3))
        assert rows(con, "SELECT s.l[1] FROM deep") == [(n - 3,), (n - 2,), (n - 1,)]
        con.register("part", pa.table({"i": list(range(10))}).slice(4, 3))
        assert rows(con, "SELECT i FROM part") == [(4,), (5,), (6,)]
