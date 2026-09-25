"""The object families that register as tables: pyarrow datasets and scanners, polars frames, pandas frames."""

from __future__ import annotations

import datetime
import decimal
import gc
import sys
import threading
import time
import uuid
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

import duckdb
from duckdb import exceptions
from duckdb._sources import adapt
from duckdb._sources.arrow import ArrowArraySource, ArrowStreamSource
from duckdb._sources.numpy import _object_kind
from duckdb._sources.pandas import PandasSource
from duckdb._sources.polars import LazyFrameSource, PolarsFrameSource
from duckdb._sources.pyarrow import PyArrowDatasetSource, PyArrowScannerSource
from duckdb.frame import col, sql, table

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from pathlib import Path

    from duckdb._expressions import Expr


@pytest.fixture
def con() -> duckdb.frame.Connection:
    return duckdb.frame.connect()


@pytest.fixture
def parquet_dir(tmp_path: Path) -> Path:
    for name, start in (("a", 0), ("b", 5)):
        pq.write_table(
            pa.table({"n": list(range(start, start + 5)), "s": [str(i) for i in range(start, start + 5)]}),
            tmp_path / f"{name}.parquet",
        )
    return tmp_path


def rows(con: duckdb.frame.Connection, query: str) -> list[tuple[object, ...]]:
    with con._execute(query) as result:
        return result.fetch_all()


def uneven_sizes(count: int) -> list[int]:
    """`count` batch sizes cycling through a mix of tiny and large, some of which split across several chunks."""
    pattern = [1, 7, 100, 2048, 3000, 5000]
    return (pattern * (count // len(pattern) + 1))[:count]


def check_order_preserved(con: duckdb.frame.Connection, register: Callable[[], None], name: str, total: int) -> None:
    """A monotone `n` column over `name` survives a plain scan, a windowed rank, a LIMIT and a CREATE TABLE AS."""
    register()
    assert [row[0] for row in rows(con, f"SELECT n FROM {name}")] == list(range(total))
    register()
    ranked = rows(con, f"SELECT row_number() OVER () AS r, n FROM {name}")
    assert ranked == [(n + 1, n) for n in range(total)]
    register()
    assert rows(con, f"SELECT n FROM {name} LIMIT 5") == [(i,) for i in range(5)]
    register()
    con.run(f'CREATE TABLE "{name}_copy" AS SELECT * FROM {name}')
    assert [row[0] for row in rows(con, f'SELECT n FROM "{name}_copy"')] == list(range(total))
    con.run(f'DROP TABLE "{name}_copy"')


class TestClassification:
    def test_each_family_gets_its_adapter(self, parquet_dir: Path) -> None:
        dataset = ds.dataset(parquet_dir)
        assert isinstance(adapt(dataset), PyArrowDatasetSource)
        assert isinstance(adapt(dataset.scanner()), PyArrowScannerSource)
        assert isinstance(adapt(pl.DataFrame({"a": [1]}).lazy()), LazyFrameSource)
        assert isinstance(adapt(pd.DataFrame({"a": [1]})), PandasSource)

    def test_every_family_is_read_as_often_as_asked(self, parquet_dir: Path) -> None:
        frame = pl.DataFrame({"a": [1]})
        for obj in (
            ds.dataset(parquet_dir),
            ds.dataset(parquet_dir).scanner(),
            frame,
            frame.lazy(),
            pd.DataFrame({"a": [1]}),
        ):
            assert not adapt(obj).one_shot

    def test_a_polars_frame_narrows_by_name(self) -> None:
        frame = pl.DataFrame({"a": [1], "b": ["x"]})
        source = adapt(frame)
        assert isinstance(source, PolarsFrameSource)
        capsule, projected = source.stream([1], ())
        assert projected
        assert pa.RecordBatchReader._import_from_c_capsule(capsule).read_all().column_names == ["b"]

    def test_a_duck_typed_scanner_is_a_scanner(self, parquet_dir: Path) -> None:
        class Mine:
            def __init__(self, inner: ds.Scanner) -> None:
                self.inner = inner

            @property
            def projected_schema(self) -> pa.Schema:
                return self.inner.projected_schema

            def to_reader(self) -> pa.RecordBatchReader:
                return self.inner.to_reader()

        assert isinstance(adapt(Mine(ds.dataset(parquet_dir).scanner())), PyArrowScannerSource)

    def test_a_class_named_like_a_reader_still_needs_the_export(self) -> None:
        class RecordBatchReader:
            pass

        with pytest.raises(TypeError, match="none of these"):
            adapt(RecordBatchReader())

    def test_series_and_fragments_are_not_datasets(self, parquet_dir: Path) -> None:
        for series in (pd.Series([1]), pl.Series([1])):
            source = adapt(series)
            assert isinstance(source, ArrowStreamSource)
            assert not source.one_shot
        assert isinstance(adapt(pa.array([1])), ArrowArraySource)
        fragment = next(iter(ds.dataset(parquet_dir).get_fragments()))
        with pytest.raises(TypeError, match="none of these"):
            adapt(fragment)

    def test_a_duck_typed_dataset_is_a_dataset(self, parquet_dir: Path) -> None:
        class Mine:
            def __init__(self, inner: ds.Dataset) -> None:
                self.inner = inner

            @property
            def schema(self) -> pa.Schema:
                return self.inner.schema

            def scanner(self) -> ds.Scanner:
                return self.inner.scanner()

        assert isinstance(adapt(Mine(ds.dataset(parquet_dir))), PyArrowDatasetSource)


class TestDatasets:
    def test_a_dataset_over_files_is_read_repeatedly(self, con: duckdb.frame.Connection, parquet_dir: Path) -> None:
        con.register("files", ds.dataset(parquet_dir))
        assert table("files").schema(con) == [("n", "BIGINT"), ("s", "VARCHAR")]
        assert rows(con, "SELECT count(*), sum(n) FROM files") == [(10, 45)]
        assert rows(con, "SELECT count(*) FROM files a JOIN files b USING (n)") == [(10,)]

    def test_a_scanner_keeps_its_own_projection_and_filter(
        self, con: duckdb.frame.Connection, parquet_dir: Path
    ) -> None:
        scanner = ds.dataset(parquet_dir).scanner(columns=["n"], filter=ds.field("n") >= 7)
        con.register("part", scanner)
        assert table("part").schema(con) == [("n", "BIGINT")]
        assert rows(con, "SELECT n FROM part ORDER BY n") == [(7,), (8,), (9,)]
        assert rows(con, "SELECT count(*) FROM part") == [(3,)]

    def test_a_partitioned_dataset_carries_its_partition_column(
        self, con: duckdb.frame.Connection, tmp_path: Path
    ) -> None:
        for year in (2020, 2021):
            (tmp_path / f"year={year}").mkdir()
            pq.write_table(pa.table({"n": [year - 2000]}), tmp_path / f"year={year}" / "part.parquet")
        con.register("years", ds.dataset(tmp_path, partitioning="hive"))
        assert table("years").schema(con) == [("n", "BIGINT"), ("year", "INTEGER")]
        assert rows(con, "SELECT year, n FROM years ORDER BY year") == [(2020, 20), (2021, 21)]
        assert rows(con, "SELECT year FROM years ORDER BY year") == [(2020,), (2021,)]
        capsule, projected = adapt(ds.dataset(tmp_path, partitioning="hive")).stream([1], ())
        assert projected
        assert columns_of(capsule) == ["year"]

    def test_a_dataset_is_scanned_per_query(self, con: duckdb.frame.Connection, parquet_dir: Path) -> None:
        con.register("files", ds.dataset(parquet_dir))
        assert rows(con, "SELECT count(*) FROM files") == [(10,)]
        assert rows(con, "SELECT max(n) FROM files") == [(9,)]


def columns_of(capsule: object) -> list[str]:
    return list(pa.RecordBatchReader._import_from_c_capsule(capsule).read_all().column_names)


class TestProjection:
    def test_a_dataset_scanner_is_asked_for_the_columns(self, con: duckdb.frame.Connection, parquet_dir: Path) -> None:
        asked: list[object] = []
        inner = ds.dataset(parquet_dir)

        class Mine:
            schema = inner.schema

            def scanner(self, **kwargs: object) -> ds.Scanner:
                asked.append(kwargs.get("columns"))
                return inner.scanner(**kwargs)

        con.register("files", Mine())
        assert rows(con, "SELECT s FROM files ORDER BY s LIMIT 1") == [("0",)]
        assert rows(con, "SELECT count(*) FROM files") == [(10,)]
        assert asked[0] == ["s"]
        assert len(asked) == 2

    def test_a_lazy_frame_is_narrowed_before_it_is_collected(self, con: duckdb.frame.Connection) -> None:
        seen: list[list[str]] = []

        def peek(frame: pl.DataFrame) -> pl.DataFrame:
            seen.append(frame.columns)
            return frame

        lazy = pl.DataFrame({"i": range(5), "s": ["a", "b", "c", "d", "e"], "unused": [0] * 5}).lazy()
        con.register("lf", lazy.map_batches(peek))
        assert rows(con, "SELECT s FROM lf WHERE i = 2") == [("c",)]
        capsule, projected = adapt(lazy).stream([1], ())
        assert projected
        assert columns_of(capsule) == ["s"]

    def test_pandas_converts_only_the_requested_columns(self) -> None:
        frame = pd.DataFrame({"i": [1, 2], "s": ["a", "b"], "o": pd.Series([object(), object()], dtype=object)})
        source = adapt(frame)
        capsule, projected = source.stream([1, 0], ())
        assert projected
        assert columns_of(capsule) == ["s", "i"]

    def test_tables_and_scanners(self, parquet_dir: Path) -> None:
        table_ = pa.table({"a": [1], "b": [2], "c": [3]})
        capsule, projected = adapt(table_).stream([2, 0], ())
        assert projected
        assert columns_of(capsule) == ["c", "a"]
        scanner = ds.dataset(parquet_dir).scanner(columns=["s", "n"])
        capsule, projected = adapt(scanner).stream([1], ())
        assert not projected
        assert columns_of(capsule) == ["s", "n"]


class TestCardinality:
    def test_frames_and_datasets_know_their_length(self, con: duckdb.frame.Connection, parquet_dir: Path) -> None:
        assert adapt(pl.DataFrame({"a": range(50)})).rows() == 50
        assert adapt(pd.DataFrame({"a": range(77)})).rows() == 77
        assert adapt(ds.dataset(parquet_dir)).rows() == 10
        assert adapt(pa.array([1, 2, 3, 4])).rows() == 4
        assert adapt(pa.chunked_array([[1, 2], [3]])).rows() == 3
        con.register("pdf", pd.DataFrame({"a": range(77)}))
        assert "~77 rows" in table("pdf").on(con).explain()
        con.register("files", ds.dataset(parquet_dir))
        assert "~10 rows" in table("files").on(con).explain()

    def test_plans_and_scanners_do_not(self, parquet_dir: Path) -> None:
        assert adapt(pl.DataFrame({"a": range(50)}).lazy()).rows() is None
        assert adapt(ds.dataset(parquet_dir).scanner(filter=ds.field("n") > 3)).rows() is None

    def test_a_filtered_parquet_dataset_counts_what_it_will_scan(
        self, con: duckdb.frame.Connection, parquet_dir: Path
    ) -> None:
        filtered = ds.dataset(parquet_dir).filter(ds.field("n") >= 7)
        assert adapt(filtered).rows() == 3
        con.register("late", filtered)
        assert "~3 rows" in table("late").on(con).explain()
        assert rows(con, "SELECT count(*) FROM late") == [(3,)]

    def test_a_csv_dataset_does_not_count_by_reading(self, tmp_path: Path) -> None:
        (tmp_path / "rows.csv").write_text("n\n1\n2\n3\n")
        assert adapt(ds.dataset(tmp_path / "rows.csv", format="csv")).rows() is None


class TestSeries:
    def test_a_pandas_series_is_one_column(self, con: duckdb.frame.Connection) -> None:
        con.register("s", pd.Series([1.5, None], name="f"))
        assert table("s").schema(con) == [("value", "DOUBLE")]
        assert rows(con, "SELECT value FROM s") == [(1.5,), (None,)]

    def test_a_polars_series_keeps_its_name(self, con: duckdb.frame.Connection) -> None:
        con.register("s", pl.Series("p", ["x", None]))
        assert table("s").schema(con) == [("p", "VARCHAR")]
        assert rows(con, "SELECT p FROM s") == [("x",), (None,)]


class TestPolars:
    def test_dtypes_arrive_as_duckdb_types(self, con: duckdb.frame.Connection) -> None:
        frame = pl.DataFrame(
            {
                "i": [1, None],
                "f": [1.5, None],
                "s": ["a", None],
                "b": [True, None],
                "bin": [b"x", None],
                "d": [datetime.date(2020, 1, 2), None],
                "dt": [datetime.datetime(2020, 1, 2, 3), None],
                "t": [datetime.time(1, 2), None],
                "dur": [datetime.timedelta(seconds=1), None],
                "l": [[1, 2], None],
                "st": [{"x": 1}, None],
                "dec": [decimal.Decimal("1.50"), None],
            }
        ).with_columns(
            pl.col("s").cast(pl.Categorical).alias("cat"),
            pl.col("s").cast(pl.Enum(["a", "b"])).alias("en"),
            pl.col("dt").dt.replace_time_zone("UTC").alias("dtz"),
            pl.col("i").cast(pl.UInt8).alias("u8"),
        )
        con.register("pf", frame)
        assert table("pf").schema(con) == [
            ("i", "BIGINT"),
            ("f", "DOUBLE"),
            ("s", "VARCHAR"),
            ("b", "BOOLEAN"),
            ("bin", "BLOB"),
            ("d", "DATE"),
            ("dt", "TIMESTAMP"),
            ("t", "TIME_NS"),
            ("dur", "INTERVAL"),
            ("l", "BIGINT[]"),
            ("st", "STRUCT(x BIGINT)"),
            ("dec", "DECIMAL(38,2)"),
            ("cat", "VARCHAR"),
            ("en", "VARCHAR"),
            ("dtz", "TIMESTAMP WITH TIME ZONE"),
            ("u8", "UTINYINT"),
        ]
        first, second = rows(con, "SELECT * FROM pf")
        assert first == (
            1,
            1.5,
            "a",
            True,
            b"x",
            datetime.date(2020, 1, 2),
            datetime.datetime(2020, 1, 2, 3),
            datetime.time(1, 2),
            datetime.timedelta(seconds=1),
            [1, 2],
            {"x": 1},
            decimal.Decimal("1.50"),
            "a",
            "a",
            datetime.datetime(2020, 1, 2, 3, tzinfo=datetime.UTC),
            1,
        )
        assert second == (None,) * 16

    def test_a_query_over_a_subset_is_narrowed_at_the_frame(self, con: duckdb.frame.Connection) -> None:
        con.register("pf", pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [0.5] * 3}))
        assert rows(con, "SELECT b FROM pf WHERE a = 2") == [("y",)]
        assert "Projections: a, b" in sql("SELECT b FROM pf WHERE a = 2").on(con).explain()

    def test_a_frame_is_read_repeatedly_and_in_order(self, con: duckdb.frame.Connection) -> None:
        con.register("pf", pl.DataFrame({"i": range(10_000)}))
        assert rows(con, "SELECT count(*), sum(i) FROM pf") == [(10_000, 49_995_000)]
        assert rows(con, "SELECT i FROM pf") == [(i,) for i in range(10_000)]

    def test_a_lazy_frame_is_collected_per_query(self, con: duckdb.frame.Connection) -> None:
        collected = 0

        def counting(batch: pl.DataFrame) -> pl.DataFrame:
            nonlocal collected
            collected += 1
            return batch

        lazy = pl.DataFrame({"i": range(10)}).lazy().map_batches(counting).filter(pl.col("i") >= 5)
        con.register("lf", lazy)
        assert table("lf").schema(con) == [("i", "BIGINT")]
        assert collected == 0
        assert rows(con, "SELECT sum(i) FROM lf") == [(35,)]
        assert rows(con, "SELECT count(*) FROM lf") == [(5,)]
        assert collected == 2

    def test_a_lazy_frame_that_cannot_run_fails_at_the_query(
        self, con: duckdb.frame.Connection, tmp_path: Path
    ) -> None:
        con.register("missing", pl.scan_parquet(tmp_path / "nope.parquet"))
        with pytest.raises(exceptions.Error, match=r"nope\.parquet"):
            rows(con, "SELECT * FROM missing")

    def test_a_plan_whose_output_differs_from_its_declared_schema_fails_at_the_query(
        self, con: duckdb.frame.Connection
    ) -> None:
        lazy = pl.DataFrame({"i": [1, 2]}).lazy().map_batches(lambda b: b.with_columns(pl.col("i").cast(pl.Utf8)))
        con.register("wrong", lazy)
        assert table("wrong").schema(con) == [("i", "BIGINT")]
        with pytest.raises(exceptions.InvalidInputError, match=r"registered as 'wrong' failed.*schema"):
            rows(con, "SELECT * FROM wrong")
        con.register("wrong", pl.DataFrame({"i": [1, 2]}).lazy())
        assert rows(con, "SELECT sum(i) FROM wrong") == [(3,)]

    def test_categoricals_read_the_same_twice_through_a_lazy_frame(self, con: duckdb.frame.Connection) -> None:
        lazy = pl.DataFrame({"c": ["x", "y", "x", None]}).lazy().with_columns(pl.col("c").cast(pl.Categorical))
        con.register("cats", lazy)
        for _ in range(2):
            assert rows(con, "SELECT c, count(*) FROM cats GROUP BY c ORDER BY c") == [("x", 2), ("y", 1), (None, 1)]

    def test_int128_is_not_supported_by_the_engine(self, con: duckdb.frame.Connection) -> None:
        con.register("wide", pl.DataFrame({"i": [1]}).cast(pl.Int128))
        with pytest.raises(exceptions.NotSupportedError, match="_pli128"):
            rows(con, "SELECT * FROM wide")


class TestPandas:
    def test_dtypes_arrive_as_duckdb_types(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame(
            {
                "i": pd.array([1, None], dtype="Int64"),
                "f": [1.5, float("nan")],
                "s": ["a", None],
                "o": pd.Series(["a", None], dtype=object),
                "b": pd.array([True, None], dtype="boolean"),
                "cat": pd.Categorical(["a", None]),
                "dt": pd.to_datetime(["2020-01-02", None]),
                "td": pd.to_timedelta(["1s", None]),
                "arrow_s": pd.array(["a", None], dtype="string[pyarrow]"),
            }
        )
        con.register("df", frame)
        assert table("df").schema(con) == [
            ("i", "BIGINT"),
            ("f", "DOUBLE"),
            ("s", "VARCHAR"),
            ("o", "VARCHAR"),
            ("b", "BOOLEAN"),
            ("cat", "VARCHAR"),
            ("dt", "TIMESTAMP"),
            ("td", "INTERVAL"),
            ("arrow_s", "VARCHAR"),
        ]
        first, second = rows(con, "SELECT * FROM df")
        assert first == (1, 1.5, "a", "a", True, "a", datetime.datetime(2020, 1, 2), datetime.timedelta(seconds=1), "a")
        assert second == (None,) * 9

    def test_the_index_is_left_out(self, con: duckdb.frame.Connection) -> None:
        con.register("df", pd.DataFrame({"a": [1, 2]}, index=pd.Index([10, 20], name="k")))
        assert table("df").schema(con) == [("a", "BIGINT")]
        assert rows(con, "SELECT * FROM df") == [(1,), (2,)]
        con.register("reset", pd.DataFrame({"a": [1, 2]}, index=pd.Index([10, 20], name="k")).reset_index())
        assert table("reset").columns(con) == ["k", "a"]

    def test_zones_and_units(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame(
            {
                "dtz": pd.to_datetime(["2020-01-02"]).tz_localize("Europe/Amsterdam"),
                "dts": pd.to_datetime(["2020-01-02"]).astype("datetime64[s]"),
            }
        )
        con.register("df", frame)
        assert table("df").types(con) == ["TIMESTAMP WITH TIME ZONE", "TIMESTAMP_S"]
        assert rows(con, "SELECT * FROM df") == [
            (datetime.datetime(2020, 1, 1, 23, tzinfo=datetime.UTC), datetime.datetime(2020, 1, 2))
        ]

    def test_a_frame_is_read_repeatedly_and_filters_apply(self, con: duckdb.frame.Connection) -> None:
        con.register("df", pd.DataFrame({"i": range(10_000)}))
        assert rows(con, "SELECT count(*), sum(i) FROM df") == [(10_000, 49_995_000)]
        assert table("df").filter(col("i") < 3).on(con).rows() == [(0,), (1,), (2,)]

    def test_an_empty_frame(self, con: duckdb.frame.Connection) -> None:
        con.register("df", pd.DataFrame({"a": pd.array([], dtype="Int64")}))
        assert table("df").schema(con) == [("a", "BIGINT")]
        assert rows(con, "SELECT * FROM df") == []
        con.register("nothing", pd.DataFrame())
        with pytest.raises(exceptions.InvalidInputError, match="did not declare any result columns"):
            rows(con, "SELECT * FROM nothing")

    def test_a_frame_is_converted_a_slice_at_a_time(
        self, con: duckdb.frame.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(PandasSource, "SLICE_ROWS", 1000)
        frame = pd.DataFrame({"i": range(10_500), "s": [str(i) for i in range(10_500)]})
        source = adapt(frame)
        assert isinstance(source, PandasSource)
        schema, forced = source._schema(frame)
        assert [batch.num_rows for batch in source._batches(frame, schema, forced)] == [1000] * 10 + [500]
        con.register("df", frame)
        assert rows(con, "SELECT count(*), sum(i), max(s) FROM df") == [(10_500, 55_119_750, "9999")]

    def test_types_come_from_a_sample_and_a_misfit_later_value_fails_the_query(
        self, con: duckdb.frame.Connection
    ) -> None:
        # Forced onto the Arrow path: the native scan's "text" kind stringifies a later int rather than refusing
        # it, since the sampled type only ever forces VARCHAR there, never a narrower one a stray int could miss.
        values: list[object] = ["text"] * 5000
        values[4999] = 12
        source = PandasSource(pd.DataFrame({"o": pd.Series(values, dtype=object)}))
        source.native = False
        con.register("df", source)
        assert table("df").schema(con) == [("o", "VARCHAR")]
        with pytest.raises(exceptions.InvalidInputError, match="registered as 'df' failed"):
            rows(con, "SELECT count(*) FROM df")

    def test_without_pyarrow_a_native_frame_still_registers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        monkeypatch.setitem(sys.modules, "pyarrow", None)
        source = adapt(pd.DataFrame({"a": [1]}))
        assert source.native

    def test_without_pyarrow_a_pyarrow_backed_column_says_so(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        # Built while pyarrow is still real: the column is pyarrow-backed before the import is blocked below.
        frame = pd.DataFrame({"a": pd.array(["a", "b"], dtype="string[pyarrow]")})
        monkeypatch.setitem(sys.modules, "pyarrow", None)
        with pytest.raises(TypeError, match="needs pyarrow"):
            adapt(frame)


@pytest.fixture
def typed_dir(tmp_path: Path) -> Path:
    """Two parquet files under hive partitions, with a NULL and a NaN in every column that can hold one."""
    for year, start in ((2020, 0), (2021, 5)):
        (tmp_path / f"year={year}").mkdir()
        ns = [start, start + 1, None, start + 3, start + 4]
        pq.write_table(
            pa.table(
                {
                    "n": pa.array(ns, pa.int32()),
                    "s": [None if n is None else str(n) for n in ns],
                    "f": [float("nan") if n == start + 3 else (None if n is None else float(n)) for n in ns],
                    "flag": [None if n is None else n % 2 == 0 for n in ns],
                    "d": [None if n is None else datetime.date(year, 1, n + 1) for n in ns],
                    "ts": pa.array(
                        [None if n is None else datetime.datetime(year, 1, 1, n) for n in ns], pa.timestamp("us")
                    ),
                    "ts_ns": pa.array(
                        [None if n is None else datetime.datetime(year, 1, 1, n) for n in ns], pa.timestamp("ns")
                    ),
                }
            ),
            tmp_path / f"year={year}" / "part.parquet",
        )
    return tmp_path


class Pushing(PyArrowDatasetSource):
    """A dataset source that remembers what it was offered and what each scan applied."""

    def __init__(self, dataset: ds.Dataset) -> None:
        super().__init__(dataset)
        self.offered: list[str] = []
        self.applied: list[list[str]] = []

    def accepts(self, predicate: Expr) -> bool:
        self.offered.append(predicate.fragment())
        return super().accepts(predicate)

    def stream(self, columns: Sequence[int] | None, filters: Sequence[Expr]) -> tuple[object, bool]:
        self.applied.append([predicate.fragment() for predicate in filters])
        return super().stream(columns, filters)


PUSHED = [
    "n > 6",
    "6 < n",
    "n >= 6",
    "n < 3",
    "n <= 3",
    "n = 3",
    "n <> 3",
    "n IN (1, 3, 5)",
    "n NOT IN (1, 3, 5)",
    "NOT (n IN (0, 5) OR s = '4')",
    "NOT (n > 6)",
    "n > 1 AND s <> '4'",
    "n < 2 OR n > 7",
    "n IS NULL",
    "s IS NOT NULL",
    "s = '5'",
    "s > '5'",
    "s IN ('1', '9')",
    "flag = true",
    "flag <> false",
    "f > 6.0",
    "f >= 8.0",
    "f < 3.0",
    "f <= 3.0",
    "f = 1.0",
    "f <> 1.0",
    "NOT (f > 6.0)",
    "d >= DATE '2021-01-02'",
    "d = DATE '2020-01-05'",
    "ts > TIMESTAMP '2021-01-01 04:00:00'",
    "ts IN (TIMESTAMP '2020-01-01 00:00:00', TIMESTAMP '2021-01-01 09:00:00')",
    "year = 2021",
    "year = 2021 AND n > 6",
]

KEPT = [
    "n BETWEEN 3 AND 6",
    "n > 1 AND n < 8",
    "n > 1.5",
    "lower(s) = '4'",
    "s = n",
    "n IN (1, NULL)",
    "n IN (1, 3.5)",
    "flag",
    "ts_ns >= TIMESTAMP '2021-01-01 00:00:00'",
    "f = 'nan'::DOUBLE",
    "d >= DATE 'infinity'",
    "d = DATE '-infinity'",
    "ts < TIMESTAMP 'infinity'",
    "ts IN (TIMESTAMP '-infinity', TIMESTAMP '2021-01-01 09:00:00')",
    "n IS DISTINCT FROM 3",
    "coalesce(n, 0) > 6",
    "CASE WHEN n > 6 THEN true ELSE false END",
    "s LIKE '%5%'",
]


def without_nan(result: list[tuple[object, ...]]) -> list[tuple[object, ...]]:
    """NaN compares unequal to itself, so rows are compared with it spelled out."""
    return [tuple("nan" if isinstance(v, float) and v != v else v for v in row) for row in result]


class TestDatasetPushdown:
    def register_both(self, con: duckdb.frame.Connection, typed_dir: Path) -> Pushing:
        source = Pushing(ds.dataset(typed_dir, partitioning="hive"))
        con.register("pushed", source)
        con.register("plain", ArrowStreamSource(ds.dataset(typed_dir, partitioning="hive").to_table()))
        return source

    @pytest.mark.parametrize("where", PUSHED)
    def test_a_pushed_predicate_selects_what_the_engine_selects(
        self, con: duckdb.frame.Connection, typed_dir: Path, where: str
    ) -> None:
        source = self.register_both(con, typed_dir)
        expected = rows(con, f"SELECT n, s, f, flag, d, ts, year FROM plain WHERE {where} ORDER BY year, n")
        pushed = rows(con, f"SELECT n, s, f, flag, d, ts, year FROM pushed WHERE {where} ORDER BY year, n")
        assert without_nan(pushed) == without_nan(expected)
        assert expected
        assert source.applied
        assert all(source.applied), where
        assert "Filter" not in sql(f"SELECT n FROM pushed WHERE {where}").on(con).explain()

    @pytest.mark.parametrize("where", KEPT)
    def test_a_predicate_outside_the_set_stays_with_the_engine(
        self, con: duckdb.frame.Connection, typed_dir: Path, where: str
    ) -> None:
        source = self.register_both(con, typed_dir)
        expected = rows(con, f"SELECT n, year FROM plain WHERE {where} ORDER BY year, n")
        assert rows(con, f"SELECT n, year FROM pushed WHERE {where} ORDER BY year, n") == expected
        assert source.applied == [[]], where

    def test_a_predicate_nothing_satisfies_yields_no_rows(self, con: duckdb.frame.Connection, typed_dir: Path) -> None:
        source = self.register_both(con, typed_dir)
        assert rows(con, "SELECT n FROM pushed WHERE n > 100") == []
        assert rows(con, "SELECT count(*) FROM pushed WHERE s = 'none'") == [(0,)]
        assert source.applied == [['("n" > 100)'], ["(\"s\" = 'none')"]]

    def test_a_conjunction_is_pushed_conjunct_by_conjunct(self, con: duckdb.frame.Connection, typed_dir: Path) -> None:
        source = self.register_both(con, typed_dir)
        where = "n > 1 AND n < 8 AND s <> '4'"
        expected = rows(con, f"SELECT n FROM plain WHERE {where} ORDER BY n")
        assert rows(con, f"SELECT n FROM pushed WHERE {where} ORDER BY n") == expected
        assert expected == [(3,), (5,), (6,)]
        assert source.offered == ["(\"s\" != '4')"]
        assert source.applied == [["(\"s\" != '4')"]]
        plan = sql(f"SELECT n FROM pushed WHERE {where}").on(con).explain()
        assert "Filter" in plan
        assert "'4'" not in plan

    def test_a_partition_filter_prunes_before_reading(self, con: duckdb.frame.Connection, typed_dir: Path) -> None:
        source = self.register_both(con, typed_dir)
        assert rows(con, "SELECT count(*) FROM pushed WHERE year = 2021 AND n IS NOT NULL") == [(4,)]
        assert sorted(source.applied[0]) == ['("n" IS NOT NULL)', '("year" = 2021)']
        fragments = ds.dataset(typed_dir, partitioning="hive").get_fragments(filter=ds.field("year") == 2021)
        assert len(list(fragments)) == 1

    def test_only_the_columns_read_are_returned(self, con: duckdb.frame.Connection, typed_dir: Path) -> None:
        source = self.register_both(con, typed_dir)
        capsule, projected = source.stream([1], (col("n") > 6,))
        assert projected
        assert columns_of(capsule) == ["s"]
        assert rows(con, "SELECT s FROM pushed WHERE n > 6 ORDER BY s") == [("8",), ("9",)]

    def test_a_scanner_keeps_refusing(self, con: duckdb.frame.Connection, typed_dir: Path) -> None:
        scanner = ds.dataset(typed_dir, partitioning="hive").scanner(columns=["n"], filter=ds.field("n") >= 7)
        assert adapt(scanner).accepts(col("n") > 8) is False
        con.register("part", scanner)
        assert rows(con, "SELECT n FROM part WHERE n > 8") == [(9,)]


class PushingLazily(LazyFrameSource):
    """A lazy frame source that remembers what each scan applied."""

    def __init__(self, plan: pl.LazyFrame) -> None:
        super().__init__(plan)
        self.applied: list[list[str]] = []

    def stream(self, columns: Sequence[int] | None, filters: Sequence[Expr]) -> tuple[object, bool]:
        self.applied.append([predicate.fragment() for predicate in filters])
        return super().stream(columns, filters)


class TestLazyFramePushdown:
    def register_both(self, con: duckdb.frame.Connection, typed_dir: Path) -> PushingLazily:
        source = PushingLazily(pl.scan_parquet(typed_dir, hive_partitioning=True))
        con.register("pushed", source)
        con.register("plain", ArrowStreamSource(pl.scan_parquet(typed_dir, hive_partitioning=True).collect()))
        return source

    @pytest.mark.parametrize("where", PUSHED)
    def test_a_pushed_predicate_selects_what_the_engine_selects(
        self, con: duckdb.frame.Connection, typed_dir: Path, where: str
    ) -> None:
        source = self.register_both(con, typed_dir)
        expected = rows(con, f"SELECT n, s, f, flag, d, ts, year FROM plain WHERE {where} ORDER BY year, n")
        pushed = rows(con, f"SELECT n, s, f, flag, d, ts, year FROM pushed WHERE {where} ORDER BY year, n")
        assert without_nan(pushed) == without_nan(expected)
        assert expected
        assert source.applied
        assert all(source.applied), where
        assert "Filter" not in sql(f"SELECT n FROM pushed WHERE {where}").on(con).explain()

    @pytest.mark.parametrize("where", KEPT)
    def test_a_predicate_outside_the_set_stays_with_the_engine(
        self, con: duckdb.frame.Connection, typed_dir: Path, where: str
    ) -> None:
        source = self.register_both(con, typed_dir)
        expected = rows(con, f"SELECT n, year FROM plain WHERE {where} ORDER BY year, n")
        assert rows(con, f"SELECT n, year FROM pushed WHERE {where} ORDER BY year, n") == expected
        assert source.applied == [[]], where

    def test_the_filter_lands_in_the_polars_plan(self, con: duckdb.frame.Connection, typed_dir: Path) -> None:
        source = self.register_both(con, typed_dir)
        capsule, projected = source.stream([1], (col("n") > 6,))
        assert projected
        assert columns_of(capsule) == ["s"]
        filtered = source.obj.filter(pl.col("n") > 6).select("s")
        assert "FILTER" in filtered.explain() or "SELECTION" in filtered.explain()
        assert rows(con, "SELECT s FROM pushed WHERE n > 6 ORDER BY s") == [("8",), ("9",)]

    def test_a_decimal_pushes_and_a_nanosecond_time_stays(self, con: duckdb.frame.Connection) -> None:
        frame = pl.DataFrame(
            {
                "money": pl.Series([decimal.Decimal("1.50"), decimal.Decimal("2.25"), None], dtype=pl.Decimal(10, 2)),
                "clock": [datetime.time(1, 2, 3), datetime.time(4, 5, 6), None],
            }
        )
        source = PushingLazily(frame.lazy())
        con.register("late", source)
        con.register("now", frame)
        assert table("late").schema(con) == [("money", "DECIMAL(10,2)"), ("clock", "TIME_NS")]
        for where, pushed in (
            ("money > 1.50", True),
            ("money IN (1.50, 9.99)", True),
            ("money IS NULL", True),
            ("clock < '04:00:00'::TIME_NS", False),
            ("clock IS NULL", True),
        ):
            expected = rows(con, f"SELECT money FROM now WHERE {where}")
            assert rows(con, f"SELECT money FROM late WHERE {where}") == expected
            assert bool(source.applied[-1]) is pushed, where

    def test_a_float32_column_compares_in_its_own_width(self, con: duckdb.frame.Connection) -> None:
        frame = pl.DataFrame({"f": pl.Series([0.1, 0.2, None], dtype=pl.Float32)})
        source = PushingLazily(frame.lazy())
        con.register("late", source)
        con.register("now", frame)
        for where in ("f > 0.1", "f = 0.1", "f >= 0.1", "f < 0.2", "f IN (0.1, 0.3)"):
            expected = rows(con, f"SELECT f FROM now WHERE {where}")
            assert rows(con, f"SELECT f FROM late WHERE {where}") == expected, where
            assert source.applied[-1], where
        assert rows(con, "SELECT count(*) FROM late WHERE f > 0.1") == [(1,)]

    def test_a_decimal_literal_beyond_the_scale_stays_with_the_engine(self, con: duckdb.frame.Connection) -> None:
        frame = pl.DataFrame(
            {"money": pl.Series([decimal.Decimal("1.50"), decimal.Decimal("-1.51")], dtype=pl.Decimal(10, 2))}
        )
        source = PushingLazily(frame.lazy())
        con.register("late", source)
        con.register("now", frame)
        for where, pushed in (
            ("money > 1.5", True),
            ("money = -1.51", True),
            ("money = 1.505", False),
            ("money > 1.499", False),
        ):
            expected = rows(con, f"SELECT money FROM now WHERE {where}")
            assert rows(con, f"SELECT money FROM late WHERE {where}") == expected, where
            assert bool(source.applied[-1]) is pushed, where

    def test_an_eager_frame_keeps_refusing(self, con: duckdb.frame.Connection) -> None:
        frame = pl.DataFrame({"n": [1, 2, 3]})
        assert adapt(frame).accepts(col("n") > 1) is False
        con.register("eager", frame)
        assert rows(con, "SELECT n FROM eager WHERE n > 1 ORDER BY n") == [(2,), (3,)]


class TestPolarsTranslation:
    schema = pl.Schema(
        {
            "n": pl.Int32(),
            "big": pl.Int64(),
            "small": pl.UInt8(),
            "f": pl.Float64(),
            "s": pl.String(),
            "d": pl.Date(),
            "ts": pl.Datetime("ms"),
            "zoned": pl.Datetime("us", "UTC"),
            "money": pl.Decimal(10, 2),
            "clock": pl.Time(),
            "tags": pl.List(pl.String()),
        }
    )

    def translate(self, predicate: Expr) -> str:
        from duckdb._expressions.polars import to_polars

        return str(to_polars(predicate, self.schema))

    def test_each_form_has_a_polars_reading(self) -> None:
        assert self.translate(col("n") > 5) == '[(col("n")) > (5)]'
        assert (
            self.translate((col("n") >= 5) & (col("s") != "x")) == '[([(col("n")) >= (5)]) & ([(col("s")) != ("x")])]'
        )
        assert (
            self.translate(~(col("n") < 5) | col("d").is_null())
            == '[([(col("n")) < (5)].not()) | (col("d").is_null())]'
        )
        assert self.translate(col("n").isin([1, 2])) == 'col("n").is_in([Series])'
        assert self.translate(col("n").between(1, 2)) == '[([(col("n")) >= (1)]) & ([(col("n")) <= (2)])]'
        assert self.translate(col("f") > 1.0) == '[(col("f")) > (1.0)]'
        assert "1.5" in self.translate(col("money") > decimal.Decimal("1.5"))
        assert "2020-01-01" in self.translate(col("ts") > datetime.datetime(2020, 1, 1))

    @pytest.mark.parametrize(
        "predicate",
        [
            col("n") > 5.5,
            col("n") > "5",
            col("s") > 5,
            col("big") > True,
            col("f") == float("nan"),
            col("n") > 5_000_000_000,
            col("small") > 256,
            col("small") > -1,
            col("zoned") > datetime.datetime(2020, 1, 1),
            col("ts") > datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC),
            col("money") > 1.5,
            col("money") > decimal.Decimal("NaN"),
            col("tags").is_null(),
            col("missing") > 1,
            col("n") > col("big"),
            col("n").isin([1, None]),
            col("n").cast("BIGINT") > 5,
            col("s").like("%x%"),
            col("n") + 1 > 5,
        ],
    )
    def test_the_rest_is_refused(self, predicate: Expr) -> None:
        from duckdb._expressions import Untranslatable

        with pytest.raises(Untranslatable):
            self.translate(predicate)


class TestArrowTranslation:
    schema = pa.schema(
        [
            ("n", pa.int32()),
            ("big", pa.int64()),
            ("f", pa.float64()),
            ("s", pa.large_string()),
            ("d", pa.date32()),
            ("ts", pa.timestamp("us")),
            ("zoned", pa.timestamp("us", tz="UTC")),
            ("money", pa.decimal128(10, 2)),
            ("tags", pa.list_(pa.string())),
        ]
    )

    def translate(self, predicate: Expr) -> str:
        from duckdb._expressions.arrow import to_arrow

        return str(to_arrow(predicate, self.schema))

    def test_each_form_has_a_pyarrow_reading(self) -> None:
        assert self.translate(col("n") > 5) == "(n > 5)"
        assert self.translate((col("n") >= 5) & (col("s") != "x")) == '((n >= 5) and (s != "x"))'
        assert (
            self.translate(~(col("n") < 5) | col("d").is_null())
            == "(invert((n < 5)) or is_null(d, {nan_is_null=false}))"
        )
        inner = "is_in(n, {value_set=int32:[\n  1,\n  2\n], null_matching_behavior=MATCH})"
        assert self.translate(col("n").isin([1, 2])) == f"if_else(is_valid(n), {inner}, null[bool])"
        assert self.translate(col("n").between(1, 2)) == "((n >= 1) and (n <= 2))"
        assert self.translate(col("ts") > datetime.datetime(2020, 1, 1)) == "(ts > 2020-01-01 00:00:00.000000)"

    def test_a_floating_point_greater_than_admits_nan(self) -> None:
        assert self.translate(col("f") > 1.0) == "((f > 1) or is_nan(f))"
        assert self.translate(col("f") >= 1.0) == "((f >= 1) or is_nan(f))"
        assert self.translate(col("f") < 1.0) == "(f < 1)"
        assert self.translate(col("f") == 1.0) == "(f == 1)"

    @pytest.mark.parametrize(
        "predicate",
        [
            col("n") > 5.5,
            col("n") > "5",
            col("s") > 5,
            col("big") > True,
            col("f") == float("nan"),
            col("n") > 5_000_000_000,
            col("zoned") > datetime.datetime(2020, 1, 1),
            col("ts") > datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC),
            col("money") > decimal.Decimal("1.5"),
            col("tags").is_null(),
            col("missing") > 1,
            col("n") > col("big"),
            col("n").isin([1, None]),
            col("n").isin([1, col("big")]),
            col("n").cast("BIGINT") > 5,
            col("s").like("%x%"),
            col("n") + 1 > 5,
            col("n").is_null().is_null(),
        ],
    )
    def test_the_rest_is_refused(self, predicate: Expr) -> None:
        from duckdb._expressions import Untranslatable

        with pytest.raises(Untranslatable):
            self.translate(predicate)


class TestPandasShapes:
    def test_a_later_value_the_type_cannot_hold_fails_instead_of_truncating(self, con: duckdb.frame.Connection) -> None:
        values: list[object] = list(range(100_000))
        values[1001] = 3.7
        con.register("df", pd.DataFrame({"a": pd.Series(values, dtype=object)}))
        assert table("df").schema(con) == [("a", "BIGINT")]
        with pytest.raises(exceptions.InvalidInputError, match="cannot hold exactly"):
            rows(con, "SELECT a FROM df WHERE a = 3")
        # Forced onto the Arrow path: its slice-at-a-time conversion only knows a value fails somewhere in the
        # slice, not which row, unlike the native scan's per-row check.
        later: list[object] = list(range(PandasSource.SLICE_ROWS + 10))
        later[PandasSource.SLICE_ROWS + 1] = 4.2
        arrow_source = PandasSource(pd.DataFrame({"a": pd.Series(later, dtype=object)}))
        arrow_source.native = False
        con.register("late", arrow_source)
        with pytest.raises(exceptions.InvalidInputError, match=f"after row {PandasSource.SLICE_ROWS}"):
            rows(con, "SELECT a FROM late WHERE a = 4")

    def test_a_later_slice_typed_differently_on_its_own_fails_too(self, con: duckdb.frame.Connection) -> None:
        # Forced onto the Arrow path: the native scan's "text" kind stringifies whatever the sample missed, and a
        # date mixed with a datetime is TIMESTAMP either way, so neither refusal below is native's to make.
        many = PandasSource.SLICE_ROWS + 4
        words: list[object] = ["a"] * many
        words[-4:] = [111, 222, 333, 444]
        words_source = PandasSource(pd.DataFrame({"s": pd.Series(words, dtype=object)}))
        words_source.native = False
        con.register("words", words_source)
        assert table("words").schema(con) == [("s", "VARCHAR")]
        with pytest.raises(exceptions.InvalidInputError, match="registered as 'words' failed"):
            rows(con, "SELECT count(*) FROM words")
        moments: list[object] = [datetime.datetime(2024, 1, 1, 12)] * many
        moments[-2:] = [datetime.date(2024, 1, 2), datetime.datetime(2024, 1, 3, 9, 15)]
        moments_source = PandasSource(pd.DataFrame({"t": pd.Series(moments, dtype=object)}))
        moments_source.native = False
        con.register("moments", moments_source)
        assert table("moments").schema(con) == [("t", "TIMESTAMP")]
        with pytest.raises(exceptions.InvalidInputError, match="registered as 'moments' failed"):
            rows(con, "SELECT max(t) FROM moments")

    def test_a_later_value_the_type_holds_exactly_is_kept(self, con: duckdb.frame.Connection) -> None:
        values: list[object] = [float(i) for i in range(5000)]
        values[4999] = 99999
        con.register("df", pd.DataFrame({"a": pd.Series(values, dtype=object)}))
        assert table("df").schema(con) == [("a", "DOUBLE")]
        assert rows(con, "SELECT a FROM df WHERE a > 5000") == [(99999.0,)]

    def test_labels_that_are_not_strings_become_their_text(self, con: duckdb.frame.Connection) -> None:
        con.register("matrix", pd.DataFrame([[1, "a", 2.5], [3, "b", 4.5]]))
        assert table("matrix").columns(con) == ["0", "1", "2"]
        assert rows(con, 'SELECT "2", "0" FROM matrix ORDER BY "0"') == [(2.5, 1), (4.5, 3)]
        levels = pd.MultiIndex.from_tuples([("a", "x"), ("a", "y")])
        con.register("pivoted", pd.DataFrame([[1, 2], [3, 4]], columns=levels))
        assert table("pivoted").columns(con) == ["('a', 'x')", "('a', 'y')"]
        assert rows(con, "SELECT sum(\"('a', 'y')\") FROM pivoted") == [(6,)]

    def test_repeated_labels_are_renamed_as_the_engine_renames_arrow_fields(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame([(1, 2, 3, 4)], columns=["a_1", "a", "a", "a_2"])
        con.register("df", frame)
        con.register(
            "arrow", pa.table([pa.array([1]), pa.array([2]), pa.array([3]), pa.array([4])], names=list(frame.columns))
        )
        assert table("df").columns(con) == table("arrow").columns(con) == ["a_1", "a", "a_2", "a_2_1"]
        assert rows(con, "SELECT a_2, a_2_1 FROM df") == [(3, 4)]
        con.register("cased", pd.DataFrame([(1, 2)], columns=["A", "a"]))
        assert table("cased").columns(con) == ["A", "a_1"]

    def test_a_frame_is_read_as_it_is_at_query_time(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        con.register("df", frame)
        assert rows(con, "SELECT sum(a) FROM df") == [(6,)]
        frame.loc[0, "a"] = 100
        assert rows(con, "SELECT sum(a) FROM df") == [(105,)]
        frame.columns = ["z", "b"]
        assert table("df").columns(con) == ["z", "b"]
        assert rows(con, "SELECT sum(z) FROM df") == [(105,)]
        frame.drop(columns=["b"], inplace=True)
        frame["c"] = [7, 8, 9]
        assert table("df").columns(con) == ["z", "c"]
        assert rows(con, "SELECT sum(c) FROM df") == [(24,)]

    def test_a_multi_level_row_index_is_left_out(self, con: duckdb.frame.Connection) -> None:
        index = pd.MultiIndex.from_tuples([("x", 1), ("y", 2)], names=["letter", "number"])
        con.register("df", pd.DataFrame({"v": [10, 20]}, index=index))
        assert table("df").columns(con) == ["v"]
        assert rows(con, "SELECT v FROM df ORDER BY v") == [(10,), (20,)]

    def test_every_missing_marker_is_null(self, con: duckdb.frame.Connection) -> None:
        moment = datetime.datetime(2024, 1, 2, 3, 4, 5)
        frame = pd.DataFrame(
            {
                "moment": pd.Series([moment, None, pd.NaT, pd.NA], dtype=object),
                "count": pd.Series([1, None, pd.NA, 4], dtype="Int64"),
                "flag": pd.Series([True, None, pd.NA, False], dtype="boolean"),
            }
        )
        con.register("df", frame)
        assert table("df").schema(con) == [("moment", "TIMESTAMP"), ("count", "BIGINT"), ("flag", "BOOLEAN")]
        assert rows(con, 'SELECT count(moment), count("count"), count(flag), count(*) FROM df') == [(1, 2, 2, 4)]

    def test_an_object_column_scans_correctly_under_four_threads(self, con: duckdb.frame.Connection) -> None:
        n = 300_000
        frame = pd.DataFrame({"i": range(n), "o": [str(i) for i in range(n)]})
        con.run("SET threads = 4")
        con.register("df", frame)
        assert rows(con, "SELECT count(*), sum(i) FROM df") == [(n, sum(range(n)))]
        assert rows(con, "SELECT i, o FROM df WHERE i IN (0, 150000, 299999) ORDER BY i") == [
            (0, "0"),
            (150000, "150000"),
            (299999, "299999"),
        ]


def register_both(
    con: duckdb.frame.Connection, frame: pd.DataFrame, *, native: str = "native", arrow: str = "arrow"
) -> None:
    """`frame` registered twice: once served by the native scan, once forced through the Arrow path."""
    native_source = PandasSource(frame)
    assert native_source.native, "the frame is not native-eligible"
    arrow_source = PandasSource(frame)
    arrow_source.native = False
    con.register(native, native_source)
    con.register(arrow, arrow_source)


def both(
    con: duckdb.frame.Connection, frame: pd.DataFrame, query: str
) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]]]:
    """`query`, with `{}` standing for the table name, run against `frame` on both paths."""
    register_both(con, frame)
    return rows(con, query.format("native")), rows(con, query.format("arrow"))


class TestPandasNativeClassification:
    def test_a_plain_frame_is_native(self) -> None:
        frame = pd.DataFrame({"i": range(3), "f": [1.0, 2.0, 3.0], "o": pd.Series(["a", "b", "c"], dtype=object)})
        assert PandasSource(frame).native

    def test_a_pyarrow_backed_string_column_is_not_native(self) -> None:
        frame = pd.DataFrame({"i": range(3), "s": pd.array(["a", "b", "c"], dtype="string[pyarrow]")})
        source = PandasSource(frame)
        assert not source.native

    def test_a_pyarrow_backed_column_sends_the_whole_frame_through_the_arrow_path(
        self, con: duckdb.frame.Connection
    ) -> None:
        frame = pd.DataFrame({"i": range(3), "s": pd.array(["a", "b", "c"], dtype="string[pyarrow]")})
        con.register("t", frame)
        assert rows(con, "SELECT i, s FROM t ORDER BY i") == [(0, "a"), (1, "b"), (2, "c")]
        assert "Python Arrow Scan" in table("t").on(con).explain()


_KIND_PLAN_CASES = {
    "int64": pd.Series([1, 2, 3], dtype="int64"),
    "float16": pd.Series([1.5, 2.5, 3.5], dtype="float16"),
    "float64": pd.Series([1.5, 2.5, 3.5], dtype="float64"),
    "bool": pd.Series([True, False, True]),
    "nullable Int64": pd.Series([1, None, 3], dtype="Int64"),
    "nullable Float64": pd.Series([1.5, None, 3.5], dtype="Float64"),
    "nullable boolean": pd.Series([True, None, False], dtype="boolean"),
    "datetime64[ns]": pd.Series(pd.to_datetime(["2020-01-01", "2020-01-02"])),
    "datetime64[us, UTC]": pd.Series(
        pd.to_datetime(["2020-01-01", "2020-01-02"]).tz_localize("UTC").astype("datetime64[us, UTC]")
    ),
    "timedelta64[ms]": pd.Series(pd.array([1000, 2000], dtype="timedelta64[ms]")),
    "string categorical": pd.Series(pd.Categorical(["a", "b", "a"])),
    "integer categorical": pd.Series(pd.Categorical([1, 2, 1])),
    "object ints": pd.Series([1, 2, 3], dtype=object),
    "object strings": pd.Series(["a", "b", "c"], dtype=object),
    "object mixed": pd.Series([1, "a", 2.5], dtype=object),
}


class TestPandasKindAndPlanAgree:
    """`describe()`'s kind-only reading and `columns()`'s full plan must never disagree over a column's type."""

    @pytest.mark.parametrize("label", list(_KIND_PLAN_CASES))
    def test_kind_matches_the_plans_kind_and_type(self, label: str) -> None:
        from duckdb._sources import pandas as _sources_pandas

        series = _KIND_PLAN_CASES[label]
        assert _sources_pandas._column_kind(series) == _sources_pandas._column_plan(series)[:2]


class TestPandasNativeFixed:
    def test_a_masked_float_keeps_a_real_nan(self, con: duckdb.frame.Connection) -> None:
        """Only a plain float column marks missing values by NaN; under a mask, a NaN is a value."""
        data = np.array([1.0, np.nan, 3.0, 4.0])
        mask = np.array([False, False, False, True])
        frame = pd.DataFrame({"a": pd.arrays.FloatingArray(data, mask)})
        con.register("t", frame)
        got = rows(con, "SELECT a, a IS NULL FROM t")
        assert [row[1] for row in got] == [False, False, False, True]
        assert got[0][0] == 1.0
        assert got[1][0] != got[1][0]

    """`"fixed"` columns: numpy int/uint/float/bool and their nullable pandas counterparts."""

    @pytest.mark.parametrize(
        ("dtype", "type_text"),
        [
            ("int8", "TINYINT"),
            ("int16", "SMALLINT"),
            ("int32", "INTEGER"),
            ("int64", "BIGINT"),
            ("uint8", "UTINYINT"),
            ("uint16", "USMALLINT"),
            ("uint32", "UINTEGER"),
            ("uint64", "UBIGINT"),
            ("float32", "FLOAT"),
            ("float64", "DOUBLE"),
        ],
    )
    def test_every_plain_numeric_dtype_matches_the_arrow_path(
        self, con: duckdb.frame.Connection, dtype: str, type_text: str
    ) -> None:
        frame = pd.DataFrame({"a": pd.Series(range(10), dtype=dtype)})
        native_rows, arrow_rows = both(con, frame, "SELECT a, typeof(a) FROM {} ORDER BY a")
        assert native_rows == arrow_rows
        assert native_rows[0][1] == type_text

    def test_plain_bool_matches_the_arrow_path(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame({"a": [i % 2 == 0 for i in range(10)]})
        native_rows, arrow_rows = both(con, frame, "SELECT a, typeof(a) FROM {}")
        assert native_rows == arrow_rows
        assert native_rows[0][1] == "BOOLEAN"

    @pytest.mark.parametrize(
        "dtype", ["Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64", "Float32", "Float64"]
    )
    def test_every_nullable_numeric_dtype_matches_the_arrow_path(
        self, con: duckdb.frame.Connection, dtype: str
    ) -> None:
        frame = pd.DataFrame({"a": pd.array([*range(9), None], dtype=dtype)})
        native_rows, arrow_rows = both(con, frame, "SELECT a FROM {}")
        assert native_rows == arrow_rows
        assert native_rows[-1] == (None,)

    def test_nullable_boolean_matches_the_arrow_path(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame({"a": pd.array([True, None, pd.NA, np.nan, True], dtype="boolean")})
        native_rows, arrow_rows = both(con, frame, "SELECT a FROM {}")
        assert native_rows == [(True,), (None,), (None,), (None,), (True,)]
        assert native_rows == arrow_rows

    def test_nullable_narrow_dtypes_use_the_mask_not_a_nan_sentinel(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame({"i": pd.array([1, None, 3], dtype="Int8"), "u": pd.array([1, None, 3], dtype="UInt8")})
        con.register("t", frame)
        assert rows(con, "SELECT i, u FROM t") == [(1, 1), (None, None), (3, 3)]

    def test_columns_of_a_shared_2d_block_stay_contiguous(self, con: duckdb.frame.Connection) -> None:
        block = np.arange(12, dtype=np.int64).reshape(4, 3)
        frame = pd.DataFrame(block, columns=["a", "b", "c"])
        native_rows, arrow_rows = both(con, frame[["a", "c"]], "SELECT a, c FROM {} ORDER BY a")
        assert native_rows == arrow_rows == [(0, 2), (3, 5), (6, 8), (9, 11)]

    def test_float16_widens_to_float32_once(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame({"a": pd.Series([1.5, 2.5, np.nan], dtype=np.float16)})
        con.register("t", frame)
        assert rows(con, "SELECT a, typeof(a) FROM t") == [(1.5, "FLOAT"), (2.5, "FLOAT"), (None, "FLOAT")]

    def test_float64_nan_and_infinities(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame({"a": [1.0, float("nan"), float("inf"), float("-inf")]})
        native_rows, arrow_rows = both(con, frame, "SELECT a FROM {}")
        assert native_rows == [(1.0,), (None,), (float("inf"),), (float("-inf"),)]
        assert without_nan(native_rows) == without_nan(arrow_rows)


class TestPandasNativeTemporal:
    """`"timestamp"` and `"interval"` columns: datetime64 and timedelta64, naive and zoned."""

    @pytest.mark.parametrize(
        ("unit", "type_text"),
        [("s", "TIMESTAMP_S"), ("ms", "TIMESTAMP_MS"), ("us", "TIMESTAMP"), ("ns", "TIMESTAMP_NS")],
    )
    def test_naive_datetime64_units_match_the_arrow_path(
        self, con: duckdb.frame.Connection, unit: str, type_text: str
    ) -> None:
        frame = pd.DataFrame({"t": pd.to_datetime(["2020-01-02 03:04:05", None]).astype(f"datetime64[{unit}]")})
        native_rows, arrow_rows = both(con, frame, "SELECT t, typeof(t) FROM {}")
        assert native_rows == arrow_rows
        assert native_rows[0] == (datetime.datetime(2020, 1, 2, 3, 4, 5), type_text)
        assert native_rows[1] == (None, type_text)

    @pytest.mark.parametrize("unit", ["s", "ms", "us"])
    def test_utc_aware_datetime64_matches_the_arrow_path(self, con: duckdb.frame.Connection, unit: str) -> None:
        frame = pd.DataFrame(
            {"t": pd.to_datetime(["2020-01-02 03:04:05"]).tz_localize("UTC").astype(f"datetime64[{unit}, UTC]")}
        )
        native_rows, arrow_rows = both(con, frame, "SELECT t, typeof(t) FROM {}")
        assert native_rows == arrow_rows
        assert native_rows == [
            (datetime.datetime(2020, 1, 2, 3, 4, 5, tzinfo=datetime.UTC), "TIMESTAMP WITH TIME ZONE")
        ]

    def test_a_nanosecond_aware_column_still_reads_as_timestamp_with_time_zone(
        self, con: duckdb.frame.Connection
    ) -> None:
        # The engine also has a nanosecond TIMESTAMP_TZ_NS, which the Arrow path picks for this dtype; the native
        # scan always normalizes an aware column to microseconds, so the two paths deliberately part ways here.
        frame = pd.DataFrame(
            {"t": pd.to_datetime(["2020-01-02 03:04:05"]).tz_localize("UTC").astype("datetime64[ns, UTC]")}
        )
        con.register("t", frame)
        assert rows(con, "SELECT t, typeof(t) FROM t") == [
            (datetime.datetime(2020, 1, 2, 3, 4, 5, tzinfo=datetime.UTC), "TIMESTAMP WITH TIME ZONE")
        ]

    @pytest.mark.parametrize("zone", ["Europe/Berlin", "Asia/Kathmandu"])
    def test_a_non_utc_zone_keeps_its_instant(self, con: duckdb.frame.Connection, zone: str) -> None:
        frame = pd.DataFrame({"t": pd.to_datetime(["2020-06-02 03:04:05"]).tz_localize(zone)})
        native_rows, arrow_rows = both(con, frame, "SELECT t FROM {}")
        assert native_rows == arrow_rows
        utc = pd.to_datetime(["2020-06-02 03:04:05"]).tz_localize(zone).tz_convert("UTC")[0].to_pydatetime()
        assert native_rows == [(utc,)]

    @pytest.mark.parametrize("year", [1680, 2260])
    def test_years_far_from_the_epoch_at_microsecond_resolution(self, con: duckdb.frame.Connection, year: int) -> None:
        frame = pd.DataFrame({"t": pd.to_datetime([f"{year}-01-02 03:04:05.123456"])})
        native_rows, arrow_rows = both(con, frame, "SELECT t FROM {}")
        assert native_rows == arrow_rows
        assert native_rows == [(datetime.datetime(year, 1, 2, 3, 4, 5, 123456),)]

    def test_a_strided_datetime_series_reads_correctly(self, con: duckdb.frame.Connection) -> None:
        strided = pd.date_range("2020-01-01", periods=100, freq="h")[::23]
        frame = pd.DataFrame({"t": pd.Series(strided)})
        native_rows, arrow_rows = both(con, frame, "SELECT t FROM {}")
        assert native_rows == arrow_rows
        assert [row[0] for row in native_rows] == list(strided.to_pydatetime())

    def test_an_object_column_of_fixed_offset_datetimes(self, con: duckdb.frame.Connection) -> None:
        early = datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone(datetime.timedelta(hours=-19)))
        late = datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone(datetime.timedelta(hours=14)))
        frame = pd.DataFrame({"t": pd.Series([early, late], dtype=object)})
        con.register("t", frame)
        assert rows(con, "SELECT t, typeof(t) FROM t") == [
            (early.astimezone(datetime.UTC), "TIMESTAMP WITH TIME ZONE"),
            (late.astimezone(datetime.UTC), "TIMESTAMP WITH TIME ZONE"),
        ]

    @pytest.mark.parametrize(("unit", "value"), [("s", 2), ("ms", 2000), ("us", 2_000_000), ("ns", 2_000_000_000)])
    def test_timedelta64_units_read_as_interval(self, con: duckdb.frame.Connection, unit: str, value: int) -> None:
        frame = pd.DataFrame({"d": pd.array([value, None], dtype=f"timedelta64[{unit}]")})
        con.register("t", frame)
        assert rows(con, "SELECT d, typeof(d) FROM t") == [
            (datetime.timedelta(seconds=2), "INTERVAL"),
            (None, "INTERVAL"),
        ]

    def test_an_object_column_of_large_and_negative_timedeltas(self, con: duckdb.frame.Connection) -> None:
        big = datetime.timedelta(days=9999, hours=24, minutes=60, seconds=60, milliseconds=999, microseconds=999999)
        small = -datetime.timedelta(days=1, seconds=1)
        frame = pd.DataFrame({"d": pd.Series([big, small], dtype=object)})
        con.register("t", frame)
        assert rows(con, "SELECT d FROM t") == [(big,), (small,)]

    def test_a_microsecond_magnitude_near_the_int64_edge(self, con: duckdb.frame.Connection) -> None:
        huge = datetime.timedelta(microseconds=9_150_000_000_000_000)
        frame = pd.DataFrame({"d": pd.Series([huge], dtype=object)})
        con.register("t", frame)
        assert rows(con, "SELECT d FROM t") == [(huge,)]


class TestPandasNativeCategorical:
    """`"enum"` columns: pandas Categorical, string and non-string."""

    def test_string_categories_with_and_without_null(self, con: duckdb.frame.Connection) -> None:
        # The native scan reads a categorical as ENUM; the Arrow path (pyarrow dictionary) reads it as VARCHAR
        # already, so only the values, not typeof(), are compared here.
        frame = pd.DataFrame({"c": pd.Categorical(["x", "y", None, "x"])})
        native_rows, arrow_rows = both(con, frame, "SELECT c FROM {}")
        assert native_rows == arrow_rows == [("x",), ("y",), (None,), ("x",)]
        con.register("t", frame)
        assert table("t").schema(con) == [("c", "ENUM('x', 'y')")]

    def test_a_declared_but_unused_category_of_another_type_does_not_change_the_type(
        self, con: duckdb.frame.Connection
    ) -> None:
        """Filtering keeps a category declared; the values, not the declaration, type the column."""
        mixed = pd.Series(pd.Categorical([1, 2, 1, datetime.date(2020, 1, 1)]))
        filtered = mixed[mixed != datetime.date(2020, 1, 1)].reset_index(drop=True)
        assert len(filtered.cat.categories) == 3
        frame = pd.DataFrame({"c": filtered})
        con.register("t", frame)
        assert table("t").schema(con) == [("c", "BIGINT")]
        assert rows(con, "SELECT c FROM t") == [(1,), (2,), (1,)]

    def test_integer_categories_are_read_through_their_own_kind(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame({"c": pd.Categorical([1, 2, 1])})
        native_rows, arrow_rows = both(con, frame, "SELECT c, typeof(c) FROM {}")
        assert native_rows == arrow_rows == [(1, "BIGINT"), (2, "BIGINT"), (1, "BIGINT")]

    def test_integer_categories_with_a_null(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame({"c": pd.Categorical([1, 2, None])})
        con.register("t", frame)
        assert rows(con, "SELECT c FROM t") == [(1,), (2,), (None,)]

    @pytest.mark.parametrize("count", [10, 160, 300, 70_000])
    def test_category_counts_spanning_every_code_width(self, con: duckdb.frame.Connection, count: int) -> None:
        categories = [f"c{i}" for i in range(count)]
        frame = pd.DataFrame({"c": pd.Categorical([categories[0], categories[-1], None], categories=categories)})
        con.register("t", frame)
        assert rows(con, "SELECT c FROM t") == [(categories[0],), (categories[-1],), (None,)]

    def test_an_empty_categorical(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame({"c": pd.Categorical([], categories=["a", "b"])})
        con.register("t", frame)
        assert table("t").schema(con) == [("c", "ENUM('a', 'b')")]
        assert rows(con, "SELECT c FROM t") == []

    def test_a_categorical_beside_a_float_column_with_nulls(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame({"c": pd.Categorical(["x", None, "y"]), "f": [1.0, None, 3.0]})
        con.register("t", frame)
        assert rows(con, "SELECT c, f FROM t") == [("x", 1.0), (None, None), ("y", 3.0)]

    def test_a_categorical_over_several_batches(self, con: duckdb.frame.Connection) -> None:
        n = 4096
        categories = [f"v{i % 5}" for i in range(n)]
        frame = pd.DataFrame({"c": pd.Categorical(categories)})
        native_rows, arrow_rows = both(con, frame, "SELECT c FROM {}")
        assert native_rows == arrow_rows == [(c,) for c in categories]

    def test_category_order_is_preserved_in_the_enum(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame({"c": pd.Categorical(["b", "a"], categories=["z", "a", "b"])})
        con.register("t", frame)
        assert table("t").schema(con) == [("c", "ENUM('z', 'a', 'b')")]


class TestPandasNativeStrings:
    """`"text"` columns: Python-backed string arrays and plain object columns of strings."""

    def test_object_strings_and_python_backed_strings_agree(self, con: duckdb.frame.Connection) -> None:
        values = ["a", "b", None]
        object_frame = pd.DataFrame({"s": pd.Series(values, dtype=object)})
        string_frame = pd.DataFrame({"s": pd.Series(values, dtype=pd.StringDtype("python"))})
        object_rows, arrow_rows = both(con, object_frame, "SELECT s FROM {}")
        con.register("string_backed", string_frame)
        assert object_rows == arrow_rows == rows(con, "SELECT s FROM string_backed") == [("a",), ("b",), (None,)]

    def test_three_million_rows_of_repeating_names(self, con: duckdb.frame.Connection) -> None:
        n = 3_000_000
        cities = ["Amsterdam", "Utrecht", "Haarlem"]
        frame = pd.DataFrame({"city": pd.Series([cities[i % 3] for i in range(n)], dtype=object)})
        con.register("t", frame)
        assert rows(con, "SELECT count(*) FROM t") == [(n,)]

    def test_unicode_text_and_a_self_join_under_four_threads(self, con: duckdb.frame.Connection) -> None:
        con.run("SET threads = 4")
        words = ["mühleisen", "鴨", "數據庫", "🤦🏼‍♂️ L🤦🏼‍♂️R 🤦🏼‍♂️", "🦆🍞🦆", "п"] * 2000
        frame = pd.DataFrame({"i": range(len(words)), "s": pd.Series(words, dtype=object)})
        con.register("t", frame)
        assert rows(con, "SELECT count(*) FROM t a JOIN t b USING (i)") == [(len(words),)]
        assert rows(con, "SELECT length('ë')") == [(1,)]

    def test_an_empty_string_is_not_null(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame({"s": pd.Series(["", "a", None], dtype=object)})
        con.register("t", frame)
        assert rows(con, "SELECT s, s IS NULL FROM t") == [("", False), ("a", False), (None, True)]

    def test_nat_and_na_in_a_string_backed_column_are_null(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame({"s": pd.Series(["a", pd.NaT, pd.NA], dtype=pd.StringDtype("python"))})
        con.register("t", frame)
        assert rows(con, "SELECT s FROM t") == [("a",), (None,), (None,)]

    def test_a_single_bytes_value_reads_as_blob(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame({"b": pd.Series([b"\xc3\x83"], dtype=object)})
        con.register("t", frame)
        assert table("t").schema(con) == [("b", "BLOB")]
        assert rows(con, "SELECT b FROM t") == [(b"\xc3\x83",)]

    def test_an_all_none_column_beside_an_int_column(self, con: duckdb.frame.Connection) -> None:
        n = 2001
        frame = pd.DataFrame({"i": range(n), "o": pd.Series([None] * n, dtype=object)})
        con.register("t", frame)
        assert table("t").schema(con) == [("i", "BIGINT"), ("o", "VARCHAR")]
        assert rows(con, "SELECT count(*), count(o) FROM t") == [(n, 0)]

    def test_a_sample_that_misses_the_only_value_still_finds_it(self, con: duckdb.frame.Connection) -> None:
        values: list[object] = [None] * 10_001
        values[1] = 5
        frame = pd.DataFrame({"a": pd.Series(values, dtype=object)})
        con.register("t", frame)
        assert table("t").schema(con) == [("a", "BIGINT")]
        assert rows(con, "SELECT a FROM t WHERE a IS NOT NULL") == [(5,)]

    def test_a_sample_that_misses_the_only_value_finds_it_by_position_under_a_repeated_label(
        self, con: duckdb.frame.Connection
    ) -> None:
        values: list[object] = [None] * 2001
        values[1999] = 5
        labels = list(range(2001))
        labels[0] = labels[1999]
        frame = pd.DataFrame({"a": pd.Series(values, index=labels, dtype=object)})
        con.register("t", frame)
        assert table("t").schema(con) == [("a", "BIGINT")]
        assert rows(con, "SELECT a FROM t WHERE a IS NOT NULL") == [(5,)]

    def test_an_object_array_whose_sample_misses_the_only_value_still_finds_it(self) -> None:
        values: list[object] = [None, float("nan"), pd.NA, pd.NaT] * 600
        values.append(5)
        assert _object_kind(np.array(values, dtype=object)) == ("objects", "BIGINT")

    def test_a_list_outside_the_sample_fails_the_query_naming_the_row(self, con: duckdb.frame.Connection) -> None:
        values: list[object] = list(range(10_000))
        values[5] = [1, 2, 3]
        frame = pd.DataFrame({"a": pd.Series(values, dtype=object)})
        con.register("t", frame)
        assert table("t").schema(con) == [("a", "BIGINT")]
        with pytest.raises(exceptions.InvalidInputError, match=r"column 'a' holds a value at row 5"):
            rows(con, "SELECT count(*) FROM t")

    @pytest.mark.parametrize("values", [[18446744073709551615, 0], [2**64, 0]])
    def test_oversized_ints_read_as_hugeint(self, con: duckdb.frame.Connection, values: list[int]) -> None:
        frame = pd.DataFrame({"a": pd.Series(values, dtype=object)})
        con.register("t", frame)
        assert table("t").schema(con) == [("a", "HUGEINT")]
        assert rows(con, "SELECT a FROM t ORDER BY a") == [(0,), (values[0],)]

    def test_an_int_beyond_hugeint_reads_as_text(self, con: duckdb.frame.Connection) -> None:
        huge = 2**10000
        frame = pd.DataFrame({"a": pd.Series([huge, 0], dtype=object)})
        con.register("t", frame)
        assert table("t").schema(con) == [("a", "VARCHAR")]
        assert rows(con, "SELECT a FROM t ORDER BY length(a)") == [("0",), (str(huge),)]

    def test_decimals_of_varied_scale_combine_into_one_decimal_type(self, con: duckdb.frame.Connection) -> None:
        values = [decimal.Decimal("1.5"), decimal.Decimal("2.125"), None]
        frame = pd.DataFrame({"a": pd.Series(values, dtype=object)})
        con.register("t", frame)
        assert table("t").schema(con) == [("a", "DECIMAL(38,3)")]
        assert rows(con, "SELECT a FROM t") == [(decimal.Decimal("1.500"),), (decimal.Decimal("2.125"),), (None,)]

    @pytest.mark.parametrize("bad", [decimal.Decimal("NaN"), decimal.Decimal("Infinity")])
    def test_a_non_finite_decimal_is_refused(self, con: duckdb.frame.Connection, bad: decimal.Decimal) -> None:
        frame = pd.DataFrame({"a": pd.Series([decimal.Decimal("1.5"), bad], dtype=object)})
        con.register("t", frame)
        with pytest.raises(exceptions.InvalidInputError, match="column 'a' holds a value at row 1"):
            rows(con, "SELECT a FROM t")

    def test_a_decimal_beyond_38_digits_is_refused(self, con: duckdb.frame.Connection) -> None:
        oversized = decimal.Decimal("1" * 40)
        frame = pd.DataFrame({"a": pd.Series([decimal.Decimal("1.5"), oversized], dtype=object)})
        con.register("t", frame)
        with pytest.raises(exceptions.InvalidInputError, match="column 'a' holds a value at row 1"):
            rows(con, "SELECT a FROM t")

    def test_a_column_of_dates_reads_as_date_on_both_paths(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame(
            {"a": pd.Series([datetime.date(2024, 1, 1), None, datetime.date(1999, 12, 31)], dtype=object)}
        )
        expected = [(datetime.date(2024, 1, 1),), (None,), (datetime.date(1999, 12, 31),)]
        con.register("t", frame)
        assert table("t").schema(con) == [("a", "DATE")]
        assert rows(con, "SELECT a FROM t") == expected
        source = PandasSource(frame)
        source.native = False
        con.register("t", source)
        assert table("t").schema(con) == [("a", "DATE")]
        assert rows(con, "SELECT a FROM t") == expected

    def test_date_mixed_with_datetime_reads_as_timestamp(self, con: duckdb.frame.Connection) -> None:
        values = [datetime.date(2024, 1, 1), datetime.datetime(2024, 1, 2, 3, 4, 5)]
        frame = pd.DataFrame({"a": pd.Series(values, dtype=object)})
        con.register("t", frame)
        assert table("t").schema(con) == [("a", "TIMESTAMP")]
        assert rows(con, "SELECT a FROM t") == [
            (datetime.datetime(2024, 1, 1, 0, 0),),
            (datetime.datetime(2024, 1, 2, 3, 4, 5),),
        ]

    def test_date_mixed_with_a_string_reads_as_text(self, con: duckdb.frame.Connection) -> None:
        values: list[object] = [datetime.date(2024, 1, 1), "not a date"]
        frame = pd.DataFrame({"a": pd.Series(values, dtype=object)})
        con.register("t", frame)
        assert table("t").schema(con) == [("a", "VARCHAR")]
        assert rows(con, "SELECT a FROM t") == [("2024-01-01",), ("not a date",)]

    def test_numpy_scalars_read_as_their_python_equivalents(self, con: duckdb.frame.Connection) -> None:
        values = [np.int64(1), np.int64(2)]
        frame = pd.DataFrame({"a": pd.Series(values, dtype=object)})
        con.register("t", frame)
        assert table("t").schema(con) == [("a", "BIGINT")]
        assert rows(con, "SELECT a FROM t") == [(1,), (2,)]

    def test_a_numpy_float_and_bool_scalar_column(self, con: duckdb.frame.Connection) -> None:
        floats = pd.DataFrame({"a": pd.Series([np.float64(1.5), np.float64(2.5)], dtype=object)})
        con.register("f", floats)
        assert table("f").schema(con) == [("a", "DOUBLE")]
        bools = pd.DataFrame({"a": pd.Series([np.bool_(True), np.bool_(False)], dtype=object)})
        con.register("b", bools)
        assert table("b").schema(con) == [("a", "BOOLEAN")]
        assert rows(con, "SELECT a FROM b") == [(True,), (False,)]

    def test_dicts_lists_and_a_mix_read_as_text(self, con: duckdb.frame.Connection) -> None:
        dicts = pd.DataFrame({"a": pd.Series([{"x": 1}, {"y": 2}], dtype=object)})
        con.register("d", dicts)
        assert table("d").schema(con) == [("a", "VARCHAR")]
        assert rows(con, "SELECT a FROM d") == [(str({"x": 1}),), (str({"y": 2}),)]

        lists = pd.DataFrame({"a": pd.Series([[1, 2], [3]], dtype=object)})
        con.register("l", lists)
        assert table("l").schema(con) == [("a", "VARCHAR")]
        assert rows(con, "SELECT a FROM l") == [(str([1, 2]),), (str([3]),)]

        mixed = pd.DataFrame({"a": pd.Series([{"x": 1}, [1, 2], {"y": [1]}], dtype=object)})
        con.register("m", mixed)
        assert table("m").schema(con) == [("a", "VARCHAR")]
        assert rows(con, "SELECT a FROM m") == [(str({"x": 1}),), (str([1, 2]),), (str({"y": [1]}),)]

    def test_a_uuid_object_column(self, con: duckdb.frame.Connection) -> None:
        identity = uuid.uuid4()
        frame = pd.DataFrame({"u": pd.Series([identity, None], dtype=object)})
        con.register("t", frame)
        assert table("t").schema(con) == [("u", "UUID")]
        assert rows(con, "SELECT u FROM t") == [(identity,), (None,)]

    def test_a_sliced_frame_reads_positionally(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame({"a": range(5)}).iloc[1:]
        con.register("t", frame)
        assert rows(con, "SELECT a FROM t") == [(1,), (2,), (3,), (4,)]


class TestPandasTextUnification:
    """The Arrow path stringifies a mixed object column the same way the native scan reads it as text."""

    def test_consistent_dicts_and_lists_of_ints_read_as_text_on_the_arrow_path(
        self, con: duckdb.frame.Connection
    ) -> None:
        dicts = pd.DataFrame({"a": pd.Series([{"x": 1}, {"x": 2}], dtype=object)})
        dicts_source = PandasSource(dicts)
        dicts_source.native = False
        con.register("d", dicts_source)
        assert table("d").schema(con) == [("a", "VARCHAR")]
        assert rows(con, "SELECT a FROM d") == [(str({"x": 1}),), (str({"x": 2}),)]

        lists = pd.DataFrame({"a": pd.Series([[1, 2], [3, 4]], dtype=object)})
        lists_source = PandasSource(lists)
        lists_source.native = False
        con.register("l", lists_source)
        assert table("l").schema(con) == [("a", "VARCHAR")]
        assert rows(con, "SELECT a FROM l") == [(str([1, 2]),), (str([3, 4]),)]

    def test_bytes_mixed_with_str_reads_as_text_on_both_paths(self, con: duckdb.frame.Connection) -> None:
        values: list[object] = [b"abc", "text"]
        frame = pd.DataFrame({"a": pd.Series(values, dtype=object)})
        native_rows, arrow_rows = both(con, frame, "SELECT a FROM {}")
        assert native_rows == arrow_rows == [(str(v),) for v in values]


class MisreportingColumns(PandasSource):
    """A native source whose columns() reply can be broken in a chosen way, to exercise the scan's own checks."""

    def __init__(self, obj: object, break_as: str) -> None:
        super().__init__(obj)
        self.break_as = break_as

    def columns(self, columns: Sequence[int] | None) -> list[tuple[str, str, str, object, object | None]]:
        answer = super().columns(columns)
        if self.break_as == "short":
            return answer[:-1]
        name, kind, kind_text, data, mask = answer[0]
        assert isinstance(data, np.ndarray)
        if self.break_as == "renamed":
            return [("not_" + name, kind, kind_text, data, mask), *answer[1:]]
        if self.break_as == "strided":
            return [(name, kind, kind_text, data[::2], mask), *answer[1:]]
        if self.break_as == "wrong_width":
            return [(name, kind, kind_text, data.astype(np.int8), mask), *answer[1:]]
        if self.break_as == "short_mask":
            return [(name, kind, kind_text, data, np.zeros(len(data) - 1, dtype=bool)), *answer[1:]]
        if self.break_as == "swapped":
            return [(name, kind, kind_text, data.astype(data.dtype.newbyteorder()), mask), *answer[1:]]
        return answer


class Counted(PandasSource):
    """A native source whose rows() answers None, which the scan cannot work without."""

    def rows(self) -> int | None:
        return None


class TestPandasNativeValidation:
    """The scan refuses a malformed columns() answer instead of misreading it, naming the offending column."""

    def test_a_short_answer_is_refused(self, con: duckdb.frame.Connection) -> None:
        con.register("t", MisreportingColumns(pd.DataFrame({"a": range(3), "b": range(3)}), "short"))
        with pytest.raises(exceptions.InvalidInputError, match=r"1 columns where 2 were requested"):
            rows(con, "SELECT * FROM t")

    def test_a_renamed_column_is_refused(self, con: duckdb.frame.Connection) -> None:
        con.register("t", MisreportingColumns(pd.DataFrame({"a": range(3)}), "renamed"))
        with pytest.raises(exceptions.InvalidInputError, match="column 'not_a' at position 0 where 'a' was expected"):
            rows(con, "SELECT * FROM t")

    def test_a_non_contiguous_buffer_is_refused(self, con: duckdb.frame.Connection) -> None:
        con.register("t", MisreportingColumns(pd.DataFrame({"a": range(10)}), "strided"))
        with pytest.raises(exceptions.InvalidInputError, match="answered columns\\(\\) for 'a'"):
            rows(con, "SELECT * FROM t")

    def test_a_mask_of_the_wrong_length_is_refused_and_the_data_buffer_released(
        self, con: duckdb.frame.Connection
    ) -> None:
        frame = pd.DataFrame({"a": range(10)})
        data = frame["a"].to_numpy(copy=False)
        before = sys.getrefcount(data)
        con.register("t", MisreportingColumns(frame, "short_mask"))
        for _ in range(3):
            with pytest.raises(exceptions.InvalidInputError, match="mask array of 9 rows where 10 were expected"):
                rows(con, "SELECT * FROM t")
        assert sys.getrefcount(data) == before

    def test_a_buffer_in_the_other_byte_order_is_refused(self, con: duckdb.frame.Connection) -> None:
        con.register("t", MisreportingColumns(pd.DataFrame({"a": range(10)}), "swapped"))
        with pytest.raises(exceptions.InvalidInputError, match="in the other byte order"):
            rows(con, "SELECT * FROM t")

    def test_a_frame_stored_in_the_other_byte_order_is_refused(self, con: duckdb.frame.Connection) -> None:
        con.register("t", pd.DataFrame({"a": np.array([1, 2, 3], dtype=">i4")}))
        with pytest.raises(exceptions.InvalidInputError, match="in the other byte order"):
            rows(con, "SELECT * FROM t")

    def test_a_source_without_a_row_count_is_refused(self, con: duckdb.frame.Connection) -> None:
        con.register("t", Counted(pd.DataFrame({"a": range(3)})))
        with pytest.raises(exceptions.InvalidInputError, match=r"something other than .* a row count"):
            rows(con, "SELECT * FROM t")

    def test_a_mismatched_element_width_is_refused(self, con: duckdb.frame.Connection) -> None:
        con.register("t", MisreportingColumns(pd.DataFrame({"a": range(10)}), "wrong_width"))
        with pytest.raises(exceptions.InvalidInputError, match="answered columns\\(\\) for 'a'"):
            rows(con, "SELECT * FROM t")


class TestPandasNativeShapes:
    def test_duplicate_labels_are_renamed(self, con: duckdb.frame.Connection) -> None:
        original = ["a_1", "a", "a"]
        frame = pd.DataFrame([[1, 2, 3]], columns=original)
        con.register("t", frame)
        assert table("t").columns(con) == ["a_1", "a", "a_2"]
        assert list(frame.columns) == original

    @pytest.mark.parametrize(
        "columns",
        [[1, 2], [1.5, 2.5], [("a", "x"), ("a", "y")]],
    )
    def test_non_string_column_labels_read_under_their_text(
        self, con: duckdb.frame.Connection, columns: list[object]
    ) -> None:
        frame = pd.DataFrame([[1, 2]], columns=columns)
        con.register("t", frame)
        assert table("t").columns(con) == [str(c) for c in columns]

    def test_a_multi_index_column_reads_under_its_text(self, con: duckdb.frame.Connection) -> None:
        levels = pd.MultiIndex.from_tuples([("a", "x"), ("a", "y")])
        frame = pd.DataFrame([[1, 2]], columns=levels)
        con.register("t", frame)
        assert table("t").columns(con) == ["('a', 'x')", "('a', 'y')"]

    def test_a_non_default_index_is_left_out(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame({"a": range(3)}, index=pd.Index([10, 20, 30], name="k"))
        con.register("t", frame)
        assert table("t").columns(con) == ["a"]
        assert rows(con, "SELECT a FROM t") == [(0,), (1,), (2,)]

    def test_a_multi_index_row_index_is_left_out(self, con: duckdb.frame.Connection) -> None:
        index = pd.MultiIndex.from_tuples([("x", 1), ("y", 2)], names=["letter", "number"])
        frame = pd.DataFrame({"v": [10, 20]}, index=index)
        con.register("t", frame)
        assert table("t").columns(con) == ["v"]
        assert rows(con, "SELECT v FROM t") == [(10,), (20,)]

    def test_zero_columns_is_refused_at_bind(self, con: duckdb.frame.Connection) -> None:
        con.register("t", pd.DataFrame())
        with pytest.raises(exceptions.InvalidInputError, match="did not declare any result columns"):
            rows(con, "SELECT * FROM t")

    def test_zero_rows_with_a_column_of_each_kind(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame(
            {
                "i": pd.array([], dtype="Int64"),
                "t": pd.array([], dtype="datetime64[us]"),
                "d": pd.array([], dtype="timedelta64[us]"),
                "c": pd.Categorical([], categories=["a"]),
                "o": pd.Series([], dtype=object),
            }
        )
        con.register("t", frame)
        assert rows(con, "SELECT * FROM t") == []

    def test_one_row(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame({"a": [1]})
        con.register("t", frame)
        assert rows(con, "SELECT a FROM t") == [(1,)]

    def test_a_wide_frame_of_mixed_kinds(self, con: duckdb.frame.Connection) -> None:
        columns: dict[str, object] = {}
        for i in range(100):
            columns[f"i{i}"] = pd.Series(range(5))
            columns[f"f{i}"] = pd.Series([float(x) for x in range(5)])
            columns[f"s{i}"] = pd.Series([str(x) for x in range(5)], dtype=object)
        frame = pd.DataFrame(columns)
        con.register("t", frame)
        assert rows(con, "SELECT count(*) FROM t") == [(5,)]
        assert rows(con, "SELECT i50, f50, s50 FROM t WHERE i0 = 2") == [(2, 2.0, "2")]


class Rendered:
    """An object whose text rendering records the thread that asked for it."""

    seen: ClassVar[set[int]] = set()

    def __str__(self) -> str:
        Rendered.seen.add(threading.get_ident())
        return "x"


class TestPandasNativeParallel:
    def test_several_engine_threads_read_one_frame(self, con: duckdb.frame.Connection) -> None:
        """A column of objects reads as text through str(), so the rendering sees which threads fill ranges."""
        con.run("SET threads = 4")
        Rendered.seen.clear()
        n = 6 * 50 * 2048
        con.register("t", pd.DataFrame({"o": pd.Series([Rendered()] * n, dtype=object)}))
        assert rows(con, "SELECT count(o), min(o) FROM t") == [(n, "x")]
        assert len(Rendered.seen) > 1

    def test_ten_million_rows_sum_matches_across_thread_counts(self, con: duckdb.frame.Connection) -> None:
        n = 10_000_000
        frame = pd.DataFrame({"i": range(n)})
        con.register("t", frame)
        con.run("SET threads = 1")
        one = rows(con, "SELECT sum(i) FROM t")
        con.run("SET threads = 4")
        four = rows(con, "SELECT sum(i) FROM t")
        assert one == four == [(sum(range(n)),)]

    def test_a_limit_after_a_disjunctive_filter_keeps_scan_order_under_eight_threads(
        self, con: duckdb.frame.Connection
    ) -> None:
        n = 10_000_000
        frame = pd.DataFrame({"i": range(n)})
        con.register("t", frame)
        con.run("SET threads = 8")
        assert rows(con, "SELECT i FROM t WHERE i = 334 OR i > 9967864 LIMIT 5") == [
            (334,),
            (9967865,),
            (9967866,),
            (9967867,),
            (9967868,),
        ]

    def test_order_preserved_over_a_million_row_frame(self, con: duckdb.frame.Connection) -> None:
        con.run("SET threads = 4")
        total = 1_000_000
        frame = pd.DataFrame({"n": range(total)})

        def register() -> None:
            con.register("src", frame)

        check_order_preserved(con, register, "src", total)

    def test_a_frame_narrower_than_one_range(self, con: duckdb.frame.Connection) -> None:
        con.run("SET threads = 4")
        frame = pd.DataFrame({"n": range(10)})
        con.register("src", frame)
        assert rows(con, "SELECT n FROM src") == [(i,) for i in range(10)]

    def test_a_frame_of_exactly_one_range_plus_one_row(self, con: duckdb.frame.Connection) -> None:
        con.run("SET threads = 4")
        vector_size = int(con._engine().get_option("standard_vector_size"))
        total = vector_size * 50 + 1
        frame = pd.DataFrame({"n": range(total)})
        con.register("src", frame)
        assert rows(con, "SELECT count(*), sum(n) FROM src") == [(total, sum(range(total)))]

    def test_projection_to_a_subset_and_to_a_single_column(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame({"a": range(5), "b": [str(i) for i in range(5)], "c": [float(i) for i in range(5)]})
        con.register("t", frame)
        assert rows(con, "SELECT c, a FROM t WHERE a = 2") == [(2.0, 2)]
        assert rows(con, "SELECT b FROM t WHERE a = 3") == [("3",)]

    def test_a_self_join_under_four_threads(self, con: duckdb.frame.Connection) -> None:
        con.run("SET threads = 4")
        n = 50_000
        frame = pd.DataFrame({"n": range(n)})
        con.register("t", frame)
        assert rows(con, "SELECT count(*), sum(a.n) FROM t a JOIN t b USING (n)") == [(n, sum(range(n)))]

    def test_a_query_interrupted_from_another_thread_stops_promptly(self, con: duckdb.frame.Connection) -> None:
        con.run("SET threads = 4")
        n = 8_000_000
        frame = pd.DataFrame({"s": pd.Series([str(i) for i in range(n)], dtype=object)})
        con.register("t", frame)
        timer = threading.Timer(0.02, con.interrupt)
        timer.start()
        start = time.time()
        try:
            with pytest.raises(exceptions.Error):
                rows(con, "SELECT sum(length(s)) FROM t")
            assert time.time() - start < 5
        finally:
            timer.cancel()


class TestPandasNativeLifetime:
    def test_a_mutated_frame_reads_the_new_state(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame({"a": [1, 2, 3]})
        con.register("t", frame)
        assert rows(con, "SELECT sum(a) FROM t") == [(6,)]
        frame.loc[0, "a"] = 100
        assert rows(con, "SELECT sum(a) FROM t") == [(105,)]

    def test_the_registry_holds_the_frame_after_it_is_dropped(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame({"a": [1, 2, 3]})
        con.register("t", frame)
        del frame
        gc.collect()
        assert rows(con, "SELECT sum(a) FROM t") == [(6,)]

    def test_an_unregistered_frame_does_not_crash_a_later_query(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame({"a": [1, 2, 3]})
        con.register("t", frame)
        con.unregister("t")
        gc.collect()
        assert rows(con, "SELECT 1") == [(1,)]

    def test_a_progress_bar_query_does_not_crash(self, con: duckdb.frame.Connection) -> None:
        con.run("SET enable_progress_bar = true")
        con.run("SET threads = 4")
        n = 1_000_000
        frame = pd.DataFrame({"n": range(n)})
        con.register("t", frame)
        assert rows(con, "SELECT count(*) FROM t a JOIN t b USING (n)") == [(n,)]


class TestPolarsShapes:
    def test_a_column_of_python_objects_is_refused(self, con: duckdb.frame.Connection) -> None:
        frame = pl.DataFrame({"n": [1, 2], "o": pl.Series([object(), object()], dtype=pl.Object)})
        with pytest.raises(TypeError, match="the polars column 'o' holds Python objects"):
            con.register("eager", frame)
        con.register("lazy", frame.lazy())
        with pytest.raises(exceptions.InvalidInputError, match="the polars column 'o' holds Python objects"):
            rows(con, "SELECT n FROM lazy")
        con.register("fine", frame.select("n"))
        assert rows(con, "SELECT sum(n) FROM fine") == [(3,)]

    def test_a_frame_just_over_one_slice_scans_whole(self, con: duckdb.frame.Connection) -> None:
        n = PolarsFrameSource.SLICE_ROWS + 1
        holes = [None if i in (0, n - 2, n - 1) else str(i) for i in range(n)]
        con.register("src", pl.DataFrame({"n": range(n), "s": holes}))
        assert rows(con, "SELECT count(*), sum(n), count(s) FROM src") == [(n, sum(range(n)), n - 3)]
        last = rows(con, "SELECT n, s FROM src ORDER BY n DESC LIMIT 3")
        assert last == [(n - 1, None), (n - 2, None), (n - 3, str(n - 3))]
        assert rows(con, "SELECT n FROM src WHERE n IN (0, 65535, 65536) ORDER BY n") == [(0,), (65535,), (65536,)]

    @pytest.mark.parametrize("lazy", [False, True])
    def test_an_empty_frame_gives_no_rows(self, con: duckdb.frame.Connection, *, lazy: bool) -> None:
        frame = pl.DataFrame({"n": pl.Series([], dtype=pl.Int64), "s": pl.Series([], dtype=pl.String)})
        con.register("empty", frame.lazy() if lazy else frame)
        assert rows(con, "SELECT * FROM empty") == []
        assert rows(con, "SELECT count(*), sum(n) FROM empty") == [(0, None)]

    def test_a_self_join_reads_a_chunked_frame_twice_at_once(self, con: duckdb.frame.Connection) -> None:
        con.run("SET threads = 4")
        parts = [pl.DataFrame({"n": range(start, start + 1000)}) for start in range(0, 5000, 1000)]
        frame = pl.concat(parts, rechunk=False)
        con.register("m", frame)
        assert rows(con, "SELECT count(*), sum(a.n) FROM m a JOIN m b USING (n)") == [(5000, sum(range(5000)))]
        assert frame.n_chunks() == 5

    def test_scanning_a_chunked_frame_leaves_its_chunk_count_unchanged(self, con: duckdb.frame.Connection) -> None:
        frame = pl.concat([pl.DataFrame({"n": [1, 2]}), pl.DataFrame({"n": [3, 4, 5]})], rechunk=False)
        assert frame.n_chunks() == 2
        con.register("src", frame)
        assert rows(con, "SELECT n FROM src") == [(1,), (2,), (3,), (4,), (5,)]
        assert frame.n_chunks() == 2

    @pytest.mark.parametrize("lazy", [False, True])
    def test_several_engine_threads_pull_from_a_polars_frame(
        self, con: duckdb.frame.Connection, monkeypatch: pytest.MonkeyPatch, *, lazy: bool
    ) -> None:
        from duckdb._sources import polars as _sources_polars

        pulled: set[int] = set()
        slices = _sources_polars._polars_slices

        def recording(frame: pl.DataFrame) -> Iterator[pl.Series]:
            for part in slices(frame):
                pulled.add(threading.get_ident())
                yield part

        monkeypatch.setattr(_sources_polars, "_polars_slices", recording)
        con.run("SET threads = 4")
        n = 2_000_000
        frame = pl.DataFrame({"n": range(n), "s": [str(i) for i in range(n)]})
        con.register("src", frame.lazy() if lazy else frame)
        assert rows(con, "SELECT count(*), sum(n) FROM src") == [(n, sum(range(n)))]
        assert len(pulled) > 1


class TestDatasetShapes:
    @pytest.mark.parametrize("kind", [pa.string_view(), pa.binary_view()])
    def test_a_view_column_keeps_every_filter_with_the_engine(
        self, con: duckdb.frame.Connection, kind: pa.DataType
    ) -> None:
        values = [b"abc", b"efg", None] if pa.types.is_binary_view(kind) else ["abc", "efg", None]
        source = Pushing(ds.dataset(pa.table({"v": pa.array(values, kind), "n": [1, 2, 3]})))
        con.register("views", source)
        assert rows(con, "SELECT n FROM views WHERE v IS NULL") == [(3,)]
        assert rows(con, "SELECT n FROM views WHERE n > 1") == [(2,), (3,)]
        assert rows(con, "SELECT v IS NULL FROM views WHERE n > 1") == [(False,), (True,)]
        assert source.applied == [[], [], []]

    def test_a_view_column_stays_unfilterable_in_this_pyarrow(self) -> None:
        views = pa.table({"v": pa.array(["a", "b"], pa.string_view()), "n": [1, 2]})
        with pytest.raises(pa.ArrowNotImplementedError, match="array_filter"):
            ds.dataset(views).scanner(filter=ds.field("n") > 1).to_table()

    def test_a_top_n_over_a_dataset_and_a_plan(self, con: duckdb.frame.Connection, typed_dir: Path) -> None:
        con.register("files", ds.dataset(typed_dir, partitioning="hive"))
        con.register("plan", pl.scan_parquet(typed_dir, hive_partitioning=True))
        for name in ("files", "plan"):
            assert rows(con, f"SELECT n FROM {name} ORDER BY n DESC LIMIT 2") == [(9,), (8,)]
            assert rows(con, f"SELECT n FROM {name} ORDER BY n ASC NULLS FIRST LIMIT 3") == [(None,), (None,), (0,)]

    def test_not_in_and_a_long_in_list(self, con: duckdb.frame.Connection, typed_dir: Path) -> None:
        source = Pushing(ds.dataset(typed_dir, partitioning="hive"))
        con.register("files", source)
        assert rows(con, "SELECT n FROM files WHERE n NOT IN (0, 1, 3, 4, 5, 6) ORDER BY n") == [(8,), (9,)]
        assert source.applied[-1] == ['(NOT ("n" IN (0, 1, 3, 4, 5, 6)))']
        members = ", ".join(str(i) for i in range(1000, 6000)) + ", 9"
        assert rows(con, f"SELECT n FROM files WHERE n IN ({members})") == [(9,)]
        [applied] = source.applied[-1]
        assert applied.startswith('("n" IN (1000, 1001')


class TestParallelOrdering:
    """The scan's batch ordering survives four threads pulling from one chunked or batched source."""

    def test_order_preserved_over_a_record_batch_reader(self, con: duckdb.frame.Connection) -> None:
        con.run("SET threads = 4")
        sizes = uneven_sizes(204)
        total = sum(sizes)
        schema = pa.schema([("n", pa.int64())])

        def batches() -> Iterator[pa.RecordBatch]:
            start = 0
            for size in sizes:
                yield pa.record_batch([pa.array(range(start, start + size))], schema=schema)
                start += size

        def register() -> None:
            con.register("src", pa.RecordBatchReader.from_batches(schema, batches()))

        check_order_preserved(con, register, "src", total)

    def test_order_preserved_over_a_table_with_many_chunks(self, con: duckdb.frame.Connection) -> None:
        con.run("SET threads = 4")
        sizes = uneven_sizes(204)
        total = sum(sizes)
        schema = pa.schema([("n", pa.int64())])
        start = 0
        batches = []
        for size in sizes:
            batches.append(pa.record_batch([pa.array(range(start, start + size))], schema=schema))
            start += size
        table_ = pa.Table.from_batches(batches)
        assert table_.column(0).num_chunks == len(sizes)
        con.register("src", table_)
        check_order_preserved(con, lambda: None, "src", total)

    def test_order_preserved_over_a_polars_frame_with_many_chunks(self, con: duckdb.frame.Connection) -> None:
        con.run("SET threads = 4")
        sizes = uneven_sizes(204)
        total = sum(sizes)
        start = 0
        parts = []
        for size in sizes:
            parts.append(pl.DataFrame({"n": range(start, start + size)}))
            start += size
        frame = pl.concat(parts, rechunk=False)
        assert frame.n_chunks() == len(sizes)
        con.register("src", frame)
        check_order_preserved(con, lambda: None, "src", total)

    def test_order_preserved_over_a_one_chunk_polars_frame(self, con: duckdb.frame.Connection) -> None:
        con.run("SET threads = 4")
        total = sum(uneven_sizes(204))
        frame = pl.DataFrame({"n": range(total)})
        assert frame.n_chunks() == 1
        con.register("src", frame)
        check_order_preserved(con, lambda: None, "src", total)


class TestParallelResultsMatch:
    """Every source family gives the same answer scanned by one thread as by four."""

    def test_matches_the_single_threaded_scan(self, con: duckdb.frame.Connection) -> None:
        n = 20_000
        chunk = 137
        schema = pa.schema([("n", pa.int64())])

        def table_batches() -> list[pa.RecordBatch]:
            return [
                pa.record_batch([pa.array(range(start, min(start + chunk, n)))], schema=schema)
                for start in range(0, n, chunk)
            ]

        table_ = pa.Table.from_batches(table_batches())
        frame = pl.concat(
            [pl.DataFrame({"n": range(start, min(start + chunk, n))}) for start in range(0, n, chunk)],
            rechunk=False,
        )
        pandas_frame = pd.DataFrame({"n": range(n)})

        makers: list[tuple[str, Callable[[], object]]] = [
            ("table", lambda: table_),
            ("batch reader", lambda: pa.RecordBatchReader.from_batches(schema, table_batches())),
            ("polars frame", lambda: frame),
            ("lazy frame", lambda: frame.lazy()),
            ("pandas frame", lambda: pandas_frame),
            ("capsule", lambda: table_.__arrow_c_stream__()),
        ]
        for name, maker in makers:
            for threads in (1, 4):
                con.run(f"SET threads = {threads}")
                con.register("src", maker())
                assert rows(con, "SELECT count(*), sum(n) FROM src") == [(n, sum(range(n)))], (name, threads)

    def test_a_dataset_and_a_scanner_match_the_single_threaded_scan(
        self, con: duckdb.frame.Connection, tmp_path: Path
    ) -> None:
        n = 200_000
        files = 4
        for part in range(files):
            start = part * n // files
            pq.write_table(pa.table({"n": range(start, start + n // files)}), tmp_path / f"{part}.parquet")
        makers: list[tuple[str, Callable[[], object]]] = [
            ("dataset", lambda: ds.dataset(tmp_path)),
            ("scanner", lambda: ds.dataset(tmp_path).scanner(batch_size=1000)),
        ]
        for name, maker in makers:
            for threads in (1, 4):
                con.run(f"SET threads = {threads}")
                con.register("files", maker())
                assert rows(con, "SELECT count(*), sum(n) FROM files") == [(n, sum(range(n)))], (name, threads)
