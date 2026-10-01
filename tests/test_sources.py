"""The object families that register as tables: pyarrow datasets and scanners, polars frames, pandas frames."""

from __future__ import annotations

import contextlib
import datetime
import decimal
import gc
import math
import re
import subprocess
import sys
import threading
import time
import uuid
import warnings
import weakref
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd
import pytest

import duckdb
from duckdb import exceptions
from duckdb._sources import NumpyScanSource, adapt
from duckdb._sources.arrow import ArrowArraySource, ArrowStreamSource
from duckdb._sources.numpy import SAMPLE_ROWS, NumpySource, _classify_objects
from duckdb._sources.pandas import PandasSource
from duckdb._sources.polars import LazyFrameSource, PolarsFrameSource
from duckdb._sources.pyarrow import PyArrowDatasetSource, PyArrowScannerSource, PyArrowTableSource
from duckdb.frame import col, sql, table

# polars publishes no free-threaded build and pyarrow none for Windows on ARM64; a test needing one is marked requires.
with contextlib.suppress(ModuleNotFoundError):
    import polars as pl
with contextlib.suppress(ModuleNotFoundError):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from pathlib import Path

    from duckdb._expressions import Expr

    ColumnsAnswer = list[tuple[str, str, str | None, object, object | None]]


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
    @pytest.mark.requires("polars", "pyarrow")
    def test_each_family_gets_its_adapter(self, parquet_dir: Path) -> None:
        dataset = ds.dataset(parquet_dir)
        assert isinstance(adapt(dataset), PyArrowDatasetSource)
        assert isinstance(adapt(dataset.scanner()), PyArrowScannerSource)
        assert isinstance(adapt(pl.DataFrame({"a": [1]}).lazy()), LazyFrameSource)
        assert isinstance(adapt(pd.DataFrame({"a": [1]})), PandasSource)

    @pytest.mark.requires("polars", "pyarrow")
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

    @pytest.mark.requires("polars", "pyarrow")
    def test_a_polars_frame_narrows_by_name(self) -> None:
        frame = pl.DataFrame({"a": [1], "b": ["x"]})
        source = adapt(frame)
        assert isinstance(source, PolarsFrameSource)
        capsule, projected = source.stream([1], ())
        assert projected
        assert pa.RecordBatchReader._import_from_c_capsule(capsule).read_all().column_names == ["b"]

    @pytest.mark.requires("pyarrow")
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

    @pytest.mark.requires("polars", "pyarrow")
    def test_series_and_fragments_are_not_datasets(self, parquet_dir: Path) -> None:
        for series in (pd.Series([1]), pl.Series([1])):
            source = adapt(series)
            assert isinstance(source, ArrowStreamSource)
            assert not source.one_shot
        assert isinstance(adapt(pa.array([1])), ArrowArraySource)
        fragment = next(iter(ds.dataset(parquet_dir).get_fragments()))
        with pytest.raises(TypeError, match="none of these"):
            adapt(fragment)

    @pytest.mark.requires("pyarrow")
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


@pytest.mark.requires("pyarrow")
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
        capsule, projected = PyArrowDatasetSource(ds.dataset(tmp_path, partitioning="hive")).stream([1], ())
        assert projected
        assert columns_of(capsule) == ["year"]

    def test_a_dataset_is_scanned_per_query(self, con: duckdb.frame.Connection, parquet_dir: Path) -> None:
        con.register("files", ds.dataset(parquet_dir))
        assert rows(con, "SELECT count(*) FROM files") == [(10,)]
        assert rows(con, "SELECT max(n) FROM files") == [(9,)]


def columns_of(capsule: object) -> list[str]:
    return list(pa.RecordBatchReader._import_from_c_capsule(capsule).read_all().column_names)


@pytest.mark.requires("pyarrow")
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

    @pytest.mark.requires("polars")
    def test_a_lazy_frame_is_narrowed_before_it_is_collected(self, con: duckdb.frame.Connection) -> None:
        seen: list[list[str]] = []

        def peek(frame: pl.DataFrame) -> pl.DataFrame:
            seen.append(frame.columns)
            return frame

        lazy = pl.DataFrame({"i": range(5), "s": ["a", "b", "c", "d", "e"], "unused": [0] * 5}).lazy()
        con.register("lf", lazy.map_batches(peek))
        assert rows(con, "SELECT s FROM lf WHERE i = 2") == [("c",)]
        capsule, projected = LazyFrameSource(lazy).stream([1], ())
        assert projected
        assert columns_of(capsule) == ["s"]

    def test_pandas_prepares_only_the_requested_columns(self) -> None:
        frame = pd.DataFrame({"i": [1, 2], "s": ["a", "b"], "o": pd.Series([object(), object()], dtype=object)})
        answer = PandasSource(frame).columns([1, 0])
        assert [(name, encoding) for name, encoding, *_ in answer] == [("s", "arrow"), ("i", "fixed")]

    def test_tables_and_scanners(self, parquet_dir: Path) -> None:
        table_ = pa.table({"a": [1], "b": [2], "c": [3]})
        capsule, projected = PyArrowTableSource(table_).stream([2, 0], ())
        assert projected
        assert columns_of(capsule) == ["c", "a"]
        scanner = ds.dataset(parquet_dir).scanner(columns=["s", "n"])
        capsule, projected = PyArrowScannerSource(scanner).stream([1], ())
        assert not projected
        assert columns_of(capsule) == ["s", "n"]


@pytest.mark.requires("pyarrow")
class TestCardinality:
    @pytest.mark.requires("polars")
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

    @pytest.mark.requires("polars")
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
    @pytest.mark.requires("pyarrow")
    def test_a_pandas_series_is_one_column(self, con: duckdb.frame.Connection) -> None:
        con.register("s", pd.Series([1.5, None], name="f"))
        assert table("s").schema(con) == [("value", "DOUBLE")]
        assert rows(con, "SELECT value FROM s") == [(1.5,), (None,)]

    @pytest.mark.requires("polars")
    def test_a_polars_series_keeps_its_name(self, con: duckdb.frame.Connection) -> None:
        con.register("s", pl.Series("p", ["x", None]))
        assert table("s").schema(con) == [("p", "VARCHAR")]
        assert rows(con, "SELECT p FROM s") == [("x",), (None,)]


@pytest.mark.requires("polars")
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
    @pytest.mark.requires("pyarrow")
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
            ("cat", "ENUM('a')"),
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

    @pytest.mark.requires("pyarrow")
    def test_a_later_value_in_a_text_column_reads_as_text(self, con: duckdb.frame.Connection) -> None:
        values: list[object] = ["text"] * 5000
        values[4999] = 12
        frame = pd.DataFrame({"o": pd.Series(values, dtype=object)})
        alone, beside = alone_and_beside(con, frame, "SELECT o, typeof(o) FROM {} WHERE o <> 'text'")
        assert alone == beside == [("12", "VARCHAR")]

    def test_a_pandas_source_over_something_else_is_refused(self) -> None:
        with pytest.raises(TypeError, match="needs a pandas DataFrame, not dict"):
            PandasSource({"a": [1]})

    def test_without_pyarrow_a_numpy_backed_frame_still_registers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        monkeypatch.setitem(sys.modules, "pyarrow", None)
        assert isinstance(adapt(pd.DataFrame({"a": [1]})), PandasSource)

    @pytest.mark.requires("pyarrow")
    def test_a_pyarrow_backed_column_is_read_without_importing_pyarrow(
        self, con: duckdb.frame.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys

        # Built while pyarrow is still importable: the column holds pyarrow data before the import is blocked.
        frame = pd.DataFrame({"a": pd.array(["a", "b"], dtype="string[pyarrow]")})
        monkeypatch.setitem(sys.modules, "pyarrow", None)
        con.register("t", frame)
        assert rows(con, "SELECT a FROM t") == [("a",), ("b",)]


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


@pytest.mark.requires("pyarrow")
class TestDatasetPushdown:
    def register_both(self, con: duckdb.frame.Connection, typed_dir: Path) -> Pushing:
        source = Pushing(ds.dataset(typed_dir, partitioning="hive"))
        con._register_source("pushed", source)
        con._register_source("plain", ArrowStreamSource(ds.dataset(typed_dir, partitioning="hive").to_table()))
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
        assert PyArrowScannerSource(scanner).accepts(col("n") > 8) is False
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


@pytest.mark.requires("polars")
class TestLazyFramePushdown:
    def register_both(self, con: duckdb.frame.Connection, typed_dir: Path) -> PushingLazily:
        source = PushingLazily(pl.scan_parquet(typed_dir, hive_partitioning=True))
        con._register_source("pushed", source)
        con._register_source("plain", ArrowStreamSource(pl.scan_parquet(typed_dir, hive_partitioning=True).collect()))
        return source

    @pytest.mark.requires("pyarrow")
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

    @pytest.mark.requires("pyarrow")
    @pytest.mark.parametrize("where", KEPT)
    def test_a_predicate_outside_the_set_stays_with_the_engine(
        self, con: duckdb.frame.Connection, typed_dir: Path, where: str
    ) -> None:
        source = self.register_both(con, typed_dir)
        expected = rows(con, f"SELECT n, year FROM plain WHERE {where} ORDER BY year, n")
        assert rows(con, f"SELECT n, year FROM pushed WHERE {where} ORDER BY year, n") == expected
        assert source.applied == [[]], where

    @pytest.mark.requires("pyarrow")
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
        con._register_source("late", source)
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
        con._register_source("late", source)
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
        con._register_source("late", source)
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
        assert PolarsFrameSource(frame).accepts(col("n") > 1) is False
        con.register("eager", frame)
        assert rows(con, "SELECT n FROM eager WHERE n > 1 ORDER BY n") == [(2,), (3,)]


@pytest.mark.requires("polars")
class TestPolarsTranslation:
    def translate(self, predicate: Expr) -> str:
        from duckdb._expressions.polars import to_polars

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
        return str(to_polars(predicate, schema))

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


@pytest.mark.requires("pyarrow")
class TestArrowTranslation:
    def translate(self, predicate: Expr) -> str:
        from duckdb._expressions.arrow import to_arrow

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
        return str(to_arrow(predicate, schema))

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

    @pytest.mark.requires("pyarrow")
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


def against_pyarrow(
    con: duckdb.frame.Connection, frame: pd.DataFrame, query: str
) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]]]:
    """`query`, `{}` standing for the table, on `frame` and on pyarrow's own conversion of it.

    The frame is read by the numpy scan and the conversion by the Arrow scan: an independent reading of the columns.
    """
    con.register("numpy", frame)
    con.register("arrow", pa.Table.from_pandas(frame, preserve_index=False))
    return rows(con, query.format("numpy")), rows(con, query.format("arrow"))


def beside_an_arrow_column(frame: pd.DataFrame) -> pd.DataFrame:
    """`frame` plus an unused pyarrow-backed string column."""
    return frame.assign(unused=pd.array(["x"] * len(frame), dtype="string[pyarrow]"))


def alone_and_beside(
    con: duckdb.frame.Connection, frame: pd.DataFrame, query: str
) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]]]:
    """`query`, `{}` standing for the table, on `frame` and on `frame` beside a pyarrow-backed column.

    Both are read by the numpy scan, and a column reads the same whatever the columns beside it.
    """
    con.register("alone", frame)
    con.register("beside", beside_an_arrow_column(frame))
    assert "Python Numpy Scan" in table("alone").on(con).explain()
    assert "Python Numpy Scan" in table("beside").on(con).explain()
    return rows(con, query.format("alone")), rows(con, query.format("beside"))


class TestPandasScanChoice:
    def test_a_plain_frame_is_read_by_the_numpy_scan(self) -> None:
        frame = pd.DataFrame({"i": range(3), "f": [1.0, 2.0, 3.0], "o": pd.Series(["a", "b", "c"], dtype=object)})
        assert isinstance(adapt(frame), PandasSource)

    @pytest.mark.requires("pyarrow")
    def test_a_frame_with_a_pyarrow_backed_column_is_read_by_the_numpy_scan_too(
        self, con: duckdb.frame.Connection
    ) -> None:
        frame = pd.DataFrame({"i": range(3), "s": pd.array(["a", "b", "c"], dtype="string[pyarrow]")})
        assert isinstance(adapt(frame), PandasSource)
        con.register("t", frame)
        assert rows(con, "SELECT i, s FROM t ORDER BY i") == [(0, "a"), (1, "b"), (2, "c")]
        assert "Python Numpy Scan" in table("t").on(con).explain()


class TestPandasNumpyScanFixed:
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

    @pytest.mark.requires("pyarrow")
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
    def test_every_plain_numeric_dtype_matches_pyarrow(
        self, con: duckdb.frame.Connection, dtype: str, type_text: str
    ) -> None:
        frame = pd.DataFrame({"a": pd.Series(range(10), dtype=dtype)})
        numpy_rows, arrow_rows = against_pyarrow(con, frame, "SELECT a, typeof(a) FROM {} ORDER BY a")
        assert numpy_rows == arrow_rows
        assert numpy_rows[0][1] == type_text

    @pytest.mark.requires("pyarrow")
    def test_plain_bool_matches_pyarrow(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame({"a": [i % 2 == 0 for i in range(10)]})
        numpy_rows, arrow_rows = against_pyarrow(con, frame, "SELECT a, typeof(a) FROM {}")
        assert numpy_rows == arrow_rows
        assert numpy_rows[0][1] == "BOOLEAN"

    @pytest.mark.requires("pyarrow")
    @pytest.mark.parametrize(
        "dtype", ["Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64", "Float32", "Float64"]
    )
    def test_every_nullable_numeric_dtype_matches_pyarrow(self, con: duckdb.frame.Connection, dtype: str) -> None:
        frame = pd.DataFrame({"a": pd.array([*range(9), None], dtype=dtype)})
        numpy_rows, arrow_rows = against_pyarrow(con, frame, "SELECT a FROM {}")
        assert numpy_rows == arrow_rows
        assert numpy_rows[-1] == (None,)

    @pytest.mark.requires("pyarrow")
    def test_nullable_boolean_matches_pyarrow(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame({"a": pd.array([True, None, pd.NA, np.nan, True], dtype="boolean")})
        numpy_rows, arrow_rows = against_pyarrow(con, frame, "SELECT a FROM {}")
        assert numpy_rows == [(True,), (None,), (None,), (None,), (True,)]
        assert numpy_rows == arrow_rows

    def test_nullable_narrow_dtypes_use_the_mask_not_a_nan_sentinel(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame({"i": pd.array([1, None, 3], dtype="Int8"), "u": pd.array([1, None, 3], dtype="UInt8")})
        con.register("t", frame)
        assert rows(con, "SELECT i, u FROM t") == [(1, 1), (None, None), (3, 3)]

    @pytest.mark.requires("pyarrow")
    def test_columns_of_a_shared_2d_block_stay_contiguous(self, con: duckdb.frame.Connection) -> None:
        block = np.arange(12, dtype=np.int64).reshape(4, 3)
        frame = pd.DataFrame(block, columns=["a", "b", "c"])
        numpy_rows, arrow_rows = against_pyarrow(con, frame[["a", "c"]], "SELECT a, c FROM {} ORDER BY a")
        assert numpy_rows == arrow_rows == [(0, 2), (3, 5), (6, 8), (9, 11)]

    @pytest.mark.parametrize("dtype", [">i4", ">u8", ">f8"])
    def test_a_column_stored_in_the_other_byte_order_reads_as_the_numpy_source_reads_it(
        self, con: duckdb.frame.Connection, dtype: str
    ) -> None:
        # pyarrow refuses byte-swapped arrays, so only a frame read without Arrow can hold one.
        data = np.array([1, 2, 300], dtype=dtype)
        con.register("frame", pd.DataFrame({"a": data}))
        con.register("arrays", {"a": data})
        assert rows(con, "SELECT a FROM frame") == rows(con, "SELECT a FROM arrays") == [(v,) for v in data.tolist()]

    def test_float16_widens_to_float32_once(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame({"a": pd.Series([1.5, 2.5, np.nan], dtype=np.float16)})
        con.register("t", frame)
        assert rows(con, "SELECT a, typeof(a) FROM t") == [(1.5, "FLOAT"), (2.5, "FLOAT"), (None, "FLOAT")]

    @pytest.mark.requires("pyarrow")
    def test_float64_nan_and_infinities(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame({"a": [1.0, float("nan"), float("inf"), float("-inf")]})
        numpy_rows, arrow_rows = against_pyarrow(con, frame, "SELECT a FROM {}")
        assert numpy_rows == [(1.0,), (None,), (float("inf"),), (float("-inf"),)]
        assert without_nan(numpy_rows) == without_nan(arrow_rows)


class TestPandasNumpyScanTemporal:
    """`"timestamp"` and `"interval"` columns: datetime64 and timedelta64, naive and zoned."""

    @pytest.mark.requires("pyarrow")
    @pytest.mark.parametrize(
        ("unit", "type_text"),
        [("s", "TIMESTAMP_S"), ("ms", "TIMESTAMP_MS"), ("us", "TIMESTAMP"), ("ns", "TIMESTAMP_NS")],
    )
    def test_naive_datetime64_units_match_the_arrow_path(
        self, con: duckdb.frame.Connection, unit: str, type_text: str
    ) -> None:
        frame = pd.DataFrame({"t": pd.to_datetime(["2020-01-02 03:04:05", None]).astype(f"datetime64[{unit}]")})
        numpy_rows, arrow_rows = against_pyarrow(con, frame, "SELECT t, typeof(t) FROM {}")
        assert numpy_rows == arrow_rows
        assert numpy_rows[0] == (datetime.datetime(2020, 1, 2, 3, 4, 5), type_text)
        assert numpy_rows[1] == (None, type_text)

    @pytest.mark.requires("pyarrow")
    @pytest.mark.parametrize(
        ("unit", "type_text"),
        [
            ("s", "TIMESTAMP WITH TIME ZONE"),
            ("ms", "TIMESTAMP WITH TIME ZONE"),
            ("us", "TIMESTAMP WITH TIME ZONE"),
            ("ns", "TIMESTAMPTZ_NS"),
        ],
    )
    def test_utc_aware_datetime64_matches_pyarrow(
        self, con: duckdb.frame.Connection, unit: str, type_text: str
    ) -> None:
        frame = pd.DataFrame(
            {"t": pd.to_datetime(["2020-01-02 03:04:05"]).tz_localize("UTC").astype(f"datetime64[{unit}, UTC]")}
        )
        numpy_rows, arrow_rows = against_pyarrow(con, frame, "SELECT t, typeof(t) FROM {}")
        assert numpy_rows == arrow_rows
        assert numpy_rows == [(datetime.datetime(2020, 1, 2, 3, 4, 5, tzinfo=datetime.UTC), type_text)]

    @pytest.mark.parametrize("zone", ["UTC", "Europe/Amsterdam"])
    def test_a_nanosecond_aware_column_keeps_its_nanoseconds(self, con: duckdb.frame.Connection, zone: str) -> None:
        instant = pd.Timestamp("2020-01-02 03:04:05.123456789", tz="UTC")
        frame = pd.DataFrame({"t": pd.Series([instant, pd.NaT], dtype="datetime64[ns, UTC]").dt.tz_convert(zone)})
        con.register("t", frame)
        assert rows(con, "SELECT typeof(t), epoch_ns(t) FROM t") == [
            ("TIMESTAMPTZ_NS", instant.value),
            ("TIMESTAMPTZ_NS", None),
        ]
        back = table("t").to_numpy(con)["t"]
        assert back.astype("int64")[0] == instant.value
        assert back.mask.tolist() == [False, True]

    @pytest.mark.requires("pyarrow")
    @pytest.mark.parametrize("zone", ["Europe/Berlin", "Asia/Kathmandu"])
    def test_a_non_utc_zone_keeps_its_instant(self, con: duckdb.frame.Connection, zone: str) -> None:
        frame = pd.DataFrame({"t": pd.to_datetime(["2020-06-02 03:04:05"]).tz_localize(zone)})
        numpy_rows, arrow_rows = against_pyarrow(con, frame, "SELECT t FROM {}")
        assert numpy_rows == arrow_rows
        utc = pd.to_datetime(["2020-06-02 03:04:05"]).tz_localize(zone).tz_convert("UTC")[0].to_pydatetime()
        assert numpy_rows == [(utc,)]

    @pytest.mark.requires("pyarrow")
    @pytest.mark.parametrize("year", [1680, 2260])
    def test_years_far_from_the_epoch_at_microsecond_resolution(self, con: duckdb.frame.Connection, year: int) -> None:
        frame = pd.DataFrame({"t": pd.to_datetime([f"{year}-01-02 03:04:05.123456"])})
        numpy_rows, arrow_rows = against_pyarrow(con, frame, "SELECT t FROM {}")
        assert numpy_rows == arrow_rows
        assert numpy_rows == [(datetime.datetime(year, 1, 2, 3, 4, 5, 123456),)]

    @pytest.mark.requires("pyarrow")
    def test_a_strided_datetime_series_reads_correctly(self, con: duckdb.frame.Connection) -> None:
        strided = pd.date_range("2020-01-01", periods=100, freq="h")[::23]
        frame = pd.DataFrame({"t": pd.Series(strided)})
        numpy_rows, arrow_rows = against_pyarrow(con, frame, "SELECT t FROM {}")
        assert numpy_rows == arrow_rows
        assert [row[0] for row in numpy_rows] == list(strided.to_pydatetime())

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


class TestPandasNumpyScanCategorical:
    """`"enum"` columns: pandas Categorical, string and non-string."""

    @pytest.mark.requires("pyarrow")
    def test_string_categories_with_and_without_null(self, con: duckdb.frame.Connection) -> None:
        # pyarrow converts a categorical to a dictionary, which the engine reads as VARCHAR, so only the values, not
        # typeof(), are compared with it here.
        frame = pd.DataFrame({"c": pd.Categorical(["x", "y", None, "x"])})
        numpy_rows, arrow_rows = against_pyarrow(con, frame, "SELECT c FROM {}")
        assert numpy_rows == arrow_rows == [("x",), ("y",), (None,), ("x",)]
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

    @pytest.mark.requires("pyarrow")
    def test_integer_categories_are_read_through_their_own_kind(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame({"c": pd.Categorical([1, 2, 1])})
        numpy_rows, arrow_rows = against_pyarrow(con, frame, "SELECT c, typeof(c) FROM {}")
        assert numpy_rows == arrow_rows == [(1, "BIGINT"), (2, "BIGINT"), (1, "BIGINT")]

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

    @pytest.mark.requires("pyarrow")
    def test_a_categorical_over_several_batches(self, con: duckdb.frame.Connection) -> None:
        n = 4096
        categories = [f"v{i % 5}" for i in range(n)]
        frame = pd.DataFrame({"c": pd.Categorical(categories)})
        numpy_rows, arrow_rows = against_pyarrow(con, frame, "SELECT c FROM {}")
        assert numpy_rows == arrow_rows == [(c,) for c in categories]

    def test_category_order_is_preserved_in_the_enum(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame({"c": pd.Categorical(["b", "a"], categories=["z", "a", "b"])})
        con.register("t", frame)
        assert table("t").schema(con) == [("c", "ENUM('z', 'a', 'b')")]


class TestPandasNumpyScanStrings:
    """`"text"` columns: Python-backed string arrays and plain object columns of strings."""

    @pytest.mark.requires("pyarrow")
    def test_object_strings_and_python_backed_strings_agree(self, con: duckdb.frame.Connection) -> None:
        values = ["a", "b", None]
        object_frame = pd.DataFrame({"s": pd.Series(values, dtype=object)})
        string_frame = pd.DataFrame({"s": pd.Series(values, dtype=pd.StringDtype("python"))})
        alone, beside = alone_and_beside(con, object_frame, "SELECT s FROM {}")
        con.register("string_backed", string_frame)
        assert alone == beside == rows(con, "SELECT s FROM string_backed") == [("a",), ("b",), (None,)]

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
        assert _classify_objects(np.array(values, dtype=object)) == ("objects", "BIGINT")

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

    @pytest.mark.requires("pyarrow")
    def test_a_column_of_dates_reads_as_date(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame(
            {"a": pd.Series([datetime.date(2024, 1, 1), None, datetime.date(1999, 12, 31)], dtype=object)}
        )
        alone, beside = alone_and_beside(con, frame, "SELECT a, typeof(a) FROM {}")
        assert (
            alone
            == beside
            == [(datetime.date(2024, 1, 1), "DATE"), (None, "DATE"), (datetime.date(1999, 12, 31), "DATE")]
        )

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


def epoch_ns(con: duckdb.frame.Connection, name: str) -> list[tuple[object, ...]]:
    return rows(con, f"SELECT epoch_ns(v), typeof(v) FROM {name}")


class TestPandasNumpyScalars:
    """numpy scalars in an object column read by what they mean, not by what `.item()` returns."""

    @pytest.mark.requires("pyarrow")
    @pytest.mark.parametrize(
        ("missing", "present"),
        [
            (np.float32("nan"), np.float32(1.5)),
            (np.float64("nan"), np.float64(1.5)),
            (np.datetime64("NaT", "ns"), np.datetime64("2020-01-01", "ns")),
            (np.datetime64("NaT", "us"), np.datetime64("2020-01-01", "us")),
            (np.timedelta64("NaT", "ns"), np.timedelta64(5, "ns")),
            (np.timedelta64("NaT", "us"), np.timedelta64(5, "us")),
        ],
        ids=repr,
    )
    def test_a_missing_marker_is_null(
        self, con: duckdb.frame.Connection, missing: np.generic, present: np.generic
    ) -> None:
        frame = pd.DataFrame({"v": pd.Series([missing, present, missing], dtype=object)})
        alone, beside = alone_and_beside(con, frame, "SELECT count(v), count(*), bool_or(v IS NULL) FROM {}")
        assert alone == beside == [(1, 3, True)]

    def test_a_half_precision_nan_is_null(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame({"v": pd.Series([np.float16("nan"), np.float16(1.5)], dtype=object)})
        con.register("t", frame)
        assert rows(con, "SELECT v, typeof(v) FROM t") == [(None, "DOUBLE"), (1.5, "DOUBLE")]

    @pytest.mark.requires("pyarrow")
    def test_a_column_of_nothing_but_nat_is_all_null(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame({"v": pd.Series([np.datetime64("NaT", "ns")] * 3, dtype=object)})
        alone, beside = alone_and_beside(con, frame, "SELECT count(v), count(*) FROM {}")
        assert alone == beside == [(0, 3)]

    @pytest.mark.requires("pyarrow")
    def test_nat_among_text_is_null_not_the_text_nat(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame({"v": pd.Series(["a", np.datetime64("NaT", "ns"), 1], dtype=object)})
        alone, beside = alone_and_beside(con, frame, "SELECT v FROM {}")
        assert alone == beside == [("a",), (None,), ("1",)]

    @pytest.mark.requires("pyarrow")
    def test_nanosecond_datetimes_read_as_timestamp_ns(self, con: duckdb.frame.Connection) -> None:
        values = [
            np.datetime64("2020-01-01T00:00:00.000000001", "ns"),
            np.datetime64("1969-12-31T23:59:59.999999999", "ns"),
            np.datetime64("NaT", "ns"),
        ]
        frame = pd.DataFrame({"v": pd.Series(values, dtype=object)})
        con.register("alone", frame)
        con.register("beside", beside_an_arrow_column(frame))
        expected = [(1_577_836_800_000_000_001, "TIMESTAMP_NS"), (-1, "TIMESTAMP_NS"), (None, "TIMESTAMP_NS")]
        assert epoch_ns(con, "alone") == epoch_ns(con, "beside") == expected

    def test_nanosecond_datetimes_mixed_with_python_datetimes(self, con: duckdb.frame.Connection) -> None:
        values = [np.datetime64("2020-01-01T00:00:00.000000001", "ns"), datetime.datetime(2020, 1, 1, 0, 0, 1)]
        frame = pd.DataFrame({"v": pd.Series(values, dtype=object)})
        con.register("numpy", frame)
        assert epoch_ns(con, "numpy") == [
            (1_577_836_800_000_000_001, "TIMESTAMP_NS"),
            (1_577_836_801_000_000_000, "TIMESTAMP_NS"),
        ]

    def test_a_datetime_past_the_nanosecond_range_keeps_the_column_at_microseconds(
        self, con: duckdb.frame.Connection
    ) -> None:
        far = datetime.datetime(2300, 1, 1, 1, 2, 3)
        near = np.datetime64("2021-06-01T00:00:00.000001", "ns")
        con.register("t", pd.DataFrame({"v": pd.Series([far, near], dtype=object)}))
        assert rows(con, "SELECT v, typeof(v) FROM t") == [
            (far, "TIMESTAMP"),
            (datetime.datetime(2021, 6, 1, 0, 0, 0, 1), "TIMESTAMP"),
        ]

    def test_a_date_past_the_nanosecond_range_keeps_the_column_at_microseconds(
        self, con: duckdb.frame.Connection
    ) -> None:
        values = [datetime.date(1500, 1, 1), np.datetime64("2021-06-01", "ns")]
        con.register("t", pd.DataFrame({"v": pd.Series(values, dtype=object)}))
        assert table("t").schema(con) == [("v", "TIMESTAMP")]
        assert rows(con, "SELECT v FROM t") == [(datetime.datetime(1500, 1, 1),), (datetime.datetime(2021, 6, 1),)]

    def test_a_sub_microsecond_value_in_a_microsecond_column_is_refused(self, con: duckdb.frame.Connection) -> None:
        values = [datetime.datetime(2300, 1, 1), np.datetime64("2021-06-01T00:00:00.000000001", "ns")]
        con.register("t", pd.DataFrame({"v": pd.Series(values, dtype=object)}))
        with pytest.raises(exceptions.InvalidInputError, match="cannot hold exactly"):
            rows(con, "SELECT v FROM t")

    @pytest.mark.parametrize(
        ("instant", "expected"),
        [
            (datetime.datetime(1677, 9, 21, 23, 59, 59, 999999), "TIMESTAMP"),
            (datetime.datetime(1677, 9, 22), "TIMESTAMP_NS"),
            (datetime.datetime(2262, 4, 11, 1), "TIMESTAMP_NS"),
            (datetime.datetime(2262, 4, 11, 23, 47, 16, 854775), "TIMESTAMP_NS"),
            (datetime.datetime(2262, 4, 11, 23, 47, 16, 854776), "TIMESTAMP"),
            (datetime.date(1677, 9, 21), "TIMESTAMP"),
            (datetime.date(2262, 4, 11), "TIMESTAMP_NS"),
        ],
        ids=str,
    )
    def test_the_nanosecond_range_is_judged_by_the_whole_instant(
        self, con: duckdb.frame.Connection, instant: datetime.date, expected: str
    ) -> None:
        near = np.datetime64("2021-06-01T00:00:00.000001", "ns")
        con.register("t", pd.DataFrame({"v": pd.Series([instant, near], dtype=object)}))
        assert table("t").schema(con) == [("v", expected)]
        read = rows(con, "SELECT v FROM t")
        whole = (
            instant if isinstance(instant, datetime.datetime) else datetime.datetime.combine(instant, datetime.time())
        )
        assert read == [(whole,), (datetime.datetime(2021, 6, 1, 0, 0, 0, 1),)]

    @pytest.mark.parametrize(
        "stepped",
        [
            np.array(10**16, dtype="datetime64[1000ps]")[()],
            np.array(10**13, dtype="datetime64[1000000fs]")[()],
            np.array(-(10**16), dtype="datetime64[1000ps]")[()],
        ],
        ids=["1000ps", "1000000fs", "negative 1000ps"],
    )
    def test_a_stepped_fine_unit_reads_like_its_nanoseconds(
        self, con: duckdb.frame.Connection, stepped: np.datetime64
    ) -> None:
        equivalent = stepped.astype("datetime64[ns]")
        con.register("t", pd.DataFrame({"v": pd.Series([stepped], dtype=object)}))
        assert epoch_ns(con, "t") == [(int(equivalent.view("i8")), "TIMESTAMP_NS")]

    def test_a_stepped_fine_duration_truncates_like_its_nanoseconds(self, con: duckdb.frame.Connection) -> None:
        stepped = [np.array(n, dtype="timedelta64[500ps]")[()] for n in (3, -3, 10**16)]
        con.register("t", pd.DataFrame({"v": pd.Series(stepped, dtype=object)}))
        assert rows(con, "SELECT v FROM t") == [
            (datetime.timedelta(0),),
            (datetime.timedelta(0),),
            (datetime.timedelta(microseconds=5 * 10**12),),
        ]

    def test_an_instant_past_the_nanosecond_range_is_refused(self, con: duckdb.frame.Connection) -> None:
        far = np.array(2 * 10**18, dtype="datetime64[10ns]")[()]
        con.register("u", pd.DataFrame({"v": pd.Series([far], dtype=object)}))
        with pytest.raises(exceptions.InvalidInputError, match="cannot hold exactly"):
            rows(con, "SELECT v FROM u")

    def test_microsecond_datetimes_still_read_as_timestamp(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame({"v": pd.Series([np.datetime64("2020-01-01T00:00:00.000001", "us")], dtype=object)})
        con.register("t", frame)
        assert rows(con, "SELECT v, typeof(v) FROM t") == [
            (datetime.datetime(2020, 1, 1, 0, 0, 0, 1), "TIMESTAMP"),
        ]

    def test_a_datetime_finer_than_a_nanosecond_is_refused(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame({"v": pd.Series([np.datetime64(1, "ps")], dtype=object)})
        con.register("t", frame)
        with pytest.raises(exceptions.InvalidInputError, match="cannot hold exactly"):
            rows(con, "SELECT v FROM t")

    def test_an_unsampled_nanosecond_datetime_is_refused_by_a_microsecond_column(
        self, con: duckdb.frame.Connection
    ) -> None:
        values: list[object] = [datetime.datetime(2020, 1, 1)] * (2 * SAMPLE_ROWS)
        values[1] = np.datetime64("2020-01-01T00:00:00.000000001", "ns")
        frame = pd.DataFrame({"v": pd.Series(values, dtype=object)})
        con.register("t", frame)
        assert table("t").schema(con) == [("v", "TIMESTAMP")]
        with pytest.raises(exceptions.InvalidInputError, match="cannot hold exactly"):
            rows(con, "SELECT v FROM t")

    @pytest.mark.requires("pyarrow")
    def test_nanosecond_timedeltas_truncate_to_microseconds(self, con: duckdb.frame.Connection) -> None:
        values = [np.timedelta64(1500, "ns"), np.timedelta64(-1500, "ns"), np.timedelta64(2_000_000_000, "ns")]
        frame = pd.DataFrame({"v": pd.Series(values, dtype=object)})
        alone, beside = alone_and_beside(con, frame, "SELECT v, typeof(v) FROM {}")
        expected = [
            (datetime.timedelta(microseconds=1), "INTERVAL"),
            (datetime.timedelta(microseconds=-1), "INTERVAL"),
            (datetime.timedelta(seconds=2), "INTERVAL"),
        ]
        assert alone == beside == expected

    def test_mixed_units_in_one_column(self, con: duckdb.frame.Connection) -> None:
        durations = [np.timedelta64(2, "s"), np.timedelta64(1500, "ns"), np.timedelta64(3, "D")]
        instants = [np.datetime64(1, "s"), np.datetime64(1, "ns")]
        con.register("d", pd.DataFrame({"v": pd.Series(durations, dtype=object)}))
        con.register("i", pd.DataFrame({"v": pd.Series(instants, dtype=object)}))
        assert rows(con, "SELECT v FROM d") == [
            (datetime.timedelta(seconds=2),),
            (datetime.timedelta(microseconds=1),),
            (datetime.timedelta(days=3),),
        ]
        assert epoch_ns(con, "i") == [(1_000_000_000, "TIMESTAMP_NS"), (1, "TIMESTAMP_NS")]

    def test_object_and_typed_timedelta_columns_agree(self, con: duckdb.frame.Connection) -> None:
        values = [np.timedelta64(n, "ns") for n in (1500, -1500, 999, -999)]
        con.register("objects", pd.DataFrame({"v": pd.Series(values, dtype=object)}))
        con.register("typed", pd.DataFrame({"v": pd.Series(values, dtype="timedelta64[ns]")}))
        assert rows(con, "SELECT v FROM objects") == rows(con, "SELECT v FROM typed")


STRIDED_COLUMNS = {
    "i": pd.Series(range(7)),
    "f": pd.Series([0.5, float("nan"), 2.5, 3.5, 4.5, 5.5, 6.5]),
    "o": pd.Series(list("abcdefg"), dtype=object),
    "m": pd.Series([1, None, 3, 4, None, 6, 7], dtype="Int64"),
    "c": pd.Categorical(list("xyzxyzx")),
    "t": pd.Series(pd.date_range("2020-01-01", periods=7, freq="h")),
    "z": pd.Series(pd.date_range("2020-01-01", periods=7, freq="h", tz="Europe/Amsterdam")),
    "d": pd.Series(pd.to_timedelta(range(7), unit="s")),
}


class Relaid(PandasSource):
    """A pandas source handing the numpy scan each data array relaid in memory, values unchanged, as `layout` says."""

    def __init__(self, obj: object, layout: str) -> None:
        super().__init__(obj)
        self.layout = layout

    def columns(self, columns: Sequence[int] | None) -> ColumnsAnswer:
        answer: ColumnsAnswer = []
        for name, encoding, type_text, data, mask in super().columns(columns):
            assert isinstance(data, np.ndarray)
            if self.layout == "unaligned":
                packed = np.zeros(len(data), dtype=np.dtype([("pad", "i1"), ("value", data.dtype)], align=False))
                packed["value"] = data
                data = packed["value"]
                assert not data.flags.aligned
            elif self.layout == "broadcast":
                data = np.broadcast_to(data[:1], data.shape)
                mask = None if mask is None else np.broadcast_to(np.asarray(mask)[:1], data.shape)
            answer.append((name, encoding, type_text, data, mask))
        return answer


class TestPandasStridedViews:
    """A row-stepped view hands over strided buffers, which the numpy scan reads in place."""

    @pytest.mark.parametrize("step", [2, -1, -3])
    @pytest.mark.parametrize("column", list(STRIDED_COLUMNS))
    def test_a_stepped_view_reads_like_its_copy(self, con: duckdb.frame.Connection, column: str, step: int) -> None:
        view = pd.DataFrame({column: STRIDED_COLUMNS[column]}).iloc[::step]
        con.register("view", view)
        con.register("copy", view.copy())
        expected = rows(con, f"SELECT {column} FROM copy")
        assert len(expected) == len(range(7)[::step])
        assert rows(con, f"SELECT {column} FROM view") == expected

    def test_an_empty_reversed_view(self, con: duckdb.frame.Connection) -> None:
        con.register("t", pd.DataFrame({"i": pd.Series([], dtype="int64")}).iloc[::-1])
        assert rows(con, "SELECT i FROM t") == []

    @pytest.mark.parametrize("step", [1, -2])
    def test_a_column_is_not_copied(self, step: int) -> None:
        frame = pd.DataFrame({"i": np.arange(10), "o": pd.Series(list("abcdefghij"), dtype=object)}).iloc[::step]
        for (_, _, _, data, _), column in zip(PandasSource(frame).columns(None), ("i", "o"), strict=True):
            assert isinstance(data, np.ndarray)
            assert data.strides[0] == step * data.itemsize
            assert np.shares_memory(data, frame[column].to_numpy(copy=False))

    def test_a_reversed_view_across_several_ranges_keeps_its_order(self, con: duckdb.frame.Connection) -> None:
        # Past three of the ranges threads claim, so a range starting mid-view is read at its own offset.
        n = 900_000
        frame = pd.DataFrame({"i": np.arange(n), "o": pd.Series(np.arange(n).astype(str), dtype=object)}).iloc[::-3]
        con.register("t", frame)
        assert rows(con, "SELECT i, o FROM t") == list(zip(frame["i"].tolist(), frame["o"].tolist(), strict=True))

    @pytest.mark.parametrize("column", ["i", "f", "m", "t", "z", "d"])
    def test_an_unaligned_buffer_reads_like_its_copy(self, con: duckdb.frame.Connection, column: str) -> None:
        frame = pd.DataFrame({column: STRIDED_COLUMNS[column]}).iloc[::2]
        con._register_source("unaligned", Relaid(frame, "unaligned"))
        con.register("copy", frame.copy())
        assert rows(con, f"SELECT {column} FROM unaligned") == rows(con, f"SELECT {column} FROM copy")

    @pytest.mark.parametrize("column", list(STRIDED_COLUMNS))
    def test_a_broadcast_buffer_repeats_its_one_element(self, con: duckdb.frame.Connection, column: str) -> None:
        frame = pd.DataFrame({column: STRIDED_COLUMNS[column]})
        con._register_source("broadcast", Relaid(frame, "broadcast"))
        first = rows(con, f"SELECT {column} FROM broadcast LIMIT 1")
        assert rows(con, f"SELECT {column} FROM broadcast") == first * len(frame)
        con.register("copy", frame.iloc[:1].copy())
        assert first == rows(con, f"SELECT {column} FROM copy")


#: Strided numpy arrays, scanned with pandas unimportable.
SCAN_WITHOUT_PANDAS = """
import sys
sys.modules["pandas"] = None
import numpy as np
import duckdb

con = duckdb.frame.connect()
con.register("t", {"o": np.array(["a", "x", None, "y", "c"], dtype=object)[::2], "i": np.arange(6)[::-2]})
assert duckdb.frame.sql("SELECT o, i FROM t").rows(con) == [("a", 5), (None, 3), ("c", 1)]
assert sys.modules["pandas"] is None
"""


class TestNumpyScanWithoutPandas:
    def test_numpy_arrays_scan_with_pandas_unimportable(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", SCAN_WITHOUT_PANDAS], capture_output=True, text=True, check=False, timeout=60
        )
        assert result.returncode == 0, result.stderr


@pytest.mark.requires("pyarrow")
class TestPandasTextUnification:
    """A mixed or nested object column reads as text, whatever the columns beside it."""

    def test_consistent_dicts_and_lists_of_ints_read_as_text(self, con: duckdb.frame.Connection) -> None:
        dicts = pd.DataFrame({"a": pd.Series([{"x": 1}, {"x": 2}], dtype=object)})
        alone, beside = alone_and_beside(con, dicts, "SELECT a, typeof(a) FROM {}")
        assert alone == beside == [(str({"x": 1}), "VARCHAR"), (str({"x": 2}), "VARCHAR")]
        lists = pd.DataFrame({"a": pd.Series([[1, 2], [3, 4]], dtype=object)})
        alone, beside = alone_and_beside(con, lists, "SELECT a, typeof(a) FROM {}")
        assert alone == beside == [(str([1, 2]), "VARCHAR"), (str([3, 4]), "VARCHAR")]

    def test_bytes_mixed_with_str_reads_as_text(self, con: duckdb.frame.Connection) -> None:
        values: list[object] = [b"abc", "text"]
        frame = pd.DataFrame({"a": pd.Series(values, dtype=object)})
        alone, beside = alone_and_beside(con, frame, "SELECT a FROM {}")
        assert alone == beside == [(str(v),) for v in values]


class MisreportingColumns(PandasSource):
    """A pandas source whose columns() reply can be broken in a chosen way, to exercise the numpy scan's own checks."""

    def __init__(self, obj: object, break_as: str) -> None:
        super().__init__(obj)
        self.break_as = break_as

    def columns(self, columns: Sequence[int] | None) -> ColumnsAnswer:
        answer = super().columns(columns)
        if self.break_as == "short":
            return answer[:-1]
        name, encoding, type_text, data, mask = answer[0]
        assert isinstance(data, np.ndarray)
        if self.break_as == "renamed":
            return [("not_" + name, encoding, type_text, data, mask), *answer[1:]]
        if self.break_as == "two_dimensional":
            return [(name, encoding, type_text, data.reshape(-1, 1), mask), *answer[1:]]
        if self.break_as == "wrong_width":
            return [(name, encoding, type_text, data.astype(np.int8), mask), *answer[1:]]
        if self.break_as == "short_mask":
            return [(name, encoding, type_text, data, np.zeros(len(data) - 1, dtype=bool)), *answer[1:]]
        if self.break_as == "swapped":
            return [(name, encoding, type_text, data.astype(data.dtype.newbyteorder()), mask), *answer[1:]]
        return answer


class TestPandasNumpyScanValidation:
    """The scan refuses a malformed columns() answer instead of misreading it, naming the offending column."""

    def test_a_short_answer_is_refused(self, con: duckdb.frame.Connection) -> None:
        con._register_source("t", MisreportingColumns(pd.DataFrame({"a": range(3), "b": range(3)}), "short"))
        with pytest.raises(exceptions.InvalidInputError, match=r"1 columns where 2 were requested"):
            rows(con, "SELECT * FROM t")

    def test_a_renamed_column_is_refused(self, con: duckdb.frame.Connection) -> None:
        con._register_source("t", MisreportingColumns(pd.DataFrame({"a": range(3)}), "renamed"))
        with pytest.raises(exceptions.InvalidInputError, match="column 'not_a' at position 0 where 'a' was expected"):
            rows(con, "SELECT * FROM t")

    def test_a_two_dimensional_buffer_is_refused(self, con: duckdb.frame.Connection) -> None:
        con._register_source("t", MisreportingColumns(pd.DataFrame({"a": range(10)}), "two_dimensional"))
        with pytest.raises(exceptions.InvalidInputError, match="for 'a' with a data array that is not one-dimensional"):
            rows(con, "SELECT * FROM t")

    def test_a_mask_of_the_wrong_length_is_refused_and_the_data_buffer_released(
        self, con: duckdb.frame.Connection
    ) -> None:
        frame = pd.DataFrame({"a": range(10)})
        data = frame["a"].to_numpy(copy=False)
        before = sys.getrefcount(data)
        con._register_source("t", MisreportingColumns(frame, "short_mask"))
        for _ in range(3):
            with pytest.raises(exceptions.InvalidInputError, match="mask array of 9 rows where 10 were expected"):
                rows(con, "SELECT * FROM t")
        assert sys.getrefcount(data) == before

    def test_a_buffer_in_the_other_byte_order_is_refused(self, con: duckdb.frame.Connection) -> None:
        con._register_source("t", MisreportingColumns(pd.DataFrame({"a": range(10)}), "swapped"))
        with pytest.raises(exceptions.InvalidInputError, match="in the other byte order"):
            rows(con, "SELECT * FROM t")

    def test_a_mismatched_element_width_is_refused(self, con: duckdb.frame.Connection) -> None:
        con._register_source("t", MisreportingColumns(pd.DataFrame({"a": range(10)}), "wrong_width"))
        with pytest.raises(exceptions.InvalidInputError, match="answered columns\\(\\) for 'a'"):
            rows(con, "SELECT * FROM t")


class TestPandasNumpyScanShapes:
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


class TestPandasNumpyScanParallel:
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


class TestPandasNumpyScanLifetime:
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


def chunked_like(values: list[object], arrow_type: pa.DataType) -> pa.ChunkedArray:
    """`values` as a chunked array of `arrow_type` in uneven chunks, one of them empty, so reads cross boundaries."""
    chunks = [pa.array([], arrow_type)]
    start = 0
    for size in uneven_sizes(len(values)):
        if start >= len(values):
            break
        chunks.append(pa.array(values[start : start + size], arrow_type))
        start += size
    return pa.chunked_array(chunks, arrow_type)


def arrow_backed(**columns: pa.ChunkedArray) -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(pd.arrays.ArrowExtensionArray(data)) for name, data in columns.items()})


N_ARROW = 5000
#: Each kind's values, with nulls among them; `arrow_column` gives each its Arrow type.
ARROW_VALUES: dict[str, list[Any]] = {
    "string": [None if i % 7 == 0 else f"s{i}" * (i % 3) for i in range(N_ARROW)],
    "large_string": [None if i % 7 == 0 else f"s{i}" for i in range(N_ARROW)],
    "string_view": [None if i % 7 == 0 else f"a string past the inline limit {i}" for i in range(N_ARROW)],
    "int64": [None if i % 5 == 0 else i for i in range(N_ARROW)],
    "bool": [None if i % 5 == 0 else i % 3 == 0 for i in range(N_ARROW)],
    "timestamp_tz": [
        None if i % 9 == 0 else datetime.datetime(2020, 1, 1) + datetime.timedelta(seconds=i) for i in range(N_ARROW)
    ],
    "timestamp_ns": [None if i % 9 == 0 else 10**18 + i for i in range(N_ARROW)],
    "date32": [None if i % 9 == 0 else 19_000 + i for i in range(N_ARROW)],
    "duration": [None if i % 9 == 0 else i * 1_000 for i in range(N_ARROW)],
    "decimal": [None if i % 7 == 0 else decimal.Decimal(i) / 100 for i in range(N_ARROW)],
    "binary": [None if i % 7 == 0 else bytes([i % 256]) * (i % 4) for i in range(N_ARROW)],
    "list": [None if i % 6 == 0 else list(range(i % 4)) for i in range(N_ARROW)],
    "large_list": [None if i % 6 == 0 else [f"x{j}" for j in range(i % 3)] for i in range(N_ARROW)],
    "fixed_size_list": [None if i % 6 == 0 else [i, -i] for i in range(N_ARROW)],
    "struct": [None if i % 6 == 0 else {"a": i, "b": None if i % 4 == 0 else f"b{i}"} for i in range(N_ARROW)],
    "map": [None if i % 8 == 0 else [(f"k{i}", i)] for i in range(N_ARROW)],
    "list_of_structs": [None if i % 6 == 0 else [{"a": j} for j in range(i % 3)] for i in range(N_ARROW)],
    "dictionary": [None if i % 10 == 0 else f"c{i % 4}" for i in range(N_ARROW)],
    "uuid": [None if i % 7 == 0 else uuid.UUID(int=i).bytes for i in range(N_ARROW)],
    "null": [None] * N_ARROW,
}


def arrow_column(kind: str) -> pa.ChunkedArray:
    types = {
        "string": pa.string(),
        "large_string": pa.large_string(),
        "string_view": pa.string_view(),
        "int64": pa.int64(),
        "bool": pa.bool_(),
        "timestamp_tz": pa.timestamp("us", tz="Europe/Amsterdam"),
        "timestamp_ns": pa.timestamp("ns"),
        "date32": pa.date32(),
        "duration": pa.duration("us"),
        "decimal": pa.decimal128(12, 2),
        "binary": pa.binary(),
        "list": pa.list_(pa.int32()),
        "large_list": pa.large_list(pa.string()),
        "fixed_size_list": pa.list_(pa.int64(), 2),
        "struct": pa.struct([("a", pa.int64()), ("b", pa.string())]),
        "map": pa.map_(pa.string(), pa.int64()),
        "list_of_structs": pa.list_(pa.struct([("a", pa.int64())])),
        "dictionary": pa.dictionary(pa.int8(), pa.string()),
        "uuid": pa.uuid(),
        "null": pa.null(),
    }
    return chunked_like(ARROW_VALUES[kind], types[kind])


class ChangingArrowColumn(PandasSource):
    """A pandas source whose frame's column `a` is replaced by `replacement` once a scan asks for its columns."""

    def __init__(self, obj: pd.DataFrame, replacement: pd.Series) -> None:
        super().__init__(obj)
        self.replacement = replacement

    def columns(self, columns: Sequence[int] | None) -> ColumnsAnswer:
        self.obj["a"] = self.replacement
        return super().columns(columns)


@pytest.mark.requires("pyarrow")
class TestPandasNumpyScanArrowColumns:
    """The numpy scan reads a pyarrow-backed column as its Arrow data, through core's Arrow importer."""

    @pytest.mark.parametrize("kind", ARROW_VALUES)
    def test_a_column_reads_as_the_arrow_scan_reads_the_same_data(
        self, con: duckdb.frame.Connection, kind: str
    ) -> None:
        data = arrow_column(kind)
        con.register("numpy_scanned", arrow_backed(v=data))
        con.register("arrow_scanned", pa.table({"v": data}))
        through_arrow = rows(con, "SELECT v, typeof(v) FROM arrow_scanned")
        assert rows(con, "SELECT v, typeof(v) FROM numpy_scanned") == through_arrow
        assert len(through_arrow) == N_ARROW

    def test_a_null_in_a_dictionary_column_stays_null_in_every_batch(self, con: duckdb.frame.Connection) -> None:
        # The engine's dictionary import reads an uncounted null count, -1, as no nulls, so every batch needs its count.
        values = [None if i % 10 == 0 else f"c{i % 4}" for i in range(N_ARROW)]
        con.register("t", arrow_backed(v=pa.chunked_array([pa.array(values).dictionary_encode()])))
        assert [row[0] for row in rows(con, "SELECT v FROM t")] == values

    def test_a_run_end_encoded_column_reads_every_run(self, con: duckdb.frame.Connection) -> None:
        values = [f"r{i // 1000}" if i % 3000 else None for i in range(N_ARROW)]
        encoded = pa.chunked_array(
            [pc.run_end_encode(pa.array(values[:4000])), pc.run_end_encode(pa.array(values[4000:]))]
        )
        con.register("t", arrow_backed(v=encoded))
        assert rows(con, "SELECT typeof(v) FROM t LIMIT 1") == [("VARCHAR",)]
        assert [row[0] for row in rows(con, "SELECT v FROM t")] == values

    def test_the_default_string_dtype_and_string_pyarrow_read_as_varchar(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame(
            {
                "inferred": pd.Series(["a", None, "c"]),
                "declared": pd.Series(["x", "y", None], dtype="string[pyarrow]"),
            }
        )
        assert isinstance(frame["inferred"].array, pd.arrays.ArrowStringArray)
        con.register("t", frame)
        assert rows(con, "SELECT inferred, declared, typeof(inferred), typeof(declared) FROM t") == [
            ("a", "x", "VARCHAR", "VARCHAR"),
            (None, "y", "VARCHAR", "VARCHAR"),
            ("c", None, "VARCHAR", "VARCHAR"),
        ]

    def test_a_struct_column_stays_one_column(self, con: duckdb.frame.Connection) -> None:
        struct = pa.chunked_array([pa.array([{"a": 1, "b": "x"}])])
        con.register("t", arrow_backed(s=struct, n=pa.chunked_array([pa.array([7])])))
        assert rows(con, "SELECT * FROM t") == [({"a": 1, "b": "x"}, 7)]

    def test_columns_chunked_differently_stay_aligned_across_threads(self, con: duckdb.frame.Connection) -> None:
        con.run("SET threads = 4")
        vector_size = int(con._engine().get_option("standard_vector_size"))
        total = vector_size * 50 * 3 + 777
        texts = [f"t{i}" for i in range(total)]
        frame = arrow_backed(
            a=chunked_like(list(range(total)), pa.int64()),
            b=pa.chunked_array([pa.array(texts[:1]), pa.array(texts[1 : total - 3]), pa.array(texts[total - 3 :])]),
        )
        frame["n"] = np.arange(total)
        con.register("t", frame)
        assert rows(con, "SELECT a, b, n FROM t") == [(i, f"t{i}", i) for i in range(total)]

    def test_projection_reorders_and_narrows_arrow_columns(self, con: duckdb.frame.Connection) -> None:
        frame = arrow_backed(a=pa.chunked_array([pa.array([1, 2])]), b=pa.chunked_array([pa.array(["x", "y"])]))
        frame["c"] = [1.5, 2.5]
        con.register("t", frame)
        assert rows(con, "SELECT b, c, a FROM t") == [("x", 1.5, 1), ("y", 2.5, 2)]
        assert rows(con, "SELECT b FROM t WHERE a = 2") == [("y",)]
        assert rows(con, "SELECT count(*) FROM t") == [(2,)]

    @pytest.mark.parametrize("chunks", [0, 2])
    def test_a_frame_of_no_rows(self, con: duckdb.frame.Connection, chunks: int) -> None:
        empty = pa.chunked_array([pa.array([], pa.string())] * chunks, pa.string())
        con.register("t", arrow_backed(s=empty))
        assert rows(con, "SELECT s FROM t") == []
        assert rows(con, "DESCRIBE t")[0][:2] == ("s", "VARCHAR")

    @pytest.mark.xfail(strict=True, reason="the engine's union import reads the members without the union's offset")
    def test_a_sparse_union_column_reads_every_batch(self, con: duckdb.frame.Connection) -> None:
        tags = pa.array([0, 1] * 3000, pa.int8())
        members = [
            pa.array([None if i % 10 == 0 else i for i in range(6000)]),
            pa.array([f"s{i}" for i in range(6000)]),
        ]
        union = pa.UnionArray.from_sparse(tags, members, ["i", "s"])
        con.register("t", arrow_backed(u=pa.chunked_array([union])))
        assert [row[0] for row in rows(con, "SELECT u FROM t")] == union.to_pylist()

    def test_an_arrow_type_the_engine_does_not_import_is_refused_when_bound(self, con: duckdb.frame.Connection) -> None:
        con.register("t", arrow_backed(h=pa.chunked_array([pa.array([1.5], pa.float16())])))
        with pytest.raises(exceptions.NotSupportedError, match="Unsupported Internal Arrow Type"):
            rows(con, "SELECT h FROM t")

    def test_an_arrow_type_changed_between_binding_and_scanning_is_refused(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame({"a": pd.array([1, 2], dtype="int64[pyarrow]")})
        con._register_source("t", ChangingArrowColumn(frame, pd.Series(["x", "y"], dtype="string[pyarrow]")))
        with pytest.raises(exceptions.InvalidInputError, match="with the type VARCHAR, but the query was bound when"):
            rows(con, "SELECT a FROM t")

    def test_an_arrow_stream_of_the_wrong_length_is_refused(self, con: duckdb.frame.Connection) -> None:
        class Shortened(PandasSource):
            def columns(self, columns: Sequence[int] | None) -> ColumnsAnswer:
                name, encoding, type_text, _, mask = super().columns(columns)[0]
                short = self.obj["a"].array.__arrow_array__().slice(1)
                return [(name, encoding, type_text, short.__arrow_c_stream__(), mask)]

        con._register_source("t", Shortened(pd.DataFrame({"a": pd.array([1, 2, 3], dtype="int64[pyarrow]")})))
        with pytest.raises(exceptions.InvalidInputError, match="an Arrow stream of 2 rows where 3 were expected"):
            rows(con, "SELECT a FROM t")

    def test_an_arrow_stream_failing_midway_is_refused(self, con: duckdb.frame.Connection) -> None:
        class Failing(PandasSource):
            def columns(self, columns: Sequence[int] | None) -> ColumnsAnswer:
                name, encoding, type_text, _, mask = super().columns(columns)[0]

                def arrays() -> Iterator[pa.RecordBatch]:
                    yield pa.record_batch({"a": [1]})
                    message = "the producer broke"
                    raise ValueError(message)

                reader = pa.RecordBatchReader.from_batches(pa.schema({"a": pa.int64()}), arrays())
                return [(name, encoding, type_text, reader.__arrow_c_stream__(), mask)]

        con._register_source("t", Failing(pd.DataFrame({"a": pd.array([1, 2], dtype="int64[pyarrow]")})))
        with pytest.raises(
            exceptions.InvalidInputError, match=r"with an Arrow stream that failed: .*the producer broke"
        ):
            rows(con, "SELECT a FROM t")

    def test_a_described_capsule_other_than_a_schema_is_refused(self, con: duckdb.frame.Connection) -> None:
        class Misdescribed(PandasSource):
            def describe(self) -> list[tuple[str, object]]:
                return [("a", self.obj["a"].array.__arrow_array__().__arrow_c_stream__())]

        con._register_source("t", Misdescribed(pd.DataFrame({"a": pd.array([1], dtype="int64[pyarrow]")})))
        with pytest.raises(exceptions.InvalidInputError, match="an 'arrow_schema' capsule but a 'arrow_array_stream'"):
            rows(con, "SELECT a FROM t")

    def test_every_scan_releases_the_arrow_data(self, con: duckdb.frame.Connection) -> None:
        gc.collect()
        before = pa.total_allocated_bytes()
        total = 300_000
        frame = arrow_backed(
            s=chunked_like([f"value {i}" for i in range(total)], pa.string()),
            d=pa.chunked_array([pa.array([f"c{i % 5}" for i in range(total)]).dictionary_encode()]),
        )
        frame["n"] = np.arange(total)
        con.run("SET threads = 4")
        con.register("t", frame)
        assert rows(con, "SELECT count(s), max(d) FROM t") == [(total, "c4")]
        assert len(rows(con, "SELECT s, d FROM t LIMIT 3")) == 3
        # The column opened before the refused one is released with it.
        con._register_source("t", ChangingArrowColumn(frame.assign(a=frame["n"]), frame["s"]))
        with pytest.raises(exceptions.InvalidInputError, match="but the query was bound when"):
            rows(con, "SELECT s, a FROM t")
        con.unregister("t")
        del frame
        gc.collect()
        assert pa.total_allocated_bytes() == before


@pytest.mark.requires("polars")
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


@pytest.mark.requires("pyarrow")
class TestDatasetShapes:
    @pytest.mark.parametrize("binary", [False, True], ids=["string_view", "binary_view"])
    def test_a_view_column_keeps_every_filter_with_the_engine(self, con: duckdb.frame.Connection, binary: bool) -> None:
        kind = pa.binary_view() if binary else pa.string_view()
        values = [b"abc", b"efg", None] if binary else ["abc", "efg", None]
        source = Pushing(ds.dataset(pa.table({"v": pa.array(values, kind), "n": [1, 2, 3]})))
        con._register_source("views", source)
        assert rows(con, "SELECT n FROM views WHERE v IS NULL") == [(3,)]
        assert rows(con, "SELECT n FROM views WHERE n > 1") == [(2,), (3,)]
        assert rows(con, "SELECT v IS NULL FROM views WHERE n > 1") == [(False,), (True,)]
        assert source.applied == [[], [], []]

    def test_a_view_column_stays_unfilterable_in_this_pyarrow(self) -> None:
        views = pa.table({"v": pa.array(["a", "b"], pa.string_view()), "n": [1, 2]})
        with pytest.raises(pa.ArrowNotImplementedError, match="array_filter"):
            ds.dataset(views).scanner(filter=ds.field("n") > 1).to_table()

    @pytest.mark.requires("polars")
    def test_a_top_n_over_a_dataset_and_a_plan(self, con: duckdb.frame.Connection, typed_dir: Path) -> None:
        con.register("files", ds.dataset(typed_dir, partitioning="hive"))
        con.register("plan", pl.scan_parquet(typed_dir, hive_partitioning=True))
        for name in ("files", "plan"):
            assert rows(con, f"SELECT n FROM {name} ORDER BY n DESC LIMIT 2") == [(9,), (8,)]
            assert rows(con, f"SELECT n FROM {name} ORDER BY n ASC NULLS FIRST LIMIT 3") == [(None,), (None,), (0,)]

    def test_not_in_and_a_long_in_list(self, con: duckdb.frame.Connection, typed_dir: Path) -> None:
        source = Pushing(ds.dataset(typed_dir, partitioning="hive"))
        con._register_source("files", source)
        assert rows(con, "SELECT n FROM files WHERE n NOT IN (0, 1, 3, 4, 5, 6) ORDER BY n") == [(8,), (9,)]
        assert source.applied[-1] == ['(NOT ("n" IN (0, 1, 3, 4, 5, 6)))']
        members = ", ".join(str(i) for i in range(1000, 6000)) + ", 9"
        assert rows(con, f"SELECT n FROM files WHERE n IN ({members})") == [(9,)]
        [applied] = source.applied[-1]
        assert applied.startswith('("n" IN (1000, 1001')


class TestParallelOrdering:
    """The scan's batch ordering survives four threads pulling from one chunked or batched source."""

    @pytest.mark.requires("pyarrow")
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

    @pytest.mark.requires("pyarrow")
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

    @pytest.mark.requires("polars")
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

    @pytest.mark.requires("polars")
    def test_order_preserved_over_a_one_chunk_polars_frame(self, con: duckdb.frame.Connection) -> None:
        con.run("SET threads = 4")
        total = sum(uneven_sizes(204))
        frame = pl.DataFrame({"n": range(total)})
        assert frame.n_chunks() == 1
        con.register("src", frame)
        check_order_preserved(con, lambda: None, "src", total)


@pytest.mark.requires("pyarrow")
class TestParallelResultsMatch:
    """Every source family gives the same answer scanned by one thread as by four."""

    @pytest.mark.requires("polars")
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


def typed(con: duckdb.frame.Connection, obj: object) -> list[tuple[object, ...]]:
    """`obj` registered as `t`, then each row of its first column with that column's engine type."""
    con.register("t", obj)
    name = table("t").schema(con)[0][0]
    return rows(con, f'SELECT "{name}", typeof("{name}") FROM t')


class TestNumpyShapes:
    """What registers as a table and how its columns are named: rows stay rows, whatever the array's layout."""

    def test_a_one_dimensional_array_is_one_column(self, con: duckdb.frame.Connection) -> None:
        con.register("t", np.arange(3))
        assert table("t").schema(con) == [("column0", "BIGINT")]
        assert rows(con, "SELECT * FROM t") == [(0,), (1,), (2,)]

    def test_a_tall_two_dimensional_array_is_two_columns(self, con: duckdb.frame.Connection) -> None:
        # The previous client made one column per row, so this array became 10,000 columns.
        data = np.arange(20_000).reshape(10_000, 2)
        con.register("t", data)
        assert table("t").schema(con) == [("column0", "BIGINT"), ("column1", "BIGINT")]
        assert rows(con, "SELECT count(*), sum(column0), sum(column1) FROM t") == [
            (10_000, int(data[:, 0].sum()), int(data[:, 1].sum()))
        ]

    @pytest.mark.parametrize("order", ["C", "F"])
    def test_either_memory_order_reads_its_columns_in_place(self, con: duckdb.frame.Connection, order: str) -> None:
        data = (np.ascontiguousarray if order == "C" else np.asfortranarray)(np.arange(12).reshape(4, 3))
        for column, (_, _, _, array, _) in enumerate(NumpySource(data).columns(None)):
            assert np.shares_memory(array, data)
            assert np.array_equal(np.asarray(array), data[:, column])
        con.register("t", data)
        assert rows(con, "SELECT * FROM t") == [tuple(row) for row in data.tolist()]

    def test_a_transposed_view_reads_as_its_copy(self, con: duckdb.frame.Connection) -> None:
        data = np.arange(12).reshape(3, 4).T
        con.register("view", data)
        con.register("copy", data.copy())
        assert rows(con, "SELECT * FROM view") == rows(con, "SELECT * FROM copy") == [tuple(r) for r in data.tolist()]

    def test_one_column_of_a_matrix_is_read(self, con: duckdb.frame.Connection) -> None:
        con.register("t", np.arange(6).reshape(3, 2))
        assert rows(con, "SELECT column1 FROM t") == [(1,), (3,), (5,)]
        assert "Projections: column1" in sql("SELECT column1 FROM t").on(con).explain()

    def test_equal_sub_arrays_of_mixed_dtypes_fold_into_an_object_matrix(self, con: duckdb.frame.Connection) -> None:
        folded = np.array([np.array([1, 9], dtype=np.int32), np.array([2.0, 8.0]), np.array(["3", "7"])], dtype=object)
        assert folded.shape == (3, 2)
        con.register("t", folded)
        assert table("t").schema(con) == [("column0", "VARCHAR"), ("column1", "VARCHAR")]
        assert rows(con, "SELECT * FROM t") == [("1", "9"), ("2.0", "8.0"), ("3", "7")]

    def test_a_matrix_subclass_reads_like_its_array(self, con: duckdb.frame.Connection) -> None:
        with pytest.warns(PendingDeprecationWarning, match="matrix subclass"):
            matrix = np.matrix([[1, 2], [3, 4]])
        con.register("t", matrix)
        assert rows(con, "SELECT * FROM t") == [(1, 2), (3, 4)]

    def test_a_dict_names_its_columns_by_key_text(self, con: duckdb.frame.Connection) -> None:
        con.register("t", {"a": np.arange(2), 1: np.array([3.5, 4.5]), "A": np.array([True, False])})
        assert table("t").schema(con) == [("a", "BIGINT"), ("1", "DOUBLE"), ("A_1", "BOOLEAN")]
        assert rows(con, "SELECT * FROM t") == [(0, 3.5, True), (1, 4.5, False)]

    def test_a_key_repeating_another_as_text_gets_a_suffix(self, con: duckdb.frame.Connection) -> None:
        con.register("t", {1: np.arange(1), "1": np.arange(1)})
        assert [name for name, _ in table("t").schema(con)] == ["1", "1_1"]

    @pytest.mark.parametrize("container", [list, tuple])
    def test_a_list_or_tuple_of_arrays_is_numbered_columns(
        self, con: duckdb.frame.Connection, container: Callable[[list[object]], object]
    ) -> None:
        con.register("t", container([np.arange(2), np.array(["x", "y"])]))
        assert table("t").schema(con) == [("column0", "BIGINT"), ("column1", "VARCHAR")]
        assert rows(con, "SELECT * FROM t") == [(0, "x"), (1, "y")]

    def test_the_plan_names_the_numpy_scan_and_the_row_count(self, con: duckdb.frame.Connection) -> None:
        con.register("t", {"a": np.arange(77)})
        plan = table("t").on(con).explain()
        assert "Python Numpy Scan" in plan
        assert "~77 rows" in plan

    def test_an_empty_array_is_an_empty_table(self, con: duckdb.frame.Connection) -> None:
        con.register("t", np.array([], dtype=np.int64))
        assert table("t").schema(con) == [("column0", "BIGINT")]
        assert rows(con, "SELECT * FROM t") == []

    @pytest.mark.parametrize(
        ("obj", "error", "message"),
        [
            (np.array(5), TypeError, "one- or two-dimensional, not 0-dimensional"),
            (np.zeros((2, 2, 2)), TypeError, "one- or two-dimensional, not 3-dimensional"),
            (np.zeros((3, 0)), ValueError, "two-dimensional numpy array needs at least one column"),
            ({}, ValueError, "a registered dict needs at least one array"),
            ([], ValueError, "a registered list needs at least one array"),
            ((), ValueError, "a registered tuple needs at least one array"),
            ({"a": np.arange(2), "b": [1, 2]}, TypeError, "the value at key 'b' is a list, not a numpy array"),
            ((np.arange(2), 7), TypeError, "the value at position 1 is a int, not a numpy array"),
            ({"a": np.zeros((2, 2))}, TypeError, "the array at key 'a' is 2-dimensional, not one-dimensional"),
            ({"a": np.arange(2), "b": np.arange(3)}, ValueError, "column 'b' has 3 rows where 'a' has 2"),
            (np.int64(4), TypeError, "a registered numpy object must be an array, not a int64"),
        ],
    )
    def test_a_shape_that_is_no_table_is_refused_at_registration(
        self, con: duckdb.frame.Connection, obj: object, error: type[Exception], message: str
    ) -> None:
        with pytest.raises(error, match=re.escape(message)):
            con.register("t", obj)


class TestNumpyDtypes:
    """Each numpy dtype's engine type and values."""

    @pytest.mark.parametrize(
        ("dtype", "type_text"),
        [
            ("i1", "TINYINT"),
            ("i2", "SMALLINT"),
            ("i4", "INTEGER"),
            ("i8", "BIGINT"),
            ("u1", "UTINYINT"),
            ("u2", "USMALLINT"),
            ("u4", "UINTEGER"),
            ("u8", "UBIGINT"),
            ("f2", "FLOAT"),
            ("f4", "FLOAT"),
            ("f8", "DOUBLE"),
            ("?", "BOOLEAN"),
        ],
    )
    def test_a_fixed_width_dtype(self, con: duckdb.frame.Connection, dtype: str, type_text: str) -> None:
        data = np.array([0, 1], dtype=dtype)
        assert typed(con, data) == [(value, type_text) for value in data.tolist()]

    @pytest.mark.parametrize("dtype", [">i2", ">i4", ">i8", ">u8", ">f4", ">f8"])
    def test_the_other_byte_order_is_swapped(self, con: duckdb.frame.Connection, dtype: str) -> None:
        data = np.array([1, -2 if dtype[1] != "u" else 2, 300], dtype=dtype)
        con.register("t", data)
        assert rows(con, "SELECT * FROM t") == [(value,) for value in data.tolist()]

    def test_a_nan_is_null_and_an_infinity_a_value(self, con: duckdb.frame.Connection) -> None:
        con.register("t", np.array([1.0, np.nan, np.inf, -np.inf]))
        assert rows(con, "SELECT * FROM t") == [(1.0,), (None,), (float("inf"),), (float("-inf"),)]

    def test_fixed_width_text_is_varchar_not_an_enum(self, con: duckdb.frame.Connection) -> None:
        # The previous client typed this as ENUM('xxx', 'zzz') through numpy.unique.
        assert typed(con, np.array(["zzz", "xxx"])) == [("zzz", "VARCHAR"), ("xxx", "VARCHAR")]

    def test_fixed_width_text_keeps_embedded_nuls_and_drops_padding(self, con: duckdb.frame.Connection) -> None:
        data = np.array(["goo\x00se", "", "héllo", "\U0001d11e", "a\x00\x00"], dtype="U8")
        con.register("t", data)
        assert rows(con, "SELECT * FROM t") == [("goo\x00se",), ("",), ("héllo",), ("\U0001d11e",), ("a",)]
        assert [value for (value,) in rows(con, "SELECT * FROM t")] == data.tolist()

    def test_fixed_width_text_in_the_other_byte_order(self, con: duckdb.frame.Connection) -> None:
        con.register("t", np.array(["ab", "\U0001d11e"], dtype=">U2"))
        assert rows(con, "SELECT * FROM t") == [("ab",), ("\U0001d11e",)]

    def test_a_code_point_that_is_not_unicode_is_refused_at_its_row(self, con: duckdb.frame.Connection) -> None:
        con.register("t", np.array([0x61, 0xD800], dtype=np.uint32).view("U1"))
        with pytest.raises(exceptions.InvalidInputError, match="column 'column0' holds a value at row 1 that is not"):
            rows(con, "SELECT * FROM t")

    def test_fixed_width_bytes_are_a_blob_with_padding_dropped(self, con: duckdb.frame.Connection) -> None:
        data = np.array([b"\x00\x00\x00a", b"", b"ab\x00"], dtype="S4")
        assert typed(con, data) == [(b"\x00\x00\x00a", "BLOB"), (b"", "BLOB"), (b"ab", "BLOB")]
        assert [value for value, _ in typed(con, data)] == data.tolist()

    def test_variable_width_strings_are_varchar(self, con: duckdb.frame.Connection) -> None:
        data = np.array(["a", None, "long " * 20], dtype=np.dtypes.StringDType(na_object=None))
        assert typed(con, data) == [("a", "VARCHAR"), (None, "VARCHAR"), ("long " * 20, "VARCHAR")]
        assert typed(con, np.array(["x"], dtype=np.dtypes.StringDType())) == [("x", "VARCHAR")]

    @pytest.mark.parametrize(
        ("unit", "type_text"),
        [("s", "TIMESTAMP_S"), ("ms", "TIMESTAMP_MS"), ("us", "TIMESTAMP"), ("ns", "TIMESTAMP_NS")],
    )
    def test_a_datetime_in_an_engine_unit(self, con: duckdb.frame.Connection, unit: str, type_text: str) -> None:
        data = np.array(["2020-01-02T03:04:05", "NaT"], dtype=f"datetime64[{unit}]")
        assert typed(con, data) == [(datetime.datetime(2020, 1, 2, 3, 4, 5), type_text), (None, type_text)]

    @pytest.mark.parametrize(
        ("dtype", "value", "expected"),
        [
            ("datetime64[D]", "2020-03-04", datetime.date(2020, 3, 4)),
            ("datetime64[W]", 2, datetime.date(1970, 1, 15)),
            ("datetime64[M]", "2020-03", datetime.date(2020, 3, 1)),
            ("datetime64[Y]", "2020", datetime.date(2020, 1, 1)),
            ("datetime64[3D]", 2, datetime.date(1970, 1, 7)),
            ("datetime64[2M]", 1, datetime.date(1970, 3, 1)),
        ],
    )
    def test_a_datetime_of_days_or_longer_is_a_date(
        self, con: duckdb.frame.Connection, dtype: str, value: object, expected: datetime.date
    ) -> None:
        data = np.array([value, "NaT"], dtype=dtype)
        assert typed(con, data) == [(expected, "DATE"), (None, "DATE")]

    def test_the_last_days_a_date_holds_are_read_and_the_next_refused(self, con: duckdb.frame.Connection) -> None:
        limit = 2**31 - 2
        con.register("edge", np.array([limit, -limit], dtype="datetime64[D]"))
        assert rows(con, "SELECT datediff('day', DATE '1970-01-01', column0) FROM edge") == [(limit,), (-limit,)]
        for beyond in (limit + 1, -limit - 1, 2**40):
            con.register("beyond", np.array([beyond], dtype="datetime64[D]"))
            with pytest.raises(exceptions.InvalidInputError, match="beyond the range of its engine type"):
                rows(con, "SELECT * FROM beyond")
        con.register("years", np.array([2**45], dtype="datetime64[Y]"))
        with pytest.raises(exceptions.InvalidInputError, match="beyond the range of its engine type"):
            rows(con, "SELECT * FROM years")

    @pytest.mark.parametrize(
        ("dtype", "value", "expected"),
        [
            ("datetime64[h]", "2020-01-02T03", datetime.datetime(2020, 1, 2, 3)),
            ("datetime64[m]", "2020-01-02T03:04", datetime.datetime(2020, 1, 2, 3, 4)),
            ("datetime64[10s]", 3, datetime.datetime(1970, 1, 1, 0, 0, 30)),
        ],
    )
    def test_a_coarse_or_stepped_datetime_is_scaled_to_seconds(
        self, con: duckdb.frame.Connection, dtype: str, value: object, expected: datetime.datetime
    ) -> None:
        data = np.array([value, "NaT"], dtype=dtype)
        assert typed(con, data) == [(expected, "TIMESTAMP_S"), (None, "TIMESTAMP_S")]

    def test_a_stepped_engine_unit_keeps_its_unit(self, con: duckdb.frame.Connection) -> None:
        data = np.array([3], dtype="datetime64[250ms]")
        assert typed(con, data) == [(datetime.datetime(1970, 1, 1, 0, 0, 0, 750_000), "TIMESTAMP_MS")]

    def test_a_scaled_datetime_past_int64_is_refused_at_its_row(self, con: duckdb.frame.Connection) -> None:
        con.register("t", np.array([1, 2**62], dtype="datetime64[h]"))
        with pytest.raises(exceptions.InvalidInputError, match="holds a value at row 1 that overflows its engine type"):
            rows(con, "SELECT * FROM t")

    @pytest.mark.parametrize(
        ("dtype", "value", "expected"),
        [
            ("timedelta64[ns]", 1500, datetime.timedelta(microseconds=1)),
            ("timedelta64[us]", -3, datetime.timedelta(microseconds=-3)),
            ("timedelta64[ms]", 4, datetime.timedelta(milliseconds=4)),
            ("timedelta64[s]", 5, datetime.timedelta(seconds=5)),
            ("timedelta64[m]", 90, datetime.timedelta(minutes=90)),
            ("timedelta64[h]", -2, datetime.timedelta(hours=-2)),
            ("timedelta64[D]", 3, datetime.timedelta(days=3)),
            ("timedelta64[W]", 1, datetime.timedelta(weeks=1)),
            ("timedelta64[10ms]", 3, datetime.timedelta(milliseconds=30)),
        ],
    )
    def test_a_duration_is_an_interval(
        self, con: duckdb.frame.Connection, dtype: str, value: int, expected: datetime.timedelta
    ) -> None:
        data = np.array([value, "NaT"], dtype=dtype)
        assert typed(con, data) == [(expected, "INTERVAL"), (None, "INTERVAL")]

    @pytest.mark.parametrize(
        ("data", "message"),
        [
            (np.array([1j]), "and no engine type holds a complex number"),
            (np.zeros(2, dtype=[("a", "i4")]), "which is structured or raw bytes"),
            (np.zeros(2, dtype="V4"), "which is structured or raw bytes"),
            (np.array([1], dtype="datetime64[ps]"), "which is finer than the engine's nanoseconds"),
            (np.array([1], dtype="datetime64[fs]"), "which is finer than the engine's nanoseconds"),
            (np.array([1], dtype="timedelta64[as]"), "which is finer than the engine's nanoseconds"),
            (np.array([1], dtype="timedelta64[M]"), "which counts months or years"),
            (np.array([1], dtype="timedelta64[Y]"), "which counts months or years"),
        ],
    )
    def test_a_dtype_no_engine_type_holds_is_refused_at_registration(
        self, con: duckdb.frame.Connection, data: np.ndarray[Any, Any], message: str
    ) -> None:
        with pytest.raises(TypeError, match=re.escape(f"column 'column0' has the numpy dtype {data.dtype}, {message}")):
            con.register("t", data)

    def test_a_duration_without_a_unit_is_refused(self, con: duckdb.frame.Connection) -> None:
        with warnings.catch_warnings():
            # numpy deprecates the unit-less duration from 2.5 on, which Python 3.11 cannot install.
            warnings.filterwarnings("ignore", "The 'generic' unit", DeprecationWarning)
            data = np.array([1], dtype="timedelta64")
        with pytest.raises(TypeError, match="has the numpy dtype timedelta64, which has no unit"):
            con.register("t", data)

    @pytest.mark.parametrize("frame", [False, True], ids=["numpy", "pandas"])
    def test_a_long_double_reads_as_double_only_where_it_is_eight_bytes(
        self, con: duckdb.frame.Connection, frame: bool
    ) -> None:
        # numpy's long double is eight bytes on some platforms, such as macOS on arm64, and wider on others.
        data = np.array([1.5, -2.25], dtype=np.longdouble)
        obj: object = pd.DataFrame({"column0": data}) if frame else data

        def read() -> list[tuple[object, ...]]:
            con.register("t", obj)
            return rows(con, "SELECT column0, typeof(column0) FROM t")

        if data.itemsize == 8:
            assert read() == [(1.5, "DOUBLE"), (-2.25, "DOUBLE")]
        else:
            with pytest.raises((TypeError, exceptions.InvalidInputError), match="no engine type has its width"):
                read()

    def test_numpy_string_scalars_in_an_object_array_are_text(self, con: duckdb.frame.Connection) -> None:
        # The previous client once read every numpy.str_ as NULL.
        data = np.array([np.str_("a"), np.str_("b")], dtype=object)
        assert typed(con, data) == [("a", "VARCHAR"), ("b", "VARCHAR")]

    def test_an_object_array_of_arrays_reads_as_their_text(self, con: duckdb.frame.Connection) -> None:
        data = np.empty(2, dtype=object)
        data[0], data[1] = np.array([1, 2]), np.array([3])
        assert typed(con, data) == [(str(data[0]), "VARCHAR"), (str(data[1]), "VARCHAR")]

    def test_none_na_and_nan_in_an_object_array_are_null(self, con: duckdb.frame.Connection) -> None:
        data = np.array([True, None, pd.NA, np.nan, True], dtype=object)
        assert typed(con, data) == [(v, "BOOLEAN") for v in (True, None, None, None, True)]


class TestNumpyMasks:
    """A numpy masked array's mask marks NULL whatever its dtype; the previous client ignored it everywhere."""

    @pytest.mark.parametrize(
        "data",
        [
            np.array([1, 999, 3]),
            np.array([1.5, 999.0, 3.5]),
            np.array([True, False, True]),
            np.array(["a", "garbage", "c"]),
            np.array([b"a", b"garbage", b"c"]),
            np.array(["2020-01-01", "1999-01-01", "2020-01-03"], dtype="datetime64[s]"),
            np.array(["2020-01-01", "1999-01-01", "2020-01-03"], dtype="datetime64[D]"),
            np.array([1, 999, 3], dtype="timedelta64[s]"),
            np.array([1, 999, 3], dtype="timedelta64[h]"),
            np.array([uuid.UUID(int=1), uuid.UUID(int=999), uuid.UUID(int=3)], dtype=object),
            np.array(["a", "garbage", "c"], dtype=np.dtypes.StringDType()),
        ],
        ids=lambda data: str(data.dtype),
    )
    def test_a_masked_value_is_null(self, con: duckdb.frame.Connection, data: np.ndarray[Any, Any]) -> None:
        masked = np.ma.array(data, mask=[False, True, False])
        con.register("masked", masked)
        con.register("plain", data)
        plain = rows(con, "SELECT * FROM plain")
        assert rows(con, "SELECT * FROM masked") == [plain[0], (None,), plain[2]]
        assert table("masked").schema(con) == table("plain").schema(con)

    def test_an_unmasked_nan_stays_a_nan(self, con: duckdb.frame.Connection) -> None:
        con.register("t", np.ma.array([np.nan, 1.0], mask=[False, True]))
        [(first,), (second,)] = rows(con, "SELECT * FROM t")
        assert isinstance(first, float)
        assert math.isnan(first)
        assert second is None

    def test_a_masked_value_the_type_cannot_hold_is_never_read(self, con: duckdb.frame.Connection) -> None:
        data = np.ma.array(np.array([1, "not a number", 3], dtype=object), mask=[False, True, False])
        assert typed(con, data) == [(1, "BIGINT"), (None, "BIGINT"), (3, "BIGINT")]

    def test_the_sample_skips_masked_cells_to_find_a_valid_one(self, con: duckdb.frame.Connection) -> None:
        values = np.array(["text"] * 5000 + [7], dtype=object)
        mask = np.ones(len(values), dtype=bool)
        mask[-1] = False
        con.register("t", np.ma.array(values, mask=mask))
        assert table("t").schema(con) == [("column0", "BIGINT")]
        assert rows(con, "SELECT count(column0), sum(column0) FROM t") == [(1, 7)]

    @pytest.mark.parametrize(
        ("dtype", "type_text"),
        [
            ("datetime64[D]", "DATE"),
            ("datetime64[W]", "DATE"),
            ("datetime64[Y]", "DATE"),
            ("datetime64[h]", "TIMESTAMP_S"),
            ("timedelta64[h]", "INTERVAL"),
        ],
    )
    def test_a_masked_value_out_of_range_is_not_checked(
        self, con: duckdb.frame.Connection, dtype: str, type_text: str
    ) -> None:
        data = np.ma.array(np.array([0, 2**62, 1]).view(dtype), mask=[False, True, False])
        con.register("t", data)
        [(first, _), (second, second_type), (third, _)] = rows(con, "SELECT column0, typeof(column0) FROM t")
        assert (second, second_type) == (None, type_text)
        con.register("plain", np.ma.getdata(data)[[0, 2]])
        assert [(first,), (third,)] == rows(con, "SELECT * FROM plain")

    def test_a_mask_and_a_nat_together(self, con: duckdb.frame.Connection) -> None:
        con.register(
            "t",
            np.ma.array(
                np.array(["2020-01-01", "NaT", "2020-01-03"], dtype="datetime64[D]"), mask=[True, False, False]
            ),
        )
        assert rows(con, "SELECT * FROM t") == [(None,), (None,), (datetime.date(2020, 1, 3),)]

    def test_a_masked_matrix_masks_each_column(self, con: duckdb.frame.Connection) -> None:
        con.register("t", np.ma.array(np.arange(4).reshape(2, 2), mask=[[False, True], [True, False]]))
        assert rows(con, "SELECT * FROM t") == [(0, None), (None, 3)]

    def test_a_masked_array_without_a_mask_has_no_nulls(self, con: duckdb.frame.Connection) -> None:
        con.register("t", {"a": np.ma.array([1, 2])})
        assert rows(con, "SELECT * FROM t") == [(1,), (2,)]

    def test_a_masked_view_reversed_across_several_ranges(self, con: duckdb.frame.Connection) -> None:
        n = 600_000
        data = np.ma.array(np.arange(n), mask=np.arange(n) % 7 == 0)[::-2]
        con.register("t", data)
        assert rows(con, "SELECT * FROM t") == [(value,) for value in data.tolist()]


class TestNumpyLifetime:
    """A registered object is read as it is at each query and kept alive by its registration."""

    def test_a_change_after_registration_shows_in_the_next_query(self, con: duckdb.frame.Connection) -> None:
        data = np.arange(3)
        con.register("t", data)
        data[0] = 999
        assert rows(con, "SELECT * FROM t") == [(999,), (1,), (2,)]

    def test_a_resized_array_is_read_at_its_new_size(self, con: duckdb.frame.Connection) -> None:
        data = np.arange(3)
        con.register("t", data)
        assert rows(con, "SELECT count(*) FROM t") == [(3,)]
        data.resize(6, refcheck=False)
        data[3:] = [7, 8, 9]
        assert rows(con, "SELECT * FROM t") == [(0,), (1,), (2,), (7,), (8,), (9,)]

    def test_a_dict_changed_after_registration_is_read_as_it_is_now(self, con: duckdb.frame.Connection) -> None:
        columns = {"a": np.arange(2)}
        con.register("t", columns)
        columns["b"] = np.array([5, 6])
        assert rows(con, "SELECT * FROM t") == [(0, 5), (1, 6)]

    def test_registration_keeps_a_column_view_and_its_memory_alive(self, con: duckdb.frame.Connection) -> None:
        def register() -> weakref.ref[np.ndarray[Any, Any]]:
            owner = np.arange(20)
            con.register("t", {"x": owner.reshape(10, 2)[:, 1]})
            return weakref.ref(owner)

        base = register()
        gc.collect()
        assert base() is not None
        assert rows(con, "SELECT sum(x) FROM t") == [(sum(range(1, 20, 2)),)]
        con.unregister("t")
        gc.collect()
        assert base() is None

    def test_an_array_over_an_offset_into_a_buffer(self, con: duckdb.frame.Connection) -> None:
        backing = np.array([1, 2, 3, 4, 5])
        view = np.ndarray((4,), buffer=backing, offset=backing.itemsize, dtype=backing.dtype)
        con.register("t", view)
        assert rows(con, "SELECT * FROM t") == [(2,), (3,), (4,), (5,)]

    @pytest.mark.parametrize("shape", ["array", "dict", "list"])
    def test_repeated_queries_hold_no_reference(self, con: duckdb.frame.Connection, shape: str) -> None:
        data = np.array(["a", "b"], dtype=object)
        obj = {"array": data, "dict": {"a": data}, "list": [data]}[shape]
        con.register("t", obj)
        rows(con, "SELECT * FROM t")
        gc.collect()
        before = sys.getrefcount(data)
        for _ in range(50):
            rows(con, "SELECT * FROM t")
        gc.collect()
        assert sys.getrefcount(data) == before


#: The numpy source's object columns, classified with pandas unimportable.
CLASSIFY_WITHOUT_PANDAS = """
import datetime, sys, uuid
sys.modules["pandas"] = None
import numpy as np
import duckdb

con = duckdb.frame.connect()
con.register("t", {
    "i": np.array([1, None, 3], dtype=object),
    "t": np.array([datetime.datetime(2020, 1, 1), None], dtype=object)[[0, 1, 1]],
    "u": np.ma.array(np.array([uuid.UUID(int=1)] * 3, dtype=object), mask=[0, 0, 1]),
    "s": np.array(["a", "b", "c"]),
})
schema = duckdb.frame.table("t").schema(con)
assert schema == [("i", "BIGINT"), ("t", "TIMESTAMP"), ("u", "UUID"), ("s", "VARCHAR")], schema
one = uuid.UUID(int=1)
assert duckdb.frame.sql("SELECT i, u, s FROM t").rows(con) == [(1, one, "a"), (None, one, "b"), (3, None, "c")]
assert sys.modules["pandas"] is None
"""


class TestNumpyWithoutPandas:
    def test_object_columns_classify_the_same_with_pandas_unimportable(self) -> None:
        # The previous client once read every object column as text when pandas was absent.
        result = subprocess.run(
            [sys.executable, "-c", CLASSIFY_WITHOUT_PANDAS], capture_output=True, text=True, check=False, timeout=60
        )
        assert result.returncode == 0, result.stderr


class TestNumpyRoundTrip:
    """What `to_numpy()` produces registers back; after one trip the dict is a fixed point."""

    QUERY = (
        "SELECT * FROM (VALUES (1, 1.5, 'a', true, TIMESTAMP '2020-01-01 01:02:03', INTERVAL 90 MINUTE, "
        "uuid '00000000-0000-0000-0000-000000000001', '\\x00ab'::BLOB, 7::TINYINT, "
        "TIMESTAMP_NS '2020-01-01 00:00:00.000000001', DATE '2020-01-02', 12.5::DECIMAL(4,1)), "
        "(NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)) "
        "t(i, f, s, b, ts, iv, u, bl, ti, tns, d, dec)"
    )

    def test_a_fetched_dict_registers_back_with_its_values_and_nulls(self, con: duckdb.frame.Connection) -> None:
        fetched = sql(self.QUERY).to_numpy(con)
        con.register("back", fetched)
        original = rows(con, self.QUERY)
        back = rows(con, "SELECT * FROM back")
        # DECIMAL leaves as float64, so it comes back as a DOUBLE of the same value.
        assert back == [(*row[:-1], None if row[-1] is None else float(str(row[-1]))) for row in original]
        types = dict(table("back").schema(con))
        assert (types["dec"], types["i"], types["ti"], types["tns"], types["d"]) == (
            "DOUBLE",
            "INTEGER",
            "TINYINT",
            "TIMESTAMP_NS",
            "DATE",
        )
        again = sql("SELECT * FROM back").to_numpy(con)
        assert again.keys() == fetched.keys()
        for name, array in fetched.items():
            assert again[name].dtype == array.dtype, name
            assert again[name].tolist() == array.tolist(), name

    def test_a_date_comes_back_as_a_date(self, con: duckdb.frame.Connection) -> None:
        fetched = sql("SELECT DATE '2020-01-02' AS d").to_numpy(con)
        assert typed(con, fetched) == [(datetime.date(2020, 1, 2), "DATE")]


class Answering(NumpyScanSource):
    """A source read by the numpy scan that describes one column as `type_text` and answers `columns()` with `plan`."""

    def __init__(self, type_text: str, plan: tuple[str, str, object, object | None]) -> None:
        super().__init__(None)
        self.type_text = type_text
        self.plan = plan

    def rows(self) -> int:
        data = self.plan[2]
        return len(data) if isinstance(data, np.ndarray) else 0

    def describe(self) -> list[tuple[str, object]]:
        return [("a", self.type_text)]

    def columns(self, columns: Sequence[int] | None) -> ColumnsAnswer:
        return [("a", *self.plan)]


class TestNumpyScanContract:
    """The numpy scan checks each `columns()` answer against the types the query was bound with, before reading it."""

    @pytest.mark.parametrize(
        ("type_text", "plan", "message"),
        [
            # A buffer of the right width and the wrong element class, the case width alone cannot catch.
            ("BIGINT", ("fixed", "BIGINT", np.array([1.0]), None), "format 'd', 8 bytes wide"),
            ("UBIGINT", ("fixed", "UBIGINT", np.array([1]), None), "does not read into UBIGINT"),
            # Integers read as Python object pointers, which is how a pandas sparse column once crashed the scan.
            ("BIGINT", ("objects", "BIGINT", np.array([1, 2]), None), "does not read into BIGINT"),
            ("VARCHAR", ("text", "VARCHAR", np.array([1, 2]), None), "does not read into VARCHAR"),
            ("BIGINT", ("timestamp:us", "BIGINT", np.array([1]), None), "does not read into BIGINT"),
            ("BIGINT", ("interval:us", "BIGINT", np.array([1]), None), "does not read into BIGINT"),
            ("BIGINT", ("enum", "BIGINT", np.array([0], dtype=np.int8), None), "does not read into BIGINT"),
            ("ENUM('x')", ("enum", "ENUM('x')", np.array([0], dtype=np.int64), None), "8 bytes wide"),
            ("BLOB", ("ucs4", "BLOB", np.array(["x"]), None), "does not read into BLOB"),
            ("VARCHAR", ("bytes", "VARCHAR", np.array([b"x"]), None), "does not read into VARCHAR"),
            ("TINYINT", ("fixed", "TINYINT", np.array([1], dtype=np.int8), np.array([0])), "where a mask holds"),
            ("BIGINT", ("fixed", "BIGINT", np.array([[1]]), None), "not one-dimensional"),
        ],
    )
    def test_a_buffer_the_encoding_cannot_read_into_the_type_is_refused(
        self, con: duckdb.frame.Connection, type_text: str, plan: tuple[str, str, object, object | None], message: str
    ) -> None:
        con._register_source("t", Answering(type_text, plan))
        with pytest.raises(exceptions.InvalidInputError, match=re.escape(message)):
            rows(con, "SELECT * FROM t")

    @pytest.mark.parametrize("mask_dtype", [np.bool_, np.uint8, np.int8])
    def test_a_mask_of_one_byte_elements_marks_missing_values(
        self, con: duckdb.frame.Connection, mask_dtype: type
    ) -> None:
        mask: np.ndarray[Any, Any] = np.array([0, 1, 0], dtype=mask_dtype)
        con._register_source("t", Answering("TINYINT", ("fixed", "TINYINT", np.array([1, 2, 3], dtype=np.int8), mask)))
        assert rows(con, "SELECT * FROM t") == [(1,), (None,), (3,)]

    def test_a_refused_second_column_releases_the_first_columns_array(self, con: duckdb.frame.Connection) -> None:
        good = np.arange(10)

        class SecondRefused(NumpyScanSource):
            def rows(self) -> int:
                return len(good)

            def describe(self) -> list[tuple[str, object]]:
                return [("a", "BIGINT"), ("b", "BIGINT")]

            def columns(self, columns: Sequence[int] | None) -> ColumnsAnswer:
                return [("a", "fixed", "BIGINT", good, None), ("b", "fixed", "BIGINT", good.astype(float), None)]

        con._register_source("t", SecondRefused(None))
        before = sys.getrefcount(good)
        for _ in range(3):
            with pytest.raises(exceptions.InvalidInputError, match="for 'b' with a data array of elements"):
                rows(con, "SELECT * FROM t")
        assert sys.getrefcount(good) == before

    def test_exporters_whose_views_point_into_themselves_read_in_every_column(
        self, con: duckdb.frame.Connection
    ) -> None:
        import array

        # bytes and bytearray point a view's shape and strides into the view itself, so the view must never move.
        columns = {
            "b": ("UTINYINT", bytes([1, 2, 3, 4, 5])),
            "a": ("UTINYINT", bytearray([6, 7, 8, 9, 10])),
            "q": ("BIGINT", array.array("q", [11, 12, 13, 14, 15])),
            "n": ("BIGINT", np.arange(16, 21)),
        }

        class Exporters(NumpyScanSource):
            def rows(self) -> int:
                return 5

            def describe(self) -> list[tuple[str, object]]:
                return [(name, type_text) for name, (type_text, _) in columns.items()]

            def columns(self, requested: Sequence[int] | None) -> ColumnsAnswer:
                answer: ColumnsAnswer = [
                    (name, "fixed", type_text, data, None) for name, (type_text, data) in columns.items()
                ]
                return answer if requested is None else [answer[i] for i in requested]

        con._register_source("t", Exporters(None))
        assert rows(con, "SELECT * FROM t") == [(1 + i, 6 + i, 11 + i, 16 + i) for i in range(5)]
        assert rows(con, "SELECT q, b FROM t") == [(11 + i, 1 + i) for i in range(5)]

    def test_a_type_other_than_the_bound_one_is_refused(self, con: duckdb.frame.Connection) -> None:
        con._register_source("t", Answering("BIGINT", ("fixed", "DOUBLE", np.array([1.0]), None)))
        with pytest.raises(exceptions.InvalidInputError, match="with the type DOUBLE, but the query was bound when"):
            rows(con, "SELECT * FROM t")

    @pytest.mark.parametrize(
        ("bound", "answered"),
        [("TIMESTAMP_S", "TIMESTAMP_NS"), ("DECIMAL(10,2)", "DECIMAL(10,3)"), ("ENUM('x', 'y')", "ENUM('y', 'x')")],
    )
    def test_a_type_differing_only_in_its_parameters_is_refused(
        self, con: duckdb.frame.Connection, bound: str, answered: str
    ) -> None:
        con._register_source("t", Answering(bound, ("objects", answered, np.array([None], dtype=object), None)))
        with pytest.raises(exceptions.InvalidInputError, match="but the query was bound when it was"):
            rows(con, "SELECT * FROM t")

    @pytest.mark.parametrize(
        "encoding",
        [
            "interval:",
            "interval:2",
            "interval:02ns",
            "interval:-3s",
            "interval: s",
            "interval:1e3ns",
            "interval:nsx",
            "interval:msx",
            "interval:ps",
            "interval:M",
            "interval:Y",
            "interval:2147483648ns",
            "timestamp",
            "timestamp:",
            "timestamp:M",
        ],
    )
    def test_a_malformed_unit_is_refused(self, con: duckdb.frame.Connection, encoding: str) -> None:
        con._register_source("t", Answering("INTERVAL", (encoding, "INTERVAL", np.array([1]), None)))
        with pytest.raises(exceptions.InvalidInputError, match=re.escape(f"the unknown encoding '{encoding}'")):
            rows(con, "SELECT * FROM t")

    @pytest.mark.parametrize(
        ("type_text", "encoding", "expected"),
        [
            ("INTERVAL", "interval:m", datetime.timedelta(minutes=3)),
            ("INTERVAL", "interval:h", datetime.timedelta(hours=3)),
            ("INTERVAL", "interval:2147483647ns", datetime.timedelta(microseconds=6442450)),
            ("TIMESTAMP_S", "timestamp:D", datetime.datetime(1970, 1, 4)),
            ("TIMESTAMP_S", "timestamp:0s", datetime.datetime(1970, 1, 1)),
            ("TIMESTAMPTZ_NS", "timestamp:us", datetime.datetime(1970, 1, 1, microsecond=3, tzinfo=datetime.UTC)),
            ("TIMESTAMPTZ_NS", "timestamp:s", datetime.datetime(1970, 1, 1, 0, 0, 3, tzinfo=datetime.UTC)),
            ("TIMESTAMPTZ_NS", "timestamp:D", datetime.datetime(1970, 1, 4, tzinfo=datetime.UTC)),
            (
                "TIMESTAMPTZ_NS",
                "timestamp:1500us",
                datetime.datetime(1970, 1, 1, microsecond=4500, tzinfo=datetime.UTC),
            ),
        ],
    )
    def test_a_unit_in_numpys_spelling_is_read(
        self, con: duckdb.frame.Connection, type_text: str, encoding: str, expected: object
    ) -> None:
        con._register_source("t", Answering(type_text, (encoding, type_text, np.array([3]), None)))
        assert rows(con, "SELECT * FROM t") == [(expected,)]

    @pytest.mark.parametrize(
        ("type_text", "encoding", "message"),
        [
            ("TIMESTAMP_S", "timestamp:ns", "whose unit is finer than TIMESTAMP_S"),
            ("TIMESTAMP", "timestamp:1500ns", "whose unit is finer than TIMESTAMP"),
            ("TIMESTAMP WITH TIME ZONE", "timestamp:ns", "whose unit is finer than TIMESTAMP WITH TIME ZONE"),
        ],
    )
    def test_a_unit_the_type_cannot_take_is_refused(
        self, con: duckdb.frame.Connection, type_text: str, encoding: str, message: str
    ) -> None:
        con._register_source("t", Answering(type_text, (encoding, type_text, np.array([3]), None)))
        with pytest.raises(exceptions.InvalidInputError, match=re.escape(message)):
            rows(con, "SELECT * FROM t")

    def test_a_count_past_a_nanosecond_zoned_timestamp_is_refused(self, con: duckdb.frame.Connection) -> None:
        # A million days is past 2262, where nanoseconds since the epoch leave int64.
        con._register_source(
            "t", Answering("TIMESTAMPTZ_NS", ("timestamp:D", "TIMESTAMPTZ_NS", np.array([10**6]), None))
        )
        with pytest.raises(exceptions.InvalidInputError, match="holds a value at row 0 that overflows its engine type"):
            rows(con, "SELECT * FROM t")

    def test_a_category_code_without_a_label_is_refused_at_its_row(self, con: duckdb.frame.Connection) -> None:
        con._register_source("t", Answering("ENUM('x')", ("enum", "ENUM('x')", np.array([0, 1], dtype=np.int8), None)))
        with pytest.raises(exceptions.InvalidInputError, match="category code at row 1 that its ENUM has no label"):
            rows(con, "SELECT * FROM t")

    def test_an_answer_that_fits_is_read(self, con: duckdb.frame.Connection) -> None:
        mask = np.array([False, True])
        con._register_source(
            "t", Answering("ENUM('x', 'y')", ("enum", "ENUM('x', 'y')", np.array([1, 0], np.int16), mask))
        )
        assert rows(con, "SELECT * FROM t") == [("y",), (None,)]

    def test_a_dtype_changed_between_binding_and_scanning_is_refused(self, con: duckdb.frame.Connection) -> None:
        class Changing(NumpySource):
            # Stands in for another thread replacing the column after the query was bound; this once overflowed
            # the scan's TINYINT vector with eight-byte timestamps.
            def columns(self, columns: Sequence[int] | None) -> ColumnsAnswer:
                self.obj["a"] = np.arange(2048).astype("datetime64[ns]")
                return super().columns(columns)

        con._register_source("t", Changing({"a": np.zeros(2048, dtype=np.int8)}))
        with pytest.raises(exceptions.InvalidInputError, match="with the type TIMESTAMP_NS, but the query was bound"):
            rows(con, "SELECT * FROM t")


class TestPandasColumnsReadAsTheirArrays:
    """A pandas column is read by the dtype of the array holding its values, not by the pandas dtype around it."""

    def test_a_sparse_column_reads_its_values(self, con: duckdb.frame.Connection) -> None:
        # This once read the integers as Python object pointers and crashed the process.
        con.register("t", pd.DataFrame({"s": pd.arrays.SparseArray([1, 0, 2])}))
        assert rows(con, "SELECT s, typeof(s) FROM t") == [(1, "BIGINT"), (0, "BIGINT"), (2, "BIGINT")]

    def test_a_complex_column_is_refused_by_name_when_a_query_is_bound(self, con: duckdb.frame.Connection) -> None:
        # It was once described as VARCHAR, a type the scan could never deliver.
        con.register("t", pd.DataFrame({"a": [1, 2], "c": [1j, 2j]}))
        with pytest.raises(exceptions.InvalidInputError, match="column 'c' has the numpy dtype complex128"):
            rows(con, "SELECT a FROM t")


def objects(*values: object) -> np.ndarray[Any, Any]:
    """An object array holding exactly `values`, numpy scalars kept as they are."""
    array = np.empty(len(values), dtype=object)
    for i, value in enumerate(values):
        array[i] = value
    return array


class TestTemporalScaling:
    """Counts in any fixed unit and step read through one conversion, in dense arrays and in object columns alike."""

    @pytest.mark.parametrize("raw", [5 * 10**18, -5 * 10**18])
    def test_a_stepped_nanosecond_duration_reads_dense_and_as_an_object(
        self, con: duckdb.frame.Connection, raw: int
    ) -> None:
        # The dense array was once refused: it was multiplied in nanoseconds before being divided to microseconds.
        expected = [(datetime.timedelta(microseconds=raw // 500),)]
        con.register("dense", np.array([raw], dtype="timedelta64[2ns]"))
        con.register("objects", objects(np.array(raw, dtype="timedelta64[2ns]")[()]))
        assert rows(con, "SELECT * FROM dense") == rows(con, "SELECT * FROM objects") == expected

    @pytest.mark.parametrize("sign", [1, -1])
    def test_an_attosecond_step_whose_remainder_product_passes_int64_reads(
        self, con: duckdb.frame.Connection, sign: int
    ) -> None:
        # 999999999999 * 2147483647 as, about 2.1 * 10^21 before dividing by 10^12, was once refused.
        con.register("t", objects(np.array(sign * 999999999999, dtype="timedelta64[2147483647as]")[()]))
        assert rows(con, "SELECT * FROM t") == [(datetime.timedelta(microseconds=sign * 2147483646),)]

    def test_a_unit_too_large_for_any_nonzero_count_still_reads_zero_and_missing_values(
        self, con: duckdb.frame.Connection
    ) -> None:
        unit = "timedelta64[2147483647W]"
        con.register("zero", np.array([0], dtype=unit))
        con.register("nat", np.array(["NaT"], dtype=unit))
        con.register("masked", np.ma.array(np.array([1], dtype=unit), mask=[True]))
        con.register("empty", np.array([], dtype=unit))
        assert rows(con, "SELECT * FROM zero") == [(datetime.timedelta(0),)]
        assert rows(con, "SELECT * FROM nat") == [(None,)]
        assert rows(con, "SELECT * FROM masked") == [(None,)]
        assert rows(con, "SELECT * FROM empty") == []
        con.register("one", np.array([1], dtype=unit))
        with pytest.raises(exceptions.InvalidInputError, match="holds a value at row 0 that overflows its engine type"):
            rows(con, "SELECT * FROM one")

    @pytest.mark.parametrize(
        ("raw", "micros"),
        [
            (-(2**62), -(2**63)),
            (-(2**62) + 1, -(2**63) + 2),
            (2**62 - 1, 2**63 - 2),
        ],
    )
    def test_the_signed_limits_of_a_duration_are_read(
        self, con: duckdb.frame.Connection, raw: int, micros: int
    ) -> None:
        # 2000 ns is two microseconds, so the negative end reaches -2^63, one further than the positive end.
        con.register("dense", np.array([raw], dtype="timedelta64[2000ns]"))
        con.register("objects", objects(np.array(raw, dtype="timedelta64[2000ns]")[()]))
        expected = [(datetime.timedelta(microseconds=micros),)]
        assert rows(con, "SELECT * FROM dense") == rows(con, "SELECT * FROM objects") == expected

    @pytest.mark.parametrize("raw", [-(2**62) - 1, 2**62])
    def test_one_past_either_signed_limit_is_refused(self, con: duckdb.frame.Connection, raw: int) -> None:
        con.register("dense", np.array([0, raw], dtype="timedelta64[2000ns]"))
        with pytest.raises(exceptions.InvalidInputError, match="holds a value at row 1 that overflows its engine type"):
            rows(con, "SELECT * FROM dense")
        con.register("objects", objects(np.array(raw, dtype="timedelta64[2000ns]")[()]))
        with pytest.raises(exceptions.InvalidInputError, match="cannot hold exactly"):
            rows(con, "SELECT * FROM objects")

    @pytest.mark.parametrize("raw", [-1, -999, 1, 999])
    def test_a_duration_under_a_microsecond_truncates_to_zero(self, con: duckdb.frame.Connection, raw: int) -> None:
        con.register("t", np.array([raw, raw - 1000 if raw < 0 else raw + 1000], dtype="timedelta64[ns]"))
        whole = -1 if raw < 0 else 1
        assert rows(con, "SELECT * FROM t") == [
            (datetime.timedelta(0),),
            (datetime.timedelta(microseconds=whole),),
        ]

    @pytest.mark.parametrize(("raw", "micros"), [(-1, 0), (-333, 0), (-334, -1), (333, 0), (334, 1)])
    def test_a_stepped_duration_under_a_microsecond_truncates_to_zero_too(
        self, con: duckdb.frame.Connection, raw: int, micros: int
    ) -> None:
        con.register("dense", np.array([raw], dtype="timedelta64[3ns]"))
        con.register("objects", objects(np.array(raw, dtype="timedelta64[3ns]")[()]))
        expected = [(datetime.timedelta(microseconds=micros),)]
        assert rows(con, "SELECT * FROM dense") == rows(con, "SELECT * FROM objects") == expected

    def test_a_zoned_timestamp_under_a_microsecond_before_the_epoch_truncates_to_the_epoch(
        self, con: duckdb.frame.Connection
    ) -> None:
        stamps = pd.Series(np.array([-1, -1001], dtype="datetime64[ns]")).dt.tz_localize("UTC")
        con.register("t", pd.DataFrame({"t": stamps}))
        epoch = datetime.datetime(1970, 1, 1, tzinfo=datetime.UTC)
        assert rows(con, "SELECT * FROM t") == [(epoch,), (epoch - datetime.timedelta(microseconds=1),)]

    @pytest.mark.parametrize(
        ("dtype", "zero"),
        [("timedelta64[0ns]", datetime.timedelta(0)), ("datetime64[0s]", datetime.datetime(1970, 1, 1))],
    )
    def test_a_zero_step_reads_as_it_always_has(self, con: duckdb.frame.Connection, dtype: str, zero: object) -> None:
        data = np.ma.array(np.array([5, 7, 9, -5], dtype=dtype), mask=[False, False, True, False])
        data[1] = np.array("NaT", dtype=dtype)
        con.register("t", data)
        assert rows(con, "SELECT * FROM t") == [(zero,), (None,), (None,), (zero,)]

    @pytest.mark.parametrize("dtype", ["datetime64[h]", "datetime64[250ms]", "timedelta64[2ns]", "timedelta64[D]"])
    def test_a_converted_unit_is_read_in_place_from_a_strided_view(
        self, con: duckdb.frame.Connection, dtype: str
    ) -> None:
        base = np.arange(40, dtype=np.int64).view(dtype)
        view = base[::-3]
        [(_, _, _, data, _)] = NumpySource(view).columns(None)
        assert isinstance(data, np.ndarray)
        assert np.shares_memory(data, base)
        con.register("view", view)
        con.register("copy", view.copy())
        assert rows(con, "SELECT * FROM view") == rows(con, "SELECT * FROM copy")

    @pytest.mark.parametrize(
        ("encoding", "type_text", "read"),
        [
            (
                "timestamp:h",
                "TIMESTAMP_S",
                lambda hours: datetime.datetime(1970, 1, 1) + datetime.timedelta(hours=hours),
            ),
            ("interval:h", "INTERVAL", lambda hours: datetime.timedelta(hours=hours)),
        ],
    )
    def test_a_reversed_count_buffer_with_a_mask_of_another_stride(
        self, con: duckdb.frame.Connection, encoding: str, type_text: str, read: Callable[[int], object]
    ) -> None:
        # The data steps back eight bytes a row and the mask forward three; the masked count would overflow.
        counts = np.array([1, 2**62, 3, 4, 5, 6], dtype=np.int64)[::-1]
        mask = np.zeros(18, dtype=bool)[::3]
        mask[4] = True
        con._register_source("masked", Answering(type_text, (encoding, type_text, counts, mask)))
        assert rows(con, "SELECT * FROM masked") == [
            (read(6),),
            (read(5),),
            (read(4),),
            (read(3),),
            (None,),
            (read(1),),
        ]
        con._register_source("unmasked", Answering(type_text, (encoding, type_text, counts, None)))
        with pytest.raises(exceptions.InvalidInputError, match="holds a value at row 4 that overflows its engine type"):
            rows(con, "SELECT * FROM unmasked")

    def test_an_object_datetime_in_picoseconds_reads_only_when_exact(self, con: duckdb.frame.Connection) -> None:
        con.register("exact", objects(np.array(10**16, dtype="datetime64[1000ps]")[()]))
        assert rows(con, "SELECT column0, typeof(column0) FROM exact") == [
            (datetime.datetime(1970, 1, 1) + datetime.timedelta(microseconds=10**13), "TIMESTAMP_NS")
        ]
        con.register("inexact", objects(np.array(1, dtype="datetime64[ps]")[()]))
        with pytest.raises(exceptions.InvalidInputError, match="cannot hold exactly"):
            rows(con, "SELECT * FROM inexact")

    @pytest.mark.requires("pyarrow")
    def test_nanosecond_timestamps_at_the_int64_ends_convert_to_python(self, con: duckdb.frame.Connection) -> None:
        # The lowest count arrives through Arrow; the SQL constructor refuses it.
        con.register("t", pa.table({"t": pa.array([-(2**63), -1, 2**63 - 2], type=pa.timestamp("ns"))}))
        assert rows(con, "SELECT * FROM t") == [
            (datetime.datetime(1677, 9, 21, 0, 12, 43, 145225),),
            (datetime.datetime(1970, 1, 1),),
            (datetime.datetime(2262, 4, 11, 23, 47, 16, 854775),),
        ]


#: A column of each value family the object sample tells apart, and the engine type it reads as.
OBJECT_FAMILIES: dict[str, tuple[list[object], str]] = {
    "bool": ([True, False], "BOOLEAN"),
    "int": ([1, -2], "BIGINT"),
    "int at 2**63": ([1, 2**63], "HUGEINT"),
    "int at -2**63-1": ([1, -(2**63) - 1], "HUGEINT"),
    "float": ([1.5, 2.0], "DOUBLE"),
    "int and float": ([1, 2.5], "DOUBLE"),
    "numpy int64 scalars": ([np.int64(1), np.int64(2)], "BIGINT"),
    "numpy float32 scalars": ([np.float32(1.5)], "DOUBLE"),
    "numpy float16 scalars": ([np.float16(1.5)], "DOUBLE"),
    "decimal": ([decimal.Decimal("1.5"), decimal.Decimal("22.25")], "DECIMAL(38,2)"),
    "decimal and int": ([decimal.Decimal("1.5"), 2], "VARCHAR"),
    "decimal exponent": ([decimal.Decimal("1E+3"), decimal.Decimal("1")], "DECIMAL(38,0)"),
    "date": ([datetime.date(2024, 1, 2)], "DATE"),
    "date and datetime": ([datetime.date(2024, 1, 2), datetime.datetime(2024, 1, 3, 4)], "TIMESTAMP"),
    "naive datetime": ([datetime.datetime(2024, 1, 2, 3)], "TIMESTAMP"),
    "naive datetime far": ([datetime.datetime(3000, 1, 2, 3)], "TIMESTAMP"),
    "aware datetime": ([datetime.datetime(2024, 1, 2, 3, tzinfo=datetime.UTC)], "TIMESTAMP WITH TIME ZONE"),
    "aware other zone": (
        [datetime.datetime(2024, 1, 2, 3, tzinfo=datetime.timezone(datetime.timedelta(hours=2)))],
        "TIMESTAMP WITH TIME ZONE",
    ),
    "datetime64[ns] and datetime": (
        [np.datetime64("2024-01-02T03:04:05.123456789", "ns"), datetime.datetime(2024, 1, 1)],
        "TIMESTAMP_NS",
    ),
    "datetime64[ns] and date": (
        [np.datetime64("2024-01-02T03:04:05.123456789", "ns"), datetime.date(2024, 1, 1)],
        "TIMESTAMP_NS",
    ),
    "datetime64[s]": ([np.datetime64("2024-01-02T03:04:05", "s")], "TIMESTAMP"),
    "pd.Timestamp": ([pd.Timestamp("2024-01-02 03:04:05.5")], "TIMESTAMP"),
    "time": ([datetime.time(1, 2, 3, 4)], "TIME"),
    "timedelta": ([datetime.timedelta(days=1, microseconds=5)], "INTERVAL"),
    "timedelta64[ns]": ([np.timedelta64(1500, "ns")], "INTERVAL"),
    "pd.Timedelta ns": ([pd.Timedelta(1500, "ns")], "INTERVAL"),
    "bytes": ([b"ab", b"c"], "BLOB"),
    "bytearray": ([bytearray(b"ab")], "VARCHAR"),
    "uuid": ([uuid.UUID(int=5)], "UUID"),
    "str and int": (["a", 1], "VARCHAR"),
    "list": ([[1, 2]], "VARCHAR"),
    "dict": ([{"a": 1}], "VARCHAR"),
}


@pytest.mark.requires("pyarrow")
class TestPandasObjectPolicy:
    """A column of Python objects is classified and converted the same way whatever the columns beside it."""

    @pytest.mark.parametrize(
        ("values", "expected_type", "expected"),
        [
            (["s", 7] + ["s"] * 1999, "VARCHAR", [("7",)]),
            ([None] * 2001, "VARCHAR", []),
            ([None, 7] + [None] * 1999, "BIGINT", [(7,)]),
            (["s", b"ab"] + ["s"] * 1999, "VARCHAR", [("b'ab'",)]),
        ],
        ids=["later integer", "all missing", "value outside the sample", "later bytes"],
    )
    def test_the_same_column_reads_the_same_beside_an_arrow_column(
        self, con: duckdb.frame.Connection, values: list[object], expected_type: str, expected: list[tuple[object]]
    ) -> None:
        frame = pd.DataFrame({"v": pd.Series(values, dtype=object)})
        alone, beside = alone_and_beside(con, frame, "SELECT v FROM {} WHERE v IS NOT NULL AND v::VARCHAR <> 's'")
        assert alone == beside == expected
        assert dict(table("alone").schema(con))["v"] == dict(table("beside").schema(con))["v"] == expected_type

    @pytest.mark.parametrize(("values", "expected_type"), OBJECT_FAMILIES.values(), ids=list(OBJECT_FAMILIES))
    def test_every_family_reads_the_same_beside_an_arrow_column(
        self, con: duckdb.frame.Connection, values: list[object], expected_type: str
    ) -> None:
        frame = pd.DataFrame({"v": pd.Series(values, dtype=object)})
        alone, beside = alone_and_beside(con, frame, "SELECT v, typeof(v) FROM {}")
        assert alone == beside
        assert {row[1] for row in alone} == {expected_type}

    @pytest.mark.parametrize(
        ("later", "expected"),
        [
            ([None, float("nan")], [(None,)] * 2),
            (
                [None, float("nan"), np.float32("nan"), pd.NA, pd.NaT, np.datetime64("NaT", "ns")],
                [(None,)] * 6,
            ),
            ([decimal.Decimal("NaN")] * 2, [("NaN",)] * 2),
        ],
        ids=["None and NaN", "every missing marker", "Decimal NaN"],
    )
    def test_later_missing_markers_or_decimal_nan_in_a_text_column(
        self, con: duckdb.frame.Connection, later: list[object], expected: list[tuple[object]]
    ) -> None:
        first: list[object] = ["a", np.float32("nan")] + ["a"] * 5000
        frame = pd.DataFrame({"v": pd.Series([*first, *later], dtype=object)})
        query = "SELECT v FROM {} WHERE v IS NULL OR v <> 'a' ORDER BY v NULLS FIRST"
        alone, beside = alone_and_beside(con, frame, query)
        assert alone == beside == [(None,), *expected]

    @pytest.mark.parametrize("value", [7, 2**60 + 1], ids=["small", "past exact doubles"])
    @pytest.mark.parametrize(
        "marker", [np.float32("nan"), np.float16("nan"), np.datetime64("NaT", "ns")], ids=["float32", "float16", "NaT"]
    )
    def test_a_numpy_missing_marker_does_not_change_a_typed_column(
        self, con: duckdb.frame.Connection, marker: object, value: int
    ) -> None:
        frame = pd.DataFrame({"v": pd.Series([marker] + [value] * 2000, dtype=object)})
        alone, beside = alone_and_beside(con, frame, "SELECT typeof(v), count(v), max(v) FROM {} GROUP BY ALL")
        assert alone == beside == [("BIGINT", 2000, value)]

    def test_float32_objects_read_as_double(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame({"v": pd.Series([np.float32("nan")] + [np.float32(1.5)] * 2000, dtype=object)})
        alone, beside = alone_and_beside(con, frame, "SELECT typeof(v), count(v), max(v) FROM {} GROUP BY ALL")
        assert alone == beside == [("DOUBLE", 2000, 1.5)]

    def test_a_str_subclass_is_read_as_the_string_it_holds(self, con: duckdb.frame.Connection) -> None:
        class Shouting(str):
            def __str__(self) -> str:
                return "SHOUTING"

        frame = pd.DataFrame({"v": pd.Series(["a", 7, Shouting("hi")] + ["a"] * 1998, dtype=object)})
        alone, beside = alone_and_beside(con, frame, "SELECT v FROM {} WHERE v <> 'a' ORDER BY v")
        assert alone == beside == [("7",), ("hi",)]

    def test_an_empty_frame_reads_as_an_empty_text_column(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame({"v": pd.Series([], dtype=object)})
        alone, beside = alone_and_beside(con, frame, "SELECT v FROM {}")
        assert alone == beside == []
        assert dict(table("alone").schema(con))["v"] == dict(table("beside").schema(con))["v"] == "VARCHAR"

    def test_repeated_index_labels_do_not_hide_a_value_outside_the_sample(self, con: duckdb.frame.Connection) -> None:
        values: list[object] = [None] * 3000
        values[1] = 7
        frame = pd.DataFrame({"v": pd.Series(values, dtype=object, index=[0] * 3000)})
        alone, beside = alone_and_beside(con, frame, "SELECT v, typeof(v) FROM {} WHERE v IS NOT NULL")
        assert alone == beside == [(7, "BIGINT")]

    def test_a_frame_changed_between_queries_is_decided_again(self, con: duckdb.frame.Connection) -> None:
        frame = beside_an_arrow_column(pd.DataFrame({"v": pd.Series([None] * 2001, dtype=object)}))
        con.register("t", frame)
        assert table("t").schema(con)[0] == ("v", "VARCHAR")
        frame.loc[1, "v"] = 7
        assert table("t").schema(con)[0] == ("v", "BIGINT")
        assert rows(con, "SELECT v FROM t WHERE v IS NOT NULL") == [(7,)]

    def test_a_query_skipping_a_column_that_cannot_convert_still_reads(self, con: duckdb.frame.Connection) -> None:
        # The fraction is outside the sample, and BIGINT cannot hold it exactly.
        numbers: list[object] = [1] * 5000
        numbers[4999] = 1.5
        frame = beside_an_arrow_column(
            pd.DataFrame({"n": pd.Series(numbers, dtype=object), "s": ["a"] * len(numbers)}).astype({"s": object})
        )
        con.register("t", frame)
        assert rows(con, "SELECT count(s) FROM t") == [(len(numbers),)]
        with pytest.raises(exceptions.InvalidInputError, match="cannot hold exactly"):
            rows(con, "SELECT sum(n) FROM t")


@pytest.mark.requires("pyarrow")
class TestPandasCategoricalsKeepTheirOwnRules:
    """Categoricals and extension arrays are not decided by the object sample; they read by their own rules."""

    def test_a_categorical_of_mixed_objects_is_text(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame({"v": pd.Series(pd.Categorical([1, "a", 1]))})
        alone, beside = alone_and_beside(con, frame, "SELECT v, typeof(v) FROM {}")
        assert alone == beside == [("1", "VARCHAR"), ("a", "VARCHAR"), ("1", "VARCHAR")]

    def test_a_categorical_of_strings_is_an_enum(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame({"v": pd.Series(pd.Categorical(["a", None, "b"]))})
        alone, beside = alone_and_beside(con, frame, "SELECT v, typeof(v) FROM {}")
        assert alone == beside == [("a", "ENUM('a', 'b')"), (None, "ENUM('a', 'b')"), ("b", "ENUM('a', 'b')")]

    def test_an_unused_category_of_another_type_does_not_change_the_type(self, con: duckdb.frame.Connection) -> None:
        frame = pd.DataFrame({"v": pd.Series(pd.Categorical([1, 2], categories=[1, 2, "x"]))})
        alone, beside = alone_and_beside(con, frame, "SELECT v, typeof(v) FROM {}")
        assert alone == beside == [(1, "BIGINT"), (2, "BIGINT")]

    @pytest.mark.parametrize("dtype", ["string[python]", "string[pyarrow]"])
    def test_a_pandas_string_column_reads_as_text(self, con: duckdb.frame.Connection, dtype: str) -> None:
        frame = pd.DataFrame({"v": pd.Series(["a", None, "c"], dtype=dtype)})
        alone, beside = alone_and_beside(con, frame, "SELECT v, typeof(v) FROM {}")
        assert alone == beside == [("a", "VARCHAR"), (None, "VARCHAR"), ("c", "VARCHAR")]
