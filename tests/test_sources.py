"""The object families that register as tables: pyarrow datasets and scanners, polars frames, pandas frames."""

from __future__ import annotations

import datetime
import decimal
from typing import TYPE_CHECKING

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

import duckdb
from duckdb import exceptions
from duckdb._sources import (
    ArraySource,
    DatasetSource,
    ExportingSource,
    LazyFrameSource,
    PandasSource,
    PolarsFrameSource,
    ScannerSource,
    adapt,
)
from duckdb.frame import col, sql, table

if TYPE_CHECKING:
    from pathlib import Path


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


class TestClassification:
    def test_each_family_gets_its_adapter(self, parquet_dir: Path) -> None:
        dataset = ds.dataset(parquet_dir)
        assert isinstance(adapt(dataset), DatasetSource)
        assert isinstance(adapt(dataset.scanner()), ScannerSource)
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
        capsule, projected = source.stream([1])
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

        assert isinstance(adapt(Mine(ds.dataset(parquet_dir).scanner())), ScannerSource)

    def test_a_class_named_like_a_reader_still_needs_the_export(self) -> None:
        class RecordBatchReader:
            pass

        with pytest.raises(TypeError, match="none of these"):
            adapt(RecordBatchReader())

    def test_series_and_fragments_are_not_datasets(self, parquet_dir: Path) -> None:
        for series in (pd.Series([1]), pl.Series([1])):
            source = adapt(series)
            assert isinstance(source, ExportingSource)
            assert not source.one_shot
        assert isinstance(adapt(pa.array([1])), ArraySource)
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

        assert isinstance(adapt(Mine(ds.dataset(parquet_dir))), DatasetSource)


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
        capsule, projected = adapt(ds.dataset(tmp_path, partitioning="hive")).stream([1])
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
        capsule, projected = adapt(lazy).stream([1])
        assert projected
        assert columns_of(capsule) == ["s"]

    def test_pandas_converts_only_the_requested_columns(self) -> None:
        frame = pd.DataFrame({"i": [1, 2], "s": ["a", "b"], "o": pd.Series([object(), object()], dtype=object)})
        source = adapt(frame)
        capsule, projected = source.stream([1, 0])
        assert projected
        assert columns_of(capsule) == ["s", "i"]

    def test_tables_and_scanners(self, parquet_dir: Path) -> None:
        table_ = pa.table({"a": [1], "b": [2], "c": [3]})
        capsule, projected = adapt(table_).stream([2, 0])
        assert projected
        assert columns_of(capsule) == ["c", "a"]
        scanner = ds.dataset(parquet_dir).scanner(columns=["s", "n"])
        capsule, projected = adapt(scanner).stream([1])
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
        schema = source._schema(frame)
        assert [batch.num_rows for batch in source._batches(frame, schema)] == [1000] * 10 + [500]
        con.register("df", frame)
        assert rows(con, "SELECT count(*), sum(i), max(s) FROM df") == [(10_500, 55_119_750, "9999")]

    def test_types_come_from_a_sample_and_a_misfit_later_value_fails_the_query(
        self, con: duckdb.frame.Connection
    ) -> None:
        values: list[object] = ["text"] * 5000
        values[4999] = 12
        con.register("df", pd.DataFrame({"o": pd.Series(values, dtype=object)}))
        assert table("df").schema(con) == [("o", "VARCHAR")]
        with pytest.raises(exceptions.InvalidInputError, match="registered as 'df' failed"):
            rows(con, "SELECT count(*) FROM df")

    def test_without_pyarrow_registration_says_so(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        monkeypatch.setitem(sys.modules, "pyarrow", None)
        with pytest.raises(TypeError, match="needs pyarrow"):
            adapt(pd.DataFrame({"a": [1]}))
