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
    from collections.abc import Sequence
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


class Pushing(DatasetSource):
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
        con.register("plain", ExportingSource(ds.dataset(typed_dir, partitioning="hive").to_table()))
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
        con.register("plain", ExportingSource(pl.scan_parquet(typed_dir, hive_partitioning=True).collect()))
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
        later: list[object] = list(range(PandasSource.SLICE_ROWS + 10))
        later[PandasSource.SLICE_ROWS + 1] = 4.2
        con.register("late", pd.DataFrame({"a": pd.Series(later, dtype=object)}))
        with pytest.raises(exceptions.InvalidInputError, match=f"after row {PandasSource.SLICE_ROWS}"):
            rows(con, "SELECT a FROM late WHERE a = 4")

    def test_a_later_slice_typed_differently_on_its_own_fails_too(self, con: duckdb.frame.Connection) -> None:
        many = PandasSource.SLICE_ROWS + 4
        words: list[object] = ["a"] * many
        words[-4:] = [111, 222, 333, 444]
        con.register("words", pd.DataFrame({"s": pd.Series(words, dtype=object)}))
        assert table("words").schema(con) == [("s", "VARCHAR")]
        with pytest.raises(exceptions.InvalidInputError, match="registered as 'words' failed"):
            rows(con, "SELECT count(*) FROM words")
        moments: list[object] = [datetime.datetime(2024, 1, 1, 12)] * many
        moments[-2:] = [datetime.date(2024, 1, 2), datetime.datetime(2024, 1, 3, 9, 15)]
        con.register("moments", pd.DataFrame({"t": pd.Series(moments, dtype=object)}))
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
