"""Columnar egress: engine chunks to numpy and pandas without Arrow.

NULLs are the axis that matters: masked arrays on the numpy path, nullable
dtypes on the pandas path, and plain dtypes wherever no NULL occurs.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

import duckdb
from duckdb import col

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def con() -> Iterator[duckdb.Connection]:
    connection = duckdb.connect()
    yield connection
    connection.close()


class TestToNumpy:
    def test_numeric_dtypes_are_exact(self, con: duckdb.Connection) -> None:
        out = duckdb.sql(
            "SELECT 1::TINYINT AS i8, 2::SMALLINT AS i16, 3::INTEGER AS i32, 4::BIGINT AS i64, "
            "5::UTINYINT AS u8, 6::UBIGINT AS u64, 1.5::FLOAT AS f4, 2.5::DOUBLE AS f8, true AS b"
        ).to_numpy(con)
        dtypes = {name: array.dtype.name for name, array in out.items()}
        assert dtypes == {
            "i8": "int8",
            "i16": "int16",
            "i32": "int32",
            "i64": "int64",
            "u8": "uint8",
            "u64": "uint64",
            "f4": "float32",
            "f8": "float64",
            "b": "bool",
        }
        assert out["i64"][0] == 4

    def test_nulls_come_back_masked(self, con: duckdb.Connection) -> None:
        out = duckdb.sql("SELECT unnest([1, NULL, 3]) AS v").to_numpy(con)
        assert isinstance(out["v"], np.ma.MaskedArray)
        assert out["v"].mask.tolist() == [False, True, False]
        assert out["v"].compressed().tolist() == [1, 3]

    def test_a_clean_column_is_a_plain_ndarray(self, con: duckdb.Connection) -> None:
        out = duckdb.sql("SELECT unnest([1, 2, 3]) AS v").to_numpy(con)
        assert not isinstance(out["v"], np.ma.MaskedArray)
        assert out["v"].tolist() == [1, 2, 3]

    def test_strings_and_nested_go_per_object(self, con: duckdb.Connection) -> None:
        out = duckdb.sql("SELECT 'hi' AS s, [1, 2] AS l, {'a': 1} AS st").to_numpy(con)
        assert out["s"].dtype == object
        assert out["s"][0] == "hi"
        assert out["l"][0] == [1, 2]
        assert out["st"][0] == {"a": 1}

    def test_temporal_units(self, con: duckdb.Connection) -> None:
        out = duckdb.sql(
            "SELECT DATE '2026-09-01' AS d, TIMESTAMP '2026-09-01 12:00:00' AS ts, "
            "TIMESTAMP_NS '2026-09-01 12:00:00.123456789' AS ns, INTERVAL 90 DAY AS iv"
        ).to_numpy(con)
        assert out["d"].dtype == np.dtype("datetime64[us]")
        assert out["ts"].dtype == np.dtype("datetime64[us]")
        assert out["ns"].dtype == np.dtype("datetime64[ns]")
        assert out["iv"].dtype == np.dtype("timedelta64[us]")
        assert out["ts"][0] == np.datetime64("2026-09-01T12:00:00", "us")

    def test_decimal_becomes_float64_at_its_scale(self, con: duckdb.Connection) -> None:
        # The numpy path trades exactness for vectors, as the old client did;
        # exact Decimals come from the row egress.
        out = duckdb.sql("SELECT 1.250::DECIMAL(18,3) AS d, 2.5::DECIMAL(38,1) AS wide").to_numpy(con)
        assert out["d"].dtype == np.dtype("float64")
        assert out["d"][0] == 1.25
        assert out["wide"][0] == 2.5

    def test_enum_becomes_strings_with_none_for_null(self, con: duckdb.Connection) -> None:
        con.run("CREATE TYPE mood AS ENUM ('sad', 'ok', 'happy')")
        out = duckdb.sql("SELECT unnest(['ok', NULL, 'happy']::mood[]) AS m").to_numpy(con)
        column = out["m"]
        assert isinstance(column, np.ma.MaskedArray)
        assert column.mask.tolist() == [False, True, False]
        assert column.data[0] == "ok"
        assert column.data[1] is None
        assert column.data[2] == "happy"

    def test_an_empty_result_keeps_its_dtypes(self, con: duckdb.Connection) -> None:
        out = duckdb.sql("SELECT 1::BIGINT AS v, 'x' AS s WHERE FALSE").to_numpy(con)
        assert out["v"].dtype == np.dtype("int64")
        assert len(out["v"]) == 0
        assert out["s"].dtype == object

    def test_many_chunks_concatenate(self, con: duckdb.Connection) -> None:
        out = duckdb.sql("SELECT range AS v FROM range(10000)").to_numpy(con)
        assert len(out["v"]) == 10000
        assert out["v"][9999] == 9999
        assert out["v"].sum() == 49995000

    def test_parameters_flow(self, con: duckdb.Connection) -> None:
        plan = duckdb.sql("SELECT range AS v FROM range(10)").filter(col("v") < duckdb.param("cap"))
        out = plan.to_numpy(con, parameters={"cap": 3})
        assert out["v"].tolist() == [0, 1, 2]


class TestToPandas:
    def test_nullable_dtypes_only_where_nulls_occur(self, con: duckdb.Connection) -> None:
        frame = duckdb.sql("SELECT unnest([1, NULL]) AS with_null, unnest([1, 2]) AS clean").to_pandas(con)
        assert str(frame["with_null"].dtype) == "Int32"
        assert frame["with_null"][1] is pd.NA
        assert str(frame["clean"].dtype) == "int32"

    def test_null_timestamps_are_nat(self, con: duckdb.Connection) -> None:
        frame = duckdb.sql("SELECT unnest([TIMESTAMP '2026-09-01', NULL]) AS ts").to_pandas(con)
        assert pd.isna(frame["ts"][1])
        assert frame["ts"][0] == pd.Timestamp("2026-09-01")

    def test_timestamptz_is_utc_aware(self, con: duckdb.Connection) -> None:
        frame = duckdb.sql("SELECT TIMESTAMPTZ '2026-09-01 12:00:00+00' AS ts").to_pandas(con)
        assert str(frame["ts"].dtype) == "datetime64[us, UTC]"

    def test_enum_becomes_categorical(self, con: duckdb.Connection) -> None:
        con.run("CREATE TYPE mood AS ENUM ('sad', 'ok', 'happy')")
        frame = duckdb.sql("SELECT unnest(['ok', 'happy']::mood[]) AS m").to_pandas(con)
        assert isinstance(frame["m"].dtype, pd.CategoricalDtype)
        assert list(frame["m"]) == ["ok", "happy"]

    def test_date_as_object(self, con: duckdb.Connection) -> None:
        plan = duckdb.sql("SELECT DATE '2026-09-01' AS d")
        as_datetime = plan.to_pandas(con)
        assert as_datetime["d"].dtype.kind == "M"
        as_object = plan.to_pandas(con, date_as_object=True)
        assert as_object["d"][0] == datetime.date(2026, 9, 1)

    def test_strings_follow_pandas_inference(self, con: duckdb.Connection) -> None:
        # The same shape a hand-built pd.DataFrame gives the same values.
        frame = duckdb.sql("SELECT unnest(['a', NULL]) AS s").to_pandas(con)
        assert frame["s"][0] == "a"
        assert pd.isna(frame["s"][1])
        nested = duckdb.sql("SELECT [1, 2] AS l").to_pandas(con)
        assert nested["l"][0] == [1, 2]


class TestBoundEgress:
    def test_bound_forwards_both(self, con: duckdb.Connection) -> None:
        bound = duckdb.sql("SELECT 1 AS v").on(con)
        assert bound.to_numpy()["v"].tolist() == [1]
        assert bound.to_pandas()["v"].tolist() == [1]


class TestTemporalInfinities:
    def test_infinite_dates_clamp_like_the_row_path(self, con: duckdb.Connection) -> None:
        plan = duckdb.sql("SELECT 'infinity'::DATE AS pos, '-infinity'::DATE AS neg")
        frame = plan.to_pandas(con)
        assert frame["pos"][0].date() == datetime.date.max
        assert frame["neg"][0].date() == datetime.date.min
        out = plan.to_numpy(con)
        assert out["pos"][0] == np.datetime64("9999-12-31", "us")
        assert out["neg"][0] == np.datetime64("0001-01-01", "us")

    def test_infinite_timestamps_agree_with_the_row_path(self, con: duckdb.Connection) -> None:
        plan = duckdb.sql("SELECT 'infinity'::TIMESTAMP AS pos, '-infinity'::TIMESTAMP AS neg")
        frame = plan.to_pandas(con)
        rows = plan.rows(con)
        assert frame["pos"][0].to_pydatetime() == rows[0][0] == datetime.datetime.max
        assert frame["neg"][0].to_pydatetime() == rows[0][1] == datetime.datetime.min

    def test_infinite_coarse_timestamps_clamp_in_their_unit(self, con: duckdb.Connection) -> None:
        frame = duckdb.sql("SELECT 'infinity'::TIMESTAMP_S AS s, 'infinity'::TIMESTAMP_MS AS ms").to_pandas(con)
        assert frame["s"][0].year == 9999
        assert frame["ms"][0].year == 9999

    def test_infinite_timestamptz_stays_aware(self, con: duckdb.Connection) -> None:
        frame = duckdb.sql("SELECT 'infinity'::TIMESTAMPTZ AS ts").to_pandas(con)
        value = frame["ts"][0]
        assert value.tzinfo is not None
        assert value.year == 9999


class TestEmptyResults:
    def test_empty_decimal_and_enum_keep_their_dtypes(self, con: duckdb.Connection) -> None:
        con.run("CREATE TYPE empty_mood AS ENUM ('sad', 'ok')")
        frame = duckdb.sql("SELECT 1.5::DECIMAL(9,2) AS d, 'ok'::empty_mood AS m WHERE FALSE").to_pandas(con)
        assert frame["d"].dtype == np.dtype("float64")
        assert isinstance(frame["m"].dtype, pd.CategoricalDtype)
        assert list(frame["m"].dtype.categories) == ["sad", "ok"]

    def test_empty_timestamptz_is_utc_aware(self, con: duckdb.Connection) -> None:
        frame = duckdb.sql("SELECT TIMESTAMPTZ '2024-01-01' AS ts WHERE FALSE").to_pandas(con)
        assert str(frame["ts"].dtype) == "datetime64[us, UTC]"

    def test_empty_date_follows_date_as_object(self, con: duckdb.Connection) -> None:
        plan = duckdb.sql("SELECT DATE '2024-01-01' AS d WHERE FALSE")
        assert plan.to_pandas(con, date_as_object=True)["d"].dtype == np.dtype(object)
        assert plan.to_pandas(con)["d"].dtype == np.dtype("datetime64[us]")

    def test_empty_interval_is_timedelta(self, con: duckdb.Connection) -> None:
        frame = duckdb.sql("SELECT INTERVAL 1 DAY AS iv WHERE FALSE").to_pandas(con)
        assert frame["iv"].dtype == np.dtype("timedelta64[us]")


class TestChunkViewContract:
    def test_buffers_keep_the_chunk_alive(self, con: duckdb.Connection) -> None:
        import gc

        result = con._execute("SELECT i::BIGINT AS a FROM range(3) t(i)")
        view = result.result.fetch_chunk_view()
        assert view is not None
        buffer = view.data(0)
        assert buffer is not None
        del view
        gc.collect()
        assert list(np.frombuffer(buffer, dtype="int64")) == [0, 1, 2]
        result.close()

    def test_a_partly_fetched_chunk_carries_its_offset(self, con: duckdb.Connection) -> None:
        result = con._execute("SELECT i::BIGINT AS a FROM range(6) t(i)")
        assert result.fetch_rows(2) == [(0,), (1,)]
        view = result.result.fetch_chunk_view()
        assert view is not None
        assert view.row_offset == 2
        assert view.row_count == 6
        result.close()


class TestInt128Egress:
    def test_hugeint_maps_to_float64_like_the_engine_cast(self, con: duckdb.Connection) -> None:
        top = 170141183460469231731687303715884105727
        out = duckdb.sql(f"SELECT (-1)::HUGEINT AS m, (-1000000)::HUGEINT AS n, {top}::HUGEINT AS p").to_numpy(con)
        engine = duckdb.sql(f"SELECT (-1)::HUGEINT::DOUBLE, (-1000000)::HUGEINT::DOUBLE, {top}::HUGEINT::DOUBLE").rows(
            con
        )[0]
        assert out["m"].dtype == np.dtype("float64")
        assert (out["m"][0], out["n"][0], out["p"][0]) == engine

    def test_negative_hugeint_extreme_matches_the_engine_cast(self, con: duckdb.Connection) -> None:
        top = 170141183460469231731687303715884105727
        out = duckdb.sql(f"SELECT (-{top})::HUGEINT AS n").to_numpy(con)
        engine = duckdb.sql(f"SELECT (-{top})::HUGEINT::DOUBLE").rows(con)[0][0]
        assert out["n"][0] == engine

    def test_uhugeint_maps_to_float64(self, con: duckdb.Connection) -> None:
        umax = 340282366920938463463374607431768211455
        out = duckdb.sql(f"SELECT {umax}::UHUGEINT AS u").to_numpy(con)
        assert out["u"][0] == float(umax)

    def test_null_hugeints_come_back_masked(self, con: duckdb.Connection) -> None:
        out = duckdb.sql("SELECT unnest([(-5)::HUGEINT, NULL, 7::HUGEINT]) AS h").to_numpy(con)
        assert isinstance(out["h"], np.ma.MaskedArray)
        assert out["h"].mask.tolist() == [False, True, False]
        assert out["h"].data[0] == -5.0
        assert out["h"].data[2] == 7.0

    def test_negative_wide_decimals_are_exact(self, con: duckdb.Connection) -> None:
        out = duckdb.sql("SELECT (-2.5)::DECIMAL(38,6) AS a, (-0.000001)::DECIMAL(38,6) AS b").to_numpy(con)
        assert out["a"][0] == -2.5
        assert out["b"][0] == -0.000001
