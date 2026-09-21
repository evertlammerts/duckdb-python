"""Results converted straight to numpy, where NULLs decide masked arrays."""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import numpy as np
import pytest

import duckdb
from duckdb.frame import col

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def con() -> Iterator[duckdb.frame.Connection]:
    connection = duckdb.frame.connect()
    yield connection
    connection.close()


class TestToNumpy:
    def test_numeric_dtypes_are_exact(self, con: duckdb.frame.Connection) -> None:
        out = duckdb.frame.sql(
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

    def test_nulls_come_back_masked(self, con: duckdb.frame.Connection) -> None:
        out = duckdb.frame.sql("SELECT unnest([1, NULL, 3]) AS v").to_numpy(con)
        assert isinstance(out["v"], np.ma.MaskedArray)
        assert out["v"].mask.tolist() == [False, True, False]
        assert out["v"].compressed().tolist() == [1, 3]

    def test_a_clean_column_is_a_plain_ndarray(self, con: duckdb.frame.Connection) -> None:
        out = duckdb.frame.sql("SELECT unnest([1, 2, 3]) AS v").to_numpy(con)
        assert not isinstance(out["v"], np.ma.MaskedArray)
        assert out["v"].tolist() == [1, 2, 3]

    def test_strings_and_nested_go_per_object(self, con: duckdb.frame.Connection) -> None:
        out = duckdb.frame.sql("SELECT 'hi' AS s, [1, 2] AS l, {'a': 1} AS st").to_numpy(con)
        assert out["s"].dtype == object
        assert out["s"][0] == "hi"
        assert out["l"][0] == [1, 2]
        assert out["st"][0] == {"a": 1}

    def test_temporal_units(self, con: duckdb.frame.Connection) -> None:
        out = duckdb.frame.sql(
            "SELECT DATE '2026-09-01' AS d, TIMESTAMP '2026-09-01 12:00:00' AS ts, "
            "TIMESTAMP_NS '2026-09-01 12:00:00.123456789' AS ns, INTERVAL 90 DAY AS iv"
        ).to_numpy(con)
        assert out["d"].dtype == np.dtype("datetime64[us]")
        assert out["ts"].dtype == np.dtype("datetime64[us]")
        assert out["ns"].dtype == np.dtype("datetime64[ns]")
        assert out["iv"].dtype == np.dtype("timedelta64[us]")
        assert out["ts"][0] == np.datetime64("2026-09-01T12:00:00", "us")

    def test_decimal_becomes_float64_at_its_scale(self, con: duckdb.frame.Connection) -> None:
        # numpy trades exactness for vectors, as the previous package did; exact Decimals come from the rows.
        out = duckdb.frame.sql("SELECT 1.250::DECIMAL(18,3) AS d, 2.5::DECIMAL(38,1) AS wide").to_numpy(con)
        assert out["d"].dtype == np.dtype("float64")
        assert out["d"][0] == 1.25
        assert out["wide"][0] == 2.5

    def test_enum_becomes_strings_with_none_for_null(self, con: duckdb.frame.Connection) -> None:
        con.run("CREATE TYPE mood AS ENUM ('sad', 'ok', 'happy')")
        out = duckdb.frame.sql("SELECT unnest(['ok', NULL, 'happy']::mood[]) AS m").to_numpy(con)
        column = out["m"]
        assert isinstance(column, np.ma.MaskedArray)
        assert column.mask.tolist() == [False, True, False]
        assert column.data[0] == "ok"
        assert column.data[1] is None
        assert column.data[2] == "happy"

    def test_an_empty_result_keeps_its_dtypes(self, con: duckdb.frame.Connection) -> None:
        out = duckdb.frame.sql("SELECT 1::BIGINT AS v, 'x' AS s WHERE FALSE").to_numpy(con)
        assert out["v"].dtype == np.dtype("int64")
        assert len(out["v"]) == 0
        assert out["s"].dtype == object

    def test_many_chunks_concatenate(self, con: duckdb.frame.Connection) -> None:
        out = duckdb.frame.sql("SELECT range AS v FROM range(10000)").to_numpy(con)
        assert len(out["v"]) == 10000
        assert out["v"][9999] == 9999
        assert out["v"].sum() == 49995000

    def test_parameters_flow(self, con: duckdb.frame.Connection) -> None:
        plan = duckdb.frame.sql("SELECT range AS v FROM range(10)").filter(col("v") < duckdb.frame.param("cap"))
        out = plan.to_numpy(con, parameters={"cap": 3})
        assert out["v"].tolist() == [0, 1, 2]


class TestBoundEgress:
    def test_bound_forwards_to_numpy(self, con: duckdb.frame.Connection) -> None:
        bound = duckdb.frame.sql("SELECT 1 AS v").on(con)
        assert bound.to_numpy()["v"].tolist() == [1]


class TestTemporalInfinities:
    def test_infinite_dates_clamp_like_the_row_path(self, con: duckdb.frame.Connection) -> None:
        plan = duckdb.frame.sql("SELECT 'infinity'::DATE AS pos, '-infinity'::DATE AS neg")
        rows = plan.rows(con)
        assert rows[0] == (datetime.date.max, datetime.date.min)
        out = plan.to_numpy(con)
        assert out["pos"][0] == np.datetime64("9999-12-31", "us")
        assert out["neg"][0] == np.datetime64("0001-01-01", "us")

    def test_infinite_timestamps_agree_with_the_row_path(self, con: duckdb.frame.Connection) -> None:
        plan = duckdb.frame.sql("SELECT 'infinity'::TIMESTAMP AS pos, '-infinity'::TIMESTAMP AS neg")
        out = plan.to_numpy(con)
        rows = plan.rows(con)
        assert out["pos"][0].item() == rows[0][0] == datetime.datetime.max
        assert out["neg"][0].item() == rows[0][1] == datetime.datetime.min

    def test_infinite_coarse_timestamps_clamp_in_their_unit(self, con: duckdb.frame.Connection) -> None:
        out = duckdb.frame.sql("SELECT 'infinity'::TIMESTAMP_S AS s, 'infinity'::TIMESTAMP_MS AS ms").to_numpy(con)
        assert out["s"].dtype == np.dtype("datetime64[s]")
        assert out["ms"].dtype == np.dtype("datetime64[ms]")
        assert out["s"][0].astype("datetime64[Y]") == np.datetime64("9999", "Y")
        assert out["ms"][0].astype("datetime64[Y]") == np.datetime64("9999", "Y")

    def test_infinite_timestamptz_clamps_as_a_utc_instant(self, con: duckdb.frame.Connection) -> None:
        out = duckdb.frame.sql("SELECT 'infinity'::TIMESTAMPTZ AS ts").to_numpy(con)
        assert out["ts"].dtype == np.dtype("datetime64[us]")
        assert out["ts"][0] == np.datetime64("9999-12-31T23:59:59.999999", "us")


class TestEmptyResults:
    def test_empty_decimal_and_enum_keep_their_dtypes(self, con: duckdb.frame.Connection) -> None:
        con.run("CREATE TYPE empty_mood AS ENUM ('sad', 'ok')")
        out = duckdb.frame.sql("SELECT 1.5::DECIMAL(9,2) AS d, 'ok'::empty_mood AS m WHERE FALSE").to_numpy(con)
        assert out["d"].dtype == np.dtype("float64")
        assert out["m"].dtype == np.dtype(object)
        assert len(out["d"]) == len(out["m"]) == 0

    def test_empty_temporals_keep_their_units(self, con: duckdb.frame.Connection) -> None:
        out = duckdb.frame.sql(
            "SELECT TIMESTAMPTZ '2024-01-01' AS ts, DATE '2024-01-01' AS d, INTERVAL 1 DAY AS iv WHERE FALSE"
        ).to_numpy(con)
        assert out["ts"].dtype == np.dtype("datetime64[us]")
        assert out["d"].dtype == np.dtype("datetime64[us]")
        assert out["iv"].dtype == np.dtype("timedelta64[us]")


class TestChunkViewContract:
    def test_buffers_keep_the_chunk_alive(self, con: duckdb.frame.Connection) -> None:
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

    def test_a_partly_fetched_chunk_carries_its_offset(self, con: duckdb.frame.Connection) -> None:
        result = con._execute("SELECT i::BIGINT AS a FROM range(6) t(i)")
        assert result.fetch_rows(2) == [(0,), (1,)]
        view = result.result.fetch_chunk_view()
        assert view is not None
        assert view.row_offset == 2
        assert view.row_count == 6
        result.close()


class TestInt128Egress:
    def test_hugeint_maps_to_float64_like_the_engine_cast(self, con: duckdb.frame.Connection) -> None:
        top = 170141183460469231731687303715884105727
        out = duckdb.frame.sql(f"SELECT (-1)::HUGEINT AS m, (-1000000)::HUGEINT AS n, {top}::HUGEINT AS p").to_numpy(
            con
        )
        engine = duckdb.frame.sql(
            f"SELECT (-1)::HUGEINT::DOUBLE, (-1000000)::HUGEINT::DOUBLE, {top}::HUGEINT::DOUBLE"
        ).rows(con)[0]
        assert out["m"].dtype == np.dtype("float64")
        assert (out["m"][0], out["n"][0], out["p"][0]) == engine

    def test_negative_hugeint_extreme_matches_the_engine_cast(self, con: duckdb.frame.Connection) -> None:
        top = 170141183460469231731687303715884105727
        out = duckdb.frame.sql(f"SELECT (-{top})::HUGEINT AS n").to_numpy(con)
        engine = duckdb.frame.sql(f"SELECT (-{top})::HUGEINT::DOUBLE").rows(con)[0][0]
        assert out["n"][0] == engine

    def test_uhugeint_maps_to_float64(self, con: duckdb.frame.Connection) -> None:
        umax = 340282366920938463463374607431768211455
        out = duckdb.frame.sql(f"SELECT {umax}::UHUGEINT AS u").to_numpy(con)
        assert out["u"][0] == float(umax)

    def test_null_hugeints_come_back_masked(self, con: duckdb.frame.Connection) -> None:
        out = duckdb.frame.sql("SELECT unnest([(-5)::HUGEINT, NULL, 7::HUGEINT]) AS h").to_numpy(con)
        assert isinstance(out["h"], np.ma.MaskedArray)
        assert out["h"].mask.tolist() == [False, True, False]
        assert out["h"].data[0] == -5.0
        assert out["h"].data[2] == 7.0

    def test_negative_wide_decimals_are_exact(self, con: duckdb.frame.Connection) -> None:
        out = duckdb.frame.sql("SELECT (-2.5)::DECIMAL(38,6) AS a, (-0.000001)::DECIMAL(38,6) AS b").to_numpy(con)
        assert out["a"][0] == -2.5
        assert out["b"][0] == -0.000001
