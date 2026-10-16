"""Timestamp comparison filter pushdown into Arrow scans.

A filter pushed into an Arrow scan has to keep exactly the rows the same filter keeps on a
native table holding the same data, for every Arrow timestamp unit, with and without a time
zone. The Arrow scan does not re-apply a pushed filter, so a constant converted to the wrong
unit returns wrong rows or raises.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest
from _pushdown_helpers import _find_arrow_scan_nodes

import duckdb

pa = pytest.importorskip("pyarrow")
pa_ds = pytest.importorskip("pyarrow.dataset")


TICKS_PER_SECOND = {"s": 1, "ms": 10**3, "us": 10**6, "ns": 10**9}
NOON_PLUS_ONE_TICK = {"s": "12:00:01", "ms": "12:00:00.001", "us": "12:00:00.000001", "ns": "12:00:00.000000001"}
MAY_FIRST_NOON = 1714564800
ONE_DAY = 86400
INFINITY_SENTINEL = 2**63 - 1

UNITS = ["s", "ms", "us", "ns"]
ZONES = {"naive": None, "utc": "UTC", "new_york": "America/New_York"}

IN_LIST_FROM_MICROSECOND_DATETIMES = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="IN builds its list through PythonObject::FromValue, a microsecond datetime; follow up",
)
POLARS_CONSTANT_FROM_MICROSECOND_DATETIME = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="the polars backend builds its constant through PythonObject::FromValue, a microsecond datetime; follow up",
)
POLARS_CONSTANT_IN_SESSION_ZONE = pytest.mark.xfail(
    strict=True,
    raises=duckdb.Error,
    reason="the polars backend's constant carries the session time zone, which polars refuses to compare with another"
    " zone; follow up",
)


def _column_values(unit):
    ticks = TICKS_PER_SECOND[unit]
    return [
        1 * ticks,
        # A one second constant multiplied by 1000 lands on this row.
        1000 * ticks,
        MAY_FIRST_NOON * ticks,
        MAY_FIRST_NOON * ticks + 1,
        (MAY_FIRST_NOON + ONE_DAY) * ticks,
        -1 * ticks,
        None,
    ]


def _literals(unit, zone):
    offset = "+00" if zone else ""
    return {
        "epoch": f"'1970-01-01 00:00:01{offset}'",
        "noon": f"'2024-05-01 12:00:00{offset}'",
        "tick": f"'2024-05-01 {NOON_PLUS_ONE_TICK[unit]}{offset}'",
    }


def _unit_zone_params(zones, marked=None):
    marked = marked or {}
    return [
        pytest.param(unit, zone, id=f"{unit}-{zone_id}", marks=marked.get((unit, zone_id), ()))
        for unit in UNITS
        for zone_id, zone in zones.items()
    ]


UNIT_ZONES = _unit_zone_params(ZONES)
UNIT_WITH_ZONES = _unit_zone_params({zone_id: zone for zone_id, zone in ZONES.items() if zone is not None})


def _as_table(table):
    return table


def _as_dataset(table):
    return pa_ds.dataset(table)


def _as_scanner(table):
    return pa_ds.dataset(table).scanner()


def _as_record_batch_reader(table):
    return table.to_reader()


def _as_polars_dataframe(table):
    pl = pytest.importorskip("polars")
    return pl.from_arrow(table)


def _as_polars_lazyframe(table):
    return _as_polars_dataframe(table).lazy()


def _as_arrow_dtype_pandas(table):
    pd = pytest.importorskip("pandas")
    return table.to_pandas(types_mapper=pd.ArrowDtype)


INPUTS = {
    "dataset": _as_dataset,
    "scanner": _as_scanner,
    "record_batch_reader": _as_record_batch_reader,
    "polars_dataframe": _as_polars_dataframe,
    "polars_lazyframe": _as_polars_lazyframe,
    "pandas_arrow_dtype": _as_arrow_dtype_pandas,
}


def _input_params(marked):
    return [
        pytest.param(
            unit, zone, factory, id=f"{unit}-{zone_id}-{input_id}", marks=marked.get((unit, zone_id, input_id), ())
        )
        for unit in UNITS
        for zone_id, zone in ZONES.items()
        for input_id, factory in INPUTS.items()
    ]


@pytest.fixture
def con(duckdb_cursor):
    duckdb_cursor.execute("SET TimeZone = 'UTC'")
    return duckdb_cursor


def _make_tables(con, unit, zone, factory=_as_table):
    values = pa.array(_column_values(unit), pa.timestamp(unit, tz=zone))
    table = pa.table(
        {
            "x": list(range(1, len(values) + 1)),
            "p": values,
            "s": pa.StructArray.from_arrays([values], ["p"]),
        }
    )
    con.register("arrow_t", factory(table))
    con.execute("CREATE OR REPLACE TABLE native_t AS SELECT * FROM arrow_t")
    return lambda: con.register("arrow_t", factory(table))


def _rows(con, table, predicate, params=None):
    return con.execute(f"SELECT x FROM {table} WHERE {predicate} ORDER BY x", params).fetchall()


def _assert_matches_native(con, predicate, params=None):
    native = _rows(con, "native_t", predicate, params)
    assert _rows(con, "arrow_t", predicate, params) == native
    return len(native)


def _scan_emitted_rows(con, query):
    plan = json.loads(con.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {query}").fetchone()[1])
    [(scan, _)] = _find_arrow_scan_nodes(plan["operator"])
    return scan["intermediate_rows"]


def _assert_pushed(con, predicate, native_count):
    query = f"SELECT x FROM arrow_t WHERE {predicate}"
    plan = json.loads(con.execute(f"EXPLAIN (FORMAT JSON) {query}").fetchone()[1])
    [(scan, has_filter_above)] = _find_arrow_scan_nodes(plan)
    assert scan["extra_info"].get("Filters"), "no filter on the Arrow scan"
    assert not has_filter_above, "FILTER operator above the Arrow scan"
    assert _scan_emitted_rows(con, query) == native_count


@pytest.mark.parametrize("constant", ["epoch", "tick"])
@pytest.mark.parametrize("op", ["=", "<>", "<", "<=", ">", ">="])
@pytest.mark.parametrize(("unit", "zone"), UNIT_ZONES)
def test_comparison(con, unit, zone, op, constant):
    _make_tables(con, unit, zone)
    predicate = f"p {op} {_literals(unit, zone)[constant]}"
    _assert_pushed(con, predicate, _assert_matches_native(con, predicate))


@pytest.mark.parametrize(("unit", "zone"), UNIT_ZONES)
def test_between(con, unit, zone):
    _make_tables(con, unit, zone)
    literals = _literals(unit, zone)
    predicate = f"p BETWEEN {literals['noon']} AND {literals['tick']}"
    _assert_pushed(con, predicate, _assert_matches_native(con, predicate))


@pytest.mark.parametrize("bound", ["'infinity'", "'-infinity'"])
@pytest.mark.parametrize(("unit", "zone"), UNIT_ZONES)
def test_infinite_bound(con, unit, zone, bound):
    _make_tables(con, unit, zone)
    predicate = f"p < {bound}" if bound == "'infinity'" else f"p > {bound}"
    _assert_pushed(con, predicate, _assert_matches_native(con, predicate))


@pytest.mark.parametrize(
    "constant",
    [
        pytest.param("'infinity'", id="infinity"),
        pytest.param("'-infinity'", id="minus_infinity"),
        pytest.param("'2262-04-11 23:47:16.854775806+00'", id="largest_finite"),
        pytest.param("'1677-09-21 00:12:43.145224194+00'", id="smallest_finite"),
    ],
)
@pytest.mark.parametrize("op", ["=", "<>", "<", "<=", ">", ">="])
def test_infinity_sentinels_in_the_data(con, op, constant):
    """The engine reads the largest int64 as infinity and its negation as minus infinity."""
    values = [-INFINITY_SENTINEL, -INFINITY_SENTINEL + 1, INFINITY_SENTINEL - 1, INFINITY_SENTINEL, None]
    con.register(
        "arrow_t",
        pa.table({"x": list(range(1, len(values) + 1)), "p": pa.array(values, pa.timestamp("ns", tz="UTC"))}),
    )
    con.execute("CREATE OR REPLACE TABLE native_t AS SELECT * FROM arrow_t")
    _assert_matches_native(con, f"p {op} {constant}")


@pytest.mark.parametrize(("unit", "zone"), UNIT_ZONES)
def test_struct_field(con, unit, zone):
    _make_tables(con, unit, zone)
    predicate = f"s.p = {_literals(unit, zone)['tick']}"
    _assert_pushed(con, predicate, _assert_matches_native(con, predicate))


@pytest.mark.parametrize(("unit", "zone"), UNIT_ZONES)
def test_in_list(con, unit, zone):
    """The optimizer pushes an IN list as an optional filter and keeps a FILTER above the scan."""
    _make_tables(con, unit, zone)
    literals = _literals(unit, zone)
    predicate = f"p IN ({literals['epoch']}, {literals['noon']})"
    native_count = _assert_matches_native(con, predicate)
    assert _scan_emitted_rows(con, f"SELECT x FROM arrow_t WHERE {predicate}") == native_count


@pytest.mark.parametrize(
    ("unit", "zone"),
    _unit_zone_params(ZONES, {("ns", zone_id): IN_LIST_FROM_MICROSECOND_DATETIMES for zone_id in ZONES}),
)
def test_in_list_at_column_precision(con, unit, zone):
    _make_tables(con, unit, zone)
    literals = _literals(unit, zone)
    _assert_matches_native(con, f"p IN ({literals['epoch']}, {literals['tick']})")


@pytest.mark.parametrize("predicate", ["p IS NULL", "p IS NOT NULL"])
@pytest.mark.parametrize(("unit", "zone"), UNIT_ZONES)
def test_null_check(con, unit, zone, predicate):
    """The optimizer keeps IS NULL above Arrow scans, see test_is_null_pushes_into_arrow_scan."""
    _make_tables(con, unit, zone)
    _assert_matches_native(con, predicate)


@pytest.mark.parametrize(("unit", "zone"), UNIT_ZONES)
def test_python_datetime_parameter(con, unit, zone):
    _make_tables(con, unit, zone)
    noon = dt.datetime(2024, 5, 1, 12, tzinfo=dt.timezone.utc if zone else None)
    _assert_matches_native(con, "p = ?", [noon])


@pytest.mark.parametrize("op", ["=", "<"])
@pytest.mark.parametrize(("unit", "zone"), UNIT_WITH_ZONES)
def test_session_time_zone_differs_from_column(con, unit, zone, op):
    con.execute("SET TimeZone = 'Asia/Kathmandu'")
    _make_tables(con, unit, zone)
    predicate = f"p {op} {_literals(unit, zone)['tick']}"
    _assert_pushed(con, predicate, _assert_matches_native(con, predicate))


@pytest.mark.parametrize(
    ("unit", "zone", "factory"),
    _input_params(
        {
            ("ns", "naive", "polars_lazyframe"): POLARS_CONSTANT_FROM_MICROSECOND_DATETIME,
            ("ns", "utc", "polars_lazyframe"): POLARS_CONSTANT_FROM_MICROSECOND_DATETIME,
            **{(unit, "new_york", "polars_lazyframe"): POLARS_CONSTANT_IN_SESSION_ZONE for unit in UNITS},
        }
    ),
)
def test_every_arrow_input(con, unit, zone, factory):
    """A RecordBatchReader can be read only once, so every scanning query gets a fresh input."""
    register_fresh_input = _make_tables(con, unit, zone, factory)
    predicate = f"p = {_literals(unit, zone)['tick']}"
    register_fresh_input()
    native_count = _assert_matches_native(con, predicate)
    register_fresh_input()
    _assert_pushed(con, predicate, native_count)
