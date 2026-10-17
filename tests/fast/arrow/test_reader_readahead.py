"""Scanning an Arrow stream holds a few batches in Arrow memory, not the whole stream.

A pyarrow RecordBatchReader, and any other object with __arrow_c_stream__ that is not a pyarrow Table, is read the
way a bare stream capsule is: DuckDB pulls one batch per scanning thread and applies projection and filters itself,
above the scan. pyarrow's Scanner, which would evaluate pushed filters, reads a stream far ahead of a slow consumer,
and that memory sits outside memory_limit.
"""

from __future__ import annotations

import threading
import time

import pytest
from _pushdown_helpers import assert_plan_pushes_nothing, physical_plan

import duckdb
from duckdb.sqltypes import BIGINT

pa = pytest.importorskip("pyarrow")
pytest.importorskip("pyarrow.dataset")

THREADS = 4
BATCHES = 40
BATCH_ROWS = 1024
PAYLOAD = "x" * 1024
SCHEMA = pa.schema([("id", pa.int64()), ("payload", pa.string())])
# A bare capsule holds one batch per scanning thread plus one; this leaves room for as many again.
PEAK_BATCHES = 2 * THREADS + 2


def _batches():
    for index in range(BATCHES):
        start = index * BATCH_ROWS
        yield pa.record_batch(
            [pa.array(range(start, start + BATCH_ROWS), pa.int64()), pa.array([PAYLOAD] * BATCH_ROWS)],
            schema=SCHEMA,
        )


class OneShotStream:
    """A third party stream: a plain object that hands out the stream of one generator."""

    def __init__(self, reader) -> None:
        self.reader = reader

    def __arrow_c_schema__(self):  # noqa: ANN204
        return self.reader.schema.__arrow_c_schema__()

    def __arrow_c_stream__(self, requested_schema=None):  # noqa: ANN204
        return self.reader.__arrow_c_stream__(requested_schema)


class StreamClaimingToBeATable(OneShotStream):
    """A third party stream whose __class__ claims pyarrow's Table, which isinstance believes."""

    @property
    def __class__(self):  # noqa: ANN204
        return pa.Table


def _slow_identity(ids):
    # Batches pile up only behind a consumer that is slower than the producer.
    time.sleep(0.02)
    return ids


def _peak_arrow_bytes_while(run):
    baseline = pa.total_allocated_bytes()
    peak = baseline
    done = threading.Event()

    def sample():
        nonlocal peak
        while not done.is_set():
            peak = max(peak, pa.total_allocated_bytes())
            time.sleep(0.001)

    sampler = threading.Thread(target=sample)
    sampler.start()
    try:
        result = run()
    finally:
        done.set()
        sampler.join()
    return result, peak - baseline


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "export",
    [
        pytest.param(lambda reader: reader, id="reader"),
        pytest.param(lambda reader: reader.__arrow_c_stream__(), id="capsule"),
        pytest.param(OneShotStream, id="third_party_stream"),
        pytest.param(StreamClaimingToBeATable, id="stream_claiming_to_be_a_table"),
    ],
)
def test_slowly_consumed_stream_holds_a_few_batches(export):
    query = "SELECT count(*), sum(strlen(payload)), sum(slow_identity(id)) FROM source"
    with duckdb.connect(config={"threads": THREADS, "memory_limit": "64MB"}) as con:
        con.create_function("slow_identity", _slow_identity, [BIGINT], BIGINT, type="arrow")
        con.register("source", export(pa.RecordBatchReader.from_batches(SCHEMA, _batches())))
        rows, peak = _peak_arrow_bytes_while(lambda: con.execute(query).fetchall())

    total = BATCHES * BATCH_ROWS
    assert rows == [(total, total * len(PAYLOAD), total * (total - 1) // 2)]
    batch_bytes = next(_batches()).nbytes
    assert peak / batch_bytes <= PEAK_BATCHES


TABLE = pa.table(
    {
        "a": [1, 2, None, 4, 2],
        "b": ["x", "y", "z", None, "y"],
        "c": [1.5, 2.5, 3.5, 4.5, 2.5],
    }
)


def _reader():
    return TABLE.to_reader(max_chunksize=2)


@pytest.mark.parametrize(
    "predicate",
    ["a > 1", "a = 2 AND b = 'y'", "a IN (1, 4)", "a IS NULL", "a > 1 OR b IS NULL"],
)
def test_filtered_reader_scan_matches_native(duckdb_cursor, predicate):
    duckdb_cursor.register("table_source", TABLE)
    duckdb_cursor.execute("CREATE TABLE native_t AS SELECT * FROM table_source")
    duckdb_cursor.register("reader_t", _reader())

    def rows(table):
        return duckdb_cursor.execute(f"SELECT c, b FROM {table} WHERE {predicate} ORDER BY ALL").fetchall()

    assert rows("reader_t") == rows("native_t")


@pytest.mark.parametrize(
    ("make_source", "entry_point"),
    [
        pytest.param(_reader, "register", id="reader-register"),
        pytest.param(_reader, "replacement", id="reader-replacement"),
        pytest.param(_reader, "from_arrow", id="reader-from_arrow"),
        pytest.param(lambda: OneShotStream(_reader()), "register", id="third_party_stream-register"),
    ],
)
def test_filtered_stream_scan_leaves_projection_and_filter_to_duckdb(duckdb_cursor, make_source, entry_point):
    query = "SELECT c, b FROM src WHERE a > 1"
    if entry_point == "register":
        duckdb_cursor.register("src", make_source())
    elif entry_point == "replacement":
        src = make_source()
    else:
        src = duckdb_cursor.from_arrow(make_source())  # noqa: F841

    # The replacement scan looks only in the frame that runs the query, so the plan is read here.
    assert_plan_pushes_nothing(physical_plan(duckdb_cursor.execute(f"EXPLAIN (FORMAT JSON) {query}").fetchall()))
    assert sorted(duckdb_cursor.execute(query).fetchall()) == [(2.5, "y"), (2.5, "y"), (4.5, None)]


@pytest.mark.parametrize(
    ("column", "predicate", "expected"),
    [
        pytest.param(
            pa.DictionaryArray.from_arrays(
                pa.array([0, 1, 2, None, 1], pa.int8()), pa.array(["a", "b", "c"], pa.string_view())
            ),
            "v = 'b'",
            [(2,), (5,)],
            id="dictionary_of_string_view",
        ),
        pytest.param(
            pa.RunEndEncodedArray.from_arrays(pa.array([2, 4, 5], pa.int32()), pa.array([10, 20, 30], pa.int64())),
            "v = 20",
            [(3,), (4,)],
            id="run_end_encoded",
        ),
    ],
)
def test_reader_filter_that_pyarrow_cannot_evaluate_returns_rows(duckdb_cursor, column, predicate, expected):
    duckdb_cursor.register("src", pa.table({"x": [1, 2, 3, 4, 5], "v": column}).to_reader())
    assert duckdb_cursor.execute(f"SELECT x FROM src WHERE {predicate} ORDER BY x").fetchall() == expected
