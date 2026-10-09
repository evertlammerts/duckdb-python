"""Results leaving as Arrow: `to_arrow`, `to_reader`, and the capsule protocol on a bound plan."""

from __future__ import annotations

import contextlib
import datetime
import decimal
import gc
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
import uuid
import weakref
from typing import TYPE_CHECKING

import pytest

import duckdb
from duckdb import exceptions
from duckdb.frame import col, fn, param

if TYPE_CHECKING:
    from collections.abc import Iterator

with contextlib.suppress(ModuleNotFoundError):
    import pyarrow as pa

pytestmark = pytest.mark.requires("pyarrow")


@pytest.fixture
def con() -> Iterator[duckdb.frame.Connection]:
    connection = duckdb.frame.connect()
    yield connection
    connection.close()


def sql(query: str) -> duckdb.frame.Frame:
    return duckdb.frame.sql(query)


def blocked_stream(
    con: duckdb.frame.Connection,
) -> tuple[duckdb.frame.connection.LiveArrowStream, threading.Event]:
    """A stream whose first batch never comes, with an event that fires once its read is provably in flight.

    The marking function runs inside the sort's input scan, so the event is a handshake from the engine work
    itself, not a guess from a sleep.
    """
    started = getattr(con, "_test_mark_started", None)
    if started is None:
        started = threading.Event()
        con._test_mark_started = started  # type: ignore[attr-defined]

        def mark(value: int) -> int:
            started.set()
            return value

        con.create_function("mark_started", mark, ["BIGINT"], "BIGINT")
    started.clear()
    held = con._execute_arrow("SELECT mark_started(i) AS v FROM range(20_000_000_000) t(i) ORDER BY v DESC", None, 0)
    return held, started


class TestToArrowValues:
    @pytest.mark.parametrize(
        "query",
        [
            "SELECT * FROM (VALUES (1::TINYINT), (NULL)) t(v)",
            "SELECT * FROM (VALUES (2::SMALLINT), (3::SMALLINT)) t(v)",
            "SELECT * FROM (VALUES (4::INTEGER), (NULL)) t(v)",
            "SELECT * FROM (VALUES (5::BIGINT), (-5::BIGINT)) t(v)",
            "SELECT * FROM (VALUES (6::UTINYINT), (NULL)) t(v)",
            "SELECT * FROM (VALUES (7::USMALLINT), (NULL)) t(v)",
            "SELECT * FROM (VALUES (8::UINTEGER), (NULL)) t(v)",
            "SELECT * FROM (VALUES (9::UBIGINT), (NULL)) t(v)",
            "SELECT * FROM (VALUES (1.5::FLOAT), (NULL)) t(v)",
            "SELECT * FROM (VALUES (2.5::DOUBLE), (-0.5::DOUBLE)) t(v)",
            "SELECT * FROM (VALUES (12.345::DECIMAL(18, 3)), (NULL)) t(v)",
            "SELECT * FROM (VALUES ('text'), (''), (NULL)) t(v)",
            "SELECT * FROM (VALUES ('\\xAA'::BLOB), (NULL)) t(v)",
            "SELECT * FROM (VALUES (DATE '2020-01-02'), (NULL)) t(v)",
            "SELECT * FROM (VALUES (TIME '01:02:03.000004'), (NULL)) t(v)",
            "SELECT * FROM (VALUES (TIMESTAMP '2020-01-02 03:04:05.000006'), (NULL)) t(v)",
            "SELECT * FROM (VALUES (TIMESTAMP_S '2020-01-02 03:04:05'), (NULL)) t(v)",
            "SELECT * FROM (VALUES (TIMESTAMP_MS '2020-01-02 03:04:05.007'), (NULL)) t(v)",
            "SELECT * FROM (VALUES (TIMESTAMP_NS '2020-01-02 03:04:05.000006'), (NULL)) t(v)",
            "SELECT * FROM (VALUES ('x'::ENUM('x', 'y')), (NULL)) t(v)",
            "SELECT * FROM (VALUES ([1, 2]), ([]), (NULL)) t(v)",
            "SELECT * FROM (VALUES ([1, 2]::INT[2]), (NULL)) t(v)",
            "SELECT * FROM (VALUES ({'a': 1, 'b': 'x'}), (NULL)) t(v)",
            "SELECT * FROM (VALUES ([{'a': [1]}]), (NULL)) t(v)",
            "SELECT NULL AS v",
        ],
    )
    def test_a_family_reads_as_rows_read(self, con: duckdb.frame.Connection, query: str) -> None:
        plan = sql(query)
        assert plan.to_arrow(con).column("v").to_pylist() == [row[0] for row in plan.rows(con)]

    def test_forms_arrow_spells_differently(self, con: duckdb.frame.Connection) -> None:
        # The engine's default export: HUGEINT and UHUGEINT as decimal128, UUID as text, MAP as pairs, a BIT as
        # its packed bytes, VARINT as an opaque extension; `rows()` keeps the native Python forms.
        table = sql(
            "SELECT 123456789012345678901234567890123::HUGEINT AS h,"
            " 98765432109876543210987654321098765432::UHUGEINT AS uh,"
            " 12345678901234567890123456789012345678901234567890::VARINT AS vi,"
            " uuid '550e8400-e29b-41d4-a716-446655440000' AS u, MAP {'k': 1} AS m, '101'::BIT AS b"
        ).to_arrow(con)
        assert table.column("h").to_pylist() == [decimal.Decimal("123456789012345678901234567890123")]
        assert table.column("uh").to_pylist() == [decimal.Decimal("98765432109876543210987654321098765432")]
        assert "arrow.opaque" in str(table.schema.field("vi").type)
        assert isinstance(table.column("vi").to_pylist()[0], bytes)
        assert table.column("u").to_pylist() == ["550e8400-e29b-41d4-a716-446655440000"]
        assert table.column("m").to_pylist() == [[("k", 1)]]
        assert table.column("b").to_pylist() == [b"\x05\xfd"]

    def test_an_interval_counts_months_days_and_nanoseconds(self, con: duckdb.frame.Connection) -> None:
        [value] = sql("SELECT INTERVAL '1 month 2 days 3 seconds' AS v").to_arrow(con).column("v").to_pylist()
        assert (value.months, value.days, value.nanoseconds) == (1, 2, 3_000_000_000)

    def test_a_union_reads_its_member_values(self, con: duckdb.frame.Connection) -> None:
        con.run("CREATE TABLE unions(v UNION(num INTEGER, txt VARCHAR))")
        con.run("INSERT INTO unions VALUES (2), ('x')")
        table = sql("SELECT v FROM unions").to_arrow(con)
        assert str(table.schema.field("v").type).startswith("sparse_union")
        assert table.column("v").to_pylist() == [2, "x"]

    def test_nanoseconds_survive_the_export(self, con: duckdb.frame.Connection) -> None:
        # rows() reads TIMESTAMP_NS into a microsecond datetime; the Arrow path keeps every digit.
        [scalar] = sql("SELECT TIMESTAMP_NS '2020-01-02 03:04:05.000000008' AS v").to_arrow(con).column("v")
        assert scalar.value % 1000 == 8

    def test_a_zoned_timestamp_keeps_its_instant(self, con: duckdb.frame.Connection) -> None:
        con.run("SET TimeZone = 'UTC'")
        plan = sql("SELECT TIMESTAMPTZ '2020-01-02 03:04:05+00' AS v")
        assert plan.to_arrow(con).column("v").to_pylist() == [
            datetime.datetime(2020, 1, 2, 3, 4, 5, tzinfo=datetime.UTC)
        ]

    def test_duplicate_column_names_survive(self, con: duckdb.frame.Connection) -> None:
        table = sql("SELECT 1 AS a, 2 AS a").to_arrow(con)
        assert table.column_names == ["a", "a"]
        assert [column.to_pylist() for column in table.columns] == [[1], [2]]

    def test_an_empty_result_keeps_its_schema(self, con: duckdb.frame.Connection) -> None:
        table = sql("SELECT 1::BIGINT AS n, 'x' AS t WHERE false").to_arrow(con)
        assert table.num_rows == 0
        assert [field.name for field in table.schema] == ["n", "t"]
        assert table.schema.field("n").type == pa.int64()

    def test_parameters_flow_through(self, con: duckdb.frame.Connection) -> None:
        plan = sql("SELECT i FROM range(5) t(i)").filter(col("i") >= param("floor"))
        assert plan.to_arrow(con, parameters={"floor": 3}).column("i").to_pylist() == [3, 4]
        reader = plan.to_reader(con, parameters={"floor": 1})
        assert reader.read_all().column("i").to_pylist() == [1, 2, 3, 4]

    def test_an_untyped_parameter_takes_the_statements_type(self, con: duckdb.frame.Connection) -> None:
        # An empty list has no type of its own; the statement says INT[], through the same fill as execute.
        plan = sql("SELECT 1").select(fn("len", param("xs").cast("INT[]")).alias("n"))
        table = plan.to_arrow(con, parameters={"xs": []})
        assert table.column("n").to_pylist() == [0]


class TestSessionSettings:
    def test_lossless_conversion_switches_the_export(self, con: duckdb.frame.Connection) -> None:
        plan = sql("SELECT uuid '550e8400-e29b-41d4-a716-446655440000' AS u")
        assert plan.to_arrow(con).column("u").to_pylist() == ["550e8400-e29b-41d4-a716-446655440000"]
        con.run("SET arrow_lossless_conversion = true")
        lossless = plan.to_arrow(con)
        assert lossless.column("u").to_pylist() == [uuid.UUID("550e8400-e29b-41d4-a716-446655440000")]
        hugeint = sql("SELECT 1::HUGEINT AS h").to_arrow(con)
        assert "hugeint" in str(hugeint.schema.field("h").type)

    def test_the_session_zone_names_the_timestamp_type(self, con: duckdb.frame.Connection) -> None:
        con.run("SET TimeZone = 'Asia/Tokyo'")
        table = sql("SELECT TIMESTAMPTZ '2020-01-02 03:04:05+00' AS v").to_arrow(con)
        assert str(table.schema.field("v").type) == "timestamp[us, tz=Asia/Tokyo]"


class TestReaderStreaming:
    def test_batch_size_caps_every_batch(self, con: duckdb.frame.Connection) -> None:
        reader = sql("SELECT i FROM range(10000) t(i)").to_reader(con, batch_size=1000)
        batches = list(reader)
        assert sum(batch.num_rows for batch in batches) == 10000
        assert max(batch.num_rows for batch in batches) <= 1000
        assert len(batches) >= 10
        values = [v for batch in batches for v in batch.column("i").to_pylist()]
        assert values == list(range(10000))

    def test_the_default_batch_is_the_engines(self, con: duckdb.frame.Connection) -> None:
        batches = list(sql("SELECT i FROM range(140000) t(i)").to_reader(con))
        assert sum(batch.num_rows for batch in batches) == 140000
        assert max(batch.num_rows for batch in batches) <= 131072
        assert len(batches) >= 2

    def test_a_batch_size_must_be_positive(self, con: duckdb.frame.Connection) -> None:
        with pytest.raises(exceptions.InvalidInputError, match="batch_size must be positive"):
            sql("SELECT 1").to_reader(con, batch_size=0)

    def test_an_open_reader_holds_the_connection(self, con: duckdb.frame.Connection) -> None:
        reader = sql("SELECT i FROM range(1000000) t(i)").to_reader(con, batch_size=1000)
        assert reader.read_next_batch().num_rows > 0
        with pytest.raises(exceptions.OperationalError, match="live result"):
            con.run("SELECT 1")
        reader.close()
        assert sql("SELECT 1").rows(con) == [(1,)]

    def test_the_first_batch_arrives_before_the_query_finishes(self, con: duckdb.frame.Connection) -> None:
        # Minutes of work in full; the reader is streaming only if a batch lands long before that.
        reader = sql("SELECT i FROM range(20_000_000_000) t(i)").to_reader(con, batch_size=1000)
        started = time.monotonic()
        try:
            assert reader.read_next_batch().num_rows > 0
            assert time.monotonic() - started < 10
        finally:
            con.interrupt()
            reader.close()

    def test_a_consumer_driven_read_is_not_throttled(self) -> None:
        # With one thread the consumer runs the engine's work itself, step by step; a pacing nap after every
        # bounded step would stretch this scan from about a second to minutes.
        con = duckdb.frame.connect(threads="1")
        try:
            started = time.monotonic()
            table = sql("SELECT count(*) AS n FROM range(600_000_000) t(i) WHERE i % 7 = 0").to_arrow(con)
            elapsed = time.monotonic() - started
            assert table.column("n").to_pylist() == [85714286]
            assert elapsed < 8, f"the single-threaded read took {elapsed:.1f}s"
        finally:
            con.close()

    def test_dropping_an_unfinished_reader_frees_the_connection(self, con: duckdb.frame.Connection) -> None:
        reader = sql("SELECT i FROM range(1000000) t(i)").to_reader(con, batch_size=100)
        assert reader.read_next_batch().num_rows == 100
        started = time.monotonic()
        reader.close()
        assert time.monotonic() - started < 5
        assert sql("SELECT 1").rows(con) == [(1,)]


class TestStreamContract:
    def test_a_stream_is_consumed_once(self, con: duckdb.frame.Connection) -> None:
        stream = con._execute_arrow("SELECT 1 AS v", None, 0)
        assert pa.RecordBatchReader.from_stream(stream).read_all().num_rows == 1
        with pytest.raises(exceptions.InterfaceError, match="already consumed"):
            pa.RecordBatchReader.from_stream(stream)

    def test_a_bound_plan_runs_again_per_export(self, con: duckdb.frame.Connection) -> None:
        bound = sql("SELECT i FROM range(100) t(i)").on(con)
        assert pa.table(bound).num_rows == 100
        assert pa.table(bound).num_rows == 100

    def test_a_statement_without_rows_is_refused(self, con: duckdb.frame.Connection) -> None:
        with pytest.raises(exceptions.InvalidInputError, match="returns no rows"):
            sql("CREATE TABLE nothing_here(x INT)").to_arrow(con)

    def test_the_schema_outlives_a_cleanly_exhausted_stream(self, con: duckdb.frame.Connection) -> None:
        # The Arrow stream contract keeps get_schema answerable until the consumer releases the stream, as
        # pyarrow's own streams do; the nested column exercises the copied schema's children.
        import ctypes

        class CSchema(ctypes.Structure):
            pass

        CSchema._fields_ = [
            ("format", ctypes.c_char_p),
            ("name", ctypes.c_char_p),
            ("metadata", ctypes.c_void_p),
            ("flags", ctypes.c_int64),
            ("n_children", ctypes.c_int64),
            ("children", ctypes.POINTER(ctypes.POINTER(CSchema))),
            ("dictionary", ctypes.c_void_p),
            ("release", ctypes.c_void_p),
            ("private_data", ctypes.c_void_p),
        ]

        counts = ("length", "null_count", "offset", "n_buffers", "n_children")
        array_pointers = ("buffers", "children", "dictionary", "release", "private_data")
        callbacks = ("get_schema", "get_next", "get_last_error", "release", "private_data")

        class CArray(ctypes.Structure):
            pass

        CArray._fields_ = [(name, ctypes.c_int64) for name in counts] + [
            (name, ctypes.c_void_p) for name in array_pointers
        ]

        class CStream(ctypes.Structure):
            pass

        CStream._fields_ = [(name, ctypes.c_void_p) for name in callbacks]

        get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
        get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        get_pointer.restype = ctypes.c_void_p

        con.run("CREATE TYPE mood AS ENUM ('sad', 'ok', 'happy')")
        con.run("SET arrow_lossless_conversion = true")
        held = con._execute_arrow(
            "SELECT {'a': [{'b': [1, 2]}], 'c': 'x'} AS s, 'ok'::mood AS m,"
            " uuid '550e8400-e29b-41d4-a716-446655440000' AS u, 42 AS i",
            None,
            0,
        )
        capsule = held.__arrow_c_stream__()
        stream = ctypes.cast(get_pointer(capsule, b"arrow_array_stream"), ctypes.POINTER(CStream))
        schema_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(CStream), ctypes.POINTER(CSchema))
        get_schema = schema_type(stream.contents.get_schema)
        next_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(CStream), ctypes.POINTER(CArray))
        get_next = next_type(stream.contents.get_next)

        def metadata_bytes(address: int | None) -> bytes | None:
            # Arrow C metadata: an int32 pair count, then length-prefixed keys and values.
            if not address:
                return None
            pairs = int.from_bytes(ctypes.string_at(address, 4), sys.byteorder)
            length = 4
            for _ in range(pairs * 2):
                piece = int.from_bytes(ctypes.string_at(address + length, 4), sys.byteorder)
                length += 4 + piece
            return ctypes.string_at(address, length)

        def shape_of(node: CSchema) -> tuple[object, ...]:
            children = tuple(shape_of(node.children[i].contents) for i in range(node.n_children))
            dictionary = None
            if node.dictionary:
                dictionary = shape_of(ctypes.cast(node.dictionary, ctypes.POINTER(CSchema)).contents)
            return (node.format, node.name, node.flags, metadata_bytes(node.metadata), children, dictionary)

        def read_schema() -> tuple[object, ...]:
            value = CSchema()
            assert get_schema(stream, ctypes.byref(value)) == 0
            shape = shape_of(value)
            ctypes.CFUNCTYPE(None, ctypes.POINTER(CSchema))(value.release)(ctypes.byref(value))
            return shape

        before = read_schema()
        while True:
            value = CArray()
            assert get_next(stream, ctypes.byref(value)) == 0
            if not value.release:
                break
            ctypes.CFUNCTYPE(None, ctypes.POINTER(CArray))(value.release)(ctypes.byref(value))
        assert read_schema() == before
        assert read_schema() == before
        assert held.error is None
        assert held.stream.live is False
        ctypes.CFUNCTYPE(None, ctypes.POINTER(CStream))(stream.contents.release)(stream)

    def test_an_expanding_statement_defers_its_schema(self, con: duckdb.frame.Connection) -> None:
        con.run("CREATE TABLE sales(product VARCHAR, quarter VARCHAR, amount INT)")
        con.run("INSERT INTO sales VALUES ('a', 'q1', 1), ('a', 'q2', 2)")
        table = sql("PIVOT sales ON quarter USING sum(amount)").to_arrow(con)
        assert table.column_names == ["product", "q1", "q2"]
        assert table.num_rows == 1


class TestReentrantClose:
    # A regression in any of these deadlocks rather than fails, so each runs in a bounded subprocess.

    def test_a_signal_handler_may_close_the_stream_it_interrupts(self) -> None:
        probe = textwrap.dedent(
            """
            import os, signal, threading
            import duckdb
            import pyarrow as pa

            con = duckdb.frame.connect()
            in_flight = threading.Event()
            con.create_function("mark", lambda v: (in_flight.set(), v)[1], ["BIGINT"], "BIGINT")
            stream = con._execute_arrow(
                "SELECT mark(i) AS v FROM range(20_000_000_000) t(i) ORDER BY v DESC", None, 0
            )
            reader = pa.RecordBatchReader.from_stream(stream)

            def handler(*_):
                stream.close()
                print("handler returned", flush=True)

            signal.signal(signal.SIGINT, handler)

            def send_signal():
                in_flight.wait(10)
                os.kill(os.getpid(), signal.SIGINT)

            threading.Thread(target=send_signal).start()
            try:
                reader.read_next_batch()
            except Exception as error:
                print(type(error).__name__, flush=True)
            con.close()
            print("ok", flush=True)
            """
        )
        done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=60, check=False)
        assert done.returncode == 0, done.stderr
        assert done.stdout.splitlines()[-1] == "ok", done.stdout

    def test_a_scalar_function_may_close_its_own_stream(self) -> None:
        probe = textwrap.dedent(
            """
            import duckdb
            import pyarrow as pa

            con = duckdb.frame.connect()
            stream = None

            def close_self(value):
                stream.close()
                return value

            con.create_function("close_self", close_self, ["BIGINT"], "BIGINT")
            stream = con._execute_arrow("SELECT close_self(i) FROM range(10) t(i)", None, 0)
            try:
                pa.RecordBatchReader.from_stream(stream).read_all()
            except Exception as error:
                print(type(error).__name__, flush=True)
            con.close()
            print("ok", flush=True)
            """
        )
        done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=60, check=False)
        assert done.returncode == 0, done.stderr
        assert done.stdout.splitlines()[-1] == "ok", done.stdout

    def test_a_scalar_function_may_close_the_reader_it_feeds(self) -> None:
        # Closing through pyarrow releases the stream from inside its own read; whichever thread the function
        # runs on, the release must neither clear the struct under the live frames nor wait for them forever.
        probe = textwrap.dedent(
            """
            import duckdb
            import pyarrow as pa

            for threads in ("1", "4"):
                con = duckdb.frame.connect(threads=threads)
                holder = {}

                def close_own_reader(value):
                    holder["reader"].close()
                    return value

                con.create_function("close_own_reader", close_own_reader, ["BIGINT"], "BIGINT")
                stream = con._execute_arrow("SELECT close_own_reader(i) FROM range(1000) t(i)", None, 10)
                holder["reader"] = pa.RecordBatchReader.from_stream(stream)
                try:
                    holder["reader"].read_all()
                except Exception as error:
                    print(threads, type(error).__name__, flush=True)
                con.close()
            print("ok", flush=True)
            """
        )
        done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=90, check=False)
        assert done.returncode == 0, done.stderr
        assert done.stdout.splitlines()[-1] == "ok", done.stdout

    def test_closing_the_handle_after_a_mid_read_release_does_not_crash(self) -> None:
        # A release during a read quarantines the capsule holder rather than freeing it, since the consumer's
        # trailing calls have no last-call marker; a close of the stream handle right after must not free it
        # out from under them either.
        probe = textwrap.dedent(
            """
            import duckdb
            import pyarrow as pa

            holder = {}

            def close_both(value):
                holder["reader"].close()
                holder["stream"].close()
                return value

            con = duckdb.frame.connect()
            con.create_function("close_both", close_both, ["BIGINT"], "BIGINT")
            holder["stream"] = con._execute_arrow("SELECT close_both(i) FROM range(1000) t(i)", None, 10)
            holder["reader"] = pa.RecordBatchReader.from_stream(holder["stream"])
            try:
                holder["reader"].read_all()
            except Exception as error:
                print(type(error).__name__, flush=True)
            con.close()
            print("ok", flush=True)
            """
        )
        done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=90, check=False)
        assert done.returncode == 0, done.stderr
        assert done.stdout.splitlines()[-1] == "ok", done.stdout

    def test_a_connections_next_statement_completes_a_deferred_close(self) -> None:
        # A close from an engine thread can only cancel and defer when the stream is idle; the stream's own
        # connection is the cleanup owner of last resort, finishing the teardown before its next statement.
        a = duckdb.frame.connect(threads="1")
        b = duckdb.frame.connect(threads="1")
        try:
            reader = sql("SELECT i FROM range(10) t(i)").to_reader(a)
            held = next(iter(a._live_streams))

            def close_a(value: int) -> int:
                held.close()
                return value

            b.create_function("close_a", close_a, ["BIGINT"], "BIGINT")
            assert sql("SELECT min(close_a(i)) AS v FROM range(3) t(i)").rows(b) == [(0,)]
            assert sql("SELECT 2").rows(a) == [(2,)]
            with pytest.raises(OSError, match="result is closed"):
                reader.read_all()
        finally:
            a.close()
            b.close()

    def test_close_waits_out_a_slow_callback(self, con: duckdb.frame.Connection) -> None:
        # The free-connection promise has no silent bound: a close during a scalar function that takes a few
        # seconds returns only once the teardown is really done.
        resume = threading.Event()
        entered = threading.Event()

        def pause(value: int) -> int:
            entered.set()
            resume.wait(10)
            return value

        con.create_function("pause", pause, ["BIGINT"], "BIGINT")
        held = con._execute_arrow("SELECT pause(i) AS v FROM range(1) t(i)", None, 1)
        reader = pa.RecordBatchReader.from_stream(held)

        def read() -> None:
            with contextlib.suppress(Exception):
                reader.read_all()

        worker = threading.Thread(target=read)
        worker.start()
        assert entered.wait(10)
        threading.Timer(2.0, resume.set).start()
        started = time.monotonic()
        held.close()
        waited = time.monotonic() - started
        assert waited > 1.5, "close returned before the callback let the read end"
        assert held.stream.live is False
        assert sql("SELECT 1").rows(con) == [(1,)]
        worker.join(10)
        assert not worker.is_alive()

    def test_ctrl_c_breaks_out_of_a_blocked_close(self, con: duckdb.frame.Connection) -> None:
        # A close is a join, and a join on work that never ends must still answer Ctrl-C; the request then
        # stands and the connection's next statement completes it.
        from ._support import ctrl_c_after

        resume = threading.Event()
        entered = threading.Event()

        def pause(value: int) -> int:
            entered.set()
            resume.wait(30)
            return value

        con.create_function("pause", pause, ["BIGINT"], "BIGINT")
        held = con._execute_arrow("SELECT pause(i) AS v FROM range(1) t(i)", None, 1)
        reader = pa.RecordBatchReader.from_stream(held)

        def read() -> None:
            with contextlib.suppress(Exception):
                reader.read_all()

        worker = threading.Thread(target=read)
        worker.start()
        assert entered.wait(10)
        started = time.monotonic()
        with pytest.raises(KeyboardInterrupt), ctrl_c_after(0.5, resume.set):
            held.close()
        assert time.monotonic() - started < 8, "the close did not poll for the interrupt"
        assert held.stream.close_pending
        resume.set()
        worker.join(10)
        assert not worker.is_alive()
        assert sql("SELECT 1").rows(con) == [(1,)]

    def test_a_failed_registration_does_not_pin_the_callable(self, con: duckdb.frame.Connection) -> None:
        def orphan(value: int) -> int:
            return value

        with pytest.raises(duckdb.exceptions.Error):
            con.create_function("orphan", orphan, ["NO SUCH TYPE"], "BIGINT")
        gone = weakref.ref(orphan)
        del orphan
        gc.collect()
        assert gone() is None, "the rejected callable stayed pinned"

    def test_two_closers_both_find_the_connection_free(self, con: duckdb.frame.Connection) -> None:
        # Whichever closer loses the race to drive the teardown still waits for it, so each one's own next
        # statement finds the slot free; the follow-ups are serialised so they do not collide with each other.
        def read(reader: pa.RecordBatchReader) -> None:
            with contextlib.suppress(Exception):
                reader.read_next_batch()

        def close_and_query(
            held: duckdb.frame.connection.LiveArrowStream, follow_up: threading.Lock, outcomes: list[str]
        ) -> None:
            held.close()
            with follow_up:
                try:
                    sql("SELECT 1").rows(con)
                    outcomes.append("ok")
                except Exception:
                    outcomes.append("busy")

        for _ in range(3):
            held, in_flight = blocked_stream(con)
            reader = pa.RecordBatchReader.from_stream(held)
            worker = threading.Thread(target=read, args=(reader,))
            worker.start()
            assert in_flight.wait(10), "the read never reached the engine"
            outcomes: list[str] = []
            follow_up = threading.Lock()
            closers = [threading.Thread(target=close_and_query, args=(held, follow_up, outcomes)) for _ in range(2)]
            for closer in closers:
                closer.start()
            for closer in closers:
                closer.join(15)
            worker.join(10)
            assert not worker.is_alive()
            assert outcomes == ["ok", "ok"]

    def test_releasing_the_reader_during_a_read_does_not_crash(self) -> None:
        # The Arrow contract forbids releasing a stream while a read runs, but breaking it must fail softly:
        # the release cancels and waits out the read, so the consumer's error path finds live callbacks.
        probe = textwrap.dedent(
            """
            import threading
            import duckdb
            import pyarrow as pa

            con = duckdb.frame.connect()
            in_flight = threading.Event()
            con.create_function("mark", lambda v: (in_flight.set(), v)[1], ["BIGINT"], "BIGINT")
            stream = con._execute_arrow(
                "SELECT mark(i) AS v FROM range(20_000_000_000) t(i) ORDER BY v DESC", None, 0
            )
            reader = pa.RecordBatchReader.from_stream(stream)

            def read():
                try:
                    reader.read_next_batch()
                except Exception as error:
                    print(type(error).__name__, flush=True)

            worker = threading.Thread(target=read)
            worker.start()
            assert in_flight.wait(10), "the read never reached the engine"
            reader.close()
            worker.join(30)
            assert not worker.is_alive()
            con.close()
            print("ok", flush=True)
            """
        )
        done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=60, check=False)
        assert done.returncode == 0, done.stderr
        assert done.stdout.splitlines()[-1] == "ok", done.stdout

    def test_close_frees_the_connection_before_returning(self, con: duckdb.frame.Connection) -> None:
        # Close waits for the cancelled read's teardown, so the very next statement finds the slot free.
        def read(reader: pa.RecordBatchReader) -> None:
            with contextlib.suppress(Exception):
                reader.read_next_batch()

        for _ in range(5):
            held, in_flight = blocked_stream(con)
            reader = pa.RecordBatchReader.from_stream(held)
            worker = threading.Thread(target=read, args=(reader,))
            worker.start()
            assert in_flight.wait(10), "the read never reached the engine"
            held.close()
            assert sql("SELECT 1").rows(con) == [(1,)]
            worker.join(10)
            assert not worker.is_alive()

    def test_a_finalizer_during_error_cleanup_may_close_the_reader(self) -> None:
        # The error teardown drops the stream's pins before the failing callback returns to the consumer; a
        # finalizer running then can release the stream, and the struct must stay whole for the consumer's
        # own error path afterwards.
        probe = textwrap.dedent(
            """
            import duckdb
            import pyarrow as pa

            reader = None

            class Marker:
                def __arrow_c_stream__(self, requested_schema=None):
                    return pa.table({"x": [1]}).__arrow_c_stream__()

                def __del__(self):
                    reader.close()
                    print("finalizer returned", flush=True)

            def make():
                con = duckdb.frame.connect()
                con.register("marker", Marker())
                plan = "SELECT CASE WHEN i < 4 THEN i ELSE CAST(error('boom') AS BIGINT) END AS v FROM range(10) t(i)"
                return duckdb.frame.sql(plan).to_reader(con, batch_size=2)

            reader = make()
            try:
                reader.read_all()
            except Exception as error:
                print(type(error).__name__, flush=True)
            print("ok", flush=True)
            """
        )
        done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=60, check=False)
        assert done.returncode == 0, done.stderr
        assert "finalizer returned" in done.stdout, done.stdout
        assert done.stdout.splitlines()[-1] == "ok", done.stdout

    def test_exporting_from_a_stream_callback_is_refused(self) -> None:
        # A signal handler running inside the stream's own read already holds the stream's lock; the export
        # must refuse rather than deadlock on it.
        probe = textwrap.dedent(
            """
            import os, signal, threading
            import duckdb
            import pyarrow as pa

            con = duckdb.frame.connect()
            stream = con._execute_arrow("SELECT i FROM range(20_000_000_000) t(i) ORDER BY i DESC", None, 0)
            reader = pa.RecordBatchReader.from_stream(stream)

            def handler(*_):
                try:
                    stream.stream.__arrow_c_stream__()
                    print("exported", flush=True)
                except Exception as error:
                    print("refused", type(error).__name__, flush=True)
                stream.close()

            signal.signal(signal.SIGINT, handler)
            threading.Timer(0.3, os.kill, (os.getpid(), signal.SIGINT)).start()
            try:
                reader.read_next_batch()
            except Exception as error:
                print(type(error).__name__, flush=True)
            con.close()
            print("ok", flush=True)
            """
        )
        done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=60, check=False)
        assert done.returncode == 0, done.stderr
        assert "refused InterfaceError" in done.stdout, done.stdout
        assert done.stdout.splitlines()[-1] == "ok", done.stdout

    def test_a_registered_source_may_close_the_stream_consuming_it(self) -> None:
        # Scan callbacks run user Python on engine threads outside scalar functions too; a close reached from
        # there must not wait on the very thread the reader needs.
        probe = textwrap.dedent(
            """
            import duckdb
            import pyarrow as pa

            holder = {}

            class SelfCloser:
                def __arrow_c_stream__(self, requested_schema=None):
                    stream = holder.get("stream")
                    if stream is not None:
                        stream.close()
                    return pa.table({"x": list(range(1000))}).__arrow_c_stream__()

            con = duckdb.frame.connect()
            con.register("self_closer", SelfCloser())
            holder["stream"] = con._execute_arrow("SELECT x FROM self_closer", None, 0)
            try:
                # The close can land during either call; both must fail loudly rather than hang or crash.
                pa.RecordBatchReader.from_stream(holder["stream"]).read_all()
            except Exception as error:
                print(type(error).__name__, flush=True)
            con.close()
            print("ok", flush=True)
            """
        )
        done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=60, check=False)
        assert done.returncode == 0, done.stderr
        assert done.stdout.splitlines()[-1] == "ok", done.stdout

    def test_a_finalizer_during_stream_teardown_may_close_the_reader(self) -> None:
        # The database pin drops when the stream ends; the finalizers that may run then must find the stream's
        # lock free.
        probe = textwrap.dedent(
            """
            import duckdb
            import pyarrow as pa

            reader = None

            class Marker:
                def __arrow_c_stream__(self, requested_schema=None):
                    return pa.table({"x": [1]}).__arrow_c_stream__()

                def __del__(self):
                    reader.close()
                    print("finalizer returned", flush=True)

            def make_reader():
                con = duckdb.frame.connect()
                con.register("marker", Marker())
                return duckdb.frame.sql("SELECT 1 AS n").to_reader(con)

            reader = make_reader()
            reader.read_all()
            print("ok", flush=True)
            """
        )
        done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=60, check=False)
        assert done.returncode == 0, done.stderr
        assert "finalizer returned" in done.stdout, done.stdout
        assert done.stdout.splitlines()[-1] == "ok", done.stdout


class TestExhaustion:
    def test_a_reader_read_to_the_end_frees_the_connection(self, con: duckdb.frame.Connection) -> None:
        reader = sql("SELECT i FROM range(10) t(i)").to_reader(con)
        reader.read_all()
        assert sql("SELECT 1").rows(con) == [(1,)]

    def test_reading_past_the_end_stays_a_clean_end(self, con: duckdb.frame.Connection) -> None:
        reader = sql("SELECT i FROM range(10) t(i)").to_reader(con)
        reader.read_all()
        for _ in range(2):
            with pytest.raises(StopIteration):
                reader.read_next_batch()

    def test_exhausted_readers_still_referenced_are_pruned(self, con: duckdb.frame.Connection) -> None:
        readers = []
        for _ in range(20):
            reader = sql("SELECT 1 AS a").to_reader(con)
            reader.read_all()
            readers.append(reader)
        sql("SELECT 2").to_arrow(con)
        assert len(con._live_streams) <= 1


class TestDatabaseRelease:
    def test_a_closed_connections_file_reopens_despite_a_retained_reader(self, tmp_path: object) -> None:
        path = f"{tmp_path}/pin.db"
        con = duckdb.frame.connect(path)
        con.run("CREATE TABLE t AS SELECT i FROM range(1000) t(i)")
        reader = sql("SELECT i FROM t").to_reader(con)
        con.close()
        duckdb.frame.connect(path).close()
        with pytest.raises(OSError, match="result is closed"):
            reader.read_next_batch()

    def test_a_cycle_through_an_unconsumed_stream_collects(self) -> None:
        # The stream's own hold on the database is reported to the collector while no capsule is out, so this
        # cycle is visible and collectable.
        def make_cycle() -> weakref.ref[object]:
            con = duckdb.frame.connect()
            box: dict[str, object] = {}

            def keep(value: int) -> int:
                box["last"] = value
                return value

            con.create_function("keep", keep, ["BIGINT"], "BIGINT")
            box["stream"] = con._engine().execute_arrow("SELECT 1")
            con.close()
            return weakref.ref(keep)

        gone = make_cycle()
        gc.collect()
        assert gone() is None, "the unconsumed stream kept its database pinned"

    def test_a_closed_stream_object_releases_the_database(self, tmp_path: object) -> None:
        path = f"{tmp_path}/pin2.db"
        con = duckdb.frame.connect(path)
        con.run("CREATE TABLE t AS SELECT 1 AS x")
        stream = con._engine().execute_arrow("SELECT x FROM t")
        stream.close()
        con.close()
        duckdb.frame.connect(path).close()
        assert stream.live is False

    def test_a_reader_with_a_scalar_function_outlives_its_database(self) -> None:
        # The stream pins the registered callables, not the Database object, so the reader keeps working
        # after everything else is dropped.
        def build() -> pa.RecordBatchReader:
            con = duckdb.frame.connect()
            con.create_function("double_it", lambda v: v * 2, ["BIGINT"], "BIGINT")
            return duckdb.frame.sql("SELECT double_it(i) AS v FROM range(1000) t(i)").to_reader(con, batch_size=100)

        reader = build()
        gc.collect()
        table = reader.read_all()
        assert table.num_rows == 1000
        assert table.column("v")[-1].as_py() == 1998

    def test_an_exported_reader_cycle_frees_the_database(self) -> None:
        # A database-owned function retaining its own unread reader keeps the function and reader alive, which
        # the open query needs, but not the database: nothing may report a leaked database at exit.
        probe = textwrap.dedent(
            """
            import gc, weakref
            import duckdb

            def build():
                con = duckdb.frame.connect()
                box = {}

                def keep(v):
                    box.get("reader")
                    return v

                con.create_function("keep", keep, ["BIGINT"], "BIGINT")
                box["reader"] = duckdb.frame.sql("SELECT keep(i) AS v FROM range(100000) t(i)").to_reader(con)
                return weakref.ref(con), box["reader"]

            connection, reader = build()
            gc.collect()
            print("connection alive", connection() is not None, flush=True)
            # Ends the still-running query, so the interpreter does not exit over a live executor.
            reader.close()
            """
        )
        done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=60, check=False)
        assert done.returncode == 0, done.stderr
        assert "connection alive False" in done.stdout, done.stdout
        assert "leaked" not in done.stderr, done.stderr

    def test_a_cycle_through_an_exhausted_reader_collects(self) -> None:
        # The capsule pins the database while the query can still call what it owns; a cycle such as this one
        # is invisible to the collector through the capsule, so an ended stream must have dropped the pin.
        con = duckdb.frame.connect()
        reader = sql("SELECT i FROM range(10) t(i)").to_reader(con)

        def keep(v: int, reader: object = reader) -> int:
            return v

        con.create_function("keep", keep, ["BIGINT"], "BIGINT")
        reader.read_all()
        gone = weakref.ref(keep)
        del keep, reader
        con.close()
        del con
        gc.collect()
        assert gone() is None, "the stream kept its database pinned after the read ended"


class TestErrors:
    def test_a_mid_stream_error_is_typed_through_to_arrow(self, con: duckdb.frame.Connection) -> None:
        plan = sql("SELECT CASE WHEN i = 9000 THEN error('boom late') ELSE i::VARCHAR END AS v FROM range(10000) t(i)")
        with pytest.raises(exceptions.InvalidInputError, match="boom late"):
            plan.to_arrow(con)
        assert sql("SELECT 1").rows(con) == [(1,)]

    def test_a_bare_reader_surfaces_pyarrows_error_with_the_engines_text(self, con: duckdb.frame.Connection) -> None:
        reader = sql(
            "SELECT CASE WHEN i = 9000 THEN error('boom late') ELSE i::VARCHAR END AS v FROM range(10000) t(i)"
        ).to_reader(con, batch_size=1000)
        with pytest.raises(OSError, match="boom late"):
            reader.read_all()

    def test_closing_the_connection_fails_the_next_read_loudly(self, con: duckdb.frame.Connection) -> None:
        # Never a quiet end of stream: a read past the close must fail, not look complete.
        reader = sql("SELECT i FROM range(1000000) t(i)").to_reader(con, batch_size=100)
        assert reader.read_next_batch().num_rows == 100
        con.close()
        with pytest.raises(OSError, match="result is closed"):
            reader.read_all()

    def test_an_interrupt_from_another_thread_lands_typed(self, con: duckdb.frame.Connection) -> None:
        # One interrupt at a fixed delay can land before the query starts and leave it running, so repeat.
        stop = threading.Event()

        def keep_interrupting() -> None:
            while not stop.is_set():
                con.interrupt()
                time.sleep(0.05)

        worker = threading.Thread(target=keep_interrupting)
        worker.start()
        try:
            with pytest.raises(exceptions.InterruptError):
                sql("SELECT count(*) FROM range(100_000_000_000)").to_arrow(con)
        finally:
            stop.set()
            worker.join()


class TestCloseUnderLoad:
    def test_closing_during_a_read_with_a_python_function_does_not_deadlock(self) -> None:
        # The reader holds the stream's lock across engine work that needs the GIL for the function; a close
        # that waited for it holding the GIL would starve the function forever.
        probe = textwrap.dedent(
            """
            import threading
            import duckdb

            for round in range(60):
                con = duckdb.frame.connect()
                con.create_function("slow", lambda v: v, ["BIGINT"], "BIGINT")
                reader = duckdb.frame.sql("SELECT slow(i) AS v FROM range(3000000) t(i)").to_reader(
                    con, batch_size=2048
                )

                def read(r=reader):
                    try:
                        while r.read_next_batch() is not None:
                            pass
                    except Exception:
                        pass

                worker = threading.Thread(target=read)
                worker.start()
                worker.join(0.005 * (round % 6))
                con.close()
                worker.join(30)
                assert not worker.is_alive(), "the reader never came back after close"
            print("ok")
            """
        )
        done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=240, check=False)
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == "ok"

    def test_closing_interrupts_a_read_waiting_on_its_first_batch(self, con: duckdb.frame.Connection) -> None:
        # An ORDER BY over this range produces nothing for hours; close must cancel it, not wait for it.
        held, in_flight = blocked_stream(con)
        reader = pa.RecordBatchReader.from_stream(held)
        failed = threading.Event()

        def read() -> None:
            try:
                reader.read_next_batch()
            except Exception:
                failed.set()

        worker = threading.Thread(target=read)
        worker.start()
        assert in_flight.wait(10), "the read never reached the engine"
        started = time.monotonic()
        con.close()
        assert time.monotonic() - started < 10, "close waited for a batch that never comes"
        worker.join(10)
        assert not worker.is_alive()
        assert failed.is_set()

    def test_live_answers_while_a_read_is_blocked(self, con: duckdb.frame.Connection) -> None:
        # A monitoring poll must not sit behind the read in flight, whose next batch may be hours away.
        held, in_flight = blocked_stream(con)
        reader = pa.RecordBatchReader.from_stream(held)

        def read() -> None:
            # The close below ends the read; any exit is fine.
            with contextlib.suppress(Exception):
                reader.read_next_batch()

        worker = threading.Thread(target=read)
        worker.start()
        assert in_flight.wait(10), "the read never reached the engine"
        started = time.monotonic()
        alive = held.stream.live
        assert time.monotonic() - started < 1, "live blocked behind the read in flight"
        assert alive
        assert held.stream.error is None
        held.close()
        worker.join(10)
        assert not worker.is_alive()

    def test_closing_a_finished_stream_leaves_the_next_query_alone(self, con: duckdb.frame.Connection) -> None:
        # Only a read in flight justifies cancelling a query; lock contention from error or liveness polling
        # on a finished stream must not reach whatever the connection runs next.
        held = con._execute_arrow("SELECT i FROM range(100) t(i)", None, 0)
        pa.RecordBatchReader.from_stream(held).read_all()
        held.close()
        stop = threading.Event()

        def poll() -> None:
            while not stop.is_set():
                _ = held.stream.live

        def close() -> None:
            while not stop.is_set():
                held.close()

        hammers = [threading.Thread(target=poll), threading.Thread(target=close)]
        for hammer in hammers:
            hammer.start()
        try:
            assert sql("SELECT sum(i) FROM range(200_000_000) t(i)").rows(con) == [(19999999900000000,)]
        finally:
            stop.set()
            for hammer in hammers:
                hammer.join()

    def test_finished_streams_do_not_accumulate(self, con: duckdb.frame.Connection) -> None:
        for _ in range(5):
            sql("SELECT i FROM range(10) t(i)").to_reader(con).read_all()
        sql("SELECT 1").to_reader(con).read_all()
        assert len(con._live_streams) <= 2

    def test_a_read_after_an_interrupt_keeps_the_interrupts_code(self, con: duckdb.frame.Connection) -> None:
        stream = con._execute_arrow("SELECT i FROM range(1000000) t(i)", None, 100)
        reader = pa.RecordBatchReader.from_stream(stream)
        assert reader.read_next_batch().num_rows == 100
        stream.close()
        with pytest.raises(OSError, match="result is closed"):
            reader.read_all()
        with pytest.raises(OSError, match="result is closed"):
            reader.read_all()
        assert stream.error == (-1, "result is closed")

    def test_a_thread_interrupt_stays_the_recorded_cause(self, con: duckdb.frame.Connection) -> None:
        # The first cause is sticky: later reads must not relabel an interrupt as anything else.
        held, in_flight = blocked_stream(con)
        reader = pa.RecordBatchReader.from_stream(held)

        def interrupt_once() -> None:
            in_flight.wait(10)
            con.interrupt()

        threading.Thread(target=interrupt_once).start()
        with pytest.raises(OSError, match="cancel"):
            reader.read_next_batch()
        first = held.error
        assert first is not None
        assert first[0] > 0
        with pytest.raises(OSError, match="cancel"):
            reader.read_next_batch()
        assert held.error == first


class TestCtrlC:
    def test_a_keyboard_interrupt_stops_a_table_read(self, con: duckdb.frame.Connection) -> None:
        # The stream ends with the engine's cadence and the signal is re-armed, so Python raises it here.
        from ._support import ctrl_c_after

        plan = sql("SELECT i FROM range(100_000_000_000) t(i)")
        started = time.monotonic()
        with pytest.raises(KeyboardInterrupt), ctrl_c_after(0.3, con.interrupt):
            plan.to_arrow(con)
        assert time.monotonic() - started < 5, "the interrupt did not land while the read ran"
        assert sql("SELECT 1").rows(con) == [(1,)]

    def test_a_keyboard_interrupt_lands_before_the_first_batch(self, con: duckdb.frame.Connection) -> None:
        # A pipeline breaker's first array may be minutes away; the read sees the signal between bounded
        # pieces of engine work rather than waiting for an array to land.
        from ._support import ctrl_c_after

        plan = sql("SELECT i FROM range(5_000_000_000) t(i) ORDER BY i DESC")
        started = time.monotonic()
        with pytest.raises(KeyboardInterrupt), ctrl_c_after(0.3, con.interrupt):
            plan.to_arrow(con)
        assert time.monotonic() - started < 8, "the interrupt waited for a batch instead of the signal"
        assert sql("SELECT 1").rows(con) == [(1,)]

    def test_another_handlers_exception_ends_the_read_typed(self, con: duckdb.frame.Connection) -> None:
        # A handler that raises something other than KeyboardInterrupt is not a Ctrl-C: its error crosses the
        # stream's error channel, and the array the step just produced goes back, since ownership never moved.
        def refuse(signum: int, frame: object) -> None:
            message = "handler says no"
            raise ValueError(message)

        previous = signal.signal(signal.SIGINT, refuse)
        interrupter = threading.Timer(0.2, os.kill, (os.getpid(), signal.SIGINT))
        interrupter.start()
        try:
            with pytest.raises(exceptions.InterfaceError, match="handler says no"):
                sql("SELECT i FROM range(10_000_000_000) t(i)").to_arrow(con)
        finally:
            interrupter.cancel()
            signal.signal(signal.SIGINT, previous)
        assert sql("SELECT 1").rows(con) == [(1,)]


class TestGilAndThreads:
    def test_a_python_function_feeds_a_pyarrow_read(self) -> None:
        # pyarrow drives the read holding the GIL while the query needs it for the function: a regression here
        # deadlocks rather than fails, so it runs in its own process under a bound.
        probe = textwrap.dedent(
            """
            import duckdb
            import pyarrow as pa

            con = duckdb.frame.connect()
            con.create_function("twice", lambda v: v * 2, ["BIGINT"], "BIGINT")
            table = duckdb.frame.sql("SELECT twice(i) AS v FROM range(100000) t(i)").to_arrow(con)
            assert table.num_rows == 100000
            assert table.column("v").to_pylist()[:3] == [0, 2, 4]
            print("ok")
            """
        )
        done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=60, check=False)
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == "ok"

    def test_two_threads_export_on_their_own_connections(self) -> None:
        out: dict[int, int] = {}

        def export(key: int) -> None:
            with duckdb.frame.connect() as con:
                out[key] = sql("SELECT i FROM range(10000) t(i)").to_arrow(con).num_rows

        threads = [threading.Thread(target=export, args=(key,)) for key in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert out == {0: 10000, 1: 10000}


# The capsule-without-pyarrow behavior lives in test_arrow_egress_no_pyarrow.py: this module requires pyarrow,
# which would skip exactly the case that matters.
