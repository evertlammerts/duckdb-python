import importlib.util
import subprocess
import sys
import textwrap
import threading
import time

import pytest

import duckdb
from duckdb.sqltypes import BIGINT

ROWS = 1_000_000
VECTOR = 2048
# An undecided result parks its first chunk at the sink, so settling runs about one vector.
SETTLED = 4 * VECTOR
OPEN_RESULT = "open result"

needs_polars = pytest.mark.skipif(importlib.util.find_spec("polars") is None, reason="polars not installed")
needs_pandas = pytest.mark.skipif(importlib.util.find_spec("pandas") is None, reason="pandas not installed")
needs_pyarrow = pytest.mark.skipif(importlib.util.find_spec("pyarrow") is None, reason="pyarrow not installed")

FAILED = (AssertionError, pytest.fail.Exception)
DEFERS = pytest.mark.xfail(strict=True, raises=FAILED, reason="execute() defers SELECT-class statements")
SUPERSEDES = pytest.mark.xfail(
    strict=True, raises=FAILED, reason="other users of a connection supersede its open result instead of erroring"
)
DRAINS = pytest.mark.xfail(
    strict=True,
    raises=duckdb.TransactionException,
    reason="a superseded statement that may write is cancelled, which invalidates the transaction",
)
CANCELS = pytest.mark.xfail(
    strict=True, raises=FAILED, reason="two lazy polars sources of one connection cancel each other"
)
HANGS = pytest.mark.xfail(
    strict=True, raises=subprocess.TimeoutExpired, reason="re-entering a connection with an open result deadlocks"
)


@pytest.fixture
def con():
    connection = duckdb.connect()
    # One thread bounds how far a statement runs before it is read.
    connection.execute("SET threads = 1")
    yield connection
    connection.close()


@pytest.fixture
def produced(con):
    count = [0]

    def tally(i):
        count[0] += 1
        return i

    con.create_function("tally", tally, [BIGINT], BIGINT, side_effects=True)
    return count


def run_isolated(script):
    # A child process keeps a deadlock from stalling the suite; the marker tells a hang from a slow start.
    try:
        proc = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)], capture_output=True, text=True, timeout=15, check=False
        )
    except subprocess.TimeoutExpired as hang:
        started = hang.stdout or ""
        if "started" not in (started.decode() if isinstance(started, bytes) else started):
            msg = "the child never reached the statement under test"
            raise AssertionError(msg) from hang
        raise
    return proc.returncode, proc.stdout + proc.stderr


class TestExecuteSettles:
    @DEFERS
    def test_an_error_in_the_first_chunk_raises_from_execute(self, con):
        with pytest.raises(duckdb.InvalidInputException, match="boom"):
            con.execute("SELECT error('boom')")

    @DEFERS
    def test_a_parameterized_error_raises_from_execute(self, con):
        with pytest.raises(duckdb.InvalidInputException, match="boom"):
            con.execute("SELECT error(?)", ["boom"])

    @DEFERS
    def test_the_last_statement_of_a_script_settles_too(self, con):
        with pytest.raises(duckdb.InvalidInputException, match="boom"):
            con.execute("SELECT 1; SELECT error('boom')")

    def test_a_failing_statement_inside_a_script_stops_it(self, con):
        with pytest.raises(duckdb.InvalidInputException, match="boom"):
            con.execute("CREATE TABLE a (i INT); SELECT error('boom'); CREATE TABLE b (i INT)")
        tables = con.execute("SELECT table_name FROM duckdb_tables() ORDER BY 1").fetchall()
        assert tables == [("a",)]

    @pytest.mark.parametrize(("parameter_sets", "ran"), [([[1], [2]], 0), ([[2], [1]], 1)])
    def test_executemany_stops_at_the_failing_parameter_set(self, con, produced, parameter_sets, ran):
        with pytest.raises(duckdb.InvalidInputException, match="boom"):
            con.executemany("SELECT tally(CASE WHEN ? = 1 THEN error('boom')::BIGINT ELSE 0 END)", parameter_sets)
        assert produced[0] == ran

    @DEFERS
    def test_the_first_chunk_runs_and_the_rest_waits(self, con, produced):
        con.execute(f"SELECT tally(i) FROM range({ROWS}) t(i)")
        assert 0 < produced[0] <= SETTLED

    def test_settling_loses_no_rows(self, con):
        con.execute(f"SELECT * FROM range({ROWS})")
        assert [column[0] for column in con.description] == ["range"]
        rows = con.fetchall()
        assert len(rows) == ROWS
        assert rows[:3] == [(0,), (1,), (2,)]
        assert rows[-1] == (ROWS - 1,)

    def test_an_empty_result_settles(self, con):
        con.execute("SELECT * FROM range(0)")
        assert con.fetchall() == []

    def test_an_error_after_the_first_chunk_waits_for_the_fetch(self, con):
        con.execute("BEGIN")
        con.execute(f"SELECT CASE WHEN i = {ROWS - 1} THEN error('late') ELSE i END FROM range({ROWS}) t(i)")
        with pytest.raises(duckdb.InvalidInputException, match="late"):
            con.fetchall()
        with pytest.raises(duckdb.TransactionException):
            con.execute("SELECT 1")
        con.execute("ROLLBACK")
        assert con.execute("SELECT 42").fetchall() == [(42,)]

    @DEFERS
    @pytest.mark.timeout(60)
    def test_an_interrupt_while_settling_raises_from_execute(self, con):
        timer = threading.Timer(0.5, con.interrupt)
        timer.start()
        try:
            with pytest.raises(duckdb.InterruptException):
                # An aggregate emits its only row at the end and needs no memory while it runs.
                con.execute("SELECT sum(a.range * b.range) FROM range(200000) a, range(200000) b")
        finally:
            timer.cancel()

    RETURNING = pytest.mark.parametrize(
        ("statement", "returned", "remaining"),
        [
            pytest.param(
                "INSERT INTO t SELECT range FROM range(5000, 10000) RETURNING i", range(5000, 10000), 10000, id="insert"
            ),
            pytest.param("DELETE FROM t WHERE i < 5000 RETURNING i", range(5000), 0, id="delete"),
        ],
    )

    @DEFERS
    @RETURNING
    def test_returning_completes_in_execute(self, con, statement, returned, remaining):
        con.execute("CREATE TABLE t AS SELECT range AS i FROM range(5000)")
        con.execute(statement)
        assert con.cursor().execute("SELECT count(*) FROM t").fetchall() == [(remaining,)]
        assert sorted(con.fetchall()) == [(i,) for i in returned]

    @DEFERS
    @RETURNING
    def test_unread_returning_survives_the_next_statement(self, con, statement, returned, remaining):
        con.execute("CREATE TABLE t AS SELECT range AS i FROM range(5000)")
        con.execute(statement)
        con.execute("SELECT 1")
        assert con.execute("SELECT count(*) FROM t").fetchall() == [(remaining,)]

    @pytest.mark.parametrize(
        "statement",
        [
            pytest.param("CREATE TABLE u AS SELECT {late} AS x FROM range({rows}) t(i)", id="ctas"),
            pytest.param("INSERT INTO t SELECT {late} FROM range({rows}) t(i)", id="insert"),
            pytest.param("UPDATE t SET i = {late}", id="update"),
        ],
    )
    def test_a_write_that_fails_late_raises_from_execute_and_leaves_no_trace(self, con, statement):
        con.execute(f"CREATE TABLE t AS SELECT range AS i FROM range({ROWS})")
        late = f"CASE WHEN i = {ROWS - 1} THEN error('boom')::BIGINT ELSE i + 1 END"
        with pytest.raises(duckdb.InvalidInputException, match="boom"):
            con.execute(statement.format(late=late, rows=ROWS))
        assert con.execute("SELECT count(*), sum(i) FROM t").fetchall() == [(ROWS, ROWS * (ROWS - 1) // 2)]
        assert con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name = 'u'").fetchall() == [(0,)]

    def test_a_copy_that_fails_late_raises_from_execute(self, con, tmp_path):
        late = f"CASE WHEN i = {ROWS - 1} THEN error('boom')::BIGINT ELSE i END"
        with pytest.raises(duckdb.InvalidInputException, match="boom"):
            con.execute(f"COPY (SELECT {late} FROM range({ROWS}) t(i)) TO '{(tmp_path / 'out.csv').as_posix()}'")

    @DEFERS
    def test_a_failed_select_aborts_the_transaction(self, con):
        con.execute("BEGIN")
        con.execute("CREATE TABLE t AS SELECT 1 AS a")
        with pytest.raises(duckdb.InvalidInputException):
            con.execute("SELECT error('boom')")
        with pytest.raises(duckdb.TransactionException):
            con.execute("SELECT 1")
        con.execute("ROLLBACK")
        assert con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name = 't'").fetchall() == [(0,)]


class TestSupersede:
    def test_a_new_statement_replaces_the_open_result(self, con):
        con.execute(f"SELECT * FROM range({ROWS})")
        con.fetchone()
        con.execute("SELECT 42")
        assert con.fetchall() == [(42,)]

    def test_a_superseded_read_only_select_leaves_the_transaction_valid(self, con):
        con.execute("BEGIN")
        con.execute("CREATE TABLE t AS SELECT 1 AS a")
        con.execute(f"SELECT * FROM range({ROWS})")
        con.fetchone()
        con.execute("INSERT INTO t VALUES (2)")
        con.execute("COMMIT")
        assert con.execute("SELECT a FROM t ORDER BY a").fetchall() == [(1,), (2,)]

    @DRAINS
    def test_a_superseded_statement_that_may_write_is_drained(self, con):
        con.execute("CREATE SEQUENCE s")
        con.execute("BEGIN")
        con.execute(f"SELECT nextval('s') FROM range({ROWS})")
        con.fetchone()
        assert con.execute("SELECT 42").fetchall() == [(42,)]
        con.execute("COMMIT")
        assert con.execute("SELECT currval('s')").fetchall() == [(ROWS,)]

    @DRAINS
    def test_an_error_in_the_drained_remainder_raises_from_the_next_execute(self, con):
        con.execute("CREATE SEQUENCE s")
        con.execute("BEGIN")
        late = f"CASE WHEN i = {ROWS - 1} THEN error('late')::BIGINT ELSE 0 END"
        con.execute(f"SELECT nextval('s') + {late} FROM range({ROWS}) t(i)")
        con.fetchone()
        with pytest.raises(duckdb.InvalidInputException, match="late"):
            con.execute("SELECT 42")
        with pytest.raises(duckdb.TransactionException):
            con.execute("SELECT 1")
        con.execute("ROLLBACK")

    @DEFERS
    def test_a_superseded_read_only_select_keeps_its_side_effects_and_runs_no_further(self, con, produced):
        con.execute(f"SELECT tally(i) FROM range({ROWS}) t(i)")
        settled = produced[0]
        assert con.execute("SELECT 42").fetchall() == [(42,)]
        time.sleep(0.5)
        assert 0 < settled <= SETTLED
        assert produced[0] == settled

    @pytest.mark.parametrize(
        "commit",
        [pytest.param(lambda c: c.execute("COMMIT"), id="statement"), pytest.param(lambda c: c.commit(), id="method")],
    )
    def test_commit_while_a_read_only_result_is_open(self, con, commit):
        con.execute("BEGIN")
        con.execute("CREATE TABLE t AS SELECT 1 AS a")
        con.execute(f"SELECT * FROM range({ROWS})")
        con.fetchone()
        commit(con)
        assert con.execute("SELECT a FROM t").fetchall() == [(1,)]


class TestOpenResultIsExclusive:
    @pytest.fixture
    def open_result(self, con):
        con.execute(f"SELECT * FROM range({ROWS})")
        con.fetchone()
        return con

    @SUPERSEDES
    @pytest.mark.parametrize(
        ("prepare", "run"),
        [
            pytest.param(lambda c: c.sql("SELECT 1"), lambda r: r.fetchall(), id="relation"),
            pytest.param(lambda c: c.table("t"), lambda r: r.fetchall(), id="table"),
            pytest.param(
                lambda c: c, lambda c: c.create_function("f", lambda x: x, [BIGINT], BIGINT), id="create_function"
            ),
        ],
    )
    def test_executing_anything_but_the_dbapi_errors(self, con, prepare, run):
        con.execute("CREATE TABLE t AS SELECT 1 AS a")
        con.execute(f"SELECT * FROM range({ROWS})")
        con.fetchone()
        # Building a relation only binds, so it succeeds; running it is what conflicts with the open result.
        target = prepare(con)
        with pytest.raises(duckdb.InvalidInputException, match=OPEN_RESULT):
            run(target)
        assert con.fetchone() == (1,)

    def test_constructing_relations_leaves_the_open_result_alone(self, con):
        con.execute("CREATE TABLE t AS SELECT 1 AS a")
        con.execute(f"SELECT * FROM range({ROWS})")
        con.fetchone()
        relations = [con.sql("SELECT 1 AS b"), con.table("t"), con.values([1, 2])]
        assert [r.columns for r in relations] == [["b"], ["a"], ["col0", "col1"]]
        assert con.fetchone() == (1,)

    @SUPERSEDES
    @needs_pandas
    def test_register_errors(self, con):
        import pandas as pd

        frame = pd.DataFrame({"a": [1]})
        con.execute(f"SELECT * FROM range({ROWS})")
        con.fetchone()
        with pytest.raises(duckdb.InvalidInputException, match=OPEN_RESULT):
            con.register("v", frame)
        assert con.fetchone() == (1,)

    @SUPERSEDES
    @pytest.mark.parametrize("release", ["complete", "abort", "fetchall"])
    def test_releasing_the_result_frees_the_connection(self, open_result, release):
        with pytest.raises(duckdb.InvalidInputException, match=OPEN_RESULT):
            open_result.sql("SELECT 1").fetchall()
        getattr(open_result, release)()
        assert open_result.sql("SELECT 42").fetchall() == [(42,)]

    def test_complete_runs_the_rest_of_the_result(self, con, produced):
        con.execute(f"SELECT tally(i) FROM range({ROWS}) t(i)")
        con.complete()
        assert produced[0] == ROWS

    def test_complete_keeps_a_transaction_valid_after_a_write(self, con):
        con.execute("CREATE SEQUENCE s")
        con.execute("BEGIN")
        con.execute(f"SELECT nextval('s') FROM range({ROWS})")
        con.complete()
        con.execute("COMMIT")
        assert con.execute("SELECT currval('s')").fetchall() == [(ROWS,)]

    def test_abort_of_a_statement_that_may_write_invalidates_the_transaction(self, con):
        con.execute("CREATE SEQUENCE s")
        con.execute("BEGIN")
        con.execute(f"SELECT nextval('s') FROM range({ROWS})")
        con.fetchone()
        con.abort()
        with pytest.raises(duckdb.TransactionException):
            con.execute("SELECT 1")
        con.execute("ROLLBACK")

    @DEFERS
    def test_abort_stops_the_result_where_it_settled(self, con, produced):
        con.execute(f"SELECT tally(i) FROM range({ROWS}) t(i)")
        settled = produced[0]
        con.abort()
        assert con.execute("SELECT 42").fetchall() == [(42,)]
        time.sleep(0.5)
        assert 0 < settled <= SETTLED
        assert produced[0] == settled

    def test_a_cursor_is_an_independent_connection(self, con):
        con.execute(f"SELECT * FROM range({ROWS})")
        assert con.fetchone() == (0,)
        cursor = con.cursor()
        assert cursor.execute("SELECT 42").fetchall() == [(42,)]
        assert con.fetchone() == (1,)
        cursor.execute(f"SELECT * FROM range({ROWS})")
        assert cursor.fetchone() == (0,)
        con.fetchall()
        assert con.sql("SELECT 43").fetchall() == [(43,)]
        assert cursor.fetchone() == (1,)

    @SUPERSEDES
    @pytest.mark.parametrize(
        "use",
        [
            pytest.param(lambda c: c.execute("SELECT 1"), id="execute"),
            pytest.param(lambda c: c.sql("SELECT 1").fetchall(), id="relation"),
        ],
    )
    @needs_pyarrow
    def test_a_relation_stream_blocks_its_connection(self, con, use):
        reader = con.sql(f"SELECT * FROM range({ROWS})").to_arrow_reader(1000)
        try:
            reader.read_next_batch()
            with pytest.raises(duckdb.InvalidInputException, match=OPEN_RESULT):
                use(con)
        finally:
            reader.close()
        assert con.execute("SELECT 42").fetchall() == [(42,)]

    @HANGS
    @needs_polars
    def test_a_lazy_polars_frame_scanned_on_its_own_connection_errors(self):
        code, output = run_isolated(
            """
            import duckdb
            con = duckdb.connect()
            con.execute("CREATE TABLE t AS SELECT range AS x FROM range(10)")
            lf = con.table("t").pl(lazy=True)
            print("started", flush=True)
            try:
                con.execute("SELECT count(*) FROM lf").fetchall()
            except Exception as error:
                print(type(error).__name__, error)
            """
        )
        assert code == 0, output
        assert "InvalidInputException" in output
        assert OPEN_RESULT in output

    @HANGS
    def test_a_udf_that_reenters_its_connection_errors(self):
        code, output = run_isolated(
            """
            import duckdb
            from duckdb.sqltypes import BIGINT
            con = duckdb.connect()
            con.create_function("inner", lambda x: con.execute("SELECT 1").fetchone()[0], [BIGINT], BIGINT)
            print("started", flush=True)
            try:
                con.execute("SELECT inner(1)").fetchall()
            except Exception as error:
                print(type(error).__name__, error)
            """
        )
        assert code == 0, output
        assert "InvalidInputException" in output
        assert OPEN_RESULT in output

    @CANCELS
    @needs_polars
    def test_two_lazy_polars_frames_of_one_connection_cannot_be_joined(self, con):
        con.execute("CREATE TABLE a AS SELECT range::INT AS k FROM range(5)")
        con.execute("CREATE TABLE b AS SELECT range::INT AS k FROM range(3)")
        joined = con.table("a").pl(lazy=True).join(con.table("b").pl(lazy=True), on="k")
        # polars re-raises a source's error under its own exception types.
        with pytest.raises(Exception, match=OPEN_RESULT):
            joined.collect()
