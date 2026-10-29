"""A VARCHAR the client cannot turn into a Python str raises InvalidInputException and ends the result."""

import importlib.util
import re

import pytest

import duckdb

pa = pytest.importorskip("pyarrow")

needs_pandas = pytest.mark.skipif(importlib.util.find_spec("pandas") is None, reason="pandas not installed")


def _icu_loads():
    with duckdb.connect() as con:
        try:
            con.execute("LOAD icu")
        except duckdb.Error:
            return False
    return True


needs_icu = pytest.mark.skipif(not _icu_loads(), reason="SET TimeZone needs the icu extension")

ROWS = 5000
INVALID_UTF8 = b"\xff"
INVALID_UNICODE = "Invalid unicode"


def strings(values):
    # The Arrow scan does not check UTF-8, so the invalid bytes reach the client unchanged.
    return pa.table({"s": pa.array(values, pa.binary()).cast(pa.string(), safe=False)})


def strings_with_one_invalid_row(row):
    values = [b"ok"] * ROWS
    values[row] = INVALID_UTF8
    return strings(values)


@pytest.fixture
def con():
    connection = duckdb.connect()
    yield connection
    connection.close()


def fetch_one_at_a_time(con):
    while con.fetchone() is not None:
        pass


def fetch_many_at_a_time(con):
    while con.fetchmany(1000):
        pass


FETCHES = [
    pytest.param(fetch_one_at_a_time, id="fetchone"),
    pytest.param(fetch_many_at_a_time, id="fetchmany"),
    pytest.param(lambda con: con.fetchall(), id="fetchall"),
    pytest.param(lambda con: con.df(), id="df", marks=needs_pandas),
    pytest.param(lambda con: con.fetchnumpy(), id="fetchnumpy"),
]
INVALID_ROWS = [pytest.param(0, id="first_vector"), pytest.param(3000, id="later_vector")]


class TestAConversionErrorEndsTheResult:
    @pytest.mark.parametrize("row", INVALID_ROWS)
    @pytest.mark.parametrize("fetch", FETCHES)
    def test_the_next_statement_runs(self, con, fetch, row):
        con.register("strings", strings_with_one_invalid_row(row))
        con.execute("SELECT s FROM strings")
        with pytest.raises(duckdb.InvalidInputException, match=INVALID_UNICODE):
            fetch(con)
        assert con.sql("SELECT 42").fetchall() == [(42,)]

    @pytest.mark.parametrize("fetch", FETCHES)
    def test_a_later_fetch_raises_the_same_error(self, con, fetch):
        con.register("strings", strings_with_one_invalid_row(3000))
        con.execute("SELECT s FROM strings")
        with pytest.raises(duckdb.InvalidInputException, match=INVALID_UNICODE) as first:
            fetch(con)
        for later in (con.fetchall, con.fetchone):
            with pytest.raises(duckdb.InvalidInputException) as again:
                later()
            assert str(again.value) == str(first.value)
        assert con.sql("SELECT 42").fetchall() == [(42,)]

    def test_a_python_error_is_raised_again_as_its_text(self, con):
        # The day count fits the engine's interval, but datetime.timedelta caps days at 999999999.
        con.execute("SELECT INTERVAL 1000000000 DAYS")
        with pytest.raises(OverflowError) as first:
            con.fetchone()
        with pytest.raises(duckdb.InvalidInputException, match=re.escape(str(first.value))):
            con.fetchall()
        assert con.sql("SELECT 42").fetchall() == [(42,)]

    @needs_icu
    @needs_pandas
    def test_a_python_error_in_a_data_frame_chunk_ends_the_result(self, con):
        # The engine accepts the old Java three letter zone names, which the IANA database pandas reads lacks.
        con.execute("SET TimeZone = 'ACT'")
        con.execute("SELECT TIMESTAMPTZ '2024-01-01 00:00:00+00' + INTERVAL (i) SECOND FROM range(10000) t(i)")
        with pytest.raises(KeyError) as first:
            con.fetch_df_chunk(1)
        with pytest.raises(duckdb.InvalidInputException, match=re.escape(str(first.value))):
            con.fetchall()
        assert con.sql("SELECT 42").fetchall() == [(42,)]

    def test_the_transaction_stays_usable(self, con):
        con.register("strings", strings_with_one_invalid_row(3000))
        con.execute("BEGIN")
        con.execute("CREATE TABLE kept (i INTEGER)")
        con.execute("SELECT s FROM strings")
        with pytest.raises(duckdb.InvalidInputException, match=INVALID_UNICODE):
            con.fetchall()
        con.sql("INSERT INTO kept VALUES (1)")
        con.execute("COMMIT")
        assert con.sql("SELECT i FROM kept").fetchall() == [(1,)]


class TestNumpyConversionOfInvalidUtf8:
    @pytest.mark.parametrize(
        "query",
        [pytest.param("SELECT s FROM strings", id="varchar"), pytest.param("SELECT [s] AS s FROM strings", id="list")],
    )
    @pytest.mark.parametrize("method", [pytest.param("fetchnumpy"), pytest.param("df", marks=needs_pandas)])
    @pytest.mark.parametrize("source", ["execute", "sql"])
    def test_raises_invalid_input_like_fetchall(self, con, source, method, query):
        con.register("strings", strings([b"ok", INVALID_UTF8, None]))
        with pytest.raises(duckdb.InvalidInputException, match=INVALID_UNICODE):
            getattr(getattr(con, source)(query), method)()
        assert con.sql("SELECT 42").fetchall() == [(42,)]
