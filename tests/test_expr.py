"""Expressions render to SQL, and the values in them are bound, not inlined."""

from __future__ import annotations

import datetime
import decimal
import uuid

import pytest

from duckdb import _duckdb, exceptions
from duckdb.expr import (
    Expr,
    ParamSink,
    coalesce,
    col,
    fn,
    lit,
    param,
    quote,
    row_number,
    sql_expr,
    sql_type_of,
    star,
    when,
)


@pytest.fixture(scope="module")
def con() -> _duckdb.Connection:
    return _duckdb.Database(":memory:").connect()


def evaluate(con: _duckdb.Connection, expression: object, source: str = "") -> object:
    """Run an expression and return its single value."""
    sql = f"SELECT {expression.fragment()}{source}"  # type: ignore[attr-defined]
    return con.execute(sql).fetch_all()[0][0]


class TestQuoting:
    def test_identifiers_are_always_quoted(self) -> None:
        assert col("x").fragment() == '"x"'

    def test_embedded_quotes_are_escaped(self) -> None:
        # Without escaping this would end the identifier and inject syntax.
        assert quote('a"b') == '"a""b"'

    def test_a_reserved_word_is_usable_as_a_column(self, con: _duckdb.Connection) -> None:
        con.execute('CREATE TABLE t ("select" INTEGER)').drain()
        con.execute("INSERT INTO t VALUES (1)").drain()
        assert evaluate(con, col("select"), " FROM t") == 1


class TestStringOperandsAreLiterals:
    """The load-bearing break from the previous client.

    There, `col("state") == "active"` compared against a *column* named active.
    Here a string is always a literal, and columns are always explicit.
    """

    def test_string_compares_as_a_value(self, con: _duckdb.Connection) -> None:
        con.execute("CREATE TABLE s (state VARCHAR, active VARCHAR)").drain()
        con.execute("INSERT INTO s VALUES ('active', 'no')").drain()
        assert evaluate(con, col("state") == "active", " FROM s") is True

    def test_a_column_operand_is_explicit(self, con: _duckdb.Connection) -> None:
        assert evaluate(con, col("state") == col("active"), " FROM s") is False


class TestNoneAndUnknownOperands:
    def test_equality_with_none_is_a_null_comparison(self, con: _duckdb.Connection) -> None:
        # SQL's answer: comparing with NULL is never true. Surfacing that beats
        # inventing something friendlier.
        assert evaluate(con, col("v") == None, " FROM (SELECT 1 AS v)") is None  # noqa: E711

    def test_equality_with_an_unconvertible_object_falls_back_to_python(self) -> None:
        # Python falls back to identity, so this is False rather than an error
        # and rather than a nonsensical expression.
        assert (col("x") == object()) is False

    def test_arithmetic_with_an_unconvertible_object_raises(self) -> None:
        with pytest.raises(TypeError):
            col("x") + object()


class TestParameterSink:
    """Values that could carry injection are bound; the rest stay inline."""

    def test_strings_are_bound(self) -> None:
        with ParamSink() as sink:
            rendered = (col("name") == "ann").fragment()
        assert rendered == '("name" = $1)'
        assert sink.entries == [("literal", "ann", "VARCHAR")]

    def test_numbers_stay_inline(self) -> None:
        # Injection-safe, and inlining keeps DuckDB's own literal typing rather
        # than forcing a cast through a bound value.
        with ParamSink() as sink:
            rendered = (col("age") > 30).fragment()
        assert rendered == '("age" > 30)'
        assert sink.entries == []

    def test_bytes_are_bound(self) -> None:
        with ParamSink() as sink:
            (col("b") == b"\xde\xad").fragment()
        assert sink.entries[0][2] == "BLOB"

    def test_without_a_sink_everything_renders_inline(self) -> None:
        # This is how the schema oracle sees it: an untyped $n would tell the
        # binder nothing, so binding always renders sink-off.
        assert (col("name") == "ann").fragment() == "(\"name\" = 'ann')"

    def test_numbering_follows_render_order(self) -> None:
        with ParamSink() as sink:
            rendered = ((col("a") == "x") & (col("b") == "y")).fragment()
        assert "$1" in rendered
        assert "$2" in rendered
        assert [e[1] for e in sink.entries] == ["x", "y"]

    def test_named_placeholders_share_the_numbering(self) -> None:
        with ParamSink() as sink:
            ((col("a") == "x") & (col("b") == param("later"))).fragment()
        assert [e[0] for e in sink.entries] == ["literal", "reference"]

    def test_a_string_with_a_quote_is_bound_not_escaped_into_sql(self) -> None:
        nasty = "'; DROP TABLE t; --"
        with ParamSink() as sink:
            rendered = (col("x") == nasty).fragment()
        assert nasty not in rendered
        assert sink.entries[0][1] == nasty


class TestSqlTypeOf:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (True, "BOOLEAN"),
            (1, "INTEGER"),
            (2**40, "BIGINT"),
            (2**70, "HUGEINT"),
            (1.5, "DOUBLE"),
            ("x", "VARCHAR"),
            (b"x", "BLOB"),
            ([1, 2], "INTEGER[]"),
            ([1, 1.5], "DOUBLE[]"),
            (["a", 1], "VARCHAR[]"),
        ],
    )
    def test_types(self, value: object, expected: str) -> None:
        assert sql_type_of(value) == expected

    def test_ambiguous_values_have_no_type(self) -> None:
        # An empty list has no element type, so it stays inline rather than
        # being bound as something invented.
        assert sql_type_of([]) is None


class TestOperatorsEvaluate:
    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            (lit(2) + 3, 5),
            (lit(10) - 4, 6),
            (lit(3) * 4, 12),
            (lit(7) / 2, 3.5),
            (lit(7) // 2, 3),
            (lit(7) % 3, 1),
            (lit(2) ** 10, 1024),
            (-lit(5), -5),
            (~lit(True), False),
            ((lit(True) & lit(False)), False),
            ((lit(True) | lit(False)), True),
            (3 + lit(4), 7),
            (10 - lit(4), 6),
        ],
    )
    def test_operator(self, con: _duckdb.Connection, expression: object, expected: object) -> None:
        assert evaluate(con, expression) == expected

    def test_precedence_is_preserved(self, con: _duckdb.Connection) -> None:
        # Every node parenthesises itself, so Python's precedence survives into
        # the SQL rather than being reinterpreted by it.
        assert evaluate(con, (lit(2) + lit(3)) * lit(4)) == 20
        assert evaluate(con, lit(2) + lit(3) * lit(4)) == 14


class TestPredicatesAndFunctions:
    def test_is_null(self, con: _duckdb.Connection) -> None:
        assert evaluate(con, lit(None).is_null()) is True
        assert evaluate(con, lit(1).is_not_null()) is True

    def test_isin(self, con: _duckdb.Connection) -> None:
        assert evaluate(con, lit(2).isin([1, 2, 3])) is True
        assert evaluate(con, lit(9).isin([1, 2, 3])) is False

    def test_isin_empty_is_false(self, con: _duckdb.Connection) -> None:
        # Nothing is a member of an empty set, and SQL's `IN ()` is a syntax
        # error, so this renders as FALSE.
        assert evaluate(con, lit(1).isin([])) is False

    def test_between(self, con: _duckdb.Connection) -> None:
        assert evaluate(con, lit(5).between(1, 10)) is True

    def test_cast(self, con: _duckdb.Connection) -> None:
        assert evaluate(con, lit("42").cast("INTEGER")) == 42

    def test_coalesce(self, con: _duckdb.Connection) -> None:
        assert evaluate(con, coalesce(lit(None), lit(None), lit(7))) == 7

    def test_fn_reaches_any_function(self, con: _duckdb.Connection) -> None:
        assert evaluate(con, fn("upper", lit("ab"))) == "AB"

    def test_case(self, con: _duckdb.Connection) -> None:
        expression = when(lit(1) > 2).then("big").otherwise("small")
        assert evaluate(con, expression) == "small"

    def test_case_without_otherwise_is_null(self, con: _duckdb.Connection) -> None:
        assert evaluate(con, when(lit(1) > 2).then("big").end()) is None

    def test_sql_expr_escape_hatch(self, con: _duckdb.Connection) -> None:
        assert evaluate(con, sql_expr("1 + 1")) == 2


class TestAggregatesAndWindows:
    def test_aggregate_shortcut(self, con: _duckdb.Connection) -> None:
        value = con.execute(f"SELECT {col('i').sum().fragment()} FROM range(4) t(i)").fetch_all()[0][0]
        assert value == 6

    def test_window(self, con: _duckdb.Connection) -> None:
        numbered = row_number().over(order_by=col("i"))
        rows = con.execute(f"SELECT {numbered.fragment()} FROM range(3) t(i)").fetch_all()
        assert [r[0] for r in rows] == [1, 2, 3]

    def test_unknown_attribute_still_raises(self) -> None:
        # __getattr__ serves the aggregate shortcuts, so it must not swallow
        # ordinary typos into a callable that fails later.
        with pytest.raises(AttributeError, match="no_such_aggregate"):
            _ = col("x").no_such_aggregate


class TestPresentation:
    def test_alias(self) -> None:
        assert col("x").alias("y").as_select() == '"x" AS "y"'

    def test_ordering(self) -> None:
        assert col("x").desc().as_order() == '"x" DESC'
        assert col("x").nulls_last().as_order() == '"x" ASC NULLS LAST'
        assert col("x").desc().nulls_first().as_order() == '"x" DESC NULLS FIRST'

    def test_expressions_are_immutable(self) -> None:
        # Aliasing returns a new node, so one expression can be reused across
        # frames without carrying another frame's presentation.
        base = col("x")
        aliased = base.alias("y")
        assert base.as_select() == '"x"'
        assert aliased.as_select() == '"x" AS "y"'
        assert base is not aliased

    def test_star(self) -> None:
        assert star().fragment() == "*"
        assert star(exclude=["a"]).fragment() == '* EXCLUDE ("a")'
        assert star(rename={"a": "b"}).fragment() == '* RENAME ("a" AS "b")'


class TestRichLiteralTypes:
    """The expression layer accepts every type the binding layer can convert.

    It used to accept fewer, so `col("d") > date(2026, 8, 1)` raised TypeError
    while the same value bound fine as a parameter. Found by writing an example.
    """

    @pytest.mark.parametrize(
        ("value", "expected_type"),
        [
            (datetime.date(2026, 8, 1), "DATE"),
            (datetime.datetime(2026, 8, 1, 12, 30), "TIMESTAMP"),
            (
                datetime.datetime(2026, 8, 1, tzinfo=datetime.UTC),
                "TIMESTAMP WITH TIME ZONE",
            ),
            (datetime.time(12, 30), "TIME"),
            (datetime.timedelta(days=2), "INTERVAL"),
            (decimal.Decimal("1.50"), "DECIMAL"),
            (uuid.UUID(int=1), "UUID"),
        ],
    )
    def test_binds_with_its_type(self, value: object, expected_type: str) -> None:
        with ParamSink() as sink:
            rendered = (col("x") > value).fragment()
        assert rendered == '("x" > $1)'
        assert sink.entries == [("literal", value, expected_type)]

    def test_renders_for_the_oracle_without_a_sink(self, con: _duckdb.Connection) -> None:
        # Sink-off rendering has to stay bindable, since that is how the schema
        # oracle sees the expression.
        expression = col("d") > datetime.date(2026, 8, 1)
        output, _ = con.bind(f"SELECT {expression.fragment()} FROM (SELECT DATE '2020-01-01' AS d)")
        assert output[0][1] == "BOOLEAN"

    def test_a_date_comparison_evaluates(self, con: _duckdb.Connection) -> None:
        expression = col("d") > datetime.date(2026, 1, 1)
        rows = con.execute(f"SELECT {expression.fragment()} FROM (SELECT DATE '2026-08-01' AS d)").fetch_all()
        assert rows[0][0] is True

    def test_decimal_keeps_its_scale_when_bound(self) -> None:
        # Rendering a Decimal as text would put its scale at the mercy of a
        # format string; binding hands the exact value to the converter.
        with ParamSink() as sink:
            (col("x") == decimal.Decimal("1.50")).fragment()
        assert str(sink.entries[0][1]) == "1.50"

    def test_non_finite_decimal_is_refused(self) -> None:
        with pytest.raises(TypeError, match="non-finite"):
            (col("x") == decimal.Decimal("NaN")).fragment()


class TestFoundByWritingTheManual:
    """Two shortcuts that rendered SQL the engine rejects.

    Both were caught by running every example in the user manual rather than
    trusting that they read correctly.
    """

    def test_n_unique_uses_distinct_syntax(self, con: _duckdb.Connection) -> None:
        # It mapped to a `count_distinct` function, which DuckDB does not have.
        # The exact form is syntax, not a function name.
        expression = col("c").n_unique()
        assert expression.fragment() == 'count(DISTINCT "c")'
        con.execute("CREATE TABLE u (c VARCHAR)").drain()
        con.execute("INSERT INTO u VALUES ('x'), ('x'), ('y')").drain()
        assert con.execute(f"SELECT {expression.fragment()} FROM u").fetch_all()[0][0] == 2

    def test_concat_is_explicit_because_plus_is_not_concatenation(self, con: _duckdb.Connection) -> None:
        # SQL concatenates with ||; `+` on two strings is an error. Expressions
        # carry no types, so `+` cannot decide which was meant.
        expression = lit("a").concat("-", lit("b"))
        assert con.execute(f"SELECT {expression.fragment()}").fetch_all()[0][0] == "a-b"

    def test_plus_on_text_still_fails_loudly(self, con: _duckdb.Connection) -> None:
        # Not silently reinterpreted as concatenation: the binder says so.
        with pytest.raises(exceptions.Error):
            con.execute(f"SELECT {(lit('a') + lit('b')).fragment()}").fetch_all()


# How to invoke each advertised name, so every one can be executed rather than
# merely rendered. Adding a shortcut without adding a spec fails the
# completeness test below, on purpose.
AGGREGATE_CALLS: dict[str, tuple[object, ...]] = {
    "sum": (),
    "mean": (),
    "avg": (),
    "min": (),
    "max": (),
    "count": (),
    "median": (),
    "std": (),
    "var": (),
    "first": (),
    "last": (),
    "any_value": (),
    "product": (),
    "skewness": (),
    "kurtosis": (),
    "entropy": (),
    "string_agg": (lit(","),),
    "bit_and": (),
    "bit_or": (),
    "bool_and": (),
    "bool_or": (),
}
WINDOW_CALLS: dict[str, tuple[object, ...]] = {
    "row_number": (),
    "rank": (),
    "dense_rank": (),
    "ntile": (lit(2),),
    "lag": (col("v"),),
    "lead": (col("v"),),
    "first_value": (col("v"),),
    "last_value": (col("v"),),
}


@pytest.fixture(scope="module")
def data(con: _duckdb.Connection) -> str:
    """A small table carrying one column of each kind the aggregates need."""
    con.execute("CREATE TABLE agg (v INTEGER, b BOOLEAN, s VARCHAR)").drain()
    con.execute("INSERT INTO agg VALUES (1, true, 'a'), (2, false, 'b'), (3, true, 'c')").drain()
    return "agg"


class TestEveryAdvertisedFunctionExecutes:
    """Execute every shortcut the DSL offers, against the real engine.

    Rendering tests cannot catch a wrong function name: `count_distinct("c")`
    is a perfectly well-formed string that DuckDB has never heard of. Line
    coverage cannot catch it either, because one dict literal and a three-line
    `__getattr__` serve all thirty names, so exercising one covers the same
    lines as exercising all of them.

    That gap shipped a broken `n_unique`. This closes it by construction.
    """

    def test_the_spec_covers_every_advertised_name(self) -> None:
        advertised = set(Expr._CALLABLES) | set(WINDOW_CALLS)
        specified = set(AGGREGATE_CALLS) | set(WINDOW_CALLS)
        assert not advertised - specified, f"advertised but never executed: {sorted(advertised - specified)}"

    @pytest.mark.parametrize("name", sorted(AGGREGATE_CALLS))
    def test_aggregate(self, con: _duckdb.Connection, data: str, name: str) -> None:
        column = "b" if name in {"bool_and", "bool_or"} else "s" if name == "string_agg" else "v"
        expression = getattr(col(column), name)(*AGGREGATE_CALLS[name])
        con.execute(f"SELECT {expression.fragment()} FROM {data}").fetch_all()

    @pytest.mark.parametrize("name", sorted(WINDOW_CALLS))
    def test_window(self, con: _duckdb.Connection, data: str, name: str) -> None:
        import duckdb.expr as expr_module

        expression = getattr(expr_module, name)(*WINDOW_CALLS[name]).over(order_by=col("v"))
        con.execute(f"SELECT {expression.fragment()} FROM {data}").fetch_all()

    def test_n_unique(self, con: _duckdb.Connection, data: str) -> None:
        # Not in the shortcut table: DISTINCT is syntax, not a function name.
        assert con.execute(f"SELECT {col('v').n_unique().fragment()} FROM {data}").fetch_all()[0][0] == 3
