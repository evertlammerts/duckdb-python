"""Expressions built through the old constructor names, used in relation verbs and checked against direct SQL."""

import decimal
import math

import pytest

import duckdb
from duckdb import CaseExpression, ColumnExpression, ConstantExpression, FunctionExpression


@pytest.fixture
def con():
    connection = duckdb.connect()
    connection.execute(
        "CREATE TABLE t AS SELECT * FROM (VALUES (1, '\\x00'::BLOB, 'p'), (2, '\\x01'::BLOB, 'q')) v(i, b, x)"
    )
    yield connection
    connection.close()


def described(con, sql, params=None):
    cursor = con.execute(sql, params)
    description = [(column[0], str(column[1])) for column in cursor.description]
    return cursor.fetchall(), description


class TestBinaryValues:
    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(
                b"\x00",
                id="bytes",
            ),
            pytest.param(bytearray(b"\x00"), id="bytearray"),
        ],
    )
    def test_a_binary_operand_compares_with_a_blob(self, con, value):
        rel = con.table("t").filter(ColumnExpression("b") == value)
        assert rel.fetchall() == con.execute("SELECT * FROM t WHERE b = '\\x00'::BLOB").fetchall()


class TestOldCallForms:
    def test_isin_with_one_expression_is_one_member(self, con):
        rel = con.table("t").filter(ColumnExpression("i").isin(ConstantExpression(2)))
        assert rel.fetchall() == con.execute("SELECT * FROM t WHERE i IN (2)").fetchall()

    def test_isin_with_a_column_is_one_member(self, con):
        rel = con.table("t").filter(ColumnExpression("i").isin(ColumnExpression("i")))
        assert rel.fetchall() == con.execute("SELECT * FROM t WHERE i IN (i)").fetchall()

    def test_a_column_named_star_is_the_star(self, con):
        rel = con.table("t").select(ColumnExpression("*"))
        assert rel.columns == ["i", "b", "x"]
        assert rel.fetchall() == con.execute("SELECT * FROM t").fetchall()

    def test_an_empty_alias_keeps_the_column_name(self, con):
        rel = con.table("t").select(ColumnExpression("i").alias(""))
        assert rel.columns == ["i"]
        assert rel.fetchall() == [(1,), (2,)]

    def test_struct_pack_keeps_the_field_name_of_an_alias(self, con):
        rel = con.table("t").select(FunctionExpression("struct_pack", ColumnExpression("x").alias("name")).alias("s"))
        rows, description = described(con, "SELECT struct_pack(name := x) AS s FROM t")
        assert [(name, str(kind)) for name, kind in zip(rel.columns, rel.types, strict=True)] == description
        assert rel.fetchall() == rows

    def test_an_alias_on_a_function_argument_is_dropped(self, con):
        rel = con.table("t").select(FunctionExpression("upper", ColumnExpression("x").alias("q")))
        rows, description = described(con, "SELECT upper(x) FROM t")
        assert [(name, str(kind)) for name, kind in zip(rel.columns, rel.types, strict=True)] == description
        assert rel.fetchall() == rows

    def test_a_column_without_a_name_is_refused(self):
        with pytest.raises(ValueError, match=r"(?i)name"):
            ColumnExpression()


class TestValueConstants:
    @pytest.mark.parametrize(
        ("value", "kind"),
        [
            pytest.param(5, duckdb.sqltypes.TINYINT, id="tinyint"),
            pytest.param("2026-10-18", duckdb.sqltypes.DATE, id="date"),
            pytest.param(
                {"a": 1, "b": "x"},
                duckdb.struct_type({"a": duckdb.sqltypes.BIGINT, "b": duckdb.sqltypes.VARCHAR}),
                id="struct",
            ),
        ],
    )
    def test_a_value_constant_stores_what_binding_the_value_stores(self, con, value, kind):
        constant = ConstantExpression(duckdb.Value(value, kind)).alias("c")
        rel = con.sql("SELECT 1 AS z").select(constant)
        rows, description = described(con, "SELECT ? AS c", [duckdb.Value(value, kind)])
        assert [(name, str(k)) for name, k in zip(rel.columns, rel.types, strict=True)] == description
        assert rel.fetchall() == rows


def unfinished_case():
    return duckdb.when(duckdb.col("i") == 1).then(1)


# The exact binary value of the float 0.1, 55 digits long, which is the ordinary way to meet a wide Decimal.
DECIMAL_OF_A_FLOAT = decimal.Decimal(0.1)  # noqa: RUF032
UNFINISHED = r"\.otherwise\(|\.end\("


class TestAnUnfinishedCase:
    """A CASE without .otherwise() or .end() is no expression, so every use of it names the way out."""

    @pytest.mark.parametrize(
        "use",
        [
            pytest.param(lambda case: case == 1, id="equals"),
            pytest.param(lambda case: case != 1, id="not_equals"),
            pytest.param(lambda case: case < 1, id="less_than"),
            pytest.param(lambda case: case + 1, id="plus"),
            pytest.param(lambda case: 1 - case, id="reflected_minus"),
            pytest.param(bool, id="truth"),
        ],
    )
    def test_comparing_computing_or_testing_it_raises(self, use):
        with pytest.raises(TypeError, match=UNFINISHED):
            use(unfinished_case())

    def test_filter_over_a_comparison_with_it_raises_rather_than_returning_no_rows(self, con):
        # The comparison is an expression position, so the expression layer refuses it with TypeError.
        with pytest.raises(TypeError, match=UNFINISHED):
            con.table("t").filter(unfinished_case() == 1)

    @pytest.mark.parametrize(
        "verb",
        [
            pytest.param(lambda rel, case: rel.update({"i": case}), id="update_set"),
            pytest.param(lambda rel, case: rel.filter(case), id="filter"),
            pytest.param(lambda rel, case: rel.select(case), id="select"),
            pytest.param(lambda rel, case: rel.sort(case), id="sort"),
            pytest.param(lambda rel, case: rel.aggregate([case]), id="aggregate"),
            pytest.param(lambda rel, case: rel.join(rel.set_alias("o"), case), id="join_condition"),
        ],
    )
    def test_a_verb_given_it_raises_the_verb_argument_class_with_the_same_message(self, con, verb):
        # A verb refuses an argument it cannot take with InvalidInputException, as the shipping client did.
        with pytest.raises(duckdb.InvalidInputException, match=UNFINISHED):
            verb(con.table("t"), unfinished_case())

    def test_a_finished_case_compares_as_an_expression(self, con):
        finished = unfinished_case().otherwise(0)
        assert con.table("t").filter(finished == 1).project("i").fetchall() == [(1,)]


class TestDecimalsTheParameterPathHolds:
    """A Decimal the parameter path types, as DOUBLE when DECIMAL(38) cannot hold it, renders on every path."""

    def parameter_answer(self, con, value):
        return con.execute("SELECT ? AS d", [value]).fetchall()

    def test_decimal_of_a_float_in_a_filter(self, con):
        rel = con.sql("SELECT 0.05::DOUBLE AS x").filter(ColumnExpression("x") < DECIMAL_OF_A_FLOAT)
        assert rel.fetchall() == [(0.05,)]

    def test_decimal_of_a_float_in_values(self, con):
        value = DECIMAL_OF_A_FLOAT
        assert con.values([value]).fetchall() == self.parameter_answer(con, value)

    def test_decimal_of_a_float_in_a_projection(self, con):
        value = DECIMAL_OF_A_FLOAT
        rel = con.sql("SELECT 1 AS z").select(ConstantExpression(value).alias("d"))
        assert rel.fetchall() == self.parameter_answer(con, value)

    @pytest.mark.parametrize("text", ["NaN", "Infinity", "-Infinity"])
    @pytest.mark.parametrize(
        "path",
        [
            pytest.param(
                lambda con, v: con.sql("SELECT 1 AS z").select(ConstantExpression(v).alias("d")), id="projection"
            ),
            pytest.param(lambda con, v: con.values([v]), id="values"),
        ],
    )
    def test_a_value_that_is_not_finite(self, con, text, path):
        value = decimal.Decimal(text)
        [(answered,)] = path(con, value).fetchall()
        [(expected,)] = self.parameter_answer(con, value)
        assert (math.isnan(answered) and math.isnan(expected)) or answered == expected

    def test_a_value_that_is_not_finite_in_a_filter(self, con):
        rel = con.sql("SELECT 1.0::DOUBLE AS x").filter(ColumnExpression("x") < decimal.Decimal("Infinity"))
        assert rel.fetchall() == [(1.0,)]


class TestDecimalWidthAtTheEdges:
    """The width is the larger of the integral digits and the scale, so DECIMAL(38) holds both edges."""

    @pytest.mark.parametrize(
        ("text", "kind"),
        [
            pytest.param("1e-38", "DECIMAL(38,38)", id="smallest_scale_38"),
            pytest.param("9" * 38, "DECIMAL(38,0)", id="largest_integral"),
            pytest.param("0." + "9" * 38, "DECIMAL(38,38)", id="largest_fraction"),
        ],
    )
    def test_a_decimal_at_the_edge_renders_at_that_width(self, con, text, kind):
        value = decimal.Decimal(text)
        rel = con.sql("SELECT 1 AS z").select(ConstantExpression(value).alias("d"))
        assert [str(t) for t in rel.types] == [kind]
        assert rel.fetchall() == [(value,)]


class TestKeywordForms:
    """The keyword names of the shipping client's constructors and methods are accepted."""

    def test_constant_takes_value(self, con):
        assert con.sql("SELECT 1 AS z").select(ConstantExpression(value=5).alias("v")).fetchall() == [(5,)]

    def test_cast_takes_type(self, con):
        rel = con.sql("SELECT 1 AS z").select(ColumnExpression("z").cast(type=duckdb.sqltypes.VARCHAR).alias("v"))
        assert rel.fetchall() == [("1",)]

    def test_try_cast_takes_type(self, con):
        rel = con.sql("SELECT 'x' AS z").select(ColumnExpression("z").try_cast(type=duckdb.sqltypes.INTEGER).alias("v"))
        assert rel.fetchall() == [(None,)]

    def test_between_takes_lower_and_upper(self, con):
        between = ColumnExpression("z").between(lower=ConstantExpression(1), upper=ConstantExpression(3))
        assert con.sql("SELECT 2 AS z").select(between.alias("v")).fetchall() == [(True,)]


class TestConditionValues:
    """A condition is any value but a string, so a Decimal is one and None, which no condition can be, is not."""

    def test_a_decimal_condition(self, con):
        case = CaseExpression(decimal.Decimal("1.5"), ConstantExpression(7)).otherwise(ConstantExpression(0))
        assert con.sql("SELECT 1 AS z").select(case.alias("v")).fetchall() == [(7,)]

    def test_none_is_no_condition(self):
        with pytest.raises(TypeError):
            CaseExpression(None, ConstantExpression(7))

    def test_where_takes_a_value_condition(self, con):
        # .where() takes any value but a string, as when() and filter() do.
        total = duckdb.col("i").sum().where(True).alias("s")
        assert "FILTER (WHERE TRUE)" in str(total)
        assert con.table("t").aggregate([total]).fetchall() == [(3,)]


class TestAMarkerInsideAValue:
    """A value holding `$1` stays a value; the text is assembled without a replacement pass."""

    @pytest.fixture
    def marked(self, con):
        return con.sql("SELECT * FROM (VALUES ('prefix$1suffix', 1), (' || version() || ', 2)) v(s, n)")

    def test_in_a_filter(self, marked):
        assert marked.filter(ColumnExpression("s") == ConstantExpression("prefix$1suffix")).fetchall() == [
            ("prefix$1suffix", 1)
        ]

    def test_in_a_function_beside_a_value_that_reads_like_sql(self, con):
        call = FunctionExpression(
            "concat", ConstantExpression(" || version() || "), ConstantExpression("prefix$1suffix")
        )
        assert con.sql("SELECT 1 AS z").select(call.alias("c")).fetchall() == [(" || version() || prefix$1suffix",)]

    def test_in_a_join_condition(self, con, marked):
        other = con.sql("SELECT * FROM (VALUES ('prefix$1suffix', 'x')) w(s, y)")
        condition = (ColumnExpression("a", "s") == ColumnExpression("b", "s")) & (
            ColumnExpression("b", "s") == ConstantExpression("prefix$1suffix")
        )
        joined = marked.set_alias("a").join(other.set_alias("b"), condition)
        assert joined.fetchall() == [("prefix$1suffix", 1, "prefix$1suffix", "x")]


class TestRawTextInsideALambda:
    """A lambda variable written into raw SQL text names the variable the lambda finally binds."""

    def test_a_variable_in_raw_text_binds(self, con):
        # The variable is interpolated before its final name is chosen, and the text still reads it.
        incremented = duckdb.fn("list_transform", [1, 2], lambda x: duckdb.SQLExpression(f"{x} + 1"))
        assert con.sql("SELECT 1").select(incremented).fetchall() == [([2, 3],)]

    def test_a_nested_lambda_with_a_column_in_raw_text_binds(self, con):
        # The inner variable and the column k both appear in the raw text of the inner lambda.
        con.execute("CREATE TABLE nested AS SELECT [[1, 2], [3]] AS xss, 10 AS k")
        shifted = duckdb.fn(
            "list_transform",
            duckdb.col("xss"),
            lambda xs: duckdb.fn("list_transform", xs, lambda x: duckdb.SQLExpression(f"{x} + k")),
        )
        direct = "SELECT list_transform(xss, lambda xs: list_transform(xs, lambda x: x + k)) FROM nested"
        expected = con.execute(direct).fetchall()
        assert con.table("nested").select(shifted).fetchall() == expected == [([[11, 12], [13]],)]
