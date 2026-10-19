"""Call forms of the relational API that user code relies on, checked against direct SQL on the same connection."""

import gc

import pytest

import duckdb

pd = pytest.importorskip("pandas")


@pytest.fixture
def con():
    connection = duckdb.connect()
    connection.execute("CREATE TABLE t AS SELECT * FROM (VALUES (1, 'a'), (2, 'b'), (3, 'c')) v(i, x)")
    connection.execute("CREATE TABLE r AS SELECT * FROM (VALUES (2, 'B'), (3, 'C')) v(i, y)")
    connection.execute("CREATE TABLE c AS SELECT * FROM (VALUES (2, 'Z')) v(i, z)")
    yield connection
    connection.close()


def rows(result):
    return sorted(result, key=repr)


def read_through_sql(con):
    local_frame = pd.DataFrame({"i": [2, 3]})  # noqa: F841
    return con.sql("SELECT i FROM local_frame")


def read_through_the_module(con):
    local_frame = pd.DataFrame({"i": [2, 3]})  # noqa: F841
    return duckdb.sql("SELECT i FROM local_frame")


def read_through_a_filter(con):
    local_frame = pd.DataFrame({"i": [2, 3]})  # noqa: F841
    return con.table("t").filter("i IN (SELECT i FROM local_frame)").project("i")


def read_through_a_join_condition(con):
    local_frame = pd.DataFrame({"i": [2, 3]})  # noqa: F841
    return con.table("t").join(con.table("r"), "t.i = r.i AND t.i IN (SELECT i FROM local_frame)").project("t.i")


class TestFunctionLocalFrames:
    @pytest.mark.parametrize(
        "build",
        [
            pytest.param(read_through_sql, id="sql"),
            pytest.param(read_through_the_module, id="module_sql"),
            pytest.param(read_through_a_filter, id="filter_text"),
            pytest.param(read_through_a_join_condition, id="join_condition"),
        ],
    )
    def test_a_frame_local_to_the_calling_function_is_read(self, con, build):
        rel = build(con)
        gc.collect()
        assert rows(rel.fetchall()) == [(2,), (3,)]

    def test_create_table_as_and_insert_select_read_a_local_frame(self, con):
        def create(con):
            local_frame = pd.DataFrame({"i": [2, 3]})  # noqa: F841
            con.sql("CREATE TABLE copied AS SELECT * FROM local_frame")

        def insert(con):
            local_frame = pd.DataFrame({"i": [4]})  # noqa: F841
            con.sql("INSERT INTO copied SELECT i FROM local_frame")

        create(con)
        insert(con)
        assert con.execute("SELECT i FROM copied ORDER BY i").fetchall() == [(2,), (3,), (4,)]

    def test_two_frames_bound_in_turn_under_one_name_each_read_their_own(self, con):
        frame = pd.DataFrame({"a": [1]})
        first = con.sql("SELECT a FROM frame")
        frame = pd.DataFrame({"a": [2]})  # noqa: F841
        second = con.sql("SELECT a FROM frame")
        assert first.fetchall() == [(1,)]
        assert second.fetchall() == [(2,)]
        assert con.execute("SELECT a FROM frame").fetchall() == [(2,)]

    def test_two_objects_captured_under_one_name_are_refused_when_combined(self, con):
        data = pd.DataFrame({"a": [1]})
        first = con.sql("SELECT a FROM data")
        data = pd.DataFrame({"a": [2]})  # noqa: F841
        second = con.sql("SELECT a FROM data")
        assert first.fetchall() == [(1,)]
        assert second.fetchall() == [(2,)]
        with pytest.raises(duckdb.Error, match="data"):
            first.union(second).fetchall()

    def test_frames_whose_names_differ_only_in_case_in_one_statement_meet_the_engine_s_ambiguity(self, con):
        # The engine refuses the statement before any capture runs, so its own error is the answer.
        tbl = pd.DataFrame({"a": [1]})  # noqa: F841
        TBL = pd.DataFrame({"a": [2]})  # noqa: F841
        with pytest.raises(duckdb.BinderException, match='Ambiguous reference to table "tbl"'):
            con.sql("SELECT * FROM tbl, TBL")

    @pytest.mark.parametrize(
        "combine",
        [
            pytest.param(lambda left, right: left.cross(right), id="cross"),
            pytest.param(lambda left, right: left.union(right), id="union"),
        ],
    )
    def test_frames_whose_names_differ_only_in_case_from_two_steps_are_refused(self, con, combine):
        tbl = pd.DataFrame({"a": [1]})  # noqa: F841
        TBL = pd.DataFrame({"a": [2]})  # noqa: F841
        left = con.sql("SELECT a FROM tbl")
        right = con.sql("SELECT a FROM TBL")
        with pytest.raises(duckdb.InvalidInputException, match="differ only in case"):
            combine(left, right).fetchall()


class TestNamesInVerbs:
    def test_a_qualified_name_of_a_table_source(self, con):
        assert con.table("t").filter("t.i > 1").fetchall() == con.execute("FROM t WHERE t.i > 1").fetchall()

    def test_rowid_of_a_table_source(self, con):
        assert con.table("t").project("rowid").fetchall() == con.execute("SELECT rowid FROM t").fetchall()

    def test_a_three_way_join_names_all_three_sides(self, con):
        rel = (
            con.table("t").join(con.table("r"), "t.i = r.i").join(con.table("c"), "r.i = c.i").project("t.x, r.y, c.z")
        )
        expected = con.execute("SELECT t.x, r.y, c.z FROM t JOIN r ON t.i = r.i JOIN c ON r.i = c.i").fetchall()
        assert rel.fetchall() == expected

    def test_project_takes_a_name_with_a_space_after_a_filter(self, con):
        rel = con.sql('SELECT 1 AS "my col", 2 AS x').filter("x > 1").project("my col", "x")
        assert rel.columns == ["my col", "x"]
        assert rel.fetchall() == [(1, 2)]

    def test_project_takes_a_qualified_name_after_a_filter(self, con):
        rel = con.table("t").filter("i > 1").project("t.x", "i")
        assert rows(rel.fetchall()) == rows(con.execute("SELECT t.x, i FROM t WHERE i > 1").fetchall())

    @pytest.mark.parametrize("verb", ["project", "select"])
    def test_a_projection_without_arguments_is_refused(self, con, verb):
        with pytest.raises(TypeError):
            getattr(con.table("t"), verb)()


class TestAggregatesInAProjection:
    @pytest.mark.parametrize(
        "spelling",
        [
            pytest.param(lambda: "sum(i) AS s", id="sql_text"),
            pytest.param(
                lambda: duckdb.FunctionExpression("sum", duckdb.ColumnExpression("i")).alias("s"), id="constructor"
            ),
            pytest.param(lambda: duckdb.fn("sum", duckdb.col("i")).alias("s"), id="fn"),
            pytest.param(lambda: duckdb.col("i").sum().alias("s"), id="aggregate_method"),
        ],
    )
    def test_a_projection_may_hold_an_aggregate(self, con, spelling):
        rel = con.table("t").project(spelling())
        assert rel.columns == ["s"]
        assert rel.fetchall() == con.execute("SELECT sum(i) AS s FROM t").fetchall()

    def test_a_projection_binds_once_whatever_its_text_holds(self, con, monkeypatch):
        from duckdb import _seam

        source = con.table("t")
        original = _seam.bind
        binds = []

        def counted(connection, sql, *args, **kwargs):
            binds.append(sql)
            return original(connection, sql, *args, **kwargs)

        monkeypatch.setattr(_seam, "bind", counted)
        source.project("i")
        plain = len(binds)
        binds.clear()
        source.project("abs(i)")
        assert (plain, len(binds)) == (1, 1)


class TestReaderOptions:
    def test_a_csv_read_that_sniffs_one_of_two_shapes_raises_at_the_first_read(self, con, tmp_path):
        # The bind sniffs only the files it is told to, so the second shape is met when the rows are read.
        (tmp_path / "a.csv").write_text("i,s\n1,a\n")
        (tmp_path / "b.csv").write_text("i,s,extra\n2,b,9\n")
        rel = con.read_csv(str(tmp_path / "*.csv"), files_to_sniff=1)
        assert rel.columns == ["i", "s"]
        with pytest.raises(duckdb.InvalidInputException, match="CSV Error"):
            rel.fetchall()


class TestSourcesOverStatementsThatDescribe:
    @pytest.mark.parametrize(
        ("sql", "condition", "direct"),
        [
            pytest.param(
                "DESCRIBE t", "column_name = 'x'", "SELECT * FROM (DESCRIBE t) WHERE column_name = 'x'", id="describe"
            ),
            pytest.param("SHOW TABLES", "name = 't'", "SELECT * FROM (SHOW TABLES) WHERE name = 't'", id="show_tables"),
            pytest.param(
                "SUMMARIZE t",
                "column_name = 'i'",
                "SELECT * FROM (SUMMARIZE t) WHERE column_name = 'i'",
                id="summarize",
            ),
        ],
    )
    def test_a_filter_over_it(self, con, sql, condition, direct):
        assert con.sql(sql).filter(condition).fetchall() == con.execute(direct).fetchall()

    @pytest.mark.parametrize("sql", ["DESCRIBE t", "SHOW TABLES", "SUMMARIZE t"])
    def test_repr_and_show(self, con, sql, capsys):
        rel = con.sql(sql)
        assert rel.columns[0] in repr(rel)
        rel.show()
        assert rel.columns[0] in capsys.readouterr().out


class TestQueryNames:
    def test_a_plain_name_folds_case(self, con):
        assert con.sql("SELECT 1 AS a").query("MyView", "SELECT * FROM myview").fetchall() == [(1,)]


class TestTheReturnType:
    """A relation carries the engine's return type of the statement it holds, so a face can tell a command apart."""

    @pytest.mark.parametrize(
        ("build", "expected"),
        [
            pytest.param(lambda con: con.sql("SELECT 1 AS a"), "query_result", id="select"),
            pytest.param(lambda con: con.table("t"), "query_result", id="table"),
            pytest.param(lambda con: con.table("t").filter("i > 1"), "query_result", id="step"),
            pytest.param(lambda con: con.sql("CREATE TABLE made (i INTEGER)"), "nothing", id="ddl"),
            pytest.param(lambda con: con.sql("INSERT INTO t VALUES (9, 'z')"), "changed_rows", id="dml"),
        ],
    )
    def test_it_is_the_engine_s_return_type(self, con, build, expected):
        assert build(con)._return_type == expected


class TestSetOperationsKeepMultiplicity:
    """`except_` and `intersect` keep ALL, so multiplicities and NULLs follow SQL's EXCEPT ALL and INTERSECT ALL."""

    LEFT = "SELECT * FROM (VALUES (1), (1), (1), (NULL), (NULL), (2)) v(a)"
    RIGHT = "SELECT * FROM (VALUES (1), (1), (NULL)) v(a)"

    def sides(self, con):
        return con.sql(self.LEFT), con.sql(self.RIGHT)

    def test_except_keeps_the_remaining_copies(self, con):
        left, right = self.sides(con)
        twin = f"{self.LEFT} EXCEPT ALL {self.RIGHT}"
        expected = sorted(con.execute(twin).fetchall(), key=repr)
        assert sorted(left.except_(right).fetchall(), key=repr) == expected
        assert expected == sorted([(1,), (None,), (2,)], key=repr)

    def test_intersect_keeps_the_lesser_count(self, con):
        left, right = self.sides(con)
        twin = f"{self.LEFT} INTERSECT ALL {self.RIGHT}"
        expected = sorted(con.execute(twin).fetchall(), key=repr)
        assert sorted(left.intersect(right).fetchall(), key=repr) == expected
        assert expected == sorted([(1,), (1,), (None,)], key=repr)


class TestTypeStrings:
    @pytest.mark.parametrize(
        ("build", "expected"),
        [
            pytest.param(lambda con: con.sql("SELECT 1 AS a").map(lambda df: df), "TABLE_FUNCTION_RELATION", id="map"),
            pytest.param(
                lambda con: con.sql("SELECT 1 AS a").query("v", "SELECT * FROM v"), "QUERY_RELATION", id="query"
            ),
            pytest.param(lambda con: con.sql("CALL range(2)"), "MATERIALIZED_RELATION", id="ran_at_the_call"),
            pytest.param(
                lambda con: con.table("t").sample(2),
                "SAMPLE_RELATION",
                id="sample",
            ),
        ],
    )
    def test_type(self, con, build, expected):
        assert build(con).type == expected


class TestModuleLevelFunctions:
    @pytest.mark.parametrize(
        ("call", "direct"),
        [
            pytest.param(lambda f: duckdb.filter(f, "a > 1"), "SELECT * FROM f WHERE a > 1", id="filter"),
            pytest.param(lambda f: duckdb.project(f, "a + 1 AS b"), "SELECT a + 1 AS b FROM f", id="project"),
            pytest.param(lambda f: duckdb.aggregate(f, "sum(a) AS s"), "SELECT sum(a) AS s FROM f", id="aggregate"),
            pytest.param(lambda f: duckdb.distinct(f), "SELECT DISTINCT * FROM f", id="distinct"),
            pytest.param(lambda f: duckdb.limit(f, 2), "SELECT * FROM f LIMIT 2", id="limit"),
            pytest.param(lambda f: duckdb.order(f, "a DESC"), "SELECT * FROM f ORDER BY a DESC", id="order"),
            pytest.param(
                lambda f: duckdb.query_df(f, "fr", "SELECT sum(a) FROM fr"), "SELECT sum(a) FROM f", id="query_df"
            ),
        ],
    )
    def test_a_verb_over_a_data_frame(self, call, direct):
        f = pd.DataFrame({"a": [1, 2, 3]})
        expected = duckdb.connect().execute(direct).fetchall()
        assert rows(call(f).fetchall()) == rows(expected)

    def test_df_of_a_data_frame_is_a_relation(self):
        assert duckdb.df(pd.DataFrame({"a": [1, 2]})).fetchall() == [(1,), (2,)]

    def test_alias_over_a_data_frame(self):
        assert duckdb.alias(pd.DataFrame({"a": [1]}), "fr").alias == "fr"

    def test_write_csv_of_a_data_frame(self, tmp_path):
        path = str(tmp_path / "out.csv")
        duckdb.write_csv(pd.DataFrame({"a": [1, 2]}), path)
        assert duckdb.connect().execute(f"SELECT a FROM read_csv('{path}')").fetchall() == [(1,), (2,)]

    def test_arrow_of_an_arrow_table(self):
        pa = pytest.importorskip("pyarrow")
        assert duckdb.arrow(pa.table({"a": [1, 2]})).fetchall() == [(1,), (2,)]
