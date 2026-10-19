"""Python objects read by relations: who keeps them alive, and the SQL text of a relation that reads one."""

import gc
import re
import weakref

import pytest

import duckdb

pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")


@pytest.fixture
def con():
    connection = duckdb.connect()
    yield connection
    connection.close()


def frame():
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


def arrow():
    return pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})


def identity(df):
    return df


def from_frame_variable(con):
    data = frame()  # noqa: F841
    return con.sql("SELECT * FROM data")


SOURCES = [
    pytest.param(lambda con: con.from_df(frame()), id="from_df"),
    pytest.param(lambda con: con.from_arrow(arrow()), id="from_arrow"),
    pytest.param(from_frame_variable, id="frame_variable"),
    pytest.param(
        lambda con: con.sql("SELECT range AS a, ['x', 'y', 'z'][range] AS b FROM range(1, 4)").materialize(),
        id="materialize",
    ),
    pytest.param(
        lambda con: con.sql("SELECT range AS a, ['x', 'y', 'z'][range] AS b FROM range(1, 4)").map(identity), id="map"
    ),
]


class TestTheRelationKeepsItsSource:
    @pytest.mark.parametrize(
        "build",
        [
            pytest.param(lambda con, obj: con.from_df(obj), id="from_df"),
            pytest.param(lambda con, obj: con.sql("SELECT * FROM obj"), id="frame_variable"),
        ],
    )
    def test_a_data_frame_lives_while_a_relation_reads_it(self, con, build):
        obj = frame()
        alive = weakref.ref(obj)
        rel = build(con, obj).filter("a > 1")
        del obj
        gc.collect()
        assert alive() is not None
        assert rel.fetchall() == [(2, "y"), (3, "z")]
        del rel
        gc.collect()
        assert alive() is None

    def test_an_arrow_table_lives_while_a_relation_reads_it(self, con):
        obj = arrow()
        alive = weakref.ref(obj)
        rel = con.from_arrow(obj).project("a * 2 AS d")
        del obj
        gc.collect()
        assert alive() is not None
        assert rel.fetchall() == [(2,), (4,), (6,)]
        del rel
        gc.collect()
        assert alive() is None

    def test_the_source_lives_until_the_last_relation_goes(self, con):
        obj = frame()
        alive = weakref.ref(obj)
        source = con.from_df(obj)
        first = source.filter("a > 1")
        second = source.project("b")
        del obj, source
        gc.collect()
        del first
        gc.collect()
        assert alive() is not None
        assert second.fetchall() == [("x",), ("y",), ("z",)]
        del second
        gc.collect()
        assert alive() is None

    def test_a_map_function_lives_while_its_relation_does(self, con):
        def double(df):
            df["a"] = df["a"] * 2
            return df

        alive = weakref.ref(double)
        rel = con.sql("SELECT range AS a FROM range(3)").map(double)
        del double
        gc.collect()
        assert alive() is not None
        assert rel.fetchall() == [(0,), (2,), (4,)]
        del rel
        gc.collect()
        assert alive() is None

    def test_a_materialized_relation_keeps_its_rows_after_its_input_goes(self, con):
        obj = frame()
        materialized = con.from_df(obj).materialize()
        del obj
        gc.collect()
        assert materialized.fetchall() == [(1, "x"), (2, "y"), (3, "z")]


class TestRegister:
    def test_register_keeps_the_object_until_unregister(self, con):
        obj = frame()
        alive = weakref.ref(obj)
        con.register("v", obj)
        del obj
        gc.collect()
        assert alive() is not None
        assert con.sql("SELECT sum(a) FROM v").fetchall() == [(6,)]
        con.unregister("v")
        gc.collect()
        assert alive() is None

    def test_a_relation_over_a_registered_name_reads_it(self, con):
        con.register("v", frame())
        rel = con.sql("SELECT a FROM v WHERE b <> 'x'")
        assert rel.fetchall() == [(2,), (3,)]
        assert con.sql(rel.sql_query()).fetchall() == [(2,), (3,)]


GENERATED_PREFIX = "__duckdb_source"


class TestUnregister:
    """`unregister` removes the view and its source, or says it did not; and it never shows a generated name."""

    def test_the_ordinary_case_removes_both(self, con):
        obj = frame()
        alive = weakref.ref(obj)
        con.register("v", obj)
        del obj
        gc.collect()
        assert con.sql("SELECT sum(a) FROM v").fetchall() == [(6,)]
        con.unregister("v")
        gc.collect()
        assert alive() is None
        with pytest.raises(duckdb.CatalogException, match="name v does not exist"):
            con.sql("SELECT a FROM v")

    def test_a_drop_that_cannot_happen_raises_instead_of_reporting_success(self, con):
        con.register("v", frame())
        con.execute("BEGIN TRANSACTION")
        with pytest.raises(duckdb.CatalogException):
            con.execute("SELECT * FROM does_not_exist")
        with pytest.raises(duckdb.Error):
            con.unregister("v")
        con.execute("ROLLBACK")

    def test_a_drop_that_cannot_happen_leaves_the_view_and_the_source(self, con):
        obj = frame()
        alive = weakref.ref(obj)
        con.register("v", obj)
        del obj
        con.execute("BEGIN TRANSACTION")
        with pytest.raises(duckdb.CatalogException):
            con.execute("SELECT * FROM does_not_exist")
        with pytest.raises(duckdb.Error):
            con.unregister("v")
        con.execute("ROLLBACK")
        gc.collect()
        assert alive() is not None
        assert con.sql("SELECT sum(a) FROM v").fetchall() == [(6,)]

    def test_a_later_unregister_on_a_usable_transaction_removes_both(self, con):
        obj = frame()
        alive = weakref.ref(obj)
        con.register("v", obj)
        del obj
        con.execute("BEGIN TRANSACTION")
        with pytest.raises(duckdb.CatalogException):
            con.execute("SELECT * FROM does_not_exist")
        with pytest.raises(duckdb.Error):
            con.unregister("v")
        con.execute("ROLLBACK")
        con.unregister("v")
        gc.collect()
        assert alive() is None
        with pytest.raises(duckdb.CatalogException, match="name v does not exist"):
            con.sql("SELECT a FROM v")


class TestNoMessageShowsAGeneratedName:
    """A name the client generates for a source is internal, so no error a user reads may carry one."""

    @pytest.mark.parametrize(
        "reach",
        [
            pytest.param(lambda con: con.sql("SELECT a FROM never_registered"), id="never_registered"),
            pytest.param(
                lambda con: (con.register("v", frame()), con.unregister("v"), con.sql("SELECT a FROM v"))[-1],
                id="after_unregister",
            ),
            pytest.param(
                lambda con: (
                    con.from_df(frame()).create_view("w"),
                    con.execute("DROP VIEW w"),
                    con.sql("SELECT a FROM w"),
                )[-1],
                id="after_the_view_is_dropped",
            ),
            pytest.param(
                lambda con: (con.register("v", frame()), con.register("v", frame()), con.sql("SELECT nope FROM v"))[-1],
                id="after_a_replacement",
            ),
        ],
    )
    def test_the_message_names_what_the_user_wrote(self, con, reach):
        with pytest.raises(duckdb.Error) as error:
            reach(con)
        assert GENERATED_PREFIX not in str(error.value)

    def test_the_refusal_of_a_view_in_a_database_file(self, tmp_path):
        # The case a reopen could not read is refused at the call, and the refusal names no key either.
        con = duckdb.connect(str(tmp_path / "saved.duckdb"))
        with pytest.raises(duckdb.Error) as error:
            con.from_df(frame()).create_view("w")
        con.close()
        assert GENERATED_PREFIX not in str(error.value)


class TestViewsOverSources:
    def test_a_view_over_a_data_frame_relation_reads_after_the_relation_goes(self, con):
        obj = frame()
        con.from_df(obj).filter("a > 1").create_view("w")
        del obj
        gc.collect()
        assert con.sql("SELECT a FROM w ORDER BY a").fetchall() == [(2,), (3,)]


class TestAViewKeepsItsSources:
    def test_dropping_the_view_through_sql_frees_the_source_once_the_drop_commits(self, con):
        # The view's definition owns what it reads, so the source lives exactly as long as the view does.
        obj = frame()
        alive = weakref.ref(obj)
        con.from_df(obj).create_view("w")
        del obj
        gc.collect()
        assert alive() is not None
        con.execute("BEGIN")
        con.execute("DROP VIEW w")
        gc.collect()
        assert alive() is not None
        con.execute("COMMIT")
        gc.collect()
        assert alive() is None

    def test_a_view_can_be_created_again_after_a_drop(self, con):
        con.from_df(frame()).create_view("w")
        con.execute("DROP VIEW w")
        con.from_df(pd.DataFrame({"a": [7]})).create_view("w")
        assert con.sql("SELECT a FROM w").fetchall() == [(7,)]

    def test_replacing_the_view_releases_the_source(self, con):
        obj = frame()
        alive = weakref.ref(obj)
        con.from_df(obj).create_view("w")
        del obj
        con.sql("SELECT 1 AS a").create_view("w")
        gc.collect()
        assert alive() is None

    def test_unregister_leaves_a_view_made_by_create_view(self, con):
        # Unregister releases only what register made, so the view and the source it owns stay.
        obj = frame()
        alive = weakref.ref(obj)
        expected = con.execute("SELECT a, b FROM obj ORDER BY a").fetchall()
        con.from_df(obj).create_view("w")
        del obj
        con.unregister("w")
        gc.collect()
        assert alive() is not None
        assert con.sql("SELECT a, b FROM w ORDER BY a").fetchall() == expected

    def test_closing_the_connection_releases_the_source(self):
        con = duckdb.connect()
        obj = frame()
        alive = weakref.ref(obj)
        con.from_df(obj).create_view("w")
        del obj
        gc.collect()
        assert alive() is not None
        con.close()
        gc.collect()
        assert alive() is None


class TestOtherConnectionsOfTheDatabase:
    def test_a_cursor_reads_a_view_over_a_data_frame(self, con):
        con.from_df(frame()).create_view("w")
        assert con.cursor().sql("SELECT a FROM w ORDER BY a").fetchall() == [(1,), (2,), (3,)]

    @pytest.mark.parametrize("build", [source for source in SOURCES if source.id != "materialize"])
    def test_a_view_over_a_python_object_is_refused_in_a_database_file(self, tmp_path, build):
        con = duckdb.connect(str(tmp_path / "saved.duckdb"))
        with pytest.raises(duckdb.InvalidInputException, match="database file"):
            build(con).create_view("w")
        assert con.execute("SELECT view_name FROM duckdb_views() WHERE NOT internal").fetchall() == []
        con.close()

    def test_a_view_over_a_materialized_relation_is_stored_in_a_database_file(self, tmp_path):
        # The view's definition carries the rows it reads, so it reads after a reopen, as the shipping client did.
        path = str(tmp_path / "saved.duckdb")
        con = duckdb.connect(path)
        con.sql("SELECT 42 AS x").materialize().create_view("v")
        assert con.sql("SELECT x FROM v").fetchall() == [(42,)]
        con.close()
        con = duckdb.connect(path)
        assert con.sql("SELECT x FROM v").fetchall() == [(42,)]
        con.close()

    def test_a_view_over_a_relation_read_by_variable_is_stored_in_a_database_file(self, tmp_path):
        # The relation is written into the view as its own text, so the view reads after a reopen.
        path = str(tmp_path / "saved.duckdb")
        con = duckdb.connect(path)
        con.execute("CREATE TABLE numbers AS SELECT * FROM (VALUES (1, 'a'), (2, 'b'), (-3, 'c')) v(a, b)")
        held = con.sql("SELECT a, b FROM numbers WHERE a > 0")  # noqa: F841
        con.sql("SELECT b, a * 10 AS t FROM held").create_view("w")
        expected = con.execute("SELECT b, a * 10 AS t FROM numbers WHERE a > 0 ORDER BY b").fetchall()
        assert expected == [("a", 10), ("b", 20)]
        assert con.sql("SELECT b, t FROM w ORDER BY b").fetchall() == expected
        con.close()
        con = duckdb.connect(path)
        assert con.sql("SELECT b, t FROM w ORDER BY b").fetchall() == expected
        con.close()

    def test_a_view_over_a_materialized_relation_is_created_in_memory(self, con):
        con.sql("SELECT 42 AS x").materialize().create_view("v")
        assert con.sql("SELECT x FROM v").fetchall() == [(42,)]

    def test_in_memory_the_view_is_persistent_and_a_cursor_reads_it(self, con):
        con.from_df(frame()).create_view("w")
        assert con.execute("SELECT temporary FROM duckdb_views() WHERE view_name = 'w'").fetchall() == [(False,)]
        assert con.cursor().sql("SELECT a FROM w ORDER BY a").fetchall() == [(1,), (2,), (3,)]

    def test_a_cursor_reads_a_relation_by_name(self, con):
        rel = con.sql("SELECT 42 AS a")  # noqa: F841
        assert con.cursor().sql("SELECT a FROM rel").fetchall() == [(42,)]


class TestARelationReadByNameAcrossConnections:
    """A relation belongs to its database: a cursor or a sibling connection reads it, another database cannot."""

    def test_a_cursor_re_binds_the_relation(self, tmp_path):
        con = duckdb.connect(str(tmp_path / "one.duckdb"))
        con.execute("CREATE TABLE numbers AS SELECT * FROM (VALUES (1), (2)) v(a)")
        numbers = con.sql("SELECT a FROM numbers ORDER BY a")  # noqa: F841
        con.execute("INSERT INTO numbers VALUES (3)")
        assert con.cursor().sql("SELECT a FROM numbers ORDER BY a").fetchall() == [(1,), (2,), (3,)]
        con.close()

    def test_a_sibling_connection_of_the_same_database_reads_it(self, tmp_path):
        path = str(tmp_path / "one.duckdb")
        con = duckdb.connect(path)
        con.execute("CREATE TABLE numbers AS SELECT * FROM (VALUES (1), (2)) v(a)")
        numbers = con.sql("SELECT a FROM numbers ORDER BY a")  # noqa: F841
        sibling = duckdb.connect(path)
        assert sibling.sql("SELECT a FROM numbers ORDER BY a").fetchall() == [(1,), (2,)]
        sibling.close()
        con.close()

    def test_a_connection_to_another_database_refuses_it(self, tmp_path):
        first = duckdb.connect(str(tmp_path / "one.duckdb"))
        second = duckdb.connect(str(tmp_path / "two.duckdb"))
        first.execute("CREATE TABLE numbers AS SELECT * FROM (VALUES (1), (2)) v(a)")
        second.execute("CREATE TABLE numbers AS SELECT * FROM (VALUES (9)) v(a)")
        held = first.sql("SELECT a FROM numbers")  # noqa: F841
        with pytest.raises(duckdb.Error, match="held"):
            second.sql("SELECT a FROM held").fetchall()
        assert second.execute("SELECT a FROM numbers").fetchall() == [(9,)]
        first.close()
        second.close()


class TestTheTextOfARelationOverASource:
    @pytest.mark.parametrize("build", SOURCES)
    def test_the_relation_reads_its_rows(self, con, build):
        assert sorted(build(con).fetchall()) == [(1, "x"), (2, "y"), (3, "z")]

    @pytest.mark.parametrize("build", SOURCES)
    def test_the_text_runs_again_on_the_same_connection(self, con, build):
        rel = build(con).filter("a > 1")
        assert sorted(rel.fetchall()) == [(2, "y"), (3, "z")]
        assert sorted(con.sql(rel.sql_query()).fetchall()) == [(2, "y"), (3, "z")]


class TestCaptureAtTheCall:
    def test_a_name_found_in_sql_keeps_the_object_of_the_call(self, con):
        data = pd.DataFrame({"a": [1]})
        rel = con.sql("SELECT a FROM data")
        data = pd.DataFrame({"a": [9]})  # noqa: F841
        assert rel.fetchall() == [(1,)]

    def test_the_object_of_the_call_lives_after_the_name_moves_on(self, con):
        data = pd.DataFrame({"a": [1]})
        alive = weakref.ref(data)
        rel = con.sql("SELECT a FROM data")
        data = None
        gc.collect()
        assert alive() is not None
        assert rel.fetchall() == [(1,)]

    def test_a_relation_found_by_name_is_read_as_it_was_built(self, con):
        first = con.sql("SELECT 1 AS a")
        rel = con.sql("SELECT a + 1 AS b FROM first")
        first = con.sql("SELECT 10 AS a")  # noqa: F841
        assert rel.fetchall() == [(2,)]


class TestARegistryReferenceKeepsEverySource:
    """A relation read through the registry keeps the sources of its inputs, not only of its own step."""

    def test_a_registered_step_over_a_frame_reads_after_the_intermediates_go(self):
        con = duckdb.connect()

        def build():
            con.register("v", con.from_df(pd.DataFrame({"a": [1, 2, 3]})).filter("a > 1"))

        build()
        gc.collect()
        assert con.sql("SELECT a FROM v ORDER BY a").fetchall() == [(2,), (3,)]

    def test_a_step_read_by_name_inside_a_function_reads_after_it_returns(self):
        con = duckdb.connect()

        def reader():
            rel = con.from_df(pd.DataFrame({"a": [1, 2, 3]})).filter("a > 1")  # noqa: F841
            return con.sql("SELECT a FROM rel ORDER BY a")

        outer = reader()
        gc.collect()
        assert outer.fetchall() == [(2,), (3,)]

    def test_a_view_over_a_step_read_by_name_reads_after_the_function_returns(self):
        con = duckdb.connect()

        def build():
            rel = con.from_df(pd.DataFrame({"a": [1, 2, 3]})).filter("a > 1")  # noqa: F841
            con.sql("SELECT a FROM rel").create_view("w")

        build()
        gc.collect()
        assert con.sql("SELECT a FROM w ORDER BY a").fetchall() == [(2,), (3,)]

    @pytest.mark.parametrize("keep", ["register", "create_view"])
    def test_a_mapped_relation_frees_its_connection(self, keep):
        # The map keeps its input's handles, not the relation, so no cycle holds the connection.
        con = duckdb.connect()
        alive = weakref.ref(con)
        mapped = con.sql("SELECT 1 AS a").map(lambda df: df)
        if keep == "register":
            con.register("m", mapped)
        else:
            mapped.create_view("m")
        del mapped, con
        gc.collect()
        assert alive() is None


class TestUnregisterAfterTheViewWent:
    """Unregister drops its own temporary view if it is there, and never a persistent view of the same name."""

    def test_unregister_after_the_user_dropped_the_view_returns_and_frees_the_object(self):
        con = duckdb.connect()
        obj = frame()
        alive = weakref.ref(obj)
        con.register("v", obj)
        del obj
        con.execute("DROP VIEW v")
        con.unregister("v")
        gc.collect()
        assert alive() is None

    def test_unregister_never_drops_a_persistent_view_of_the_same_name(self):
        con = duckdb.connect()
        con.register("v", frame())
        con.execute("DROP VIEW v")
        con.execute("CREATE VIEW main.v AS SELECT 2 AS a")
        con.unregister("v")
        assert con.execute("SELECT a FROM main.v").fetchall() == [(2,)]


class TestAMissingSourceIsNamed:
    """When a source is gone, the error names its kind and the name it was registered under, never a key."""

    def test_a_rolled_back_unregister_either_restores_the_source_or_names_what_is_missing(self):
        con = duckdb.connect()
        con.register("v", frame())
        con.execute("BEGIN")
        con.unregister("v")
        con.execute("ROLLBACK")
        gc.collect()
        # A view that owns its source reads the rows; one that does not raises an error naming the kind and "v".
        try:
            rows = con.execute("SELECT a FROM v ORDER BY a").fetchall()
        except duckdb.InvalidInputException as error:
            message = str(error)
            assert "v" in re.findall(r"[\"']([^\"']+)[\"']", message)
            assert "DataFrame" in message
            assert "released" in message
            assert GENERATED_PREFIX not in message
        else:
            assert rows == [(1,), (2,), (3,)]


def registry_key(rel):
    return '"' + re.search(GENERATED_PREFIX + r"\w*", rel.sql_query()).group(0) + '"'


class TestAViewOverAQueryThatNamesARegistryKey:
    """A relation's key shows in its rendering and stays live while the relation does, so a query can name it."""

    def test_a_common_table_expression_named_like_a_live_key_reads_its_own_rows(self, con):
        # The CTE shadows the registry entry of the same name inside its query, so its rows are written in.
        held = con.from_df(pd.DataFrame({"x": [1]}))
        key = registry_key(held)
        assert con.execute(f"SELECT x FROM {key}").fetchall() == [(1,)]
        text = f"WITH {key} AS (SELECT 99 AS x) SELECT * FROM {key}"
        expected = con.execute(text).fetchall()
        assert expected == [(99,)]
        con.sql(text).create_view("v")
        assert con.sql("SELECT x FROM v").fetchall() == expected
        assert held.fetchall() == [(1,)]

    def test_a_view_over_a_pivot_whose_value_list_is_a_subquery_reads_after_both_frames_go(self, con):
        # Such a PIVOT runs at the call, so the view reads the held rows, as the shipping client's did.
        con.register("facts", pd.DataFrame({"year": [2020, 2021], "v": [7, 8]}))
        con.register("years", pd.DataFrame({"year": [2020]}))
        text = "PIVOT facts ON year IN (SELECT year FROM years) USING sum(v)"
        expected = con.execute(text).fetchall()
        assert expected == [(7,)]
        con.sql(text).create_view("v")
        con.unregister("facts")
        con.unregister("years")
        gc.collect()
        assert con.sql("SELECT * FROM v").fetchall() == expected

    def test_a_pivot_over_a_registry_key_with_a_literal_value_list_reads_after_the_frame_goes(self, con):
        # The PIVOT's source is written into the view, so the view reads after the frame's relation goes.
        facts = pd.DataFrame({"year": [2020, 2021], "v": [7, 8]})
        held = con.from_df(facts)
        alive = weakref.ref(facts)
        del facts
        text = f"PIVOT {registry_key(held)} ON year IN (2020) USING sum(v)"
        expected = con.execute(text).fetchall()
        assert expected == [(7,)]
        con.sql(text).create_view("v")
        del held
        gc.collect()
        assert alive() is not None
        assert con.sql("SELECT * FROM v").fetchall() == expected


class TestUnregisterAfterARename:
    def test_unregister_of_the_new_name_drops_the_view_and_frees_the_object(self, con):
        # The renamed view is still the one register made, so unregister finds it under its new name.
        obj = pd.DataFrame({"x": [1]})
        alive = weakref.ref(obj)
        con.register("original", obj)
        del obj
        con.execute("ALTER VIEW original RENAME TO renamed")
        assert con.sql("SELECT x FROM renamed").fetchall() == [(1,)]
        con.unregister("renamed")
        gc.collect()
        assert alive() is None
        assert con.execute("SELECT view_name FROM duckdb_views() WHERE NOT internal").fetchall() == []
