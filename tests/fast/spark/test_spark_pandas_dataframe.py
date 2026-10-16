import datetime
import math

import pytest

_ = pytest.importorskip("duckdb.experimental.spark")
pd = pytest.importorskip("pandas")

from pandas.testing import assert_frame_equal
from spark_namespace import USE_ACTUAL_SPARK
from spark_namespace.errors import PySparkTypeError, PySparkValueError
from spark_namespace.sql.types import (
    ArrayType,
    IntegerType,
    Row,
    StringType,
    StructField,
    StructType,
)

if USE_ACTUAL_SPARK:
    import pyspark


@pytest.fixture
def pandasDF(spark):
    data = [["Scott", 50], ["Jeff", 45], ["Thomas", 54], ["Ann", 34]]
    # Create the pandas DataFrame
    df = pd.DataFrame(data, columns=["Name", "Age"])
    return df


class TestPandasDataFrame:
    def test_pd_conversion_basic(self, spark, pandasDF):
        sparkDF = spark.createDataFrame(pandasDF)
        res = sparkDF.collect()
        sparkDF.show()
        expected = [
            Row(Name="Scott", Age=50),
            Row(Name="Jeff", Age=45),
            Row(Name="Thomas", Age=54),
            Row(Name="Ann", Age=34),
        ]
        assert res == expected

    def test_pd_conversion_schema(self, spark, pandasDF):
        mySchema = StructType([StructField("First Name", StringType(), True), StructField("Age", IntegerType(), True)])
        sparkDF = spark.createDataFrame(pandasDF, schema=mySchema)
        sparkDF.show()
        res = sparkDF.collect()
        expected = (
            "[Row(First Name='Scott', Age=50), Row(First Name='Jeff', Age=45), "
            "Row(First Name='Thomas', Age=54), Row(First Name='Ann', Age=34)]"
        )
        assert str(res) == expected

    def test_spark_to_pandas_dataframe(self, spark, pandasDF):
        sparkDF = spark.createDataFrame(pandasDF)
        res = sparkDF.toPandas()
        assert_frame_equal(res, pandasDF)

    def test_categorical_column(self, spark):
        pdf = pd.DataFrame({"c": pd.Categorical(["a", "b", None, "a"])})
        df = spark.createDataFrame(pdf)
        assert df.dtypes == [("c", "string")]
        # Spark 3.5 stores the text of the missing value, as it does for other pandas missing values.
        missing = "NaN" if USE_ACTUAL_SPARK and pyspark.__version__.startswith("3.") else None
        assert [row.c for row in df.collect()] == ["a", "b", missing, "a"]


def _local(value):
    return value.astimezone().replace(tzinfo=None)


class TestPandasTimestamps:
    def test_naive_datetimes_are_timestamps(self, spark):
        values = [datetime.datetime(2024, 1, 1, 12), datetime.datetime(2024, 7, 1, 12), None]
        df = spark.createDataFrame(pd.DataFrame({"t": values, "i": [1, 2, 3]}))
        assert df.dtypes == [("t", "timestamp"), ("i", "bigint")]
        assert df.collect() == [Row(t=values[0], i=1), Row(t=values[1], i=2), Row(t=None, i=3)]
        assert df.toPandas()["t"].tolist()[:2] == values[:2]

    @pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
    def test_naive_datetime_units(self, spark, unit):
        series = pd.Series([datetime.datetime(2024, 1, 1, 12)]).astype(f"datetime64[{unit}]")
        df = spark.createDataFrame(pd.DataFrame({"t": series}))
        assert df.dtypes == [("t", "timestamp")]
        assert df.collect() == [Row(t=datetime.datetime(2024, 1, 1, 12))]

    @pytest.mark.parametrize("zone", ["UTC", "Asia/Tokyo"])
    def test_aware_datetimes(self, spark, zone):
        series = pd.Series([datetime.datetime(2024, 1, 1, 12)]).dt.tz_localize(zone)
        df = spark.createDataFrame(pd.DataFrame({"t": series}))
        expected = _local(series[0].to_pydatetime())
        assert df.dtypes == [("t", "timestamp")]
        assert df.collect() == [Row(t=expected)]
        assert df.toPandas()["t"].tolist() == [expected]

    def test_to_pandas_gives_naive_local_times(self, spark):
        value = datetime.datetime(2024, 1, 1, 12, tzinfo=datetime.timezone.utc)
        df = spark.createDataFrame([(value, datetime.datetime(2024, 1, 1, 12))], "t timestamp, n timestamp_ntz")
        res = df.toPandas()
        assert res["t"].dt.tz is None
        assert res["t"].tolist() == [_local(value)]
        assert res["n"].tolist() == [datetime.datetime(2024, 1, 1, 12)]

    def test_to_pandas_after_a_join(self, spark):
        left = spark.createDataFrame([(1, datetime.datetime(2024, 1, 1, 12))], "id int, t timestamp")
        right = spark.createDataFrame([(1, datetime.datetime(2024, 2, 1, 12))], "id int, t timestamp")
        res = left.join(right, left.id == right.id).toPandas()
        assert res.values.tolist() == [[1, datetime.datetime(2024, 1, 1, 12), 1, datetime.datetime(2024, 2, 1, 12)]]

    def test_to_pandas_of_dates(self, spark):
        df = spark.createDataFrame([(datetime.date(2024, 1, 2), 1), (None, 2)], "d date, i int")
        res = df.toPandas()
        assert str(res["d"].dtype) == "object"
        assert res["d"].tolist() == [datetime.date(2024, 1, 2), None]
        missing = spark.createDataFrame([(None,)], "d date").toPandas()
        assert str(missing["d"].dtype) == "object"
        assert missing["d"].tolist() == [None]
        res = spark.createDataFrame(
            [(datetime.date(1, 1, 1), {"k": datetime.date(9999, 12, 31)})], "d date, m map<string,date>"
        )
        assert res.toPandas().values.tolist() == [[datetime.date(1, 1, 1), {"k": datetime.date(9999, 12, 31)}]]

    def test_to_pandas_keeps_repeated_labels(self, spark):
        df = spark.createDataFrame([(1, 2)], ["a", "b"])
        assert df.toDF("x", "x").toPandas().columns.tolist() == ["x", "x"]
        assert df.toDF("x", "X").toPandas().columns.tolist() == ["x", "X"]
        left = spark.createDataFrame([(1, "l")], ["id", "v"])
        right = spark.createDataFrame([(1, "r")], ["id", "v"])
        res = left.join(right, left.id == right.id).toPandas()
        assert res.columns.tolist() == ["id", "v", "id", "v"]
        assert res.values.tolist() == [[1, "l", 1, "r"]]


class TestPandasNullability:
    @pytest.mark.parametrize(
        ("data", "schema"),
        [
            ({"v": [None]}, "v string not null"),
            ({"a": ["x"], "v": [None]}, "a string not null, v string not null"),
            ({"v": pd.Series([1, None], dtype="object")}, "v bigint not null"),
            ({"v": [pd.NaT, pd.Timestamp("2024-01-01")]}, "v timestamp not null"),
        ],
    )
    def test_none_in_a_field_that_is_not_nullable(self, spark, data, schema):
        with pytest.raises(PySparkValueError):
            spark.createDataFrame(pd.DataFrame(data), schema)

    def test_values_in_fields_that_are_not_nullable(self, spark):
        df = spark.createDataFrame(pd.DataFrame({"a": ["x", "y"], "v": [None, "z"]}), "a string not null, v string")
        assert df.collect() == [Row(a="x", v=None), Row(a="y", v="z")]
        floats = pd.DataFrame({"v": [1.0, None]})
        if USE_ACTUAL_SPARK:
            # Spark keeps a NaN, which is how pandas stores a missing float.
            (_, row) = spark.createDataFrame(floats, "v double not null").collect()
            assert math.isnan(row.v)
        else:
            # DuckDB reads a NaN in a pandas float column as NULL, which the field does not allow.
            for schema in ("v double not null", "v float not null"):
                with pytest.raises(PySparkValueError):
                    spark.createDataFrame(floats, schema)

    @pytest.mark.parametrize(
        "values", [pd.Series([float("nan"), "a"], dtype=object), pd.Series([pd.NA, "a"], dtype=object), [1.5, None]]
    )
    def test_missing_values_in_a_string_field_that_is_not_nullable(self, spark, values):
        df = pd.DataFrame({"v": values})
        if USE_ACTUAL_SPARK:
            # Spark keeps the text of the missing value.
            assert None not in [row.v for row in spark.createDataFrame(df, "v string not null").collect()]
        else:
            # DuckDB would read the missing value as NULL.
            with pytest.raises(PySparkValueError):
                spark.createDataFrame(df, "v string not null")

    def test_none_in_a_nested_field_that_is_not_nullable(self, spark):
        with pytest.raises(PySparkValueError):
            spark.createDataFrame(pd.DataFrame({"s": [{"a": None}]}), "s struct<a:int not null>")
        elements_not_null = StructType([StructField("l", ArrayType(IntegerType(), False))])
        with pytest.raises(PySparkValueError):
            spark.createDataFrame(pd.DataFrame({"l": [[1, None]]}), elements_not_null)
        df = spark.createDataFrame(pd.DataFrame({"s": [{"a": 1}, None]}), "s struct<a:int not null>")
        assert df.collect() == [Row(s=Row(a=1)), Row(s=None)]


class TestPandasValueTypes:
    @pytest.mark.parametrize(
        ("value", "schema"),
        [
            (["001"], "v array<int>"),
            (["001"], "v array<int> not null"),
            ({"a": "001"}, "v struct<a:int>"),
            ({"k": "001"}, "v map<string,int>"),
            ([1, "x"], "v array<bigint>"),
        ],
    )
    def test_nested_value_of_another_type(self, spark, value, schema):
        with pytest.raises(PySparkTypeError):
            spark.createDataFrame(pd.DataFrame({"v": [value]}), schema)

    def test_nested_values_that_fit(self, spark):
        stamp = pd.Timestamp("2024-01-02 03:04:05")
        df = pd.DataFrame({"v": [[1, None], None], "s": [{"a": "x"}, {"a": None}], "t": [[stamp], [None]]})
        res = spark.createDataFrame(df, "v array<int>, s struct<a:string>, t array<timestamp>")
        assert res.collect() == [
            Row(v=[1, None], s=Row(a="x"), t=[stamp.to_pydatetime()]),
            Row(v=None, s=Row(a=None), t=[None]),
        ]

    def test_integers_for_a_double_field(self, spark):
        df = pd.DataFrame({"v": [[1, 2]], "d": pd.Series([1], dtype=object)})
        if USE_ACTUAL_SPARK and not pyspark.__version__.startswith("3."):
            # Spark 4 converts the integers; the module follows Spark 3.5's verifier.
            assert spark.createDataFrame(df, "v array<double>, d double").collect() == [Row(v=[1.0, 2.0], d=1.0)]
        else:
            for schema in ("v array<double>, d string", "v string, d double"):
                with pytest.raises(PySparkTypeError):
                    spark.createDataFrame(df, schema)

    @pytest.mark.parametrize(
        ("values", "schema"),
        [
            ([1.5], "a int"),
            ([1.0, None], "a bigint not null"),
            ([1.5], "a decimal(10,2)"),
        ],
    )
    def test_float_that_does_not_fit(self, spark, values, schema):
        with pytest.raises(PySparkTypeError):
            spark.createDataFrame(pd.DataFrame({"a": values}), schema)

    def test_whole_floats_for_an_integral_field(self, spark):
        data = pd.DataFrame({"a": [2.0, None]})
        if USE_ACTUAL_SPARK and pyspark.__version__.startswith("3."):
            # Spark 3.5 rejects every float for an integral field; the module follows Spark 4.
            with pytest.raises(PySparkTypeError):
                spark.createDataFrame(data, "a bigint")
        else:
            assert spark.createDataFrame(data, "a bigint").collect() == [Row(a=2), Row(a=None)]

    def test_values_that_fit(self, spark):
        assert spark.createDataFrame(pd.DataFrame({"a": [1, 2]}), "a bigint").collect() == [Row(a=1), Row(a=2)]
        assert spark.createDataFrame(pd.DataFrame({"a": [1.5]}), "a double").collect() == [Row(a=1.5)]
