"""The support for scaning over numpy arrays reuses many codes for pandas.
Therefore, we only test the new codes and exec paths.
"""  # noqa: D205

import subprocess
import sys
import textwrap
from datetime import timedelta

import pytest
from conftest import PANDAS_GE_3

import duckdb
import numpy as np


class TestScanNumpy:
    def test_scan_numpy(self, duckdb_cursor):
        z = np.array([1, 2, 3])
        res = duckdb_cursor.sql("select * from z").fetchall()
        assert res == [(1,), (2,), (3,)]

        z = np.array([[1, 2, 3], [4, 5, 6]])
        res = duckdb_cursor.sql("select * from z").fetchall()
        assert res == [(1, 4), (2, 5), (3, 6)]

        z = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        res = duckdb_cursor.sql("select * from z").fetchall()
        assert res == [(1, 4), (2, 5), (3, 6)]

        z = {"z": np.array([1, 2, 3]), "x": np.array([4, 5, 6])}
        res = duckdb_cursor.sql("select * from z").fetchall()
        assert res == [(1, 4), (2, 5), (3, 6)]

        z = np.array(["zzz", "xxx"])
        res = duckdb_cursor.sql("select * from z").fetchall()
        assert res == [("zzz",), ("xxx",)]

        z = [np.array(["zzz", "xxx"]), np.array([1, 2])]
        res = duckdb_cursor.sql("select * from z").fetchall()
        assert res == [("zzz", 1), ("xxx", 2)]

        # test ndarray with dtype = object (python dict)
        z = []
        for i in range(3):
            z.append({str(3 - i): i})
        z = np.array(z)
        res = duckdb_cursor.sql("select * from z").fetchall()
        assert res == [
            ({"3": 0},),
            ({"2": 1},),
            ({"1": 2},),
        ]

        # test timedelta
        delta = timedelta(days=50, seconds=27, microseconds=10, milliseconds=29000, minutes=5, hours=8, weeks=2)
        delta2 = timedelta(days=5, seconds=27, microseconds=10, milliseconds=29000, minutes=5, hours=8, weeks=2)
        z = np.array([delta, delta2])
        res = duckdb_cursor.sql("select * from z").fetchall()
        assert res == [
            (timedelta(days=64, seconds=29156, microseconds=10),),
            (timedelta(days=19, seconds=29156, microseconds=10),),
        ]

        # np.empty
        z = np.empty((3,))
        res = duckdb_cursor.sql("select * from z").fetchall()
        assert len(res) == 3

        # empty list
        z = np.array([])
        res = duckdb_cursor.sql("select * from z").fetchall()
        assert res == []

        # np.nan
        z = np.array([np.nan])
        res = duckdb_cursor.sql("select * from z").fetchall()
        assert res == [(None,)]

        # dict of mixed types
        z = {"z": np.array([1, 2, 3]), "x": np.array(["z", "x", "c"])}
        res = duckdb_cursor.sql("select * from z").fetchall()
        assert res == [(1, "z"), (2, "x"), (3, "c")]

        # list of mixed types
        z = [np.array([1, 2, 3]), np.array(["z", "x", "c"])]
        res = duckdb_cursor.sql("select * from z").fetchall()
        assert res == [(1, "z"), (2, "x"), (3, "c")]

        # currently unsupported formats, will throw duckdb.InvalidInputException

        # list of arrays with different length
        z = [np.array([1, 2]), np.array([3])]
        with pytest.raises(duckdb.InvalidInputException):
            duckdb_cursor.sql("select * from z")

        # dict of ndarrays of different length
        z = {"z": np.array([1, 2]), "x": np.array([3])}
        with pytest.raises(duckdb.InvalidInputException):
            duckdb_cursor.sql("select * from z")

        # high dimensional tensors
        z = np.array([[[1, 2]]])
        with pytest.raises(duckdb.InvalidInputException):
            duckdb_cursor.sql("select * from z")

        # list of ndarrys with len(shape) > 1
        z = [np.array([[1, 2], [3, 4]])]
        with pytest.raises(duckdb.InvalidInputException):
            duckdb_cursor.sql("select * from z")

        # dict of ndarrays with len(shape) > 1
        z = {"x": np.array([[1, 2], [3, 4]])}
        with pytest.raises(duckdb.InvalidInputException):
            duckdb_cursor.sql("select * from z")


@pytest.mark.parametrize(
    "mask",
    [
        pytest.param(None, id="plain"),
        pytest.param([(0, 0), (0, 0), (0, 0)], id="all_false"),
        pytest.param([(0, 0), (0, 1), (0, 0)], id="partial"),
    ],
)
def test_masked_structured_array_fails_like_the_plain_array(duckdb_cursor, mask):
    source = np.zeros(3, dtype=[("a", "i4"), ("b", "i4")])
    if mask is not None:
        source = np.ma.masked_array(source, mask=mask)
    with pytest.raises(duckdb.NotImplementedException, match="not recognized"):
        duckdb_cursor.sql("SELECT * FROM source")


@pytest.mark.parametrize("route", ["register", "variable"])
@pytest.mark.parametrize("length", [2, 60])
def test_first_column_mask_of_another_length_is_refused(duckdb_cursor, route, length):
    source = np.ma.masked_array(np.arange(50, dtype="int64"), mask=np.zeros(50, dtype=bool))
    source._mask = np.zeros(length, dtype=bool)
    message = f"Column 'column0' has {length} rows where its data has 50"
    if route == "register":
        with pytest.raises(duckdb.InvalidInputException, match=message):
            duckdb_cursor.register("registered", source)
    else:
        with pytest.raises(duckdb.InvalidInputException, match=message):
            duckdb_cursor.sql("SELECT * FROM source")


@pytest.mark.skipif(
    np.lib.NumpyVersion(np.__version__) < "2.0.0", reason="numpy below 2 imports numpy.ma together with numpy"
)
def test_mask_of_a_column_that_imports_numpy_ma_when_read():
    script = textwrap.dedent(
        """
        import sys

        import duckdb
        import numpy as np

        assert "numpy.ma" not in sys.modules


        class ImportsOnRead(dict):
            def __getitem__(self, key):
                if key == "masked":
                    import numpy.ma

                    return numpy.ma.masked_array(np.array([1, 2, 3]), mask=[False, True, False])
                return super().__getitem__(key)


        source = ImportsOnRead(plain=np.array([1, 2, 3]), masked=np.zeros(3, dtype="int64"))
        print(duckdb.connect().execute("SELECT * FROM source").fetchall())
        """
    )
    child = subprocess.run(
        [sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=120, check=False
    )
    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == "[(1, 1), (2, None), (3, 3)]"


@pytest.mark.parametrize("query", ["SELECT k FROM {}", "SELECT * FROM {}", "DESCRIBE SELECT * FROM {}"])
def test_object_field_of_a_packed_structured_array(duckdb_cursor, query):
    packed = np.zeros(10, dtype=[("i", "i4"), ("o", "O")])
    packed["o"] = [7, None] * 5
    source = {"k": np.arange(10, dtype="int64"), "x": packed["o"]}
    plain = {"k": source["k"], "x": np.ascontiguousarray(packed["o"])}  # noqa: F841
    expected = duckdb_cursor.execute(query.format("plain")).fetchall()
    assert duckdb_cursor.execute(query.format("source")).fetchall() == expected


@pytest.mark.skipif(not PANDAS_GE_3, reason="before pandas 3 assigning a column copies its values")
def test_object_field_of_a_packed_structured_array_in_a_frame(duckdb_cursor):
    pd = pytest.importorskip("pandas")
    packed = np.zeros(10, dtype=[("i", "i4"), ("o", "O")])
    packed["o"] = [7, None] * 5
    frame = pd.DataFrame({"k": np.arange(10, dtype="int64")})
    frame["x"] = pd.Series(packed["o"], copy=False)
    assert frame["x"].to_numpy().strides == (12,)
    assert duckdb_cursor.execute("SELECT k FROM frame").fetchall() == [(i,) for i in range(10)]
    assert duckdb_cursor.execute("SELECT x FROM frame").fetchall() == [(7,), (None,)] * 5


@pytest.mark.parametrize("dtype", ["int64", "float64", "datetime64[us]"])
def test_numeric_field_of_a_packed_structured_array(duckdb_cursor, dtype):
    packed = np.zeros(4, dtype=[("a", dtype), ("b", "i4")])
    packed["a"] = np.arange(1, 5).astype(dtype)
    packed["b"] = [10, 20, 30, 40]
    source = {"a": packed["a"], "b": packed["b"]}  # noqa: F841
    plain = {"a": np.ascontiguousarray(packed["a"]), "b": np.ascontiguousarray(packed["b"])}  # noqa: F841
    expected = duckdb_cursor.execute("SELECT * FROM plain").fetchall()
    assert duckdb_cursor.execute("SELECT * FROM source").fetchall() == expected


def test_unaligned_contiguous_column(duckdb_cursor):
    unaligned = np.frombuffer(bytes(1) + np.arange(5, dtype="int64").tobytes(), dtype="int64", offset=1)
    assert unaligned.flags.c_contiguous
    assert not unaligned.flags.aligned
    assert duckdb_cursor.execute("SELECT * FROM unaligned").fetchall() == [(i,) for i in range(5)]


class _ZeroItemsize(np.ndarray):
    @property
    def itemsize(self):
        return 0


def test_array_subclass_overriding_itemsize(duckdb_cursor):
    source = {"a": np.arange(5, dtype="int64").view(_ZeroItemsize)}  # noqa: F841
    assert duckdb_cursor.execute("SELECT * FROM source").fetchall() == [(i,) for i in range(5)]
