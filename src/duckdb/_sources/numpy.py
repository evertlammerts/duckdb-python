"""The numpy source, and how the numpy scan reads a column of it or of a pandas DataFrame.

The table function in `numpy_scan.cpp` reads each column from a `ScanColumn`: a data array, a mask array or None
(True marking a missing value), the engine type of the column, and the encoding of the data array, which says how
the scan turns its bytes into that type. The encodings are `"fixed"` (fixed-width numbers or booleans copied as they
are, a raw NaN also missing for FLOAT and DOUBLE without a mask), `"timestamp:<unit>"` and `"interval:<unit>"`
(int64 counts of a timestamp, naive or normalized to UTC as its engine type says, or of a duration, in a unit
spelled as numpy spells it inside a datetime64 or timedelta64 dtype's brackets, such as `ns` or `2ns`), `"enum"`
(categorical codes), `"ucs4"` and `"bytes"` (numpy's fixed-width `U` and `S` strings) and `"text"` or `"objects"`
(Python objects, read one at a time as text or converted to the engine type).

`ColumnReading` holds the one decision made about a column: its engine type, which is what a query is bound
against, and how to produce its `ScanColumn`, which is done only when a query's scan starts and only for the
columns it reads. An object array's engine type is judged from a sample of its values spread evenly over the
array: the sample unifies under one family below for a typed reading, or falls back to text when it does not.

Nothing here imports pandas: a pandas Series is unwrapped to its numpy array by the caller, in `pandas.py`, before
any function here sees it, and a pandas NA or NaT singleton reaches `_missing` as a plain object an array can hold.
"""

from __future__ import annotations

import datetime
import decimal
import uuid as uuid_module
from typing import TYPE_CHECKING, Any, NamedTuple

from . import Source, _unique_names

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy as np

#: Rows an object array's engine type is inferred from, spread evenly over the array.
SAMPLE_ROWS = 1000

#: numpy dtype code (byte order dropped) to the engine type of matching width.
_FIXED_TYPES = {
    "i1": "TINYINT",
    "i2": "SMALLINT",
    "i4": "INTEGER",
    "i8": "BIGINT",
    "u1": "UTINYINT",
    "u2": "USMALLINT",
    "u4": "UINTEGER",
    "u8": "UBIGINT",
    "f4": "FLOAT",
    "f8": "DOUBLE",
}

#: A naive datetime64 unit to the engine type storing that same unit, so the ints copy straight through; hours and
#: minutes are read into TIMESTAMP_S.
_NAIVE_TIMESTAMP_TYPES = {"s": "TIMESTAMP_S", "ms": "TIMESTAMP_MS", "us": "TIMESTAMP", "ns": "TIMESTAMP_NS"}

#: The first and last instants a TIMESTAMP casts to TIMESTAMP_NS, which is how the scan reads a Python datetime;
#: the cast refuses every instant of 1677-09-21, although TIMESTAMP_NS itself starts during that day.
_NANOSECOND_FIRST = datetime.datetime(1677, 9, 22)
_NANOSECOND_LAST = datetime.datetime(2262, 4, 11, 23, 47, 16, 854775)

_SUB_NANOSECOND_UNITS = ("ps", "fs", "as")

#: numpy's NaT, as the int64 its datetime64 and timedelta64 buffers hold.
_NAT = -(2**63)

#: The most days a DATE holds either side of 1970-01-01; the next day on each side is an infinity.
_DATE_LIMIT = 2**31 - 2

#: A bound on a count of months or years, checked before numpy converts it to days, since that conversion wraps on
#: overflow; far past DATE's range, which the days are checked against after.
_CALENDAR_LIMIT = 2**40

#: The largest magnitude BIGINT and HUGEINT can each hold; a sampled int past both is read as text instead.
_INT64_MAX = 2**63 - 1
_INT128_MAX = 2**127 - 1


class ScanColumn(NamedTuple):
    """`columns()`'s answer for one column, after its name: encoding, engine type, data array, mask array or None."""

    encoding: str
    type_text: str
    data: object
    mask: object | None


class ColumnReading(NamedTuple):
    """A column's engine type, and a function producing the `ScanColumn` the scan reads it from.

    `prepare` may copy or convert the data, so describing a table to bind a query never calls it.
    """

    type_text: str
    prepare: Callable[[], ScanColumn]


def _stride(length: int, sample_rows: int) -> slice:
    """Every step-th index over `length` items, the step chosen so at most `sample_rows` of them land in the stride."""
    step = max(1, length // sample_rows)
    return slice(None, None, step)


def _missing(value: object) -> bool:
    """Whether `value` is a null marker: None, a float NaN of any width, numpy's NaT, or pandas' NA/NaT singletons.

    Never calls pandas' own `isna`, which turns a list or dict argument into an array rather than a bool. The same
    rule as `IsNoneLike` in numpy_scan.cpp.
    """
    if value is None:
        return True
    if isinstance(value, float):
        return value != value
    cls = type(value)
    if cls.__module__ == "numpy":
        import numpy as np

        return isinstance(value, (np.floating, np.datetime64, np.timedelta64)) and bool(value != value)
    return cls.__module__.startswith("pandas") and cls.__name__ in ("NAType", "NaTType")


def _first_valid_position(data: np.ndarray[Any, Any]) -> int | None:
    """The position of the first value in `data` that is not missing, or None when every value is."""
    for position, value in enumerate(data):
        if not _missing(value):
            return position
    return None


def _object_sample(
    data: np.ndarray[Any, Any],
    first_valid: Callable[[np.ndarray[Any, Any]], int | None],
    mask: np.ndarray[Any, Any] | None = None,
) -> list[object]:
    """Up to SAMPLE_ROWS unmasked values spread over `data`, or its first non-missing unmasked one if none is."""
    stride = _stride(len(data), SAMPLE_ROWS)
    picked = data[stride][:SAMPLE_ROWS]
    if mask is not None:
        picked = picked[~mask[stride][:SAMPLE_ROWS]]
    sample: list[object] = picked.tolist()
    if len(data) and all(_missing(v) for v in sample):
        position = first_valid(data) if mask is None else _first_unmasked_valid_position(data, mask)
        if position is not None:
            return [data[position]]
    return sample


def _first_unmasked_valid_position(data: np.ndarray[Any, Any], mask: np.ndarray[Any, Any]) -> int | None:
    import numpy as np

    for position in np.flatnonzero(~mask):
        if not _missing(data[position]):
            return int(position)
    return None


def _classify_objects(
    data: np.ndarray[Any, Any],
    first_valid: Callable[[np.ndarray[Any, Any]], int | None] = _first_valid_position,
    mask: np.ndarray[Any, Any] | None = None,
) -> tuple[str, str]:
    return _classify_sample(_object_sample(data, first_valid, mask))


#: Python type to the family it unifies under; checked in order, so a bool (an int subclass) is caught first.
_FAMILY_TYPES: tuple[tuple[type, str], ...] = (
    (bool, "bool"),
    (int, "number"),
    (float, "number"),
    (decimal.Decimal, "decimal"),
    (datetime.time, "time"),
    (datetime.timedelta, "interval"),
    (bytes, "bytes"),
    (uuid_module.UUID, "uuid"),
)

#: A singleton family's engine type, for the families that need no further bookkeeping to pick one.
_SIMPLE_FAMILY_TYPES = {"bool": "BOOLEAN", "time": "TIME", "interval": "INTERVAL", "bytes": "BLOB", "uuid": "UUID"}


def _value_family(value: object) -> str | None:
    """Which family `value` unifies under, or None when its mere presence takes the whole column to text."""
    if type(value).__module__ == "numpy":
        # Only the temporal scalars `_significant_values` leaves wrapped reach here as numpy scalars.
        import numpy as np

        if isinstance(value, np.datetime64):
            return "naive_ns"
        if isinstance(value, np.timedelta64):
            return "interval"
        return None
    if isinstance(value, datetime.datetime):
        return "aware" if value.tzinfo is not None else "naive"
    if isinstance(value, datetime.date):
        return "date"
    for cls, family in _FAMILY_TYPES:
        if isinstance(value, cls):
            return family
    return None


def _significant_values(sample: list[object]) -> list[object]:
    """`sample` with missing markers dropped and numpy scalars unwrapped through `.item()`.

    A datetime64 or timedelta64 that `.item()` would turn into a bare int, as it does for a nanosecond unit, stays
    wrapped so it is still read as a time.
    """
    import numpy as np

    def unwrap(value: object) -> object:
        if not isinstance(value, np.generic):
            return value
        item = value.item()
        if isinstance(item, int) and isinstance(value, (np.datetime64, np.timedelta64)):
            return value
        return item

    return [unwrap(v) for v in sample if not _missing(v)]


def _classify_sample(sample: list[object]) -> tuple[str, str]:
    """Encoding and engine type for an object column, from a sample of its non-null values.

    `"text"` (VARCHAR) when the sample is empty, holds only strings, or does not unify under one family below;
    `"objects"` otherwise, with each value later read through `PythonToValue` and cast to the chosen type.
    """
    values = _significant_values(sample)
    if not values or all(isinstance(v, str) for v in values):
        return "text", "VARCHAR"

    temporal: set[str] = set()
    beyond_nanoseconds = False
    other: set[str] = set()
    has_float = False
    max_abs_int = 0
    max_scale = 0
    for value in values:
        family = _value_family(value)
        if family is None:
            return "text", "VARCHAR"
        if family in ("date", "naive", "naive_ns", "aware"):
            temporal.add(family)
            if family != "aware" and isinstance(value, datetime.date):
                midnight = datetime.time()
                instant = value if isinstance(value, datetime.datetime) else datetime.datetime.combine(value, midnight)
                beyond_nanoseconds = beyond_nanoseconds or not _NANOSECOND_FIRST <= instant <= _NANOSECOND_LAST
            continue
        other.add(family)
        if isinstance(value, float):
            has_float = True
        elif isinstance(value, int):
            max_abs_int = max(max_abs_int, abs(value))
        elif isinstance(value, decimal.Decimal):
            exponent = value.as_tuple().exponent
            if isinstance(exponent, int):
                max_scale = max(max_scale, -exponent)

    if len(other) + bool(temporal) > 1 or ("aware" in temporal and temporal != {"aware"}):
        return "text", "VARCHAR"

    if "aware" in temporal:
        return "objects", "TIMESTAMP WITH TIME ZONE"
    if "naive_ns" in temporal and not beyond_nanoseconds:
        return "objects", "TIMESTAMP_NS"
    if "naive_ns" in temporal:
        # A sampled value past TIMESTAMP_NS's range needs the wider type, which refuses a sub-microsecond value.
        return "objects", "TIMESTAMP"
    if "naive" in temporal:
        return "objects", "TIMESTAMP"
    if "date" in temporal:
        return "objects", "DATE"

    [family] = other
    if family == "number":
        if has_float:
            return "objects", "DOUBLE"
        if max_abs_int > _INT128_MAX:
            return "text", "VARCHAR"
        if max_abs_int > _INT64_MAX:
            return "objects", "HUGEINT"
        return "objects", "BIGINT"
    if family == "decimal":
        return "objects", f"DECIMAL(38,{min(max_scale, 38)})"
    return "objects", _SIMPLE_FAMILY_TYPES[family]


class NumpySource(Source):
    """A numpy array, or a dict, list or tuple of one-dimensional numpy arrays, read off the arrays' own buffers.

    A one-dimensional array is one column and a two-dimensional one of shape (n, k) is k columns of n rows, named
    `column0`, `column1` and so on, as are the arrays of a list or tuple; a dict's keys name its columns as their
    text, a repeat getting a numbered suffix. The mask of a numpy masked array marks missing values. The object is
    read as it is at each query, so a change to it after registration shows in the next one.
    """

    native = True

    def __init__(self, obj: object) -> None:
        super().__init__(obj)
        try:
            import numpy  # noqa: F401
        except ImportError as error:
            message = f"registering a {type(obj).__name__} reads it as numpy arrays, which needs numpy"
            raise TypeError(message) from error
        for name, array in self._arrays():
            _numpy_reading(name, array)

    def _arrays(self) -> list[tuple[str, np.ndarray[Any, Any]]]:
        """The object's columns as named one-dimensional arrays, all of one length."""
        import numpy as np

        obj = self.obj
        if isinstance(obj, np.ndarray):
            array = _plain(obj)
            if array.ndim == 1:
                arrays = [("column0", array)]
            elif array.ndim == 2 and array.shape[1] > 0:
                arrays = [(f"column{j}", array[:, j]) for j in range(array.shape[1])]
            elif array.ndim == 2:
                message = "a registered two-dimensional numpy array needs at least one column"
                raise ValueError(message)
            else:
                message = f"a registered numpy array must be one- or two-dimensional, not {array.ndim}-dimensional"
                raise TypeError(message)
        elif isinstance(obj, dict):
            if not obj:
                message = "a registered dict needs at least one array"
                raise ValueError(message)
            names = _unique_names([str(key) for key in obj])
            arrays = [
                (name, _one_dimensional(value, f"key {key!r}"))
                for name, (key, value) in zip(names, obj.items(), strict=True)
            ]
        elif isinstance(obj, (list, tuple)):
            if not obj:
                message = f"a registered {type(obj).__name__} needs at least one array"
                raise ValueError(message)
            arrays = [(f"column{i}", _one_dimensional(value, f"position {i}")) for i, value in enumerate(obj)]
        else:
            message = f"a registered numpy object must be an array, not a {type(obj).__name__}"
            raise TypeError(message)
        first, length = arrays[0][0], len(arrays[0][1])
        for name, array in arrays[1:]:
            if len(array) != length:
                message = f"column '{name}' has {len(array)} rows where '{first}' has {length}"
                raise ValueError(message)
        return arrays

    def rows(self) -> int | None:
        return len(self._arrays()[0][1])

    def describe(self) -> list[tuple[str, str]]:
        """Column names and engine types, which a query is bound against."""
        return [(name, _numpy_reading(name, array).type_text) for name, array in self._arrays()]

    def columns(self, columns: Sequence[int] | None) -> list[tuple[str, str, str, object, object | None]]:
        """The name and `ScanColumn` of each requested column, all of them when None."""
        arrays = self._arrays()
        chosen = arrays if columns is None else [arrays[i] for i in columns]
        return [(name, *_numpy_reading(name, array).prepare()) for name, array in chosen]


def _plain(array: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """`array` as a base ndarray or a masked array, sharing its data; a column of a `numpy.matrix` stays 2-D."""
    import numpy as np

    return array if isinstance(array, np.ma.MaskedArray) else np.asarray(array)


def _one_dimensional(value: object, where: str) -> np.ndarray[Any, Any]:
    import numpy as np

    if not isinstance(value, np.ndarray):
        message = f"the value at {where} is a {type(value).__name__}, not a numpy array"
        raise TypeError(message)
    if value.ndim != 1:
        message = f"the array at {where} is {value.ndim}-dimensional, not one-dimensional"
        raise TypeError(message)
    return _plain(value)


def _refused(name: str, dtype: np.dtype[Any], reason: str) -> TypeError:
    return TypeError(f"column '{name}' has the numpy dtype {dtype}, {reason}")


def _numpy_reading(name: str, array: np.ndarray[Any, Any]) -> ColumnReading:
    import numpy as np

    mask = None if np.ma.getmask(array) is np.ma.nomask else np.ma.getmaskarray(array)
    return _array_reading(name, np.ma.getdata(array), mask)


def _array_reading(
    name: str,
    data: np.ndarray[Any, Any],
    mask: np.ndarray[Any, Any] | None,
    first_valid: Callable[[np.ndarray[Any, Any]], int | None] = _first_valid_position,
) -> ColumnReading:
    """How the scan reads the one-dimensional array `data`, True in `mask` marking a missing value.

    Refuses a dtype no engine type holds. `first_valid` finds the first value that is not missing in an object
    array whose sample holds none.
    """
    dtype = data.dtype
    if dtype.kind == "O":
        encoding, type_text = _classify_objects(data, first_valid, mask)
        return ColumnReading(type_text, lambda: ScanColumn(encoding, type_text, data, mask))
    type_text, convert = _dtype_reading(name, dtype)

    def prepare() -> ScanColumn:
        # The scan copies bytes as they are, so an array in the other byte order is swapped by a copy first.
        native = data if dtype.isnative else data.astype(dtype.newbyteorder("="))
        encoding, values, missing = convert(native, mask)
        return ScanColumn(encoding, type_text, values, missing)

    return ColumnReading(type_text, prepare)


_Converted = tuple[str, "np.ndarray[Any, Any]", "np.ndarray[Any, Any] | None"]


def _dtype_reading(
    name: str, dtype: np.dtype[Any]
) -> tuple[str, Callable[[np.ndarray[Any, Any], np.ndarray[Any, Any] | None], _Converted]]:
    """The engine type of a column of `dtype`, other than objects, and how its array becomes what the scan reads."""
    import numpy as np

    kind = dtype.kind
    if kind == "b":
        return "BOOLEAN", lambda data, mask: ("fixed", data, mask)
    if kind in "iu" or (kind == "f" and dtype.itemsize in (4, 8)):
        return _FIXED_TYPES[dtype.str[1:]], lambda data, mask: ("fixed", data, mask)
    if kind == "f" and dtype.itemsize == 2:
        # float16 has no engine type, so it is widened to FLOAT.
        return "FLOAT", lambda data, mask: ("fixed", data.astype(np.float32), mask)
    if kind in "Mm":
        unit = np.datetime_data(dtype)[0]
        if unit == "generic":
            raise _refused(name, dtype, "which has no unit")
        if unit in _SUB_NANOSECOND_UNITS:
            raise _refused(name, dtype, "which is finer than the engine's nanoseconds")
        if kind == "m" and unit in ("M", "Y"):
            raise _refused(name, dtype, "which counts months or years, and those have no fixed length")
        if kind == "M" and unit in ("D", "W", "M", "Y"):
            return "DATE", lambda data, mask: _date_arrays(name, data, mask)
        # The scan converts the counts itself, from numpy's own spelling of the unit and its step, such as "2ns".
        spelled = dtype.str.partition("[")[2][:-1]
        if kind == "m":
            return "INTERVAL", lambda data, mask: (f"interval:{spelled}", data.view(np.int64), mask)
        return _NAIVE_TIMESTAMP_TYPES.get(unit, "TIMESTAMP_S"), lambda data, mask: (
            f"timestamp:{spelled}",
            data.view(np.int64),
            mask,
        )
    if kind == "U":
        return "VARCHAR", lambda data, mask: ("ucs4", data, mask)
    if kind == "S":
        return "BLOB", lambda data, mask: ("bytes", data, mask)
    if kind == "T":
        # A StringDType array exports no buffer, so its values are read as Python strings.
        return "VARCHAR", lambda data, mask: ("text", data.astype(object), mask)
    reasons = {
        "f": "and no engine type has its width",
        "c": "and no engine type holds a complex number",
        "V": "which is structured or raw bytes, and no single engine type holds it",
    }
    raise _refused(name, dtype, reasons.get(kind, "and no engine type holds it"))


def _checked_scale(
    name: str, raw: np.ndarray[Any, Any], factor: int, limit: int, mask: np.ndarray[Any, Any] | None
) -> np.ndarray[Any, Any]:
    """`raw` times `factor`, refusing a column whose product would pass `limit` in a cell neither NaT nor masked."""
    import numpy as np

    valid = raw != _NAT
    if mask is not None:
        valid &= ~mask
    largest = max(int(np.max(raw, where=valid, initial=0)), -int(np.min(raw, where=valid, initial=0)))
    if largest * factor > limit:
        message = f"column '{name}' holds a value beyond the range of its engine type"
        raise ValueError(message)
    return raw if factor == 1 else np.where(valid, raw * factor, raw)


def _date_arrays(name: str, data: np.ndarray[Any, Any], mask: np.ndarray[Any, Any] | None) -> _Converted:
    """A datetime64 of days, weeks, months or years as DATE: int32 days, NaT and masked cells marked missing."""
    import numpy as np

    unit, count = np.datetime_data(data.dtype)
    raw = data.view(np.int64)
    if unit in ("M", "Y"):
        _checked_scale(name, raw, count, _CALENDAR_LIMIT, mask)
        # A masked cell may hold anything, and numpy's conversion to days wraps on overflow.
        cleared = raw if mask is None else np.where(mask, 0, raw)
        days = cleared.view(data.dtype).astype("datetime64[D]").view(np.int64)
    else:
        days = raw
    days = _checked_scale(name, days, count if unit == "D" else 7 * count if unit == "W" else 1, _DATE_LIMIT, mask)
    missing = days == _NAT if mask is None else (days == _NAT) | mask
    return "fixed", np.where(missing, 0, days).astype(np.int32), missing if missing.any() else None
