"""The array-level column plans a source answers `columns()` with, read by the table function in `numpy_scan.cpp`.

A plan is a `ColumnPlan`: a kind, an engine type text, a data array and a mask array or None. The kinds it
understands are `"fixed"` (a raw fixed-width buffer, missing values from a mask array or, for FLOAT and DOUBLE, a
raw NaN), `"timestamp"` (a naive datetime64 buffer, copied straight through), `"timestamp:<unit>"` (a
UTC-normalized aware timestamp, tagged with its source unit), `"interval:<unit>"` (a timedelta64 buffer, tagged the
same way), `"enum"` (categorical codes) and `"text"` or `"objects"` (a Python-object array, read one value at a
time). An object array's kind and engine type are judged from a sample of its values spread evenly over the array:
the sample unifies under one family below for a typed reading, or falls back to text when it does not.

Nothing here imports pandas: a pandas Series is unwrapped to its numpy array by the caller, in `pandas.py`, before
any function here sees it, and a pandas NA or NaT singleton reaches `_missing` as a plain object an array can hold.
"""

from __future__ import annotations

import datetime
import decimal
import uuid as uuid_module
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

#: Rows a column's kind or an object array's type is inferred from, spread evenly over the array.
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

#: A naive datetime64 unit to the engine type storing that same unit, so the ints copy straight through.
_NAIVE_TIMESTAMP_TYPES = {"s": "TIMESTAMP_S", "ms": "TIMESTAMP_MS", "us": "TIMESTAMP", "ns": "TIMESTAMP_NS"}

#: The first and last instants a TIMESTAMP casts to TIMESTAMP_NS, which is how the scan reads a Python datetime;
#: the cast refuses every instant of 1677-09-21, although TIMESTAMP_NS itself starts during that day.
_NANOSECOND_FIRST = datetime.datetime(1677, 9, 22)
_NANOSECOND_LAST = datetime.datetime(2262, 4, 11, 23, 47, 16, 854775)

#: The largest magnitude BIGINT and HUGEINT can each hold; a sampled int past both is read as text instead.
_INT64_MAX = 2**63 - 1
_INT128_MAX = 2**127 - 1


class ColumnPlan(NamedTuple):
    """`columns()`'s answer for one column: kind, engine type text, data array, mask array or None."""

    kind: str
    type_text: str
    data: object
    mask: object | None


def contiguous(plan: ColumnPlan) -> ColumnPlan:
    """`plan` with its arrays in one contiguous run each, as the scan reads them; a strided view is copied."""
    import numpy as np

    mask = None if plan.mask is None else np.ascontiguousarray(plan.mask)
    return plan._replace(data=np.ascontiguousarray(plan.data), mask=mask)


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
    kind = type(value)
    if kind.__module__ == "numpy":
        import numpy as np

        return isinstance(value, (np.floating, np.datetime64, np.timedelta64)) and bool(value != value)
    return kind.__module__.startswith("pandas") and kind.__name__ in ("NAType", "NaTType")


def _fixed_type_text(dtype: np.dtype[Any]) -> str:
    """The engine type of a fixed-width numpy dtype: boolean, a half-, single- or double-width float, int or uint."""
    if dtype.kind == "b":
        return "BOOLEAN"
    if dtype.kind == "f" and dtype.itemsize == 2:
        # A half-precision column is widened to FLOAT on read; float16 has no native engine type.
        return "FLOAT"
    return _FIXED_TYPES[dtype.str[1:]]


def _float_plan(data: np.ndarray[Any, Any], dtype: np.dtype[Any]) -> ColumnPlan:
    import numpy as np

    if dtype.itemsize == 2:
        data = data.astype(np.float32)
    return ColumnPlan("fixed", _fixed_type_text(dtype), data, None)


def _naive_timestamp_plan(data: np.ndarray[Any, Any], dtype: np.dtype[Any]) -> ColumnPlan:
    import numpy as np

    unit = np.datetime_data(dtype)[0]
    return ColumnPlan("timestamp", _NAIVE_TIMESTAMP_TYPES[unit], data.view(np.int64), None)


def _interval_plan(data: np.ndarray[Any, Any], dtype: np.dtype[Any]) -> ColumnPlan:
    import numpy as np

    unit = np.datetime_data(dtype)[0]
    return ColumnPlan(f"interval:{unit}", "INTERVAL", data.view(np.int64), None)


def _first_valid_position(data: np.ndarray[Any, Any]) -> int | None:
    """The position of the first value in `data` that is not missing, or None when every value is."""
    for position, value in enumerate(data):
        if not _missing(value):
            return position
    return None


def _object_sample(
    data: np.ndarray[Any, Any], first_valid: Callable[[np.ndarray[Any, Any]], int | None]
) -> list[object]:
    """Up to SAMPLE_ROWS values spread over `data`, or its first non-missing value when none of those is one."""
    sample: list[object] = data[_stride(len(data), SAMPLE_ROWS)][:SAMPLE_ROWS].tolist()
    if sample and all(_missing(v) for v in sample):
        position = first_valid(data)
        if position is not None:
            return [data[position]]
    return sample


def _object_kind(
    data: np.ndarray[Any, Any], first_valid: Callable[[np.ndarray[Any, Any]], int | None] = _first_valid_position
) -> tuple[str, str]:
    return _classify_sample(_object_sample(data, first_valid))


def _object_plan(
    data: np.ndarray[Any, Any], first_valid: Callable[[np.ndarray[Any, Any]], int | None] = _first_valid_position
) -> ColumnPlan:
    kind, type_text = _object_kind(data, first_valid)
    return ColumnPlan(kind, type_text, data, None)


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
    for kind, family in _FAMILY_TYPES:
        if isinstance(value, kind):
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
    """Kind and engine type text for an object column, from a sample of its non-null values.

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
