"""The pandas source: read natively off its own buffers, or through pyarrow a slice at a time.

A pandas DataFrame whose columns are all numpy- or Python-object-backed is read natively: `describe()` names its
columns and their engine types for `pandas_scan.cpp`'s bind, and `columns()` answers each requested column as a
kind, an engine type, a data array and a mask array or None, straight off the DataFrame's own buffers. A DataFrame
with a pyarrow-backed column (an `ArrowExtensionArray`, which includes `ArrowStringArray`, the default for strings
when pyarrow is installed) is read through pyarrow instead, converted a slice of rows at a time so the DataFrame is
never copied whole. Either way, an object column's engine type is judged from a sample of its values spread evenly
over the column: the sample unifies under one family below for a typed reading, or falls back to text.
"""

from __future__ import annotations

import datetime
import decimal
import uuid as uuid_module
from typing import TYPE_CHECKING, Any, NamedTuple

from . import Source, _unique_names

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import numpy as np
    import pandas as pd
    from pandas import DataFrame
    from pyarrow import RecordBatch, Schema, Table

    from .._expressions import Expr

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

#: The largest magnitude BIGINT and HUGEINT can each hold; a sampled int past both is read as text instead.
_INT64_MAX = 2**63 - 1
_INT128_MAX = 2**127 - 1


class ColumnPlan(NamedTuple):
    """`columns()`'s answer for one column: kind, engine type text, data array, mask array or None."""

    kind: str
    type_text: str
    data: object
    mask: object | None


class PandasSource(Source):
    """A pandas DataFrame, read natively when every column is numpy- or Python-object-backed, through pyarrow otherwise.

    A pyarrow-backed column (an `ArrowExtensionArray`, which includes `ArrowStringArray`, the default for strings
    when pyarrow is installed) sends the whole DataFrame through the Arrow path below, converted a slice of rows at a
    time so the DataFrame is never copied whole; `native` is False for such a DataFrame. Otherwise `describe()` and
    `columns()` serve `pandas_scan.cpp`'s table function directly off the DataFrame's own buffers.

    The index is left out, as the previous client did. Column labels become their text, and a label repeating
    another gets a numbered suffix, as the engine names a repeated Arrow field.
    """

    #: Rows the types are inferred from, spread evenly over the DataFrame.
    SAMPLE_ROWS = 1000
    #: Rows converted per slice on the Arrow path; the memory held at any time is one slice's worth of Arrow.
    SLICE_ROWS = 1 << 16

    def __init__(self, obj: object) -> None:
        super().__init__(obj)
        try:
            import pyarrow
        except ImportError:
            pyarrow = None
        self.native = _is_native_pandas_frame(obj)
        if not self.native and pyarrow is None:
            message = (
                "registering a pandas DataFrame with a pyarrow-backed column needs pyarrow, which converts it to Arrow"
            )
            raise TypeError(message)
        self.pyarrow = pyarrow

    def _named(self, columns: Sequence[int] | None) -> DataFrame:
        """The DataFrame as it is now, or the requested columns of it, under the text names; a view sharing its data."""
        names = _unique_names([str(label) for label in self.obj.columns])
        frame = self.obj if columns is None else self.obj.iloc[:, list(columns)]
        chosen = names if columns is None else [names[i] for i in columns]
        return frame.set_axis(chosen, axis=1)

    def describe(self) -> list[tuple[str, str]]:
        """Column names and engine types, for the native scan's bind."""
        frame = self._named(None)
        return [(name, _column_kind(frame[name])[1]) for name in frame.columns]

    def columns(self, columns: Sequence[int] | None) -> list[tuple[str, str, str, object, object | None]]:
        """For the requested columns (all when None): name, kind, engine type, data array, mask array or None."""
        frame = self._named(columns)
        return [(name, *_column_plan(frame[name])) for name in frame.columns]

    def _schema(self, frame: DataFrame) -> tuple[Schema, set[str]]:
        """The Arrow schema of a sample of the DataFrame `frame`, and the object columns read through `str()`."""
        sample = _row_sample(frame, self.SAMPLE_ROWS)
        forced = self._text_conversions(sample)
        if forced:
            sample = sample.assign(**{name: _stringified(sample[name]) for name in forced})
        return self.pyarrow.Schema.from_pandas(sample, preserve_index=False), forced

    def _text_conversions(self, sample: DataFrame) -> set[str]:
        """Object columns of `sample` the classifier reads as text from something other than strings alone.

        A sample that is empty or holds only strings is left for pyarrow's own inference, which already reads it
        as a string array; every other text-classified sample mixes values pyarrow cannot build one array from, so
        it is run through `str()` instead, on this sample and on every later slice of the same column.
        """
        forced = set()
        for name, dtype in sample.dtypes.items():
            if dtype.kind != "O":
                continue
            values = sample[name].tolist()
            kind, _ = _classify_sample(values)
            if kind == "text" and _needs_text_conversion(values):
                forced.add(str(name))
        return forced

    def _batches(self, frame: DataFrame, schema: Schema, forced: set[str]) -> Iterator[RecordBatch]:
        loose = [str(name) for name, dtype in frame.dtypes.items() if dtype.kind == "O" and name not in forced]
        for start in range(0, len(frame), self.SLICE_ROWS):
            part = frame.iloc[start : start + self.SLICE_ROWS]
            if forced:
                part = part.assign(**{name: _stringified(part[name]) for name in forced})
            converted = self.pyarrow.Table.from_pandas(part, schema=schema, preserve_index=False)
            for name in loose:
                self._check_exact(part, name, converted, schema, start)
            yield from converted.to_batches()

    def _check_exact(self, part: DataFrame, name: str, converted: Table, schema: Schema, start: int) -> None:
        """Refuses a slice whose Python objects the sampled type holds only approximately.

        Under a forced type pyarrow truncates a float into an integer column silently, and typed on its own a
        slice can lose as much, so the slice's own typing is cast to the sampled type with the cast that refuses
        loss, and the two readings must agree.
        """
        kind = schema.field(name).type
        message = f"column '{name}' holds a value after row {start} that its sampled type {kind} cannot hold exactly"
        natural = self.pyarrow.array(part[name], from_pandas=True)
        try:
            checked = natural.cast(kind)
        except self.pyarrow.ArrowException as error:
            detail = f"{message}: {error}"
            raise ValueError(detail) from None
        if not checked.equals(converted[name].combine_chunks()):
            raise ValueError(message)

    def __arrow_c_schema__(self) -> object:
        return self._schema(self._named(None))[0].__arrow_c_schema__()

    def rows(self) -> int | None:
        return len(self.obj)

    def stream(self, columns: Sequence[int] | None, filters: Sequence[Expr]) -> tuple[object, bool]:
        frame = self._named(columns)
        schema, forced = self._schema(frame)
        reader = self.pyarrow.RecordBatchReader.from_batches(schema, self._batches(frame, schema, forced))
        return reader.__arrow_c_stream__(), columns is not None


def _is_native_pandas_frame(obj: DataFrame) -> bool:
    """Whether every column of `obj` is numpy- or Python-object-backed rather than pyarrow-backed."""
    try:
        import pandas as pd
    except ImportError as error:
        message = "registering a pandas DataFrame needs pandas"
        raise TypeError(message) from error
    if not isinstance(obj, pd.DataFrame):
        message = f"registering as a pandas DataFrame needs a pandas DataFrame, not {type(obj).__name__}"
        raise TypeError(message)

    string_kind = getattr(pd.arrays, "ArrowStringArray", None)
    # A DataFrame's .values is the whole block as one array, not a per-column iterator like a dict's.
    for _, column in obj.items():  # noqa: PERF102
        array = column.array
        if isinstance(array, pd.arrays.ArrowExtensionArray):
            return False
        if string_kind is not None and isinstance(array, string_kind):
            return False
    return True


def _stride(length: int, sample_rows: int) -> slice:
    """Every step-th index over `length` items, the step chosen so at most `sample_rows` of them land in the stride."""
    step = max(1, length // sample_rows)
    return slice(None, None, step)


def _row_sample(frame: DataFrame, sample_rows: int) -> DataFrame:
    """Up to `sample_rows` rows of the DataFrame `frame`, spread evenly, which is how column types are inferred."""
    return frame.iloc[_stride(len(frame), sample_rows)].head(sample_rows)


def _object_sample(series: pd.Series, data: np.ndarray[Any, Any]) -> list[object]:
    """Up to `PandasSource.SAMPLE_ROWS` values spread over `data`, or the first valid one when none of those landed."""
    sample: list[object] = data[_stride(len(data), PandasSource.SAMPLE_ROWS)][: PandasSource.SAMPLE_ROWS].tolist()
    if sample and all(_missing(v) for v in sample):
        first_label = series.first_valid_index()
        if first_label is not None:
            value = series.loc[first_label]
            if isinstance(value, type(series)):
                # A duplicate label reads back several rows; the first is as good a sample as any.
                value = value.iloc[0]
            return [value]
    return sample


def _missing(value: object) -> bool:
    """Whether `value` is a null marker: None, a float NaN, or pandas' NA/NaT singletons.

    Never calls pandas' own `isna`, which turns a list or dict argument into an array rather than a bool.
    """
    if value is None:
        return True
    if isinstance(value, float):
        return value != value
    kind = type(value)
    return kind.__module__.startswith("pandas") and kind.__name__ in ("NAType", "NaTType")


def _stringified(series: pd.Series) -> pd.Series:
    """`series` with every non-null value replaced by its text, so pyarrow can hold the column as VARCHAR."""
    return series.map(lambda value: value if _missing(value) else str(value))


def _sql_quote(text: str) -> str:
    """`text` as a single-quoted SQL string literal."""
    return "'" + text.replace("'", "''") + "'"


def _column_kind(series: pd.Series) -> tuple[str, str]:
    """The kind and engine type text `describe()` reports for `series`, without materializing its data."""
    import numpy as np
    import pandas as pd

    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        if all(isinstance(c, str) for c in dtype.categories):
            return "enum", _enum_type_text(dtype.categories)
        return _classify_sample(_used_categories(series))
    if isinstance(dtype, pd.DatetimeTZDtype):
        return f"timestamp:{dtype.unit}", "TIMESTAMP WITH TIME ZONE"
    array = series.array
    if hasattr(array, "_data") and hasattr(array, "_mask"):
        return "fixed", _fixed_type_text(array._data.dtype)
    if isinstance(dtype, np.dtype):
        if dtype.kind == "M":
            return "timestamp", _NAIVE_TIMESTAMP_TYPES[np.datetime_data(dtype)[0]]
        if dtype.kind == "m":
            return f"interval:{np.datetime_data(dtype)[0]}", "INTERVAL"
        if dtype.kind in "biuf":
            return "fixed", _fixed_type_text(dtype)
    return _object_kind(series)


def _column_plan(series: pd.Series) -> ColumnPlan:
    """kind, engine type text, data array, mask array or None: the native reading of one pandas column."""
    import numpy as np
    import pandas as pd

    dtype = series.dtype
    array = series.array
    if isinstance(dtype, pd.CategoricalDtype):
        categories = dtype.categories
        if all(isinstance(c, str) for c in categories):
            return _enum_plan(series, categories)
        kind, type_text = _classify_sample(_used_categories(series))
        return ColumnPlan(kind, type_text, _object_data(series.astype(object)), None)
    if isinstance(dtype, pd.DatetimeTZDtype):
        return _aware_timestamp_plan(series, dtype)
    if hasattr(array, "_data") and hasattr(array, "_mask"):
        return _masked_plan(array)
    if isinstance(dtype, np.dtype):
        if dtype.kind == "M":
            return _naive_timestamp_plan(series, dtype)
        if dtype.kind == "m":
            return _interval_plan(series, dtype)
        if dtype.kind in "biu":
            return ColumnPlan("fixed", _fixed_type_text(dtype), series.to_numpy(copy=False), None)
        if dtype.kind == "f":
            return _float_plan(series, dtype)
    return _object_plan(series)


def _fixed_type_text(dtype: np.dtype[Any]) -> str:
    """The engine type of a fixed-width numpy dtype: boolean, a half-, single- or double-width float, int or uint."""
    if dtype.kind == "b":
        return "BOOLEAN"
    if dtype.kind == "f" and dtype.itemsize == 2:
        # A half-precision column is widened to FLOAT on read; float16 has no native engine type.
        return "FLOAT"
    return _FIXED_TYPES[dtype.str[1:]]


def _float_plan(series: pd.Series, dtype: np.dtype[Any]) -> ColumnPlan:
    import numpy as np

    data = series.to_numpy(copy=False)
    if dtype.itemsize == 2:
        data = data.astype(np.float32)
    return ColumnPlan("fixed", _fixed_type_text(dtype), data, None)


def _masked_plan(array: object) -> ColumnPlan:
    data = getattr(array, "_data")  # noqa: B009
    mask = getattr(array, "_mask")  # noqa: B009
    return ColumnPlan("fixed", _fixed_type_text(data.dtype), data, mask)


def _naive_timestamp_plan(series: pd.Series, dtype: np.dtype[Any]) -> ColumnPlan:
    import numpy as np

    unit = np.datetime_data(dtype)[0]
    array = series.array
    raw = array._ndarray if hasattr(array, "_ndarray") else series.to_numpy(copy=False)
    return ColumnPlan("timestamp", _NAIVE_TIMESTAMP_TYPES[unit], raw.view(np.int64), None)


def _aware_timestamp_plan(series: pd.Series, dtype: pd.DatetimeTZDtype) -> ColumnPlan:
    import numpy as np

    array = series.array
    if str(array.tz) != "UTC":
        array = array.tz_convert("UTC")
    return ColumnPlan(f"timestamp:{dtype.unit}", "TIMESTAMP WITH TIME ZONE", array._ndarray.view(np.int64), None)


def _interval_plan(series: pd.Series, dtype: np.dtype[Any]) -> ColumnPlan:
    import numpy as np

    unit = np.datetime_data(dtype)[0]
    array = series.array
    raw = array._ndarray if hasattr(array, "_ndarray") else series.to_numpy(copy=False)
    return ColumnPlan(f"interval:{unit}", "INTERVAL", raw.view(np.int64), None)


def _enum_type_text(categories: Sequence[str]) -> str:
    return "ENUM(" + ", ".join(_sql_quote(str(c)) for c in categories) + ")"


def _enum_plan(series: pd.Series, categories: Sequence[str]) -> ColumnPlan:
    codes = series.cat.codes.to_numpy()
    return ColumnPlan("enum", _enum_type_text(categories), codes, None)


def _object_kind(series: pd.Series) -> tuple[str, str]:
    kind, type_text, _ = _object_kind_and_data(series)
    return kind, type_text


def _object_plan(series: pd.Series) -> ColumnPlan:
    kind, type_text, data = _object_kind_and_data(series)
    return ColumnPlan(kind, type_text, data, None)


def _object_kind_and_data(series: pd.Series) -> tuple[str, str, np.ndarray[Any, Any]]:
    data = _object_data(series)
    kind, type_text = _classify_sample(_object_sample(series, data))
    return kind, type_text, data


def _used_categories(series: pd.Series) -> list[object]:
    """The distinct values a categorical series holds, which type it exactly with no row sample and no boxing.

    Filtering leaves a category declared but unused, and one of another type than the values would otherwise turn
    the column into text, so the declared categories are not enough.
    """
    used: list[object] = series.cat.remove_unused_categories().cat.categories.tolist()
    return used


def _object_data(series: pd.Series) -> np.ndarray[Any, Any]:
    """The object-dtype series `series` as a numpy array, its own buffer where the series already has one."""
    array = series.array
    data: np.ndarray[Any, Any] = array._ndarray if hasattr(array, "_ndarray") else series.to_numpy(copy=False)
    return data


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
    if isinstance(value, datetime.datetime):
        return "aware" if value.tzinfo is not None else "naive"
    if isinstance(value, datetime.date):
        return "date"
    for kind, family in _FAMILY_TYPES:
        if isinstance(value, kind):
            return family
    return None


def _significant_values(sample: list[object]) -> list[object]:
    """`sample` with missing markers dropped and numpy scalars unwrapped through `.item()`."""
    import numpy as np

    def unwrap(value: object) -> object:
        return value.item() if isinstance(value, np.generic) else value

    return [unwrap(v) for v in sample if not _missing(v)]


def _needs_text_conversion(sample: list[object]) -> bool:
    """Whether an object column's sample must be run through `str()` before pyarrow can type it as VARCHAR.

    False for an empty sample or one that already holds only strings: pyarrow reads either as a string array on
    its own. True for anything else a text classification covers, since pyarrow cannot build one array from it.
    """
    values = _significant_values(sample)
    return bool(values) and not all(isinstance(v, str) for v in values)


def _classify_sample(sample: list[object]) -> tuple[str, str]:
    """Kind and engine type text for an object column, from a sample of its non-null values.

    `"text"` (VARCHAR) when the sample is empty, holds only strings, or does not unify under one family below;
    `"objects"` otherwise, with each value later read through `PythonToValue` and cast to the chosen type.
    """
    values = _significant_values(sample)
    if not values or all(isinstance(v, str) for v in values):
        return "text", "VARCHAR"

    temporal: set[str] = set()
    other: set[str] = set()
    has_float = False
    max_abs_int = 0
    max_scale = 0
    for value in values:
        family = _value_family(value)
        if family is None:
            return "text", "VARCHAR"
        if family in ("date", "naive", "aware"):
            temporal.add(family)
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
