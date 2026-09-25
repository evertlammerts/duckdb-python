"""The pandas source: read natively off its own buffers, or through pyarrow a slice at a time.

A pandas DataFrame whose columns are all numpy- or Python-object-backed is read natively: `describe()` names its
columns and their engine types for `numpy_scan.cpp`'s bind, and `columns()` answers each requested column as a
kind, an engine type, a data array and a mask array or None, straight off the DataFrame's own buffers. A DataFrame
with a pyarrow-backed column (an `ArrowExtensionArray`, which includes `ArrowStringArray`, the default for strings
when pyarrow is installed) is read through pyarrow instead, converted a slice of rows at a time so the DataFrame is
never copied whole. Either way, an object column's engine type is judged from a sample of its values spread evenly
over the column, classified in `numpy.py`: the sample unifies under one family there for a typed reading, or falls
back to text.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import Source, _unique_names
from .numpy import (
    _NAIVE_TIMESTAMP_TYPES,
    SAMPLE_ROWS,
    ColumnPlan,
    _classify_sample,
    _fixed_type_text,
    _float_plan,
    _interval_plan,
    _missing,
    _naive_timestamp_plan,
    _object_kind,
    _object_plan,
    _significant_values,
    _stride,
    contiguous,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import numpy as np
    import pandas as pd
    from pandas import DataFrame
    from pyarrow import RecordBatch, Schema, Table

    from .._expressions import Expr


class PandasSource(Source):
    """A pandas DataFrame, read natively when every column is numpy- or Python-object-backed, through pyarrow otherwise.

    A pyarrow-backed column (an `ArrowExtensionArray`, which includes `ArrowStringArray`, the default for strings
    when pyarrow is installed) sends the whole DataFrame through the Arrow path below, converted a slice of rows at a
    time so the DataFrame is never copied whole; `native` is False for such a DataFrame. Otherwise `describe()` and
    `columns()` serve `numpy_scan.cpp`'s table function directly off the DataFrame's own buffers.

    The index is left out, as the previous client did. Column labels become their text, and a label repeating
    another gets a numbered suffix, as the engine names a repeated Arrow field.
    """

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
        return [(name, *contiguous(_column_plan(frame[name]))) for name in frame.columns]

    def _schema(self, frame: DataFrame) -> tuple[Schema, set[str]]:
        """The Arrow schema of a sample of the DataFrame `frame`, and the object columns read through `str()`."""
        sample = _row_sample(frame, SAMPLE_ROWS)
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
            if loose:
                part = part.assign(**{name: _nulled(part[name]) for name in loose})
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


def _row_sample(frame: DataFrame, sample_rows: int) -> DataFrame:
    """Up to `sample_rows` rows of the DataFrame `frame`, spread evenly, which is how column types are inferred."""
    return frame.iloc[_stride(len(frame), sample_rows)].head(sample_rows)


def _stringified(series: pd.Series) -> pd.Series:
    """`series` with every non-null value replaced by its text, so pyarrow can hold the column as VARCHAR."""
    return series.map(lambda value: None if _missing(value) else str(value))


def _nulled(series: pd.Series) -> pd.Series:
    """An object `series` with every missing marker as None, since pyarrow keeps a numpy float32 NaN as a NaN."""
    missing = series.isna()
    return series.where(~missing, None) if missing.any() else series


def _needs_text_conversion(sample: list[object]) -> bool:
    """Whether an object column's sample must be run through `str()` before pyarrow can type it as VARCHAR.

    False for an empty sample or one that already holds only strings: pyarrow reads either as a string array on
    its own. True for anything else a text classification covers, since pyarrow cannot build one array from it.
    """
    values = _significant_values(sample)
    return bool(values) and not all(isinstance(v, str) for v in values)


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
    return _object_kind(_backing_array(series), _pandas_first_valid_position)


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
        return ColumnPlan(kind, type_text, _backing_array(series.astype(object)), None)
    if isinstance(dtype, pd.DatetimeTZDtype):
        return _aware_timestamp_plan(series, dtype)
    if hasattr(array, "_data") and hasattr(array, "_mask"):
        return _masked_plan(array)
    if isinstance(dtype, np.dtype):
        if dtype.kind == "M":
            raw = _backing_array(series)
            return _naive_timestamp_plan(raw, dtype)
        if dtype.kind == "m":
            raw = _backing_array(series)
            return _interval_plan(raw, dtype)
        if dtype.kind in "biu":
            return ColumnPlan("fixed", _fixed_type_text(dtype), series.to_numpy(copy=False), None)
        if dtype.kind == "f":
            return _float_plan(series.to_numpy(copy=False), dtype)
    return _object_plan(_backing_array(series), _pandas_first_valid_position)


def _masked_plan(array: object) -> ColumnPlan:
    data = getattr(array, "_data")  # noqa: B009
    mask = getattr(array, "_mask")  # noqa: B009
    return ColumnPlan("fixed", _fixed_type_text(data.dtype), data, mask)


def _aware_timestamp_plan(series: pd.Series, dtype: pd.DatetimeTZDtype) -> ColumnPlan:
    import numpy as np

    array = series.array
    if str(array.tz) != "UTC":
        array = array.tz_convert("UTC")
    return ColumnPlan(f"timestamp:{dtype.unit}", "TIMESTAMP WITH TIME ZONE", array._ndarray.view(np.int64), None)


def _enum_type_text(categories: Sequence[str]) -> str:
    return "ENUM(" + ", ".join(_sql_quote(str(c)) for c in categories) + ")"


def _enum_plan(series: pd.Series, categories: Sequence[str]) -> ColumnPlan:
    codes = series.cat.codes.to_numpy()
    return ColumnPlan("enum", _enum_type_text(categories), codes, None)


def _pandas_first_valid_position(data: np.ndarray[Any, Any]) -> int | None:
    """The first position of an object array taken from a pandas column that pandas does not count as missing.

    pandas' own missing test runs over the whole array in C, where a loop over the values in Python costs several
    times as much on a column that holds nothing but missing values.
    """
    import pandas as pd

    valid = pd.notna(data)
    return int(valid.argmax()) if valid.any() else None


def _used_categories(series: pd.Series) -> list[object]:
    """The distinct values a categorical series holds, which type it exactly with no row sample and no boxing.

    Filtering leaves a category declared but unused, and one of another type than the values would otherwise turn
    the column into text, so the declared categories are not enough.
    """
    used: list[object] = series.cat.remove_unused_categories().cat.categories.tolist()
    return used


def _backing_array(series: pd.Series) -> np.ndarray[Any, Any]:
    """The numpy array holding the values of `series`, its own buffer where the series already has one."""
    array = series.array
    data: np.ndarray[Any, Any] = array._ndarray if hasattr(array, "_ndarray") else series.to_numpy(copy=False)
    return data
