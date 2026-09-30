"""The pandas sources: a DataFrame read off its own buffers, or through pyarrow a slice at a time.

`adapt` picks one at registration. A DataFrame whose columns are all numpy- or Python-object-backed is a
`PandasSource`: `describe()` names its columns and their engine types for `numpy_scan.cpp`'s bind, and `columns()`
answers each requested column as an encoding, an engine type, a data array and a mask array or None, straight off
the DataFrame's own buffers. A DataFrame with a pyarrow-backed column (an `ArrowExtensionArray`, which includes
`ArrowStringArray`, the default for strings when pyarrow is installed) is a `PandasArrowSource`, converted to Arrow a
slice of rows at a time so the DataFrame is never copied whole. Either way, an object column's engine type is judged
from the same sample of its values spread evenly over the column, classified in `numpy.py`: the sample unifies under
one family there for a typed reading, or falls back to text, which both sources then apply to every value alike.

Both leave the index out, as the previous client did. Column labels become their text, and a label repeating another
gets a numbered suffix, as the engine names a repeated Arrow field.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any

from . import ArrowScanSource, NumpyScanSource, _unique_names
from .numpy import (
    SAMPLE_ROWS,
    ColumnReading,
    ScanColumn,
    _array_reading,
    _classify_sample,
    _missing,
    _object_sample,
    _significant_values,
    _stride,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import numpy as np
    import pandas as pd
    from pandas import DataFrame
    from pyarrow import RecordBatch, Schema, Table

    from .._expressions import Expr


class PandasSource(NumpyScanSource):
    """A pandas DataFrame read by the numpy scan, off the DataFrame's own buffers."""

    def rows(self) -> int:
        return len(self.obj)

    def describe(self) -> list[tuple[str, object]]:
        frame = _named(self.obj, None)
        return [(name, _column_reading(name, frame[name]).engine_type) for name in frame.columns]

    def columns(self, columns: Sequence[int] | None) -> list[tuple[str, str, str | None, object, object | None]]:
        frame = _named(self.obj, columns)
        return [(name, *_column_reading(name, frame[name]).prepare()) for name in frame.columns]


class PandasArrowSource(ArrowScanSource):
    """A pandas DataFrame read by the Arrow scan, converted through pyarrow a slice of rows at a time."""

    #: Rows converted per slice; the memory held at any time is one slice's worth of Arrow.
    SLICE_ROWS = 1 << 16

    def __init__(self, obj: object) -> None:
        super().__init__(obj)
        try:
            self.pyarrow  # noqa: B018
        except ImportError as error:
            message = (
                "registering a pandas DataFrame with a pyarrow-backed column needs pyarrow, which converts it to Arrow"
            )
            raise TypeError(message) from error

    @cached_property
    def pyarrow(self) -> Any:  # noqa: ANN401
        """pyarrow, imported only once a DataFrame is read through it."""
        import pyarrow

        return pyarrow

    def _schema(self, frame: DataFrame) -> tuple[Schema, dict[str, str]]:
        """The Arrow schema of the DataFrame `frame`, and which columns convert as `"text"` or checked `"exact"`.

        A column of Python objects is classified from the numpy scan's own sample, so both transports read it alike.
        """
        import numpy as np
        import pandas as pd

        pa = self.pyarrow
        sample = _row_sample(frame, SAMPLE_ROWS)
        rules: dict[str, str] = {}
        fields: dict[str, object] = {}
        for name, dtype in frame.dtypes.items():
            label = str(name)
            if isinstance(dtype, np.dtype) and dtype.kind == "O":
                values = _object_sample(_backing_array(frame[label]), _pandas_first_valid_position)
                if _classify_sample(values)[0] == "text":
                    rules[label], fields[label] = "text", pa.string()
                else:
                    typed = [v for v in values if not _missing(v)]
                    rules[label], fields[label] = "exact", pa.array(typed, from_pandas=True).type
            elif isinstance(dtype, pd.CategoricalDtype):
                values = sample[label].tolist()
                if _classify_sample(values)[0] == "text" and _needs_text_conversion(values):
                    rules[label], fields[label] = "text", pa.string()
                else:
                    rules[label] = "exact"
        inferred = pa.Schema.from_pandas(sample.drop(columns=list(fields)), preserve_index=False)
        names = [str(name) for name in frame.columns]
        schema = pa.schema([pa.field(n, fields[n]) if n in fields else inferred.field(n) for n in names])
        return schema, rules

    def _batches(self, frame: DataFrame, schema: Schema, rules: dict[str, str]) -> Iterator[RecordBatch]:
        text = [name for name, rule in rules.items() if rule == "text"]
        exact = [name for name, rule in rules.items() if rule == "exact"]
        for start in range(0, len(frame), self.SLICE_ROWS):
            part = frame.iloc[start : start + self.SLICE_ROWS]
            if rules:
                converted_columns = {name: _as_text(part[name]) for name in text}
                converted_columns.update({name: _nulled(part[name]) for name in exact})
                part = part.assign(**converted_columns)
            converted = self.pyarrow.Table.from_pandas(part, schema=schema, preserve_index=False)
            for name in exact:
                self._check_exact(part, name, converted, schema, start)
            yield from converted.to_batches()

    def _check_exact(self, part: DataFrame, name: str, converted: Table, schema: Schema, start: int) -> None:
        """Refuses a slice whose Python objects the sampled type holds only approximately.

        Under a forced type pyarrow truncates a float into an integer column silently, and typed on its own a
        slice can lose as much, so the slice's own typing is cast to the sampled type with the cast that refuses
        loss, and the two readings must agree.
        """
        arrow_type = schema.field(name).type
        message = (
            f"column '{name}' holds a value after row {start} that its sampled type {arrow_type} cannot hold exactly"
        )
        natural = self.pyarrow.array(part[name], from_pandas=True)
        try:
            checked = natural.cast(arrow_type)
        except self.pyarrow.ArrowException as error:
            detail = f"{message}: {error}"
            raise ValueError(detail) from None
        if not checked.equals(converted[name].combine_chunks()):
            raise ValueError(message)

    def __arrow_c_schema__(self) -> object:
        return self._schema(_named(self.obj, None))[0].__arrow_c_schema__()

    def rows(self) -> int | None:
        return len(self.obj)

    def stream(self, columns: Sequence[int] | None, filters: Sequence[Expr]) -> tuple[object, bool]:
        frame = _named(self.obj, columns)
        schema, rules = self._schema(frame)
        reader = self.pyarrow.RecordBatchReader.from_batches(schema, self._batches(frame, schema, rules))
        return reader.__arrow_c_stream__(), columns is not None


def has_pyarrow_columns(obj: DataFrame) -> bool:
    """Whether a column of `obj` is pyarrow-backed rather than numpy- or Python-object-backed."""
    try:
        import pandas as pd
    except ImportError as error:
        message = "registering a pandas DataFrame needs pandas"
        raise TypeError(message) from error
    if not isinstance(obj, pd.DataFrame):
        message = f"registering as a pandas DataFrame needs a pandas DataFrame, not {type(obj).__name__}"
        raise TypeError(message)

    return any(_held_as_arrow(column.array) for _, column in obj.items())


def _held_as_arrow(array: object) -> bool:
    """Whether a pandas column's array holds its values as pyarrow data."""
    import pandas as pd

    string_array = getattr(pd.arrays, "ArrowStringArray", None)
    return isinstance(array, pd.arrays.ArrowExtensionArray) or (
        string_array is not None and isinstance(array, string_array)
    )


def _named(frame: DataFrame, columns: Sequence[int] | None) -> DataFrame:
    """`frame` as it is now, or the requested columns of it, under the text names; a view sharing its data."""
    names = _unique_names([str(label) for label in frame.columns])
    selected = frame if columns is None else frame.iloc[:, list(columns)]
    chosen = names if columns is None else [names[i] for i in columns]
    return selected.set_axis(chosen, axis=1)


def _row_sample(frame: DataFrame, sample_rows: int) -> DataFrame:
    """Up to `sample_rows` rows of the DataFrame `frame`, spread evenly, which is how column types are inferred."""
    return frame.iloc[_stride(len(frame), sample_rows)].head(sample_rows)


def _stringified(series: pd.Series) -> pd.Series:
    """`series` with every non-null value replaced by its text, so pyarrow can hold the column as VARCHAR."""
    return series.map(lambda value: None if _missing(value) else value if isinstance(value, str) else str(value))


def _as_text(series: pd.Series) -> pd.Series:
    """`series` under the numpy scan's text rule: strings as they are, missing markers None, anything else `str()`."""
    from pandas.api.types import infer_dtype

    if infer_dtype(series, skipna=True) in ("string", "empty"):
        # Strings and missing markers only; the markers become None first, since pyarrow refuses a float32 NaN.
        return _nulled(series)
    return _stringified(series)


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


def _column_reading(name: str, series: pd.Series) -> ColumnReading:
    """How the numpy scan reads one pandas column.

    A pyarrow-backed column is read as its Arrow data. pandas' categorical and time zone dtypes and its nullable
    arrays, which pair a data array with a mask, are read here; every other column is read as the numpy array holding
    its values, as the numpy source reads an array.
    """
    import pandas as pd

    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        categories = dtype.categories
        if all(isinstance(c, str) for c in categories):
            type_text = _enum_type_text(categories)
            return ColumnReading(type_text, lambda: ScanColumn("enum", type_text, series.cat.codes.to_numpy(), None))
        encoding, type_text = _classify_sample(_used_categories(series))
        return ColumnReading(
            type_text, lambda: ScanColumn(encoding, type_text, _backing_array(series.astype(object)), None)
        )
    if isinstance(dtype, pd.DatetimeTZDtype):
        type_text = "TIMESTAMP WITH TIME ZONE"
        return ColumnReading(
            type_text, lambda: ScanColumn(f"timestamp:{dtype.unit}", type_text, _utc_counts(series), None)
        )
    array = series.array
    if _held_as_arrow(array):
        chunked = array.__arrow_array__()
        return ColumnReading(
            chunked.type.__arrow_c_schema__(), lambda: ScanColumn("arrow", None, chunked.__arrow_c_stream__(), None)
        )
    if hasattr(array, "_data") and hasattr(array, "_mask"):
        return _array_reading(name, array._data, array._mask)
    return _array_reading(name, _backing_array(series), None, _pandas_first_valid_position)


def _utc_counts(series: pd.Series) -> np.ndarray[Any, Any]:
    """A time zone aware series as int64 counts since the epoch in UTC, in the series' own unit."""
    import numpy as np

    array = series.array
    if str(array.tz) != "UTC":
        array = array.tz_convert("UTC")
    counts: np.ndarray[Any, Any] = array._ndarray.view(np.int64)
    return counts


def _enum_type_text(categories: Sequence[str]) -> str:
    return "ENUM(" + ", ".join(_sql_quote(str(c)) for c in categories) + ")"


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
