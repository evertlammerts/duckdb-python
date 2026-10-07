"""The pandas source: a DataFrame read by the numpy scan, column by column, off the DataFrame's own data.

`describe()` names its columns and their engine types for `numpy_scan.cpp`'s bind, and `columns()` answers each
requested column as an encoding, an engine type, a data array and a mask array or None, straight off the column's
own buffers. A pyarrow-backed column (an `ArrowExtensionArray`, which includes `ArrowStringArray`, the default for
strings when pyarrow is installed) is answered as its Arrow data instead, whose engine type the engine's Arrow import
decides. An object column's engine type is judged from a sample of its values spread evenly over the column,
classified in `numpy.py`. Each column is read by its own dtype, whatever the columns beside it.

The index is left out, as the previous client did. Column labels become their text, and a label repeating another
gets a numbered suffix, as the engine names a repeated Arrow field.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import NumpyScanSource, _unique_names
from .numpy import ColumnReading, ScanColumn, _array_reading, _classify_sample

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    import pandas as pd
    from pandas import DataFrame


class PandasSource(NumpyScanSource):
    """A pandas DataFrame read by the numpy scan, off the DataFrame's own data."""

    def __init__(self, obj: object) -> None:
        super().__init__(obj)
        try:
            import pandas as pd
        except ImportError as error:
            message = "registering a pandas DataFrame needs pandas"
            raise TypeError(message) from error
        if not isinstance(obj, pd.DataFrame):
            message = f"registering as a pandas DataFrame needs a pandas DataFrame, not {type(obj).__name__}"
            raise TypeError(message)

    def rows(self) -> int:
        return len(self.obj)

    def describe(self) -> list[tuple[str, object]]:
        frame = _named(self.obj, None)
        return [(name, _column_reading(name, frame[name]).engine_type) for name in frame.columns]

    def columns(self, columns: Sequence[int] | None) -> list[tuple[str, str, str | None, object, object | None]]:
        frame = _named(self.obj, columns)
        return [(name, *_column_reading(name, frame[name]).prepare()) for name in frame.columns]


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


def _sql_quote(text: str) -> str:
    """`text` as a single-quoted SQL string literal."""
    return "'" + text.replace("'", "''") + "'"


def _column_reading(name: str, series: pd.Series) -> ColumnReading:
    """How the numpy scan reads one pandas column.

    A pyarrow-backed column is read as its Arrow data. pandas' categorical and time zone dtypes and its nullable
    arrays, which pair a data array with a mask, are read here; every other column is read as the numpy array holding
    its values, as the numpy source reads an array. A period has no engine type and is refused.
    """
    import pandas as pd

    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        categories = dtype.categories
        # An ENUM needs at least one label.
        if len(categories) and all(isinstance(c, str) for c in categories):
            type_text = _enum_type_text(categories)
            return ColumnReading(type_text, lambda: ScanColumn("enum", type_text, series.cat.codes.to_numpy(), None))
        return _categorical_reading(name, series)
    if isinstance(dtype, pd.PeriodDtype):
        message = f"column '{name}' has the pandas dtype {dtype}, and no engine type holds a period"
        raise TypeError(message)
    if isinstance(dtype, pd.DatetimeTZDtype):
        type_text = "TIMESTAMPTZ_NS" if dtype.unit == "ns" else "TIMESTAMP WITH TIME ZONE"
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


def _categorical_reading(name: str, series: pd.Series) -> ColumnReading:
    """A categorical series whose categories are not all text, typed from its categories and decoded only when read.

    Categories of one dtype give that dtype's engine type. Object categories are classified as a whole, over every
    category a row uses, so no row sample decides the type and a category declared but unused does not count.
    """
    import pandas as pd

    categories = series.cat.categories
    if categories.dtype == object:
        encoding, type_text = _classify_sample(_used_categories(series))
        return ColumnReading(
            type_text, lambda: ScanColumn(encoding, type_text, _backing_array(series.astype(object)), None)
        )
    engine_type = _column_reading(name, pd.Series(categories[:0])).engine_type
    return ColumnReading(engine_type, lambda: _decoded_reading(name, series).prepare())


def _used_categories(series: pd.Series) -> list[object]:
    """The distinct values a categorical series holds, which type it exactly with no row sample and no boxing."""
    used: list[object] = series.cat.remove_unused_categories().cat.categories.tolist()
    return used


def _decoded_reading(name: str, series: pd.Series) -> ColumnReading:
    """A categorical series decoded into a column of its categories' own dtype, each row the category its code names.

    Integer and boolean categories take a missing code as their first category, or zero when there is none, marked
    missing by a mask, since filling in pandas' own missing value would widen them to float or object.
    """
    import numpy as np
    import pandas as pd

    categories = series.cat.categories
    codes = series.cat.codes.to_numpy()
    if isinstance(categories.dtype, np.dtype) and categories.dtype.kind in "iub":
        missing = codes < 0
        if len(categories):
            values = categories.to_numpy().take(np.where(missing, 0, codes))
        else:
            values = np.zeros(len(codes), dtype=categories.dtype)
        return _array_reading(name, values, missing if missing.any() else None)
    return _column_reading(name, pd.Series(categories.array.take(codes, allow_fill=True)))


def _backing_array(series: pd.Series) -> np.ndarray[Any, Any]:
    """The numpy array holding the values of `series`, its own buffer where the series already has one."""
    array = series.array
    data: np.ndarray[Any, Any] = array._ndarray if hasattr(array, "_ndarray") else series.to_numpy(copy=False)
    return data
