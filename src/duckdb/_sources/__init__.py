"""The Python objects that register as tables, each behind a source that exports Arrow for a scan.

There are two scans: the Arrow scan reads an Arrow C stream from the Arrow, pyarrow and polars sources in
`arrow.py`, `pyarrow.py` and `polars.py`, and the pandas scan reads a pandas DataFrame natively off its own buffers
through the source in `pandas.py`, which hands a DataFrame with a pyarrow-backed column to the Arrow scan instead.

A source exposes `__arrow_c_schema__`, when the object can say its schema without producing data, `accepts`, which
says whether a scan will apply a predicate itself, `stream(columns, filters)`, which exports an Arrow stream
capsule holding either exactly the requested columns, in that order, or every column; it says which, and
`pull_under_gil`, which says whether the scan must hold the GIL while pulling from that stream. Nothing here
imports pyarrow, polars or pandas at module level: a family is recognised by its class's module and name, or by the
methods it carries, and its library is imported only inside the source that needs it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .. import _duckdb

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .._expressions import Expr

#: The capsule names the Arrow PyCapsule interface reserves, by which a bare capsule says what it holds.
STREAM_CAPSULE = "arrow_array_stream"
ARROW_CAPSULES = (STREAM_CAPSULE, "arrow_schema", "arrow_array")


class Source:
    """What the scan talks to. `one_shot` sources are read once; every other source is read as often as asked."""

    one_shot = False
    #: Whether the exported stream needs the scan to hold the GIL while it pulls the next array. A source whose
    #: stream is pure C, C++ or Rust, or takes the GIL itself where it runs Python, sets this False, and its pulls
    #: then run on engine threads with no GIL held.
    pull_under_gil = True
    #: Whether the registered name resolves to the native pandas scan rather than the Arrow scan. Only a pandas
    #: frame whose columns are all numpy- or Python-object-backed sets this True.
    native = False

    def __init__(self, obj: object) -> None:
        self.obj: Any = obj

    def rows(self) -> int | None:
        """How many rows a scan will produce, when the object knows without reading itself; None otherwise."""
        return None

    def accepts(self, predicate: Expr) -> bool:
        """Whether every scan will apply `predicate`, a `duckdb._expressions` tree over the object's columns, itself.

        The engine offers each predicate a query applies to the rows, and stops applying one that is accepted, so
        True is a promise: the stream must then hold only rows satisfying it, with the engine's meaning of the
        comparison. Anything else answers False and the engine filters as usual. A column is named as the Arrow
        schema names it, so the one column of a nameless array, `value` in SQL, is named by the empty string.
        """
        return False

    def stream(self, columns: Sequence[int] | None, filters: Sequence[Expr]) -> tuple[object, bool]:
        """An Arrow stream capsule over the object and whether it holds only `columns`, in that order.

        `None` asks for every column. A source that cannot narrow the stream returns every column and False, in
        the order its schema declared them: the scan picks columns by position, so that order must not change
        between calls. `filters` holds the predicates `accepts` promised, to apply over every column before the
        stream is narrowed, since a predicate may name a column the query does not read. In place of a stream
        capsule, the pair of schema and array capsules that `__arrow_c_array__` returns stands for one batch. A
        stream or array whose top level is not a struct is read as one column.
        """
        return self.obj.__arrow_c_stream__(), False


def _from(obj: object, library: str) -> bool:
    module = type(obj).__module__
    return module == library or module.startswith(library + ".")


def _is(obj: object, library: str, *names: str) -> bool:
    return type(obj).__name__ in names and _from(obj, library)


def _unique_names(labels: list[str]) -> list[str]:
    """`labels` with a repeat renamed as the engine renames a repeated Arrow field: the first free numbered suffix."""
    taken: set[str] = set()
    names = []
    for label in labels:
        name = label
        suffix = 0
        while name.lower() in taken:
            suffix += 1
            name = f"{label}_{suffix}"
        taken.add(name.lower())
        names.append(name)
    return names


def adapt(obj: object) -> Source:
    """The source to register for `obj`.

    An Arrow stream capsule and a pyarrow RecordBatchReader are read once. A pyarrow Table or RecordBatch, a polars
    DataFrame or LazyFrame, a pandas DataFrame, a pyarrow Dataset or Scanner, and anything else exporting
    `__arrow_c_stream__` or `__arrow_c_array__`, are read as often as asked. A `Source` of one's own is registered
    as it is.
    """
    from .arrow import ArrowArraySource, ArrowCapsuleSource, ArrowStreamSource
    from .pandas import PandasSource
    from .polars import LazyFrameSource, PolarsFrameSource
    from .pyarrow import PyArrowDatasetSource, PyArrowReaderSource, PyArrowScannerSource, PyArrowTableSource

    if isinstance(obj, Source):
        return obj
    capsule = _duckdb.capsule_name(obj)
    if capsule == STREAM_CAPSULE:
        return ArrowCapsuleSource(obj)
    if capsule in ARROW_CAPSULES:
        message = f"a bare '{capsule}' capsule is not an Arrow stream; register the object it came from instead"
        raise TypeError(message)
    exports = hasattr(obj, "__arrow_c_stream__")
    if exports and any(cls.__name__ == "RecordBatchReader" for cls in type(obj).__mro__):
        return PyArrowReaderSource(obj)
    if _is(obj, "polars", "LazyFrame"):
        return LazyFrameSource(obj)
    if _is(obj, "pandas", "DataFrame"):
        return PandasSource(obj)
    if exports and _is(obj, "pyarrow", "Table", "RecordBatch"):
        return PyArrowTableSource(obj)
    if exports and _is(obj, "polars", "DataFrame"):
        return PolarsFrameSource(obj)
    if exports:
        return ArrowStreamSource(obj)
    if hasattr(obj, "__arrow_c_array__"):
        return ArrowArraySource(obj)
    if hasattr(obj, "to_reader") and hasattr(obj, "projected_schema"):
        return PyArrowScannerSource(obj)
    if hasattr(obj, "scanner") and hasattr(obj, "schema"):
        return PyArrowDatasetSource(obj)
    message = (
        f"a registered object must export Arrow through __arrow_c_stream__ or __arrow_c_array__, be an Arrow stream "
        f"capsule, a pyarrow Dataset or Scanner, a polars LazyFrame or a pandas DataFrame; {type(obj).__name__} is "
        f"none of these"
    )
    raise TypeError(message)
