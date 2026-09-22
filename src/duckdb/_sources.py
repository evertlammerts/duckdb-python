"""The Python objects that register as tables, each behind a source that exports Arrow for the scan.

A source exposes `__arrow_c_schema__`, when the object can say its schema without producing data, `accepts`, which
says whether a scan will apply a predicate itself, and `stream(columns, filters)`, which exports an Arrow stream
capsule holding either exactly the requested columns, in that order, or every column; it says which. Nothing here
imports pyarrow, polars or pandas at module level: a family is recognised by its class's module and name, or by the
methods it carries, and its library is imported only inside the source that needs it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import _duckdb

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from pandas import DataFrame
    from pyarrow import RecordBatch, Schema

    from ._expressions import Expr

#: The capsule names the Arrow PyCapsule interface reserves, by which a bare capsule says what it holds.
STREAM_CAPSULE = "arrow_array_stream"
ARROW_CAPSULES = (STREAM_CAPSULE, "arrow_schema", "arrow_array")


class Source:
    """What the scan talks to. `one_shot` sources are read once; every other source is read as often as asked."""

    one_shot = False

    def __init__(self, obj: object) -> None:
        self.obj: Any = obj

    def rows(self) -> int | None:
        """How many rows a scan will produce, when the object knows without reading itself; None otherwise."""
        return None

    def accepts(self, predicate: Expr) -> bool:
        """Whether every scan will apply `predicate`, a frame expression over the object's own column names, itself.

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


class ExportingSource(Source):
    """Anything with `__arrow_c_stream__`, every column; its schema comes from `__arrow_c_schema__` or `schema`."""

    def __init__(self, obj: object) -> None:
        super().__init__(obj)
        exporter: Any = obj if hasattr(obj, "__arrow_c_schema__") else getattr(obj, "schema", None)
        if hasattr(exporter, "__arrow_c_schema__"):
            self.__dict__["__arrow_c_schema__"] = exporter.__arrow_c_schema__

    def rows(self) -> int | None:
        """The length of a pyarrow, polars or pandas object is its row count; anything else's may mean anything."""
        return _known_length(self.obj)


class StreamSource(ExportingSource):
    """A RecordBatchReader: the same, read once, its length unknown."""

    one_shot = True

    def rows(self) -> int | None:
        return None


class ArraySource(Source):
    """Anything with `__arrow_c_array__` and no stream: one array, read as often as asked.

    A struct array is a table of its fields; any other array is one column, named `value` unless the array names it.
    """

    def __init__(self, obj: object) -> None:
        super().__init__(obj)
        kind: Any = getattr(obj, "type", None)
        if hasattr(kind, "__arrow_c_schema__"):
            self.__dict__["__arrow_c_schema__"] = kind.__arrow_c_schema__

    def rows(self) -> int | None:
        return _known_length(self.obj)

    def stream(self, columns: Sequence[int] | None, filters: Sequence[Expr]) -> tuple[object, bool]:
        return self.obj.__arrow_c_array__(), False


class CapsuleSource(Source):
    """A bare stream capsule: read once, every column, its schema peeked by the scan."""

    one_shot = True

    def stream(self, columns: Sequence[int] | None, filters: Sequence[Expr]) -> tuple[object, bool]:
        return self.obj, False


class TableSource(ExportingSource):
    """A pyarrow Table or RecordBatch: `select` by position narrows it without copying."""

    def stream(self, columns: Sequence[int] | None, filters: Sequence[Expr]) -> tuple[object, bool]:
        if columns is None:
            return self.obj.__arrow_c_stream__(), False
        return self.obj.select(list(columns)).__arrow_c_stream__(), True


class PolarsFrameSource(ExportingSource):
    """A polars DataFrame: `select` by name narrows it without copying."""

    def stream(self, columns: Sequence[int] | None, filters: Sequence[Expr]) -> tuple[object, bool]:
        if columns is None:
            return self.obj.__arrow_c_stream__(), False
        names = self.obj.columns
        return self.obj.select([names[i] for i in columns]).__arrow_c_stream__(), True


class LazyFrameSource(Source):
    """A polars LazyFrame: a plan, narrowed to the requested columns and collected once per scan, never held.

    The schema is the plan's prediction; a step whose declared output differs from what it produces, such as a
    `map_batches` without a return type, is refused by polars when the plan runs, and the query fails with that.
    """

    def __arrow_c_schema__(self) -> object:
        return self.obj.collect_schema().__arrow_c_schema__()

    def stream(self, columns: Sequence[int] | None, filters: Sequence[Expr]) -> tuple[object, bool]:
        if columns is None:
            return self.obj.collect().__arrow_c_stream__(), False
        names = self.obj.collect_schema().names()
        return self.obj.select([names[i] for i in columns]).collect().__arrow_c_stream__(), True


class PandasSource(Source):
    """A pandas DataFrame through pyarrow, converted a slice of rows at a time so the frame is never copied whole.

    The index is left out, as the previous client did. Column types come from a sample of rows spread over the
    frame, as the previous client's analyzer sampled, since pyarrow can only type an object column by converting it;
    every slice is then converted to those types, and a value a later slice holds that does not fit fails the query.
    Only the requested columns are converted.
    """

    #: Rows the types are inferred from, spread evenly over the frame.
    SAMPLE_ROWS = 1000
    #: Rows converted per slice; the memory held at any time is one slice's worth of Arrow.
    SLICE_ROWS = 1 << 16

    def __init__(self, obj: object) -> None:
        try:
            import pyarrow
        except ImportError as error:
            message = "registering a pandas DataFrame needs pyarrow, which converts it to Arrow"
            raise TypeError(message) from error
        super().__init__(obj)
        self.pyarrow = pyarrow

    def _schema(self, frame: DataFrame) -> Schema:
        step = max(1, len(frame) // self.SAMPLE_ROWS)
        sample = frame.iloc[::step].head(self.SAMPLE_ROWS)
        return self.pyarrow.Schema.from_pandas(sample, preserve_index=False)

    def _batches(self, frame: DataFrame, schema: Schema) -> Iterator[RecordBatch]:
        for start in range(0, len(frame), self.SLICE_ROWS):
            part = frame.iloc[start : start + self.SLICE_ROWS]
            yield from self.pyarrow.Table.from_pandas(part, schema=schema, preserve_index=False).to_batches()

    def __arrow_c_schema__(self) -> object:
        return self._schema(self.obj).__arrow_c_schema__()

    def rows(self) -> int | None:
        return len(self.obj)

    def stream(self, columns: Sequence[int] | None, filters: Sequence[Expr]) -> tuple[object, bool]:
        frame = self.obj if columns is None else self.obj.iloc[:, list(columns)]
        schema = self._schema(frame)
        reader = self.pyarrow.RecordBatchReader.from_batches(schema, self._batches(frame, schema))
        return reader.__arrow_c_stream__(), columns is not None


class DatasetSource(Source):
    """A pyarrow Dataset, or anything with its `scanner()` and `schema`: scanned afresh per read, narrowed by name.

    A predicate the pyarrow translator models is applied by the scanner, which prunes partitions and row groups by
    it before reading.
    """

    def __arrow_c_schema__(self) -> object:
        return self.obj.schema.__arrow_c_schema__()

    def accepts(self, predicate: Expr) -> bool:
        from ._expressions import Untranslatable
        from ._expressions.arrow import to_arrow

        try:
            to_arrow(predicate, self.obj.schema)
        except Untranslatable:
            return False
        return True

    def rows(self) -> int | None:
        """A parquet dataset counts its rows from file metadata; any other format would read the files to count."""
        layout = getattr(self.obj, "format", None)
        if not _is(layout, "pyarrow", "ParquetFileFormat") or not callable(getattr(self.obj, "count_rows", None)):
            return None
        return int(self.obj.count_rows())

    def stream(self, columns: Sequence[int] | None, filters: Sequence[Expr]) -> tuple[object, bool]:
        from ._expressions.arrow import to_arrow

        mask = None
        for predicate in filters:
            term = to_arrow(predicate, self.obj.schema)
            mask = term if mask is None else mask & term
        names = self.obj.schema.names
        chosen = None if columns is None else [names[i] for i in columns]
        reader = self.obj.scanner(columns=chosen, filter=mask).to_reader()
        return reader.__arrow_c_stream__(), columns is not None


class ScannerSource(Source):
    """A pyarrow Scanner: its projection and filter are its own, so it always yields every column it was given."""

    def __arrow_c_schema__(self) -> object:
        return self.obj.projected_schema.__arrow_c_schema__()

    def stream(self, columns: Sequence[int] | None, filters: Sequence[Expr]) -> tuple[object, bool]:
        return self.obj.to_reader().__arrow_c_stream__(), False


def _from(obj: object, library: str) -> bool:
    module = type(obj).__module__
    return module == library or module.startswith(library + ".")


def _is(obj: object, library: str, *names: str) -> bool:
    return type(obj).__name__ in names and _from(obj, library)


def _known_length(obj: object) -> int | None:
    """`len()` when the object comes from a library where a length is a row count, and it has one."""
    if not any(_from(obj, library) for library in ("pyarrow", "polars", "pandas")):
        return None
    try:
        return len(obj)  # type: ignore[arg-type]
    except TypeError:
        return None


def adapt(obj: object) -> Source:
    """The source to register for `obj`.

    An Arrow stream capsule and a pyarrow RecordBatchReader are read once. A pyarrow Table or RecordBatch, a polars
    DataFrame or LazyFrame, a pandas DataFrame, a pyarrow Dataset or Scanner, and anything else exporting
    `__arrow_c_stream__` or `__arrow_c_array__`, are read as often as asked. A `Source` of one's own is registered
    as it is.
    """
    if isinstance(obj, Source):
        return obj
    capsule = _duckdb.capsule_name(obj)
    if capsule == STREAM_CAPSULE:
        return CapsuleSource(obj)
    if capsule in ARROW_CAPSULES:
        message = f"a bare '{capsule}' capsule is not an Arrow stream; register the object it came from instead"
        raise TypeError(message)
    exports = hasattr(obj, "__arrow_c_stream__")
    if exports and any(cls.__name__ == "RecordBatchReader" for cls in type(obj).__mro__):
        return StreamSource(obj)
    if _is(obj, "polars", "LazyFrame"):
        return LazyFrameSource(obj)
    if _is(obj, "pandas", "DataFrame"):
        return PandasSource(obj)
    if exports and _is(obj, "pyarrow", "Table", "RecordBatch"):
        return TableSource(obj)
    if exports and _is(obj, "polars", "DataFrame"):
        return PolarsFrameSource(obj)
    if exports:
        return ExportingSource(obj)
    if hasattr(obj, "__arrow_c_array__"):
        return ArraySource(obj)
    if hasattr(obj, "to_reader") and hasattr(obj, "projected_schema"):
        return ScannerSource(obj)
    if hasattr(obj, "scanner") and hasattr(obj, "schema"):
        return DatasetSource(obj)
    message = (
        f"a registered object must export Arrow through __arrow_c_stream__ or __arrow_c_array__, be an Arrow stream "
        f"capsule, a pyarrow Dataset or Scanner, a polars LazyFrame or a pandas DataFrame; {type(obj).__name__} is "
        f"none of these"
    )
    raise TypeError(message)
