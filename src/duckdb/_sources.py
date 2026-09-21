"""What a Python object registered as a table is, and the adapters that make each family export an Arrow stream.

Nothing here imports pyarrow, polars or pandas at module level: a family is recognised by its class's module and
name, or by the methods it carries, and its library is imported only inside the adapter that needs it.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pyarrow import RecordBatch, Schema

#: The capsule type, which the C API exposes no other way; the datetime module's own capsule has it.
_CAPSULE = type(datetime.datetime_CAPI)

#: Read once: a stream, or a reader whose exported streams all drain the same source.
STREAM = "stream"
#: Exports a fresh stream every time it is read: an in-memory representation, or a plan that runs per read.
OBJECT = "object"


class LazyFrameSource:
    """A polars LazyFrame: a plan, collected once per scan and never held.

    The schema is the plan's prediction; a step whose declared output differs from what it produces, such as a
    `map_batches` without a return type, is refused by polars when the plan runs, and the query fails with that.
    """

    def __init__(self, lazy: object) -> None:
        self.lazy = lazy

    def __arrow_c_schema__(self) -> object:
        return self.lazy.collect_schema().__arrow_c_schema__()  # type: ignore[attr-defined]

    def __arrow_c_stream__(self, requested_schema: object = None) -> object:
        return self.lazy.collect().__arrow_c_stream__()  # type: ignore[attr-defined]


class PandasSource:
    """A pandas DataFrame through pyarrow, converted a slice of rows at a time so the frame is never copied whole.

    The index is left out, as the previous client did. Column types come from a sample of rows spread over the
    frame, as the previous client's analyzer sampled, since pyarrow can only type an object column by converting it;
    every slice is then converted to those types, and a value a later slice holds that does not fit fails the query.
    """

    #: Rows the types are inferred from, spread evenly over the frame.
    SAMPLE_ROWS = 1000
    #: Rows converted per slice; the memory held at any time is one slice's worth of Arrow.
    SLICE_ROWS = 1 << 16

    def __init__(self, frame: object) -> None:
        try:
            import pyarrow
        except ImportError as error:
            message = "registering a pandas DataFrame needs pyarrow, which converts it to Arrow"
            raise TypeError(message) from error
        self.frame = frame
        self.pyarrow = pyarrow

    def _schema(self) -> Schema:
        frame: Any = self.frame
        step = max(1, len(frame) // self.SAMPLE_ROWS)
        sample = frame.iloc[::step].head(self.SAMPLE_ROWS)
        return self.pyarrow.Schema.from_pandas(sample, preserve_index=False)

    def __arrow_c_schema__(self) -> object:
        return self._schema().__arrow_c_schema__()

    def _batches(self, schema: Schema) -> Iterator[RecordBatch]:
        frame: Any = self.frame
        for start in range(0, len(frame), self.SLICE_ROWS):
            part = frame.iloc[start : start + self.SLICE_ROWS]
            yield from self.pyarrow.Table.from_pandas(part, schema=schema, preserve_index=False).to_batches()

    def __arrow_c_stream__(self, requested_schema: object = None) -> object:
        schema = self._schema()
        return self.pyarrow.RecordBatchReader.from_batches(schema, self._batches(schema)).__arrow_c_stream__()


class DatasetSource:
    """A pyarrow Dataset, or anything with its `scanner()` and `schema`: scanned afresh per read."""

    def __init__(self, dataset: object) -> None:
        self.dataset = dataset

    def __arrow_c_schema__(self) -> object:
        return self.dataset.schema.__arrow_c_schema__()  # type: ignore[attr-defined]

    def __arrow_c_stream__(self, requested_schema: object = None) -> object:
        return self.dataset.scanner().to_reader().__arrow_c_stream__()  # type: ignore[attr-defined]


class ScannerSource:
    """A pyarrow Scanner: its projection and filter are its own, and it runs afresh per read."""

    def __init__(self, scanner: object) -> None:
        self.scanner = scanner

    def __arrow_c_schema__(self) -> object:
        return self.scanner.projected_schema.__arrow_c_schema__()  # type: ignore[attr-defined]

    def __arrow_c_stream__(self, requested_schema: object = None) -> object:
        return self.scanner.to_reader().__arrow_c_stream__()  # type: ignore[attr-defined]


def _is(obj: object, library: str, name: str) -> bool:
    cls = type(obj)
    return cls.__name__ == name and (cls.__module__ == library or cls.__module__.startswith(library + "."))


def adapt(obj: object) -> tuple[object, str]:
    """The object to register for `obj`, and whether it is read once or as often as asked.

    An Arrow stream capsule and a pyarrow RecordBatchReader are streams. A polars LazyFrame, a pandas DataFrame,
    a pyarrow Dataset and a pyarrow Scanner are wrapped so they export a stream per read. Anything else exporting
    `__arrow_c_stream__`, a pyarrow Table or RecordBatch or a polars DataFrame for instance, is registered as it is.
    """
    if isinstance(obj, _CAPSULE):
        return obj, STREAM
    exports = hasattr(obj, "__arrow_c_stream__")
    if exports and any(cls.__name__ == "RecordBatchReader" for cls in type(obj).__mro__):
        return obj, STREAM
    if _is(obj, "polars", "LazyFrame"):
        return LazyFrameSource(obj), OBJECT
    if _is(obj, "pandas", "DataFrame"):
        return PandasSource(obj), OBJECT
    if exports:
        return obj, OBJECT
    if hasattr(obj, "to_reader") and hasattr(obj, "projected_schema"):
        return ScannerSource(obj), OBJECT
    if hasattr(obj, "scanner") and hasattr(obj, "schema"):
        return DatasetSource(obj), OBJECT
    message = (
        f"a registered object must export an Arrow stream through __arrow_c_stream__, be an Arrow stream capsule, "
        f"a pyarrow Dataset or Scanner, a polars LazyFrame or a pandas DataFrame; {type(obj).__name__} is none of these"
    )
    raise TypeError(message)
