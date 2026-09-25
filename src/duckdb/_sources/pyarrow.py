"""The pyarrow sources: a Table or RecordBatch, a RecordBatchReader, and a Dataset or Scanner.

`PyArrowTableSource` and `PyArrowReaderSource` subclass `ArrowStreamSource` in `arrow.py`, narrowing a Table or
RecordBatch by position without copying and reading a RecordBatchReader once. `PyArrowDatasetSource` and
`PyArrowScannerSource` are scanned afresh on every read; a Dataset also applies the query's own simple filters
through its scanner, pruning partitions and row groups before a row is read.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import Source, _is
from .arrow import ArrowStreamSource

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .._expressions import Expr


class PyArrowTableSource(ArrowStreamSource):
    """A pyarrow Table or RecordBatch: `select` by position narrows it without copying."""

    pull_under_gil = False

    def stream(self, columns: Sequence[int] | None, filters: Sequence[Expr]) -> tuple[object, bool]:
        if columns is None:
            return self.obj.__arrow_c_stream__(), False
        return self.obj.select(list(columns)).__arrow_c_stream__(), True


class PyArrowReaderSource(ArrowStreamSource):
    """A RecordBatchReader: the same, read once, its length unknown."""

    one_shot = True

    def rows(self) -> int | None:
        return None


class PyArrowDatasetSource(Source):
    """A pyarrow Dataset, or anything with its `scanner()` and `schema`: scanned afresh per read, narrowed by name.

    A predicate the pyarrow translator models is applied by the scanner, which prunes partitions and row groups by
    it before reading.
    """

    pull_under_gil = False

    def __arrow_c_schema__(self) -> object:
        return self.obj.schema.__arrow_c_schema__()

    def accepts(self, predicate: Expr) -> bool:
        from .._expressions import Untranslatable
        from .._expressions.arrow import to_arrow

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
        from .._expressions.arrow import to_arrow

        mask = None
        for predicate in filters:
            term = to_arrow(predicate, self.obj.schema)
            mask = term if mask is None else mask & term
        names = self.obj.schema.names
        chosen = None if columns is None else [names[i] for i in columns]
        reader = self.obj.scanner(columns=chosen, filter=mask).to_reader()
        return reader.__arrow_c_stream__(), columns is not None


class PyArrowScannerSource(Source):
    """A pyarrow Scanner: its projection and filter are its own, so it always yields every column it was given."""

    pull_under_gil = False

    def __arrow_c_schema__(self) -> object:
        return self.obj.projected_schema.__arrow_c_schema__()

    def stream(self, columns: Sequence[int] | None, filters: Sequence[Expr]) -> tuple[object, bool]:
        return self.obj.to_reader().__arrow_c_stream__(), False
