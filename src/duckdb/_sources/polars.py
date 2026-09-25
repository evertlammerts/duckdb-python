"""The polars sources: an eager DataFrame chained into slices, and a LazyFrame collected once per scan.

A polars DataFrame or LazyFrame that holds a column of Python objects (`pl.Object`) cannot export that column as
Arrow and is refused up front. Otherwise a DataFrame is narrowed by name and chained into slices of `SLICE_ROWS`
rows, so a frame held in one chunk does not leave every scanning thread sharing the same array; a LazyFrame is
filtered and narrowed as the query asks, collected once per scan, and sliced the same way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .. import _duckdb
from . import Source
from .arrow import ArrowStreamSource

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    import polars as pl

    from .._expressions import Expr


class PolarsFrameSource(ArrowStreamSource):
    """A polars DataFrame: `select` by name narrows it without copying, and slices of it are exported in a chain.

    Polars exports a whole frame as one array however many chunks it holds, and rechunks the frame in place doing
    so, which would leave the scan one array for all its threads. A slice is a view whose export keeps the chunks
    it spans, so the chain hands the scan one array per slice or chunk. The chain takes the GIL itself around the
    slicing, so a pull needs none.
    """

    pull_under_gil = False
    #: Rows per slice, which bounds how many threads can share a frame held in one chunk.
    SLICE_ROWS = 1 << 16

    def __init__(self, obj: object) -> None:
        super().__init__(obj)
        _refuse_object_columns(self.obj.schema)

    def stream(self, columns: Sequence[int] | None, filters: Sequence[Expr]) -> tuple[object, bool]:
        if columns is None:
            narrowed = self.obj
        else:
            names = self.obj.columns
            narrowed = self.obj.select([names[i] for i in columns])
        stream = _duckdb.chain_streams(narrowed.head(0).to_struct(), _polars_slices(narrowed))
        return stream, columns is not None


class LazyFrameSource(Source):
    """A polars LazyFrame: a plan, filtered and narrowed as the query asks and collected once per scan, never held.

    The schema is the plan's prediction; a step whose declared output differs from what it produces, such as a
    `map_batches` without a return type, is refused by polars when the plan runs, and the query fails with that.
    A predicate the polars translator models becomes a `filter` step, which polars pushes into its own scan. The
    collected frame is sliced and chained as `PolarsFrameSource` does.
    """

    pull_under_gil = False

    def __arrow_c_schema__(self) -> object:
        schema = self.obj.collect_schema()
        _refuse_object_columns(schema)
        return schema.__arrow_c_schema__()

    def accepts(self, predicate: Expr) -> bool:
        from .._expressions import Untranslatable
        from .._expressions.polars import to_polars

        try:
            to_polars(predicate, self.obj.collect_schema())
        except Untranslatable:
            return False
        return True

    def stream(self, columns: Sequence[int] | None, filters: Sequence[Expr]) -> tuple[object, bool]:
        from .._expressions.polars import to_polars

        schema = self.obj.collect_schema()
        plan = self.obj
        for predicate in filters:
            plan = plan.filter(to_polars(predicate, schema))
        if columns is not None:
            names = schema.names()
            plan = plan.select([names[i] for i in columns])
        collected = plan.collect()
        stream = _duckdb.chain_streams(collected.head(0).to_struct(), _polars_slices(collected))
        return stream, columns is not None


def _refuse_object_columns(schema: Mapping[str, object]) -> None:
    """Polars exports a column of Python objects as the objects' addresses, so such a frame is refused up front."""
    import polars

    for name, dtype in schema.items():
        if dtype == polars.Object:
            message = f"the polars column '{name}' holds Python objects, which polars cannot export as Arrow"
            raise TypeError(message)


def _polars_slices(frame: pl.DataFrame) -> Iterator[pl.Series]:
    """`frame` as views of `PolarsFrameSource.SLICE_ROWS` rows, each a struct series, which exports per chunk."""
    for part in frame.iter_slices(PolarsFrameSource.SLICE_ROWS):
        yield part.to_struct()
