"""The Arrow C interface sources: read anything exporting the Arrow C stream or array interface, library-agnostic.

`ArrowStreamSource`, `ArrowArraySource` and `ArrowCapsuleSource` know nothing of pyarrow, polars or pandas; they
read `__arrow_c_stream__`, `__arrow_c_array__` or a bare stream capsule off whatever object carries one, regardless
of what library produced it. What they hand back is what the Arrow scan, the table function in `arrow_scan.cpp`,
reads: a stream capsule, or the schema and array capsules of one array standing for one batch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import Source, _from

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .._expressions import Expr


class ArrowStreamSource(Source):
    """Anything with `__arrow_c_stream__`, every column; its schema comes from `__arrow_c_schema__` or `schema`."""

    def __init__(self, obj: object) -> None:
        super().__init__(obj)
        exporter: Any = obj if hasattr(obj, "__arrow_c_schema__") else getattr(obj, "schema", None)
        if hasattr(exporter, "__arrow_c_schema__"):
            self.__dict__["__arrow_c_schema__"] = exporter.__arrow_c_schema__

    def rows(self) -> int | None:
        """The length of a pyarrow, polars or pandas object is its row count; anything else's may mean anything."""
        return _known_length(self.obj)


class ArrowArraySource(Source):
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


class ArrowCapsuleSource(Source):
    """A bare stream capsule: read once, every column, its schema peeked by the scan."""

    one_shot = True

    def stream(self, columns: Sequence[int] | None, filters: Sequence[Expr]) -> tuple[object, bool]:
        return self.obj, False


def _known_length(obj: object) -> int | None:
    """`len()` when the object comes from a library where a length is a row count, and it has one."""
    if not any(_from(obj, library) for library in ("pyarrow", "polars", "pandas")):
        return None
    try:
        return len(obj)  # type: ignore[arg-type]
    except TypeError:
        return None
