"""Type stubs for the nanobind extension module."""

from collections.abc import Mapping, Sequence
from typing import Any

class ChunkView:
    """One fetched chunk column-wise, for the numpy converter.

    The buffer views borrow the chunk's memory and are valid only while
    this object lives; consumers copy out of them immediately.
    """

    @property
    def row_count(self) -> int: ...
    @property
    def column_count(self) -> int: ...
    def type_id(self, column: int) -> int:
        """The facade's LogicalTypeId value for the column."""

    def type_text(self, column: int) -> str:
        """The column's type in its text form."""

    def data(self, column: int) -> memoryview | None:
        """Zero-copy view over the flattened data, or None without a fixed-width layout."""

    def validity(self, column: int) -> memoryview | None:
        """Validity bitmask as 64-bit words, LSB first, or None when all rows are valid."""

    def decimal_scale(self, column: int) -> int:
        """A DECIMAL column's scale."""

    def enum_values(self, column: int) -> list[str]:
        """The ENUM dictionary, index to string."""

    def values(self, column: int) -> list[Any]:
        """Per-cell object fallback for the columns data() cannot serve."""

class Result:
    @property
    def schema(self) -> list[tuple[str, str]]:
        """Column names paired with the text form of their type."""

    def fetch_all(self) -> list[tuple[Any, ...]]:
        """Drain the result into a list of row tuples."""

    @property
    def result_type(self) -> str:
        """One of "rows", "changed_rows", or "nothing"."""

    def close(self) -> None:
        """Release the result so the connection can run another query."""

    def drain(self) -> int:
        """Run to completion and report how many rows changed."""

    def fetch_rows(self, count: int) -> list[tuple[Any, ...]]:
        """Up to `count` more rows, or every remaining row when `count` is zero."""

    def fetch_chunk_view(self) -> ChunkView | None:
        """The next chunk column-wise, or None at the end. Not to be mixed with the row fetches."""

class Connection:
    def execute(self, sql: str, parameters: Sequence[Any] | Mapping[str, Any] | None = None) -> Result:
        """Run one statement, binding parameters positionally or by name."""

    def bind(self, sql: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """What a statement produces and expects, without running it.

        Returns (output columns, parameters), each a list of (name, type).
        """

    def interrupt(self) -> None: ...
    def get_option(self, name: str) -> str: ...
    def set_option(self, name: str, value: str) -> None: ...

class Database:
    def __init__(self, path: str = ":memory:", options: list[tuple[str, str]] | None = None) -> None: ...
    def connect(self) -> Connection: ...

def library_version() -> str:
    """The version of the DuckDB engine this extension is linked against."""
