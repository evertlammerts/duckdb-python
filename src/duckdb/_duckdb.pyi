"""Type stubs for the compiled extension module."""

import enum
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

class FunctionNullHandling(enum.Enum):
    """Whether a scalar function handles NULL arguments itself."""

    DEFAULT = 0
    SPECIAL = 1

class FunctionStability(enum.Enum):
    """How far DuckDB may reuse a scalar function's result."""

    CONSISTENT = 0
    VOLATILE = 1
    CONSISTENT_WITHIN_QUERY = 2

class ChunkView:
    """One fetched batch of rows, column by column; its buffer views borrow memory that dies with this object."""

    @property
    def row_count(self) -> int: ...
    @property
    def row_offset(self) -> int:
        """Leading rows a prior row fetch already consumed; the buffers still cover the whole batch."""

    @property
    def column_count(self) -> int: ...
    def type_id(self, column: int) -> int:
        """DuckDB's type-id number for the column."""

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
    """One statement's result, streamed and read from a single thread; only `close()` is safe from another."""

    @property
    def schema(self) -> list[tuple[str, str]]:
        """Column names paired with the text form of their type."""

    def fetch_all(self) -> list[tuple[Any, ...]]:
        """Every remaining row, as a list of tuples."""

    @property
    def result_type(self) -> str:
        """One of "rows", "changed_rows", or "nothing"."""

    @property
    def statement_type(self) -> str:
        """The kind of statement behind the result; the PIVOT family cannot answer until the result is stepped."""

    def close(self) -> None:
        """Release the result so the connection can run another query."""

    def drain(self) -> int:
        """Run the statement to completion and report how many rows changed."""

    def fetch_rows(self, count: int) -> list[tuple[Any, ...]]:
        """Up to `count` more rows, or every remaining row when `count` is zero."""

    def fetch_chunk_view(self) -> ChunkView | None:
        """The next batch column by column, or None at the end, carrying the offset a prior row fetch left."""

    @property
    def schema_types(self) -> list[tuple[int, int, list[str] | None]]:
        """Per-column (type id, decimal scale, enum dictionary or None), from the schema alone."""

class Connection:
    def execute(self, sql: str, parameters: Sequence[Any] | Mapping[str, Any] | None = None) -> Result:
        """Run one statement, binding parameters positionally or by name."""

    def bind(self, sql: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """The output columns and the parameters a statement would have, without running it, as (name, type) pairs."""

    def create_scalar_function(
        self,
        name: str,
        callable: Callable[..., Any],
        parameters: list[str],
        returns: str,
        null_handling: FunctionNullHandling,
        stability: FunctionStability,
    ) -> None:
        """Register a Python callable as a scalar SQL function; the type texts reach DuckDB's parser unchanged."""

    def register_object(self, name: str, obj: object, one_shot: bool, native: bool) -> None:
        """Make `obj` readable as the table `name` on this database.

        A one-shot object is a stream, read once. A native object is read through the numpy scan rather than the
        Arrow scan.
        """

    def unregister_object(self, name: str) -> bool:
        """Forget the object registered as `name`; False when there was none."""

    def registered_kind(self, name: str) -> str | None:
        """The kind of a registered name, "stream" or "object", or None when nothing is registered as it."""

    def interrupt(self) -> None: ...
    def get_option(self, name: str) -> str: ...
    def set_option(self, name: str, value: str) -> None: ...
    def close(self) -> None:
        """Close the connection now. Idempotent; every other method raises InterfaceError afterwards."""

class Database:
    """One open database, kept alive by every connection and result on it."""

    def __init__(self, path: str = ":memory:", options: list[tuple[str, str]] | None = None) -> None: ...
    def connect(self) -> Connection: ...

def library_version() -> str:
    """The DuckDB version this extension module is linked against."""

def capsule_name(object: object) -> str | None:
    """The name a capsule carries, which for Arrow data says what it holds; None for anything else."""

def chain_streams(schema: object, parts: Iterable[object]) -> object:
    """Several exports read as one stream, each part with the schema of `schema`, which the caller guarantees."""
