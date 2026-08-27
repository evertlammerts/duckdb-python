"""Type stubs for the nanobind extension module."""

from collections.abc import Mapping, Sequence
from typing import Any

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

class Connection:
    def execute(self, sql: str, parameters: Sequence[Any] | Mapping[str, Any] | None = None) -> Result:
        """Run one statement, binding parameters positionally or by name."""

    def interrupt(self) -> None: ...
    def get_option(self, name: str) -> str: ...
    def set_option(self, name: str, value: str) -> None: ...

class Database:
    def __init__(self, path: str = ":memory:", options: list[tuple[str, str]] | None = None) -> None: ...
    def connect(self) -> Connection: ...

def library_version() -> str:
    """The version of the DuckDB engine this extension is linked against."""
