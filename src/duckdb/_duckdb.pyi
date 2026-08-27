"""Type stubs for the nanobind extension module."""

from typing import Any

class Result:
    @property
    def schema(self) -> list[tuple[str, str]]:
        """Column names paired with the text form of their type."""

    def fetch_all(self) -> list[tuple[Any, ...]]:
        """Drain the result into a list of row tuples."""

class Connection:
    def execute(self, sql: str) -> Result: ...
    def interrupt(self) -> None: ...
    def get_option(self, name: str) -> str: ...
    def set_option(self, name: str, value: str) -> None: ...

class Database:
    def __init__(self, path: str = ":memory:", options: list[tuple[str, str]] | None = None) -> None: ...
    def connect(self) -> Connection: ...

def library_version() -> str:
    """The version of the DuckDB engine this extension is linked against."""
