"""
This module contains classes and methods related to typing
"""

from __future__ import annotations
import duckdb
import typing

__all__: list[str] = [
    "BIGINT",
    "BIT",
    "BLOB",
    "BOOLEAN",
    "DATE",
    "DOUBLE",
    "DuckDBPyType",
    "FLOAT",
    "HUGEINT",
    "INTEGER",
    "INTERVAL",
    "SMALLINT",
    "SQLNULL",
    "TIME",
    "TIMESTAMP",
    "TIMESTAMP_MS",
    "TIMESTAMP_NS",
    "TIMESTAMP_S",
    "TIMESTAMP_TZ",
    "TIME_TZ",
    "TINYINT",
    "UBIGINT",
    "UHUGEINT",
    "UINTEGER",
    "USMALLINT",
    "UTINYINT",
    "UUID",
    "VARCHAR",
]

class DuckDBPyType:
    @typing.overload
    def __eq__(self, other: DuckDBPyType) -> bool:
        """
        Compare two types for equality
        """
    @typing.overload
    def __eq__(self, other: str) -> bool:
        """
        Compare two types for equality
        """
    def __getattr__(self, name: str) -> DuckDBPyType:
        """
        Get the child type by 'name'
        """
    def __getitem__(self, name: str) -> DuckDBPyType:
        """
        Get the child type by 'name'
        """
    def __hash__(self) -> int: ...
    @typing.overload
    def __init__(self, type_str: str, connection: typing.Optional[duckdb.DuckDBPyConnection] = None) -> None:
        """
        Construct a DuckDBPyType from a type name and optional connection.
        """

    @typing.overload
    def __init__(self, obj: object) -> None:
        """
        Construct a DuckDBPyType from a Python type (types.*) or object.
        """
    def __repr__(self) -> str:
        """
        Stringified representation of the type object
        """
    @property
    def children(self) -> list: ...
    @property
    def id(self) -> str: ...

BIGINT: DuckDBPyType  # value = BIGINT
BIT: DuckDBPyType  # value = BIT
BLOB: DuckDBPyType  # value = BLOB
BOOLEAN: DuckDBPyType  # value = BOOLEAN
DATE: DuckDBPyType  # value = DATE
DOUBLE: DuckDBPyType  # value = DOUBLE
FLOAT: DuckDBPyType  # value = FLOAT
HUGEINT: DuckDBPyType  # value = HUGEINT
INTEGER: DuckDBPyType  # value = INTEGER
INTERVAL: DuckDBPyType  # value = INTERVAL
SMALLINT: DuckDBPyType  # value = SMALLINT
SQLNULL: DuckDBPyType  # value = "NULL"
TIME: DuckDBPyType  # value = TIME
TIMESTAMP: DuckDBPyType  # value = TIMESTAMP
TIMESTAMP_MS: DuckDBPyType  # value = TIMESTAMP_MS
TIMESTAMP_NS: DuckDBPyType  # value = TIMESTAMP_NS
TIMESTAMP_S: DuckDBPyType  # value = TIMESTAMP_S
TIMESTAMP_TZ: DuckDBPyType  # value = TIMESTAMP WITH TIME ZONE
TIME_TZ: DuckDBPyType  # value = TIME WITH TIME ZONE
TINYINT: DuckDBPyType  # value = TINYINT
UBIGINT: DuckDBPyType  # value = UBIGINT
UHUGEINT: DuckDBPyType  # value = UHUGEINT
UINTEGER: DuckDBPyType  # value = UINTEGER
USMALLINT: DuckDBPyType  # value = USMALLINT
UTINYINT: DuckDBPyType  # value = UTINYINT
UUID: DuckDBPyType  # value = UUID
VARCHAR: DuckDBPyType  # value = VARCHAR
