from __future__ import annotations
from typing import TypeAlias, TYPE_CHECKING
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from uuid import UUID
from collections.abc import Mapping

if TYPE_CHECKING:
    from ._expression import Expression

NumericLiteral: TypeAlias = int | float | Decimal
"""Python objects that can be converted to a numerical `ConstantExpression` (integer or floating points numbers.)"""
TemporalLiteral: TypeAlias = date | datetime | time | timedelta
BlobLiteral: TypeAlias = bytes | bytearray | memoryview
"""Python objects that can be converted to a `BLOB` `ConstantExpression`.

Note:
    `bytes` can also be converted to a `BITSTRING`.
"""
NonNestedLiteral: TypeAlias = NumericLiteral | TemporalLiteral | str | bool | BlobLiteral | UUID
PythonLiteral: TypeAlias = (
    NonNestedLiteral | list[PythonLiteral] | tuple[PythonLiteral, ...] | dict[PythonLiteral, PythonLiteral] | None
)
"""Python objects that can be converted to a `ConstantExpression`."""
# the field_ids argument to to_parquet and write_parquet has a recursive structure
ParquetFieldIdsType: TypeAlias = Mapping[str, int | ParquetFieldIdsType]

IntoExprColumn: TypeAlias = Expression | str
"""Types that are, or can be used as a `ColumnExpression`."""
IntoExpr: TypeAlias = IntoExprColumn | PythonLiteral
"""Any type that can be converted to an `Expression` (or is already one).

See Also:
    https://duckdb.org/docs/stable/clients/python/conversion
"""
