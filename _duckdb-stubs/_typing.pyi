from __future__ import annotations
from typing import TypeAlias, TYPE_CHECKING, Protocol, Any, TypeVar, Generic
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from uuid import UUID
from collections.abc import Mapping, Iterator

if TYPE_CHECKING:
    from ._expression import Expression

_T_co = TypeVar("_T_co", covariant=True)
_S_co = TypeVar("_S_co", bound=tuple[Any, ...], covariant=True)
_D_co = TypeVar("_D_co", covariant=True)

class NPTypeLike(Protocol, Generic[_T_co]): ...

class NPArrayLike(Protocol, Generic[_S_co, _D_co]):
    def __len__(self) -> int: ...
    def __contains__(self, value: object, /) -> bool: ...
    def __iter__(self) -> Iterator[_D_co]: ...
    def __array__(self, *args: Any, **kwargs: Any) -> Any: ...
    def __array_finalize__(self, *args: Any, **kwargs: Any) -> None: ...
    def __array_wrap__(self, *args: Any, **kwargs: Any) -> Any: ...
    def __getitem__(self, *args: Any, **kwargs: Any) -> Any: ...
    def __setitem__(self, *args: Any, **kwargs: Any) -> None: ...
    @property
    def shape(self) -> _S_co: ...
    @property
    def dtype(self) -> Any: ...
    @property
    def ndim(self) -> int: ...
    @property
    def size(self) -> int: ...

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
    NonNestedLiteral
    | list[PythonLiteral]
    | tuple[PythonLiteral, ...]
    | dict[NonNestedLiteral, PythonLiteral]
    | NPArrayLike[Any, Any]
    | None
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
