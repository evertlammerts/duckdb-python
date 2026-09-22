"""A frame expression as a polars expression, for a source that filters through polars."""

from __future__ import annotations

import datetime
import decimal
import math
import operator
from typing import TYPE_CHECKING, Any

from .expr import Between, Binary, Col, Expr, In, Lit, Postfix, Unary, Untranslatable

if TYPE_CHECKING:
    from collections.abc import Callable

    import polars as pl

_COMPARISONS: dict[str, Callable[[Any, Any], Any]] = {
    "=": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}

_INTEGER_BOUNDS: dict[str, tuple[int, int]] = {
    "Int8": (-(1 << 7), (1 << 7) - 1),
    "Int16": (-(1 << 15), (1 << 15) - 1),
    "Int32": (-(1 << 31), (1 << 31) - 1),
    "Int64": (-(1 << 63), (1 << 63) - 1),
    "UInt8": (0, (1 << 8) - 1),
    "UInt16": (0, (1 << 16) - 1),
    "UInt32": (0, (1 << 32) - 1),
    "UInt64": (0, (1 << 64) - 1),
}


def to_polars(node: Expr, schema: pl.Schema) -> pl.Expr:
    """`node` over the columns of `schema`, or `Untranslatable` when polars would not answer as the engine does.

    What translates: a comparison of a column with a value of the column's type, `BETWEEN`, `IN`, `IS NULL`,
    `IS NOT NULL`, `AND`, `OR` and `NOT`, over boolean, integer, floating-point, string, decimal, date, time and
    naive datetime columns. Polars orders NaN above every number as the engine does, and combines NULL with the
    engine's three-valued logic, so no comparison is rewritten; a NaN value is refused all the same. `schema` must
    be the schema the scan declared to the engine, since the value is typed as the column is.
    """
    import polars as pl

    match node:
        case Binary(op="AND", left=left, right=right):
            return to_polars(left, schema) & to_polars(right, schema)
        case Binary(op="OR", left=left, right=right):
            return to_polars(left, schema) | to_polars(right, schema)
        case Unary(op="NOT", operand=operand):
            return ~to_polars(operand, schema)
        case Postfix(op="IS NULL", operand=Col(parts=(name,))):
            _column(schema, name)
            return pl.col(name).is_null()
        case Postfix(op="IS NOT NULL", operand=Col(parts=(name,))):
            _column(schema, name)
            return pl.col(name).is_not_null()
        case Binary(op=op, left=Col(parts=(name,)), right=Lit(value=value)) if op in _COMPARISONS:
            kind = _column(schema, name)
            compared: pl.Expr = _COMPARISONS[op](pl.col(name), _literal(value, kind))
            return compared
        case Between(operand=Col(parts=(name,)), low=Lit(value=low), high=Lit(value=high)):
            kind = _column(schema, name)
            return (pl.col(name) >= _literal(low, kind)) & (pl.col(name) <= _literal(high, kind))
        case In(operand=Col(parts=(name,)), values=values):
            kind = _column(schema, name)
            candidates = [value.value for value in values if isinstance(value, Lit) and _fits(value.value, kind)]
            if len(candidates) != len(values):
                raise Untranslatable(node)
            if not candidates:
                return pl.lit(False)
            return pl.col(name).is_in(pl.Series(candidates, dtype=kind))
    raise Untranslatable(node)


def _column(schema: pl.Schema, name: str) -> pl.DataType:
    import polars as pl

    kind = schema.get(name)
    if kind is None:
        raise Untranslatable(name)
    if not (
        kind == pl.Boolean
        or kind.is_integer()
        or kind.is_float()
        or kind == pl.String
        or kind.is_decimal()
        or kind == pl.Date
        or kind == pl.Time
        or (isinstance(kind, pl.Datetime) and kind.time_zone is None)
    ):
        raise Untranslatable(kind)
    return kind


def _fits(value: object, kind: pl.DataType) -> bool:
    """Whether `value` is a Python value of the column's type, so it compares without a cast on either side."""
    import polars as pl

    if isinstance(value, bool):
        return kind == pl.Boolean
    if isinstance(value, int):
        bounds = _INTEGER_BOUNDS.get(str(kind))
        return bounds is not None and bounds[0] <= value <= bounds[1]
    if isinstance(value, float):
        return kind.is_float() and not math.isnan(value)
    if isinstance(value, str):
        return kind == pl.String
    if isinstance(value, decimal.Decimal):
        return kind.is_decimal() and value.is_finite()
    if isinstance(value, datetime.datetime):
        return isinstance(kind, pl.Datetime) and kind.time_zone is None and value.tzinfo is None
    if isinstance(value, datetime.date):
        return kind == pl.Date
    if isinstance(value, datetime.time):
        return kind == pl.Time and value.tzinfo is None
    return False


def _literal(value: object, kind: pl.DataType) -> pl.Expr:
    import polars as pl

    if not _fits(value, kind):
        raise Untranslatable(value)
    return pl.lit(value, dtype=kind)
