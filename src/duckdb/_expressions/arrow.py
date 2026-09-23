"""An expression of this package as a pyarrow compute expression, for a registered object read through pyarrow."""

from __future__ import annotations

import datetime
import math
import operator
from typing import TYPE_CHECKING, Any

from .expr import Between, Binary, Col, Expr, In, Lit, Postfix, Unary, Untranslatable

if TYPE_CHECKING:
    from collections.abc import Callable

    import pyarrow as pa
    import pyarrow.compute as pc

_COMPARISONS: dict[str, Callable[[Any, Any], Any]] = {
    "=": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


def to_arrow(node: Expr, schema: pa.Schema) -> pc.Expression:
    """`node` over the columns of `schema`, or `Untranslatable` when pyarrow would not answer as the engine does.

    What translates: a comparison of a column with a value of the column's type, `BETWEEN`, `IN`, `IS NULL`,
    `IS NOT NULL`, `AND`, `OR` and `NOT`, over boolean, integer, floating-point, string, date and naive timestamp
    columns. The engine orders NaN above every number and pyarrow orders it nowhere, so `>` and `>=` on a
    floating-point column admit NaN rows, and a NaN value is refused. `schema` must be the schema the object
    declares to the engine when a query binds over it, since the value is typed as the column is and a unit or width
    that differs would compare differently.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    _filterable(schema)
    match node:
        case Binary(op="AND", left=left, right=right):
            return to_arrow(left, schema) & to_arrow(right, schema)
        case Binary(op="OR", left=left, right=right):
            return to_arrow(left, schema) | to_arrow(right, schema)
        case Unary(op="NOT", operand=operand):
            return ~to_arrow(operand, schema)
        case Postfix(op="IS NULL", operand=Col(parts=(name,))):
            _column(schema, name)
            return pc.field(name).is_null()
        case Postfix(op="IS NOT NULL", operand=Col(parts=(name,))):
            _column(schema, name)
            return pc.field(name).is_valid()
        case Binary(op=op, left=Col(parts=(name,)), right=Lit(value=value)) if op in _COMPARISONS:
            kind = _column(schema, name)
            compared = _COMPARISONS[op](pc.field(name), _scalar(value, kind))
            if op in (">", ">=") and _floating(kind):
                return compared | pc.is_nan(pc.field(name))
            return compared
        case Between(operand=Col(parts=(name,)), low=Lit(value=low), high=Lit(value=high)):
            kind = _column(schema, name)
            return (pc.field(name) >= _scalar(low, kind)) & (pc.field(name) <= _scalar(high, kind))
        case In(operand=Col(parts=(name,)), values=values):
            kind = _column(schema, name)
            candidates = [_scalar(value.value, kind) for value in values if isinstance(value, Lit)]
            if len(candidates) != len(values):
                raise Untranslatable(node)
            if not candidates:
                return pc.scalar(False)
            # `is_in` answers false for a NULL, where the engine answers NULL; the difference shows under NOT.
            member = pc.field(name).isin(candidates)
            return pc.if_else(pc.field(name).is_valid(), member, pa.scalar(None, pa.bool_()))
    raise Untranslatable(node)


def _filterable(schema: pa.Schema) -> None:
    """Refuses a schema with a view column, whose rows pyarrow 25 cannot take by mask, so no predicate applies."""
    import pyarrow as pa

    for field in schema:
        if pa.types.is_string_view(field.type) or pa.types.is_binary_view(field.type):
            raise Untranslatable(field)


def _column(schema: pa.Schema, name: str) -> pa.DataType:
    import pyarrow as pa

    index = schema.get_field_index(name)
    if index < 0:
        raise Untranslatable(name)
    kind = schema.field(index).type
    if not (
        pa.types.is_boolean(kind)
        or pa.types.is_integer(kind)
        or _floating(kind)
        or _string(kind)
        or pa.types.is_date(kind)
        or (pa.types.is_timestamp(kind) and kind.tz is None)
    ):
        raise Untranslatable(kind)
    return kind


def _floating(kind: pa.DataType) -> bool:
    import pyarrow as pa

    return bool(pa.types.is_floating(kind))


def _string(kind: pa.DataType) -> bool:
    import pyarrow as pa

    return bool(pa.types.is_string(kind) or pa.types.is_large_string(kind))


def _fits(value: object, kind: pa.DataType) -> bool:
    """Whether `value` is a Python value of the column's type, so it compares without a cast on either side."""
    import pyarrow as pa

    if isinstance(value, bool):
        return bool(pa.types.is_boolean(kind))
    if isinstance(value, int):
        return bool(pa.types.is_integer(kind))
    if isinstance(value, float):
        return _floating(kind) and not math.isnan(value)
    if isinstance(value, str):
        return _string(kind)
    if isinstance(value, datetime.datetime):
        return bool(pa.types.is_timestamp(kind) and kind.tz is None and value.tzinfo is None)
    if isinstance(value, datetime.date):
        return bool(pa.types.is_date(kind))
    return False


def _scalar(value: object, kind: pa.DataType) -> pa.Scalar:
    import pyarrow as pa

    if not _fits(value, kind):
        raise Untranslatable(value)
    try:
        return pa.scalar(value, type=kind)
    except (pa.ArrowException, OverflowError, ValueError) as error:
        raise Untranslatable(value) from error
