"""The types of this package's arguments, each naming the one way the argument reads a Python string.

A string in an argument is read in exactly one way, and the annotation says which: `ColumnName` is a column,
`ValueOrExpr` takes it as text, `SqlText` splices it into the query as written. Where no reading makes sense, as in a
condition, the annotation is `Expr` and a string raises `TypeError`. A string is always one name, never split at a dot
and never taken as a sequence of its characters.

The names in `__all__` are stable, for annotating code that wraps this package:

    from duckdb.frame.typing import ColumnOrExpr

    def newest(plan: Frame, by: ColumnOrExpr) -> Frame: ...

Each alias's docstring names its reader, the function that turns an argument into what the package works with.
"""

from __future__ import annotations

import datetime
import decimal
import os
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Literal, Protocol, TypeAlias

from .._expressions.expr import Expr

__all__ = [
    "ColumnName",
    "ColumnNames",
    "ColumnOrExpr",
    "ColumnsOrExprs",
    "Exprs",
    "FilePath",
    "JoinHow",
    "NewName",
    "Parameters",
    "PythonValue",
    "QualifiedName",
    "SqlText",
    "ValueOrExpr",
    "ValueSequence",
]


class _NumpyScalar(Protocol):
    """A numpy scalar, recognised by shape so numpy need not be installed; datetime64 and timedelta64 convert."""

    @property
    def dtype(self) -> object: ...

    def astype(self, dtype: Any, /) -> object: ...  # noqa: ANN401


class _Pair(Protocol):
    """Two items read by position, as from a tuple, a list or a numpy row."""

    def __getitem__(self, index: int, /) -> object: ...


# --- names ---------------------------------------------------------------------

ColumnName: TypeAlias = str
"""One column the input has, by its whole name. Reader: `duckdb._expressions.expr._as_name`."""

ColumnNames: TypeAlias = ColumnName | Iterable[ColumnName]
"""One column or several; a bare string is one column. Reader: `duckdb._expressions.expr._as_names`."""

NewName: TypeAlias = str
"""A name the call gives: an alias, a new or renamed column, a parameter.

Reader: `duckdb._expressions.expr._as_name`.
"""

_ObjectName: TypeAlias = str
"""One bare name in the catalog: a table or function. Reader: `duckdb._expressions.expr._as_name`."""

QualifiedName: TypeAlias = str | tuple[str, ...]
"""A table, view, macro or function; a tuple qualifies it by schema. Reader: `duckdb._expressions.expr.name_parts`."""

_FieldKey: TypeAlias = str | int | _NumpyScalar
"""A struct field's name, a map key, or a list position; a numpy integer is its value.

Reader: `duckdb._expressions.expr._as_field_key`.
"""

_OptionName: TypeAlias = str
"""A COPY option name, written into the SQL bare, so an identifier. Reader: `duckdb.frame.plan._option`."""

_Renames: TypeAlias = Mapping[ColumnName, NewName] | Iterable[Iterable[ColumnName | NewName]]
"""Columns the input has, each to its new name: a mapping, or (old, new) pairs; a string is never a pair.

Reader: `duckdb._expressions.expr._as_renames`.
"""

_AttributeName: TypeAlias = str
"""An attribute Python looks up; a join side has none, and its error names the bracket to use instead.

Reader: `duckdb._expressions.expr.Side.__getattr__`.
"""

# --- SQL ------------------------------------------------------------------------

SqlText: TypeAlias = str
"""SQL, spliced into the query as written and never escaped, so never built from untrusted input.

A type, as `cast()` and `values()` take one, is SQL text too. Reader: `duckdb._expressions.expr._as_sql`.
"""

# --- expressions -----------------------------------------------------------------

ColumnOrExpr: TypeAlias = Expr | ColumnName
"""An expression, or a string naming one column. Reader: `duckdb.frame.plan._as_expr`."""

ColumnsOrExprs: TypeAlias = ColumnOrExpr | Iterable[ColumnOrExpr]
"""One or several of `ColumnOrExpr`; a bare string is one column. Reader: `duckdb.frame.plan._as_exprs`."""

PythonValue: TypeAlias = (
    bool
    | int
    | float
    | str
    | bytes
    | datetime.date
    | datetime.time
    | datetime.timedelta
    | decimal.Decimal
    | uuid.UUID
    | list[Any]
    | tuple[Any, ...]
    | dict[Any, Any]
    | _NumpyScalar
    | None
)
"""A Python value the package converts; a string is text. Reader: `duckdb._expressions.expr.Lit`.

The containers hold `Any` because a list is invariant: `list[PythonValue]` would refuse a `list[int]`.
"""

ValueOrExpr: TypeAlias = Expr | PythonValue | Callable[..., object]
"""An expression, or a value it stands for; a string is text, a function a SQL lambda.

Reader: `duckdb._expressions.expr._lift`.
"""

Exprs: TypeAlias = Expr | Sequence[Expr]
"""One expression, or a list or tuple of them in order; a string raises `TypeError`.

Reader: `duckdb._expressions.expr._as_expressions`.
"""

# --- values ----------------------------------------------------------------------

ValueSequence: TypeAlias = Iterable[ValueOrExpr]
"""Values to test membership in; a bare string raises `TypeError`. Reader: `duckdb._expressions.expr.Expr.isin`."""

_Rows: TypeAlias = Iterable[Iterable[PythonValue]]
"""The rows of `values()`; a bare string, as the rows or as one row, raises `TypeError`.

Reader: `duckdb.frame.plan._as_rows`.
"""

_Heading: TypeAlias = NewName | Iterable[NewName | Iterable[NewName | SqlText | None]]
"""The columns of `values()`: names, or pairs of a name and its type, where a bare string is one column.

A pair is any iterable but text, such as a tuple, a list or a numpy row; without a type, or with None, the column
is untyped.

Reader: `duckdb.frame.plan._as_heading`.
"""

Parameters: TypeAlias = Iterable[PythonValue] | Mapping[NewName, PythonValue]
"""Values for a statement's placeholders, in order or by name; a bare string raises `TypeError`.

Reader: `duckdb.frame.connection._as_parameters`.
"""

_NamedParameters: TypeAlias = Mapping[NewName, PythonValue]
"""A value for each `param()` of a plan, by its name. Reader: `duckdb.frame.plan._as_named_parameters`."""

_MacroParameters: TypeAlias = NewName | Iterable[NewName | _Pair]
"""A macro's parameters: names, or (name, default) pairs read by position; a bare string is one parameter.

Reader: `duckdb.frame.connection._as_macro_parameters`.
"""

_FunctionParameters: TypeAlias = Sequence[SqlText] | Mapping[NewName, SqlText]
"""The SQL types of a Python function's parameters, in order or by name.

Reader: `duckdb.frame._signature.sql_signature`.
"""

_TextValue: TypeAlias = str
"""A value that must be text. Reader: `duckdb._expressions.expr._as_text`."""

FilePath: TypeAlias = str | os.PathLike[str]
"""A file or directory, as text or a path object. Reader: `os.fspath`."""

_TableArgument: TypeAlias = Expr | PythonValue | os.PathLike[str]
"""An argument of a table function; a path object is its text. Reader: `duckdb.frame.plan._argument`."""

_OptionValue: TypeAlias = Expr | PythonValue
"""A COPY option's value, written into the SQL; a list is a column list. Reader: `duckdb.frame.plan._option_value`."""

_SettingValue: TypeAlias = object
"""A setting's value, any value at all: DuckDB is given its `str()`. Reader: `duckdb.frame.connection.connect`."""

_RequestedSchema: TypeAlias = object
"""The schema an Arrow consumer may ask for, as a capsule; ignored, as the protocol permits.

Reader: `duckdb.frame.plan.Bound.__arrow_c_stream__`.
"""

_StepState: TypeAlias = dict[str, object]
"""A step's fields by name, as pickle hands them back. Reader: `duckdb.frame.plan.Step.__setstate__`."""

_TableSource: TypeAlias = object
"""Any object `register()` reads as a table; what qualifies is decided by looking at it, and a string raises.

Reader: `duckdb._sources.adapt`.
"""

# --- words from a fixed set ----------------------------------------------------------

JoinHow: TypeAlias = Literal[
    "inner", "left", "right", "outer", "full", "semi", "anti", "cross", "positional", "natural", "asof"
]
"""How two plans join; the case of the letters does not matter. Reader: `duckdb.frame.plan.Join`."""

_SampleMethod: TypeAlias = Literal["reservoir", "bernoulli", "system"]
"""How `sample()` draws rows; the case of the letters does not matter. Reader: `duckdb.frame.plan._as_sample_method`."""

_NullHandling: TypeAlias = Literal["default", "special"]
"""Whether a NULL argument skips the call. Reader: `duckdb.frame.connection._NULL_HANDLING`."""

_Stability: TypeAlias = Literal["consistent", "volatile", "consistent_within_query"]
"""When DuckDB may reuse a function's result. Reader: `duckdb.frame.connection._STABILITY`."""
