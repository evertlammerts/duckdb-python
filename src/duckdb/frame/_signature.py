"""The SQL signature of a Python function.

A Python function's parameters keep their names, kinds and defaults in SQL: positional-only, standard (positional or
named), `*args`, named-only and `**kwargs` are kinds DuckDB has too. Each parameter's type comes from `parameters`
where the caller gives one, else from its annotation, else it is ANY. A list of types shorter than the parameters
leaves the rest to Python, which fills them in from their defaults. A function Python has no signature for (some
built-ins) takes its parameters from a list of type texts alone, where `*T` is the type of extra positional
arguments and `**T` the type of extra named ones.
"""

from __future__ import annotations

import dataclasses
import datetime
import decimal
import inspect
import sys
import types
import typing
import uuid
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from .. import _duckdb
from ..exceptions import InvalidInputError

if TYPE_CHECKING:
    from collections.abc import Callable

Kind = _duckdb.ParameterKind

_KINDS = {
    inspect.Parameter.POSITIONAL_ONLY: Kind.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD: Kind.STANDARD,
    inspect.Parameter.VAR_POSITIONAL: Kind.POSITIONAL_VARIADIC,
    inspect.Parameter.KEYWORD_ONLY: Kind.NAMED_ONLY,
    inspect.Parameter.VAR_KEYWORD: Kind.NAMED_VARIADIC,
}

_KIND_WORDS = {
    Kind.POSITIONAL_ONLY: "positional-only",
    Kind.STANDARD: "positional or named",
    Kind.POSITIONAL_VARIADIC: "*args",
    Kind.NAMED_ONLY: "named-only",
    Kind.NAMED_VARIADIC: "**kwargs",
}

_PREFIXES: dict[object, str] = {inspect.Parameter.VAR_POSITIONAL: "*", inspect.Parameter.VAR_KEYWORD: "**"}

_ENTRY_ORDER = [None, Kind.POSITIONAL_VARIADIC, Kind.NAMED_VARIADIC]

#: The SQL type an annotation of exactly this class names, which is the type the client gives a value of the class.
#: A subclass is a different annotation: a function annotated with one expects its instances, which SQL cannot give.
CLASS_TYPES: dict[type, str] = {
    bool: "BOOLEAN",
    int: "BIGINT",
    float: "DOUBLE",
    str: "VARCHAR",
    bytes: "BLOB",
    datetime.datetime: "TIMESTAMP",
    datetime.date: "DATE",
    datetime.time: "TIME",
    datetime.timedelta: "INTERVAL",
    uuid.UUID: "UUID",
}

# Since Python 3.14 a signature evaluates its annotations; this format turns a missing name into a reference rather
# than failing, and leaves an annotation object as it is, where asking for text would format it.
_LENIENT: dict[str, object] = {}
if sys.version_info >= (3, 14):
    import annotationlib

    _LENIENT = {"annotation_format": annotationlib.Format.FORWARDREF}

# Python 3.12.0 to 3.12.4 evaluate a text annotation without the function's own type parameters in scope.
_UNSCOPED_TYPE_PARAMETERS = (3, 12) <= sys.version_info < (3, 12, 5)


@dataclasses.dataclass(frozen=True)
class Parameter:
    """One parameter as the engine declares it; `default` is `inspect.Parameter.empty` when there is none."""

    name: str
    kind: _duckdb.ParameterKind
    type: str
    default: object = inspect.Parameter.empty

    def for_engine(self) -> tuple[str, _duckdb.ParameterKind, str, bool, object]:
        """The parameter as the extension takes it, with a flag for the default, since None is one too."""
        has_default = self.default is not inspect.Parameter.empty
        return self.name, self.kind, self.type, has_default, self.default if has_default else None


def sql_signature(
    function: Callable[..., object],
    parameters: Sequence[str] | Mapping[str, str] | None,
    returns: str | None,
) -> tuple[list[Parameter], str]:
    """The parameters and the return type a Python function registers with."""
    if not callable(function):
        message = f"a function is a callable, not {type(function).__name__}"  # type: ignore[unreachable]
        raise TypeError(message)
    if parameters is not None and (
        isinstance(parameters, (str, bytes)) or not isinstance(parameters, (Sequence, Mapping))
    ):
        message = f"parameters is a list or a dict of SQL type texts, not {type(parameters).__name__}"
        raise TypeError(message)
    python = _python_signature(function)
    if python is None:
        declared = _without_signature(parameters)
        return_annotation: object = inspect.Parameter.empty
    else:
        declared = _with_signature(python, parameters)
        return_annotation = python.return_annotation
    return declared, _return_type(returns, return_annotation)


def _is_any(text: str) -> bool:
    """Whether a type text is the word ANY, trimmed of ASCII blanks and in any case."""
    return text.strip(" \t\n\r\f\v").upper() == "ANY"


def _python_signature(function: Callable[..., object]) -> inspect.Signature | None:
    """The function's signature with its annotations evaluated, or None when Python has no signature for it.

    Python evaluates a function's annotations together, so when one fails none is read: each stands for that
    failure, and a parameter needs its type in `parameters` before its annotation would be read.
    """
    try:
        lenient = inspect.signature(function, **_LENIENT)  # type: ignore[arg-type]
    except Exception as error:
        if isinstance(error, (ValueError, TypeError)) and not _from_annotations(error):
            return None
        # A lazy annotation whose code raises, other than for a missing name, leaves Python no signature to give.
        message = (
            f"Invalid Input Error: Python cannot give the function's signature, since evaluating its annotations "
            f"raises {type(error).__name__}: {error}"
        )
        raise InvalidInputError(message) from None
    try:
        scope = None
        # On Python 3.12.0 to 3.12.4, a type parameter named in an annotation of a function reached through
        # functools.partial, a wrapper or a callable object resolves as a module name, as it does in Python there.
        if (
            _UNSCOPED_TYPE_PARAMETERS
            and isinstance(function, (types.FunctionType, types.MethodType))
            and not hasattr(function, "__wrapped__")
        ):
            scope = {parameter.__name__: parameter for parameter in getattr(function, "__type_params__", ())}
        return inspect.signature(function, eval_str=True, locals=scope)
    except Exception as failure:
        return _unread(lenient, failure)


def _unread(signature: inspect.Signature, failure: Exception) -> inspect.Signature:
    """The signature with each annotation standing for the failure that kept Python from evaluating them."""

    def unread(annotation: object) -> object:
        return annotation if annotation is inspect.Parameter.empty else _Unevaluated(annotation, failure)

    parameters = [p.replace(annotation=unread(p.annotation)) for p in signature.parameters.values()]
    return signature.replace(parameters=parameters, return_annotation=unread(signature.return_annotation))


def _from_annotations(error: BaseException) -> bool:
    """Whether the error came out of an annotation's code: `eval` of its text or a function's `__annotate__`."""
    traceback = error.__traceback__
    while traceback is not None:
        code = traceback.tb_frame.f_code
        if code.co_filename == "<string>" or code.co_name == "__annotate__":
            return True
        traceback = traceback.tb_next
    return False


@dataclasses.dataclass(frozen=True)
class _Unevaluated:
    """An annotation, as Python kept it, of a function whose annotations Python could not evaluate, and why."""

    annotation: object
    error: Exception


def _without_signature(parameters: Sequence[str] | Mapping[str, str] | None) -> list[Parameter]:
    if parameters is None or isinstance(parameters, Mapping):
        message = (
            "Invalid Input Error: Python has no signature for the function, so its parameters cannot be read or "
            "named: give parameters as a list of type texts, with '*' before the type of extra positional "
            "arguments and '**' before the type of extra named ones"
        )
        raise InvalidInputError(message)
    split = [_split(text) for text in parameters]
    # Python's order: positional parameters, then the type of extra positional ones, then that of extra named ones.
    ranks = [_ENTRY_ORDER.index(prefix) for prefix, _ in split]
    if ranks != sorted(ranks) or ranks.count(1) > 1 or ranks.count(2) > 1:
        message = (
            f"Invalid Input Error: parameters {list(parameters)!r} is out of order: plain types come first, then at "
            f"most one '*' type, then at most one '**' type"
        )
        raise InvalidInputError(message)
    return [
        Parameter(f"arg{i}", Kind.POSITIONAL_ONLY, sql_type)
        if prefix is None
        else Parameter("args" if prefix is Kind.POSITIONAL_VARIADIC else "kwargs", prefix, sql_type)
        for i, (prefix, sql_type) in enumerate(split)
    ]


def _with_signature(python: inspect.Signature, parameters: Sequence[str] | Mapping[str, str] | None) -> list[Parameter]:
    entries = _declared_entries(list(python.parameters.values()), parameters)
    given = {} if parameters is None else _given_types(entries, parameters)
    declared = []
    for entry in entries:
        kind = _KINDS[entry.kind]
        if entry.name not in given:
            sql_type = _annotation_type(entry.annotation, f"parameter {entry.name!r}", "give its type in parameters")
        else:
            text = given[entry.name]
            prefix, sql_type = _split(text)
            if prefix is not None and prefix is not kind:
                message = (
                    f"Invalid Input Error: the type {text!r} for parameter {entry.name!r} is marked "
                    f"{_KIND_WORDS[prefix]} by its prefix, but the parameter is {_KIND_WORDS[kind]}"
                )
                raise InvalidInputError(message)
        declared.append(Parameter(entry.name, kind, sql_type, entry.default))
    return declared


def _declared_entries(
    entries: list[inspect.Parameter], parameters: Sequence[str] | Mapping[str, str] | None
) -> list[inspect.Parameter]:
    """The parameters SQL sees: all of them, or as many as a list of types covers, the rest left to Python."""
    if parameters is None or isinstance(parameters, Mapping) or len(parameters) == len(entries):
        return entries
    if len(parameters) > len(entries):
        names = ", ".join(_PREFIXES.get(entry.kind, "") + entry.name for entry in entries)
        message = (
            f"Invalid Input Error: parameters gives {_count(len(parameters), 'type')}, but the function has "
            f"{_count(len(entries), 'parameter')}" + (f": {names}" if entries else "")
        )
        raise InvalidInputError(message)
    for entry in entries[len(parameters) :]:
        if entry.default is inspect.Parameter.empty and entry.kind not in _PREFIXES:
            message = (
                f"Invalid Input Error: parameters gives {_count(len(parameters), 'type')}, but parameter "
                f"{entry.name!r} after them has no default, so SQL must pass it and it needs a type too"
            )
            raise InvalidInputError(message)
    return entries[: len(parameters)]


def _given_types(entries: list[inspect.Parameter], parameters: Sequence[str] | Mapping[str, str]) -> dict[str, str]:
    """The type texts `parameters` gives, by parameter name."""
    if not isinstance(parameters, Mapping):
        return {entry.name: text for entry, text in zip(entries, parameters, strict=True)}
    for key in parameters:
        if key not in {entry.name for entry in entries}:
            names = ", ".join(entry.name for entry in entries)
            message = (
                f"Invalid Input Error: parameters names {key!r}, which is not a parameter of the function: {names}"
            )
            raise InvalidInputError(message)
    return dict(parameters)


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _split(text: object) -> tuple[_duckdb.ParameterKind | None, str]:
    """A type text's kind prefix, if it has one, and the type, with ANY in the one spelling the extension takes."""
    if not isinstance(text, str):
        message = f"a parameter type is SQL type text, not {type(text).__name__}"
        raise TypeError(message)
    stripped = text.lstrip(" \t\n\r\f\v")
    prefix = None
    if stripped.startswith("**"):
        prefix, stripped = Kind.NAMED_VARIADIC, stripped[2:]
    elif stripped.startswith("*"):
        prefix, stripped = Kind.POSITIONAL_VARIADIC, stripped[1:]
    return prefix, "ANY" if _is_any(stripped) else stripped


def _return_type(returns: str | None, annotation: object) -> str:
    if returns is not None:
        if not isinstance(returns, str):
            message = f"returns is SQL type text, not {type(returns).__name__}"  # type: ignore[unreachable]
            raise TypeError(message)
        sql_type = returns
    elif annotation is inspect.Parameter.empty:
        message = "Invalid Input Error: the function's return type is unknown: give returns or annotate the return"
        raise InvalidInputError(message)
    else:
        sql_type = _annotation_type(annotation, "the return", "give returns")
    # The return type must be known before the first call, and a Python function's is known only after one.
    if _is_any(sql_type):
        message = "Invalid Input Error: an ANY return type is not supported yet"
        raise InvalidInputError(message)
    return sql_type


class _Unreadable(Exception):
    """Why no SQL type can be read from an annotation."""


def _annotation_type(annotation: object, owner: str, remedy: str) -> str:
    if annotation is inspect.Parameter.empty:
        return "ANY"
    try:
        return _read(annotation, nested=False)
    except _Unreadable as reason:
        shown = _shown(annotation.annotation if isinstance(annotation, _Unevaluated) else annotation)
        message = f"Invalid Input Error: no SQL type is read from the annotation {shown} of {owner}: {reason}; {remedy}"
        raise InvalidInputError(message) from None


def _shown(annotation: object) -> str:
    """An annotation as a refusal names it; formatting it runs the annotation's own code, which may fail."""
    if isinstance(annotation, str):
        return repr(annotation)
    if isinstance(annotation, typing.ForwardRef):
        return repr(annotation.__forward_arg__)
    try:
        return repr(inspect.formatannotation(annotation))
    except Exception:
        return f"of type {type(annotation).__name__}"


def _read(annotation: object, *, nested: bool) -> str:
    """The SQL type text an annotation names."""
    if annotation is typing.Any:
        if nested:
            message = "ANY can only be a whole parameter's type"
            raise _Unreadable(message)
        return "ANY"
    if isinstance(annotation, _Unevaluated):
        error = annotation.error
        message = f"Python could not evaluate the function's annotations ({type(error).__name__}: {error})"
        raise _Unreadable(message)
    if isinstance(annotation, str):
        message = "Python could not evaluate it"
        raise _Unreadable(message)
    if isinstance(annotation, typing.ForwardRef):
        message = f"Python could not evaluate {annotation.__forward_arg__!r}"
        raise _Unreadable(message)
    if isinstance(annotation, type) and annotation in CLASS_TYPES:
        return CLASS_TYPES[annotation]
    if annotation is decimal.Decimal:
        message = "a Decimal's width and scale are not in its class; give DECIMAL(width, scale)"
        raise _Unreadable(message)
    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if origin is typing.Union or origin is types.UnionType:
        members = [a for a in args if a is not type(None)]
        if len(members) != 1:
            message = "a union of several types has no one SQL type"
            raise _Unreadable(message)
        # Every SQL value can be NULL, so `T | None` is T.
        return _read(members[0], nested=nested)
    if origin is list and len(args) == 1:
        return f"{_read(args[0], nested=True)}[]"
    if origin is tuple and len(args) == 2 and args[1] is Ellipsis:
        return f"{_read(args[0], nested=True)}[]"
    if origin is dict and len(args) == 2:
        return f"MAP({_read(args[0], nested=True)}, {_read(args[1], nested=True)})"
    if annotation in (list, tuple, dict):
        message = "it does not say what it holds"
        raise _Unreadable(message)
    message = "the client knows no SQL type for it"
    raise _Unreadable(message)
