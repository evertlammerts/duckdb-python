"""The exception hierarchy.

Rooted in PEP 249 so DB-API consumers can catch what they expect, with a
concrete leaf per engine error code that carries a distinct meaning.

The leaves are derived from the engine's error-code space, not from the
previous client's class list. Some of its classes drew distinctions the V2 code
space does not make, and inventing leaves the engine cannot produce would mean
raising them from guesswork, so those classes are deliberately gone.
"""

from __future__ import annotations

from ._error_codes import ERROR_CODES

__all__ = [
    "CatalogError",
    "ConfigurationError",
    "ConnectionError",
    "ConstraintError",
    "ConversionError",
    "DataError",
    "DatabaseError",
    "Error",
    "FatalError",
    "IOError",
    "IntegrityError",
    "InterfaceError",
    "InternalError",
    "InterruptError",
    "InvalidInputError",
    "NotSupportedError",
    "OperationalError",
    "OutOfMemoryError",
    "ParserError",
    "ProgrammingError",
    "TransactionError",
    "Warning",
]


# --- PEP 249 roots ---------------------------------------------------------


class Warning(Exception):
    """Warnings raised during processing, per PEP 249."""


class Error(Exception):
    """Base of every error the engine reports, and of `InterfaceError`.

    A plan refused before the engine sees it raises Python's own
    `TypeError` or `ValueError` instead, `NeedsConnection` among them.
    """


class InterfaceError(Error):
    """Misuse of this package itself, rather than a failure in the database."""


class DatabaseError(Error):
    """A failure reported by the database."""


class DataError(DatabaseError):
    """The data was wrong: bad value, out of range, division by zero."""


class OperationalError(DatabaseError):
    """A failure outside the programmer's control: I/O, memory, transactions."""


class IntegrityError(DatabaseError):
    """A constraint was violated."""


class InternalError(DatabaseError):
    """The database reached a state it considers impossible."""


class ProgrammingError(DatabaseError):
    """The statement or its arguments were wrong."""


class NotSupportedError(DatabaseError):
    """The operation is not implemented by this engine."""


# --- concrete leaves -------------------------------------------------------


class IOError(OperationalError):
    """Reading or writing failed."""


class OutOfMemoryError(OperationalError):
    """An allocation failed."""


class ConnectionError(OperationalError):
    """The connection is closed or no longer usable."""


class TransactionError(OperationalError):
    """The transaction could not proceed."""


class InterruptError(OperationalError):
    """The query was interrupted."""


class FatalError(InternalError):
    """The database is no longer usable and must be reopened."""


class ParserError(ProgrammingError):
    """The SQL could not be parsed."""


class CatalogError(ProgrammingError):
    """A referenced object does not exist, or already does."""


class InvalidInputError(ProgrammingError):
    """An argument was malformed or out of range."""


class ConfigurationError(ProgrammingError):
    """A setting was unknown, invalid, or not permitted."""


class ConstraintError(IntegrityError):
    """A key, NOT NULL, or CHECK constraint was violated."""


class ConversionError(DataError):
    """A value could not be converted to the requested type."""


# --- code to class ---------------------------------------------------------

# Every code that carries a distinct meaning gets an entry. Codes sharing a
# meaning share a class; the message keeps the detail.
_BY_NAME: dict[str, type[Error]] = {
    "API": InterfaceError,
    # I/O
    "IO_FILE_NOT_FOUND": IOError,
    "IO_READ_FAILURE": IOError,
    "IO_EOF": IOError,
    "IO_GENERAL": IOError,
    "IO_NETWORK": IOError,
    "IO_HTTP": IOError,
    # caller-supplied arguments
    "INPUT_INVALID": InvalidInputError,
    "INPUT_PARAMETER_INVALID": InvalidInputError,
    "INPUT_OUT_OF_RANGE": DataError,
    "INPUT_OBJECT_SIZE": DataError,
    # resources
    "RESOURCE_IN_USE": ProgrammingError,
    "RESOURCE_OUT_OF_MEMORY": OutOfMemoryError,
    "RESOURCE_CONNECTION": ConnectionError,
    "RESOURCE_DEPENDENCY": CatalogError,
    "RESOURCE_MISSING_EXTENSION": CatalogError,
    "RESOURCE_AUTOLOAD": IOError,
    # types and values
    "TYPE_CONVERSION": ConversionError,
    "TYPE_UNKNOWN": CatalogError,
    "TYPE_INVALID": DataError,
    "TYPE_MISMATCH": DataError,
    "TYPE_DECIMAL": DataError,
    "TYPE_DIVIDE_BY_ZERO": DataError,
    # statement lifecycle
    "QUERY_PARSER": ParserError,
    "QUERY_SYNTAX": ParserError,
    "QUERY_BINDER": ProgrammingError,
    "QUERY_PLANNER": ProgrammingError,
    "QUERY_OPTIMIZER": InternalError,
    "QUERY_EXPRESSION": ProgrammingError,
    "QUERY_EXECUTOR": InternalError,
    "QUERY_SCHEDULER": InternalError,
    "QUERY_NOT_IMPLEMENTED": NotSupportedError,
    "QUERY_PARAMETER_NOT_RESOLVED": ProgrammingError,
    "QUERY_PARAMETER_NOT_ALLOWED": ProgrammingError,
    # catalog and storage
    "DATABASE_CATALOG": CatalogError,
    "DATABASE_TRANSACTION": TransactionError,
    "DATABASE_CONSTRAINT": ConstraintError,
    "DATABASE_INDEX": ConstraintError,
    "DATABASE_SEQUENCE": CatalogError,
    "DATABASE_STATISTICS": InternalError,
    "DATABASE_SERIALIZATION": InternalError,
    # configuration
    "CONFIGURATION_SETTINGS": ConfigurationError,
    "CONFIGURATION_INVALID": ConfigurationError,
    "CONFIGURATION_PERMISSION": ConfigurationError,
    # runtime
    "RUNTIME_INTERNAL": InternalError,
    "RUNTIME_FATAL": FatalError,
    "RUNTIME_INTERRUPT": InterruptError,
    "RUNTIME_NULL_POINTER": InterfaceError,
}

_BY_CODE: dict[int, type[Error]] = {ERROR_CODES[name]: cls for name, cls in _BY_NAME.items() if name in ERROR_CODES}


def class_for_code(code: int) -> type[Error]:
    """The exception class for an engine error code.

    Unknown codes map to `DatabaseError` rather than raising: a newer engine
    may report a code this build has never seen, and losing the error to a
    lookup failure would be worse than reporting it imprecisely.
    """
    return _BY_CODE.get(code, DatabaseError)
