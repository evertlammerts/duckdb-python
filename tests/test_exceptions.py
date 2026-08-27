"""The exception hierarchy covers the engine's code space and stays PEP 249."""

from __future__ import annotations

import pytest

from duckdb import exceptions
from duckdb._error_codes import ERROR_CODES

# PEP 249 section "Exceptions": these names, and these parent relationships.
PEP249_HIERARCHY = [
    ("Error", Exception),
    ("InterfaceError", "Error"),
    ("DatabaseError", "Error"),
    ("DataError", "DatabaseError"),
    ("OperationalError", "DatabaseError"),
    ("IntegrityError", "DatabaseError"),
    ("InternalError", "DatabaseError"),
    ("ProgrammingError", "DatabaseError"),
    ("NotSupportedError", "DatabaseError"),
]


def test_every_engine_code_maps_to_a_class() -> None:
    # An engine bump that adds a code should fail here rather than silently
    # start reporting it as a bare DatabaseError.
    unmapped = sorted(set(ERROR_CODES) - set(exceptions._BY_NAME))
    assert not unmapped, f"engine codes with no exception class: {unmapped}"


def test_no_mapping_refers_to_a_code_the_engine_dropped() -> None:
    stale = sorted(set(exceptions._BY_NAME) - set(ERROR_CODES))
    assert not stale, f"mappings for codes the engine no longer defines: {stale}"


@pytest.mark.parametrize(("name", "parent"), PEP249_HIERARCHY)
def test_pep249_hierarchy(name: str, parent: str | type) -> None:
    cls = getattr(exceptions, name)
    expected = parent if isinstance(parent, type) else getattr(exceptions, parent)
    assert issubclass(cls, expected)


def test_every_leaf_descends_from_error() -> None:
    for cls in exceptions._BY_NAME.values():
        assert issubclass(cls, exceptions.Error), f"{cls.__name__} is outside the hierarchy"


def test_unknown_code_degrades_instead_of_raising() -> None:
    # A newer engine may report a code this build has never seen. Losing the
    # error to a lookup failure would be worse than reporting it imprecisely.
    assert exceptions.class_for_code(999_999) is exceptions.DatabaseError


def test_warning_is_not_an_error() -> None:
    # PEP 249 keeps Warning outside the Error tree, so `except Error` must not
    # swallow it.
    assert not issubclass(exceptions.Warning, exceptions.Error)
