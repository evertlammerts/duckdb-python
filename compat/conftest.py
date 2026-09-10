"""Test setup for the previous duckdb package's tests, copied unchanged under suite/ and never edited here."""

import os

import pytest

#: Copied tests this package deliberately does not satisfy, keyed by test name and explained in compat/BEHAVIOR.md.
DIVERGENT = {
    **dict.fromkeys(
        [
            "test_fetch_dict_coverage[test_case18]",
            "test_fetch_dict_coverage[test_case19]",
            "test_fetch_dict_coverage[test_case21]",
            "test_fetch_dict_coverage[test_case22]",
            "test_fetch_dict_coverage[test_case24]",
        ],
        "a temporal beyond Python's range raises ConversionError here; the old client degraded it to text",
    ),
    "test_fetch_dict_key_not_hashable[VARCHAR[]]": (
        "a MAP with unhashable keys comes back as (key, value) pairs, not the old key/value dict of lists"
    ),
    "test_comment_is_harmless": (
        "a -- comment in a shorthand aggregate operand is not stripped; it fails loudly as an unknown column"
    ),
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark the copied tests whose expectation this package deliberately does not meet."""
    for item in items:
        reason = DIVERGENT.get(item.name)
        if reason is not None:
            item.add_marker(pytest.mark.xfail(reason=f"{reason} (compat/BEHAVIOR.md)", strict=False))


# Set this to run the copied tests against duckdb.compat instead of the default API.
if os.environ.get("DUCKDB_COMPAT_FACADE"):
    import duckdb
    from duckdb import compat

    # Patching from __all__ keeps this in step with duckdb.compat as it grows.
    for _name in (*compat.__all__, "__formatted_python_version__"):
        setattr(duckdb, _name, getattr(compat, _name))
