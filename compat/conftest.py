"""Ours, not adopted: the harness layer over the verbatim suite.

With DUCKDB_COMPAT_FACADE set, `duckdb.connect` and the old module-level
names are replaced from `duckdb.compat` before the adopted tests run,
previewing what a drop-in `connect()` would score. And where the old suite
asserts behavior this client deliberately does not have, the test is marked
xfail from here with a reason naming its entry in BEHAVIOR.md. The verbatim
files under suite/ are never edited; everything lives here, one directory
up.
"""

import os

import pytest

#: Adopted tests asserting a recorded divergence, by test name. Each reason
#: names its entry in compat/BEHAVIOR.md.
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
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark the adopted tests that assert a recorded divergence."""
    for item in items:
        reason = DIVERGENT.get(item.name)
        if reason is not None:
            item.add_marker(pytest.mark.xfail(reason=f"{reason} (compat/BEHAVIOR.md)", strict=False))


if os.environ.get("DUCKDB_COMPAT_FACADE"):
    import duckdb
    from duckdb import compat

    # Everything the face exports IS the old module surface; patching from
    # __all__ keeps this list from trailing the face as it grows.
    for _name in (*compat.__all__, "__formatted_python_version__"):
        setattr(duckdb, _name, getattr(compat, _name))
