"""DuckDB Python packaging, versioning, and build tooling.

This module wraps the scikit-build-core build backend because we want to use a custom
version scheme with setuptools-scm, and PEP 621 provides no way to specify local code
as a build-backend plugin. However, PEP 517 allows us to put our own build backend on
the python path with the `build.backend-path` key. The side effect is that out version
scheme is also on the path during the build.

Also see https://peps.python.org/pep-0517/#in-tree-build-backends.
"""
from scikit_build_core.build import (
    build_wheel,
    build_sdist,
    build_editable,
    get_requires_for_build_wheel,
    get_requires_for_build_sdist,
    get_requires_for_build_editable,
    prepare_metadata_for_build_wheel,
    prepare_metadata_for_build_editable,
)

__all__ = [
    "build_wheel",
    "build_sdist",
    "build_editable",
    "get_requires_for_build_wheel",
    "get_requires_for_build_sdist",
    "get_requires_for_build_editable",
    "prepare_metadata_for_build_wheel",
    "prepare_metadata_for_build_editable",
]