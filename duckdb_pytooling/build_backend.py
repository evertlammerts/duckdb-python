"""Minimal custom build backend that allows us to prepare for building an sdist.
"""

from typing import Any, Dict, Optional

# Import scikit-build-core backend functions
from scikit_build_core.build import (
    build_wheel,
    build_sdist as skbuild_build_sdist,
    build_editable,
    get_requires_for_build_wheel,
    get_requires_for_build_sdist,
    get_requires_for_build_editable,
    prepare_metadata_for_build_wheel,
    prepare_metadata_for_build_editable,
)


def build_sdist(
        sdist_directory: str,
        config_settings: Optional[Dict[str, Any]] = None,
) -> str:
    """Build an sdist with custom pre-processing."""

    # Your custom logic here
    print("Running custom sdist hook...")
    # TODO: Add your custom sdist logic

    # Hand off to scikit-build-core
    return skbuild_build_sdist(sdist_directory, config_settings)


# Re-export all other functions as-is
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