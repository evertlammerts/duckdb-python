"""Loads and validates the PEP 517/660 build backend configuration from `pyproject.toml`.

This module reads the `[tool.duckdb]` section of `pyproject.toml`, extracts settings necessary for building source
distributions, wheels and editable insgtalls, and validates them.  The configuration is cached on first load, so it's
fine to call `load()` multiple times.

Example:
    >>> from duckdb_packaging._build_config import DuckDBBuildConfig
    >>> cfg = DuckDBBuildConfig.load()
    >>> cfg.duckdb_target_dir
    'extracted/duckdb_src'
    >>> cfg.extensions
    ['core_functions', 'json', 'parquet', 'icu']
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any, Optional
from ._tomllib import load as toml_load

# Build config for all build types
_CONF_TOOL_NAME = "duckdb"
_CONF_EXTENSIONS = "extensions"
_CONF_DISABLE_JEMALLOC = "disable_jemalloc"
_CONF_CUSTOM_PLATFORM = "custom_platform_name"
# Build config for sdists
_CONF_SDIST_NAME = "sdist"
_CONF_SDIST_DUCKDB_SRC_TARGET = "duckdb_src_target"
_CONF_SDIST_INCLUDE_LINE_NUMBERS = "include_line_numbers"
_CONF_SDIST_UNITY_COUNT = "unity_count"
_CONF_SDIST_SHORT_PATHS = "short_paths"


@dataclass(frozen=True)
class DuckDBBuildConfig:
    """All of the settings needed to build a wheel or sdist."""

    duckdb_target_dir: str
    """Target directory of duckdb source when shipping an sdist."""
    extensions: List[str]
    """List of extensions to include in build."""
    include_line_numbers: bool = False
    """Whether to include line numbers in sdist when building wheels."""
    unity_count: int = 32
    """Amount of unity build files (currently unused)."""
    short_paths: bool = False
    """Whether to use short paths in unity build."""
    disable_jemalloc: bool = False
    """Whether to disable jemalloc. Note that cmake will only enable jemalloc if its present."""
    custom_platform_name: Optional[str] = None
    """Custom platform name to use when building wheels."""

    @classmethod
    @lru_cache(maxsize=1)
    def load(cls) -> DuckDBBuildConfig:
        """Load and validate all configuration; cached so only runs once."""
        config = _pyproject_config()
        sdist_config = config.get(_CONF_SDIST_NAME, {})

        # 1. get and validate duckdb_src_target
        target_dir = sdist_config.get(_CONF_SDIST_DUCKDB_SRC_TARGET, "")
        assert target_dir != "", (
            f"{_CONF_SDIST_DUCKDB_SRC_TARGET} must be set to a directory where we can store duckdb source files"
        )
        target_path = Path(target_dir).absolute()
        if target_path.exists():
            assert target_path.is_dir(), f"{target_path} is not a directory"
        target_parent_path = target_path.parent
        if not target_parent_path.exists():
            target_parent_path.mkdir(parents=True)
        assert target_parent_path.is_dir(), f"{target_parent_path} is not a directory"
        target_dir = str(target_path)

        # 2. include_line_numbers
        include_line_numbers = bool(
            sdist_config.get(_CONF_SDIST_INCLUDE_LINE_NUMBERS, cls.include_line_numbers)
        )

        # 3. unity_count
        unity_count = int(
            sdist_config.get(_CONF_SDIST_UNITY_COUNT, cls.unity_count)
        )

        # 4. short_paths
        short_paths = bool(
            sdist_config.get(_CONF_SDIST_SHORT_PATHS, cls.short_paths)
        )

        # 5. extensions
        extensions = config.get(_CONF_EXTENSIONS, [])
        assert isinstance(extensions, list), f"{_CONF_EXTENSIONS} is not a list"

        # 6. jemalloc
        disable_jemalloc = bool(
            sdist_config.get(_CONF_DISABLE_JEMALLOC, cls.disable_jemalloc)
        )

        # 7. custom_platform_name
        custom_platform_name = sdist_config.get(_CONF_SDIST_DUCKDB_SRC_TARGET, cls.custom_platform_name)

        return cls(
            duckdb_target_dir=target_dir,
            include_line_numbers=include_line_numbers,
            unity_count=unity_count,
            short_paths=short_paths,
            extensions=extensions,
            disable_jemalloc=disable_jemalloc,
            custom_platform_name=custom_platform_name,
        )

def _pyproject_config() -> Dict[str, Any]:
    """Load pyproject.toml configuration file"""
    pyproject_path = Path("pyproject.toml")
    with pyproject_path.open("rb") as ft:
        pyproject = toml_load(ft)
    return pyproject.get("tool", {}).get(_CONF_TOOL_NAME, {})