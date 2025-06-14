"""Minimal custom build backend that allows us to prepare for building an sdist.
"""
import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Callable, List, Tuple

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
from duckdb_pytooling import _tomllib

_DUCKDB_SCRIPTS_RELPATH = "scripts/"
_DUCKDB_BUILD_BACKEND_MODNAME = "package_build"

_CONF_TOOL_NAME = "duckdb"
_CONF_EXTENSIONS = "extensions"
_CONF_SDIST_NAME = "sdist"
_CONF_SDIST_DUCKDB_SRC_TARGET = "duckdb_src_target"
_CONF_SDIST_INCLUDE_LINE_NUMBERS = "include_line_numbers"
_CONF_SDIST_UNITY_COUNT = "unity_count"
_CONF_SDIST_SHORT_PATHS = "short_paths"

_SKBUILD_SDIST_INCLUDE_KEY = "sdist.include"


def _duckdb_submodule_path() -> str:
    """ Verify that the duckdb submodule is checked out and usable, and return its path."""
    assert Path(".git").exists(), "Not in a git repository"
    # search the duckdb submodule
    gitmodules_path = Path(".gitmodules")
    modules = dict()
    with gitmodules_path.open("r") as f:
        cur_module_path = None
        cur_module_reponame = None
        for line in f:
            if line.strip().startswith("[submodule"):
                if cur_module_reponame is not None and cur_module_path is not None:
                    modules[cur_module_reponame] = cur_module_path
                    cur_module_reponame = None
                    cur_module_path = None
            elif line.strip().startswith("path"):
                cur_module_path = line.split('=')[-1].strip()
            elif line.strip().startswith("url"):
                basename = os.path.basename(line.split('=')[-1].strip())
                cur_module_reponame = basename[:-4] if basename.endswith(".git") else basename
        if cur_module_reponame is not None and cur_module_path is not None:
            modules[cur_module_reponame] = cur_module_path

    if "duckdb" not in modules:
        raise ValueError("DuckDB submodule missing")

    duckdb_path = modules["duckdb"]
    # now check that the submodule is usable
    proc = subprocess.Popen(["git", "submodule", "status", duckdb_path], stdout=subprocess.PIPE)
    status, _ = proc.communicate()
    status = status.decode("ascii", "replace")
    for line in status.splitlines():
        if line.startswith("-"):
            raise ValueError(f"Duckdb submodule not initialized: {line}")
        if line.startswith("U"):
            raise ValueError(f"Duckdb submodule has merge conflicts: {line}")
        if line.startswith("+"):
            print(f"WARNING: Duckdb submodule not clean: {line}")
    # all good
    return duckdb_path

def _pyproject_config() -> Dict[str, Any]:
    """Load pyproject.toml configuration file"""
    pyproject_path = Path("pyproject.toml")
    with pyproject_path.open("rb") as ft:
        pyproject = _tomllib.load(ft)
    return pyproject.get("tool", {}).get(_CONF_TOOL_NAME, {})

def _build_package(target_dir, extensions, linenumbers, unity_count, short_paths) -> Tuple[
    List[str], List[str], List[str]
]:
    """Function that loads and wraps duckdb's `package_build.build_package(...)` function."""
    # resolve and load the build_package function
    duckdb_path = _duckdb_submodule_path()
    module_path = Path(duckdb_path) / _DUCKDB_SCRIPTS_RELPATH
    print(module_path.absolute())
    sys.path.append(str(module_path.absolute()))
    mod = importlib.import_module(_DUCKDB_BUILD_BACKEND_MODNAME)
    # return the sources, include_list, and original_sources
    return mod.build_package(
        target_dir,
        extensions,
        linenumbers=linenumbers,
        unity_count=unity_count,
        short_paths=short_paths
    )


def _sdist_config() -> Tuple[str, bool, int, bool, List[str]]:
    """Load and validate all configuration needed for building an sdist."""
    config = _pyproject_config()
    sdist_config = config.get(_CONF_SDIST_NAME, {})
    # 1. get and validate duckdb_src_target
    duckdb_target_dir = sdist_config.get(_CONF_SDIST_DUCKDB_SRC_TARGET, "")
    assert duckdb_target_dir != "", \
        f"{_CONF_SDIST_DUCKDB_SRC_TARGET} must be set to a directory where we can store duckdb source files"
    duckdb_target_dir_path = Path(duckdb_target_dir).absolute()
    if duckdb_target_dir_path.exists():
        assert duckdb_target_dir_path.is_dir(), f"{duckdb_target_dir} is not a directory"
    duckdb_target_dir_parent_path = duckdb_target_dir_path.parent
    if not duckdb_target_dir_parent_path.exists():
        duckdb_target_dir_parent_path.mkdir(parents=True)
    assert duckdb_target_dir_parent_path.is_dir(), f"{duckdb_target_dir_parent_path} is not a directory"
    duckdb_target_dir = str(duckdb_target_dir_path)
    # 2. get include_line_numbers
    include_line_numbers = sdist_config.get(_CONF_SDIST_INCLUDE_LINE_NUMBERS, False)
    # 3. get unity_count
    unity_count = sdist_config.get(_CONF_SDIST_UNITY_COUNT, 32)
    # 4. get short_paths
    short_paths = sdist_config.get(_CONF_SDIST_SHORT_PATHS, False)
    # 5. get and validate extensions
    extensions = sdist_config.get(_CONF_EXTENSIONS, [])
    assert isinstance(extensions, list), f"{_CONF_EXTENSIONS} is not a list"
    return duckdb_target_dir, include_line_numbers, unity_count, short_paths, extensions


def build_sdist(sdist_directory: str, config_settings: Optional[Dict[str, Any]] = None) -> str:
    """Use build_package to get the duckdb source list, then add the includes to scikit-build-core's config
    settings."""
    duckdb_target_dir, include_line_numbers, unity_count, short_paths, extensions = _sdist_config()
    # get duckdb sources
    source_list, include_list, _ = _build_package(
        duckdb_target_dir,
        extensions,
        include_line_numbers,
        unity_count,
        short_paths
    )
    # Amend the include list with the generated sources
    config_settings = config_settings or {}
    skbuild_full_sdist_include_key = "skbuild." + _SKBUILD_SDIST_INCLUDE_KEY
    sdist_include = config_settings.get(
        _SKBUILD_SDIST_INCLUDE_KEY, config_settings.get(skbuild_full_sdist_include_key, [])
    )
    sdist_include.append(str(Path(duckdb_target_dir).relative_to(Path('.').absolute())) + "/**")
    config_settings[skbuild_full_sdist_include_key] = sdist_include
    # Hand off to scikit-build-core
    print(config_settings)
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