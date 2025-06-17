"""Custom PEP 517/660 build backend that prepares sdist and wheel builds before handing off to scikit-build-core.

The DuckDB Python package supports all build types:
- Editable install: **[no custom logic]** Compiles DuckDB sources from the git submodule using DuckDB's CMake config.
- Source distribution (sdist): **[build_sdist]** Extracts DuckDB sources from the git submodule and includes only these
    sources in the resulting tarball, including two files with the list of sources files and the list of include paths.
    Note that cmake doesn't run for this step.
- Wheel build: **[build_wheel]** Wheels can be built either from the git submodule or from pre-extracted sources. The
    build backend (see build_wheel) assumes it should use the git submodule if it is executing from within a git
    repository.
"""
import importlib
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

# Import scikit-build-core backend functions
from scikit_build_core.build import (
    build_wheel as skbuild_build_wheel,
    build_sdist as skbuild_build_sdist,
    build_editable,
    get_requires_for_build_wheel,
    get_requires_for_build_sdist,
    get_requires_for_build_editable,
    prepare_metadata_for_build_wheel,
    prepare_metadata_for_build_editable,
)
from ._tomllib import load as toml_load

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
_SKBUILD_CMAKE_VAR_UB_INCLUDE_DIRS_KEY = "cmake.define.DUCKDB_UNITY_BUILD_INCLUDE_LIST"
_SKBUILD_CMAKE_VAR_UB_SOURCES_LIST_KEY = "cmake.define.DUCKDB_UNITY_BUILD_SOURCES_LIST"

_INCLUDE_DIRS_FILENAME = "duckdb_include_dirs.txt"
_SOURCE_FILES_FILENAME = "duckdb_source_files.txt"

_LOGGING_FORMAT = "[duckdb_pytooling.build_backend] {}"


def _log(msg: str, is_error: bool=False) -> None:
    print(_LOGGING_FORMAT.format(msg), flush=True, file=sys.stderr if is_error else sys.stdout)


def _in_git_repository() -> bool:
    return Path(".git").exists()


def _duckdb_submodule_path() -> str:
    """Verify that the duckdb submodule is checked out and usable and return its path."""
    assert _in_git_repository(), "Not in a git repository, no duckdb submodule present"
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
            _log(f"WARNING: Duckdb submodule not clean: {line}")
    # all good
    return duckdb_path

def _pyproject_config() -> Dict[str, Any]:
    """Load pyproject.toml configuration file"""
    pyproject_path = Path("pyproject.toml")
    with pyproject_path.open("rb") as ft:
        pyproject = toml_load(ft)
    return pyproject.get("tool", {}).get(_CONF_TOOL_NAME, {})

def _build_package(target_dir, extensions, linenumbers, unity_count, short_paths) -> Tuple[
    List[str], List[str], List[str]
]:
    """Function that loads and wraps duckdb's `package_build.build_package(...)` function."""
    # resolve and load the build_package function
    duckdb_path = _duckdb_submodule_path()
    module_path = Path(duckdb_path) / _DUCKDB_SCRIPTS_RELPATH
    sys.path.append(str(module_path.absolute()))
    mod = importlib.import_module(_DUCKDB_BUILD_BACKEND_MODNAME)
    # return the sources, include_list, and original_sources
    sources_list, include_dirs_list, all_sources_list = mod.build_package(
        target_dir,
        extensions,
        linenumbers=linenumbers,
        unity_count=unity_count,
        short_paths=short_paths,
        folder_name="" # disable prefixing sources
    )
    # unity build files paths are prefixed with target_dir so we strip the prefix.
    target_dir_path = Path(target_dir).absolute()
    ub_to_rel = lambda p: p if not Path(p).is_absolute() else str(Path(p).relative_to(target_dir_path))
    sources_list = [ub_to_rel(p) for p in sources_list]
    return sources_list, include_dirs_list, all_sources_list


@lru_cache(maxsize=1)
def _duckdb_build_config() -> Tuple[str, bool, int, bool, List[str]]:
    """Load and validate all configuration needed for building a wheel or sdist. The return value is cached so the
    config can be read at any time by any caller in this module."""
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
    extensions = config.get(_CONF_EXTENSIONS, [])
    assert isinstance(extensions, list), f"{_CONF_EXTENSIONS} is not a list"
    return duckdb_target_dir, include_line_numbers, unity_count, short_paths, extensions


def _skbuild_config_add(key: str, value: list | str, config_settings: Dict[str, List[str]|str]) -> None:
    """Add the given value to the given key in the config settings for skbuild. Only for list and string-typed
    settings.

    If the key is not found, will add the setting under the key while prepending the "skbuild." prefix.
    """
    assert config_settings is not None, "config_settings must not be None"
    store_key = key if key in config_settings else "skbuild." + key
    if isinstance(value, str):
        base_val = f"{config_settings[store_key]};" if store_key in config_settings else ""
        config_settings[store_key] = f"{base_val}{value}"
    elif isinstance(value, list):
        new_val = config_settings.get(store_key, [])
        new_val.extend(value)
        config_settings[store_key] = new_val
    else:
        raise ValueError(f"{store_key} is not a string or list")

def _write_sources_and_includes_files(
        source_files: List[str], include_dirs: List[str], target_dir_path: Path
)-> None:
    """Write the given source files list and include dirs list as semicolon separated lists to file in the given target
    directory."""
    with open(str(target_dir_path / _SOURCE_FILES_FILENAME), mode="w") as f:
        f.write(";".join(source_files))
    with open(str(target_dir_path / _INCLUDE_DIRS_FILENAME), mode="w") as f:
        f.write(";".join(include_dirs))


def _read_sources_and_includes_files(target_dir_path: Path) -> Tuple[str, str]:
    """Read the source files list and include dirs list as a semicolon separated lists from files in the given target
    directory."""
    with open(str(target_dir_path / _SOURCE_FILES_FILENAME)) as f:
        source_files_list = f.read().strip()
    with open(str(target_dir_path / _INCLUDE_DIRS_FILENAME)) as f:
        include_dirs_list = f.read().strip()
    return source_files_list, include_dirs_list


def _extracted_duckdb_sources_path() -> Path:
    """Get the path to duckdb's extracted sources.

    Note: if the target directory exists, then the unity build files (ub_*) will be removed.
    """
    # get the settings
    abs_duckdb_target_dir, include_line_numbers, unity_count, short_paths, extensions = _duckdb_build_config()
    # if the target dir exists we remove all unity build files from it to make sure previously generated files do not
    # clutter the sources
    duckdb_target_dir_path = Path(abs_duckdb_target_dir).relative_to(Path().absolute())
    if duckdb_target_dir_path.exists():
        for file in duckdb_target_dir_path.iterdir():
            if file.is_file() and file.name.startswith("ub_"):
                file.unlink()
    # extract the duckdb sources into duckdb_target_dir
    source_files_list, include_dirs_list, all_sources_list = _build_package(
        abs_duckdb_target_dir,
        extensions,
        include_line_numbers,
        unity_count,
        short_paths
    )
    # save the source files list and include dirs list relative to project root (will overwrite if the file exists)
    source_files_list_relative_to_root = [str(duckdb_target_dir_path / s) for s in source_files_list]
    include_dirs_list_relative_to_root = [str(duckdb_target_dir_path / i) for i in include_dirs_list]
    _write_sources_and_includes_files(
        source_files_list_relative_to_root,
        include_dirs_list_relative_to_root,
        duckdb_target_dir_path
    )
    return duckdb_target_dir_path


def build_sdist(sdist_directory: str, config_settings: Optional[Dict[str, List[str]|str]] = None) -> str:
    """Build a source dist including duckdb's extracted sources."""
    _log("Building duckdb sdist")
    duckdb_target_dir_path = _extracted_duckdb_sources_path()
    config_settings = config_settings or {}
    # glob for the included duckdb sources in the sdist
    unity_build_sources_glob = str(duckdb_target_dir_path / "**")
    _skbuild_config_add(_SKBUILD_SDIST_INCLUDE_KEY, unity_build_sources_glob, config_settings)
    # hand off to scikit-build-core
    return skbuild_build_sdist(sdist_directory, config_settings)


def build_wheel(
        wheel_directory: str,
        config_settings: Optional[Dict[str, List[str]|str]] = None,
        metadata_directory: Optional[str] = None,
) -> str:
    """Build a wheel either against the git submodule (if we're in a git repository) or from extracted sources
    (probably because we're in an sdist)."""
    if _in_git_repository():
        # if we're in a repo then we extract the sources from the submodule
        _log("Building duckdb wheel using git submodule")
        duckdb_target_dir_path = _extracted_duckdb_sources_path()
    else:
        # otherwise we just get the target dir from the config
        _log("Building duckdb wheel using extracted sources")
        abs_duckdb_target_dir, _, _ , _, _ = _duckdb_build_config()
        duckdb_target_dir_path = Path(abs_duckdb_target_dir).relative_to(Path().absolute())
        assert duckdb_target_dir_path.exists(), \
            f"Can't build a wheel without duckdb sources (none found in {abs_duckdb_target_dir})"

    config_settings = config_settings or {}
    # add the source list and include dirs to the cmake config
    source_files_str, include_dirs_str = _read_sources_and_includes_files(duckdb_target_dir_path)
    _skbuild_config_add(_SKBUILD_CMAKE_VAR_UB_INCLUDE_DIRS_KEY, include_dirs_str, config_settings)
    _skbuild_config_add(_SKBUILD_CMAKE_VAR_UB_SOURCES_LIST_KEY, source_files_str, config_settings)
    # hand off to scikit-build-core
    return skbuild_build_wheel(wheel_directory, config_settings, metadata_directory)


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