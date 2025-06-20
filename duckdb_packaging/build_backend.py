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
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from ._build_config import DuckDBBuildConfig

# Import scikit-build-core backend functions
from scikit_build_core.build import (
    build_wheel as skbuild_build_wheel,
    build_sdist as skbuild_build_sdist,
    build_editable as skbuild_build_editable,
    get_requires_for_build_wheel,
    get_requires_for_build_sdist,
    get_requires_for_build_editable,
    prepare_metadata_for_build_wheel,
    prepare_metadata_for_build_editable,
)

_DUCKDB_SCRIPTS_RELPATH = "scripts/"
_DUCKDB_BUILD_BACKEND_MODNAME = "package_build"

_SKBUILD_SDIST_INCLUDE_KEY = "sdist.include"
_SKBUILD_CMAKE_DEF_UB_INCLUDE_DIRS_KEY = "cmake.define.DUCKDB_UNITY_BUILD_INCLUDE_LIST"
_SKBUILD_CMAKE_DEF_UB_SOURCES_LIST_KEY = "cmake.define.DUCKDB_UNITY_BUILD_SOURCES_LIST"
_SKBUILD_CMAKE_DEF_ENABLED_EXTENSIONS = "cmake.define.DUCKDB_ENABLED_EXTENSIONS"
_SKBUILD_CMAKE_DEF_DISABLE_JEMALLOC = "cmake.define.DUCKDB_DISABLE_JEMALLOC"
_SKBUILD_CMAKE_DEF_CUSTOM_PLATFORM = "cmake.define.DUCKDB_CUSTOM_PLATFORM"

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
    conf = DuckDBBuildConfig.load()
    # if the target dir exists we remove all unity build files from it to make sure previously generated files do not
    # clutter the sources
    duckdb_target_dir_path = Path(conf.duckdb_target_dir).relative_to(Path().absolute())
    if duckdb_target_dir_path.exists():
        for file in duckdb_target_dir_path.iterdir():
            if file.is_file() and file.name.startswith("ub_"):
                file.unlink()
    # extract the duckdb sources into duckdb_target_dir
    source_files_list, include_dirs_list, all_sources_list = _build_package(
        conf.duckdb_target_dir,
        conf.extensions,
        conf.include_line_numbers,
        conf.unity_count,
        conf.short_paths
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


def _set_cmake_build_config(config_settings: Dict[str, List[str]|str]):
    """
    Update the scikit-build-core CMake config settings from DuckDBBuildConfig.

    Reads the project’s build configuration and injects the corresponding
    CMake definitions into the given `config_settings` dict:

      1. Disables jemalloc if `disable_jemalloc` is set.
      2. Exports a semicolon-separated list of enabled extensions (excluding jemalloc).
      3. Adds a custom platform name if one was specified.
    """
    conf = DuckDBBuildConfig.load()

    # 1. Explicitly disable jemalloc if configured (cmake will include jemalloc automatically if it is present)
    if conf.disable_jemalloc:
        _skbuild_config_add(_SKBUILD_CMAKE_DEF_DISABLE_JEMALLOC, "1", config_settings)
    # 2. Enable all configured extensions (minus jemalloc)
    extensions = list(set(conf.extensions)-{"jemalloc"})
    enabled_extensions = ";".join(extensions)
    _skbuild_config_add(_SKBUILD_CMAKE_DEF_ENABLED_EXTENSIONS, enabled_extensions, config_settings)
    # 3. Set a custom platform name
    if conf.custom_platform_name is not None:
        _skbuild_config_add(_SKBUILD_CMAKE_DEF_CUSTOM_PLATFORM, conf.custom_platform_name, config_settings)


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
        conf = DuckDBBuildConfig.load()
        duckdb_target_dir_path = Path(conf.duckdb_target_dir).relative_to(Path().absolute())
        assert duckdb_target_dir_path.exists(), \
            f"Can't build a wheel without duckdb sources (none found in {conf.duckdb_target_dir})"

    config_settings = config_settings or {}
    # add the source list and include dirs to the cmake config
    source_files_str, include_dirs_str = _read_sources_and_includes_files(duckdb_target_dir_path)
    _skbuild_config_add(_SKBUILD_CMAKE_DEF_UB_INCLUDE_DIRS_KEY, include_dirs_str, config_settings)
    _skbuild_config_add(_SKBUILD_CMAKE_DEF_UB_SOURCES_LIST_KEY, source_files_str, config_settings)
    # set cmake build config
    _set_cmake_build_config(config_settings)
    # hand off to scikit-build-core
    return skbuild_build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(
        wheel_directory: str,
        config_settings: Optional[Dict[str, List[str]|str]] = None,
        metadata_directory: Optional[str] = None,
) -> str:
    # set cmake build config
    config_settings = config_settings or {}
    _set_cmake_build_config(config_settings)
    return skbuild_build_editable(wheel_directory, config_settings, metadata_directory)


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