"""DuckDB PEP 517 and PEP 660 build backend.

This module wraps the scikit-build-core build backend because:
1. We need to be able to determine the version of the DuckDB submodule while building
   a source distribution, so that we can pass it in when building a wheel. The backend
   tries to figure out which duckdb version will be included in the sdist and saves
   the output
2. We want to use a custom version scheme with setuptools-scm, and PEP 621 provides no
   way to specify local code as a build-backend plugin. However, PEP 517 allows us to
   put our own build backend on the python path with the `build.backend-path` key. The
   side effect is that our version scheme is also on the path during the build.

Also see https://peps.python.org/pep-0517/#in-tree-build-backends.
"""
import sys
import os
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Union
from scikit_build_core.build import (
    build_wheel as skbuild_build_wheel,
    build_editable,
    build_sdist as skbuild_build_sdist,
    get_requires_for_build_wheel,
    get_requires_for_build_sdist,
    get_requires_for_build_editable,
    prepare_metadata_for_build_wheel,
    prepare_metadata_for_build_editable,
)

_DUCKDB_VERSION_FILENAME = "duckdb_version.txt"
_LOGGING_FORMAT = "[duckdb_pytooling.build_backend] {}"
_SKBUILD_CMAKE_OVERRIDE_GIT_DESCRIBE = "cmake.define.OVERRIDE_GIT_DESCRIBE"


def _log(msg: str, is_error: bool=False) -> None:
    print(_LOGGING_FORMAT.format(msg), flush=True, file=sys.stderr if is_error else sys.stdout)


def _in_git_repository() -> bool:
    return Path(".git").exists()


def _in_sdist() -> bool:
    """We assume we're in an sdist iff the duckdb version file exists and PKG-INFO exists."""
    return _version_file_path().exists() and Path("PKG-INFO").exists()


def _duckdb_submodule_path() -> Path:
    """Verify that the duckdb submodule is checked out and usable and return its path."""
    if not _in_git_repository():
        raise RuntimeError("Not in a git repository, no duckdb submodule present")
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
        raise RuntimeError("DuckDB submodule missing")

    duckdb_path = modules["duckdb"]
    # now check that the submodule is usable
    proc = subprocess.Popen(["git", "submodule", "status", duckdb_path], stdout=subprocess.PIPE)
    status, _ = proc.communicate()
    status = status.decode("ascii", "replace")
    for line in status.splitlines():
        if line.startswith("-"):
            raise RuntimeError(f"Duckdb submodule not initialized: {line}")
        if line.startswith("U"):
            raise RuntimeError(f"Duckdb submodule has merge conflicts: {line}")
        if line.startswith("+"):
            _log(f"WARNING: Duckdb submodule not clean: {line}")
    # all good
    return Path(duckdb_path)


def _duckdb_long_version(submodule_path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--long", "--match", "v*.*.*"],
            cwd=str(submodule_path),
            capture_output=True,
            text=True
        )
        # check_returncode() will raise an exception if the result's exit code != 0
        result.check_returncode()
    except FileNotFoundError:
        raise RuntimeError("git executable can't be found")
    return result.stdout.strip()


def _version_file_path() -> Path:
    package_dir = Path(__file__).parent
    return package_dir / _DUCKDB_VERSION_FILENAME


def _write_duckdb_long_version(long_version: str)-> None:
    """Write the given version string to a file in the same directory as this module."""
    _version_file_path().write_text(long_version, encoding="utf-8")


def _read_duckdb_long_version() -> str:
    """Read the given version string from a file in the same directory as this module."""
    return _version_file_path().read_text(encoding="utf-8").strip()


def build_sdist(sdist_directory: str, config_settings: Optional[Dict[str, Union[List[str],str]]] = None) -> str:
    """Build an sdist using the duckdb submoule"""
    if not _in_git_repository():
        raise RuntimeError("Not in a git repository, can't create an sdist")
    submodule_path = _duckdb_submodule_path()
    duckdb_version = _duckdb_long_version(submodule_path)
    _write_duckdb_long_version(duckdb_version)
    return skbuild_build_sdist(sdist_directory, config_settings=config_settings)


def build_wheel(
        wheel_directory: str,
        config_settings: Optional[Dict[str, List[str]|str]] = None,
        metadata_directory: Optional[str] = None,
) -> str:
    """Build a wheel either against the git submodule (if we're in a git repository) or from extracted sources
    (probably because we're in an sdist)."""
    if not _in_git_repository():
        if not _in_sdist():
            raise RuntimeError("Not in a git repository nor in an sdist, can't build a wheel")
        _log("Building duckdb wheel from sdist. Reading git describe override value.")
        config_settings = config_settings or {}
        duckdb_version = _read_duckdb_long_version()
        git_decribe_override_key = f"skbuild.{_SKBUILD_CMAKE_OVERRIDE_GIT_DESCRIBE}"
        if _SKBUILD_CMAKE_OVERRIDE_GIT_DESCRIBE in config_settings or git_decribe_override_key in config_settings:
            raise RuntimeError(f"{_SKBUILD_CMAKE_OVERRIDE_GIT_DESCRIBE} already set, can't build a wheel.")
        config_settings[git_decribe_override_key] = duckdb_version
        _log(f"{git_decribe_override_key} set to {duckdb_version}")
    else:
        _log(f"Building wheel from git repository")

    return skbuild_build_wheel(wheel_directory, config_settings=config_settings, metadata_directory=metadata_directory)


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