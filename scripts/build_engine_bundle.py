"""Build a DuckDB checkout into a directory the build takes as DUCKDB_ROOT.

The output mirrors an unpacked duckdb-shared-libs archive: the library and the C headers side by side.
This is how an unreleased engine gets under the client; released engines come from scripts/engine.py fetch.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# The C headers the shared-libs archives carry.
HEADER_FILES = (
    "src/include/duckdb.h",
    "src/include/duckdb_v2.h",
    "src/include/duckdb_extension.h",
    "src/include/duckdb_extension_v2.h",
)
# Absent from older checkouts, and nothing in the build reads it.
OPTIONAL_HEADER_FILES = ("src/include/duckdb_static_extension.h",)

# The runtime library, and on Windows also the import library needed to link.
RUNTIME_NAMES = ("libduckdb.dylib", "libduckdb.so", "duckdb.dll")
IMPORT_NAMES = ("duckdb.lib",)


def run(cmd: list[str]) -> None:
    """Echo a command and run it, failing the script if it fails."""
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True)


def find_first(root: Path, names: tuple[str, ...]) -> Path | None:
    """First file under `root` matching any of `names`, in the order given."""
    for name in names:
        hits = sorted(root.rglob(name))
        if hits:
            return hits[0]
    return None


def build_engine(src: Path, build_dir: Path, build_type: str, duckdb_version: str | None) -> None:
    """Configure and build libduckdb with CMake directly, because DuckDB's Makefile is not portable to Windows."""
    configure = [
        "cmake",
        "-S",
        str(src),
        "-B",
        str(build_dir),
        f"-DCMAKE_BUILD_TYPE={build_type}",
        "-DBUILD_SHELL=0",
        "-DBUILD_UNITTESTS=0",
        "-DENABLE_EXTENSION_AUTOLOADING=1",
        "-DENABLE_EXTENSION_AUTOINSTALL=1",
    ]
    # Default to what the published shared-libs archives link in, so the output behaves like one.
    core_extensions = os.environ.get("CORE_EXTENSIONS", "autocomplete;core_functions;icu;json;parquet")
    if core_extensions:
        configure.append(f"-DCORE_EXTENSIONS={core_extensions}")
    if duckdb_version:
        # DuckDB names itself by this and finds extensions under that name, so a checkout without tags installs none.
        configure.append(f"-DDUCKDB_EXPLICIT_VERSION={duckdb_version}")
    else:
        # The CMake cache keeps the last value, so a new commit would otherwise build under the old version name.
        configure.append("-UDUCKDB_EXPLICIT_VERSION")
    if sys.platform == "win32":
        # With Ninja on PATH the Windows runners pick a MinGW gcc that rejects DuckDB's -march flags.
        if platform := os.environ.get("CMAKE_GENERATOR_PLATFORM"):
            configure += ["-A", platform]
    elif shutil.which("ninja"):
        configure += ["-G", "Ninja"]
    run(configure)
    run(["cmake", "--build", str(build_dir), "--config", build_type, "--parallel", str(os.cpu_count() or 2)])


def exports_v2(lib: Path) -> bool | None:
    """Whether the library exports the V2 C API. None when no symbol reader could look, which is normal on Windows."""
    if sys.platform != "win32" and shutil.which("nm"):
        out = subprocess.run(["nm", "-g", str(lib)], capture_output=True, text=True, check=False)
        if out.returncode == 0:
            return "duckdb_v2_" in out.stdout
    if sys.platform == "win32" and shutil.which("dumpbin"):
        out = subprocess.run(["dumpbin", "/exports", str(lib)], capture_output=True, text=True, check=False)
        if out.returncode == 0:
            return "duckdb_v2_" in out.stdout
    return None


def main() -> int:
    """Assemble the engine directory. Returns a process exit code."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path, help="a DuckDB checkout")
    ap.add_argument("output", type=Path, help="engine directory to create")
    ap.add_argument("--build-type", default="Release")
    ap.add_argument(
        "--duckdb-version", default=None, help="the version name the engine reports, as core's nightly named it"
    )
    args = ap.parse_args()

    src, out = args.source.resolve(), args.output.resolve()
    if not (src / "src" / "include" / "duckdb_v2.h").is_file():
        sys.exit(f"{src} has no src/include/duckdb_v2.h: not a DuckDB checkout with the V2 C API")

    build_dir = src / "build" / args.build_type.lower()
    build_engine(src, build_dir, args.build_type, args.duckdb_version)
    runtime = find_first(build_dir, RUNTIME_NAMES)
    if runtime is None:
        sys.exit(f"no engine library produced under {build_dir}")

    match exports_v2(runtime):
        case False:
            sys.exit(f"{runtime} exports no duckdb_v2_* symbols: pre-V2 engine")
        case None:
            print(
                f"note: cannot verify V2 exports in {runtime} on this platform; "
                "DuckDBCppApi.cmake's link probe is the real gate",
                file=sys.stderr,
            )

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # Plain copy, not copy2: preserved mtimes can predate a consumer's build objects, which ninja reads as up to date.
    for rel in HEADER_FILES:
        shutil.copy(src / rel, out / Path(rel).name)
    for rel in OPTIONAL_HEADER_FILES:
        if (src / rel).is_file():
            shutil.copy(src / rel, out / Path(rel).name)
    # Copy the real file, not the symlink standing in for it.
    shutil.copy(runtime.resolve(), out / runtime.name)
    # Windows links against the import library and loads the DLL, so it needs both.
    if imp := find_first(build_dir, IMPORT_NAMES):
        shutil.copy(imp.resolve(), out / imp.name)

    sha = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"engine: {out} ({size / 1e6:.0f}MB), built from {sha.stdout.strip()[:12] or 'unknown'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
