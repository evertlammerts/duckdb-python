"""Assemble a self-contained DuckDB engine bundle.

The bundle holds the engine library plus the C++ API and the two headers it
needs, so that pointing both DUCKDB_CPP_DIR and DUCKDB_ROOT at it is enough to
build the extension. CI produces one per platform and passes it to the wheel
jobs as an artifact.

Reconstructs the scripts/package_cpp_api.py referenced by DuckDB's
tools/cpp/example/README.md, which did not land in main. Delete this once it
does.

    build_engine_bundle.py <duckdb-checkout> <output-dir> [--build-type Release]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Only what duckdb_cpp.cpp pulls in: duckdb_cpp.hpp, duckdb_extension_v2.h, and
# duckdb_v2.h, which itself includes nothing but libc headers.
CPP_API_FILES = ("tools/cpp/duckdb_cpp.hpp", "tools/cpp/duckdb_cpp.cpp")
HEADER_FILES = ("src/include/duckdb_v2.h", "src/include/duckdb_extension_v2.h")
CMAKE_FILE = "tools/cpp/cmake/DuckDBCppApi.cmake"

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


def build_engine(src: Path, build_dir: Path, build_type: str) -> None:
    """Configure and build libduckdb through CMake directly.

    DuckDB's Makefile is a wrapper around exactly this, but it is not portable
    to the Windows runners, so we call CMake ourselves.
    """
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
    if core_extensions := os.environ.get("CORE_EXTENSIONS"):
        configure.append(f"-DCORE_EXTENSIONS={core_extensions}")
    if sys.platform == "win32":
        # Let CMake pick the Visual Studio generator. With Ninja on PATH the
        # Windows runners resolve the compiler to a MinGW gcc that rejects
        # DuckDB's -march flags ("bad value 'armv8-a' for '-march=' switch").
        # DuckDB's own Makefile passes the platform the same way.
        if platform := os.environ.get("CMAKE_GENERATOR_PLATFORM"):
            configure += ["-A", platform]
    elif shutil.which("ninja"):
        configure += ["-G", "Ninja"]
    run(configure)
    run(["cmake", "--build", str(build_dir), "--config", build_type, "--parallel", str(os.cpu_count() or 2)])


def exports_v2(lib: Path) -> bool | None:
    """Whether the library exports the V2 C API. None when undeterminable.

    Only ever authoritative when a real symbol reader answered. `nm` does not
    read PE reliably and `dumpbin` needs a developer shell, so on Windows this
    usually returns None. That is deliberate: the real gate is the link probe
    in DuckDBCppApi.cmake, which compiles against `duckdb_v2_library_version`
    at the point of use. This check exists only to fail early and legibly when
    someone points the build at a released libduckdb, and a check that cannot
    see the answer must say so rather than guess.
    """
    if sys.platform != "win32" and shutil.which("nm"):
        out = subprocess.run(["nm", "-g", str(lib)], capture_output=True, text=True, check=False)
        if out.returncode == 0:
            return "duckdb_v2_" in out.stdout
    if sys.platform == "win32" and shutil.which("dumpbin"):
        out = subprocess.run(
            ["dumpbin", "/exports", str(lib)], capture_output=True, text=True, check=False
        )
        if out.returncode == 0:
            return "duckdb_v2_" in out.stdout
    return None


def main() -> int:
    """Assemble the bundle. Returns a process exit code."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path, help="a DuckDB checkout")
    ap.add_argument("output", type=Path, help="bundle directory to create")
    ap.add_argument("--build-type", default="Release")
    args = ap.parse_args()

    src, out = args.source.resolve(), args.output.resolve()
    if not (src / "tools" / "cpp" / "duckdb_cpp.cpp").is_file():
        sys.exit(f"{src} has no tools/cpp: not a DuckDB checkout with the C++ API")

    build_dir = src / "build" / args.build_type.lower()
    runtime = find_first(build_dir, RUNTIME_NAMES) if build_dir.is_dir() else None
    if runtime is None:
        build_engine(src, build_dir, args.build_type)
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
    (out / "lib").mkdir(parents=True)
    (out / "cmake").mkdir()

    for rel in CPP_API_FILES + HEADER_FILES:
        shutil.copy2(src / rel, out / Path(rel).name)
    shutil.copy2(src / CMAKE_FILE, out / "cmake" / Path(CMAKE_FILE).name)
    # Follow any symlink chain (libduckdb.dylib -> libduckdb.1.5.dylib).
    shutil.copy2(runtime.resolve(), out / "lib" / runtime.name)
    # Windows links against the import library and loads the DLL, so it needs both.
    if imp := find_first(build_dir, IMPORT_NAMES):
        shutil.copy2(imp.resolve(), out / "lib" / imp.name)

    sha = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True)
    (out / "ENGINE_SHA").write_text((sha.stdout.strip() or "unknown") + "\n")

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"bundle: {out} ({size / 1e6:.0f}MB), engine {(out / 'ENGINE_SHA').read_text()[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
