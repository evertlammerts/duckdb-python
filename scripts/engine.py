"""Resolve, download and pin the DuckDB engine the client builds against.

One engine spec format serves all subcommands: `v<version>` for a release
(`v2.0.0`), or `<commit10>/<version>` for a staged build as the staging
pointer prints it (`96063b9e39/v2.0.0-alpha43763`).

- `resolve` prints the spec of the newest published 2.0 engine.
- `fetch` downloads that engine's shared library and C headers into a
  directory the build reads as `DUCKDB_ROOT`.
- `update-cpp-api` copies DuckDB's C++ API source at the spec's commit into
  `third_party/duckdb_cpp` and records the spec in its `REF` file.
"""

# tarfile's extraction filter arrived in 3.11.4.
# /// script
# requires-python = ">=3.11.4"
# ///

from __future__ import annotations

import argparse
import ctypes
import http.client
import platform
import re
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar, cast

if TYPE_CHECKING:
    from collections.abc import Callable
    from http.client import HTTPResponse

T = TypeVar("T")

STABLE_POINTER = "https://duckdb.org/data/latest_stable_version.txt"
ALPHA_POINTER = "https://duckdb-staging.duckdb.org/latest_alpha_version.txt"
STAGING = "https://duckdb-staging.duckdb.org"
RELEASES = "https://install.duckdb.org"
RAW = "https://raw.githubusercontent.com/duckdb/duckdb"

#: The line this client supports; `resolve` refuses an engine outside it.
LINE = "v2.0."

REPO_ROOT = Path(__file__).resolve().parents[1]
CPP_API_DIR = REPO_ROOT / "third_party" / "duckdb_cpp"

#: What `update-cpp-api` ships: the C++ API source and DuckDB's license, never the C headers.
CPP_API_FILES = {
    "tools/cpp/duckdb_cpp.hpp": "duckdb_cpp.hpp",
    "tools/cpp/duckdb_cpp.cpp": "duckdb_cpp.cpp",
    "tools/cpp/cmake/DuckDBCppApi.cmake": "cmake/DuckDBCppApi.cmake",
    "LICENSE": "LICENSE",
}

LIBRARY_NAMES = ("libduckdb.dylib", "libduckdb.so", "duckdb.dll")

PLATFORMS = (
    "osx-arm64",
    "osx-amd64",
    "linux-amd64",
    "linux-arm64",
    "linux-amd64-musl",
    "windows-amd64",
    "windows-arm64",
)


@dataclass(frozen=True)
class Spec:
    """One published engine: its version name, and for a staged build the commit it was built from."""

    version: str
    commit: str | None

    def __str__(self) -> str:
        return self.version if self.commit is None else f"{self.commit}/{self.version}"

    def archive_url(self, target: str) -> str:
        """Where the shared-libs archive for `target` lives; fails for a release until that channel is confirmed."""
        if self.commit is None:
            sys.exit(
                f"no confirmed channel serves a released shared-libs archive yet; cannot fetch {self.version}. "
                "Pass a staged spec (<commit10>/<version>), or extend archive_url once 2.0.0 publishes one."
            )
        return f"{STAGING}/{self.commit}/{self.version}/duckdb/duckdb/github_release/duckdb-shared-libs-{target}.tar.gz"

    def ref(self) -> str:
        """The git ref the engine was built from: the commit for a staged build, the tag for a release."""
        return self.version if self.commit is None else self.commit


def parse_spec(text: str, *, require_line: bool = True) -> Spec:
    """Read a spec in either format, refusing an engine off the supported line wherever the spec came from."""
    if "/" in text:
        commit, _, version = text.partition("/")
        if not re.fullmatch(r"[0-9a-f]{10}", commit) or not version.startswith("v"):
            sys.exit(f"not an engine spec: {text!r} (want v<version> or <commit10>/<version>)")
        spec = Spec(version=version, commit=commit)
    elif text.startswith("v"):
        spec = Spec(version=text, commit=None)
    else:
        sys.exit(f"not an engine spec: {text!r} (want v<version> or <commit10>/<version>)")
    if require_line and not spec.version.startswith(LINE):
        sys.exit(f"{spec} is not on the {LINE}x line this client supports")
    return spec


def transfer(url: str, timeout: int, consume: Callable[[HTTPResponse], T]) -> T:
    """Open `url` and let `consume` read the body, retrying the whole transfer on transient failures.

    The custom User-Agent is load-bearing: duckdb.org answers 403 to Python's default one.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "duckdb-python-build"})
    last: Exception | None = None
    for attempt in range(3):
        if attempt:
            time.sleep(2**attempt)
        try:
            with cast("HTTPResponse", urllib.request.urlopen(request, timeout=timeout)) as response:
                return consume(response)
        except urllib.error.HTTPError as error:
            # Closed by hand: an HTTPError left to the collector warns, and 3.14 escalates that in tests.
            error.close()
            # 429 and the 5xx family are worth retrying; any other status is a fact about the URL.
            if error.code < 500 and error.code != 429:
                sys.exit(f"{url}: {error}")
            last = error
        # A connection dropped mid-body raises OSError, or IncompleteRead, which is no OSError at all.
        except (OSError, http.client.HTTPException) as error:
            last = error
    sys.exit(f"{url}: {last} (after 3 attempts)")


def fetch_text(url: str) -> str:
    """One small text file over HTTPS."""
    return transfer(url, 60, lambda response: response.read().decode().strip())


def resolve() -> Spec:
    """The newest published engine on the line: the latest stable once one exists, the latest alpha until then."""
    stable = fetch_text(STABLE_POINTER).removeprefix("v")
    if not re.fullmatch(r"[0-9]+(\.[0-9]+)*", stable):
        sys.exit(f"unrecognized stable version pointer: {stable!r}")
    if f"v{stable}".startswith(LINE):
        return Spec(version=f"v{stable}", commit=None)
    spec = parse_spec(fetch_text(ALPHA_POINTER), require_line=False)
    if not spec.version.startswith(LINE):
        sys.exit(
            f"the latest alpha is {spec} and the latest stable v{stable}: neither is on the {LINE}x line. "
            "No per-line pointer is published, so the engine must now be named by hand: pass --engine."
        )
    return spec


def host_target() -> str | None:
    """The archive platform name for this host, or None when none matches it. Linux is read as glibc."""
    os_name = {"darwin": "osx", "linux": "linux", "win32": "windows"}.get(sys.platform)
    arch = {"x86_64": "amd64", "amd64": "amd64", "arm64": "arm64", "aarch64": "arm64"}.get(platform.machine().lower())
    if os_name is None or arch is None:
        return None
    return f"{os_name}-{arch}"


def load_library(directory: Path) -> ctypes.CDLL:
    """The DuckDB shared library in `directory`, loaded."""
    for name in LIBRARY_NAMES:
        if (directory / name).is_file():
            try:
                return ctypes.CDLL(str(directory / name))
            except OSError as error:
                sys.exit(f"cannot load {directory / name} on this host: {error}")
    sys.exit(f"no DuckDB library ({', '.join(LIBRARY_NAMES)}) in {directory}")


def library_version(lib: ctypes.CDLL) -> str:
    """What the library calls itself, e.g. `v2.0.0-alpha43763`."""
    lib.duckdb_library_version.restype = ctypes.c_char_p
    return cast(bytes, lib.duckdb_library_version()).decode()


def install_extensions(lib: ctypes.CDLL, extensions: list[str]) -> None:
    """INSTALL each extension through the library itself, into the default extension directory."""
    database, connection = ctypes.c_void_p(), ctypes.c_void_p()
    if lib.duckdb_open(None, ctypes.byref(database)) != 0:
        sys.exit("duckdb_open failed on the fetched library")
    if lib.duckdb_connect(database, ctypes.byref(connection)) != 0:
        sys.exit("duckdb_connect failed on the fetched library")
    for extension in extensions:
        # A null result pointer is accepted, though not documented as such; it keeps this independent of
        # duckdb_result's layout, at the cost of the engine's error text.
        if lib.duckdb_query(connection, f"INSTALL {extension}".encode(), None) != 0:
            sys.exit(f"INSTALL {extension} failed; no error text comes through this path")
        print(f"installed {extension}", file=sys.stderr)
    lib.duckdb_disconnect(ctypes.byref(connection))
    lib.duckdb_close(ctypes.byref(database))


def download(url: str, path: Path) -> None:
    """One file over HTTPS, streamed to `path`; a retried attempt rewrites the file from the start."""
    print(f"fetching {url}", file=sys.stderr)

    def write(response: HTTPResponse) -> None:
        with path.open("wb") as out:
            shutil.copyfileobj(response, out)

    transfer(url, 300, write)


def fresh_mtime(member: tarfile.TarInfo, dest: str) -> tarfile.TarInfo:
    """The data filter plus extraction-time mtimes: a build dir must see a switched engine as newer than its objects."""
    return tarfile.data_filter(member, dest).replace(mtime=time.time(), deep=False)


def extension_name(text: str) -> str:
    """An extension name as INSTALL accepts it, refused before anything is downloaded."""
    if not re.fullmatch(r"[a-z0-9_]+", text):
        msg = f"not an extension name: {text!r}"
        raise argparse.ArgumentTypeError(msg)
    return text


def cmd_resolve(_args: argparse.Namespace) -> int:
    """Print the resolved spec, nothing else, so a caller can capture it."""
    print(resolve())
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    """Download the engine archive into the destination, verify it, and optionally install extensions through it."""
    spec = parse_spec(args.engine) if args.engine else resolve()
    host = host_target()
    target = args.platform or host
    if target is None:
        sys.exit(f"no prebuilt DuckDB for {sys.platform}/{platform.machine()}; pass --platform")
    if target != host and args.install:
        sys.exit(f"cannot run INSTALL through a {target} library on this host")
    dest = Path(args.dest)
    if not dest.is_absolute():
        dest = REPO_ROOT / dest
    print(f"engine {spec} for {target} -> {dest}", file=sys.stderr)

    archive_dir = Path(args.archive_dir) if args.archive_dir else None
    if archive_dir and not archive_dir.is_absolute():
        archive_dir = REPO_ROOT / archive_dir
    # The file name carries the spec and the platform: a kept archive may only ever serve the engine
    # it was downloaded for, so mere existence at some path must not count as a match.
    kept = archive_dir / f"{str(spec).replace('/', '-')}-{target}.tar.gz" if archive_dir else None

    dest.parent.mkdir(parents=True, exist_ok=True)
    # The scratch directory sits beside the destination so the final rename cannot cross a filesystem.
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix=".engine-fetch-") as scratch:
        if kept and kept.exists():
            archive = kept
            print(f"using the archive at {archive}", file=sys.stderr)
        else:
            archive = Path(scratch) / "engine.tar.gz"
            download(spec.archive_url(target), archive)
        unpacked = Path(scratch) / "unpacked"
        unpacked.mkdir()
        try:
            with tarfile.open(archive) as tar:
                tar.extractall(unpacked, filter=fresh_mtime)
        # A truncated gzip raises EOFError, a bad header OSError, and a damaged deflate stream a bare
        # zlib.error; none of them is a TarError.
        except (tarfile.TarError, EOFError, OSError, zlib.error) as error:
            sys.exit(f"{archive} is not a readable archive ({error}); delete it and retry")
        if not (unpacked / "duckdb_v2.h").is_file():
            sys.exit("the archive carried no duckdb_v2.h and cannot serve as DUCKDB_ROOT")
        if target == host:
            lib = load_library(unpacked)
            reported = library_version(lib)
            if reported != spec.version:
                sys.exit(f"the downloaded library calls itself {reported}, not {spec.version}")
            if sys.platform == "win32":
                # Windows refuses to move a loaded DLL, and the rename below moves it.
                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel32.FreeLibrary.argtypes = [ctypes.c_void_p]
                kernel32.FreeLibrary(lib._handle)
        # The old engine goes only now, when a verified replacement is complete beside it.
        if dest.is_symlink() or dest.exists():
            replaceable = (dest / "duckdb_v2.h").is_file() and any((dest / n).is_file() for n in LIBRARY_NAMES)
            if dest.is_symlink() or not dest.is_dir() or (any(dest.iterdir()) and not replaceable):
                sys.exit(f"{dest} exists and does not look like a fetched engine; refusing to replace it")
            shutil.rmtree(dest)
        unpacked.rename(dest)

        # Kept only now, when it extracted and verified, so an interrupted download never poisons the location.
        if kept and archive != kept:
            kept.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(archive, kept)

    if target != host:
        print(f"ready: {dest} holds a {target} engine; nothing more can be checked on this host", file=sys.stderr)
        return 0
    if args.install:
        install_extensions(load_library(dest), args.install)
    print(f"ready: {dest} holds {spec.version}", file=sys.stderr)
    return 0


def cmd_install(args: argparse.Namespace) -> int:
    """INSTALL extensions through an already fetched engine."""
    dest = Path(args.dest)
    if not dest.is_absolute():
        dest = REPO_ROOT / dest
    install_extensions(load_library(dest), args.extensions)
    return 0


def cmd_update_cpp_api(args: argparse.Namespace) -> int:
    """Move the shipped C++ API to the spec's commit and record the spec in REF."""
    spec = parse_spec(args.engine) if args.engine else resolve()
    ref_file = CPP_API_DIR / "REF"
    before = ref_file.read_text().strip() if ref_file.is_file() else "nothing"
    # Staged first, moved in only complete, so a failed download cannot leave the files at two commits.
    with tempfile.TemporaryDirectory() as scratch:
        staged: list[tuple[Path, Path]] = []
        for source, name in CPP_API_FILES.items():
            fetched = Path(scratch) / name.replace("/", "-")
            download(f"{RAW}/{spec.ref()}/{source}", fetched)
            staged.append((fetched, CPP_API_DIR / name))
        for fetched, out in staged:
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(fetched, out)
    ref_file.write_text(f"{spec}\n")
    print(f"{CPP_API_DIR} now at {spec} (was {before})", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Dispatch a subcommand. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def engine_argument(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--engine", help="engine spec; default: resolve the newest published engine")

    resolve_cmd = commands.add_parser("resolve", help="print the spec of the newest published 2.0 engine")
    resolve_cmd.set_defaults(handler=cmd_resolve)

    fetch = commands.add_parser("fetch", help="download an engine into a DUCKDB_ROOT directory")
    engine_argument(fetch)
    fetch.add_argument("--dest", default="engine", help="directory to (re)create, relative to the repo root")
    fetch.add_argument(
        "--archive-dir", help="directory of kept archives, reused per spec and platform, relative to the repo root"
    )
    fetch.add_argument("--platform", choices=PLATFORMS, help="default: detected from this host, glibc assumed on Linux")
    fetch.add_argument(
        "--install", action="append", default=[], type=extension_name, metavar="EXT", help="INSTALL this extension"
    )
    fetch.set_defaults(handler=cmd_fetch)

    install = commands.add_parser("install", help="INSTALL extensions through an already fetched engine")
    install.add_argument("--dest", default="engine", help="the fetched engine directory, relative to the repo root")
    install.add_argument("extensions", nargs="+", type=extension_name, metavar="EXT")
    install.set_defaults(handler=cmd_install)

    update = commands.add_parser("update-cpp-api", help="re-pin third_party/duckdb_cpp to an engine's commit")
    engine_argument(update)
    update.set_defaults(handler=cmd_update_cpp_api)

    args = parser.parse_args(argv)
    return cast(int, args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
