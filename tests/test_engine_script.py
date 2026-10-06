"""scripts/engine.py: spec parsing, engine resolution, host detection and the offline paths of `fetch`."""

from __future__ import annotations

import argparse
import http.client
import importlib.util
import io
import platform
import shutil
import struct
import sys
import tarfile
import time
import urllib.error
import zlib
from pathlib import Path
from types import ModuleType

import pytest


def _load() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "engine.py"
    spec = importlib.util.spec_from_file_location("engine_script", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # The Spec dataclass resolves its string annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


engine = _load()

STAGED = "ab12cd34ef/v2.0.0-alpha1"
STAGED_ARCHIVE = (
    "https://duckdb-staging.duckdb.org/ab12cd34ef/v2.0.0-alpha1/duckdb/duckdb/github_release/"
    "duckdb-shared-libs-linux-amd64.tar.gz"
)
STAGED_KEPT = "ab12cd34ef-v2.0.0-alpha1-linux-amd64.tar.gz"
OTHER_STAGED = "ef98ab76cd/v2.0.0-alpha2"
OTHER_STAGED_ARCHIVE = (
    "https://duckdb-staging.duckdb.org/ef98ab76cd/v2.0.0-alpha2/duckdb/duckdb/github_release/"
    "duckdb-shared-libs-linux-amd64.tar.gz"
)
A_YEAR = 365 * 24 * 3600


def test_parse_spec_reads_a_staged_spec() -> None:
    assert engine.parse_spec(STAGED) == engine.Spec(version="v2.0.0-alpha1", commit="ab12cd34ef")


def test_parse_spec_reads_a_release_spec() -> None:
    spec = engine.parse_spec("v2.0.1")
    assert spec.version == "v2.0.1"
    assert spec.commit is None


@pytest.mark.parametrize("text", [STAGED, "v2.0.1"])
def test_a_spec_prints_in_the_format_parse_spec_reads(text: str) -> None:
    assert str(engine.parse_spec(text)) == text


@pytest.mark.parametrize(
    "text",
    ["xyz/v2.0.0", "AB12CD34EF/v2.0.0", "ab12cd34e/v2.0.0", "ab12cd34ef/2.0.0", "2.0.0", ""],
    ids=["non-hex-commit", "uppercase-commit", "short-commit", "staged-without-v", "release-without-v", "empty"],
)
def test_parse_spec_refuses_a_malformed_spec(text: str) -> None:
    with pytest.raises(SystemExit, match="not an engine spec"):
        engine.parse_spec(text)


@pytest.mark.parametrize("text", ["v2.1.0", "0123456789/v2.1.0-alpha1", "v1.5.6"])
def test_parse_spec_refuses_a_version_off_the_supported_line(text: str) -> None:
    with pytest.raises(SystemExit, match="line this client supports"):
        engine.parse_spec(text)


@pytest.mark.parametrize("text", ["v1.5.6", "0123456789/v2.1.0-alpha1"])
def test_parse_spec_accepts_an_off_line_version_when_the_line_is_not_required(text: str) -> None:
    assert str(engine.parse_spec(text, require_line=False)) == text


def test_the_archive_of_a_staged_spec_lives_on_the_staging_host() -> None:
    assert engine.parse_spec(STAGED).archive_url("linux-amd64") == STAGED_ARCHIVE


def test_the_archive_of_a_release_has_no_confirmed_channel() -> None:
    with pytest.raises(SystemExit, match=r"cannot fetch v2\.0\.1"):
        engine.parse_spec("v2.0.1").archive_url("linux-amd64")


def test_a_staged_spec_is_built_from_its_commit() -> None:
    assert engine.parse_spec(STAGED).ref() == "ab12cd34ef"


def test_a_release_spec_is_built_from_its_tag() -> None:
    assert engine.parse_spec("v2.0.1").ref() == "v2.0.1"


def run(*argv: str) -> int:
    return int(engine.main(list(argv)))


def serve_pointers(monkeypatch: pytest.MonkeyPatch, **pages: str) -> None:
    """Answer the version pointers from `stable` and `alpha`; fetching a pointer not given is a KeyError."""
    urls = {"stable": engine.STABLE_POINTER, "alpha": engine.ALPHA_POINTER}
    served = {urls[name]: text for name, text in pages.items()}
    monkeypatch.setattr(engine, "fetch_text", lambda url: served[url])


@pytest.mark.parametrize("pointer", ["2.0.1", "v2.0.1"])
def test_resolve_takes_a_stable_release_on_the_line(monkeypatch: pytest.MonkeyPatch, pointer: str) -> None:
    serve_pointers(monkeypatch, stable=pointer)
    assert engine.resolve() == engine.Spec(version="v2.0.1", commit=None)


def test_resolve_falls_back_to_the_alpha_while_stable_is_off_the_line(monkeypatch: pytest.MonkeyPatch) -> None:
    serve_pointers(monkeypatch, stable="1.5.6", alpha="ab12cd34ef/v2.0.0-alpha9")
    assert engine.resolve() == engine.Spec(version="v2.0.0-alpha9", commit="ab12cd34ef")


def test_resolve_refuses_a_garbled_stable_pointer(monkeypatch: pytest.MonkeyPatch) -> None:
    serve_pointers(monkeypatch, stable="<html>")
    with pytest.raises(SystemExit, match="unrecognized stable version pointer"):
        engine.resolve()


def test_resolve_refuses_an_alpha_off_the_line(monkeypatch: pytest.MonkeyPatch) -> None:
    serve_pointers(monkeypatch, stable="1.5.6", alpha="ab12cd34ef/v2.1.0-alpha9")
    with pytest.raises(SystemExit, match="neither is on"):
        engine.resolve()


def test_resolve_refuses_a_garbled_alpha_pointer(monkeypatch: pytest.MonkeyPatch) -> None:
    serve_pointers(monkeypatch, stable="1.5.6", alpha="<html>")
    with pytest.raises(SystemExit, match="not an engine spec"):
        engine.resolve()


def test_the_resolve_command_prints_the_spec_alone(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    serve_pointers(monkeypatch, stable="1.5.6", alpha="ab12cd34ef/v2.0.0-alpha9")
    assert run("resolve") == 0
    assert capsys.readouterr().out == "ab12cd34ef/v2.0.0-alpha9\n"


@pytest.mark.parametrize(
    ("os_name", "machine", "expected"),
    [
        ("darwin", "arm64", "osx-arm64"),
        ("darwin", "x86_64", "osx-amd64"),
        ("linux", "x86_64", "linux-amd64"),
        ("linux", "aarch64", "linux-arm64"),
        ("win32", "AMD64", "windows-amd64"),
        ("win32", "ARM64", "windows-arm64"),
        ("linux", "riscv64", None),
        ("freebsd14", "amd64", None),
    ],
)
def test_host_target_names_the_archive_platform(
    monkeypatch: pytest.MonkeyPatch, os_name: str, machine: str, expected: str | None
) -> None:
    monkeypatch.setattr(sys, "platform", os_name)
    monkeypatch.setattr(platform, "machine", lambda: machine)
    assert engine.host_target() == expected


def test_extension_name_passes_a_plain_name_through() -> None:
    assert engine.extension_name("tpch") == "tpch"


@pytest.mark.parametrize("text", ["tpch;DROP", "TPCH", "", "tp ch"])
def test_extension_name_refuses_anything_but_a_lowercase_identifier(text: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="not an extension name"):
        engine.extension_name(text)


def test_load_library_refuses_a_file_that_is_not_a_library(tmp_path: Path) -> None:
    for name in engine.LIBRARY_NAMES:
        (tmp_path / name).write_bytes(b"not a shared library")
    with pytest.raises(SystemExit, match="cannot load"):
        engine.load_library(tmp_path)


def test_load_library_reports_a_directory_without_one(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no DuckDB library"):
        engine.load_library(tmp_path)


def make_archive(path: Path, files: dict[str, bytes]) -> Path:
    """A tar.gz whose members all carry an mtime a year in the past."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tar:
        for name, content in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            member.mtime = time.time() - A_YEAR
            member.mode = 0o644
            tar.addfile(member, io.BytesIO(content))
    return path


def damaged_deflate_archive() -> bytes:
    """A gzip that opens and starts extracting, then hits an invalid deflate block: zlib.error, not OSError."""
    plain = io.BytesIO()
    with tarfile.open(fileobj=plain, mode="w") as tar:
        for name in ("duckdb_v2.h", "libduckdb.so"):
            member = tarfile.TarInfo(name)
            member.size = 20000
            tar.addfile(member, io.BytesIO(bytes(20000)))
    raw = plain.getvalue()
    deflate = zlib.compressobj(9, zlib.DEFLATED, -15)
    valid_prefix = deflate.compress(raw[: len(raw) // 2]) + deflate.flush(zlib.Z_SYNC_FLUSH)
    header = b"\x1f\x8b\x08\x00" + struct.pack("<I", 0) + b"\x00\xff"
    reserved_block_type = b"\x07"
    return header + valid_prefix + reserved_block_type


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    return make_archive(
        tmp_path / "served" / "engine.tar.gz", {"duckdb_v2.h": b"new header", "libduckdb.so": b"new library"}
    )


@pytest.fixture
def downloads(monkeypatch: pytest.MonkeyPatch, archive: Path) -> list[str]:
    """Serves `archive` for every download on a macOS arm64 host and returns the URLs asked for."""
    urls: list[str] = []

    def serve(url: str, path: Path) -> None:
        urls.append(url)
        shutil.copyfile(archive, path)

    def forbidden(directory: Path) -> None:
        pytest.fail(f"a foreign library was loaded from {directory}")

    monkeypatch.setattr(engine, "download", serve)
    monkeypatch.setattr(engine, "host_target", lambda: "osx-arm64")
    monkeypatch.setattr(engine, "load_library", forbidden)
    return urls


class _FakeLibrary:
    """The host's shared library as the script sees it: loads, version reports and INSTALLs are recorded."""

    def __init__(self) -> None:
        self.handle = object()
        self.version = "v2.0.0-alpha1"
        self.loaded: list[Path] = []
        self.installed: list[tuple[object, list[str]]] = []

    def load(self, directory: Path) -> object:
        self.loaded.append(directory)
        return self.handle

    def report(self, _lib: object) -> str:
        return self.version

    def install(self, lib: object, extensions: list[str]) -> None:
        self.installed.append((lib, extensions))


@pytest.fixture
def library(monkeypatch: pytest.MonkeyPatch, downloads: list[str]) -> _FakeLibrary:
    """A linux-amd64 host with a fake library; depends on `downloads` so it is applied over that fixture's host."""
    fake = _FakeLibrary()
    monkeypatch.setattr(engine, "host_target", lambda: "linux-amd64")
    monkeypatch.setattr(engine, "load_library", fake.load)
    monkeypatch.setattr(engine, "library_version", fake.report)
    monkeypatch.setattr(engine, "install_extensions", fake.install)
    return fake


def fetch_foreign(dest: Path, *extra: str, spec: str = STAGED, target: str = "linux-amd64") -> int:
    return run("fetch", "--engine", spec, "--platform", target, "--dest", str(dest), *extra)


def kept_name(spec: str, target: str = "linux-amd64") -> str:
    return f"{spec.replace('/', '-')}-{target}.tar.gz"


def fetch_native(dest: Path, *extra: str) -> int:
    return run("fetch", "--engine", STAGED, "--dest", str(dest), *extra)


def scratch_directories(dest: Path) -> list[Path]:
    return list(dest.parent.glob(".engine-fetch-*"))


def test_fetch_unpacks_the_engine_with_fresh_mtimes(tmp_path: Path, downloads: list[str]) -> None:
    dest = tmp_path / "out" / "engine"
    assert fetch_foreign(dest) == 0
    assert downloads == [STAGED_ARCHIVE]
    assert (dest / "duckdb_v2.h").read_bytes() == b"new header"
    assert (dest / "libduckdb.so").read_bytes() == b"new library"
    for name in ("duckdb_v2.h", "libduckdb.so"):
        assert time.time() - (dest / name).stat().st_mtime < 3600
    assert scratch_directories(dest) == []


def test_fetch_replaces_a_previously_fetched_engine(tmp_path: Path, downloads: list[str]) -> None:
    dest = tmp_path / "engine"
    dest.mkdir()
    (dest / "duckdb_v2.h").write_bytes(b"old header")
    (dest / "libduckdb.so").write_bytes(b"old library")
    (dest / "stale.txt").write_text("left over from the old engine")
    assert fetch_foreign(dest) == 0
    assert (dest / "duckdb_v2.h").read_bytes() == b"new header"
    assert (dest / "libduckdb.so").read_bytes() == b"new library"
    assert not (dest / "stale.txt").exists()
    assert scratch_directories(dest) == []


def test_fetch_replaces_an_empty_directory(tmp_path: Path, downloads: list[str]) -> None:
    dest = tmp_path / "engine"
    dest.mkdir()
    assert fetch_foreign(dest) == 0
    assert (dest / "duckdb_v2.h").is_file()


def test_fetch_refuses_to_replace_a_regular_file(tmp_path: Path, downloads: list[str]) -> None:
    dest = tmp_path / "engine"
    dest.write_text("not an engine")
    with pytest.raises(SystemExit, match="refusing to replace"):
        fetch_foreign(dest)
    assert dest.read_text() == "not an engine"
    assert scratch_directories(dest) == []


def test_fetch_refuses_to_replace_a_directory_that_is_not_an_engine(tmp_path: Path, downloads: list[str]) -> None:
    dest = tmp_path / "engine"
    dest.mkdir()
    (dest / "notes.txt").write_text("keep me")
    with pytest.raises(SystemExit, match="refusing to replace"):
        fetch_foreign(dest)
    assert (dest / "notes.txt").read_text() == "keep me"
    assert scratch_directories(dest) == []


def test_fetch_refuses_to_replace_a_directory_with_a_header_but_no_library(
    tmp_path: Path, downloads: list[str]
) -> None:
    dest = tmp_path / "engine"
    dest.mkdir()
    (dest / "duckdb_v2.h").write_text("a header some other project owns")
    with pytest.raises(SystemExit, match="refusing to replace"):
        fetch_foreign(dest)
    assert (dest / "duckdb_v2.h").read_text() == "a header some other project owns"


def test_fetch_refuses_to_replace_a_symlink_even_to_an_engine(tmp_path: Path, downloads: list[str]) -> None:
    target = tmp_path / "elsewhere"
    target.mkdir()
    (target / "duckdb_v2.h").write_bytes(b"old header")
    (target / "libduckdb.so").write_bytes(b"old library")
    dest = tmp_path / "engine"
    try:
        dest.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("creating a symlink needs a privilege this host withholds")
    with pytest.raises(SystemExit, match="refusing to replace"):
        fetch_foreign(dest)
    assert dest.is_symlink()
    assert (target / "duckdb_v2.h").read_bytes() == b"old header"
    assert scratch_directories(dest) == []


def test_fetch_keeps_the_old_engine_when_the_archive_has_no_header(
    tmp_path: Path, archive: Path, downloads: list[str]
) -> None:
    make_archive(archive, {"libduckdb.so": b"new library"})
    dest = tmp_path / "engine"
    dest.mkdir()
    (dest / "duckdb_v2.h").write_bytes(b"old header")
    (dest / "libduckdb.so").write_bytes(b"old library")
    with pytest.raises(SystemExit, match=r"carried no duckdb_v2\.h"):
        fetch_foreign(dest)
    assert (dest / "duckdb_v2.h").read_bytes() == b"old header"
    assert (dest / "libduckdb.so").read_bytes() == b"old library"
    assert scratch_directories(dest) == []


@pytest.mark.skipif(sys.platform == "win32", reason="the Windows path unloads a real DLL before the rename")
def test_fetch_on_the_host_platform_checks_the_library_version(
    tmp_path: Path, downloads: list[str], library: _FakeLibrary
) -> None:
    dest = tmp_path / "engine"
    assert fetch_native(dest) == 0
    assert downloads == [STAGED_ARCHIVE]
    assert len(library.loaded) == 1
    assert (dest / "libduckdb.so").is_file()


def test_fetch_keeps_the_old_engine_when_the_library_reports_another_version(
    tmp_path: Path, downloads: list[str], library: _FakeLibrary
) -> None:
    library.version = "v2.0.0-alpha2"
    dest = tmp_path / "engine"
    dest.mkdir()
    (dest / "duckdb_v2.h").write_bytes(b"old header")
    (dest / "libduckdb.so").write_bytes(b"old library")
    with pytest.raises(SystemExit, match=r"calls itself v2\.0\.0-alpha2, not v2\.0\.0-alpha1"):
        fetch_native(dest)
    assert (dest / "libduckdb.so").read_bytes() == b"old library"
    assert scratch_directories(dest) == []


@pytest.mark.skipif(sys.platform == "win32", reason="the Windows path unloads a real DLL before the rename")
def test_fetch_installs_extensions_through_the_fetched_engine(
    tmp_path: Path, downloads: list[str], library: _FakeLibrary
) -> None:
    dest = tmp_path / "engine"
    assert fetch_native(dest, "--install", "tpch", "--install", "json") == 0
    assert library.installed == [(library.handle, ["tpch", "json"])]
    assert library.loaded[-1] == dest


def test_fetch_extracts_from_an_archive_that_is_already_there(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, downloads: list[str]
) -> None:
    cache = tmp_path / "cache"
    cached = make_archive(
        cache / kept_name(STAGED), {"duckdb_v2.h": b"cached header", "libduckdb.so": b"cached library"}
    )

    def refuse(url: str, _path: Path) -> None:
        pytest.fail(f"downloaded {url} although the archive exists")

    monkeypatch.setattr(engine, "download", refuse)
    dest = tmp_path / "engine"
    assert fetch_foreign(dest, "--archive-dir", str(cache)) == 0
    assert (dest / "duckdb_v2.h").read_bytes() == b"cached header"
    assert (dest / "libduckdb.so").read_bytes() == b"cached library"
    assert time.time() - (dest / "duckdb_v2.h").stat().st_mtime < 3600
    assert cached.is_file()
    assert scratch_directories(dest) == []


def test_fetch_keeps_a_downloaded_archive_in_the_archive_dir(
    tmp_path: Path, archive: Path, downloads: list[str]
) -> None:
    cache = tmp_path / "cache" / "nested"
    dest = tmp_path / "engine"
    assert fetch_foreign(dest, "--archive-dir", str(cache)) == 0
    assert downloads == [STAGED_ARCHIVE]
    assert [entry.name for entry in cache.iterdir()] == [STAGED_KEPT]
    assert (cache / STAGED_KEPT).read_bytes() == archive.read_bytes()
    assert (dest / "duckdb_v2.h").read_bytes() == b"new header"
    assert scratch_directories(dest) == []


def test_a_second_fetch_of_the_same_spec_reuses_the_kept_archive(tmp_path: Path, downloads: list[str]) -> None:
    cache = tmp_path / "cache"
    dest = tmp_path / "engine"
    assert fetch_foreign(dest, "--archive-dir", str(cache)) == 0
    assert fetch_foreign(dest, "--archive-dir", str(cache)) == 0
    assert downloads == [STAGED_ARCHIVE]
    assert (dest / "libduckdb.so").read_bytes() == b"new library"


def test_a_relative_archive_dir_is_resolved_against_the_repo_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, downloads: list[str]
) -> None:
    repo = tmp_path / "repo"
    monkeypatch.setattr(engine, "REPO_ROOT", repo)
    assert fetch_foreign(tmp_path / "engine", "--archive-dir", "cache") == 0
    assert (repo / "cache" / STAGED_KEPT).is_file()


def test_fetches_of_different_specs_do_not_share_a_kept_archive(tmp_path: Path, downloads: list[str]) -> None:
    cache = tmp_path / "cache"
    dest = tmp_path / "engine"
    assert fetch_foreign(dest, "--archive-dir", str(cache)) == 0
    assert [entry.name for entry in cache.iterdir()] == [STAGED_KEPT]
    assert fetch_foreign(dest, "--archive-dir", str(cache), spec=OTHER_STAGED) == 0
    assert downloads == [STAGED_ARCHIVE, OTHER_STAGED_ARCHIVE]
    assert sorted(entry.name for entry in cache.iterdir()) == sorted([STAGED_KEPT, kept_name(OTHER_STAGED)])


def test_the_second_engine_of_a_run_is_downloaded_not_taken_from_the_first_ones_archive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, downloads: list[str]
) -> None:
    contents = {
        STAGED_ARCHIVE: b"header of the old engine",
        OTHER_STAGED_ARCHIVE: b"header of the new engine",
    }

    def serve(url: str, path: Path) -> None:
        make_archive(path, {"duckdb_v2.h": contents[url], "libduckdb.so": b"library"})

    monkeypatch.setattr(engine, "download", serve)
    cache = tmp_path / "cache"
    dest = tmp_path / "engine"
    assert fetch_foreign(dest, "--archive-dir", str(cache)) == 0
    assert (dest / "duckdb_v2.h").read_bytes() == b"header of the old engine"
    assert fetch_foreign(dest, "--archive-dir", str(cache), spec=OTHER_STAGED) == 0
    assert (dest / "duckdb_v2.h").read_bytes() == b"header of the new engine"


def test_a_kept_archive_for_another_platform_is_not_reused(tmp_path: Path, downloads: list[str]) -> None:
    cache = tmp_path / "cache"
    dest = tmp_path / "engine"
    assert fetch_foreign(dest, "--archive-dir", str(cache)) == 0
    assert fetch_foreign(dest, "--archive-dir", str(cache), target="linux-arm64") == 0
    assert downloads == [STAGED_ARCHIVE, STAGED_ARCHIVE.replace("linux-amd64", "linux-arm64")]
    assert sorted(entry.name for entry in cache.iterdir()) == sorted([STAGED_KEPT, kept_name(STAGED, "linux-arm64")])


def test_fetch_leaves_files_it_did_not_create_in_the_archive_dir_alone(tmp_path: Path, downloads: list[str]) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "photos-backup.tar.gz").write_bytes(b"not ours to delete")
    (cache / "notes.txt").write_text("not an archive")
    assert fetch_foreign(tmp_path / "engine", "--archive-dir", str(cache)) == 0
    assert sorted(entry.name for entry in cache.iterdir()) == sorted(["notes.txt", "photos-backup.tar.gz", STAGED_KEPT])


def test_fetch_checks_the_library_version_of_an_archive_that_is_already_there(
    tmp_path: Path, downloads: list[str], library: _FakeLibrary
) -> None:
    library.version = "v2.0.0-alpha2"
    cache = tmp_path / "cache"
    cached = make_archive(cache / kept_name(STAGED), {"duckdb_v2.h": b"h", "libduckdb.so": b"l"})
    dest = tmp_path / "engine"
    dest.mkdir()
    (dest / "duckdb_v2.h").write_bytes(b"old header")
    (dest / "libduckdb.so").write_bytes(b"old library")
    with pytest.raises(SystemExit, match=r"calls itself v2\.0\.0-alpha2, not v2\.0\.0-alpha1"):
        fetch_native(dest, "--archive-dir", str(cache))
    assert downloads == []
    assert library.loaded != []
    assert (dest / "libduckdb.so").read_bytes() == b"old library"
    assert cached.is_file()
    assert scratch_directories(dest) == []


def test_a_downloaded_archive_that_fails_the_version_check_is_not_kept(
    tmp_path: Path, downloads: list[str], library: _FakeLibrary
) -> None:
    library.version = "v2.0.0-alpha2"
    cache = tmp_path / "cache"
    earlier = make_archive(cache / kept_name(OTHER_STAGED), {"duckdb_v2.h": b"h", "libduckdb.so": b"l"})
    with pytest.raises(SystemExit, match="calls itself"):
        fetch_native(tmp_path / "engine", "--archive-dir", str(cache))
    assert downloads == [STAGED_ARCHIVE]
    assert [entry.name for entry in cache.iterdir()] == [earlier.name]


def test_a_downloaded_archive_without_the_header_is_not_kept(
    tmp_path: Path, archive: Path, downloads: list[str]
) -> None:
    make_archive(archive, {"libduckdb.so": b"new library"})
    cache = tmp_path / "cache"
    with pytest.raises(SystemExit, match=r"carried no duckdb_v2\.h"):
        fetch_foreign(tmp_path / "engine", "--archive-dir", str(cache))
    assert list(cache.glob("*")) == []


def test_fetch_exits_cleanly_on_an_unreadable_kept_archive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, downloads: list[str]
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / kept_name(STAGED)).write_bytes(b"not a tar at all")
    monkeypatch.setattr(engine, "download", lambda _url, _path: pytest.fail("a kept archive must not be downloaded"))
    dest = tmp_path / "eng"
    with pytest.raises(SystemExit, match="not a readable archive"):
        fetch_foreign(dest, "--archive-dir", str(cache))
    assert not dest.exists()


def test_fetch_exits_cleanly_on_a_truncated_kept_archive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, archive: Path, downloads: list[str]
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    whole = archive.read_bytes()
    (cache / kept_name(STAGED)).write_bytes(whole[: len(whole) // 2])
    monkeypatch.setattr(engine, "download", lambda _url, _path: pytest.fail("a kept archive must not be downloaded"))
    dest = tmp_path / "eng"
    with pytest.raises(SystemExit, match="not a readable archive"):
        fetch_foreign(dest, "--archive-dir", str(cache))
    assert not dest.exists()
    assert scratch_directories(dest) == []


def test_fetch_exits_cleanly_on_a_kept_archive_with_a_damaged_deflate_stream(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, downloads: list[str]
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / kept_name(STAGED)).write_bytes(damaged_deflate_archive())
    monkeypatch.setattr(engine, "download", lambda _url, _path: pytest.fail("a kept archive must not be downloaded"))
    dest = tmp_path / "eng"
    with pytest.raises(SystemExit, match="not a readable archive"):
        fetch_foreign(dest, "--archive-dir", str(cache))
    assert not dest.exists()
    assert scratch_directories(dest) == []


def test_a_truncated_download_exits_cleanly_and_leaves_nothing_in_the_archive_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, archive: Path, downloads: list[str]
) -> None:
    whole = archive.read_bytes()
    monkeypatch.setattr(engine, "download", lambda _url, path: path.write_bytes(whole[: len(whole) // 2]))
    cache = tmp_path / "cache"
    dest = tmp_path / "eng"
    with pytest.raises(SystemExit, match="not a readable archive"):
        fetch_foreign(dest, "--archive-dir", str(cache))
    assert list(cache.glob("*")) == []
    assert not dest.exists()
    assert scratch_directories(dest) == []


def test_fetch_needs_a_platform_when_the_host_has_no_prebuilt_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, downloads: list[str]
) -> None:
    monkeypatch.setattr(engine, "host_target", lambda: None)
    with pytest.raises(SystemExit, match="pass --platform"):
        fetch_native(tmp_path / "engine")
    assert downloads == []


def test_fetch_refuses_to_install_through_a_foreign_library_before_downloading(
    tmp_path: Path, downloads: list[str]
) -> None:
    dest = tmp_path / "out" / "engine"
    argv = ("fetch", "--engine", STAGED, "--platform", "linux-amd64", "--dest", str(dest), "--install", "tpch")
    with pytest.raises(SystemExit, match="cannot run INSTALL"):
        run(*argv)
    assert downloads == []
    assert not dest.parent.exists()


def test_fetch_refuses_a_malformed_extension_name_before_downloading(tmp_path: Path, downloads: list[str]) -> None:
    argv = ("fetch", "--engine", STAGED, "--dest", str(tmp_path / "engine"), "--install", "tpch;DROP")
    with pytest.raises(SystemExit) as excinfo:
        run(*argv)
    assert excinfo.value.code == 2
    assert downloads == []


def test_fetch_of_a_release_exits_without_downloading(tmp_path: Path, downloads: list[str]) -> None:
    dest = tmp_path / "out" / "engine"
    with pytest.raises(SystemExit, match=r"cannot fetch v2\.0\.1"):
        run("fetch", "--engine", "v2.0.1", "--platform", "linux-amd64", "--dest", str(dest))
    assert downloads == []
    assert scratch_directories(dest) == []


def test_install_runs_the_extensions_through_the_fetched_library(tmp_path: Path, library: _FakeLibrary) -> None:
    assert run("install", "--dest", str(tmp_path), "tpch", "json") == 0
    assert library.loaded == [tmp_path]
    assert library.installed == [(library.handle, ["tpch", "json"])]


def test_install_refuses_a_malformed_extension_name_before_loading_anything(
    tmp_path: Path, library: _FakeLibrary
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        run("install", "--dest", str(tmp_path), "tpch", "json;DROP")
    assert excinfo.value.code == 2
    assert library.loaded == []
    assert library.installed == []


def test_install_needs_at_least_one_extension(tmp_path: Path, library: _FakeLibrary) -> None:
    with pytest.raises(SystemExit) as excinfo:
        run("install", "--dest", str(tmp_path))
    assert excinfo.value.code == 2
    assert library.loaded == []


RAW = "https://raw.githubusercontent.com/duckdb/duckdb"
CPP_API_SOURCES = {
    "duckdb_cpp.hpp": "tools/cpp/duckdb_cpp.hpp",
    "duckdb_cpp.cpp": "tools/cpp/duckdb_cpp.cpp",
    "cmake/DuckDBCppApi.cmake": "tools/cpp/cmake/DuckDBCppApi.cmake",
    "LICENSE": "LICENSE",
}
OLD_REF = "0123456789/v2.0.0-alpha0\n"


@pytest.fixture
def cpp_api(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    directory = tmp_path / "duckdb_cpp"
    monkeypatch.setattr(engine, "CPP_API_DIR", directory)
    return directory


def seed_cpp_api(directory: Path) -> None:
    for name in CPP_API_SOURCES:
        (directory / name).parent.mkdir(parents=True, exist_ok=True)
        (directory / name).write_bytes(b"old " + name.encode())
    (directory / "REF").write_text(OLD_REF)


def assert_cpp_api_untouched(directory: Path) -> None:
    for name in CPP_API_SOURCES:
        assert (directory / name).read_bytes() == b"old " + name.encode()
    assert (directory / "REF").read_text() == OLD_REF


def test_update_cpp_api_refuses_an_engine_off_the_line_before_downloading(
    monkeypatch: pytest.MonkeyPatch, cpp_api: Path
) -> None:
    def refuse(url: str, _path: Path) -> None:
        pytest.fail(f"downloaded {url} for an engine off the supported line")

    monkeypatch.setattr(engine, "download", refuse)
    seed_cpp_api(cpp_api)
    with pytest.raises(SystemExit, match="line this client supports"):
        run("update-cpp-api", "--engine", "0123456789/v2.1.0-alpha1")
    assert_cpp_api_untouched(cpp_api)


def test_update_cpp_api_creates_nothing_when_it_refuses_the_engine(
    monkeypatch: pytest.MonkeyPatch, cpp_api: Path
) -> None:
    monkeypatch.setattr(engine, "download", lambda url, _path: pytest.fail(f"downloaded {url}"))
    with pytest.raises(SystemExit, match="line this client supports"):
        run("update-cpp-api", "--engine", "0123456789/v2.1.0-alpha1")
    assert not cpp_api.exists()


@pytest.mark.parametrize(
    ("spec", "ref"), [(STAGED, "ab12cd34ef"), ("v2.0.1", "v2.0.1")], ids=["staged-by-commit", "release-by-tag"]
)
def test_update_cpp_api_vendors_the_files_at_the_specs_ref(
    monkeypatch: pytest.MonkeyPatch, cpp_api: Path, spec: str, ref: str
) -> None:
    urls: list[str] = []

    def serve(url: str, path: Path) -> None:
        urls.append(url)
        path.write_bytes(b"content-of:" + url.encode())

    monkeypatch.setattr(engine, "download", serve)
    seed_cpp_api(cpp_api)
    assert run("update-cpp-api", "--engine", spec) == 0
    for name, source in CPP_API_SOURCES.items():
        assert (cpp_api / name).read_bytes() == b"content-of:" + f"{RAW}/{ref}/{source}".encode()
    assert sorted(urls) == sorted(f"{RAW}/{ref}/{source}" for source in CPP_API_SOURCES.values())
    assert (cpp_api / "REF").read_text() == f"{spec}\n"


def test_update_cpp_api_creates_the_directory_on_a_first_run(monkeypatch: pytest.MonkeyPatch, cpp_api: Path) -> None:
    monkeypatch.setattr(engine, "download", lambda url, path: path.write_bytes(url.encode()))
    assert run("update-cpp-api", "--engine", STAGED) == 0
    assert (cpp_api / "cmake" / "DuckDBCppApi.cmake").is_file()
    assert (cpp_api / "REF").read_text() == f"{STAGED}\n"


def test_update_cpp_api_leaves_every_file_at_the_old_commit_when_a_download_fails(
    monkeypatch: pytest.MonkeyPatch, cpp_api: Path
) -> None:
    served: list[str] = []

    def flaky(url: str, path: Path) -> None:
        if len(served) == 2:
            sys.exit(f"{url}: gave up")
        served.append(url)
        path.write_bytes(b"new content")

    monkeypatch.setattr(engine, "download", flaky)
    seed_cpp_api(cpp_api)
    with pytest.raises(SystemExit, match="gave up"):
        run("update-cpp-api", "--engine", STAGED)
    assert len(served) == 2
    assert_cpp_api_untouched(cpp_api)


class _FlakyUrlopen:
    """Serves failures from a list (an HTTP status int or an exception to raise mid-body), then bytes."""

    def __init__(self, failures: list[object]) -> None:
        self.failures = failures
        self.calls = 0

    def __call__(self, request: object, timeout: int) -> io.BytesIO:
        self.calls += 1
        if self.failures:
            failure = self.failures.pop(0)
            if isinstance(failure, int):
                url = "http://x"
                raise urllib.error.HTTPError(url, failure, "boom", None, None)  # type: ignore[arg-type]
            assert isinstance(failure, Exception)
            return _BrokenBody(failure)
        return io.BytesIO(b"ok")


class _BrokenBody(io.BytesIO):
    """A response whose body fails as a dropped connection does."""

    def __init__(self, error: Exception) -> None:
        super().__init__(b"partial")
        self.error = error

    def read(self, size: int | None = -1) -> bytes:
        raise self.error


def _flaky(monkeypatch: pytest.MonkeyPatch, failures: list[object]) -> tuple[_FlakyUrlopen, list[float]]:
    flaky = _FlakyUrlopen(failures)
    naps: list[float] = []
    monkeypatch.setattr(engine.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(engine.time, "sleep", naps.append)
    return flaky, naps


def test_a_transfer_retries_rate_limiting_and_server_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    flaky, naps = _flaky(monkeypatch, [429, 503])
    assert engine.fetch_text("http://x") == "ok"
    assert flaky.calls == 3
    assert naps == [2, 4]


def test_a_transfer_treats_other_statuses_as_facts(monkeypatch: pytest.MonkeyPatch) -> None:
    flaky, _ = _flaky(monkeypatch, [404])
    monkeypatch.setattr(engine.time, "sleep", lambda _delay: pytest.fail("a 404 must not be retried"))
    with pytest.raises(SystemExit, match="404"):
        engine.fetch_text("http://x")
    assert flaky.calls == 1


def test_a_transfer_gives_up_after_three_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    flaky, naps = _flaky(monkeypatch, [503, 503, 503])
    with pytest.raises(SystemExit, match="after 3 attempts"):
        engine.fetch_text("http://x")
    assert flaky.calls == 3
    assert len(naps) == 2


def test_a_download_retries_a_body_that_ends_short(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    flaky, naps = _flaky(monkeypatch, [http.client.IncompleteRead(b"partial")])
    out = tmp_path / "archive"
    engine.download("http://x", out)
    assert out.read_bytes() == b"ok"
    assert flaky.calls == 2
    assert naps == [2]


def test_a_download_retries_a_connection_reset_mid_body(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    resets: list[object] = [ConnectionResetError("gone") for _ in range(3)]
    flaky, _ = _flaky(monkeypatch, resets)
    with pytest.raises(SystemExit, match="after 3 attempts"):
        engine.download("http://x", tmp_path / "archive")
    assert flaky.calls == 3
