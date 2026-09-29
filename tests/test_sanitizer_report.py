"""The sanitizer report judge: which leaks count as ours, and what fails a run."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from sanitizer_report import errors, leaks, main, owner

ASAN = "#{} 0x7f00 in malloc (/usr/lib/gcc/x86_64-linux-gnu/13/libasan.so+0xfb3c)"
NEW = "#{} 0x7f01 in operator new(unsigned long) (/usr/lib/gcc/x86_64-linux-gnu/13/libasan.so+0xfd11)"
STDCXX = "#{} 0x7f02 in std::__cxx11::basic_string::_M_create (/usr/lib/x86_64-linux-gnu/libstdc++.so.6+0x1a2b)"
EXT = "#{} 0x7f03 in duckdb_python::Registry::Add src/_duckdb/registry.cpp:88:3 (/venv/duckdb/_duckdb.abi3.so+0x1234)"
ENGINE = "#{} 0x7f04 in duckdb::Allocator::Allocate (/venv/duckdb/libduckdb.so+0x5678)"
PYALLOC = "#{} 0x7f05 in PyObject_Malloc (/python/bin/python3.12+0x10)"
PYLONG = "#{} 0x7f06 in PyLong_FromLong (/python/bin/python3.12+0x20)"
MARSHAL = "#{} 0x7f07 in marshal_loads (/python/bin/python3.12+0x30)"
EVAL = "#{} 0x7f08 in _PyEval_EvalFrameDefault (/python/bin/python3.12+0x40)"
NUMPY = "#{} 0x7f09 in PyArray_NewFromDescr (/venv/numpy/_core/_multiarray_umath.cpython-312.so+0x50)"
EXEC = "#{} 0x7f0a in PyModule_ExecDef (/python/bin/python3.12+0x60)"


def stack(*frames: str) -> list[str]:
    return [frame.format(i) for i, frame in enumerate(frames)]


def report(kind: str, size: int, frames: list[str]) -> str:
    body = "\n".join(f"    {frame}" for frame in frames)
    header = f"{kind} leak of {size} byte(s) in 1 object(s) allocated from:"
    summary = f"SUMMARY: AddressSanitizer: {size} byte(s) leaked in 1 allocation(s)."
    return f"==1==ERROR: LeakSanitizer: detected memory leaks\n\n{header}\n{body}\n\n{summary}\n"


class TestOwner:
    def test_what_the_extension_allocates_is_the_extensions(self) -> None:
        assert owner(stack(ASAN, NEW, EXT, PYLONG)) == "extension"

    def test_the_runtimes_are_looked_through(self) -> None:
        assert owner(stack(ASAN, NEW, STDCXX, EXT)) == "extension"

    def test_what_the_engine_allocates_is_the_engines_even_when_the_extension_called_it(self) -> None:
        assert owner(stack(ASAN, ENGINE, EXT)) == "engine"

    def test_a_python_object_the_extension_made_through_the_c_api_is_the_extensions(self) -> None:
        assert owner(stack(ASAN, PYALLOC, PYLONG, EXT)) == "extension"

    def test_python_code_the_extension_ran_allocates_for_python(self) -> None:
        # An import started from the extension leaves the module's code behind, which CPython never frees.
        assert owner(stack(ASAN, PYALLOC, MARSHAL, EVAL, EXT)) == "other"

    def test_a_library_between_python_and_the_extension_owns_the_block(self) -> None:
        assert owner(stack(ASAN, PYALLOC, NUMPY, EXT)) == "other"

    def test_what_a_module_allocates_while_it_is_imported_is_import_time(self) -> None:
        assert owner(stack(ASAN, PYALLOC, PYLONG, EXT, EXEC, EVAL)) == "import"
        assert owner(stack(ASAN, NEW, EXT, EXEC)) == "import"
        assert owner(stack(ASAN, ENGINE, EXT, EXEC)) == "import"

    def test_python_on_its_own_is_other(self) -> None:
        assert owner(stack(ASAN, PYALLOC, EVAL)) == "other"


class TestParsing:
    def test_frames_as_ubuntus_runtime_prints_them(self) -> None:
        # A runtime with debug info prints a source line, and a library with a build id prints it between modules.
        asan = (
            "#0 0x7f0f in malloc ../../../../src/libsanitizer/asan/asan_malloc_linux.cpp:69 "
            "(/usr/lib/gcc/x86_64-linux-gnu/13/libasan.so+0xfd11)"
        )
        ffi = "#1 0x7f1f  (/lib/libffi.so.8+0x7b15) (BuildId: c914) (/lib/libffi.so.8+0x7b15)"
        ext = "#1 0x7f2f  (/venv/duckdb/_duckdb.abi3.so+0x1234) (BuildId: a1b2) (/venv/duckdb/_duckdb.abi3.so+0x1234)"
        assert owner([asan, ffi]) == "other"
        assert owner([asan, ext]) == "extension"

    def test_each_leak_record_keeps_its_own_frames(self) -> None:
        text = report("Direct", 64, stack(ASAN, NEW, EXT)) + report("Indirect", 8, stack(ASAN, PYALLOC, EVAL))
        found = leaks(text)
        assert [(leak.kind, leak.size, len(leak.frames)) for leak in found] == [("Direct", 64, 3), ("Indirect", 8, 3)]
        assert owner(found[0].frames) == "extension"

    def test_memory_errors_and_undefined_behaviour_are_errors(self) -> None:
        text = (
            "==2==ERROR: AddressSanitizer: heap-use-after-free on address 0x6020\n"
            "src/_duckdb/pyconv.cpp:12:5: runtime error: signed integer overflow\n"
            "==3==ERROR: LeakSanitizer: detected memory leaks\n"
            "SUMMARY: AddressSanitizer: 64 byte(s) leaked in 1 allocation(s).\n"
        )
        assert errors(text) == [
            "==2==ERROR: AddressSanitizer: heap-use-after-free on address 0x6020",
            "src/_duckdb/pyconv.cpp:12:5: runtime error: signed integer overflow",
        ]

    def test_the_runtimes_own_failures_are_errors(self) -> None:
        # These carry no ERROR: prefix, and the process dies with them.
        text = (
            'AddressSanitizer: CHECK failed: asan_interceptors.cpp:458 "((real___cxa_throw)) != (0)" (0x0, 0x0)\n'
            "    #0 0xfff6 in CheckUnwind ../../src/libsanitizer/asan/asan_rtl.cpp:69 (/usr/lib/libasan.so+0xf2970)\n"
            "AddressSanitizer:DEADLYSIGNAL\n"
            "==4==LeakSanitizer has encountered a fatal error.\n"
        )
        assert errors(text) == [
            'AddressSanitizer: CHECK failed: asan_interceptors.cpp:458 "((real___cxa_throw)) != (0)" (0x0, 0x0)',
            "AddressSanitizer:DEADLYSIGNAL",
            "==4==LeakSanitizer has encountered a fatal error.",
        ]


class TestVerdict:
    def judge(self, monkeypatch: pytest.MonkeyPatch, logs: Path) -> int:
        monkeypatch.setattr(sys, "argv", ["sanitizer_report.py", str(logs)])
        return main()

    def test_leaks_python_leaves_at_exit_pass(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "asan.11").write_text(report("Direct", 900, stack(ASAN, PYALLOC, MARSHAL, EVAL, EXT)))
        assert self.judge(monkeypatch, tmp_path) == 0

    def test_what_the_extension_allocates_while_imported_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "asan.15").write_text(report("Direct", 64, stack(ASAN, PYALLOC, PYLONG, EXT, EXEC)))
        assert self.judge(monkeypatch, tmp_path) == 0

    def test_a_leak_of_ours_fails(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "asan.12").write_text(report("Direct", 64, stack(ASAN, NEW, EXT)))
        assert self.judge(monkeypatch, tmp_path) == 1

    def test_a_leak_of_the_engines_fails(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "asan.16").write_text(report("Direct", 32, stack(ASAN, ENGINE, EXT)))
        assert self.judge(monkeypatch, tmp_path) == 1

    def test_frames_without_modules_fail_rather_than_pass_as_python(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        frames = ["#0 0x7f00 in malloc", "#1 0x7f03 in duckdb_python::Registry::Add src/_duckdb/registry.cpp:88"]
        (tmp_path / "asan.17").write_text(report("Direct", 64, frames))
        assert self.judge(monkeypatch, tmp_path) == 1

    def test_a_failure_of_the_runtime_itself_fails(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "asan.18").write_text("AddressSanitizer: CHECK failed: asan_interceptors.cpp:458\n")
        assert self.judge(monkeypatch, tmp_path) == 1

    def test_a_memory_error_fails(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "asan.13").write_text("==13==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x1\n")
        assert self.judge(monkeypatch, tmp_path) == 1

    def test_undefined_behaviour_fails(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "ubsan.14").write_text("src/_duckdb/pyconv.cpp:3:1: runtime error: load of misaligned address\n")
        assert self.judge(monkeypatch, tmp_path) == 1

    def test_no_reports_pass(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self.judge(monkeypatch, tmp_path) == 0
