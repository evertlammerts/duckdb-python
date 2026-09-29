"""Judge a run under AddressSanitizer, LeakSanitizer and UBSan from the reports it wrote, since the run exits 0.

Any memory error or undefined behaviour fails it, and so does a leak the extension or the engine allocated. CPython
frees little at exit, so the leaks it and the libraries leave are counted but pass, as does whatever a module
allocated while being imported, which happens once. Frames must carry their module, as `stack_trace_format` with
`(%m+%o)` guarantees; the run writes `asan.<pid>` and `ubsan.<pid>` through `log_path`.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

LEAK_HEADER = re.compile(r"^(Direct|Indirect) leak of (\d+) byte\(s\) in \d+ object\(s\) allocated from:")
FRAME = re.compile(r"^\s*#\d+ 0x[0-9a-f]+ ")
MODULE = re.compile(r"\(([^()\s]+)\+0x[0-9a-f]+\)")
#: The allocator and the C and C++ runtimes, which allocate on behalf of whoever called them.
RUNTIME = re.compile(r"/(libasan|libstdc\+\+|libc|libm|libgcc_s|ld-linux[^/]*)\.so")
EXTENSION = re.compile(r"/_duckdb[^/]*\.so")
ENGINE = re.compile(r"/libduckdb\.so")
PYTHON = re.compile(r"/(python3[^/]*|libpython3[^/]*)$")
#: Python code running, so what is allocated below it is the interpreter's, even when the extension called in.
INTERPRETER = re.compile(r"_PyEval_EvalFrameDefault|PyImport_|marshal_loads|_PyRun_")
#: A module's initialisation running, so the allocation happens once per process however long it runs.
IMPORT = re.compile(r" in PyModule_ExecDef\b")


@dataclass
class Leak:
    """One leak record: how the block was lost, its size, and the stack that allocated it, innermost frame first."""

    kind: str
    size: int
    frames: list[str]


def module_of(frame: str) -> str | None:
    """The shared object or executable a frame ran in."""
    found = MODULE.search(frame)
    return found[1] if found else None


def owner(frames: list[str]) -> str:
    """Who a leaked block belongs to: "extension", "engine", "import", or "other" for Python and its libraries.

    The owner is the first caller of the allocator outside the runtimes. A Python object the extension created
    through the C API is the extension's too: CPython allocated it, but only the extension could have lost it.
    """
    if any(IMPORT.search(frame) for frame in frames):
        return "import"
    callers = [(frame, module) for frame in frames if (module := module_of(frame)) and not RUNTIME.search(module)]
    if callers and EXTENSION.search(callers[0][1]):
        return "extension"
    if callers and ENGINE.search(callers[0][1]):
        return "engine"
    for frame, module in callers:
        if EXTENSION.search(module):
            return "extension"
        if INTERPRETER.search(frame) or not PYTHON.search(module):
            break
    return "other"


def leaks(text: str) -> list[Leak]:
    """Every leak record in one LeakSanitizer report."""
    found: list[Leak] = []
    current: Leak | None = None
    for line in text.splitlines():
        if header := LEAK_HEADER.match(line):
            current = Leak(header[1], int(header[2]), [])
            found.append(current)
        elif current is not None and FRAME.match(line):
            current.frames.append(line.strip())
        else:
            current = None
    return found


def errors(text: str) -> list[str]:
    """Every AddressSanitizer or UBSan error report in a log, each as its first line."""
    lines = text.splitlines()
    heads = [line for line in lines if "ERROR: AddressSanitizer" in line or "runtime error:" in line]
    return [line.strip() for line in heads]


def main() -> int:
    """Print what the logs hold, and exit 1 if the run found a memory error, undefined behaviour or a leak of ours."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log_dir", type=Path)
    args = ap.parse_args()

    logs = sorted(path for path in args.log_dir.iterdir() if path.name.startswith(("asan.", "ubsan.")))
    failures: list[str] = []
    counted: Counter[tuple[str, str]] = Counter()
    sizes: Counter[tuple[str, str]] = Counter()
    for log in logs:
        text = log.read_text(errors="replace")
        failures += [f"{log.name}: {head}" for head in errors(text)]
        for leak in leaks(text):
            if not any(module_of(frame) for frame in leak.frames):
                # Without modules every leak would read as Python's, so a missing format must not pass as clean.
                failures.append(f"{log.name}: a leak's frames name no module; is stack_trace_format set?")
                continue
            who = owner(leak.frames)
            counted[(who, leak.kind)] += 1
            sizes[(who, leak.kind)] += leak.size
            if who in ("extension", "engine"):
                failures.append(f"{log.name}: {leak.kind.lower()} leak of {leak.size} bytes by the {who}")
                failures += [f"    {frame}" for frame in leak.frames[:30]]

    print(f"{len(logs)} report files in {args.log_dir}")
    for (who, kind), count in sorted(counted.items()):
        print(f"  {kind} leaks by {who}: {count} blocks, {sizes[(who, kind)]} bytes")
    if failures:
        print("\n".join(failures))
        return 1
    print("no memory errors, no undefined behaviour, no leaks by the extension or the engine")
    return 0


if __name__ == "__main__":
    sys.exit(main())
