"""Measure and gate line coverage of the C++ seam.

Almost all of this package's logic is C++, so `coverage.py` reports close to
100% while saying nothing about the code that matters. This measures the seam
itself, through clang's instrumentation.

Run against a build made with `-fprofile-instr-generate -fcoverage-mapping`.

    seam_coverage.py <profraw-dir> <extension.so> [--min-lines 90]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def tool(name: str) -> str:
    """Locate an llvm tool, preferring the one Xcode selects."""
    found = shutil.which(name)
    if found:
        return found
    try:
        return subprocess.run(["xcrun", "--find", name], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        sys.exit(f"{name} not found; a clang toolchain is required for seam coverage")


def main() -> int:
    """Report seam coverage and fail if it is below the floor."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("profraw_dir", type=Path)
    ap.add_argument("extension", type=Path)
    ap.add_argument("--min-lines", type=float, default=90.0)
    args = ap.parse_args()

    raw = sorted(args.profraw_dir.glob("*.profraw"))
    if not raw:
        sys.exit(f"no .profraw files in {args.profraw_dir}; was the build instrumented?")

    merged = args.profraw_dir / "merged.profdata"
    subprocess.run([tool("llvm-profdata"), "merge", "-sparse", *map(str, raw), "-o", str(merged)], check=True)

    sources = sorted((Path(__file__).resolve().parent.parent / "src" / "_duckdb").glob("*.cpp"))

    def run_cov(verb: str, *extra: str) -> str:
        return subprocess.run(
            [
                tool("llvm-cov"),
                verb,
                str(args.extension),
                f"-instr-profile={merged}",
                *extra,
                *map(str, sources),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    print(run_cov("report"))
    report = json.loads(run_cov("export", "-summary-only"))
    totals = report["data"][0]["totals"]
    lines = totals["lines"]["percent"]
    functions = totals["functions"]["percent"]
    print(f"seam lines {lines:.2f}%  functions {functions:.2f}%  (floor {args.min_lines:.0f}%)")

    if functions < 100.0:
        # A bound method with no test at all is a different problem from a few
        # uncovered branches, and DESIGN asks for every seam entry point to
        # have a direct test.
        print("FAIL: some seam functions are never executed", file=sys.stderr)
        return 1
    if lines < args.min_lines:
        print(f"FAIL: seam line coverage {lines:.2f}% is below {args.min_lines:.2f}%", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
