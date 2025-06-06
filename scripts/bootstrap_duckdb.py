#!/usr/bin/env python3
"""
bootstrap_duckdb.py
-------------------
Initialise or reconfigure the DuckDB git submodule used by the Python client.

Environment variables
---------------------
DUCKDB_REMOTE   Optional. If set, overrides the submodule’s remote URL.
DUCKDB_REF      Optional. Branch, tag or commit to checkout (default: "main").

Usage examples
--------------
    # vanilla setup (tracks upstream/main)
    python scripts/bootstrap_duckdb.py

    # point at your own fork + branch
    set DUCKDB_REMOTE=git@github.com:alice/duckdb.git   # Windows (PowerShell/CMD)
    export DUCKDB_REMOTE=git@github.com:alice/duckdb.git # Linux/macOS
    DUCKDB_REF=my-feature-branch python scripts/bootstrap_duckdb.py
"""

from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path
from textwrap import indent

DUCKDB_DIR = Path("external/duckdb")         # location of the submodule
REMOTE     = os.environ.get("DUCKDB_REMOTE") # override or None
REF        = os.environ.get("DUCKDB_REF", "main")


def run(cmd: list[str]) -> None:
    """Run *cmd*, streaming stdout/stderr. Raises on failure."""
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"\n❌ Command failed: {' '.join(exc.cmd)}\n", file=sys.stderr)
        sys.exit(exc.returncode)


def main() -> None:
    if REMOTE:
        print(f"► Setting submodule remote to {REMOTE}")
        run(["git", "submodule", "set-url", str(DUCKDB_DIR), REMOTE])

    print("► Initialising / updating submodule")
    run(["git", "submodule", "update", "--init", str(DUCKDB_DIR)])

    print("► Fetching tags")
    run(["git", "-C", str(DUCKDB_DIR), "fetch", "--tags"])

    print(f"► Checking out {REF}")
    run(["git", "-C", str(DUCKDB_DIR), "checkout", REF])

    # Report the exact commit we ended up on
    head = subprocess.check_output(
        ["git", "-C", str(DUCKDB_DIR), "rev-parse", "--short", "HEAD"],
        text=True,
    ).strip()
    print(f"✓ DuckDB pinned at commit {head}")


if __name__ == "__main__":
    shutil = getattr(sys.modules.get("builtins"), "shutil", None)
    if not shutil:
        import shutil  # noqa: E402 (lazy import to stay std-lib only)

    # Basic sanity checks
    if not shutil.which("git"):
        sys.exit("Git is not on PATH – cannot continue.")
    if not (Path.cwd() / ".git").is_dir():
        sys.exit("Run this script from the root of the repository that owns the submodule.")

    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted.")
