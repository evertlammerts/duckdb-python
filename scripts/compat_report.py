"""Run the adopted old-client tests and report how this client measures up.

The files under compat/suite are copied verbatim from duckdb-python main and
are never edited to pass: the repo's own suite is the gate, this one is the
parity measurement. Outcomes cluster into passed, missing surface (the old
API point does not exist here), behavior (it exists and answers differently),
and skipped. The report prints the top failure signatures and always exits 0.

Each file runs in its own subprocess with a timeout. The old suite assumes
the old client's semantics of its own surface: one adopted test leaves a
delayed interrupt_main() behind that aborts whatever pytest session it lands
in, and a lost interrupt leaves a query running unbounded. Isolation keeps
one file's bomb or hang from hiding every file after it.

    compat_report.py [extra pytest args]
"""

from __future__ import annotations

import collections
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

#: A failure whose message reads like an absent name is a surface gap; the
#: rest is the old surface existing here and answering differently.
MISSING = re.compile(r"AttributeError|ImportError|ModuleNotFoundError|has no attribute|cannot import name")

#: A third-party module absent from the venv measures the environment, not
#: this client; only duckdb's own names count against the surface.
ENVIRONMENT = re.compile(r"No module named '(?!duckdb)")

MARKER = "COMPATJSON:"
FILE_TIMEOUT = 120


class Recorder:
    """Per-test outcomes and failure signatures, collected over one run."""

    def __init__(self) -> None:
        self.outcomes: collections.Counter[str] = collections.Counter()
        self.signatures: collections.Counter[str] = collections.Counter()
        self._call_passed: set[str] = set()

    def _failure(self, report: pytest.TestReport | pytest.CollectReport) -> None:
        crash = getattr(report.longrepr, "reprcrash", None)
        if crash is not None:
            lines = str(crash.message).splitlines()
        elif report.longrepr is not None:
            lines = str(report.longrepr).splitlines()[-1:]
        else:
            lines = []
        message = lines[0] if lines else "unknown failure"
        self.signatures[message[:150]] += 1
        if ENVIRONMENT.search(message):
            self.outcomes["environment"] += 1
        elif MISSING.search(message):
            self.outcomes["missing surface"] += 1
        else:
            self.outcomes["behavior"] += 1

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        """Tally one phase of one test."""
        if report.when == "call":
            if report.passed:
                self.outcomes["passed"] += 1
                self._call_passed.add(report.nodeid)
            elif report.skipped:
                if hasattr(report, "wasxfail"):
                    self.outcomes["recorded divergence"] += 1
                else:
                    self.outcomes["skipped"] += 1
            else:
                self._failure(report)
        elif report.when == "setup" and not report.passed:
            if report.skipped:
                self.outcomes["skipped"] += 1
            else:
                self._failure(report)
        elif report.when == "teardown" and report.failed:
            # A test whose fixtures failed to tear down did not fully pass;
            # counting it as passed would overstate the parity number.
            if report.nodeid in self._call_passed:
                self._call_passed.discard(report.nodeid)
                self.outcomes["passed"] -= 1
            self._failure(report)

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        """Tally a file that failed or skipped at collection."""
        if report.failed:
            self._failure(report)
        elif report.skipped:
            self.outcomes["skipped"] += 1


def run_child(arguments: list[str]) -> int:
    """Run one file under the recorder and print the tallies as one JSON line."""
    recorder = Recorder()
    code = pytest.main([*arguments, "-q", "--tb=no", "-p", "no:cacheprovider"], plugins=[recorder])
    payload = {
        "outcomes": dict(recorder.outcomes),
        "signatures": dict(recorder.signatures),
        "interrupted": code == pytest.ExitCode.INTERRUPTED,
    }
    print(MARKER + json.dumps(payload), flush=True)
    return 0


def main() -> int:
    """Run every suite file in its own subprocess and print the clustered report."""
    if sys.argv[1:2] == ["--child"]:
        return run_child(sys.argv[2:])
    arguments = [a for a in sys.argv[1:] if a != "--facade"]
    facade = len(arguments) != len(sys.argv) - 1
    environment = dict(os.environ)
    if facade:
        # compat/conftest.py reads this and patches duckdb.connect to the
        # migration face, previewing what a drop-in connect() would score.
        environment["DUCKDB_COMPAT_FACADE"] = "1"
    suite = Path(__file__).resolve().parent.parent / "compat" / "suite"
    outcomes: collections.Counter[str] = collections.Counter()
    signatures: collections.Counter[str] = collections.Counter()
    # A scratch working directory: the adopted tests write files like test.db
    # into their cwd, which must never be the repository.
    scratch = tempfile.mkdtemp(prefix="compat-")
    for path in sorted(suite.rglob("test_*.py")):
        name = str(path.relative_to(suite))
        try:
            child = subprocess.run(
                [sys.executable, __file__, "--child", str(path), *arguments],
                capture_output=True,
                text=True,
                timeout=FILE_TIMEOUT,
                env=environment,
                cwd=scratch,
            )
        except subprocess.TimeoutExpired:
            outcomes["hung"] += 1
            signatures[f"{name}: no answer within {FILE_TIMEOUT}s"] += 1
            continue
        line = next((li for li in reversed(child.stdout.splitlines()) if li.startswith(MARKER)), None)
        if line is None:
            outcomes["harness casualty"] += 1
            signatures[f"{name}: exited {child.returncode} without a report"] += 1
            continue
        payload = json.loads(line[len(MARKER) :])
        outcomes.update(payload["outcomes"])
        signatures.update(payload["signatures"])
        if payload["interrupted"]:
            outcomes["truncated file"] += 1
            signatures[f"{name}: a stray interrupt ended the session early"] += 1
    total = sum(outcomes.values())
    passed = outcomes.get("passed", 0)
    mode = "with the migration facade" if facade else "as drop-in"
    print(f"\ncompat: {passed}/{total} of the adopted slice passes {mode}")
    for outcome, count in outcomes.most_common():
        print(f"  {count:4}  {outcome}")
    if signatures:
        print("top signatures:")
        for message, count in signatures.most_common(15):
            print(f"  {count:4}  {message}")
    # A measurement, not a gate.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
