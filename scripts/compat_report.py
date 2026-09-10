"""Run the previous duckdb package's tests, copied unchanged under compat/suite, and report what passes here."""

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

#: A message about a missing name means the API point is absent, not that it behaves differently.
MISSING = re.compile(r"AttributeError|ImportError|ModuleNotFoundError|has no attribute|cannot import name")

#: A missing third-party module measures the environment, so only duckdb's own names count as gaps.
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
            # A test whose teardown failed did not fully pass; counting it would overstate the number.
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
    # Resolved here: the children run in a scratch directory, where a relative path would name nothing.
    chosen = [Path(a).resolve() for a in arguments if Path(a).exists()]
    passthrough = [a for a in arguments if not Path(a).exists()]
    environment = dict(os.environ)
    if facade:
        # compat/conftest.py reads this and points duckdb.connect at the compatibility API.
        environment["DUCKDB_COMPAT_FACADE"] = "1"
    suite = Path(__file__).resolve().parent.parent / "compat" / "suite"
    files = sorted(suite.rglob("test_*.py"))
    if chosen:
        files = [f for f in files if any(f == c or f.is_relative_to(c) for c in chosen)]
        if not files:
            listed = ", ".join(str(c) for c in chosen)
            print(f"no suite files under {listed}; the suite is {suite}", file=sys.stderr)
    outcomes: collections.Counter[str] = collections.Counter()
    signatures: collections.Counter[str] = collections.Counter()
    # The copied tests write files like test.db into the working directory, which must never be the repository.
    scratch = tempfile.mkdtemp(prefix="compat-")
    # One copied test leaves a delayed interrupt_main() behind, so each file gets a process it can only hurt itself in.
    for path in files:
        name = str(path.relative_to(suite))
        try:
            child = subprocess.run(
                [sys.executable, __file__, "--child", str(path), *passthrough],
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
