"""Run the engine's sqllogictest corpus through the bridge and report.

    uv run python scripts/sqllogic_report.py [directory ...]

Prints, per directory, how many files ran and why the rest were skipped,
how many statements the bridge carried, how many query results matched,
and every statement it did not carry or result it did not match. The test
in tests/test_sqllogic.py holds floors; this is where to look when one
moves.
"""

from __future__ import annotations

import collections
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests._sqllogic import run_file
from tests.test_feature_coverage import corpus
from tests.test_sqllogic import DIRECTORIES


def main(directories: list[str]) -> int:
    """Print the report; exit 1 if the corpus is not at hand."""
    root = corpus()
    if root is None:
        print("no duckdb checkout: set DUCKDB_SOURCE", file=sys.stderr)
        return 1
    tmp = Path(tempfile.mkdtemp())
    for directory in directories or DIRECTORIES:
        outcomes = [run_file(p, tmp, root.parent.parent) for p in sorted((root / directory).rglob("*.test"))]
        ran = [o for o in outcomes if o.skipped is None]
        skipped = collections.Counter(o.skipped for o in outcomes if o.skipped)
        total = sum(o.statements + o.queries for o in ran)
        carried, matched = sum(o.carried for o in ran), sum(o.matched for o in ran)
        compared = sum(o.compared for o in ran)
        heading = f"{len(ran)}/{len(outcomes)} files ran; carried {carried}/{total}; matched {matched}/{compared}"
        print(f"\n== {directory}: {heading}")
        for reason, n in skipped.most_common(6):
            print(f"   skipped {n:3}  {reason}")
        for o in ran:
            for item in o.not_carried:
                print(f"   NOT CARRIED  {o.path.relative_to(root)}: {item}")
            for item in o.mismatched:
                print(f"   mismatched   {o.path.relative_to(root)}: {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
