#!/usr/bin/env python3
"""Compare benchmark instruction counts against the committed baseline. See benchmarks/README.md."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 1
GATE_DEFAULT_THRESHOLD_PCT = 5.0  # run-to-run noise is about 0.1%, so this sits far above it
BINDING_FRACTION_CUTOFF = 0.25  # below this share, a bench is mostly DuckDB and a threshold on it means little

# Which DuckDB-only bench each numeric bench is measured against; the others are so clearly ours it is not needed.
_E = "benchmarks/test_engine_control_perf.py"
FLOOR_MAP = {
    "benchmarks/test_produce_numpy_perf.py::test_to_numpy_numeric": f"{_E}::test_engine_sum_2col_500k",
    "benchmarks/test_produce_numpy_perf.py::test_to_numpy_null_int": f"{_E}::test_engine_sum_1col_200k",
}

_TRIGGER_RE = re.compile(r"^desc:\s*Trigger:\s*Client Request:\s*(?P<uri>.+?)\s*$")
_TOTALS_RE = re.compile(r"^totals:\s*(?P<ir>\d+)\s*$")


# -- callgrind parsing


def _normalize_uri(raw: str) -> str:
    """Return a repo-relative benchmark key, dropping the absolute prefix a run outside the repo leaves."""
    raw = raw.strip()
    if "::" not in raw:
        return raw
    path, _, rest = raw.partition("::")
    idx = path.find("benchmarks/")
    if idx > 0:
        path = path[idx:]
    return f"{path}::{rest}"


def parse_profiles(profile_dir: Path) -> dict[str, int]:
    """Every benchmark's instruction count, read from the callgrind dumps in `profile_dir`."""
    counts: dict[str, int] = {}
    files = sorted(profile_dir.rglob("*")) if profile_dir.exists() else []
    for f in files:
        if not f.is_file():
            continue
        uri: str | None = None
        ir: int | None = None
        try:
            text = f.read_text(errors="replace")
        except (OSError, UnicodeError):
            continue
        for line in text.splitlines():
            m = _TRIGGER_RE.match(line)
            if m:
                uri = _normalize_uri(m.group("uri"))
                continue
            m = _TOTALS_RE.match(line)
            if m:
                ir = int(m.group("ir"))
        if uri and "::" in uri and ir is not None:
            counts[uri] = max(counts.get(uri, 0), ir)
    return counts


# -- helpers


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def _load_gate_set(gate_list: Path | None) -> set[str]:
    """Load the benchmarks that may be gated, from a `pytest -m gate --collect-only -q` listing."""
    if not gate_list or not gate_list.exists():
        return set()
    out = set()
    for raw in gate_list.read_text().splitlines():
        line = raw.strip()
        if "::" in line:  # a pytest node id; the workflow already filtered the listing down to these
            out.add(_normalize_uri(line))
    return out


def _pct(base: int, new: int) -> float:
    return 0.0 if base == 0 else (new - base) / base * 100.0


# -- writing a baseline


def regen(args: argparse.Namespace) -> int:
    """Write baseline.json from a valgrind run: the counts, where they came from, and what may be gated."""
    counts = parse_profiles(Path(args.profiles))
    if not counts:
        print(f"ERROR: no benchmark dumps found under {args.profiles}", file=sys.stderr)
        return 2
    gate_set = _load_gate_set(Path(args.gate_list) if args.gate_list else None)

    benches: dict[str, dict] = {}
    auto_moved: list[str] = []
    for uri, ir in sorted(counts.items()):
        source_marker = "gate" if uri in gate_set else "informational"
        marker = source_marker
        binding_fraction = None
        floor_uri = FLOOR_MAP.get(uri)
        if source_marker == "gate" and floor_uri and floor_uri in counts and ir > 0:
            binding_fraction = round(max(0.0, 1.0 - counts[floor_uri] / ir), 4)
            if binding_fraction < args.cutoff:
                marker = "informational"  # mostly DuckDB, so a threshold on the total would not mean much
                auto_moved.append(uri)
        benches[uri] = {
            "marker": marker,
            "source_marker": source_marker,
            "auto_moved": marker != source_marker,
            "instructions": ir,
            "binding_fraction": binding_fraction,
            "threshold_pct": GATE_DEFAULT_THRESHOLD_PCT if marker == "gate" else None,
        }

    baseline = {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_commit": args.git_commit,
            "duckdb_engine_sha": args.engine_sha,
            "requirements_bench_sha256": _sha256(Path(args.pins)) if args.pins else "",
            "measurement": {"tool": "valgrind callgrind", "event": "Ir", "pythonhashseed": "0"},
            "bench_scale": os.environ.get("BENCH_SCALE", ""),  # counts are only comparable at the same scale
            "gate_default_threshold_pct": GATE_DEFAULT_THRESHOLD_PCT,
            "binding_fraction_cutoff": args.cutoff,
            "noise_note": "callgrind Ir observed ~0.1% run-to-run; gate threshold set well above.",
        },
        "benchmarks": benches,
    }
    Path(args.out).write_text(json.dumps(baseline, indent=2) + "\n")
    n_gate = sum(1 for b in benches.values() if b["marker"] == "gate")
    n_info = len(benches) - n_gate
    print(f"Wrote {args.out}: {len(benches)} benchmarks ({n_gate} gate, {n_info} informational).")
    if auto_moved:
        print(f"Option-B auto-moved {len(auto_moved)} engine-diluted gate(s) to informational:")
        for uri in auto_moved:
            print(f"  {uri}  (binding_fraction={benches[uri]['binding_fraction']})")
        print("Recommend updating these benches' @pytest.mark.gate -> informational so code matches the baseline.")
    return 0


# -- comparing against one


def compare(args: argparse.Namespace) -> int:
    """Diff a fresh valgrind run against baseline.json and print a report; only --enforce lets it fail."""
    new_counts = parse_profiles(Path(args.profiles))
    if not new_counts:
        print(f"ERROR: no benchmark dumps found under {args.profiles}", file=sys.stderr)
        return 2
    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        # No committed baseline yet: report the run and never fail.
        print(f"No baseline at {baseline_path} yet -- run the workflow with regen=true to create it.")
        print(f"This run produced {len(new_counts)} benchmark instruction counts.")
        return 0
    baseline = json.loads(baseline_path.read_text())
    meta = baseline.get("meta", {})
    base_benches = baseline.get("benchmarks", {})

    # A baseline built at one BENCH_SCALE is only comparable to a run at the same scale.
    run_scale = os.environ.get("BENCH_SCALE", "")
    base_scale = meta.get("bench_scale", "")
    if run_scale != base_scale:
        print(
            f"WARNING: BENCH_SCALE differs (run={run_scale!r}, baseline={base_scale!r}) -> instruction counts are "
            "not comparable. Regenerate the baseline at this scale."
        )

    # The counts only compare cleanly against the pinned data libraries the baseline was built with.
    if args.pins:
        cur = _sha256(Path(args.pins))
        base_pins = meta.get("requirements_bench_sha256", "")
        if cur and base_pins and cur != base_pins:
            print(
                "WARNING: benchmarks/requirements-bench.txt differs from the baseline's pins -> data-lib deltas "
                "may not be pure binding. Regenerate the baseline with the current pins."
            )

    # The counts include DuckDB, so when its version differs from the baseline's a rise may not be ours.
    engine_changed = bool(
        args.engine_sha and meta.get("duckdb_engine_sha") and args.engine_sha != meta["duckdb_engine_sha"]
    )

    regressions: list[str] = []
    rows: list[tuple[str, str, str]] = []  # (status, uri, detail)
    for uri, ir in sorted(new_counts.items()):
        b = base_benches.get(uri)
        if b is None:
            rows.append(("NEW", uri, f"{ir} Ir (no baseline)"))
            continue
        base_ir = b["instructions"]
        delta = _pct(base_ir, ir)
        marker = b.get("marker", "informational")
        thr = b.get("threshold_pct") or GATE_DEFAULT_THRESHOLD_PCT
        detail = f"{base_ir} -> {ir} Ir  ({delta:+.2f}%, thr {thr:.1f}%, {marker})"
        if marker == "gate" and delta > thr:
            if engine_changed:
                rows.append(("ENGINE?", uri, detail + "  [engine pin changed -> not enforced]"))
            else:
                rows.append(("REGRESSION", uri, detail))
                regressions.append(uri)
        else:
            rows.append(("ok" if marker == "gate" else "info", uri, detail))
    rows.extend(
        ("MISSING", uri, "in baseline, absent from run (rename/removal?)")
        for uri in sorted(set(base_benches) - set(new_counts))
    )

    _print_report(meta, rows, engine_changed=engine_changed, enforce=args.enforce)

    if not args.enforce:
        return 0
    if engine_changed:
        print("\nNOT ENFORCING: the engine pin differs from the baseline; regenerate the baseline.")
        return 0
    return 1 if regressions else 0


def _print_report(meta: dict, rows: list[tuple[str, str, str]], *, engine_changed: bool, enforce: bool) -> None:
    mode = "ENFORCING" if enforce else "REPORT-ONLY (not failing the job)"
    print("=" * 100)
    print(f"CodSpeed instruction-count baseline comparison  [{mode}]")
    print(
        f"baseline: commit {meta.get('git_commit', '?')[:12]}  engine {str(meta.get('duckdb_engine_sha'))[:12]}"
        f"  generated {meta.get('generated_at_utc', '?')}"
    )
    if engine_changed:
        print(
            "WARNING: the engine pin SHA differs from the baseline -> engine-inclusive deltas may reflect the "
            "engine bump, not the binding. Regenerate the baseline for this engine."
        )
    print("=" * 100)
    order = {"REGRESSION": 0, "ENGINE?": 1, "MISSING": 2, "NEW": 3, "ok": 4, "info": 5}
    for status, uri, detail in sorted(rows, key=lambda r: (order.get(r[0], 9), r[1])):
        print(f"  [{status:>10}] {uri}\n               {detail}")
    n_reg = sum(1 for s, _, _ in rows if s == "REGRESSION")
    print("-" * 100)
    print(f"Summary: {len(rows)} benchmarks, {n_reg} gate regression(s)" + ("" if enforce else "  (report-only)"))


# -- command line


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: dispatch to the `regen` or `compare` subcommand."""
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("regen", help="write baseline.json from a valgrind run")
    r.add_argument("--profiles", required=True, help="CODSPEED_PROFILE_FOLDER with callgrind dumps")
    r.add_argument("--out", default="benchmarks/baseline.json")
    r.add_argument("--gate-list", help="file of gate node-ids (pytest -m gate --collect-only -q)")
    r.add_argument("--git-commit", default="")
    r.add_argument("--engine-sha", default="", help="engine.pin SHA the run was built against")
    r.add_argument("--pins", default="benchmarks/requirements-bench.txt")
    r.add_argument("--cutoff", type=float, default=BINDING_FRACTION_CUTOFF)
    r.set_defaults(func=regen)

    c = sub.add_parser("compare", help="compare a valgrind run against baseline.json")
    c.add_argument("--profiles", required=True)
    c.add_argument("--baseline", default="benchmarks/baseline.json")
    c.add_argument("--engine-sha", default="", help="engine.pin SHA the run was built against")
    c.add_argument(
        "--pins", default="benchmarks/requirements-bench.txt", help="warn if pins differ from the baseline's"
    )
    c.add_argument("--enforce", action="store_true", help="exit non-zero on a gate regression (default: report-only)")
    c.set_defaults(func=compare)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
