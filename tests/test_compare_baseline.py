"""The benchmark gate's decision logic: what enforces, what fails closed, and how a baseline names its engine."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def _load() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "compare_baseline.py"
    spec = importlib.util.spec_from_file_location("compare_baseline", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cb = _load()

GATE = "benchmarks/test_fetch_perf.py::test_x"
ENGINE = ("v2.0.0-alpha1", "ab12cd34ef")
PINS_TEXT = "numpy==2.0.0\n"
PINS_SHA256 = hashlib.sha256(PINS_TEXT.encode()).hexdigest()


@pytest.fixture
def profiles(tmp_path: Path) -> Path:
    directory = tmp_path / "profiles"
    directory.mkdir()
    (directory / "cg.1").write_text(f"desc: Trigger: Client Request: {GATE}\ntotals: 1000\n")
    return directory


@pytest.fixture
def pins(tmp_path: Path) -> Path:
    path = tmp_path / "requirements-bench.txt"
    # Bytes, not text: text mode would write \r\n on Windows and the recorded hash covers the exact bytes.
    path.write_bytes(PINS_TEXT.encode())
    return path


@pytest.fixture(autouse=True)
def unset_bench_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BENCH_SCALE", raising=False)


def baseline_file(
    tmp_path: Path,
    version: str = ENGINE[0],
    commit: str = ENGINE[1],
    instructions: int = 1000,
    bench_scale: str = "",
    pins_sha256: str = PINS_SHA256,
) -> Path:
    path = tmp_path / "baseline.json"
    path.write_text(
        json.dumps(
            {
                "meta": {
                    "duckdb_engine_version": version,
                    "duckdb_engine_commit": commit,
                    "bench_scale": bench_scale,
                    "requirements_bench_sha256": pins_sha256,
                },
                "benchmarks": {GATE: {"marker": "gate", "instructions": instructions, "threshold_pct": 5.0}},
            }
        )
    )
    return path


@pytest.fixture
def same_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cb, "engine_identity", lambda: ENGINE)


def compare(profiles: Path, baseline: Path, pins: Path | str, *extra: str) -> int:
    return int(
        cb.main(["compare", "--profiles", str(profiles), "--baseline", str(baseline), "--pins", str(pins), *extra])
    )


def test_baseline_engine_prints_a_staged_spec(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = baseline_file(tmp_path)
    assert cb.main(["baseline-engine", "--baseline", str(path)]) == 0
    assert capsys.readouterr().out.strip() == f"{ENGINE[1]}/{ENGINE[0]}"


def test_baseline_engine_prints_a_release_tag_alone(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = baseline_file(tmp_path, version="v2.0.0")
    assert cb.main(["baseline-engine", "--baseline", str(path)]) == 0
    assert capsys.readouterr().out.strip() == "v2.0.0"


def test_baseline_engine_is_silent_without_a_baseline(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cb.main(["baseline-engine", "--baseline", str(tmp_path / "none.json")]) == 0
    assert capsys.readouterr().out == ""


def test_a_baseline_that_cannot_name_its_engine_is_an_error_not_an_absence(tmp_path: Path) -> None:
    path = baseline_file(tmp_path, version="v2.0.0-alpha1", commit="")
    assert cb.main(["baseline-engine", "--baseline", str(path)]) == 1


def test_enforce_passes_on_the_same_engine_within_the_threshold(
    tmp_path: Path, profiles: Path, pins: Path, same_engine: None
) -> None:
    assert compare(profiles, baseline_file(tmp_path), pins, "--enforce") == 0


def test_enforce_fails_on_a_gate_regression(tmp_path: Path, profiles: Path, pins: Path, same_engine: None) -> None:
    assert compare(profiles, baseline_file(tmp_path, instructions=100), pins, "--enforce") == 1


def test_enforce_fails_when_the_engine_differs(
    tmp_path: Path, profiles: Path, pins: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cb, "engine_identity", lambda: ("v2.0.0-alpha2", "ffffffffff"))
    assert compare(profiles, baseline_file(tmp_path), pins, "--enforce") == 1


def test_enforce_fails_when_the_baseline_records_no_engine(
    tmp_path: Path, profiles: Path, pins: Path, same_engine: None
) -> None:
    assert compare(profiles, baseline_file(tmp_path, version="", commit=""), pins, "--enforce") == 1


def test_enforce_fails_when_a_gate_did_not_run(tmp_path: Path, pins: Path, same_engine: None) -> None:
    other = tmp_path / "other-profiles"
    other.mkdir()
    (other / "cg.1").write_text("desc: Trigger: Client Request: benchmarks/test_fetch_perf.py::test_y\ntotals: 7\n")
    assert compare(other, baseline_file(tmp_path), pins, "--enforce") == 1


def test_report_only_marks_an_engine_change_and_passes(
    tmp_path: Path, profiles: Path, pins: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cb, "engine_identity", lambda: ("v2.0.0-alpha2", "ffffffffff"))
    assert compare(profiles, baseline_file(tmp_path, instructions=100), pins) == 0
    assert "ENGINE?" in capsys.readouterr().out


def test_enforce_fails_without_a_baseline(tmp_path: Path, profiles: Path, pins: Path, same_engine: None) -> None:
    assert compare(profiles, tmp_path / "none.json", pins, "--enforce") == 1


def test_report_only_passes_without_a_baseline(
    tmp_path: Path, profiles: Path, pins: Path, same_engine: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert compare(profiles, tmp_path / "none.json", pins) == 0
    assert "No baseline" in capsys.readouterr().out


def test_enforce_fails_when_the_bench_scale_differs(
    tmp_path: Path, profiles: Path, pins: Path, same_engine: None
) -> None:
    assert compare(profiles, baseline_file(tmp_path, bench_scale="20"), pins, "--enforce") == 1


def test_report_only_passes_when_the_bench_scale_differs(
    tmp_path: Path, profiles: Path, pins: Path, same_engine: None
) -> None:
    assert compare(profiles, baseline_file(tmp_path, bench_scale="20"), pins) == 0


def test_enforce_passes_when_the_bench_scale_matches(
    tmp_path: Path, profiles: Path, pins: Path, same_engine: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BENCH_SCALE", "20")
    assert compare(profiles, baseline_file(tmp_path, bench_scale="20"), pins, "--enforce") == 0


def test_enforce_fails_when_the_pins_differ(tmp_path: Path, profiles: Path, pins: Path, same_engine: None) -> None:
    assert compare(profiles, baseline_file(tmp_path, pins_sha256="0" * 64), pins, "--enforce") == 1


def test_report_only_passes_when_the_pins_differ(tmp_path: Path, profiles: Path, pins: Path, same_engine: None) -> None:
    assert compare(profiles, baseline_file(tmp_path, pins_sha256="0" * 64), pins) == 0


def test_enforce_fails_without_a_pins_file_to_check(
    tmp_path: Path, profiles: Path, same_engine: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert compare(profiles, baseline_file(tmp_path), "", "--enforce") == 1
    assert "no pins file" in capsys.readouterr().out


def test_report_only_passes_without_a_pins_file_to_check(tmp_path: Path, profiles: Path, same_engine: None) -> None:
    assert compare(profiles, baseline_file(tmp_path), "") == 0


def test_enforce_fails_when_the_pins_file_does_not_exist(
    tmp_path: Path, profiles: Path, same_engine: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert compare(profiles, baseline_file(tmp_path), tmp_path / "gone.txt", "--enforce") == 1
    assert "no pins file" in capsys.readouterr().out


def test_report_only_passes_when_the_pins_file_does_not_exist(
    tmp_path: Path, profiles: Path, same_engine: None
) -> None:
    assert compare(profiles, baseline_file(tmp_path), tmp_path / "gone.txt") == 0


def test_enforce_fails_when_the_baseline_records_no_pins_hash(
    tmp_path: Path, profiles: Path, pins: Path, same_engine: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert compare(profiles, baseline_file(tmp_path, pins_sha256=""), pins, "--enforce") == 1
    assert "records no pins hash" in capsys.readouterr().out


def test_report_only_passes_when_the_baseline_records_no_pins_hash(
    tmp_path: Path, profiles: Path, pins: Path, same_engine: None
) -> None:
    assert compare(profiles, baseline_file(tmp_path, pins_sha256=""), pins) == 0


def regen(profiles: Path, out: Path, pins: Path) -> int:
    argv = ["regen", "--profiles", str(profiles), "--out", str(out), "--git-commit", "abc", "--pins", str(pins)]
    return int(cb.main(argv))


def test_regen_records_the_engine_it_ran_on(
    tmp_path: Path, profiles: Path, pins: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cb, "engine_identity", lambda: ("v2.0.0-alpha7", "0123456789"))
    out = tmp_path / "regenerated.json"
    assert regen(profiles, out, pins) == 0
    meta = json.loads(out.read_text())["meta"]
    assert meta["duckdb_engine_version"] == "v2.0.0-alpha7"
    assert meta["duckdb_engine_commit"] == "0123456789"
    assert meta["git_commit"] == "abc"


def test_regen_records_the_hash_of_the_pins_it_ran_with(
    tmp_path: Path, profiles: Path, pins: Path, same_engine: None
) -> None:
    out = tmp_path / "regenerated.json"
    assert regen(profiles, out, pins) == 0
    assert json.loads(out.read_text())["meta"]["requirements_bench_sha256"] == PINS_SHA256


def test_regen_records_the_bench_scale_it_ran_at(
    tmp_path: Path, profiles: Path, pins: Path, same_engine: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BENCH_SCALE", "20")
    out = tmp_path / "regenerated.json"
    assert regen(profiles, out, pins) == 0
    assert json.loads(out.read_text())["meta"]["bench_scale"] == "20"


def test_a_regenerated_baseline_passes_enforce_on_the_run_it_came_from(
    tmp_path: Path, profiles: Path, pins: Path, same_engine: None
) -> None:
    out = tmp_path / "regenerated.json"
    assert regen(profiles, out, pins) == 0
    assert compare(profiles, out, pins, "--enforce") == 0
