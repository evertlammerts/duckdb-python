"""The generated expression modules match what their generators make of this build's catalog."""

import importlib
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[3]


def generator(monkeypatch, name):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module(name)


def committed(module):
    return (ROOT / "duckdb" / f"{module}.py").read_text()


def test_the_keyword_module_is_current(monkeypatch):
    gen_keywords = generator(monkeypatch, "gen_keywords")
    assert committed("_sql_keywords") == gen_keywords.render(duckdb.connect()), "run scripts/gen_keywords.py"


def test_the_aggregate_module_is_current(monkeypatch):
    gen_aggregates = generator(monkeypatch, "gen_aggregates")
    assert gen_aggregates.verify(duckdb.connect()) == []
    assert committed("_expression_aggregates") == gen_aggregates.render(), "run scripts/gen_aggregates.py"


def test_the_function_family_module_is_current(monkeypatch):
    gen_func_namespaces = generator(monkeypatch, "gen_func_namespaces")
    resolved, report = gen_func_namespaces.build(duckdb.connect())
    assert report["problems"] == []
    assert committed("_expression_namespaces") == gen_func_namespaces.render(resolved), (
        "run scripts/gen_func_namespaces.py"
    )
